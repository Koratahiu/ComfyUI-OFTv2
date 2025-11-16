import logging
import torch
from typing import Optional

import comfy.utils
import comfy.sd
import folder_paths

from comfy.weight_adapter.base import WeightAdapterBase, weight_decompose
import comfy.weight_adapter


class OFTRotationUtil:
    def __init__(
        self,
        weight: torch.Tensor,
        block_size: int,
        coft: bool = False,
        eps: float = 6e-5,
        use_cayley_neumann: bool = True,
        num_cayley_neumann_terms: int = 5,
    ):
        self.weight = weight
        self.block_size = block_size
        self.coft = coft
        self.eps = eps
        self.use_cayley_neumann = use_cayley_neumann
        self.num_cayley_neumann_terms = num_cayley_neumann_terms
        self.rows, self.cols = torch.triu_indices(self.block_size, self.block_size, 1)

    def _get_triu_indices(self, device):
        if self.rows.device != device:
            self.rows = self.rows.to(device)
            self.cols = self.cols.to(device)
        return self.rows, self.cols

    def _pytorch_skew_symmetric(self, vec: torch.Tensor) -> torch.Tensor:
        batch_size = vec.shape[0]
        matrix = torch.zeros(batch_size, self.block_size, self.block_size, device=vec.device, dtype=vec.dtype)
        rows, cols = self._get_triu_indices(vec.device)
        matrix[:, rows, cols] = vec
        matrix = matrix - matrix.transpose(-2, -1)
        return matrix

    def _pytorch_skew_symmetric_inv(self, matrix: torch.Tensor) -> torch.Tensor:
        rows, cols = self._get_triu_indices(matrix.device)
        vec = matrix[:, rows, cols]
        return vec

    def _project_batch(self) -> torch.Tensor:
        oft_R = self._pytorch_skew_symmetric(self.weight)
        eps = self.eps * (1 / torch.sqrt(torch.tensor(oft_R.shape[0], device=oft_R.device)))
        origin_matrix = torch.zeros_like(oft_R)
        diff = oft_R - origin_matrix
        norm_diff = torch.norm(diff, dim=(1, 2), keepdim=True)
        mask = (norm_diff <= eps).bool()
        out = torch.where(mask, oft_R, origin_matrix + eps * (diff / norm_diff))
        return self._pytorch_skew_symmetric_inv(out)

    def _cayley_batch(self, Q: torch.Tensor) -> torch.Tensor:
        b, _ = Q.shape
        previous_dtype = Q.dtype
        Q_skew = self._pytorch_skew_symmetric(Q)
        if self.use_cayley_neumann:
            R = torch.eye(self.block_size, device=Q.device, dtype=Q.dtype).repeat(b, 1, 1)
            if self.num_cayley_neumann_terms > 1:
                R.add_(Q_skew, alpha=2.0)
                if self.num_cayley_neumann_terms > 2:
                    Q_squared = torch.bmm(Q_skew, Q_skew)
                    R.add_(Q_squared, alpha=2.0)
                    Q_power = Q_squared
                    for _ in range(3, self.num_cayley_neumann_terms - 1):
                        Q_power = torch.bmm(Q_power, Q_skew)
                        R.add_(Q_power, alpha=2.0)
                    Q_power = torch.bmm(Q_power, Q_skew)
                    R.add_(Q_power)
        else:
            id_mat = torch.eye(self.block_size, device=Q_skew.device).unsqueeze(0).expand_as(Q_skew)
            R = torch.linalg.solve(id_mat + Q_skew, id_mat - Q_skew, left=False)
        return R.to(previous_dtype)

    def get_rotation_matrix(self) -> torch.Tensor:
        weight = self.weight
        if self.coft:
            with torch.no_grad():
                projected_weight = self._project_batch()
                weight.copy_(projected_weight)
        return self._cayley_batch(weight)


class OFTv2Adapter(WeightAdapterBase):
    name = "oftv2"

    def __init__(self, loaded_keys: set[str], weights: tuple):
        self.loaded_keys = loaded_keys
        self.weights = weights

    @classmethod
    def load(
        cls,
        x: str,
        lora: dict[str, torch.Tensor],
        alpha: float,
        dora_scale: torch.Tensor,
        loaded_keys: Optional[set[str]] = None,
    ) -> Optional["OFTv2Adapter"]:
        if loaded_keys is None:
            loaded_keys = set()
        oft_r_weight_name = f"{x}.oft_R.weight"
        if oft_r_weight_name in lora:
            oft_r_weight = lora[oft_r_weight_name]
            loaded_keys.add(oft_r_weight_name)
            weights = (oft_r_weight, alpha, dora_scale)
            return cls(loaded_keys, weights)
        return None

    def calculate_weight(
        self,
        weight,
        key,
        strength,
        strength_model,
        offset,
        function,
        intermediate_dtype=torch.float32,
        original_weight=None,
    ):
        if strength == 0.0:
            return weight

        oft_r_weight_orig, alpha, dora_scale = self.weights

        try:
            oft_r_weight_processed = oft_r_weight_orig.to(weight.device, dtype=intermediate_dtype)
            
            r_loaded, n_elements = oft_r_weight_processed.shape
            block_size_f = (1 + (1 + 8 * n_elements) ** 0.5) / 2
            if abs(block_size_f - round(block_size_f)) > 1e-6:
                logging.error(f"OFTv2: Could not determine integer block_size for {key}. n_elements={n_elements} is invalid.")
                return weight
            block_size = int(round(block_size_f))

            base_weight = original_weight if original_weight is not None else weight
            out_features, *in_dims_tuple = base_weight.shape
            in_features = torch.prod(torch.tensor(in_dims_tuple)).item()

            if in_features % block_size != 0:
                logging.warning(f"OFTv2: in_features ({in_features}) not divisible by block_size ({block_size}) for {key}.")
                return weight

            r_actual = in_features // block_size
            block_share = r_loaded == 1

            if not block_share and r_loaded != r_actual:
                logging.error(f"OFTv2: Mismatch in number of blocks for {key}. Loaded: {r_loaded}, Expected: {r_actual}.")
                return weight

            # Pass the unscaled weight to the utility to get the full rotation matrix
            util = OFTRotationUtil(oft_r_weight_processed, block_size)
            orth_rotate = util.get_rotation_matrix()


            # For Linear layers,  rotates the input (x @ R), equivalent to rotating weights by R.T (W @ R.T).
            # For Conv2d layers,  rotates the weights directly (W @ R) to preserve spatial information.

            # Linear delta: W @ (R.T - I)
            # Conv2d delta: W @ (R - I)
            I = torch.eye(block_size, device=orth_rotate.device, dtype=orth_rotate.dtype)

            # Use weight dimension to determine layer type. Linear is 2D, Conv2d is 4D.
            is_conv2d = base_weight.dim() == 4
            
            if is_conv2d:
                # Use R for Conv2d layers
                rotation_matrix_for_weight = orth_rotate
            else:
                # Use R.T for Linear layers
                rotation_matrix_for_weight = orth_rotate.transpose(-1, -2)

            if block_share:
                diff_matrix = (rotation_matrix_for_weight - I.unsqueeze(0))
            else:
                diff_matrix = (rotation_matrix_for_weight - I)

            w_flat = base_weight.view(out_features, in_features)
            w_reshaped = w_flat.view(out_features, r_actual, block_size).to(intermediate_dtype)

            if block_share:
                w_diff_reshaped = torch.einsum("ork, kc -> orc", w_reshaped, diff_matrix.squeeze(0))
            else:
                w_diff_reshaped = torch.einsum("ork, rkc -> orc", w_reshaped, diff_matrix)
            
            lora_diff = w_diff_reshaped.reshape(base_weight.shape)

            if dora_scale is not None:
                weight = weight_decompose(dora_scale, weight, lora_diff, strength, 1.0, intermediate_dtype, function)
            else:
                weight += function((lora_diff * strength).type(weight.dtype))

        except Exception as e:
            logging.error(f"ERROR applying OFTv2 for {key}: {e}", exc_info=True)

        return weight


class OFTv2Loader:
    """
    This node loads an OFTv2 model and applies it to the UNet/Transformer and/or CLIP models.
    """
    def __init__(self):
        self.loaded_oft = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the OFTv2 will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the OFTv2 will be applied to."}),
                "oftv2_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the OFTv2 model file."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. Can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. Can be negative."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_oftv2"
    CATEGORY = "loaders"
    DESCRIPTION = "OFTv2 models modify diffusion and CLIP models, altering the denoising process. They can be used for style application and other fine-tuning tasks."

    def load_oftv2(self, model, clip, oftv2_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        oftv2_path = folder_paths.get_full_path_or_raise("loras", oftv2_name)
        oftv2_model = None
        if self.loaded_oft is not None:
            if self.loaded_oft[0] == oftv2_path:
                oftv2_model = self.loaded_oft[1]
            else:
                self.loaded_oft = None

        if oftv2_model is None:
            oftv2_model = comfy.utils.load_torch_file(oftv2_path, safe_load=True)
            self.loaded_oft = (oftv2_path, oftv2_model)

        model_oft, clip_oft = comfy.sd.load_lora_for_models(model, clip, oftv2_model, strength_model, strength_clip)
        return (model_oft, clip_oft)

class OFTv2LoaderModelOnly(OFTv2Loader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "lora_name": (folder_paths.get_filename_list("loras"), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_oftv2_model_only"

    def load_oftv2_model_only(self, model, lora_name, strength_model):
        return (self.load_oftv2(model, None, lora_name, strength_model, 0)[0],)

comfy.weight_adapter.adapters.append(OFTv2Adapter)

NODE_CLASS_MAPPINGS = {
    "OFTv2Loader": OFTv2Loader,
    "OFTv2LoaderModelOnly": OFTv2LoaderModelOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OFTv2Loader": "Load OFTv2",
    "OFTv2LoaderModelOnly": "Load OFTv2 Model Only",
}