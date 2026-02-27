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
        scaled_oft: bool = False,
    ):
        self.weight = weight
        self.block_size = block_size
        self.coft = coft
        self.eps = eps
        self.use_cayley_neumann = use_cayley_neumann
        self.num_cayley_neumann_terms = num_cayley_neumann_terms
        self.use_scaled_oft = scaled_oft
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
        if self.use_scaled_oft:
            # Apply the scaled_oft factor: weight / sqrt(n_elements)
            n_elements = weight.shape[-1]
            effective_weight = weight / (n_elements ** 0.5)
        else:
            effective_weight = weight
        return self._cayley_batch(effective_weight)


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
        scaled_oft_flag_name = f"{x}.oft_R.scaled_oft"
        if oft_r_weight_name in lora:
            oft_r_weight = lora[oft_r_weight_name]
            loaded_keys.add(oft_r_weight_name)
            is_scaled = False
            if scaled_oft_flag_name in lora:
                is_scaled = True
                loaded_keys.add(scaled_oft_flag_name)

            # DoRA-OFT
            dora_scale_name = f"{x}.dora_scale"
            if dora_scale_name in lora:
                loaded_keys.add(dora_scale_name)
            initial_norm = None
            initial_norm_name = f"{x}.initial_norm"
            if initial_norm_name in lora:
                initial_norm = lora[initial_norm_name]
                loaded_keys.add(initial_norm_name)

            weights = (oft_r_weight, alpha, dora_scale, is_scaled, initial_norm)
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

        oft_r_weight_orig, alpha, dora_scale, is_scaled, initial_norm = self.weights

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
            util = OFTRotationUtil(oft_r_weight_processed, block_size, scaled_oft=is_scaled)
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
                # Reshape dora_scale to [out_features, 1, ...] for correct per-row broadcasting
                dora_scale = dora_scale.reshape(base_weight.shape[0], *([1] * (base_weight.dim() - 1)))
                dora_scale = dora_scale.to(weight.device, dtype=intermediate_dtype)
                # Use initial_norm from training when available, since dora_scale
                # was learned relative to the training base model's norms.
                if initial_norm is not None:
                    norm = initial_norm.reshape(base_weight.shape[0], *([1] * (base_weight.dim() - 1)))
                    norm = norm.to(weight.device, dtype=intermediate_dtype)
                elif base_weight.dim() == 4:
                    norm = torch.norm(base_weight.reshape(base_weight.shape[0], -1), dim=1).reshape(base_weight.shape[0], 1, 1, 1)
                else: # Linear
                    norm = torch.norm(base_weight, dim=1, keepdim=True)
                # W_rotated is original weight plus the OFT rotation difference
                W_rotated = base_weight + lora_diff
                # Scale the rotated weights to match the learned DoRA magnitude
                scale = dora_scale / norm
                W_dora = W_rotated * scale
                final_diff = W_dora - base_weight
                weight += function((final_diff * strength).type(weight.dtype))
            else:
                weight += function((lora_diff * strength).type(weight.dtype))

        except Exception as e:
            logging.error(f"ERROR applying OFTv2 for {key}: {e}", exc_info=True)

        return weight


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}