"""
@author: Koratahiu
@title: ComfyUI-OFTv2
@nickname: ComfyUI-OFTv2
@description: A custom node to support and load OFTv2 adapters.
"""

import comfy.weight_adapter
from .node.oftv2_loader import OFTv2Adapter

# Register the OFTv2Adapter globally with ComfyUI's weight adapter system
comfy.weight_adapter.adapters.append(OFTv2Adapter)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}