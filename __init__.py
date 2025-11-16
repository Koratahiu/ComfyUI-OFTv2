"""
@author: Koratahiu
@title: ComfyUI-OFTv2
@nickname: ComfyUI-OFTv2
@description: A custom node to support and load OFTv2 adapters.
"""

from .node import oftv2_loader

NODE_CLASS_MAPPINGS = {
    **oftv2_loader.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **oftv2_loader.NODE_DISPLAY_NAME_MAPPINGS,
}