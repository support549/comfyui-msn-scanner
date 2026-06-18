import os
import sys


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from comfydeploy_compat import install_validate_prompt_compat

install_validate_prompt_compat()

from hash_paths import resolve_perceptual_hash_dir, sibling_perceptual_hash_installed

if not sibling_perceptual_hash_installed():
    _vendor_dir = resolve_perceptual_hash_dir()
    if _vendor_dir not in sys.path:
        sys.path.insert(0, _vendor_dir)
    from perceptual_hash import MSNPerceptualHash

    VENDOR_HASH_CLASS_MAPPINGS = {
        "MSNPerceptualHash": MSNPerceptualHash,
    }
    VENDOR_HASH_DISPLAY_MAPPINGS = {
        "MSNPerceptualHash": "MSN Perceptual Hash (pHash / dHash)",
    }
else:
    VENDOR_HASH_CLASS_MAPPINGS = {}
    VENDOR_HASH_DISPLAY_MAPPINGS = {}

from msn_nodes import (
    NODE_CLASS_MAPPINGS as GATEWAY_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as GATEWAY_DISPLAY_MAPPINGS,
)
from content_filter import (
    NODE_CLASS_MAPPINGS as FILTER_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as FILTER_DISPLAY_MAPPINGS,
)
from media_loader import (
    NODE_CLASS_MAPPINGS as MEDIA_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MEDIA_DISPLAY_MAPPINGS,
)
from thumbnail_save import (
    NODE_CLASS_MAPPINGS as THUMB_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as THUMB_DISPLAY_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(GATEWAY_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(FILTER_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(MEDIA_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(THUMB_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(VENDOR_HASH_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(GATEWAY_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(FILTER_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(MEDIA_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(THUMB_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(VENDOR_HASH_DISPLAY_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
