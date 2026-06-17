"""
Save poster-frame thumbnails for MSN moderation (images + video keyframes).
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from PIL import Image

import folder_paths


def save_poster_thumbnail(
    image: torch.Tensor,
    filename_prefix: str = "msn_poster",
    poster_frame_index: int = 0,
) -> dict[str, Any]:
    """
    Save a single poster frame PNG to ComfyUI output directory.

    Returns ComfyUI image descriptor: filename, subfolder, type.
    """
    if image is None or not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] < 1:
        return {}

    index = max(0, min(int(poster_frame_index), int(image.shape[0]) - 1))
    frame = image[index]
    height = int(frame.shape[0])
    width = int(frame.shape[1])

    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, subfolder, _prefix = folder_paths.get_save_image_path(
        filename_prefix,
        output_dir,
        width,
        height,
    )

    arr = 255.0 * frame.detach().cpu().numpy()
    pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    file_name = f"{filename}_{counter:05}_.png"
    pil.save(os.path.join(full_output_folder, file_name), compress_level=4)

    descriptor = {
        "filename": file_name,
        "subfolder": subfolder,
        "type": "output",
        "poster_frame_index": index,
        "media_frames": int(image.shape[0]),
    }
    print(
        f"MSN THUMBNAIL: saved poster frame {index + 1}/{image.shape[0]} -> "
        f"{subfolder}/{file_name}" if subfolder else file_name
    )
    return descriptor


def attach_thumbnail_fields(
    moderation_output: dict[str, Any],
    thumbnail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(moderation_output)
    thumb = thumbnail or {}
    if not thumb.get("filename"):
        out["thumbnail_filename"] = ""
        out["thumbnail_subfolder"] = ""
        out["thumbnail_type"] = ""
        out["poster_frame_index"] = None
        out["media_frame_count"] = 0
        return out

    out["thumbnail_filename"] = str(thumb.get("filename") or "")
    out["thumbnail_subfolder"] = str(thumb.get("subfolder") or "")
    out["thumbnail_type"] = str(thumb.get("type") or "output")
    out["poster_frame_index"] = int(thumb.get("poster_frame_index", 0))
    out["media_frame_count"] = int(thumb.get("media_frames", 1))
    return out


class MSNPosterThumbnail:
    """
    Extract and save poster frame (first keyframe) for Next.js thumbnails.

    Wire after MSNMediaLoader. Videos use frame 0 (first of first/middle/last batch).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "msn_poster",
                        "tooltip": "Output PNG prefix in ComfyUI/output.",
                    },
                ),
                "poster_frame_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Frame index in batch to use as poster (0 = first keyframe).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("poster_image", "thumbnail_info")
    FUNCTION = "save_poster"
    CATEGORY = "My Secret Needs/Scanner"

    def save_poster(self, image: torch.Tensor, filename_prefix: str = "msn_poster", poster_frame_index: int = 0):
        descriptor = save_poster_thumbnail(
            image,
            filename_prefix=filename_prefix or "msn_poster",
            poster_frame_index=poster_frame_index,
        )
        index = int(descriptor.get("poster_frame_index", 0)) if descriptor else 0
        if image is not None and image.shape[0] > index:
            poster = image[index : index + 1]
        else:
            poster = image[0:1] if image is not None and image.shape[0] > 0 else image
        return (poster, json.dumps(descriptor, ensure_ascii=False))


NODE_CLASS_MAPPINGS = {
    "MSNPosterThumbnail": MSNPosterThumbnail,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MSNPosterThumbnail": "MSN Poster Thumbnail (video/image)",
}
