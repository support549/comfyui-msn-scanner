"""
MSN moderation media loader — images and videos (representative frame sampling).

Drop-in replacement for LoadImage at API node 1. Keeps the same input key ``image``
so Next.js can continue injecting filenames unchanged.
"""

from __future__ import annotations

import mimetypes
import os

import av
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths
import comfy.model_management
import node_helpers
from comfy_api.latest import InputImpl


_VIDEO_EXTENSIONS = frozenset({
    "mp4", "mov", "avi", "mkv", "webm", "m4v", "wmv", "flv", "mpeg", "mpg", "3gp",
})


def _is_video_path(path: str) -> bool:
    mime, _ = mimetypes.guess_type(path, strict=False)
    if mime and mime.startswith("video/"):
        return True
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    return ext in _VIDEO_EXTENSIONS


def _representative_frame_indices(total: int) -> list[int]:
    """First, middle, and last frame indices (up to 3 unique indices)."""
    if total <= 0:
        return []
    if total == 1:
        return [0]
    if total == 2:
        return [0, 1]
    return [0, total // 2, total - 1]


def _pil_image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _load_static_image(image_path: str) -> torch.Tensor:
    dtype = comfy.model_management.intermediate_dtype()
    device = comfy.model_management.intermediate_device()

    components = InputImpl.VideoFromFile(image_path).get_components()
    if components.images.shape[0] > 0:
        if components.images.shape[0] > 1:
            print(
                f"MSN MEDIA: animated input with {components.images.shape[0]} frames — "
                "using first frame only for image path"
            )
        return components.images[:1].to(device=device, dtype=dtype)

    img = node_helpers.pillow(Image.open, image_path)
    output_images: list[torch.Tensor] = []
    w, h = None, None

    for frame in ImageSequence.Iterator(img):
        frame = node_helpers.pillow(ImageOps.exif_transpose, frame)
        rgb = frame.convert("RGB")
        if not output_images:
            w, h = rgb.size
        if rgb.size[0] != w or rgb.size[1] != h:
            continue
        arr = np.array(rgb).astype(np.float32) / 255.0
        output_images.append(torch.from_numpy(arr).unsqueeze(0))

    if not output_images:
        raise ValueError(f"MSN MEDIA: no decodable frames in image file '{image_path}'")

    if len(output_images) > 1:
        print(
            f"MSN MEDIA: animated image with {len(output_images)} frames — "
            "using first frame only"
        )

    batch = output_images[0].to(device=device, dtype=dtype)
    return batch


def _decode_frame_at_index(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
    index: int,
    fps: float,
) -> np.ndarray:
    time_base = float(stream.time_base)
    timestamp_sec = index / fps if fps > 0 else 0.0
    seek_pts = int(timestamp_sec / time_base) if time_base > 0 else 0
    container.seek(max(seek_pts, 0), stream=stream, backward=True)

    for frame in container.decode(stream):
        if isinstance(frame, av.VideoFrame):
            return frame.to_ndarray(format="rgb24")

    raise ValueError(f"MSN MEDIA: could not decode video frame at index {index}")


def _extract_video_frames(image_path: str, indices: list[int]) -> torch.Tensor:
    dtype = comfy.model_management.intermediate_dtype()
    device = comfy.model_management.intermediate_device()

    video = InputImpl.VideoFromFile(image_path)
    fps = float(video.get_frame_rate()) or 30.0

    frames_rgb: list[np.ndarray] = []
    with av.open(image_path, mode="r") as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise ValueError(f"MSN MEDIA: no video stream in '{image_path}'")

        for index in indices:
            rgb = _decode_frame_at_index(container, stream, index, fps)
            frames_rgb.append(rgb)

    tensors = [
        torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)
        for arr in frames_rgb
    ]
    batch = torch.cat(tensors, dim=0).to(device=device, dtype=dtype)
    return batch


def load_moderation_media(image_path: str) -> tuple[torch.Tensor, str, int]:
    """
    Load one image or sample up to 3 representative video frames.

    Returns (image_batch, media_type, frame_count).
    """
    if _is_video_path(image_path):
        video = InputImpl.VideoFromFile(image_path)
        total = max(1, int(video.get_frame_count()))
        indices = _representative_frame_indices(total)
        batch = _extract_video_frames(image_path, indices)
        extracted = batch.shape[0]
        print(
            f"MSN MEDIA: Video detected — extracted {extracted} frame(s) for moderation "
            f"(indices {indices} of {total} total in '{os.path.basename(image_path)}')"
        )
        return batch, "video", extracted

    batch = _load_static_image(image_path)
    print(f"MSN MEDIA: Image detected — 1 frame for moderation ('{os.path.basename(image_path)}')")
    return batch, "image", 1


class MSNMediaLoader:
    """
    API node 1 entry: image OR video filename (same ``image`` input key as LoadImage).

    Videos: first + middle + last frame (3 max). Images: single frame unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        image_files = folder_paths.filter_files_content_types(files, ["image"])
        video_files = folder_paths.filter_files_content_types(files, ["video"])
        options = sorted(set(image_files + video_files))
        return {
            "required": {
                "image": (options, {"image_upload": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("image", "media_type", "frame_count")
    FUNCTION = "load_media"
    CATEGORY = "My Secret Needs/Scanner"

    def load_media(self, image: str):
        image_path = folder_paths.get_annotated_filepath(image)
        batch, media_type, frame_count = load_moderation_media(image_path)
        return (batch, media_type, frame_count)

    @classmethod
    def IS_CHANGED(cls, image: str):
        image_path = folder_paths.get_annotated_filepath(image)
        return os.path.getmtime(image_path)

    @classmethod
    def VALIDATE_INPUTS(cls, image: str):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid media file: {image}"
        return True


NODE_CLASS_MAPPINGS = {
    "MSNMediaLoader": MSNMediaLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MSNMediaLoader": "MSN Media Loader (Image / Video)",
}
