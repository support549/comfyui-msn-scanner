"""
Fast perceptual hashing (pHash / dHash) for MSN duplicate detection.

Uses downscaled grayscale frames — suitable for images and video keyframe batches
from MSNMediaLoader (first / middle / last).
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.fftpack import dct

CORE_MODERATION_FIELDS = frozenset({
    "approved",
    "flagged",
    "moderation_status",
    "reason",
    "model",
})

_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


def _tensor_frame_to_pil(image: torch.Tensor, index: int = 0) -> Image.Image:
    frame = image[index].detach().cpu().numpy()
    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(frame, mode="RGB")


def _grayscale_array(pil_image: Image.Image, max_side: int) -> np.ndarray:
    gray = pil_image.convert("L")
    if max_side > 0:
        w, h = gray.size
        scale = max_side / float(max(w, h))
        if scale < 1.0:
            gray = gray.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.BILINEAR,
            )
    return np.asarray(gray, dtype=np.float32)


def compute_dhash(pil_image: Image.Image, max_side: int = 256) -> str:
    gray = _grayscale_array(pil_image, max_side)
    small = Image.fromarray(gray.astype(np.uint8)).resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.float32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = "".join("1" if bit else "0" for bit in diff.flatten())
    return f"{int(bits, 2):016x}"


def compute_phash(pil_image: Image.Image, max_side: int = 256) -> str:
    gray = _grayscale_array(pil_image, max_side)
    small = Image.fromarray(gray.astype(np.uint8)).resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.float32)
    dct_rows = dct(pixels, axis=0, norm="ortho")
    dct_2d = dct(dct_rows, axis=1, norm="ortho")
    low_freq = dct_2d[:8, :8].copy()
    low_freq[0, 0] = 0.0
    median = float(np.median(low_freq))
    bits = "".join("1" if value > median else "0" for value in low_freq.flatten())
    return f"{int(bits, 2):016x}"


def compute_hash_hex(
    pil_image: Image.Image,
    algorithm: str = "phash",
    max_side: int = 256,
) -> str:
    algo = (algorithm or "phash").strip().lower()
    if algo == "dhash":
        return compute_dhash(pil_image, max_side)
    if algo == "both":
        return f"{compute_phash(pil_image, max_side)}:{compute_dhash(pil_image, max_side)}"
    return compute_phash(pil_image, max_side)


def hamming_distance_hex(a: str, b: str) -> int:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return 64
    if ":" in a or ":" in b:
        parts_a = a.split(":")
        parts_b = b.split(":")
        size = max(len(parts_a), len(parts_b))
        distances = []
        for i in range(size):
            pa = parts_a[i] if i < len(parts_a) else ""
            pb = parts_b[i] if i < len(parts_b) else ""
            if pa and pb and len(pa) == len(pb):
                distances.append(hamming_distance_hex(pa, pb))
        return min(distances) if distances else 64
    if len(a) != len(b):
        return 64
    try:
        ai = int(a, 16)
        bi = int(b, 16)
    except ValueError:
        return 64
    return (ai ^ bi).bit_count()


def similarity_score_from_hamming(hamming: int, bits: int = 64) -> float:
    bits = max(1, int(bits))
    return round(max(0.0, 1.0 - (float(hamming) / float(bits))), 4)


def parse_reference_hashes(raw: str) -> list[str]:
    if not raw:
        return []
    tokens = re.split(r"[\s,;]+", str(raw).strip())
    refs: list[str] = []
    for token in tokens:
        token = token.strip().lower()
        if not token:
            continue
        if _HEX_RE.match(token.replace(":", "")):
            refs.append(token)
    return refs


def compare_against_references(
    frame_hashes: list[str],
    reference_hashes: list[str],
    threshold: int = 10,
) -> dict[str, Any]:
    if not frame_hashes or not reference_hashes:
        return {
            "hamming_distance": None,
            "similarity_score": 0.0,
            "potential_duplicate": False,
            "matched_reference_hash": "",
            "matched_frame_index": None,
        }

    best_distance = 64
    best_ref = ""
    best_frame = 0
    for frame_index, candidate in enumerate(frame_hashes):
        for ref in reference_hashes:
            distance = hamming_distance_hex(candidate, ref)
            if distance < best_distance:
                best_distance = distance
                best_ref = ref
                best_frame = frame_index

    potential = best_distance <= max(0, int(threshold))
    return {
        "hamming_distance": best_distance,
        "similarity_score": similarity_score_from_hamming(best_distance),
        "potential_duplicate": potential,
        "matched_reference_hash": best_ref,
        "matched_frame_index": best_frame,
    }


def hash_image_batch(
    image: torch.Tensor,
    algorithm: str = "phash",
    max_side: int = 256,
) -> dict[str, Any]:
    if image is None or image.ndim != 4 or image.shape[0] < 1:
        return {
            "perceptual_hash": "",
            "algorithm": algorithm,
            "frame_hashes": [],
            "frame_count": 0,
            "primary_frame_index": 0,
        }

    frame_count = int(image.shape[0])
    frame_hashes: list[str] = []
    for index in range(frame_count):
        pil = _tensor_frame_to_pil(image, index)
        frame_hashes.append(compute_hash_hex(pil, algorithm=algorithm, max_side=max_side))

    return {
        "perceptual_hash": frame_hashes[0],
        "algorithm": (algorithm or "phash").strip().lower(),
        "frame_hashes": frame_hashes,
        "frame_count": frame_count,
        "primary_frame_index": 0,
    }


def attach_perceptual_hash_fields(
    moderation_output: dict[str, Any],
    *,
    perceptual_hash: str = "",
    hash_algorithm: str = "phash",
    frame_hashes: list[str] | None = None,
    similarity_score: float = 0.0,
    potential_duplicate: bool = False,
    hamming_distance: int | None = None,
    matched_reference_hash: str = "",
    matched_frame_index: int | None = None,
) -> dict[str, Any]:
    out = dict(moderation_output)
    out["perceptual_hash"] = perceptual_hash or ""
    out["similarity_score"] = float(similarity_score)
    out["potential_duplicate"] = bool(potential_duplicate)
    if hash_algorithm:
        out["perceptual_hash_algorithm"] = hash_algorithm
    if frame_hashes:
        out["frame_hashes"] = list(frame_hashes)
    if hamming_distance is not None:
        out["hamming_distance"] = int(hamming_distance)
    if matched_reference_hash:
        out["matched_reference_hash"] = matched_reference_hash
    if matched_frame_index is not None:
        out["matched_frame_index"] = int(matched_frame_index)
    return out


class MSNPerceptualHash:
    """Generate perceptual hash(es) from IMAGE batches (image or video keyframes)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "algorithm": (
                    ["phash", "dhash", "both"],
                    {
                        "default": "phash",
                        "tooltip": "phash = DCT perceptual (robust). dhash = faster difference hash.",
                    },
                ),
                "max_side": (
                    "INT",
                    {
                        "default": 256,
                        "min": 32,
                        "max": 1024,
                        "step": 32,
                        "tooltip": "Downscale longest side before hashing (speed vs accuracy).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("perceptual_hash", "hash_details")
    FUNCTION = "compute"
    CATEGORY = "My Secret Needs/Scanner"

    def compute(self, image: torch.Tensor, algorithm: str = "phash", max_side: int = 256):
        payload = hash_image_batch(image, algorithm=algorithm, max_side=max_side)
        details_json = json.dumps(payload, ensure_ascii=False)
        print(
            f"MSN HASH: {payload['frame_count']} frame(s), algo={payload['algorithm']}, "
            f"primary={payload['perceptual_hash'][:16]}..."
        )
        return (payload["perceptual_hash"], details_json)
