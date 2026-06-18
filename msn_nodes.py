import json
from typing import Any

from content_filter import (
    MODERATION_MODEL,
    _normalize_caption,
    _scanner_bypass_enabled,
    attach_underage_fields,
    build_moderation_output,
    normalize_moderation_contract,
    scan_caption,
)
import os
import sys

from hash_paths import resolve_perceptual_hash_dir

_HASH_PKG = resolve_perceptual_hash_dir()
if _HASH_PKG not in sys.path:
    sys.path.insert(0, _HASH_PKG)

from perceptual_hash import (
    CORE_MODERATION_FIELDS,
    attach_perceptual_hash_fields,
    compare_against_references,
    parse_reference_hashes,
)
from thumbnail_save import attach_thumbnail_fields, save_poster_thumbnail


class MSNModerationGateway:
    """
    Final OUTPUT_NODE for API workflows (node 99).

    When DrugSafetyFilter is wired in, pass through its moderation_output JSON.
    History exposes outputs["99"]["moderation_output"][0] (value must live under "ui").
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Florence-2 caption (fallback scan when filter not wired).",
                    },
                ),
                "metadata_flag": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "bypass_scan": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Skip scan and always return approved moderation_output for Next.js.",
                    },
                ),
                "is_safe": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "From DrugSafetyFilter output 0.",
                    },
                ),
                "filter_reason": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "From DrugSafetyFilter output 1 (reason).",
                    },
                ),
                "safe_image": (
                    "IMAGE",
                    {"tooltip": "From DrugSafetyFilter output 2 — chains execution order."},
                ),
                "blocked_image": (
                    "IMAGE",
                    {"tooltip": "From DrugSafetyFilter output 3 — chains execution order."},
                ),
                "filter_moderation_output": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "From DrugSafetyFilter output 4 — authoritative verdict JSON.",
                    },
                ),
                "perceptual_hash": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "From MSNPerceptualHash output 0 — primary frame hash.",
                    },
                ),
                "hash_details": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "From MSNPerceptualHash output 1 — JSON with frame_hashes for video.",
                    },
                ),
                "reference_hashes": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Known-bad or prior-upload hashes (hex, comma/newline separated) from Next.js.",
                    },
                ),
                "hamming_threshold": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Hamming distance at or below this flags potential_duplicate.",
                    },
                ),
                "thumbnail_info": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "From MSNPosterThumbnail output 1 — saved poster frame descriptor JSON.",
                    },
                ),
                "source_image": (
                    "IMAGE",
                    {
                        "tooltip": "Optional MSNMediaLoader batch — fallback poster save when thumbnail_info not wired.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("moderation_output",)
    FUNCTION = "evaluate"
    OUTPUT_NODE = True
    CATEGORY = "My Secret Needs/Scanner"

    def evaluate(
        self,
        tags="",
        metadata_flag="",
        bypass_scan=False,
        is_safe=True,
        filter_reason="",
        safe_image=None,
        blocked_image=None,
        filter_moderation_output="",
        perceptual_hash="",
        hash_details="",
        reference_hashes="",
        hamming_threshold=10,
        thumbnail_info="",
        source_image=None,
    ):
        _ = blocked_image  # execution chain only

        caption_norm = _normalize_caption(tags)
        print("\n================ MSN SCANNER DEBUG ================")
        print(f"RAW CAPTION / TAGS: {caption_norm[:500]}")
        print(f"FILTER is_safe={is_safe} reason={filter_reason[:120]!r}")
        print("===================================================\n")

        upstream = str(filter_moderation_output or "").strip()
        if upstream:
            try:
                parsed = json.loads(upstream)
                if not isinstance(parsed, dict):
                    raise ValueError("filter_moderation_output must be a JSON object")
            except (json.JSONDecodeError, ValueError):
                parsed = {}
            moderation_output = normalize_moderation_contract(
                parsed,
                is_safe=bool(is_safe),
                reason=filter_reason,
            )
            if "is_underage" not in moderation_output:
                _, caption_violations = scan_caption(caption_norm, strictness=2)
                moderation_output = attach_underage_fields(moderation_output, caption_violations)
        elif bypass_scan or _scanner_bypass_enabled():
            print("MSN SCANNER: bypass_scan enabled — Image is SAFE")
            moderation_output = build_moderation_output(False, [], metadata_flag)
        else:
            blocked, violations = scan_caption(caption_norm, strictness=2)
            moderation_output = build_moderation_output(blocked, violations, metadata_flag)

        hash_payload: dict[str, Any] = {}
        details_raw = str(hash_details or "").strip()
        if details_raw:
            try:
                parsed_details = json.loads(details_raw)
                if isinstance(parsed_details, dict):
                    hash_payload = parsed_details
            except json.JSONDecodeError:
                hash_payload = {}

        primary_hash = str(perceptual_hash or hash_payload.get("perceptual_hash") or "").strip()
        frame_hashes = hash_payload.get("frame_hashes") or []
        if not isinstance(frame_hashes, list):
            frame_hashes = []
        if primary_hash and not frame_hashes:
            frame_hashes = [primary_hash]
        hash_algorithm = str(hash_payload.get("algorithm") or "phash")

        refs = parse_reference_hashes(reference_hashes)
        comparison = compare_against_references(
            [str(h) for h in frame_hashes if h],
            refs,
            threshold=int(hamming_threshold),
        )

        moderation_output = attach_perceptual_hash_fields(
            moderation_output,
            perceptual_hash=primary_hash,
            hash_algorithm=hash_algorithm,
            frame_hashes=[str(h) for h in frame_hashes if h] or None,
            similarity_score=float(comparison["similarity_score"]),
            potential_duplicate=bool(comparison["potential_duplicate"]),
            hamming_distance=comparison["hamming_distance"],
            matched_reference_hash=str(comparison.get("matched_reference_hash") or ""),
            matched_frame_index=comparison.get("matched_frame_index"),
        )

        thumbnail_descriptor: dict[str, Any] = {}
        thumb_raw = str(thumbnail_info or "").strip()
        if thumb_raw:
            try:
                parsed_thumb = json.loads(thumb_raw)
                if isinstance(parsed_thumb, dict) and parsed_thumb.get("filename"):
                    thumbnail_descriptor = parsed_thumb
            except json.JSONDecodeError:
                thumbnail_descriptor = {}
        if not thumbnail_descriptor:
            poster_source = source_image if source_image is not None else safe_image
            thumbnail_descriptor = save_poster_thumbnail(poster_source, filename_prefix="msn_poster")

        moderation_output = attach_thumbnail_fields(moderation_output, thumbnail_descriptor)

        if comparison["potential_duplicate"]:
            print(
                f"MSN HASH: potential duplicate — distance={comparison['hamming_distance']} "
                f"score={comparison['similarity_score']} ref={comparison.get('matched_reference_hash', '')[:16]}"
            )

        if moderation_output.get("is_underage"):
            print(
                f"MSN UNDERAGE: is_underage={moderation_output.get('is_underage')} "
                f"underage_ai={moderation_output.get('underage_ai')} "
                f"reason={str(moderation_output.get('underage_reason', ''))[:120]}"
            )

        assert moderation_output["model"] == MODERATION_MODEL
        assert CORE_MODERATION_FIELDS.issubset(moderation_output.keys())

        ui_message = (
            "MSN Local Scan Complete - Flagged for Review"
            if moderation_output["flagged"]
            else "MSN Local Scan Complete - Clear Pass"
        )

        moderation_output_json = json.dumps(moderation_output, ensure_ascii=False)

        ui_payload: dict[str, Any] = {
            "text": [ui_message],
            "moderation_output": [moderation_output_json],
        }
        if thumbnail_descriptor.get("filename"):
            ui_payload["images"] = [thumbnail_descriptor]

        return {
            "ui": ui_payload,
            "result": (moderation_output_json,),
        }


NODE_CLASS_MAPPINGS = {
    "MSNModerationGateway": MSNModerationGateway,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MSNModerationGateway": "MSN Moderation Gateway",
}
