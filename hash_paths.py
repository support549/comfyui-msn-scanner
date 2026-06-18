"""Resolve perceptual_hash module directory (sibling package or bundled vendor copy)."""

from __future__ import annotations

import os


def resolve_perceptual_hash_dir(scanner_dir: str | None = None) -> str:
    base = scanner_dir or os.path.dirname(os.path.realpath(__file__))
    sibling = os.path.normpath(os.path.join(base, "..", "msn-perceptual-hash"))
    if os.path.isfile(os.path.join(sibling, "perceptual_hash.py")):
        return sibling
    vendor = os.path.join(base, "vendor")
    if os.path.isfile(os.path.join(vendor, "perceptual_hash.py")):
        return vendor
    raise ImportError(
        "msn-perceptual-hash is not installed. Clone "
        "https://github.com/support549/msn-perceptual-hash.git into "
        "/comfyui/custom_nodes/msn-perceptual-hash or reinstall comfyui-msn-scanner "
        "with the bundled vendor copy."
    )


def sibling_perceptual_hash_installed(scanner_dir: str | None = None) -> bool:
    base = scanner_dir or os.path.dirname(os.path.realpath(__file__))
    sibling = os.path.normpath(os.path.join(base, "..", "msn-perceptual-hash"))
    return os.path.isfile(os.path.join(sibling, "perceptual_hash.py"))
