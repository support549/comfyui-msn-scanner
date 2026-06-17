"""
MSN Drug / NSFW Content Filter + Safety Gate for ComfyUI

Pipeline:
  Flux/SD image -> LoadImage -> Florence2Run (more_detailed_caption, empty prompt)
            -> MSNSafetyGate (caption scan + route save) -> approved / blocked outputs

Edit tiered drug term sets below (DRUG_TERMS_STRICT / STANDARD / PHRASES / VISUAL_HINTS).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import folder_paths


# ==================== TIERED CAPTION POLICY FILTER (drugs + weapons) ====================
# Level 1: hard drugs only (low FP on adult/NSFW captions)
# Level 2: + slang, cannabis, pills, injection, weapons singles/phrases
# Level 3: + extra visual hints, ambiguous singles (pipe, pill), weapon scene phrases

DRUG_TERMS_STRICT = {
    "heroin", "fentanyl", "morphine", "opium", "oxycodone", "hydrocodone",
    "codeine", "tramadol", "methadone", "buprenorphine",
    "cocaine", "crack", "meth", "methamphetamine", "amphetamine",
    "lsd", "ketamine", "pcp", "dmt", "psilocybin",
    "xanax", "alprazolam", "valium", "diazepam", "ativan", "lorazepam",
    "oxycontin", "percocet", "vicodin", "adderall", "ritalin",
    "ecstasy", "mdma", "overdose", "narcotic", "narcotics",
}

DRUG_TERMS_STANDARD = {
    "coke", "blow", "snow", "yayo", "speed", "crank", "ice",
    "acid", "tabs", "shrooms", "mushrooms", "molly",
    "percs", "perc", "xans", "benzos", "lean", "sizzurp",
    "dope", "smack", "tar", "rock", "freebase",
    "weed", "marijuana", "cannabis", "hashish", "hash", "edibles",
    "joint", "blunt", "bong", "roach", "buds", "nug", "nugs",
    "pills", "pill", "tablets", "tablet", "capsules", "capsule",
    "medication", "medications", "pharmaceutical", "pharmaceuticals",
    "syringe", "needle", "injecting", "injection", "tourniquet",
    "snorting", "huffing",
}

DRUG_TERMS_LEVEL3_ONLY = {
    "pipe",
}

FLORENCE_MODERATION_PROMPT = (
    "Estimate the apparent age of every person in years (e.g. 8 years old, 16 years old, 25 years old). "
    "State clearly if they look under 18, a teenager, or an adult. "
    "Note if the image looks AI-generated, CGI, anime, cartoon, or synthetic. "
    "Describe any pills, capsules, tablets, syringes, powders, or pharmaceutical items in extreme detail. "
    "Do not call them candies or sweets. "
    "Describe each person in detail including apparent age, gender, clothing, and pose. "
    "Describe any plants, dried herbs, green buds, substances or materials in extreme detail and name them specifically."
)

CANNABIS_PHRASES = frozenset({
    "dried cannabis", "dried herb", "dried herbs", "dried plant material",
    "dried green plant", "dried marijuana", "dried bud", "dried buds",
    "dried plant", "dried leaves", "dried leafy",
    "green buds", "green bud", "plant buds", "cannabis buds", "weed buds", "marijuana buds",
    "cannabis bud", "flower buds", "cannabis flower", "cannabis flowers",
    "dried leaves and buds", "small green pieces", "crumbly green",
    "handling dried", "handling weed", "drying cannabis", "trimming cannabis",
    "bag of weed", "bunch of weed", "bag of cannabis",
    "marijuana plant", "cannabis plant", "weed plant",
    "cannabis leaf", "cannabis leaves", "marijuana leaf", "marijuana leaves",
    "plant material in hand", "green plant material", "leafy green plant",
    "cluster of buds", "holding buds", "hands holding buds",
    "cannabis in hand", "weed in hand", "marijuana in hand",
    "cannabis like", "weed like substance", "marijuana like",
    # Visual descriptions Florence often uses instead of drug vocabulary
    "leafy green substance", "green leafy substance", "leafy green plant",
    "green plant with serrated leaves", "serrated leaves", "green leaves",
    "pile of green leaves", "green herbs", "herbal material", "plant material",
    "green nug", "green nugs", "handful of green", "pile of green",
    "green material in", "leafy material", "leafy green",
    # Temporary broad visual triggers (Level 2) — tune down if false positives appear
    "holding a small", "small green", "green material", "dried green", "buds in hand",
})

BANNED_DRUG_PHRASES = {
    "colorful pills", "scattered pills", "pills scattered", "pills on table",
    "tablets scattered", "ecstasy pills", "xanax bars", "bag of pills",
    "pill bottle", "pill bottle scattered", "pharmaceutical pills",
    "pile of pills", "assorted pills", "round tablets", "capsules scattered",
    "colorful candies", "multicolored candies", "scattered candies", "assorted candies",
    "round candies", "colorful candy", "pile of candies", "bright colored candies",
    "small colorful candies", "colorful sweets", "scattered sweets", "assorted sweets",
    "gel capsules", "gel capsule", "two-tone capsules", "capsule-shaped", "pill-shaped",
    "prescription pills", "prescription medication", "blister pack", "blister packs",
    "round colorful objects", "colorful round objects", "small round colorful",
    "tie off arm", "tourniquet on arm", "arm tie off", "injecting arm",
    "arm injection", "vein injection", "hypodermic needle", "used syringe",
    "needle mark", "track marks", "spoon and lighter", "cooking drugs",
    "burnt spoon", "syringe on table", "needle on table", "holding syringe",
    "lines of coke", "bag of coke", "bag of meth", "crystal meth", "meth lab",
    "drug deal", "drug dealer", "doing lines", "snorting lines", "white lines",
    "cutting powder", "razor blade and powder", "powder on mirror",
    "rolled up bill", "rolled bill", "glass pipe", "crack pipe", "meth pipe",
    "smoking crack", "chasing the dragon", "foil strip", "burnt foil",
    "rolling paper", "rolling papers",
    "purple drank", "lean drink", "codeine syrup",
    "white powder", "powder residue", "powder lines", "line of powder",
    "bag of white powder", "powder on table",
}

DRUG_VISUAL_HINTS = {
    "lines on mirror", "mirror with powder", "powder arranged in lines",
    "substance on spoon", "lighter under spoon", "foil with burn marks",
    "bag of powder", "scale with powder", "drug scale", "digital scale and bag",
    "rolled banknote", "straw on mirror", "snorting straw",
    "baggie of powder", "zip lock bag of white", "small plastic bags of powder",
    "drug paraphernalia", "paraphernalia on", "substance abuse",
    "smoking pipe with", "glass stem pipe",
    "trimmed cannabis",
    "multicolored pills", "bright colored pills", "pills in palm",
    "colorful round pills", "round colorful pills", "assorted capsules",
    "scattered capsules", "tablets in palm", "capsules in palm",
}

# Level-2 visual hints (promoted from L3-only subset — cannabis / substance wording)
DRUG_VISUAL_HINTS_LEVEL2 = frozenset({
    "leafy green substance",
    "green leafy substance",
    "leafy green plant",
    "green plant with leaves",
    "serrated leaves",
    "herbal material",
    "plant material on",
    "green substance on",
    "crystalline substance",
    "sticky green",
    "aromatic herbs",
})

_DRUG_HARD = frozenset(DRUG_TERMS_STRICT)
_DRUG_CANNABIS = frozenset({
    "weed", "marijuana", "cannabis", "hashish", "hash", "edibles",
    "joint", "blunt", "bong", "roach", "buds", "nug", "nugs",
})
_DRUG_PILLS = frozenset({"pills", "pill", "tablets", "tablet", "capsules", "capsule"})
_DRUG_INJECTION = frozenset({
    "syringe", "needle", "injecting", "injection", "tourniquet",
})
_CANNABIS_PHRASE_MARKERS = (
    "cannabis", "weed", "marijuana", "buds", "nug", "dried herb", "plant material",
    "dried green", "green plant", "flower bud", "crumbly green", "small green",
    "like substance", "handling dried", "plant buds",
    "holding a small", "green material", "dried green", "buds in hand",
    "leafy green", "leafy material", "serrated leaves", "herbal material", "green herbs",
    "green substance", "green leaves", "dried plant",
)
_PILL_PHRASE_MARKERS = (
    "pill", "tablet", "ecstasy pill", "xanax bar", "pharmaceutical", "capsule",
    "candy", "candies", "sweet", "sweets", "blister", "prescription",
)
_INJECTION_PHRASE_MARKERS = (
    "syringe", "needle", "inject", "injection", "track mark", "spoon and lighter",
    "cooking drug", "tie off", "tourniquet", "vein",
)

WEAPON_TERMS_LEVEL2 = frozenset({
    "gun", "guns", "firearm", "firearms", "pistol", "rifle", "shotgun",
    "revolver", "handgun", "weapon", "weapons", "ammunition", "ammo",
    "bullet", "bullets", "cartridge", "magazine", "silencer", "suppressor",
    "ak47", "ak-47", "ar15", "ar-15", "glock", "uzi", "mp5", "sniper",
    "9mm", "caliber", "m16", "m4",
})

WEAPON_TERMS_LEVEL3_ONLY = frozenset()

WEAPON_PHRASES = frozenset({
    "holding gun", "holding a gun", "holding the gun",
    "holding a pistol", "holding a rifle", "holding a shotgun",
    "pointing gun", "pointing a gun", "aiming gun", "aiming a gun",
    "pointing a pistol", "pointing a rifle", "pointing a shotgun",
    "with a gun", "with gun", "with firearm", "with pistol", "with rifle",
    "carrying gun", "carrying a gun", "carrying a pistol", "carrying a rifle",
    "carrying a weapon",
    "gun in hand", "weapon in hand", "firearm in hand", "pistol in hand",
    "rifle in hand", "shotgun in hand", "handgun in",
    "gun in his hand", "gun in her hand", "gun in their hand",
    "armed with", "holding weapon", "holding firearm", "holding rifle",
    "holding pistol", "holding handgun", "brandishing", "brandishing a",
    "assault rifle", "semi-automatic", "machine gun", "submachine gun",
    "holstered gun", "gun holster", "holstered pistol", "holstered weapon",
    "drawn gun", "gun pointed", "gun aimed", "weapon aimed", "firearm aimed",
    "gun on table", "rifle on table", "pistol on table", "weapon on table",
    "gun visible", "visible gun", "firearm visible",
    "black gun", "silver gun",
    "aiming pistol", "aiming rifle",
    "shooting gun", "firing gun",
    # Promoted from L3 visual hints (Level 2)
    "man with gun", "woman with gun", "person with gun",
    "man with rifle", "woman with rifle", "person with rifle",
    "man holding gun", "woman holding gun", "person holding gun",
    "armed man", "armed woman", "armed person",
    "pointing a rifle", "pointing a pistol", "aiming a rifle",
})

WEAPON_VISUAL_HINTS = frozenset({
    "gun to head", "gun at camera", "pointed at camera",
    "gun in foreground", "rifle slung", "tactical vest and rifle",
})

_WEAPON_PHRASE_MARKERS = (
    "gun", "firearm", "weapon", "rifle", "pistol", "shotgun", "revolver",
    "ammunition", "armed", "holster", "pointing", "aiming",
)

# Caption keyword block for minors (strictness 1–3; no bare "boy"/"girl" — high FP on adult copy).
BANNED_MINOR_TERMS = {
    "little girl",
    "little boy",
    "young child",
    "young girl",
    "young boy",
    "young asian girl",
    "young asian boy",
    "asian girl",
    "asian boy",
    "teen girl",
    "teen boy",
    "under age",
    "child",
    "children",
    "childlike",
    "kid",
    "kids",
    "baby",
    "infant",
    "toddler",
    "newborn",
    "minor",
    "minors",
    "juvenile",
    "underage",
    "preteen",
    "pre-teen",
    "schoolgirl",
    "schoolboy",
    "school girl",
    "school boy",
    "loli",
    "lolita",
    "shota",
    "teenager",
    "adolescent",
    "pedophile",
    "pedophilia",
    "paedophile",
    "jailbait",
}

# Florence-friendly phrases (substring match); hyphen variants for "african-american".
BANNED_MINOR_PHRASES = frozenset({
    "african american boy",
    "african-american boy",
    "black boy",
    "young african american boy",
    "young african-american boy",
    "young black boy",
    "african american girl",
    "african-american girl",
    "black girl",
    "young african american girl",
    "young african-american girl",
    "young black girl",
    "young boy",
    "little boy",
    "young girl",
    "little girl",
    "teenage girl",
    "teenage boy",
    "young african",
    "young african-american",
    "young black",
    "young child",
    "child smiling",
    "boy smiling",
    "girl smiling",
    "teen boy",
    "teen girl",
    "late teens",
    "early teens",
    "childlike face",
    "young face",
    "youthful face",
    "youthful appearance",
    "school uniform",
    "wearing a school uniform",
    "girl is posing",
    "girl posing",
    "boy is posing",
    "boy posing",
    "posing for the camera",
    "young asian girl",
    "young asian boy",
    "cute girl",
    "cute boy",
    "little kid",
    # Florence mislabels two minors as parent + child (e.g. "mother and daughter")
    "mother and daughter",
    "mom and daughter",
    "mother and son",
    "mom and son",
    "father and daughter",
    "father and son",
    "portrait of a mother and daughter",
    "portrait of a mother and son",
    "portrait of a father and daughter",
    "mother and her daughter",
    "father and his son",
    "daughter is sitting next",
    "son is sitting next",
    # AI / synthetic youth character phrasing
    "anime girl",
    "anime boy",
    "anime character",
    "manga girl",
    "manga boy",
    "cartoon girl",
    "cartoon boy",
    "cgi girl",
    "cgi boy",
    "3d rendered girl",
    "3d rendered boy",
    "ai generated girl",
    "ai generated boy",
    "ai-generated girl",
    "ai-generated boy",
    "digital illustration of a young girl",
    "digital illustration of a young boy",
    "illustration of a young girl",
    "illustration of a young boy",
    "young anime",
    "young cartoon",
})

# AI / synthetic rendering cues — used with age estimates and youth subjects.
AI_SYNTHETIC_CUES = frozenset({
    "ai generated",
    "ai-generated",
    "ai generated image",
    "generated by ai",
    "computer generated",
    "computer-generated",
    "cgi",
    "3d render",
    "3d rendered",
    "digital art",
    "digital illustration",
    "digital painting",
    "anime",
    "manga",
    "cartoon",
    "cartoonish",
    "illustration",
    "illustrated",
    "synthetic",
    "hyperrealistic",
    "uncanny",
    "stable diffusion",
    "midjourney",
    "novelai",
    "waifu",
    "rendered character",
    "virtual character",
    "game character",
})

# Regex: extract numeric apparent ages from Florence captions.
_AGE_YEARS_OLD_RE = re.compile(
    r"\b(\d{1,2})\s*(?:years?\s*old|y/?o\b)\b",
    re.IGNORECASE,
)
_AGE_HYPHEN_RE = re.compile(
    r"\b(\d{1,2})[-\s]year[-\s]old\b",
    re.IGNORECASE,
)
_AGE_EXPLICIT_RE = re.compile(
    r"\b(?:age[d]?|approximately|about|around|appears? to be|looks? to be|looks? like)\s+(\d{1,2})\b",
    re.IGNORECASE,
)

# Soft minor signals — only flag when paired with pose/clothing context (see _minor_context_boosted_match).
MINOR_SOFT_SIGNALS = frozenset({
    "young woman",
    "young man",
    "early 20s",
    "early twenties",
    "in her early twenties",
    "in his early twenties",
    "looks young",
    "appears young",
})

MINOR_CONTEXT_BOOSTERS = frozenset({
    "bikini",
    "bikini top",
    "bikini bottom",
    "lingerie",
    "underwear",
    "swimwear",
    "posing",
    "posing for the camera",
    "modeling",
    "suggestive",
    "skimpy",
    "revealing outfit",
    "revealing clothing",
    "black bikini",
    "wearing a bikini",
    "school uniform",
    "cute",
    "youthful",
    "childlike",
})

# Regex: flexible "young [0-4 words] girl/boy/child/kid" — "young asian girl" etc.
_MINOR_YOUNG_SUBJECT_RE = re.compile(
    r"\byoung\b(?:\s+\w+){0,4}\s+(?:girl|boy|child|kid)\b",
    re.IGNORECASE,
)
_MINOR_FLEXIBLE_YOUNG_GIRL_RE = re.compile(
    r"\byoung\b(?:\s+\w+){0,4}\s+girl\b",
    re.IGNORECASE,
)
_MINOR_FLEXIBLE_YOUNG_BOY_RE = re.compile(
    r"\byoung\b(?:\s+\w+){0,4}\s+boy\b",
    re.IGNORECASE,
)

# Adult-age words — if present, skip family-pair heuristic (real adults in caption).
_MINOR_FAMILY_ADULT_MARKERS = (
    "elderly", "middle-aged", "middle aged", "adult woman", "adult man",
    "grown woman", "grown man", "mature woman", "mature man", "senior ",
    "in her twenties", "in his twenties", "in her 30s", "in his 30s",
    "in her thirties", "in his thirties", "in her forties", "in his forties",
    "middle-aged woman", "middle-aged man",
)


def _normalize_caption(caption: Any) -> str:
    if isinstance(caption, list):
        caption = " ".join(str(item) for item in caption if item is not None)
    return str(caption or "").strip().lower()


def _florence_captions(caption: Any) -> list[str]:
    """Split Florence-2 output into per-frame caption strings."""
    if isinstance(caption, list):
        return [str(item).strip() for item in caption if str(item or "").strip()]
    text = str(caption or "").strip()
    return [text] if text else [""]


def _aggregate_frame_scans(
    frame_results: list[tuple[bool, list]],
) -> tuple[bool, list]:
    """Most severe wins: any frame flagged blocks the asset."""
    blocked = False
    violations: list[str] = []
    seen: set[str] = set()
    for is_blocked, matched in frame_results:
        if is_blocked:
            blocked = True
        for item in matched:
            key = str(item)
            if key not in seen:
                seen.add(key)
                violations.append(key)
    return blocked, violations


def scan_captions_per_frame(
    captions: list[str],
    strictness: int = 2,
    extra_keywords: str = "",
) -> tuple[bool, list, int | None]:
    """Run policy scan on each frame caption; aggregate with worst-case logic."""
    if _scanner_bypass_enabled():
        _log_scanner_safe()
        print("MSN SCANNER: Global bypass active (MSN_SCANNER_BYPASS=1)")
        return False, [], None

    frame_results: list[tuple[bool, list]] = []
    first_blocked_index: int | None = None
    for index, caption in enumerate(captions):
        blocked, matched = scan_caption(
            _normalize_caption(caption),
            strictness,
            extra_keywords,
        )
        frame_results.append((blocked, matched))
        if blocked and first_blocked_index is None:
            first_blocked_index = index
        if len(captions) > 1:
            status = "FLAGGED" if blocked else "PASS"
            print(
                f"MSN SCANNER: Frame {index + 1}/{len(captions)} {status}"
                + (f" — {matched}" if matched else "")
            )

    blocked, violations = _aggregate_frame_scans(frame_results)
    if len(captions) > 1:
        summary = "FLAGGED" if blocked else "PASS"
        print(
            f"MSN SCANNER: Aggregated {len(captions)}-frame verdict — {summary}"
            + (f" ({len(violations)} violation(s))" if violations else "")
        )
    return blocked, violations, first_blocked_index


def _strictness_from_int(level: int) -> str:
    """Map Safety Strictness slider 1-3 to internal scan profiles."""
    return {1: "low", 2: "medium", 3: "high"}.get(int(level), "medium")


def _strictness_to_int(strictness: str) -> int:
    """Map profile label to is_drug_related strictness level (1-3)."""
    return {"low": 1, "medium": 2, "high": 3}.get(strictness, 2)


def _log_scanner_safe() -> None:
    try:
        print("âœ… MSN SCANNER: Image is SAFE")
    except UnicodeEncodeError:
        print("MSN SCANNER: Image is SAFE")


def _term_matches_text(term: str, text: str) -> bool:
    """
    Match whole words/phrases only â€” avoids false positives like 'pill' inside 'pillows'
    or 'pipe' inside 'painting'.
    """
    term = term.strip().lower()
    if not term:
        return False
    if " " in term:
        return term in text
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def _scanner_bypass_enabled() -> bool:
    """Set MSN_SCANNER_BYPASS=1 in the environment to force-approve all captions."""
    return os.environ.get("MSN_SCANNER_BYPASS", "").strip().lower() in ("1", "true", "yes", "on")


def _drug_terms_for_strictness(level: int) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Return (single_terms, phrases, visual_hints) active for strictness 1-3."""
    level = max(1, min(3, int(level)))
    singles: set[str] = set(DRUG_TERMS_STRICT)
    phrases: set[str] = set()
    hints: set[str] = set()

    if level >= 2:
        singles.update(DRUG_TERMS_STANDARD)
        phrases.update(BANNED_DRUG_PHRASES)
        phrases.update(CANNABIS_PHRASES)
        hints.update(DRUG_VISUAL_HINTS_LEVEL2)

    if level >= 3:
        singles.update(DRUG_TERMS_LEVEL3_ONLY)
        hints.update(DRUG_VISUAL_HINTS)

    return frozenset(singles), frozenset(phrases), frozenset(hints)


# Skip cannabis heuristics when caption clearly describes real meal/kitchen context.
_CANNABIS_REAL_FOOD_EXEMPT = (
    "kitchen", "restaurant", "dining", "chef", "cooking", "oven", "stove",
    "plate of food", "bowl of soup", "pizza", "burger", "sandwich", "pasta",
    "salad bowl", "breakfast", "dinner table", "fast food",
)


def _is_likely_real_food_context(text: str) -> bool:
    return any(marker in text for marker in _CANNABIS_REAL_FOOD_EXEMPT)


def _cannabis_heuristic_match(text: str) -> bool:
    """
    Level-2 fallback when Florence describes cannabis visually without saying 'cannabis'.
    Includes food-mislabel recovery (e.g. 'pile of food' + gloves/hands).
    """
    if "marijuana" in text or "cannabis" in text or " weed" in text or text.startswith("weed"):
        return True

    has_dried = "dried" in text or "drying" in text
    has_green = "green" in text
    plant_words = (
        "plant material", "plant in hand", "herb", " herbs", " buds", "bud ",
        "leafy", "serrated", "substance", "material", "nug", "nugs",
    )
    handling_words = ("hands", "hand", "gloves", "glove", "holding", "pile")

    # Florence often mislabels trimmed cannabis as 'pile of food' near gloved hands.
    if not _is_likely_real_food_context(text):
        if any(p in text for p in ("pile of food", "pile of herbs", "pile of material", "pile of green")):
            if any(w in text for w in handling_words):
                return True

    # green + buds/dried/pile/hands/gloves (explicit user fallback)
    if has_green and any(w in text for w in ("buds", "bud", "nug", "nugs", "dried", "pile", "hands", "gloves", "hand", "glove")):
        if not _is_likely_real_food_context(text):
            return True

    if has_dried and any(w in text for w in plant_words):
        return True
    if has_green and ("bud" in text or "buds" in text or "nug" in text or "nugs" in text):
        return True
    if has_green and any(w in text for w in ("leafy", "serrated", "herb", "substance", "plant")):
        return True
    if "green leaves" in text and any(w in text for w in ("pile", "hand", "table", "spread", "bunch", "plastic")):
        return True
    if "herbs" in text and any(w in text for w in ("bag", "plastic", "jar", "pile", "dried", "hand", "gloves", "glove")):
        if not _is_likely_real_food_context(text):
            return True
    if "leafy green" in text or "green leafy" in text:
        return True
    return False


# Skip pill heuristics when caption clearly describes real confectionery context.
_PILL_REAL_CANDY_EXEMPT = (
    "chocolate", "lollipop", "lollipops", "halloween", "birthday cake",
    "dessert", "bakery", "cupcake", "ice cream", "cotton candy",
    "candy store", "candy shop", "trick or treat", "wrapped candy",
    "candy bar", "candy cane", "gingerbread", "frosting", "sprinkles on cake",
)


def _is_likely_real_candy_context(text: str) -> bool:
    return any(marker in text for marker in _PILL_REAL_CANDY_EXEMPT)


def _pill_heuristic_match(text: str) -> bool:
    """
    Level-2 fallback when Florence mislabels pills/capsules as candy/sweets
    or describes them as colorful round objects without drug vocabulary.
    """
    if _is_likely_real_candy_context(text):
        return False

    candy_words = ("candy", "candies", "sweet", "sweets", "gummy", "gummies")
    pill_visual = (
        "colorful", "multicolored", "multi-colored", "bright colored", "assorted",
        "round", "small round", "scattered", "pile", "capsule-shaped", "pill-shaped",
        "oblong", "two-tone", "two tone", "gel cap", "gelcap",
        "in palm", "in hand", "on table", "plastic bottle", "pill bottle",
        "blister pack", "prescription bottle",
    )

    if any(c in text for c in candy_words):
        if any(v in text for v in pill_visual):
            return True
        if any(p in text for p in ("scattered", "pile", "assorted", "multicolored", "colorful", "round")):
            return True

    has_color = any(c in text for c in ("colorful", "multicolored", "bright colored", "assorted"))
    has_round = any(
        r in text
        for r in ("round", "small round", "round objects", "round pieces", "round items")
    )
    has_spread = any(p in text for p in ("scattered", "pile", "spread", "arranged", "assorted"))
    if has_color and has_round and has_spread:
        if not any(
            e in text
            for e in ("coin", "bead", "button", "balloon", "marble", "toy", "lego", "cereal", "fruit")
        ):
            return True

    return False


def _collect_drug_violations(text: str, strictness: int) -> list[str]:
    violations: list[str] = []
    level = max(1, min(3, int(strictness)))
    singles, phrases, hints = _drug_terms_for_strictness(strictness)

    for term in singles:
        if not _term_matches_text(term, text):
            continue
        if term in _DRUG_CANNABIS:
            violations.append(f"drug:cannabis:{term}")
        else:
            violations.append(f"drug:{term}")

    for phrase in phrases:
        if phrase not in text:
            continue
        if phrase in CANNABIS_PHRASES:
            violations.append(f"drug:cannabis:phrase:{phrase}")
        elif any(m in phrase for m in _CANNABIS_PHRASE_MARKERS):
            violations.append(f"drug:cannabis:phrase:{phrase}")
        else:
            violations.append(f"drug:phrase:{phrase}")

    for hint in hints:
        if hint not in text:
            continue
        if any(m in hint for m in _CANNABIS_PHRASE_MARKERS):
            violations.append(f"drug:cannabis:visual:{hint}")
        else:
            violations.append(f"drug:visual:{hint}")

    if level >= 2 and _cannabis_heuristic_match(text):
        if not any(str(v).startswith("drug:cannabis:") for v in violations):
            violations.append("drug:cannabis:heuristic:dried_plant_or_buds")

    if level >= 2 and _pill_heuristic_match(text):
        if not any(
            str(v).startswith("drug:")
            and (
                "pill" in str(v)
                or "tablet" in str(v)
                or "capsule" in str(v)
                or "candy" in str(v)
                or "candies" in str(v)
                or "sweet" in str(v)
                or "pharmaceutical" in str(v)
                or "medication" in str(v)
            )
            for v in violations
        ):
            violations.append("drug:pill:heuristic:candy_or_round_objects")

    return violations


def _weapon_terms_for_strictness(level: int) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Return (single_terms, phrases, visual_hints) for weapons at strictness 1-3."""
    level = max(1, min(3, int(level)))
    singles: set[str] = set()
    phrases: set[str] = set()
    hints: set[str] = set()

    if level >= 2:
        singles.update(WEAPON_TERMS_LEVEL2)
        phrases.update(WEAPON_PHRASES)

    if level >= 3:
        singles.update(WEAPON_TERMS_LEVEL3_ONLY)
        hints.update(WEAPON_VISUAL_HINTS)

    return frozenset(singles), frozenset(phrases), frozenset(hints)


def _weapon_heuristic_match(text: str) -> bool:
    """
    Level-2 fallback when Florence describes a firearm without a clean keyword hit.
    Conservative: explicit gun phrases OR hold/point/aim + firearm noun.
    """
    if any(p in text for p in (
        "holding gun", "pointing gun", "with a gun", "with gun",
        "firearm in hand", "gun in hand",
    )):
        return True
    if any(v in text for v in ("holding", "pointing", "aiming", "brandishing")):
        if any(w in text for w in (
            "pistol", "rifle", "shotgun", "revolver", "handgun", "firearm",
            " gun", " guns",
        )):
            return True
    return False


def _collect_weapon_violations(text: str, strictness: int) -> list[str]:
    violations: list[str] = []
    level = max(1, min(3, int(strictness)))
    singles, phrases, hints = _weapon_terms_for_strictness(strictness)

    for term in singles:
        if _term_matches_text(term, text):
            violations.append(f"weapon:{term}")

    for phrase in phrases:
        if phrase in text:
            violations.append(f"weapon:phrase:{phrase}")

    for hint in hints:
        if hint in text:
            violations.append(f"weapon:visual:{hint}")

    if level >= 2 and _weapon_heuristic_match(text):
        if not any(str(v).startswith("weapon:") for v in violations):
            violations.append("weapon:heuristic:holding_or_pointing_firearm")

    return violations


def _minor_family_mislabel_heuristic(text: str) -> bool:
    """
    Florence often describes two minors as 'mother and daughter' (or similar).
    Catch parent + child labels in portrait-style captions when no adult-age markers.
    """
    if any(m in text for m in _MINOR_FAMILY_ADULT_MARKERS):
        return False
    has_parent = any(p in text for p in ("mother", "mom", "father", "dad", "parent"))
    has_child_role = any(
        c in text for c in (
            "daughter", " son", "son ", "child", "children", "kids", "kid ",
        )
    )
    if not (has_parent and has_child_role):
        return False
    portrait_cues = (
        "portrait", "sitting next", "smiling at the camera", "looking at the camera",
        "on the left side", "on the right side", "next to her", "next to him",
    )
    return any(cue in text for cue in portrait_cues)


def _minor_context_boosted_match(text: str) -> str | None:
    """
    Context-weighted minor detection for borderline adult phrasing.
    Soft signals (e.g. 'young woman') only flag when pose/clothing boosters co-occur.
    Skipped when caption clearly describes mature adults.
    """
    if any(m in text for m in _MINOR_FAMILY_ADULT_MARKERS):
        return None

    matched_soft = [signal for signal in MINOR_SOFT_SIGNALS if signal in text]
    if not matched_soft:
        return None

    if not any(boost in text for boost in MINOR_CONTEXT_BOOSTERS):
        return None

    return f"minor:context:{matched_soft[0]}+pose/clothing"


def _minor_interleaved_young_match(text: str) -> str | None:
    """Match 'young [optional adjective] girl/boy' when phrase lists miss (e.g. young asian girl)."""
    if any(m in text for m in _MINOR_FAMILY_ADULT_MARKERS):
        return None
    match = _MINOR_YOUNG_SUBJECT_RE.search(text)
    if match:
        return f"minor:heuristic:{match.group(0).strip().lower()}"
    return None


def _minor_girl_boy_youth_match(text: str) -> str | None:
    """
    Flag girl/boy captions with youth pose/appearance cues.
    Catches posing children when Florence omits 'young' adjacent to 'girl'.
    """
    if any(m in text for m in _MINOR_FAMILY_ADULT_MARKERS):
        return None

    has_girl = _term_matches_text("girl", text)
    has_boy = _term_matches_text("boy", text)
    if not (has_girl or has_boy):
        return None

    youth_cues = (
        "posing", "school uniform", "schoolgirl", "schoolboy",
        "youthful", "childlike", "cute", "kid", "child",
    )
    if not any(cue in text for cue in youth_cues):
        return None

    # Require explicit youth language OR school/childlike context (not bare 'woman').
    if "young" in text or "child" in text or "kid" in text or "school" in text:
        subject = "girl" if has_girl else "boy"
        return f"minor:heuristic:{subject}_youth_context"

    if "posing" in text and (has_girl or has_boy) and "woman" not in text and "man" not in text:
        subject = "girl" if has_girl else "boy"
        return f"minor:heuristic:{subject}_posing"

    return None


def _minor_term_matches(term: str, text: str) -> bool:
    """Minor term match — flexible gap for 'young * girl/boy' multi-word terms."""
    term = term.strip().lower()
    if not term:
        return False
    if term == "young girl":
        return _MINOR_FLEXIBLE_YOUNG_GIRL_RE.search(text) is not None
    if term == "young boy":
        return _MINOR_FLEXIBLE_YOUNG_BOY_RE.search(text) is not None
    if " " in term:
        return term in text
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def _log_minor_violations(violations: list[str], text: str) -> None:
    minor_hits = [v for v in violations if str(v).startswith("minor:")]
    if not minor_hits:
        return
    print(
        f"MSN SCANNER MINOR DETECT: {len(minor_hits)} hit(s) — {minor_hits}"
    )
    print(f"MSN SCANNER MINOR CAPTION: {text[:320]}")


def _has_ai_synthetic_cue(text: str) -> bool:
    return any(cue in text for cue in AI_SYNTHETIC_CUES)


def _extract_numeric_ages(text: str) -> list[int]:
    ages: list[int] = []
    for pattern in (_AGE_YEARS_OLD_RE, _AGE_HYPHEN_RE, _AGE_EXPLICIT_RE):
        for match in pattern.finditer(text):
            try:
                age = int(match.group(1))
            except (ValueError, IndexError):
                continue
            if 1 <= age <= 99:
                ages.append(age)
    return ages


def _minor_numeric_age_match(text: str) -> list[str]:
    """
    Flag explicit numeric ages <=17 always; ages 18-20 when AI/synthetic cues present.
    Skips when caption clearly describes mature adults (30s+ markers only).
    """
    if any(
        m in text
        for m in (
            "in her thirties", "in his thirties", "in her forties", "in his forties",
            "middle-aged", "middle aged", "mature woman", "mature man",
        )
    ):
        return []

    hits: list[str] = []
    for age in _extract_numeric_ages(text):
        if age <= 17:
            label = f"minor:age:explicit_{age}"
            if label not in hits:
                hits.append(label)
        elif age <= 20 and _has_ai_synthetic_cue(text):
            label = f"minor:ai:age_{age}_synthetic"
            if label not in hits:
                hits.append(label)
    return hits


def _minor_ai_character_match(text: str) -> list[str]:
    """
    Detect underage-looking AI/anime/CGI characters even when age is not numeric.
  Conservative: requires synthetic cue + youth subject; skips mature-adult captions.
    """
    if any(m in text for m in _MINOR_FAMILY_ADULT_MARKERS):
        return []
    if not _has_ai_synthetic_cue(text):
        return []

    hits: list[str] = []
    youth_subjects = (
        "girl", "boy", "child", "kid", "teen", "teenager", "adolescent",
        "schoolgirl", "schoolboy", "loli", "shota", "young girl", "young boy",
    )
    if any(subj in text for subj in youth_subjects):
        hits.append("minor:ai:synthetic_youth_character")
    elif ("young woman" in text or "young man" in text) and any(
        b in text for b in ("anime", "manga", "cartoon", "cgi", "illustration", "rendered")
    ):
        hits.append("minor:ai:synthetic_young_borderline")

    under_18_cues = (
        "under 18", "under eighteen", "looks under 18", "appears under 18",
        "minor", "childlike", "preteen", "pre-teen",
    )
    if any(cue in text for cue in under_18_cues):
        hits.append("minor:ai:under_18_appearance")

    return hits


def _minor_flexible_young_match(text: str) -> list[str]:
    """First-pass flexible young-subject detection (highest priority)."""
    hits: list[str] = []
    if _MINOR_FLEXIBLE_YOUNG_GIRL_RE.search(text):
        hits.append("minor:flex:young_girl")
    if _MINOR_FLEXIBLE_YOUNG_BOY_RE.search(text):
        hits.append("minor:flex:young_boy")
    match = _MINOR_YOUNG_SUBJECT_RE.search(text)
    if match:
        token = match.group(0).strip().lower()
        label = f"minor:flex:{token}"
        if label not in hits:
            hits.append(label)
    return hits


def _collect_minor_violations(text: str, strictness: int) -> list[str]:
    """Minor safety — active at strictness 1–3 (caption phrases + word-boundary terms)."""
    level = max(1, min(3, int(strictness)))
    if level < 1:
        return []
    violations: list[str] = []

    # Priority 0: numeric age estimates + AI/synthetic youth characters
    violations.extend(_minor_numeric_age_match(text))
    violations.extend(_minor_ai_character_match(text))

    # Priority 1: flexible young * girl/boy (fixes "young asian girl" vs contiguous "young girl")
    violations.extend(_minor_flexible_young_match(text))

    for phrase in BANNED_MINOR_PHRASES:
        if phrase in text:
            label = f"minor:phrase:{phrase}"
            if label not in violations:
                violations.append(label)
    for term in BANNED_MINOR_TERMS:
        if _minor_term_matches(term, text):
            label = f"minor:{term}"
            if label not in violations:
                violations.append(label)
    if not any(str(v).startswith("minor:") for v in violations):
        if _minor_family_mislabel_heuristic(text):
            violations.append("minor:heuristic:family_portrait_mislabel")
    if not any(str(v).startswith("minor:") for v in violations):
        context_hit = _minor_context_boosted_match(text)
        if context_hit:
            violations.append(context_hit)
    if not any(str(v).startswith("minor:") for v in violations):
        interleaved = _minor_interleaved_young_match(text)
        if interleaved:
            violations.append(interleaved)
    if not any(str(v).startswith("minor:") for v in violations):
        youth_subject = _minor_girl_boy_youth_match(text)
        if youth_subject:
            violations.append(youth_subject)

    _log_minor_violations(violations, text)
    return violations


def _collect_policy_violations(text: str, strictness: int) -> list[str]:
    """Drug, cannabis, weapon, and minor caption violations for the given strictness level."""
    violations: list[str] = []
    violations.extend(_collect_drug_violations(text, strictness))
    violations.extend(_collect_weapon_violations(text, strictness))
    violations.extend(_collect_minor_violations(text, strictness))
    return violations


MODERATION_MODEL = "comfyui:msn_safety_gate"

def _is_ai_minor_hit(hit: str) -> bool:
    if hit.startswith("minor:ai:"):
        return True
    body = hit.lower()
    return any(marker in body for marker in (
        "anime", "manga", "cgi", "cartoon", "ai generated", "illustration",
        "synthetic", "3d render", "digital illustration", "loli", "shota",
    ))


def extract_underage_verdict(violations: list | None) -> dict[str, Any]:
    """Derive is_underage, underage_ai, and underage_reason from scan violations."""
    violations = violations or []
    minor_hits = [str(v) for v in violations if str(v).startswith("minor:")]
    is_underage = len(minor_hits) > 0
    underage_ai = any(_is_ai_minor_hit(h) for h in minor_hits)

    if not is_underage:
        return {
            "is_underage": False,
            "underage_ai": False,
            "underage_reason": "",
        }

    reason = _underage_reason_from_violations(minor_hits, underage_ai)
    return {
        "is_underage": True,
        "underage_ai": underage_ai,
        "underage_reason": reason,
    }


def _underage_reason_from_violations(minor_hits: list[str], underage_ai: bool) -> str:
    if not minor_hits:
        return ""
    labels: list[str] = []
    for hit in minor_hits[:5]:
        if hit.startswith("minor:age:explicit_"):
            age = hit.replace("minor:age:explicit_", "")
            labels.append(f"apparent age {age}")
        elif hit.startswith("minor:ai:age_"):
            labels.append(hit.replace("minor:ai:", "").replace("_", " "))
        elif hit.startswith("minor:ai:"):
            labels.append(hit.replace("minor:ai:", "").replace("_", " "))
        elif hit.startswith("minor:phrase:"):
            labels.append(hit.replace("minor:phrase:", ""))
        elif hit.startswith("minor:flex:"):
            labels.append(hit.replace("minor:flex:", ""))
        else:
            labels.append(hit.replace("minor:", ""))
    prefix = "Underage AI character" if underage_ai else "Minor safety"
    detail = ", ".join(dict.fromkeys(labels))
    return f"{prefix}: {detail}"


def attach_underage_fields(
    moderation_output: dict[str, Any],
    violations: list | None = None,
) -> dict[str, Any]:
    verdict = extract_underage_verdict(violations)
    out = dict(moderation_output)
    out["is_underage"] = bool(verdict["is_underage"])
    out["underage_ai"] = bool(verdict["underage_ai"])
    out["underage_reason"] = str(verdict.get("underage_reason") or "")
    if verdict["is_underage"]:
        out["flagged"] = True
        out["approved"] = False
        out["moderation_status"] = "flagged"
        if verdict["underage_reason"] and verdict["underage_reason"] not in str(out.get("reason") or ""):
            base = str(out.get("reason") or "").strip()
            out["reason"] = (
                f"{verdict['underage_reason']}; {base}" if base and base != "No policy violations detected" else verdict["underage_reason"]
            )
    return out


def build_moderation_output(
    blocked: bool,
    violations: list | None = None,
    metadata_flag: str = "",
) -> dict[str, Any]:
    """Strict 5-field contract for Next.js outputs[\"99\"][\"moderation_output\"]."""
    _ = metadata_flag
    violations = violations or []

    if blocked:
        return attach_underage_fields(
            normalize_moderation_contract(
                {
                    "approved": False,
                    "flagged": True,
                    "moderation_status": "flagged",
                    "reason": _violations_to_reason(violations),
                }
            ),
            violations,
        )

    return attach_underage_fields(
        normalize_moderation_contract(
            {
                "approved": True,
                "flagged": False,
                "moderation_status": "approved",
                "reason": "No policy violations detected",
            }
        ),
        violations,
    )


def normalize_moderation_contract(
    payload: dict[str, Any] | None = None,
    *,
    is_safe: bool | None = None,
    reason: str = "",
) -> dict[str, Any]:
    """Coerce any payload to the exact Next.js moderation_output shape."""
    payload = payload or {}
    approved = bool(is_safe) if is_safe is not None else bool(payload.get("approved", True))
    flagged = not approved
    status = "approved" if approved else "flagged"
    reason_text = (reason or str(payload.get("reason") or "")).strip()
    if not reason_text:
        reason_text = (
            "No policy violations detected"
            if approved
            else "Blocked by MSN safety scan"
        )
    return {
        "approved": approved,
        "flagged": flagged,
        "moderation_status": status,
        "reason": reason_text,
        "model": MODERATION_MODEL,
    }


def scan_caption(caption: str, strictness: int = 2, extra_keywords: str = "") -> tuple[bool, list]:
    if _scanner_bypass_enabled():
        _log_scanner_safe()
        print("MSN SCANNER: Global bypass active (MSN_SCANNER_BYPASS=1)")
        return False, []

    if not caption:
        _log_scanner_safe()
        return False, []

    level = max(1, min(3, int(strictness)))
    text = _normalize_caption(caption)
    violations: list[str] = []

    violations.extend(_collect_policy_violations(text, level))

    if extra_keywords:
        for term in [w.strip().lower() for w in extra_keywords.split(",") if w.strip()]:
            if _term_matches_text(term, text):
                violations.append(f"extra:{term}")

    if violations:
        policy_hits = [
            v for v in violations if str(v).startswith(("drug:", "weapon:", "minor:"))
        ]
        print(
            f"!!! MSN SCANNER [strictness={level}]: "
            f"{len(policy_hits)} policy hit(s), {len(violations)} total — {violations} !!!"
        )
        return True, violations

    _log_scanner_safe()
    return False, []


def _violation_category(label: str) -> str:
    """Map a matched term/phrase to an admin-friendly category."""
    if label.startswith("weapon:"):
        return "Weapons / firearms"

    if label.startswith("drug:cannabis:"):
        return "Cannabis"

    if label.startswith("drug:visual:"):
        body = label[12:]
        if any(m in body for m in _CANNABIS_PHRASE_MARKERS):
            return "Cannabis"
        if any(m in body for m in _PILL_PHRASE_MARKERS):
            return "Pills / tablets"
        if any(m in body for m in _INJECTION_PHRASE_MARKERS):
            return "Injection / paraphernalia"
        return "Visual drug scene"

    if label.startswith("drug:phrase:"):
        body = label[12:]
        if any(m in body for m in _CANNABIS_PHRASE_MARKERS):
            return "Cannabis"
        if any(m in body for m in _PILL_PHRASE_MARKERS):
            return "Pills / tablets"
        if any(m in body for m in _INJECTION_PHRASE_MARKERS):
            return "Injection / paraphernalia"
        return "Drug paraphernalia"

    if label.startswith("drug:"):
        term = label[5:]
        if term.startswith("pill:heuristic:"):
            return "Pills / tablets"
        if term in _DRUG_HARD:
            return "Hard drugs"
        if term in _DRUG_CANNABIS:
            return "Cannabis"
        if term in _DRUG_PILLS:
            return "Pills / tablets"
        if term in _DRUG_INJECTION:
            return "Injection / paraphernalia"
        return "Substances / slang"

    if label.startswith("minor:ai:") or (
        label.startswith("minor:age:") and "synthetic" in label
    ):
        return "Underage AI character"
    if label.startswith("minor:phrase:") or label.startswith("minor:"):
        return "Minor safety"
    if label.startswith("extra:"):
        return "Custom keyword"
    return "Other"


def _violations_to_reason(violations: list) -> str:
    if not violations:
        return "No policy violations detected"

    groups: dict[str, list[str]] = {
        "Hard drugs": [],
        "Weapons / firearms": [],
        "Cannabis": [],
        "Pills / tablets": [],
        "Injection / paraphernalia": [],
        "Drug paraphernalia": [],
        "Visual drug scene": [],
        "Substances / slang": [],
        "Minor safety": [],
        "Underage AI character": [],
        "Custom keyword": [],
        "Other": [],
    }

    for v in violations:
        s = str(v)
        cat = _violation_category(s)
        if s.startswith("weapon:phrase:"):
            display = s[14:]
        elif s.startswith("weapon:visual:"):
            display = s[14:]
        elif s.startswith("weapon:"):
            display = s[7:]
        elif s.startswith("drug:cannabis:phrase:"):
            display = s[21:]
        elif s.startswith("drug:cannabis:visual:"):
            display = s[21:]
        elif s.startswith("drug:cannabis:"):
            display = s[14:]
        elif s.startswith("drug:phrase:"):
            display = s[12:]
        elif s.startswith("drug:visual:"):
            display = s[12:]
        elif s.startswith("drug:"):
            display = s[5:]
        elif s.startswith("minor:phrase:"):
            display = s[13:]
        elif s.startswith("minor:"):
            display = s[6:]
        elif s.startswith("extra:"):
            display = s[6:]
        else:
            display = s
        groups.setdefault(cat, []).append(display)

    parts: list[str] = []
    for label in (
        "Hard drugs",
        "Weapons / firearms",
        "Cannabis",
        "Pills / tablets",
        "Injection / paraphernalia",
        "Drug paraphernalia",
        "Visual drug scene",
        "Substances / slang",
        "Minor safety",
        "Underage AI character",
        "Custom keyword",
        "Other",
    ):
        items = groups.get(label) or []
        if not items:
            continue
        shown = ", ".join(items[:3])
        if len(items) > 3:
            shown += f" (+{len(items) - 3} more)"
        parts.append(f"{label}: {shown}")

    summary = parts[0] if len(parts) == 1 else f"{len(parts)} categories"
    return f"Blocked ({summary}) — " + "; ".join(parts[:4])


def is_drug_related(caption: str, strictness: int = 2, extra_keywords: str = "") -> tuple[bool, str]:
    blocked, violations = scan_caption(caption, strictness, extra_keywords)
    if blocked:
        return True, _violations_to_reason(violations)
    if not caption:
        return False, "Empty caption — no scan performed"
    return False, "No policy violations detected"


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    frame = image[0] if image.ndim == 4 else image
    arr = np.clip(255.0 * frame.detach().cpu().numpy(), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def apply_red_overlay(
    image: torch.Tensor,
    label: str = "BLOCKED â€” POLICY VIOLATION",
    overlay_alpha: float = 0.45,
) -> torch.Tensor:
    """Tint image red and stamp a visible blocked label for audit saves."""
    base = _tensor_to_pil(image).convert("RGBA")
    red_layer = Image.new("RGBA", base.size, (220, 20, 20, int(255 * overlay_alpha)))
    composed = Image.alpha_composite(base, red_layer).convert("RGB")

    draw = ImageDraw.Draw(composed)
    font = ImageFont.load_default()
    margin = 12
    text = label[:80]
    bbox = draw.textbbox((0, 0), text, font=font)
    box_w = bbox[2] - bbox[0] + margin * 2
    box_h = bbox[3] - bbox[1] + margin * 2
    draw.rectangle((0, 0, box_w, box_h), fill=(120, 0, 0))
    draw.text((margin, margin), text, fill=(255, 255, 255), font=font)

    return _pil_to_tensor(composed)


def _save_image_batch(
    images: torch.Tensor,
    filename_prefix: str,
    extra_metadata: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix,
        output_dir,
        images[0].shape[1],
        images[0].shape[0],
    )
    results: list[dict[str, str]] = []
    for batch_number, image in enumerate(images):
        arr = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        file_name = f"{filename.replace('%batch_num%', str(batch_number))}_{counter:05}_.png"
        file_path = os.path.join(full_output_folder, file_name)
        img.save(file_path)
        results.append({"filename": file_name, "subfolder": subfolder, "type": "output"})
        counter += 1

    if extra_metadata:
        meta_path = os.path.join(full_output_folder, f"{filename_prefix}_block_log.json")
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(extra_metadata, handle, indent=2)

    return results


class MSNContentFilter:
    """
    Caption-only filter (Florence-2 / WD14 text).

    Outputs: blocked (BOOL), reason, image (pass-through / red overlay), message
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "Florence2Run caption (index 2)."},
                ),
                "strictness": (
                    ["low", "medium", "high"],
                    {"default": "medium"},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "block_visual": (
                    ["pass_through", "red_overlay", "black"],
                    {"default": "red_overlay"},
                ),
                "blocked_message": (
                    "STRING",
                    {"default": "Content blocked: drug or policy-restricted material detected."},
                ),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("blocked", "reason", "image", "message")
    FUNCTION = "filter_content"
    OUTPUT_NODE = True
    CATEGORY = "My Secret Needs/Scanner"

    def filter_content(
        self,
        caption: str,
        strictness: str = "medium",
        image: torch.Tensor | None = None,
        block_visual: str = "red_overlay",
        blocked_message: str = "Content blocked: drug or policy-restricted material detected.",
    ):
        caption_norm = _normalize_caption(caption)
        blocked, violations = scan_caption(caption_norm, _strictness_to_int(strictness))
        reason = _violations_to_reason(violations)
        message = blocked_message.strip() if blocked else ""
        out_image = _resolve_visual(image, blocked, block_visual, reason)

        return {
            "ui": {"text": [message or "MSN Content Filter â€” Clear Pass"]},
            "result": (blocked, reason, out_image, message),
        }


class MSNSafetyGate:
    """
    Full Florence-2 safety gate: scan caption, route visuals, save approved/blocked.

    Wire:
      image  <- LoadImage / Flux output
      caption <- Florence2Run.caption (output index 2)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "caption": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "Florence2Run caption output."},
                ),
                "strictness": (
                    ["low", "medium", "high"],
                    {"default": "medium"},
                ),
            },
            "optional": {
                "save_outputs": ("BOOLEAN", {"default": True}),
                "approved_prefix": ("STRING", {"default": "MSN_Approved"}),
                "blocked_prefix": ("STRING", {"default": "MSN_Blocked"}),
                "blocked_message": (
                    "STRING",
                    {"default": "Content blocked: drug or policy-restricted material detected."},
                ),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("blocked", "reason", "approved_image", "blocked_image", "message")
    FUNCTION = "run_gate"
    OUTPUT_NODE = True
    CATEGORY = "My Secret Needs/Scanner"

    def run_gate(
        self,
        image: torch.Tensor,
        caption: str,
        strictness: str = "medium",
        save_outputs: bool = True,
        approved_prefix: str = "MSN_Approved",
        blocked_prefix: str = "MSN_Blocked",
        blocked_message: str = "Content blocked: drug or policy-restricted material detected.",
    ):
        caption_norm = _normalize_caption(caption)
        blocked, matched = scan_caption(caption_norm, _strictness_to_int(strictness))
        reason = _violations_to_reason(matched)
        message = blocked_message.strip() if blocked else ""

        approved_image = image
        blocked_image = apply_red_overlay(image, label=message or "BLOCKED")

        ui_images: list[dict[str, str]] = []
        if save_outputs:
            if blocked:
                log_meta = {
                    "blocked": True,
                    "reason": reason,
                    "matched": matched,
                    "caption": caption_norm,
                }
                ui_images = _save_image_batch(
                    blocked_image,
                    blocked_prefix,
                    extra_metadata=log_meta,
                )
            else:
                ui_images = _save_image_batch(approved_image, approved_prefix)

        ui_payload: dict[str, Any] = {"text": [message or "MSN Safety Gate â€” Clear Pass"]}
        if ui_images:
            ui_payload["images"] = ui_images

        return {
            "ui": ui_payload,
            "result": (blocked, reason, approved_image, blocked_image, message),
        }


def _resolve_visual(
    image: torch.Tensor | None,
    blocked: bool,
    block_visual: str,
    reason: str,
) -> torch.Tensor:
    if image is None:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    if not blocked:
        return image

    mode = (block_visual or "red_overlay").strip().lower()
    if mode == "black":
        return torch.zeros_like(image)
    if mode == "red_overlay":
        return apply_red_overlay(image, label=reason[:72])
    return image


class DrugSafetyFilter:
    """
    Drug-focused Florence-2 safety filter for the MSN Safety Gate workflow.

    Outputs:
      is_safe (BOOL)       - True when caption passes all checks
      reason (STRING)      - human-readable pass/block explanation
      safe_image (IMAGE)   - original image when safe; black frame when blocked
      blocked_image (IMAGE)- red overlay + reason text when blocked; black when safe
    """

    DEFAULT_EXTRA_KEYWORDS = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "caption": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Connect Florence2Run -> caption (output index 2).",
                    },
                ),
                "strictness": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 3,
                        "step": 1,
                        "tooltip": "1=hard drugs only | 2=+cannabis/pills/weapons | 3=+visual hints, pipe/pill",
                    },
                ),
            },
            "optional": {
                "extra_banned_keywords": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Comma-separated extra drug terms (always scanned when non-empty).",
                    },
                ),
                "bypass_scan": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When True, skip drug scan and always approve.",
                    },
                ),
                "metadata_flag": ("STRING", {"default": "", "multiline": False}),
                "is_underage": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "From UnderageFilterNode/AgeCheckerNode. True forces block.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("is_safe", "reason", "safe_image", "blocked_image", "moderation_output")
    FUNCTION = "filter_drug_content"
    OUTPUT_NODE = False
    CATEGORY = "My Secret Needs/Scanner"

    def filter_drug_content(
        self,
        image: torch.Tensor,
        caption: str,
        strictness: int = 2,
        extra_banned_keywords: str = "",
        bypass_scan: bool = False,
        metadata_flag: str = "",
        is_underage: bool = False,
    ):
        captions = _florence_captions(caption)
        frame_count = max(int(image.shape[0]), len(captions))
        if len(captions) < frame_count:
            captions.extend([""] * (frame_count - len(captions)))
        elif len(captions) > frame_count:
            captions = captions[:frame_count]

        level = max(1, min(3, int(strictness)))

        if bypass_scan or _scanner_bypass_enabled():
            blocked, matched = False, []
            first_blocked_index = None
            print(f"MSN SCANNER: bypass_scan enabled - Image is SAFE (strictness={level})")
        else:
            blocked, matched, first_blocked_index = scan_captions_per_frame(
                captions,
                level,
                extra_banned_keywords or "",
            )

        # OR logic: block if underage OR drug terms
        if is_underage:
            blocked = True
            if "underage_detected" not in matched:
                matched = ["underage_detected", *matched]

        is_safe = not blocked
        reason = _violations_to_reason(matched)
        moderation_output = build_moderation_output(blocked, matched, metadata_flag)
        moderation_output_json = json.dumps(moderation_output, ensure_ascii=False)

        display_image = image[0:1]
        if not is_safe and first_blocked_index is not None and first_blocked_index < image.shape[0]:
            display_image = image[first_blocked_index : first_blocked_index + 1]

        if is_safe:
            safe_image = image
            blocked_image = torch.zeros_like(image)
            ui_text = f"Drug Safety Filter - PASS (strictness {level})"
            print(f"MSN SCANNER: DrugSafetyFilter PASS strictness={level}")
        else:
            safe_image = torch.zeros_like(image)
            blocked_image = apply_red_overlay(display_image, label=reason[:80])
            if image.shape[0] > 1:
                blocked_image = blocked_image.repeat(image.shape[0], 1, 1, 1)
            ui_text = f"BLOCKED (strictness {level}): {reason}"
            print(f"MSN SCANNER: DrugSafetyFilter BLOCKED strictness={level} — {reason}")

        return {
            "ui": {"text": [ui_text]},
            "result": (is_safe, reason, safe_image, blocked_image, moderation_output_json),
        }


class MSNScannerBypass:
    """
    Force-approve path for scanContent.ts.
    Passes the image through unchanged and emits approved moderation_output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                "metadata_flag": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "moderation_output")
    FUNCTION = "bypass"
    OUTPUT_NODE = True
    CATEGORY = "My Secret Needs/Scanner"

    def bypass(self, image: torch.Tensor, metadata_flag: str = ""):
        moderation_output = build_moderation_output(False, [], metadata_flag)
        moderation_output_json = json.dumps(moderation_output, ensure_ascii=False)
        print("MSN SCANNER: Bypass node - Image is SAFE")
        return {
            "ui": {"text": ["MSN Scanner Bypass - approved"]},
            "result": (image, moderation_output_json),
        }


NODE_CLASS_MAPPINGS = {
    "MSNContentFilter": MSNContentFilter,
    "MSNSafetyGate": MSNSafetyGate,
    "DrugSafetyFilter": DrugSafetyFilter,
    "MSNScannerBypass": MSNScannerBypass,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MSNContentFilter": "MSN Drug/NSFW Content Filter",
    "MSNSafetyGate": "MSN Florence-2 Safety Gate",
    "DrugSafetyFilter": "Drug Safety Filter (Florence-2)",
    "MSNScannerBypass": "MSN Scanner Bypass (force approve)",
}


