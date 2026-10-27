"""Readable numeric features for one labelled lost/found report pair."""

from __future__ import annotations

from .models import Report
from .normalization import normalize_scalar, tokenize


FEATURE_NAMES = (
    "description_similarity",
    "category_match", "category_conflict",
    "colour_match", "colour_conflict",
    "brand_match", "brand_conflict",
    "model_match", "model_conflict",
    "material_match", "material_conflict",
    "distinctive_feature_similarity",
    "location_similarity", "time_difference_hours",
)


def jaccard(left: set[str], right: set[str]) -> float:
    """Share of unique words present in both descriptions."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _agreement(lost: Report, found: Report, field: str) -> tuple[int, int]:
    left = normalize_scalar(getattr(lost, field))
    right = normalize_scalar(getattr(found, field))
    if not left or not right:
        return 0, 0  # Unknown is not a contradiction.
    return int(left == right), int(left != right)


def pair_features(lost: Report, found: Report) -> dict[str, float | int]:
    if lost.report_type != "lost" or found.report_type != "found":
        raise ValueError("Expected a lost report followed by a found report")
    result: dict[str, float | int] = {
        "description_similarity": jaccard(tokenize(lost.description), tokenize(found.description)),
        "distinctive_feature_similarity": jaccard(
            tokenize(" ".join(lost.distinctive_features)),
            tokenize(" ".join(found.distinctive_features)),
        ),
        "location_similarity": jaccard(tokenize(lost.location), tokenize(found.location)),
    }
    for field in ("category", "colour", "brand", "model", "material"):
        match, conflict = _agreement(lost, found, field)
        result[f"{field}_match"] = match
        result[f"{field}_conflict"] = conflict
    # Keep a fixed numeric value for unknown times; the 48-hour cap also stops
    # extreme date gaps from dominating the other comparison features.
    if lost.reported_at and found.reported_at:
        hours = abs((found.reported_at - lost.reported_at).total_seconds()) / 3600
        result["time_difference_hours"] = min(hours, 48.0)
    else:
        result["time_difference_hours"] = 48.0
    return {name: result[name] for name in FEATURE_NAMES}
