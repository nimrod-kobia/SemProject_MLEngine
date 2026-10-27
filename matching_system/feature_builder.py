
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
    """Return the Jaccard similarity of two sets."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _agreement(lost: Report, found: Report, field: str) -> tuple[int, int]:
    """Return a pair of mutually exclusive signals for one structured attribute."""
    left = normalize_scalar(getattr(lost, field))
    right = normalize_scalar(getattr(found, field))
    if not left or not right:
        return 0, 0  # Unknown is not a contradiction.
    return int(left == right), int(left != right)


def pair_features(lost: Report, found: Report) -> dict[str, float | int]:
    """Return the complete, ordered feature vector for one candidate pair. """
    if lost.report_type != "lost" or found.report_type != "found":
        raise ValueError("Expected a lost report followed by a found report")
    result: dict[str, float | int] = {
        # Similarity features preserve partial evidence from free-text fields.
        "description_similarity": jaccard(tokenize(lost.description), tokenize(found.description)),
        "distinctive_feature_similarity": jaccard(
            tokenize(" ".join(lost.distinctive_features)),
            tokenize(" ".join(found.distinctive_features)),
        ),
        "location_similarity": jaccard(tokenize(lost.location), tokenize(found.location)),
    }
    # Match/conflict pairs let the classifier learn different weights for
    # positive evidence and for an explicit disagreement.
    for field in ("category", "colour", "brand", "model", "material"):
        match, conflict = _agreement(lost, found, field)
        result[f"{field}_match"] = match
        result[f"{field}_conflict"] = conflict
    # Use one bounded numeric feature so large date gaps cannot dominate the
    # other comparisons. Unknown timestamps receive the same maximum value.
    if lost.reported_at and found.reported_at:
        hours = abs((found.reported_at - lost.reported_at).total_seconds()) / 3600
        result["time_difference_hours"] = min(hours, 48.0)
    else:
        result["time_difference_hours"] = 48.0
    return {name: result[name] for name in FEATURE_NAMES}
