"""Build a reproducible synthetic pairing set from real found-item rows.
The source has no true lost claims or confirmed matches, so labels are
constructed assumptions. 
"""

import csv
import json
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from matching_system.feature_builder import FEATURE_NAMES, pair_features
from matching_system.models import Report
from scripts.prepare_delhi_data import OUTPUT as CLEANED, ROOT, prepare


SEED = 42
CLAIMS = ROOT / "data/processed/synthetic_delhi_lost_claims.csv"
PAIRS = ROOT / "data/processed/delhi_labelled_pairs.csv"
FEATURES = ROOT / "data/processed/delhi_modelling_pairs.csv"
SPLIT = ROOT / "data/processed/delhi_split.json"


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build() -> dict:
    audit = prepare()
    with CLEANED.open(encoding="utf-8", newline="") as handle:
        found_rows = list(csv.DictReader(handle))

    # Use item names with enough records for genuinely similar distractors.
    by_name = defaultdict(list)
    for row in found_rows:
        name = row["item_name_normalized"].strip()
        if name and row["receiving_date_iso"] and row["station_name"]:
            by_name[name].append(row)
    eligible = {name: rows for name, rows in by_name.items() if len(rows) >= 30}
    if len(eligible) < 5:
        raise ValueError("At least five frequent item names are needed")

    rng = random.Random(SEED)
    for rows in eligible.values():
        rng.shuffle(rows)
    names = sorted(eligible, key=lambda name: (-len(eligible[name]), name))[:10]
    # Twelve claims per item name: 120 positives, 120 close distractors.
    positives, hard_negatives = [], []
    for name in names:
        positives.extend(eligible[name][:12])
        hard_negatives.extend(eligible[name][12:24])
    used = {int(row["source_row"]) for row in positives + hard_negatives}
    random_pool = [row for row in found_rows if int(row["source_row"]) not in used]
    rng.shuffle(random_pool)
    random_negatives = []
    for positive in positives:
        for index, candidate in enumerate(random_pool):
            if candidate["item_name_normalized"] != positive["item_name_normalized"]:
                random_negatives.append(random_pool.pop(index))
                break
        else:
            raise ValueError("Not enough distinct-category distractors")

    claim_rows, pair_rows, feature_rows = [], [], []
    found_ids = set()
    # Each claim is derived from its positive found row. This defines an
    # exercise label; it does not establish a real passenger's ownership.
    for positive, hard, random_negative in zip(positives, hard_negatives, random_negatives):
        source_id = int(positive["source_row"])
        claim_id = f"synthetic-delhi-{source_id:05d}"
        found_date = datetime.fromisoformat(positive["receiving_date_iso"])
        # Dates and locations on lost claims are invented. They are plausible
        # variants, not recovered historical claims.
        lost_date = found_date - timedelta(days=rng.choice([0, 1, 2]))
        lost_location = positive["station_name"] if rng.random() < 0.8 else ""
        claim = {
            "report_id": claim_id,
            "report_type": "lost",
            "item_name": positive["item_name"].title(),
            "category": positive["item_name"].title(),
            "description": f"I lost a {positive['item_name'].lower()}.",
            "colour": "",  # The source has no reliable colour column.
            "location": lost_location,
            "date": lost_date.date().isoformat(),
            "source_found_row": source_id,
            "data_origin": "synthetic_claim_from_real_found_row",
        }
        claim_rows.append(claim)
        lost = Report(claim_id, "lost", claim["description"], category=claim["category"],
                      location=lost_location, reported_at=lost_date)
        for candidate, label, kind in ((positive, 1, "constructed_positive"),
                                       (hard, 0, "assumed_same_name_negative"),
                                       (random_negative, 0, "assumed_other_name_negative")):
            found_id = f"delhi-{int(candidate['source_row']):05d}"
            if found_id in found_ids:
                raise ValueError(f"Found row reused: {found_id}")
            found_ids.add(found_id)
            found = Report(
                found_id, "found",
                " ".join(part for part in (candidate["item_name"], candidate["description"]) if part),
                category=candidate["item_name"].title(),
                location=candidate["station_name"],
                reported_at=datetime.fromisoformat(candidate["receiving_date_iso"]),
            )
            pair_rows.append({"lost_report_id": claim_id, "found_report_id": found_id,
                              "match": label, "label_basis": kind})
            feature_rows.append({"lost_report_id": claim_id, "found_report_id": found_id,
                                 "original_label": "match" if label else "non_match",
                                 **pair_features(lost, found), "target": label})

    write_csv(CLAIMS, list(claim_rows[0]), claim_rows)
    write_csv(PAIRS, list(pair_rows[0]), pair_rows)
    write_csv(FEATURES, ["lost_report_id", "found_report_id", "original_label", *FEATURE_NAMES, "target"], feature_rows)

    lost_ids = sorted(row["report_id"] for row in claim_rows)
    rng.shuffle(lost_ids)
    # Split claim IDs before evaluation so all three candidates for a claim
    # remain together and its wording cannot leak into the other partition.
    test_ids = set(lost_ids[:round(len(lost_ids) * 0.3)])
    split = {"seed": SEED, "strategy": "Group all candidate pairs by synthetic lost claim; found rows are unique", "splits": {}}
    for name, ids in (("development", set(lost_ids) - test_ids), ("test", test_ids)):
        selected = [row for row in pair_rows if row["lost_report_id"] in ids]
        split["splits"][name] = {"lost_ids": sorted(ids),
                                   "found_ids": sorted(row["found_report_id"] for row in selected),
                                   "pair_count": len(selected)}
    SPLIT.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    return {"real_found_rows": audit["final_rows"], "synthetic_lost_claims": len(claim_rows),
            "constructed_positive_pairs": len(claim_rows), "assumed_negative_pairs": len(pair_rows) - len(claim_rows),
            "development_pairs": split["splits"]["development"]["pair_count"],
            "test_pairs": split["splits"]["test"]["pair_count"]}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
