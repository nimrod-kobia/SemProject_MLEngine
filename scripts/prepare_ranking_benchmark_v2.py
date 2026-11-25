"""Build a harder controlled-fiction benchmark without altering version 1."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "fixtures"
OUTPUT = ROOT / "data" / "processed" / "ranking_v2"
LOCATIONS = ("library", "msb", "cafeteria")


def read(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [{key: (value or "").strip() for key, value in row.items()} for row in csv.DictReader(stream)]


def write(name, fields, rows):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prior_case(item):
    """Vary observed details for development cases only; preserve original wording."""
    index = int(item["item_group_id"][1:])
    found_location = item["location"]
    lost_location = LOCATIONS[(LOCATIONS.index(found_location) + 1) % len(LOCATIONS)] if index % 2 else found_location
    found_colour = item["colour"]
    lost_colour = "" if index % 4 == 0 else ("grey" if found_colour == "silver" and index % 3 == 0 else found_colour)
    found_brand = item["brand"]
    lost_brand = "" if index % 3 == 0 else found_brand
    lost_date = date(2026, 10, 1) + timedelta(days=index)
    found_date = lost_date + timedelta(days=1 + index % 3)
    return dict(item_group_id=item["item_group_id"], split="train" if index <= 16 else "validation",
                category=item["category"], found_description=item["found_description"],
                lost_description=item["lost_description"], found_colour=found_colour,
                lost_colour=lost_colour, found_brand=found_brand, lost_brand=lost_brand,
                found_location=found_location, lost_location=lost_location,
                found_date=found_date.isoformat(), lost_date=lost_date.isoformat(),
                scenario="development_metadata_variation", source_type="ai_constructed_fictional")


def main():
    older = read(FIXTURES / "ranking" / "controlled_items.csv")
    new_test_path = FIXTURES / "ranking_v2" / "held_out_items.csv"
    test = read(new_test_path)
    cases = [prior_case(item) for item in older] + [
        {**item, "split": "test", "source_type": "ai_constructed_fictional"} for item in test
    ]
    groups = [item["item_group_id"] for item in cases]
    if len(groups) != len(set(groups)):
        raise ValueError("Controlled identities must be unique")
    for item in cases:
        if not item["category"] or not item["found_description"] or not item["lost_description"]:
            raise ValueError(f"Incomplete case {item['item_group_id']}")
        if date.fromisoformat(item["found_date"]) < date.fromisoformat(item["lost_date"]):
            raise ValueError(f"Found date precedes lost date for {item['item_group_id']}")
    found, lost, labels, queries = [], [], [], []
    for item in cases:
        group, split = item["item_group_id"], item["split"]
        found.append(dict(found_id=f"f-{group}", item_group_id=group, category=item["category"],
                          description=item["found_description"], colour=item["found_colour"],
                          brand=item["found_brand"], distinguishing_attributes="",
                          location=item["found_location"], event_date=item["found_date"],
                          report_status="active", source_type=item["source_type"], split=split))
        lost.append(dict(lost_id=f"l-{group}", item_group_id=group, category=item["category"],
                         description=item["lost_description"], colour=item["lost_colour"],
                         brand=item["lost_brand"], last_seen_location=item["lost_location"],
                         date_lost=item["lost_date"], source_type=item["source_type"], split=split))
        for candidate in cases:
            if candidate["split"] == split:
                labels.append(dict(lost_id=f"l-{group}", found_id=f"f-{candidate['item_group_id']}",
                                   label=int(candidate["item_group_id"] == group),
                                   label_source="controlled_fictional_identity", item_group_id=group,
                                   reviewer="pending_independent_review", split=split))
        queries.append(dict(query_id=f"q-{group}", lost_id=f"l-{group}", relevant_found_ids=f"f-{group}",
                            candidate_pool_id="all_constructed_found_v2", has_match=1, split=split))
    absent = read(FIXTURES / "ranking" / "no_match_queries.csv") + read(FIXTURES / "ranking_v2" / "held_out_no_match.csv")
    for row in absent:
        split = "validation" if row["lost_id"] in {"nm001", "nm002", "nm003"} else "test"
        lost.append(dict(lost_id=row["lost_id"], item_group_id=row["lost_id"], category=row["category"],
                         description=row["description"], colour=row["colour"], brand=row["brand"],
                         last_seen_location=row["location"], date_lost=row.get("date_lost", ""),
                         source_type="ai_constructed_fictional", split=split))
        queries.append(dict(query_id=f"q-{row['lost_id']}", lost_id=row["lost_id"], relevant_found_ids="",
                            candidate_pool_id="all_constructed_found_v2", has_match=0, split=split))
    write("found_reports.csv", list(found[0]), found)
    write("lost_reports.csv", list(lost[0]), lost)
    write("match_labels.csv", list(labels[0]), labels)
    write("evaluation_queries.csv", list(queries[0]), queries)
    (OUTPUT / "candidate_pools.json").write_text(json.dumps({"all_constructed_found_v2": [r["found_id"] for r in found]}, indent=2) + "\n", encoding="utf-8")
    counts = Counter(item["split"] for item in cases)
    audit = {"version": "v2", "source_type": "AI-authored controlled fiction, inventory-informed categories",
             "previously_observed_v1_cases_used_for": "training and validation only",
             "new_held_out_test_file_sha256": hashlib.sha256(new_test_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
             "item_groups_by_split": dict(counts), "found_count": len(found), "lost_count": len(lost),
             "positive_labels": len(cases), "negative_labels": len(labels) - len(cases),
             "no_match_queries": sum(q["has_match"] == 0 for q in queries),
             "candidate_pool_size": len(found), "independent_review": "pending"}
    (OUTPUT / "provenance.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
