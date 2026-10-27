"""Deterministic, non-destructive processing of the Delhi Metro CSV."""

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "delhi" / "delhi_lost_and_found.csv"
OUTPUT = ROOT / "data" / "processed" / "cleaned_items.csv"
REPORT = ROOT / "data" / "processed" / "delhi_preprocessing_report.json"
COLUMNS = ["item_name", "description", "item_quantity", "station_name", "receiving_date", "receiving_time"]


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def prepare(source: Path = SOURCE, output: Path = OUTPUT, report_path: Path = REPORT) -> dict:
    if not source.is_file():
        raise FileNotFoundError(
            f"Delhi CSV missing at {source}. Download it from the source linked in "
            "docs/dataset_sources.md and place it at this local, Git-ignored path."
        )
    raw = source.read_bytes()
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"Expected {COLUMNS}; found {reader.fieldnames}")
        rows = list(reader)
    missing = Counter()
    parsing_failures = Counter()
    modified = Counter()
    station_raw_values = set()
    station_clean_values = set()
    station_casefold_values = set()
    station_raw_by_clean = defaultdict(set)
    station_clean_by_casefold = defaultdict(set)
    duplicates = 0
    seen = set()
    processed = []
    for index, row in enumerate(rows, 1):
        if None in row:
            raise ValueError(f"Unexpected extra field on data row {index}")
        # Count exact repeats for the audit, but keep every source row so the
        # cleaned copy can still be traced back to the original CSV.
        key = tuple(row[name] for name in COLUMNS)
        if key in seen:
            duplicates += 1
        seen.add(key)
        clean = {"source_row": index}
        # Empty values stay empty. Filling them with guessed item attributes
        # would turn missing source evidence into invented model inputs.
        for name in COLUMNS:
            value = normalize(row[name] or "")
            if value != (row[name] or ""):
                modified[name] += 1
            if not value:
                missing[name] += 1
            clean[name] = value
        raw_station = row["station_name"] or ""
        clean_station = clean["station_name"]
        if clean_station:
            station_raw_values.add(raw_station)
            station_clean_values.add(clean_station)
            station_casefold_values.add(clean_station.casefold())
            station_raw_by_clean[clean_station].add(raw_station)
            station_clean_by_casefold[clean_station.casefold()].add(clean_station)
        for name in ("item_name", "description", "station_name"):
            clean[name + "_normalized"] = clean[name].casefold()
        for name, fmt in (("receiving_date", "%d/%m/%Y"), ("receiving_time", "%H:%M:%S")):
            value = clean[name]
            try:
                parsed = datetime.strptime(value, fmt) if value else None
            except ValueError:
                parsed = None
                parsing_failures[name] += 1
            clean[name + "_iso"] = parsed.strftime("%Y-%m-%d" if name.endswith("date") else "%H:%M:%S") if parsed else ""
        processed.append(clean)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(processed[0]))
        writer.writeheader()
        writer.writerows(processed)
    report = {
        "source": str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "original_rows": len(rows), "final_rows": len(processed),
        "duplicates_identified": duplicates, "duplicates_removed": 0,
        "missing_values": dict(missing), "parsing_failures": dict(parsing_failures),
        "whitespace_modified_rows_by_field": dict(modified),
        "station_name_audit": {
            "non_empty_rows": sum(1 for row in processed if row["station_name"]),
            "distinct_source_values": len(station_raw_values),
            "distinct_whitespace_normalized_values": len(station_clean_values),
            "distinct_casefolded_values": len(station_casefold_values),
            "rows_changed_by_whitespace": modified.get("station_name", 0),
            "whitespace_collision_groups": sum(
                len(values) > 1 for values in station_raw_by_clean.values()
            ),
            "casefold_collision_groups": sum(
                len(values) > 1 for values in station_clean_by_casefold.values()
            ),
            "semantic_alias_mapping_applied": False,
        },
        "normalization": "Whitespace collapsed; separate casefolded text columns; date DD/MM/YYYY and time HH:MM:SS parsed to ISO columns.",
        "excluded_records": [],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
