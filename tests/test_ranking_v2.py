import csv
import hashlib
import json
import unittest
from pathlib import Path

from scripts.validate_ranking_dataset import validate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "ranking_v2"
TEST_SOURCE = ROOT / "data" / "fixtures" / "ranking_v2" / "held_out_items.csv"
FROZEN_TEST_SHA256 = "4d7f0b831ae2141316559d113bc41cd473d403158dd8fa87dcf4fe64196e6cb5"


def read(name):
    with (DATA / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class RealisticRankingFixtureTests(unittest.TestCase):
    def test_frozen_test_source_and_group_separation(self):
        # Git checkouts may convert the fixture's LF line endings on Windows.
        frozen_source = TEST_SOURCE.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(frozen_source).hexdigest(), FROZEN_TEST_SHA256)
        provenance = json.loads((DATA / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["new_held_out_test_file_sha256"], FROZEN_TEST_SHA256)
        found = read("found_reports.csv")
        groups = {split: {r["item_group_id"] for r in found if r["split"] == split}
                  for split in ("train", "validation", "test")}
        self.assertEqual([len(groups[s]) for s in groups], [16, 8, 12])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])

    def test_metadata_and_no_match_variation(self):
        found = {r["item_group_id"]: r for r in read("found_reports.csv")}
        lost = read("lost_reports.csv")
        matched = [r for r in lost if r["item_group_id"] in found]
        self.assertTrue(any(r["colour"] != found[r["item_group_id"]]["colour"] for r in matched))
        self.assertTrue(any(r["last_seen_location"] != found[r["item_group_id"]]["location"] for r in matched))
        queries = read("evaluation_queries.csv")
        self.assertEqual(sum(r["split"] == "test" and r["has_match"] == "0" for r in queries), 4)
        self.assertTrue(validate(DATA)["valid"])
        pools = json.loads((DATA / "candidate_pools.json").read_text(encoding="utf-8"))
        self.assertEqual(len(next(iter(pools.values()))), 36)


if __name__ == "__main__":
    unittest.main()
