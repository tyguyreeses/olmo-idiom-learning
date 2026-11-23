#!/usr/bin/env python3
"""Create reproducible, equally sized evaluation samples from normalized data."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

# Configure the number of idioms selected from EACH task/language cohort here.
SAMPLE_SIZE_PER_COHORT = 300
RANDOM_SEED = 479

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
SAMPLE_DIR = PROCESSED_DIR / "samples"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    cohorts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for filename in ("midas.jsonl", "petci.jsonl"):
        for record in read_jsonl(PROCESSED_DIR / filename):
            # Duplicate source pairs remain in normalized data for auditing; do not
            # let exact duplicate pairs count twice in the evaluation sample.
            if record.get("metadata", {}).get("duplicate_of"):
                continue
            if record["dataset"] == "MIDAS":
                cohort = f"midas_{record['language']}"
            else:
                cohort = "petci_zh_to_en"
            cohorts[cohort].append(record)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "sample_size_per_cohort": SAMPLE_SIZE_PER_COHORT,
        "random_seed": RANDOM_SEED,
        "cohorts": {},
    }
    rng = random.Random(RANDOM_SEED)
    for cohort in sorted(cohorts):
        candidates = sorted(cohorts[cohort], key=lambda record: record["idiom_id"])
        if len(candidates) < SAMPLE_SIZE_PER_COHORT:
            raise ValueError(
                f"Cohort {cohort} has {len(candidates)} eligible records; "
                f"cannot sample {SAMPLE_SIZE_PER_COHORT}."
            )
        selected = sorted(rng.sample(candidates, SAMPLE_SIZE_PER_COHORT), key=lambda record: record["idiom_id"])
        write_jsonl(SAMPLE_DIR / f"{cohort}.jsonl", selected)
        summary["cohorts"][cohort] = {
            "eligible_records": len(candidates),
            "sampled_records": len(selected),
            "idiom_ids": [record["idiom_id"] for record in selected],
        }

    expected_cohorts = {"midas_ar", "midas_en", "midas_ko", "midas_zh", "petci_zh_to_en"}
    if set(cohorts) != expected_cohorts:
        raise ValueError(f"Expected cohorts {sorted(expected_cohorts)}, found {sorted(cohorts)}")
    (SAMPLE_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {SAMPLE_SIZE_PER_COHORT} records for each of {len(cohorts)} cohorts to {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
