#!/usr/bin/env python3
"""Normalize the supplied MIDAS and PETCI JSON files to JSONL.

Run from the repository root: python normalize_datasets.py
Only writes under data/processed; source files are never modified.
"""
from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data"
OUTPUT = SOURCE / "processed"
LANGUAGES = {
    "ZH_Idioms.json": "zh",
    "EN_Idioms.json": "en",
    "KO_Idioms.json": "ko",
    "AR_Idioms.json": "ar",
}


def clean_text(value: Any) -> str:
    """Convert a scalar string to NFC and trim surrounding whitespace."""
    if not isinstance(value, str):
        return ""
    return unicodedata.normalize("NFC", value).strip()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_array(path: Path) -> list[Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, list):
        raise ValueError(f"Expected a top-level JSON array in {path}")
    return value


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(compact_json(record) + "\n")


def rejected_record(dataset: str, source_file: str, index: int,
                    source_id: Any, reason: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "source_file": source_file,
        "source_index": index,
        "source_id": source_id,
        "reason": reason,
    }


def normalize_midas() -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    stats: Counter = Counter()
    seen: dict[tuple[str, str, str], str] = {}

    for filename, language in LANGUAGES.items():
        path = SOURCE / "MIDAS" / filename
        for index, row in enumerate(load_array(path)):
            stats["source_records"] += 1
            source_id = row.get("ID") if isinstance(row, dict) else None
            if not isinstance(row, dict):
                rejected.append(rejected_record("MIDAS", filename, index, source_id, "record is not an object"))
                continue

            raw_idioms = row.get("Idiom")
            variants = raw_idioms if isinstance(raw_idioms, list) else [raw_idioms]
            variants = [clean_text(item) for item in variants if clean_text(item)]
            explanation = clean_text(row.get("Meaning"))
            if not variants:
                rejected.append(rejected_record("MIDAS", filename, index, source_id, "missing or empty Idiom"))
                continue
            if not explanation:
                stats["missing_explanations"] += 1
                rejected.append(rejected_record("MIDAS", filename, index, source_id, "missing or empty Meaning"))
                continue

            identifier = "" if source_id is None else clean_text(str(source_id))
            if not identifier:
                rejected.append(rejected_record("MIDAS", filename, index, source_id, "missing ID"))
                continue

            idiom = variants[0]
            key = (language, idiom.casefold(), explanation.casefold())
            metadata: dict[str, Any] = {
                "source_id": identifier,
                "source_file": filename,
                "reference_explanation_language": language,
            }
            if len(variants) > 1:
                metadata["idiom_variants"] = variants[1:]
            context = row.get("Sentence")
            if isinstance(context, list):
                context = [clean_text(item) for item in context if clean_text(item)]
                if context:
                    metadata["context"] = context
            else:
                context = clean_text(context)
                if context:
                    metadata["context"] = context

            record = {
                "idiom_id": f"MIDAS:{language}:{identifier}",
                "dataset": "MIDAS",
                "language": language,
                "target_language": language,
                "idiom": idiom,
                "reference_explanation": explanation,
                "metadata": metadata,
            }
            if key in seen:
                stats["duplicate_records"] += 1
                record["metadata"]["duplicate_of"] = seen[key]
            else:
                seen[key] = record["idiom_id"]
            records.append(record)
            stats["retained_records"] += 1
    return records, rejected, stats


def normalize_petci() -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    path = SOURCE / "PETCI" / "filtered.json"
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    stats: Counter = Counter()
    seen: dict[tuple[str, str], str] = {}

    for index, row in enumerate(load_array(path)):
        stats["source_records"] += 1
        source_id = row.get("id") if isinstance(row, dict) else None
        if not isinstance(row, dict):
            rejected.append(rejected_record("PETCI", path.name, index, source_id, "record is not an object"))
            continue
        idiom = clean_text(row.get("chinese"))
        explanation = clean_text(row.get("gold"))
        if not idiom:
            rejected.append(rejected_record("PETCI", path.name, index, source_id, "missing or empty chinese idiom"))
            continue
        if not explanation:
            stats["missing_explanations"] += 1
            rejected.append(rejected_record("PETCI", path.name, index, source_id, "missing or empty gold explanation"))
            continue
        if source_id is None or not clean_text(str(source_id)):
            rejected.append(rejected_record("PETCI", path.name, index, source_id, "missing id"))
            continue

        metadata: dict[str, Any] = {
            "source_id": source_id,
            "source_file": path.name,
            "reference_explanation_language": "en",
        }
        alternates: dict[str, list[str]] = {}
        for field in ("human", "machine"):
            values = row.get(field)
            if isinstance(values, list):
                cleaned = list(dict.fromkeys(clean_text(value) for value in values if clean_text(value)))
                if cleaned:
                    alternates[field] = cleaned
            elif clean_text(values):
                alternates[field] = [clean_text(values)]
        if alternates:
            metadata["alternate_references"] = alternates

        identifier = str(source_id)
        record = {
            "idiom_id": f"PETCI:zh:{identifier}",
            "dataset": "PETCI",
            "language": "zh",
            "target_language": "en",
            "idiom": idiom,
            "reference_explanation": explanation,
            "metadata": metadata,
        }
        key = (idiom.casefold(), explanation.casefold())
        if key in seen:
            stats["duplicate_records"] += 1
            record["metadata"]["duplicate_of"] = seen[key]
        else:
            seen[key] = record["idiom_id"]
        records.append(record)
        stats["retained_records"] += 1
    return records, rejected, stats


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"datasets": {}, "total_retained_records": 0}
    all_rejected: list[dict[str, Any]] = []
    language_counts: Counter = Counter()

    for name, normalize in (("MIDAS", normalize_midas), ("PETCI", normalize_petci)):
        records, rejected, stats = normalize()
        write_jsonl(OUTPUT / f"{name.lower()}.jsonl", records)
        all_rejected.extend(rejected)
        for record in records:
            language_counts[f"{record['language']}->{record['target_language']}"] += 1
        summary["datasets"][name] = dict(stats)
        summary["datasets"][name]["excluded_records"] = len(rejected)
        summary["total_retained_records"] += len(records)

    summary["records_by_idiom_to_target_language"] = dict(sorted(language_counts.items()))
    summary["excluded_records"] = len(all_rejected)
    with (OUTPUT / "rejected.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for item in all_rejected:
            stream.write(compact_json(item) + "\n")
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nNormalized files written to: {OUTPUT}")


if __name__ == "__main__":
    main()
