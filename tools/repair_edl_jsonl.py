#!/usr/bin/env python3
"""Validate EDL JSONL sidecar files and optionally remove truncated tail lines."""

import argparse
import csv
import json
from pathlib import Path


JSONL_COLUMNS = (
    "edl_scene_summary_jsonl",
    "edl_prediction_light_jsonl",
    "edl_query_semantic_jsonl",
)


def iter_manifest_jsonl_paths(manifest_path):
    with manifest_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            for key in JSONL_COLUMNS:
                path = Path(row.get(key, ""))
                if path.exists():
                    yield row, key, path


def validate_or_repair(path, repair_tail=False):
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    bad = None
    for lineno, line in enumerate(lines, 1):
        try:
            json.loads(line)
        except Exception as exc:  # noqa: BLE001 - report exact decoder failure.
            bad = (lineno, exc)
            break

    if bad is None:
        return True, None

    lineno, exc = bad
    if repair_tail and lineno == len(lines):
        path.write_text("".join(lines[:lineno - 1]), encoding="utf-8")
        return True, f"repaired tail line {lineno}: {exc!r}"
    return False, f"bad line {lineno}: {exc!r}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="EDL log manifest TSV")
    parser.add_argument(
        "--repair-tail",
        action="store_true",
        help="Drop a malformed final JSONL line. Non-tail corruption still fails.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    failures = []
    repaired = []
    for row, key, path in iter_manifest_jsonl_paths(manifest_path):
        ok, message = validate_or_repair(path, repair_tail=args.repair_tail)
        label = f"{row.get('corruption', 'unknown')}-s{row.get('severity', '?')}:{key}"
        if ok and message:
            repaired.append((label, path, message))
        elif not ok:
            failures.append((label, path, message))

    for label, path, message in repaired:
        print(f"[EDL JSONL] Repaired {label}: {path} ({message})")

    if failures:
        for label, path, message in failures:
            print(f"[EDL JSONL] Invalid {label}: {path} ({message})")
        raise SystemExit(1)

    print("[EDL JSONL] All manifest JSONL files parsed successfully.")


if __name__ == "__main__":
    main()
