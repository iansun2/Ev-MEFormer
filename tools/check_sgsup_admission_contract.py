#!/usr/bin/env python3
"""Fail-closed audit for SGSUP scene-mapper admission telemetry.

This tool verifies feature availability at the supplement decision point.  It
does not impose a gate-rate or performance threshold; those values are only
reported as diagnostics because they are distribution-specific.
"""

import argparse
import gzip
import json
import math
import os
import sys


EXPECTED_FEATURE_COUNT = 18
CONTRACT_FIELDS = {
    "configured": "scene_supp_scene_mapper_configured",
    "expected": "scene_supp_scene_mapper_feature_expected_count",
    "available": "scene_supp_scene_mapper_feature_available_count",
    "finite": "scene_supp_scene_mapper_feature_finite_count",
    "missing": "scene_supp_scene_mapper_feature_missing_count",
    "ready": "scene_supp_scene_mapper_contract_ready",
    "q95_available": "scene_supp_scene_mapper_q95_available",
    "q95": "scene_supp_scene_mapper_q95_runtime_mean",
    "score": "scene_supp_scene_mapper_score_mean",
}


def open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    if path.endswith(".zst"):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError(
                "zstandard is required to audit .zst sidecars") from exc
        raw = open(path, "rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        import io
        return io.TextIOWrapper(reader, encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def finite_float(record, key):
    if key not in record:
        raise ValueError("missing field: {}".format(key))
    value = float(record[key])
    if not math.isfinite(value):
        raise ValueError("non-finite field: {}={}".format(key, record[key]))
    return value


def audit_paths(
        paths, min_records=1, expected_model_label=None,
        expect_supplement_off=False, require_positive_action=False):
    errors = []
    totals = {
        "records": 0,
        "scene_gate_pre_guard_sum": 0.0,
        "scene_gate_post_guard_sum": 0.0,
        "added_ratio_sum": 0.0,
    }
    noadd_reasons = {}
    sample_tokens = set()
    for path in paths:
        if not os.path.isfile(path):
            errors.append("missing sidecar: {}".format(path))
            continue
        try:
            with open_text(path) as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        token = str(record.get("sample_token") or "")
                        if not token:
                            raise ValueError("missing field: sample_token")
                        if token in sample_tokens:
                            raise ValueError(
                                "duplicate sample_token: {}".format(token))
                        if (expected_model_label is not None and
                                record.get("ev_model_label") !=
                                expected_model_label):
                            raise ValueError(
                                "ev_model_label={} (expected {})".format(
                                    record.get("ev_model_label"),
                                    expected_model_label))
                        if expect_supplement_off:
                            unexpected = sorted(
                                key for key in record
                                if key.startswith("scene_supp_"))
                            if unexpected:
                                raise ValueError(
                                    "supplement-off record contains admission/action "
                                    "telemetry: {}".format(unexpected))
                        else:
                            values = {
                                name: finite_float(record, field)
                                for name, field in CONTRACT_FIELDS.items()
                            }
                            expected = {
                                "configured": 1.0,
                                "expected": float(EXPECTED_FEATURE_COUNT),
                                "available": float(EXPECTED_FEATURE_COUNT),
                                "finite": float(EXPECTED_FEATURE_COUNT),
                                "missing": 0.0,
                                "ready": 1.0,
                                "q95_available": 1.0,
                            }
                            mismatches = [
                                "{}={} (expected {})".format(
                                    name, values[name], expected_value)
                                for name, expected_value in expected.items()
                                if values[name] != expected_value
                            ]
                            if mismatches:
                                raise ValueError("; ".join(mismatches))
                        sample_tokens.add(token)
                        totals["records"] += 1
                        if not expect_supplement_off:
                            for source_key, total_key in (
                                    ("scene_supp_scene_gate_pre_guard_ratio",
                                     "scene_gate_pre_guard_sum"),
                                    ("scene_supp_scene_gate_post_guard_ratio",
                                     "scene_gate_post_guard_sum"),
                                    ("scene_supp_added_ratio", "added_ratio_sum")):
                                totals[total_key] += finite_float(
                                    record, source_key)
                            reason = finite_float(
                                record, "scene_supp_noadd_reason_code")
                            noadd_reasons[str(reason)] = (
                                noadd_reasons.get(str(reason), 0) + 1)
                    except Exception as exc:
                        errors.append("{}:{}: {}".format(
                            path, line_number, exc))
        except Exception as exc:
            errors.append("{}: {}".format(path, exc))

    if totals["records"] <= 0:
        errors.append("no valid configured SGSUP scene records were found")
    if totals["records"] < int(min_records):
        errors.append(
            "insufficient records: {} (required at least {})".format(
                totals["records"], int(min_records)))
    if (require_positive_action and
            totals["added_ratio_sum"] <= 0.0):
        errors.append("no positive supplement action was observed")
    count = max(totals["records"], 1)
    summary = {
        "status": "PASS" if not errors else "FAIL",
        "records": totals["records"],
        "expected_features": EXPECTED_FEATURE_COUNT,
        "expected_model_label": expected_model_label,
        "minimum_records": int(min_records),
        "expect_supplement_off": bool(expect_supplement_off),
        "require_positive_action": bool(require_positive_action),
        "mean_scene_gate_pre_guard_ratio": (
            totals["scene_gate_pre_guard_sum"] / count),
        "mean_scene_gate_post_guard_ratio": (
            totals["scene_gate_post_guard_sum"] / count),
        "mean_added_ratio": totals["added_ratio_sum"] / count,
        "noadd_reason_counts": noadd_reasons,
        "errors": errors,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sidecars", nargs="+",
        help="Final guarded candidate scene-summary JSONL files (.jsonl/.gz/.zst).")
    parser.add_argument(
        "--min-records", type=int, default=1,
        help="Minimum unique scene records required across all sidecars.")
    parser.add_argument(
        "--expected-model-label", default=None,
        help="Require every record to carry this exact ev_model_label.")
    parser.add_argument(
        "--expect-supplement-off", action="store_true",
        help="Require matched-base records to contain no scene_supp telemetry.")
    parser.add_argument(
        "--require-positive-action", action="store_true",
        help="Require at least one record with a positive supplement added ratio.")
    args = parser.parse_args()

    if args.min_records <= 0:
        parser.error("--min-records must be positive")
    if args.expect_supplement_off and args.require_positive_action:
        parser.error(
            "--expect-supplement-off and --require-positive-action are mutually exclusive")
    summary = audit_paths(
        args.sidecars, min_records=args.min_records,
        expected_model_label=args.expected_model_label,
        expect_supplement_off=args.expect_supplement_off,
        require_positive_action=args.require_positive_action)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if summary["status"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
