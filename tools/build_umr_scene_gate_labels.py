#!/usr/bin/env python3
"""Build train-only scene damage applicability labels for UMR Phase 1.

This consumes EV scene sidecar logs collected on train clean / train-O.C only.
The labels are for an offline clean-safe applicability gate, not for final
thesis evaluation metrics. Runtime must use only learned scene features; the
corruption type and severity are used here only as train-time supervision.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict


DEFAULT_SCENE_FEATURES = [
    "scene_fused_weight",
    "scene_selector_entropy",
    "scene_v8_a4_bev_damage_z_mean",
    "scene_v8_a4_route_score_bev_mean",
    "scene_v8_bev_damage_mean",
    "scene_v8_context_bev_damage_mean",
    "scene_v8_high_damage_query_ratio",
    "scene_v8_bev_reliability_mean",
    "scene_v8_reliability_margin_mean",
    "scene_v9_a_bev_mean",
    "scene_v9_a_bev_above_tau_mean",
    "scene_v9_tail_prob_bev_mean",
    "scene_v9_tail_excess_bev_mean",
    "scene_v9_q95_shift_bev_mean",
    "scene_v9_route_score_bev_mean",
    "scene_v9_state_confirmed_bev_mean",
    # Runtime maps this field to v8_output_delta_center before supplement
    # admission, matching the sidecar alias written after inference.
    "scene_v9_center_l1_p95",
    "scene_v9_fused_keep_unit_q95",
]

UNSAFE_SCENE_FEATURES = {
    # These are branch-local z scores but were repeatedly treated as risky when
    # mixed as raw cross-branch comparators. Keep them out of formal defaults.
    "scene_v8_a4_cls_unc_z_bev_mean",
    "scene_v8_a4_cls_unc_z_img_mean",
    "scene_v8_gscrg_raw_unc_fused_top10",
    "scene_v8_gscrg_raw_unc_bev_top10",
    "scene_v8_gscrg_raw_unc_img_top10",
}

FORBIDDEN_INPUT_MARKERS = (
    "nuscenes_infos_val",
    "multicorrupt",
    "multicorrupt_results",
    "targeted_probe",
    "corrupted_val",
    "corrupted-val",
    "validation",
    "val_clean",
)
FORBIDDEN_INPUT_TOKEN_SETS = {
    "val_token": {"val"},
    "multicorrupt_token": {"multicorrupt"},
    "nuscenes_infos_val": {"nuscenes", "infos", "val"},
    "multicorrupt_results": {"multicorrupt", "results"},
    "targeted_probe": {"targeted", "probe"},
    "corrupted_val": {"corrupted", "val"},
    "val_clean": {"val", "clean"},
}
ALLOWED_SPLITS = {"train", "train_clean", "train_oc", "train_online_corrupt"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def open_text(path):
    if path.endswith((".gz", ".gzip")):
        return gzip.open(path, "rt", encoding="utf-8")
    if path.endswith((".zst", ".zstd")):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("Reading zstd logs requires Python package 'zstandard'.") from exc
        raw = open(path, "rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        return _ZstdTextWrapper(raw, reader)
    return open(path, "r", encoding="utf-8")


class _ZstdTextWrapper:
    def __init__(self, raw, reader):
        import io

        self._raw = raw
        self._reader = reader
        self._text = io.TextIOWrapper(reader, encoding="utf-8")

    def __enter__(self):
        return self._text

    def __exit__(self, exc_type, exc, tb):
        self._text.close()
        self._reader.close()
        self._raw.close()
        return False


def find_scene_logs(path):
    if os.path.isfile(path):
        return [path]
    matches = []
    for root, _dirs, files in os.walk(path):
        for name in files:
            if (
                name.startswith("ev_scene_summary")
                or name.startswith("edl_scene_summary")
            ) and (
                name.endswith(".jsonl")
                or name.endswith(".jsonl.gz")
                or name.endswith(".jsonl.zst")
            ):
                matches.append(os.path.join(root, name))
    return sorted(matches)


def path_tokens(path):
    text = str(path).lower().replace("\\", "/")
    raw_parts = []
    for part in text.split("/"):
        raw_parts.extend(re.split(r"[_\-.]+", part))
    return {part for part in raw_parts if part}


def forbidden_path_markers(path):
    text = str(path).lower().replace("\\", "/")
    compact = re.sub(r"[^a-z0-9]+", "_", text)
    tokens = path_tokens(path)
    found = set()
    for marker in FORBIDDEN_INPUT_MARKERS:
        marker_text = marker.lower()
        marker_compact = re.sub(r"[^a-z0-9]+", "_", marker_text)
        if marker_text in text or marker_compact in compact:
            found.add(marker)
    for name, required in FORBIDDEN_INPUT_TOKEN_SETS.items():
        if required.issubset(tokens):
            found.add(name)
    return sorted(found)


def check_formal_inputs(path, allow_debug_val=False):
    forbidden = forbidden_path_markers(path)
    if forbidden and not allow_debug_val:
        raise RuntimeError(
            f"Refusing non-train scene logs for formal calibration: {path}; "
            f"forbidden markers={forbidden}"
        )


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def condition_name(row):
    corruption = str(row.get("corruption_type", "unknown") or "unknown").lower()
    severity = safe_int(row.get("corruption_severity"), 0)
    if corruption == "clean" or severity <= 0:
        return "clean"
    return f"{corruption}_s{severity}"


def label_for(row, args):
    corruption = str(row.get("corruption_type", "unknown") or "unknown").lower()
    severity = safe_int(row.get("corruption_severity"), 0)
    if corruption == "clean" or severity <= 0:
        return 0.0, args.clean_weight, "clean_negative"
    if corruption == "beamsreducing":
        if severity >= 3:
            return 1.0, args.beams_s3_weight, "beams_s3_strong_positive"
        if severity == 2:
            return args.beams_s2_target, args.beams_s2_weight, "beams_s2_weak_positive"
        return 0.0, args.beams_s1_weight, "beams_s1_negative"
    if corruption in {"fog", "motionblur"}:
        return 0.0, args.hard_negative_weight, f"{corruption}_hard_negative"
    if args.other_corruptions == "ignore":
        return None, 0.0, "ignored_other_corruption"
    return 0.0, args.other_negative_weight, f"{corruption}_negative"


def split_is_allowed(row, allow_debug_val=False):
    split = str(row.get("ev_split", "") or "").lower()
    split = re.sub(r"[\s\-]+", "_", split)
    if not split:
        return bool(allow_debug_val)
    if split in ALLOWED_SPLITS:
        return True
    return bool(allow_debug_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--features", default=",".join(DEFAULT_SCENE_FEATURES))
    parser.add_argument("--beams-s2-target", type=float, default=0.35)
    parser.add_argument("--beams-s2-weight", type=float, default=0.45)
    parser.add_argument("--beams-s3-weight", type=float, default=1.0)
    parser.add_argument("--beams-s1-weight", type=float, default=1.0)
    parser.add_argument("--clean-weight", type=float, default=1.5)
    parser.add_argument("--hard-negative-weight", type=float, default=2.0)
    parser.add_argument("--other-negative-weight", type=float, default=1.5)
    parser.add_argument("--other-corruptions", default="negative", choices=["negative", "ignore"])
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=5000)
    parser.add_argument("--max-missing-feature-rate", type=float, default=0.0)
    parser.add_argument("--allow-missing-features", action="store_true")
    parser.add_argument("--allow-debug-val", action="store_true")
    parser.add_argument("--allow-unsafe-cross-branch-features", action="store_true")
    args = parser.parse_args()

    check_formal_inputs(args.scene_log, allow_debug_val=args.allow_debug_val)
    features = [name.strip() for name in args.features.split(",") if name.strip()]
    unsafe = sorted(set(features) & UNSAFE_SCENE_FEATURES)
    if unsafe and not args.allow_unsafe_cross_branch_features:
        raise SystemExit(
            "Refusing unsafe scene features by default: "
            f"{unsafe}. Use route-derived or branch-local formal features."
        )

    paths = find_scene_logs(args.scene_log)
    if not paths:
        raise FileNotFoundError(f"No EV scene sidecar logs found: {args.scene_log}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    summary_path = args.summary or os.path.splitext(args.output)[0] + "_summary.json"

    counts = Counter()
    missing = Counter()
    by_condition = defaultdict(lambda: {"n": 0, "target_sum": 0.0, "weight_sum": 0.0})
    start = time.time()
    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for path in paths:
            check_formal_inputs(path, allow_debug_val=args.allow_debug_val)
            with open_text(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    counts["seen"] += 1
                    if not split_is_allowed(row, allow_debug_val=args.allow_debug_val):
                        counts["skip_non_train_split"] += 1
                        continue
                    target, weight, policy = label_for(row, args)
                    if target is None:
                        counts["skip_policy_ignore"] += 1
                        continue
                    cond = condition_name(row)
                    feat = {}
                    for name in features:
                        if name not in row:
                            missing[name] += 1
                        feat[name] = safe_float(row.get(name), 0.0)
                    rec = {
                        "sample_token": row.get("sample_token"),
                        "scene_token": row.get("scene_token"),
                        "condition": cond,
                        "ev_split": row.get("ev_split"),
                        "features": feat,
                        "targets": {"scene_bev_damage_applicable": float(target)},
                        "weights": {"scene_bev_damage_applicable": float(weight)},
                        "label_policy": policy,
                    }
                    out.write(json.dumps(rec, sort_keys=True) + "\n")
                    written += 1
                    counts[f"policy_{policy}"] += 1
                    by_condition[cond]["n"] += 1
                    by_condition[cond]["target_sum"] += float(target)
                    by_condition[cond]["weight_sum"] += float(weight)
                    if args.max_records and written >= args.max_records:
                        break
                    if args.progress_interval and written % args.progress_interval == 0:
                        elapsed = max(time.time() - start, 1e-6)
                        print(
                            f"[UMR scene labels] {written} rows, {written / elapsed:.1f} row/s",
                            file=sys.stderr,
                            flush=True,
                        )
            if args.max_records and written >= args.max_records:
                break

    condition_summary = {}
    for cond, item in sorted(by_condition.items()):
        condition_summary[cond] = {
            "n": item["n"],
            "target_mean": item["target_sum"] / max(item["n"], 1),
            "weight_mean": item["weight_sum"] / max(item["n"], 1),
        }
    missing_rates = {
        name: count / max(written, 1)
        for name, count in sorted(missing.items())
    }
    max_missing_rate = max(float(args.max_missing_feature_rate), 0.0)
    bad_missing = {
        name: rate
        for name, rate in missing_rates.items()
        if rate > max_missing_rate
    }
    if bad_missing and not args.allow_missing_features:
        counts["unsafe_missing_features"] = len(bad_missing)
    payload = {
        "scene_log": args.scene_log,
        "log_paths": paths,
        "log_path_sha256": {path: sha256_file(path) for path in paths},
        "features": features,
        "counts": dict(counts),
        "written": written,
        "output_labels": args.output,
        "output_labels_sha256": sha256_file(args.output),
        "missing_features": dict(missing),
        "missing_feature_rates": missing_rates,
        "bad_missing_features": bad_missing,
        "conditions": condition_summary,
        "label_policy": {
            "beams_s3": {"target": 1.0, "weight": args.beams_s3_weight},
            "beams_s2": {"target": args.beams_s2_target, "weight": args.beams_s2_weight},
            "beams_s1": {"target": 0.0, "weight": args.beams_s1_weight},
            "clean": {"target": 0.0, "weight": args.clean_weight},
            "fog_motionblur": {"target": 0.0, "weight": args.hard_negative_weight},
            "other_corruptions": args.other_corruptions,
        },
        "formal_calibration_safe": bool(
            not args.allow_debug_val and
            not unsafe and
            not bad_missing
        ),
        "debug_only": bool(args.allow_debug_val or unsafe or bad_missing),
        "unsafe_features": unsafe,
        "notes": [
            "corruption_type/severity are train-time labels only.",
            "runtime mapper uses learned scene features only.",
            "beams_s2 is intentionally weak-positive to encourage severity-aware budgets.",
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"[UMR scene labels] wrote: {args.output}")
    print(f"[UMR scene labels] wrote: {summary_path}")
    print(f"[UMR scene labels] written={written} seen={counts['seen']}")
    if bad_missing and not args.allow_missing_features:
        raise RuntimeError(
            "Scene feature schema mismatch: missing feature rates exceed "
            f"{max_missing_rate}: {bad_missing}. Summary was written to {summary_path}."
        )


if __name__ == "__main__":
    main()
