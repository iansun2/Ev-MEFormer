#!/usr/bin/env python3
"""Diagnose why UMR SUP/UGSUP edl_logs remain conservative.

The targeted RA table only shows AP/NDS. This tool reads EV sidecar logs and
breaks the supplement path into gates:

  scene gate -> replace slots -> IMG score candidates -> utility candidates
  -> actual added predictions.

It supports the sidecar manifest written under targeted probe output dirs.
"""

import argparse
import csv
import glob
import gzip
import json
import math
import os
import re
from collections import Counter, defaultdict


STATUS = {
    1: "max_add_disabled",
    2: "missing_source_tensor",
    3: "v7_info_missing",
    4: "no_added",
    100: "added",
}

NOADD = {
    0: "added_or_not_applicable",
    1: "max_add_disabled",
    2: "missing_source_tensor",
    3: "v7_info_missing",
    10: "scene_gate_zero",
    11: "replace_slot_zero",
    12: "candidate_zero",
    13: "utility_gate_zero",
    14: "marginal_gate_zero",
    15: "modality_guard_abstain",
}


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


def iter_jsonl(path):
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_float(value):
    try:
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def safe_int(value):
    value = safe_float(value)
    if value is None:
        return None
    return int(round(value))


def mean(values):
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def percentile(values, q):
    vals = sorted(v for v in (safe_float(x) for x in values) if v is not None)
    if not vals:
        return None
    pos = (len(vals) - 1) * float(q) / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    w = pos - lo
    return vals[lo] * (1.0 - w) + vals[hi] * w


def describe(values):
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"count": 0}
    return {
        "count": len(vals),
        "mean": mean(vals),
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "p95": percentile(vals, 95),
        "p99": percentile(vals, 99),
        "min": min(vals),
        "max": max(vals),
    }


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def resolve_existing(path, base_dir):
    if not path:
        return path
    candidates = [path]
    if not os.path.isabs(path):
        candidates.append(os.path.join(base_dir, path))
        candidates.append(os.path.abspath(path))
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(path)


def find_manifest(root):
    if os.path.isfile(root):
        return os.path.abspath(root)
    candidates = [
        os.path.join(root, "edl_logs", "log_manifest.tsv"),
        os.path.join(root, "log_manifest.tsv"),
    ]
    candidates.extend(glob.glob(os.path.join(root, "**", "log_manifest.tsv"), recursive=True))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(f"No log_manifest.tsv found under: {root}")


def read_manifest(manifest_path):
    base_dir = os.path.dirname(manifest_path)
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            for key in (
                    "edl_scene_summary_jsonl",
                    "edl_prediction_light_jsonl",
                    "edl_sidecar_meta_json",
                    "edl_branch_candidates_jsonl"):
                row[key] = resolve_existing(row.get(key, ""), base_dir)
            if not row.get("edl_branch_candidates_jsonl"):
                row["edl_branch_candidates_jsonl"] = resolve_branch_candidates(row)
            rows.append(row)
    return rows


def resolve_branch_candidates(row):
    meta_path = row.get("edl_sidecar_meta_json")
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            rel_path = meta.get("branch_candidates_path")
            if rel_path:
                return resolve_existing(rel_path, os.path.dirname(meta_path))
        except Exception:
            pass
    search_dirs = []
    for key in ("edl_scene_summary_jsonl", "edl_prediction_light_jsonl"):
        path = row.get(key)
        if path:
            search_dirs.append(os.path.dirname(path))
    for search_dir in search_dirs:
        matches = sorted(glob.glob(os.path.join(search_dir, "ev_branch_candidates*.jsonl*")))
        if matches:
            return os.path.abspath(matches[0])
    return ""


def condition_name(row):
    split = str(row.get("split", ""))
    corruption = str(row.get("corruption", ""))
    severity = str(row.get("severity", ""))
    if split == "clean" or corruption in ("", "clean", "none"):
        return "clean"
    return f"{corruption}_s{severity}"


def parse_metrics(root):
    metrics = {}
    for path in glob.glob(os.path.join(root, "*_output.txt")):
        name = os.path.basename(path)
        if name == "clean_output.txt":
            condition = "clean"
        else:
            condition = name[:-len("_output.txt")].replace("_3", "_s3").replace("_2", "_s2").replace("_1", "_s1")
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue
        row = {}
        for key in ("NDS", "mAP", "mATE", "mASE", "mAOE", "mAVE", "mAAE"):
            match = re.search(rf"{key}:\s*([0-9.]+)", text)
            if match:
                row[key] = float(match.group(1))
        if row:
            metrics[condition] = row
    return metrics


def summarize_branch_candidates(path, entry):
    if not path or not os.path.exists(path):
        return
    entry["branch_candidate_paths"].add(path)
    max_records = int(os.environ.get("UGSUP_BRANCH_CANDIDATE_MAX_RECORDS", "0") or 0)
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            entry["branch_candidate_rows"] += 1
            if "sample_token" in row:
                entry["branch_candidate_samples"].add(str(row["sample_token"]))
            branch = str(row.get("branch", "unknown"))
            entry["branch_candidate_branch"][branch] += 1
            score = safe_float(row.get("score"))
            if score is not None:
                entry["branch_candidate_score"].append(score)
            valid_value = safe_float(row.get("valid"))
            if valid_value is not None and valid_value > 0.5:
                entry["branch_candidate_valid"] += 1
            if max_records and entry["branch_candidate_rows"] >= max_records:
                entry["branch_candidate_truncated"] = True
                break


def classify_bottleneck(row):
    def stat(name, key, default=0.0):
        item = row.get(name, {})
        value = item.get(key) if isinstance(item, dict) else None
        return default if value is None else value

    if row["samples"] <= 0:
        return "missing_logs"
    if row.get("scene_supp_field_presence", {}).get("scene_supp_enabled_runtime", 0.0) <= 0.0:
        return "supp_scene_telemetry_missing"
    if (
        row.get("preds", 0) > 0 and
        row.get("pred_supp_field_presence", {}).get("supp_added", 0.0) <= 0.0
    ):
        return "supp_prediction_telemetry_missing"
    noadd = row["noadd_reason_top"]
    if noadd and noadd[0][1] > 0:
        code = noadd[0][0]
        if code in NOADD and code != 0:
            return NOADD[code]
    if stat("scene_gate", "count") <= 0:
        return "telemetry_missing"
    added = stat("scene_added_ratio", "mean")
    gate = stat("scene_gate", "positive_ratio")
    candidate = stat("scene_candidate_ratio", "positive_ratio")
    utility_count = stat("scene_utility_candidate_ratio", "count")
    utility = stat("scene_utility_candidate_ratio", "positive_ratio")
    slots = stat("scene_replace_slot_ratio", "positive_ratio")
    budget = stat("scene_effective_max_add_queries", "mean")
    if gate <= 0.01:
        return "scene_gate_too_rare"
    if budget <= 0.0:
        return "dynamic_budget_zero"
    if slots <= 0.01:
        return "replace_slots_too_rare"
    if candidate <= 0.01:
        return "candidate_too_rare"
    if utility_count > 0 and utility <= 0.01:
        return "utility_gate_too_rare"
    if added <= 0.002:
        return "selection_or_nms_too_strict"
    return "added_but_metric_gain_small"


def ratio_positive(values):
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(1 for v in vals if v > 0.0) / len(vals)


def summarize_condition(cond, entry):
    status_counts = Counter(safe_int(v) for v in entry["status"] if safe_int(v) is not None)
    noadd_counts = Counter(safe_int(v) for v in entry["noadd"] if safe_int(v) is not None)
    scene_denom = max(entry["samples"], 1)
    pred_denom = max(entry["preds"], 1)
    row = {
        "condition": cond,
        "samples": entry["samples"],
        "preds": entry["preds"],
        "pred_added": entry["pred_added"],
        "pred_added_ratio": entry["pred_added"] / entry["preds"] if entry["preds"] else 0.0,
        "scene_gate": describe(entry["scene_gate"]),
        "scene_candidate_ratio": describe(entry["scene_candidate_ratio"]),
        "scene_img_candidate_ratio": describe(entry["scene_img_candidate_ratio"]),
        "scene_bev_candidate_ratio": describe(entry["scene_bev_candidate_ratio"]),
        "scene_source_bev_query_ratio": describe(entry["scene_source_bev_query_ratio"]),
        "scene_selected_img_ratio": describe(entry["scene_selected_img_ratio"]),
        "scene_selected_bev_ratio": describe(entry["scene_selected_bev_ratio"]),
        "scene_branch_aware_enabled": describe(entry["scene_branch_aware_enabled"]),
        "scene_branch_mapper_enabled": describe(entry["scene_branch_mapper_enabled"]),
        "scene_img_utility_candidate_ratio": describe(entry["scene_img_utility_candidate_ratio"]),
        "scene_bev_utility_candidate_ratio": describe(entry["scene_bev_utility_candidate_ratio"]),
        "scene_img_utility_score_mean": describe(entry["scene_img_utility_score_mean"]),
        "scene_bev_utility_score_mean": describe(entry["scene_bev_utility_score_mean"]),
        "scene_img_utility_threshold": describe(entry["scene_img_utility_threshold"]),
        "scene_bev_utility_threshold": describe(entry["scene_bev_utility_threshold"]),
        "scene_utility_candidate_ratio": describe(entry["scene_utility_candidate_ratio"]),
        "scene_fused_slot_utility_score_mean": describe(entry["scene_fused_slot_utility_score_mean"]),
        "scene_marginal_enabled": describe(entry["scene_marginal_enabled"]),
        "scene_marginal_margin": describe(entry["scene_marginal_margin"]),
        "scene_dual_marginal_margin": describe(entry["scene_dual_marginal_margin"]),
        "scene_marginal_checked_ratio": describe(entry["scene_marginal_checked_ratio"]),
        "scene_marginal_reject_ratio": describe(entry["scene_marginal_reject_ratio"]),
        "scene_marginal_pass_ratio": describe(entry["scene_marginal_pass_ratio"]),
        "scene_selected_margin_score_mean": describe(entry["scene_selected_margin_score_mean"]),
        "scene_added_ratio": describe(entry["scene_added_ratio"]),
        "scene_replace_slot_ratio": describe(entry["scene_replace_slot_ratio"]),
        "scene_gate_pre_guard_ratio": describe(entry["scene_gate_pre_guard_ratio"]),
        "scene_gate_post_guard_ratio": describe(entry["scene_gate_post_guard_ratio"]),
        "scene_img_damage_gate_ratio": describe(entry["scene_img_damage_gate_ratio"]),
        "scene_img_damage_score": describe(entry["scene_img_damage_score"]),
        "scene_dual_damage_gate_ratio": describe(entry["scene_dual_damage_gate_ratio"]),
        "scene_dual_gate_mode_code": describe(entry["scene_dual_gate_mode_code"]),
        "scene_scene_mapper_dual_gate_ratio": describe(entry["scene_scene_mapper_dual_gate_ratio"]),
        "scene_bev_damage_gate_ratio": describe(entry["scene_bev_damage_gate_ratio"]),
        "scene_bev_damage_upper_gate_ratio": describe(
            entry["scene_bev_damage_upper_gate_ratio"]),
        "scene_fused_damage_gate_ratio": describe(entry["scene_fused_damage_gate_ratio"]),
        "scene_bev_fused_dominance_gate_ratio": describe(
            entry["scene_bev_fused_dominance_gate_ratio"]),
        "scene_bev_fused_normalized_ratio_mean": describe(
            entry["scene_bev_fused_normalized_ratio_mean"]),
        "scene_dual_abstain_gate_ratio": describe(entry["scene_dual_abstain_gate_ratio"]),
        "scene_img_abstain_gate_ratio": describe(entry["scene_img_abstain_gate_ratio"]),
        "scene_dual_abstain_score": describe(entry["scene_dual_abstain_score"]),
        "scene_img_abstain_score": describe(entry["scene_img_abstain_score"]),
        "scene_dual_abstain_threshold": describe(entry["scene_dual_abstain_threshold"]),
        "scene_img_abstain_threshold": describe(entry["scene_img_abstain_threshold"]),
        "scene_dual_abstain_available": describe(entry["scene_dual_abstain_available"]),
        "scene_img_abstain_available": describe(entry["scene_img_abstain_available"]),
        "scene_qsem_gate_enabled": describe(entry["scene_qsem_gate_enabled"]),
        "scene_qsem_gate_ratio": describe(entry["scene_qsem_gate_ratio"]),
        "scene_qsem_disagree_gate_ratio": describe(entry["scene_qsem_disagree_gate_ratio"]),
        "scene_qsem_margin_gate_ratio": describe(entry["scene_qsem_margin_gate_ratio"]),
        "scene_qsem_disagree_score": describe(entry["scene_qsem_disagree_score"]),
        "scene_qsem_margin_score": describe(entry["scene_qsem_margin_score"]),
        "scene_qsem_disagree_threshold": describe(entry["scene_qsem_disagree_threshold"]),
        "scene_qsem_margin_threshold": describe(entry["scene_qsem_margin_threshold"]),
        "scene_qsem_disagree_available": describe(entry["scene_qsem_disagree_available"]),
        "scene_qsem_margin_available": describe(entry["scene_qsem_margin_available"]),
        "scene_contextheight_gate_enabled": describe(entry["scene_contextheight_gate_enabled"]),
        "scene_contextheight_gate_ratio": describe(entry["scene_contextheight_gate_ratio"]),
        "scene_contextheight_context_gate_ratio": describe(
            entry["scene_contextheight_context_gate_ratio"]),
        "scene_contextheight_height_gate_ratio": describe(
            entry["scene_contextheight_height_gate_ratio"]),
        "scene_contextheight_v9_bev_gate_ratio": describe(
            entry["scene_contextheight_v9_bev_gate_ratio"]),
        "scene_contextheight_context_score": describe(
            entry["scene_contextheight_context_score"]),
        "scene_contextheight_height_score": describe(
            entry["scene_contextheight_height_score"]),
        "scene_contextheight_v9_bev_score": describe(
            entry["scene_contextheight_v9_bev_score"]),
        "scene_contextheight_context_threshold": describe(
            entry["scene_contextheight_context_threshold"]),
        "scene_contextheight_height_threshold": describe(
            entry["scene_contextheight_height_threshold"]),
        "scene_contextheight_v9_bev_threshold": describe(
            entry["scene_contextheight_v9_bev_threshold"]),
        "scene_contextheight_context_available": describe(
            entry["scene_contextheight_context_available"]),
        "scene_contextheight_height_available": describe(
            entry["scene_contextheight_height_available"]),
        "scene_contextheight_v9_bev_available": describe(
            entry["scene_contextheight_v9_bev_available"]),
        "scene_qsem_top100_count": describe(entry["scene_qsem_top100_count"]),
        "scene_qsem_top100_max_score": describe(entry["scene_qsem_top100_max_score"]),
        "scene_qsem_top100_disagree": describe(entry["scene_qsem_top100_disagree"]),
        "scene_qsem_top100_margin": describe(entry["scene_qsem_top100_margin"]),
        "scene_qsem_top300_count": describe(entry["scene_qsem_top300_count"]),
        "scene_qsem_top300_max_score": describe(entry["scene_qsem_top300_max_score"]),
        "scene_qsem_top300_disagree": describe(entry["scene_qsem_top300_disagree"]),
        "scene_qsem_top300_margin": describe(entry["scene_qsem_top300_margin"]),
        "scene_img_damage_guard_enabled": describe(entry["scene_img_damage_guard_enabled"]),
        "scene_img_damage_guard_mode_code": describe(entry["scene_img_damage_guard_mode_code"]),
        "scene_img_damage_threshold": describe(entry["scene_img_damage_threshold"]),
        "scene_img_damage_threshold_available": describe(entry["scene_img_damage_threshold_available"]),
        "scene_img_damage_threshold_scale": describe(
            entry["scene_img_damage_threshold_scale"]),
        "scene_bev_damage_threshold_available": describe(entry["scene_bev_damage_threshold_available"]),
        "scene_bev_damage_upper_threshold_available": describe(
            entry["scene_bev_damage_upper_threshold_available"]),
        "scene_fused_damage_threshold_available": describe(
            entry["scene_fused_damage_threshold_available"]),
        "scene_bev_fused_reference_available": describe(
            entry["scene_bev_fused_reference_available"]),
        "scene_fused_damage_threshold_scale": describe(
            entry["scene_fused_damage_threshold_scale"]),
        "scene_bev_fused_dominance_ratio": describe(
            entry["scene_bev_fused_dominance_ratio"]),
        "scene_bev_damage_upper_threshold_scale": describe(
            entry["scene_bev_damage_upper_threshold_scale"]),
        "scene_dual_bev_rank_boost": describe(entry["scene_dual_bev_rank_boost"]),
        "scene_dual_scene_mapper_threshold": describe(entry["scene_dual_scene_mapper_threshold"]),
        "scene_dual_bev_threshold_scale": describe(entry["scene_dual_bev_threshold_scale"]),
        "scene_utility_score_mean": describe(entry["scene_utility_score_mean"]),
        "scene_mapper_score_mean": describe(entry["scene_mapper_score_mean"]),
        "scene_mapper_gate_ratio": describe(entry["scene_mapper_gate_ratio"]),
        "scene_dynamic_budget_ratio": describe(entry["scene_dynamic_budget_ratio"]),
        "scene_effective_max_add_queries": describe(entry["scene_effective_max_add_queries"]),
        "added_img_score": describe(entry["added_img_score"]),
        "added_branch_score": describe(entry["added_branch_score"]),
        "added_utility_score": describe(entry["added_utility_score"]),
        "replaced_fused_score": describe(entry["replaced_fused_score"]),
        "status_top": status_counts.most_common(),
        "noadd_reason_top": noadd_counts.most_common(),
        "scene_supp_field_presence": {
            key: entry["scene_field_seen"].get(key, 0) / scene_denom
            for key in (
                "scene_supp_enabled_runtime",
                "scene_supp_status_code",
                "scene_supp_noadd_reason_code",
                "scene_supp_scene_gate_ratio",
                "scene_supp_candidate_ratio",
                "scene_supp_img_candidate_ratio",
                "scene_supp_bev_candidate_ratio",
                "scene_supp_source_bev_query_ratio",
                "scene_supp_selected_img_ratio",
                "scene_supp_selected_bev_ratio",
                "scene_supp_branch_aware_enabled",
                "scene_supp_branch_mapper_enabled",
                "scene_supp_img_utility_candidate_ratio",
                "scene_supp_bev_utility_candidate_ratio",
                "scene_supp_img_utility_score_mean",
                "scene_supp_bev_utility_score_mean",
                "scene_supp_img_utility_threshold",
                "scene_supp_bev_utility_threshold",
                "scene_supp_utility_candidate_ratio",
                "scene_supp_fused_slot_utility_score_mean",
                "scene_supp_marginal_enabled",
                "scene_supp_marginal_margin",
                "scene_supp_dual_marginal_margin",
                "scene_supp_marginal_checked_ratio",
                "scene_supp_marginal_reject_ratio",
                "scene_supp_marginal_pass_ratio",
                "scene_supp_selected_margin_score_mean",
                "scene_supp_added_ratio",
                "scene_supp_replace_slot_ratio",
                "scene_supp_scene_gate_pre_guard_ratio",
                "scene_supp_scene_gate_post_guard_ratio",
                "scene_supp_img_damage_guard_enabled",
                "scene_supp_img_damage_guard_mode_code",
                "scene_supp_img_damage_gate_ratio",
                "scene_supp_img_damage_score",
                "scene_supp_img_damage_threshold",
                "scene_supp_dual_damage_gate_ratio",
                "scene_supp_dual_gate_mode_code",
                "scene_supp_scene_mapper_dual_gate_ratio",
                "scene_supp_bev_damage_gate_ratio",
                "scene_supp_bev_damage_upper_gate_ratio",
                "scene_supp_fused_damage_gate_ratio",
                "scene_supp_bev_fused_dominance_gate_ratio",
                "scene_supp_bev_fused_normalized_ratio_mean",
                "scene_supp_dual_abstain_gate_ratio",
                "scene_supp_img_abstain_gate_ratio",
                "scene_supp_dual_abstain_score",
                "scene_supp_img_abstain_score",
                "scene_supp_dual_abstain_threshold",
                "scene_supp_img_abstain_threshold",
                "scene_supp_dual_abstain_available",
                "scene_supp_img_abstain_available",
                "scene_supp_qsem_gate_enabled",
                "scene_supp_qsem_gate_ratio",
                "scene_supp_qsem_disagree_gate_ratio",
                "scene_supp_qsem_margin_gate_ratio",
                "scene_supp_qsem_disagree_score",
                "scene_supp_qsem_margin_score",
                "scene_supp_qsem_disagree_threshold",
                "scene_supp_qsem_margin_threshold",
                "scene_supp_qsem_disagree_available",
                "scene_supp_qsem_margin_available",
                "scene_supp_contextheight_gate_enabled",
                "scene_supp_contextheight_gate_ratio",
                "scene_supp_contextheight_context_gate_ratio",
                "scene_supp_contextheight_height_gate_ratio",
                "scene_supp_contextheight_v9_bev_gate_ratio",
                "scene_supp_contextheight_context_score",
                "scene_supp_contextheight_height_score",
                "scene_supp_contextheight_v9_bev_score",
                "scene_supp_contextheight_context_threshold",
                "scene_supp_contextheight_height_threshold",
                "scene_supp_contextheight_v9_bev_threshold",
                "scene_supp_contextheight_context_available",
                "scene_supp_contextheight_height_available",
                "scene_supp_contextheight_v9_bev_available",
                "scene_supp_img_damage_threshold_available",
                "scene_supp_img_damage_threshold_scale",
                "scene_supp_bev_damage_threshold_available",
                "scene_supp_bev_damage_upper_threshold_available",
                "scene_supp_fused_damage_threshold_available",
                "scene_supp_bev_fused_reference_available",
                "scene_supp_fused_damage_threshold_scale",
                "scene_supp_bev_fused_dominance_ratio",
                "scene_supp_bev_damage_upper_threshold_scale",
                "scene_supp_dual_bev_rank_boost",
                "scene_supp_dual_scene_mapper_threshold",
                "scene_supp_dual_bev_threshold_scale",
                "scene_supp_scene_mapper_score_mean",
                "scene_supp_scene_mapper_gate_ratio",
                "scene_supp_dynamic_budget_ratio",
                "scene_supp_effective_max_add_queries",
            )
        },
        "pred_supp_field_presence": {
            key: entry["pred_field_seen"].get(key, 0) / pred_denom
            for key in (
                "supp_added",
                "supp_img_score",
                "supp_bev_score",
                "supp_branch_score",
                "supp_fused_score",
                "supp_utility_score",
                "supp_fused_slot_utility_score",
                "supp_marginal_utility_score",
                "supp_source_branch",
            )
        },
        "added_class_top": entry["added_class"].most_common(10),
        "added_source_top": entry["added_source"].most_common(),
        "branch_candidate": {
            "paths": sorted(entry["branch_candidate_paths"]),
            "rows": entry["branch_candidate_rows"],
            "samples": len(entry["branch_candidate_samples"]),
            "branch_counts": dict(entry["branch_candidate_branch"]),
            "valid": entry["branch_candidate_valid"],
            "score": describe(entry["branch_candidate_score"]),
            "truncated": bool(entry["branch_candidate_truncated"]),
        },
    }
    for key in ("scene_gate", "scene_candidate_ratio", "scene_utility_candidate_ratio",
                "scene_added_ratio", "scene_replace_slot_ratio"):
        row[key]["positive_ratio"] = ratio_positive(entry[key])
    row["bottleneck"] = classify_bottleneck(row)
    return row


def analyze_root(root):
    manifest = find_manifest(root)
    rows = read_manifest(manifest)
    summary = {}
    warnings = []
    for row in rows:
        cond = condition_name(row)
        entry = summary.setdefault(cond, {
            "samples": 0,
            "preds": 0,
            "pred_added": 0,
            "scene_gate": [],
            "scene_candidate_ratio": [],
            "scene_img_candidate_ratio": [],
            "scene_bev_candidate_ratio": [],
            "scene_source_bev_query_ratio": [],
            "scene_selected_img_ratio": [],
            "scene_selected_bev_ratio": [],
            "scene_branch_aware_enabled": [],
            "scene_branch_mapper_enabled": [],
            "scene_img_utility_candidate_ratio": [],
            "scene_bev_utility_candidate_ratio": [],
            "scene_img_utility_score_mean": [],
            "scene_bev_utility_score_mean": [],
            "scene_img_utility_threshold": [],
            "scene_bev_utility_threshold": [],
            "scene_utility_candidate_ratio": [],
            "scene_utility_score_mean": [],
            "scene_fused_slot_utility_score_mean": [],
            "scene_marginal_enabled": [],
            "scene_marginal_margin": [],
            "scene_dual_marginal_margin": [],
            "scene_marginal_checked_ratio": [],
            "scene_marginal_reject_ratio": [],
            "scene_marginal_pass_ratio": [],
            "scene_selected_margin_score_mean": [],
            "scene_mapper_score_mean": [],
            "scene_mapper_gate_ratio": [],
            "scene_dynamic_budget_ratio": [],
            "scene_effective_max_add_queries": [],
            "scene_added_ratio": [],
            "scene_replace_slot_ratio": [],
            "scene_gate_pre_guard_ratio": [],
            "scene_gate_post_guard_ratio": [],
            "scene_img_damage_gate_ratio": [],
            "scene_img_damage_score": [],
            "scene_dual_damage_gate_ratio": [],
            "scene_dual_gate_mode_code": [],
            "scene_scene_mapper_dual_gate_ratio": [],
            "scene_bev_damage_gate_ratio": [],
            "scene_bev_damage_upper_gate_ratio": [],
            "scene_fused_damage_gate_ratio": [],
            "scene_bev_fused_dominance_gate_ratio": [],
            "scene_bev_fused_normalized_ratio_mean": [],
            "scene_dual_abstain_gate_ratio": [],
            "scene_img_abstain_gate_ratio": [],
            "scene_dual_abstain_score": [],
            "scene_img_abstain_score": [],
            "scene_dual_abstain_threshold": [],
            "scene_img_abstain_threshold": [],
            "scene_dual_abstain_available": [],
            "scene_img_abstain_available": [],
            "scene_qsem_gate_enabled": [],
            "scene_qsem_gate_ratio": [],
            "scene_qsem_disagree_gate_ratio": [],
            "scene_qsem_margin_gate_ratio": [],
            "scene_qsem_disagree_score": [],
            "scene_qsem_margin_score": [],
            "scene_qsem_disagree_threshold": [],
            "scene_qsem_margin_threshold": [],
            "scene_qsem_disagree_available": [],
            "scene_qsem_margin_available": [],
            "scene_contextheight_gate_enabled": [],
            "scene_contextheight_gate_ratio": [],
            "scene_contextheight_context_gate_ratio": [],
            "scene_contextheight_height_gate_ratio": [],
            "scene_contextheight_v9_bev_gate_ratio": [],
            "scene_contextheight_context_score": [],
            "scene_contextheight_height_score": [],
            "scene_contextheight_v9_bev_score": [],
            "scene_contextheight_context_threshold": [],
            "scene_contextheight_height_threshold": [],
            "scene_contextheight_v9_bev_threshold": [],
            "scene_contextheight_context_available": [],
            "scene_contextheight_height_available": [],
            "scene_contextheight_v9_bev_available": [],
            "scene_qsem_top100_count": [],
            "scene_qsem_top100_max_score": [],
            "scene_qsem_top100_disagree": [],
            "scene_qsem_top100_margin": [],
            "scene_qsem_top300_count": [],
            "scene_qsem_top300_max_score": [],
            "scene_qsem_top300_disagree": [],
            "scene_qsem_top300_margin": [],
            "scene_img_damage_guard_enabled": [],
            "scene_img_damage_guard_mode_code": [],
            "scene_img_damage_threshold": [],
            "scene_img_damage_threshold_available": [],
            "scene_img_damage_threshold_scale": [],
            "scene_bev_damage_threshold_available": [],
            "scene_bev_damage_upper_threshold_available": [],
            "scene_fused_damage_threshold_available": [],
            "scene_bev_fused_reference_available": [],
            "scene_fused_damage_threshold_scale": [],
            "scene_bev_fused_dominance_ratio": [],
            "scene_bev_damage_upper_threshold_scale": [],
            "scene_dual_bev_rank_boost": [],
            "scene_dual_scene_mapper_threshold": [],
            "scene_dual_bev_threshold_scale": [],
            "status": [],
            "noadd": [],
            "scene_field_seen": Counter(),
            "pred_field_seen": Counter(),
            "added_img_score": [],
            "added_branch_score": [],
            "added_utility_score": [],
            "replaced_fused_score": [],
            "added_class": Counter(),
            "added_source": Counter(),
            "branch_candidate_rows": 0,
            "branch_candidate_samples": set(),
            "branch_candidate_branch": Counter(),
            "branch_candidate_score": [],
            "branch_candidate_valid": 0,
            "branch_candidate_paths": set(),
            "branch_candidate_truncated": False,
        })
        scene_path = row.get("edl_scene_summary_jsonl")
        pred_path = row.get("edl_prediction_light_jsonl")
        branch_path = row.get("edl_branch_candidates_jsonl")
        if not scene_path or not os.path.exists(scene_path):
            warnings.append(f"missing scene sidecar for {cond}: {scene_path}")
        else:
            for scene in iter_jsonl(scene_path):
                entry["samples"] += 1
                for field in (
                        "scene_supp_enabled_runtime",
                        "scene_supp_status_code",
                        "scene_supp_noadd_reason_code",
                        "scene_supp_scene_gate_ratio",
                        "scene_supp_candidate_ratio",
                        "scene_supp_img_candidate_ratio",
                        "scene_supp_bev_candidate_ratio",
                        "scene_supp_source_bev_query_ratio",
                        "scene_supp_selected_img_ratio",
                        "scene_supp_selected_bev_ratio",
                        "scene_supp_branch_aware_enabled",
                        "scene_supp_branch_mapper_enabled",
                        "scene_supp_img_utility_candidate_ratio",
                        "scene_supp_bev_utility_candidate_ratio",
                        "scene_supp_img_utility_score_mean",
                        "scene_supp_bev_utility_score_mean",
                        "scene_supp_img_utility_threshold",
                        "scene_supp_bev_utility_threshold",
                        "scene_supp_utility_candidate_ratio",
                        "scene_supp_fused_slot_utility_score_mean",
                        "scene_supp_marginal_enabled",
                        "scene_supp_marginal_margin",
                        "scene_supp_dual_marginal_margin",
                        "scene_supp_marginal_checked_ratio",
                        "scene_supp_marginal_reject_ratio",
                        "scene_supp_marginal_pass_ratio",
                        "scene_supp_selected_margin_score_mean",
                        "scene_supp_added_ratio",
                        "scene_supp_replace_slot_ratio",
                        "scene_supp_scene_gate_pre_guard_ratio",
                        "scene_supp_scene_gate_post_guard_ratio",
                        "scene_supp_img_damage_guard_enabled",
                        "scene_supp_img_damage_guard_mode_code",
                        "scene_supp_img_damage_gate_ratio",
                        "scene_supp_img_damage_score",
                        "scene_supp_img_damage_threshold",
                        "scene_supp_dual_damage_gate_ratio",
                        "scene_supp_dual_gate_mode_code",
                        "scene_supp_scene_mapper_dual_gate_ratio",
                        "scene_supp_bev_damage_gate_ratio",
                        "scene_supp_bev_damage_upper_gate_ratio",
                        "scene_supp_fused_damage_gate_ratio",
                        "scene_supp_bev_fused_dominance_gate_ratio",
                        "scene_supp_bev_fused_normalized_ratio_mean",
                        "scene_supp_dual_abstain_gate_ratio",
                        "scene_supp_img_abstain_gate_ratio",
                        "scene_supp_dual_abstain_score",
                        "scene_supp_img_abstain_score",
                        "scene_supp_dual_abstain_threshold",
                        "scene_supp_img_abstain_threshold",
                        "scene_supp_dual_abstain_available",
                        "scene_supp_img_abstain_available",
                        "scene_supp_qsem_gate_enabled",
                        "scene_supp_qsem_gate_ratio",
                        "scene_supp_qsem_disagree_gate_ratio",
                        "scene_supp_qsem_margin_gate_ratio",
                        "scene_supp_qsem_disagree_score",
                        "scene_supp_qsem_margin_score",
                        "scene_supp_qsem_disagree_threshold",
                        "scene_supp_qsem_margin_threshold",
                        "scene_supp_qsem_disagree_available",
                        "scene_supp_qsem_margin_available",
                        "scene_supp_contextheight_gate_enabled",
                        "scene_supp_contextheight_gate_ratio",
                        "scene_supp_contextheight_context_gate_ratio",
                        "scene_supp_contextheight_height_gate_ratio",
                        "scene_supp_contextheight_v9_bev_gate_ratio",
                        "scene_supp_contextheight_context_score",
                        "scene_supp_contextheight_height_score",
                        "scene_supp_contextheight_v9_bev_score",
                        "scene_supp_contextheight_context_threshold",
                        "scene_supp_contextheight_height_threshold",
                        "scene_supp_contextheight_v9_bev_threshold",
                        "scene_supp_contextheight_context_available",
                        "scene_supp_contextheight_height_available",
                        "scene_supp_contextheight_v9_bev_available",
                        "scene_qsem_top100_query_count",
                        "scene_qsem_top100_max_score_mean",
                        "scene_qsem_top100_img_bev_class_disagree_ratio",
                        "scene_qsem_top100_img_margin_mean",
                        "scene_qsem_top300_query_count",
                        "scene_qsem_top300_max_score_mean",
                        "scene_qsem_top300_img_bev_class_disagree_ratio",
                        "scene_qsem_top300_img_margin_mean",
                        "scene_supp_img_damage_threshold_available",
                        "scene_supp_img_damage_threshold_scale",
                        "scene_supp_bev_damage_threshold_available",
                        "scene_supp_bev_damage_upper_threshold_available",
                        "scene_supp_fused_damage_threshold_available",
                        "scene_supp_bev_fused_reference_available",
                        "scene_supp_fused_damage_threshold_scale",
                        "scene_supp_bev_fused_dominance_ratio",
                        "scene_supp_bev_damage_upper_threshold_scale",
                        "scene_supp_dual_bev_rank_boost",
                        "scene_supp_dual_scene_mapper_threshold",
                        "scene_supp_dual_bev_threshold_scale",
                        "scene_supp_scene_mapper_score_mean",
                        "scene_supp_scene_mapper_gate_ratio",
                        "scene_supp_dynamic_budget_ratio",
                        "scene_supp_effective_max_add_queries"):
                    if field in scene:
                        entry["scene_field_seen"][field] += 1
                entry["scene_gate"].append(scene.get("scene_supp_scene_gate_ratio"))
                entry["scene_candidate_ratio"].append(scene.get("scene_supp_candidate_ratio"))
                entry["scene_img_candidate_ratio"].append(scene.get("scene_supp_img_candidate_ratio"))
                entry["scene_bev_candidate_ratio"].append(scene.get("scene_supp_bev_candidate_ratio"))
                entry["scene_source_bev_query_ratio"].append(scene.get("scene_supp_source_bev_query_ratio"))
                entry["scene_selected_img_ratio"].append(scene.get("scene_supp_selected_img_ratio"))
                entry["scene_selected_bev_ratio"].append(scene.get("scene_supp_selected_bev_ratio"))
                entry["scene_branch_aware_enabled"].append(scene.get("scene_supp_branch_aware_enabled"))
                entry["scene_branch_mapper_enabled"].append(scene.get("scene_supp_branch_mapper_enabled"))
                entry["scene_img_utility_candidate_ratio"].append(scene.get("scene_supp_img_utility_candidate_ratio"))
                entry["scene_bev_utility_candidate_ratio"].append(scene.get("scene_supp_bev_utility_candidate_ratio"))
                entry["scene_img_utility_score_mean"].append(scene.get("scene_supp_img_utility_score_mean"))
                entry["scene_bev_utility_score_mean"].append(scene.get("scene_supp_bev_utility_score_mean"))
                entry["scene_img_utility_threshold"].append(scene.get("scene_supp_img_utility_threshold"))
                entry["scene_bev_utility_threshold"].append(scene.get("scene_supp_bev_utility_threshold"))
                entry["scene_utility_candidate_ratio"].append(scene.get("scene_supp_utility_candidate_ratio"))
                entry["scene_utility_score_mean"].append(scene.get("scene_supp_utility_score_mean"))
                entry["scene_fused_slot_utility_score_mean"].append(
                    scene.get("scene_supp_fused_slot_utility_score_mean"))
                entry["scene_marginal_enabled"].append(scene.get("scene_supp_marginal_enabled"))
                entry["scene_marginal_margin"].append(scene.get("scene_supp_marginal_margin"))
                entry["scene_dual_marginal_margin"].append(
                    scene.get("scene_supp_dual_marginal_margin"))
                entry["scene_marginal_checked_ratio"].append(
                    scene.get("scene_supp_marginal_checked_ratio"))
                entry["scene_marginal_reject_ratio"].append(
                    scene.get("scene_supp_marginal_reject_ratio"))
                entry["scene_marginal_pass_ratio"].append(
                    scene.get("scene_supp_marginal_pass_ratio"))
                entry["scene_selected_margin_score_mean"].append(
                    scene.get("scene_supp_selected_margin_score_mean"))
                entry["scene_mapper_score_mean"].append(scene.get("scene_supp_scene_mapper_score_mean"))
                entry["scene_mapper_gate_ratio"].append(scene.get("scene_supp_scene_mapper_gate_ratio"))
                entry["scene_dynamic_budget_ratio"].append(scene.get("scene_supp_dynamic_budget_ratio"))
                entry["scene_effective_max_add_queries"].append(scene.get("scene_supp_effective_max_add_queries"))
                entry["scene_added_ratio"].append(scene.get("scene_supp_added_ratio"))
                entry["scene_replace_slot_ratio"].append(scene.get("scene_supp_replace_slot_ratio"))
                entry["scene_gate_pre_guard_ratio"].append(
                    scene.get("scene_supp_scene_gate_pre_guard_ratio"))
                entry["scene_gate_post_guard_ratio"].append(
                    scene.get("scene_supp_scene_gate_post_guard_ratio"))
                entry["scene_img_damage_gate_ratio"].append(
                    scene.get("scene_supp_img_damage_gate_ratio"))
                entry["scene_img_damage_score"].append(
                    scene.get("scene_supp_img_damage_score"))
                entry["scene_dual_damage_gate_ratio"].append(
                    scene.get("scene_supp_dual_damage_gate_ratio"))
                entry["scene_dual_gate_mode_code"].append(
                    scene.get("scene_supp_dual_gate_mode_code"))
                entry["scene_scene_mapper_dual_gate_ratio"].append(
                    scene.get("scene_supp_scene_mapper_dual_gate_ratio"))
                entry["scene_bev_damage_gate_ratio"].append(
                    scene.get("scene_supp_bev_damage_gate_ratio"))
                entry["scene_bev_damage_upper_gate_ratio"].append(
                    scene.get("scene_supp_bev_damage_upper_gate_ratio"))
                entry["scene_fused_damage_gate_ratio"].append(
                    scene.get("scene_supp_fused_damage_gate_ratio"))
                entry["scene_bev_fused_dominance_gate_ratio"].append(
                    scene.get("scene_supp_bev_fused_dominance_gate_ratio"))
                entry["scene_bev_fused_normalized_ratio_mean"].append(
                    scene.get("scene_supp_bev_fused_normalized_ratio_mean"))
                entry["scene_dual_abstain_gate_ratio"].append(
                    scene.get("scene_supp_dual_abstain_gate_ratio"))
                entry["scene_img_abstain_gate_ratio"].append(
                    scene.get("scene_supp_img_abstain_gate_ratio"))
                entry["scene_dual_abstain_score"].append(
                    scene.get("scene_supp_dual_abstain_score"))
                entry["scene_img_abstain_score"].append(
                    scene.get("scene_supp_img_abstain_score"))
                entry["scene_dual_abstain_threshold"].append(
                    scene.get("scene_supp_dual_abstain_threshold"))
                entry["scene_img_abstain_threshold"].append(
                    scene.get("scene_supp_img_abstain_threshold"))
                entry["scene_dual_abstain_available"].append(
                    scene.get("scene_supp_dual_abstain_available"))
                entry["scene_img_abstain_available"].append(
                    scene.get("scene_supp_img_abstain_available"))
                entry["scene_qsem_gate_enabled"].append(
                    scene.get("scene_supp_qsem_gate_enabled"))
                entry["scene_qsem_gate_ratio"].append(
                    scene.get("scene_supp_qsem_gate_ratio"))
                entry["scene_qsem_disagree_gate_ratio"].append(
                    scene.get("scene_supp_qsem_disagree_gate_ratio"))
                entry["scene_qsem_margin_gate_ratio"].append(
                    scene.get("scene_supp_qsem_margin_gate_ratio"))
                entry["scene_qsem_disagree_score"].append(
                    scene.get("scene_supp_qsem_disagree_score"))
                entry["scene_qsem_margin_score"].append(
                    scene.get("scene_supp_qsem_margin_score"))
                entry["scene_qsem_disagree_threshold"].append(
                    scene.get("scene_supp_qsem_disagree_threshold"))
                entry["scene_qsem_margin_threshold"].append(
                    scene.get("scene_supp_qsem_margin_threshold"))
                entry["scene_qsem_disagree_available"].append(
                    scene.get("scene_supp_qsem_disagree_available"))
                entry["scene_qsem_margin_available"].append(
                    scene.get("scene_supp_qsem_margin_available"))
                entry["scene_contextheight_gate_enabled"].append(
                    scene.get("scene_supp_contextheight_gate_enabled"))
                entry["scene_contextheight_gate_ratio"].append(
                    scene.get("scene_supp_contextheight_gate_ratio"))
                entry["scene_contextheight_context_gate_ratio"].append(
                    scene.get("scene_supp_contextheight_context_gate_ratio"))
                entry["scene_contextheight_height_gate_ratio"].append(
                    scene.get("scene_supp_contextheight_height_gate_ratio"))
                entry["scene_contextheight_v9_bev_gate_ratio"].append(
                    scene.get("scene_supp_contextheight_v9_bev_gate_ratio"))
                entry["scene_contextheight_context_score"].append(
                    scene.get("scene_supp_contextheight_context_score"))
                entry["scene_contextheight_height_score"].append(
                    scene.get("scene_supp_contextheight_height_score"))
                entry["scene_contextheight_v9_bev_score"].append(
                    scene.get("scene_supp_contextheight_v9_bev_score"))
                entry["scene_contextheight_context_threshold"].append(
                    scene.get("scene_supp_contextheight_context_threshold"))
                entry["scene_contextheight_height_threshold"].append(
                    scene.get("scene_supp_contextheight_height_threshold"))
                entry["scene_contextheight_v9_bev_threshold"].append(
                    scene.get("scene_supp_contextheight_v9_bev_threshold"))
                entry["scene_contextheight_context_available"].append(
                    scene.get("scene_supp_contextheight_context_available"))
                entry["scene_contextheight_height_available"].append(
                    scene.get("scene_supp_contextheight_height_available"))
                entry["scene_contextheight_v9_bev_available"].append(
                    scene.get("scene_supp_contextheight_v9_bev_available"))
                entry["scene_qsem_top100_count"].append(
                    scene.get("scene_qsem_top100_query_count"))
                entry["scene_qsem_top100_max_score"].append(
                    scene.get("scene_qsem_top100_max_score_mean"))
                entry["scene_qsem_top100_disagree"].append(
                    scene.get("scene_qsem_top100_img_bev_class_disagree_ratio"))
                entry["scene_qsem_top100_margin"].append(
                    scene.get("scene_qsem_top100_img_margin_mean"))
                entry["scene_qsem_top300_count"].append(
                    scene.get("scene_qsem_top300_query_count"))
                entry["scene_qsem_top300_max_score"].append(
                    scene.get("scene_qsem_top300_max_score_mean"))
                entry["scene_qsem_top300_disagree"].append(
                    scene.get("scene_qsem_top300_img_bev_class_disagree_ratio"))
                entry["scene_qsem_top300_margin"].append(
                    scene.get("scene_qsem_top300_img_margin_mean"))
                entry["scene_img_damage_guard_enabled"].append(
                    scene.get("scene_supp_img_damage_guard_enabled"))
                entry["scene_img_damage_guard_mode_code"].append(
                    scene.get("scene_supp_img_damage_guard_mode_code"))
                entry["scene_img_damage_threshold"].append(
                    scene.get("scene_supp_img_damage_threshold"))
                entry["scene_img_damage_threshold_available"].append(
                    scene.get("scene_supp_img_damage_threshold_available"))
                entry["scene_img_damage_threshold_scale"].append(
                    scene.get("scene_supp_img_damage_threshold_scale"))
                entry["scene_bev_damage_threshold_available"].append(
                    scene.get("scene_supp_bev_damage_threshold_available"))
                entry["scene_bev_damage_upper_threshold_available"].append(
                    scene.get("scene_supp_bev_damage_upper_threshold_available"))
                entry["scene_fused_damage_threshold_available"].append(
                    scene.get("scene_supp_fused_damage_threshold_available"))
                entry["scene_bev_fused_reference_available"].append(
                    scene.get("scene_supp_bev_fused_reference_available"))
                entry["scene_fused_damage_threshold_scale"].append(
                    scene.get("scene_supp_fused_damage_threshold_scale"))
                entry["scene_bev_fused_dominance_ratio"].append(
                    scene.get("scene_supp_bev_fused_dominance_ratio"))
                entry["scene_bev_damage_upper_threshold_scale"].append(
                    scene.get("scene_supp_bev_damage_upper_threshold_scale"))
                entry["scene_dual_bev_rank_boost"].append(
                    scene.get("scene_supp_dual_bev_rank_boost"))
                entry["scene_dual_scene_mapper_threshold"].append(
                    scene.get("scene_supp_dual_scene_mapper_threshold"))
                entry["scene_dual_bev_threshold_scale"].append(
                    scene.get("scene_supp_dual_bev_threshold_scale"))
                entry["status"].append(scene.get("scene_supp_status_code"))
                entry["noadd"].append(scene.get("scene_supp_noadd_reason_code"))
        if not pred_path or not os.path.exists(pred_path):
            warnings.append(f"missing prediction sidecar for {cond}: {pred_path}")
        else:
            for pred in iter_jsonl(pred_path):
                entry["preds"] += 1
                for field in (
                        "supp_added", "supp_img_score", "supp_bev_score", "supp_branch_score",
                        "supp_fused_score", "supp_utility_score",
                        "supp_fused_slot_utility_score", "supp_marginal_utility_score",
                        "supp_source_branch"):
                    if field in pred:
                        entry["pred_field_seen"][field] += 1
                if safe_float(pred.get("supp_added")) and safe_float(pred.get("supp_added")) > 0.5:
                    entry["pred_added"] += 1
                    entry["added_img_score"].append(pred.get("supp_img_score"))
                    entry["added_branch_score"].append(
                        pred.get("supp_branch_score", pred.get("supp_img_score")))
                    entry["added_utility_score"].append(pred.get("supp_utility_score"))
                    entry["replaced_fused_score"].append(pred.get("supp_fused_score"))
                    cls = safe_int(pred.get("supp_img_class"))
                    if cls is not None:
                        entry["added_class"][cls] += 1
                    source = safe_int(pred.get("supp_source_branch"))
                    if source is not None:
                        entry["added_source"][source] += 1
        if branch_path and os.path.exists(branch_path):
            summarize_branch_candidates(branch_path, entry)
        elif branch_path:
            warnings.append(f"missing branch-candidate sidecar for {cond}: {branch_path}")
    conditions = {cond: summarize_condition(cond, entry) for cond, entry in sorted(summary.items())}
    return {
        "root": os.path.abspath(root),
        "manifest": manifest,
        "metrics": parse_metrics(os.path.abspath(root)),
        "conditions": conditions,
        "warnings": warnings,
    }


def expand_roots(args):
    roots = []
    for root in args.roots:
        matches = sorted(glob.glob(root))
        roots.extend(matches if matches else [root])
    if args.candidates:
        for name in args.candidates.split():
            roots.append(os.path.join(args.results_dir, f"UMR_{name}_targeted_probe"))
    seen = set()
    out = []
    for root in roots:
        key = os.path.abspath(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def write_markdown(results, path):
    lines = ["# UGSUP EDL Log Diagnostic", ""]
    lines += [
        "## Summary",
        "",
        "| model | condition | NDS | mAP | pred_added | scene_added | gate+ | mapper_gate+ | budget | cand+ | util+ | slots+ | status | noadd | bottleneck |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for result in results:
        model = os.path.basename(result["root"]).replace("_targeted_probe", "")
        metrics = result.get("metrics", {})
        for cond, row in result["conditions"].items():
            metric = metrics.get(cond, {})
            status = ", ".join(f"{STATUS.get(k, k)}:{v}" for k, v in row["status_top"][:3]) or "N/A"
            noadd = ", ".join(f"{NOADD.get(k, k)}:{v}" for k, v in row["noadd_reason_top"][:3]) or "N/A"
            lines.append(
                f"| {model} | {cond} | {fmt(metric.get('NDS'))} | {fmt(metric.get('mAP'))} | "
                f"{fmt(row['pred_added_ratio'])} | {fmt(row['scene_added_ratio'].get('mean'))} | "
                f"{fmt(row['scene_gate'].get('positive_ratio'))} | "
                f"{fmt(row['scene_mapper_gate_ratio'].get('positive_ratio'))} | "
                f"{fmt(row['scene_effective_max_add_queries'].get('mean'))} | "
                f"{fmt(row['scene_candidate_ratio'].get('positive_ratio'))} | "
                f"{fmt(row['scene_utility_candidate_ratio'].get('positive_ratio'))} | "
                f"{fmt(row['scene_replace_slot_ratio'].get('positive_ratio'))} | "
                f"{status} | {noadd} | {row['bottleneck']} |"
            )
    lines.append("")
    lines += [
        "## Telemetry Presence",
        "",
        "| model | condition | scene_enabled | scene_status | scene_gate | mapper_score | mapper_gate | dyn_budget | eff_max_add | scene_candidate | scene_utility | img_damage_score | dual_abstain | img_abstain | dual_abstain_avail | img_abstain_avail | slot_util | margin_on | scene_added | pred_added_field | pred_utility_field | pred_slot_util | pred_margin |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        model = os.path.basename(result["root"]).replace("_targeted_probe", "")
        for cond, row in result["conditions"].items():
            scene_presence = row.get("scene_supp_field_presence", {})
            pred_presence = row.get("pred_supp_field_presence", {})
            lines.append(
                f"| {model} | {cond} | "
                f"{fmt(scene_presence.get('scene_supp_enabled_runtime'))} | "
                f"{fmt(scene_presence.get('scene_supp_status_code'))} | "
                f"{fmt(scene_presence.get('scene_supp_scene_gate_ratio'))} | "
                f"{fmt(scene_presence.get('scene_supp_scene_mapper_score_mean'))} | "
                f"{fmt(scene_presence.get('scene_supp_scene_mapper_gate_ratio'))} | "
                f"{fmt(scene_presence.get('scene_supp_dynamic_budget_ratio'))} | "
                f"{fmt(scene_presence.get('scene_supp_effective_max_add_queries'))} | "
                f"{fmt(scene_presence.get('scene_supp_candidate_ratio'))} | "
                f"{fmt(scene_presence.get('scene_supp_utility_candidate_ratio'))} | "
                f"{fmt(scene_presence.get('scene_supp_img_damage_score'))} | "
                f"{fmt(scene_presence.get('scene_supp_dual_abstain_gate_ratio'))} | "
                f"{fmt(scene_presence.get('scene_supp_img_abstain_gate_ratio'))} | "
                f"{fmt(scene_presence.get('scene_supp_dual_abstain_available'))} | "
                f"{fmt(scene_presence.get('scene_supp_img_abstain_available'))} | "
                f"{fmt(scene_presence.get('scene_supp_fused_slot_utility_score_mean'))} | "
                f"{fmt(scene_presence.get('scene_supp_marginal_enabled'))} | "
                f"{fmt(scene_presence.get('scene_supp_added_ratio'))} | "
                f"{fmt(pred_presence.get('supp_added'))} | "
                f"{fmt(pred_presence.get('supp_utility_score'))} | "
                f"{fmt(pred_presence.get('supp_fused_slot_utility_score'))} | "
                f"{fmt(pred_presence.get('supp_marginal_utility_score'))} |"
            )
    lines.append("")
    lines += [
        "## ACT-M Marginal Utility",
        "",
        "| model | condition | margin_on | base_margin | dual_margin | slot_util | checked | reject | pass | selected_margin | noadd |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        model = os.path.basename(result["root"]).replace("_targeted_probe", "")
        for cond, row in result["conditions"].items():
            noadd = ", ".join(f"{NOADD.get(k, k)}:{v}" for k, v in row["noadd_reason_top"][:3]) or "N/A"
            lines.append(
                f"| {model} | {cond} | "
                f"{fmt(row['scene_marginal_enabled'].get('mean'))} | "
                f"{fmt(row['scene_marginal_margin'].get('mean'))} | "
                f"{fmt(row['scene_dual_marginal_margin'].get('mean'))} | "
                f"{fmt(row['scene_fused_slot_utility_score_mean'].get('mean'))} | "
                f"{fmt(row['scene_marginal_checked_ratio'].get('mean'))} | "
                f"{fmt(row['scene_marginal_reject_ratio'].get('mean'))} | "
                f"{fmt(row['scene_marginal_pass_ratio'].get('mean'))} | "
                f"{fmt(row['scene_selected_margin_score_mean'].get('mean'))} | "
                f"{noadd} |"
            )
    lines.append("")
    lines += [
        "## QSEM Gate",
        "",
        "| model | condition | enabled | qsem_gate | disagree_gate | margin_gate | disagree_score | margin_score | disagree_thr | margin_thr | disagree_avail | margin_avail |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        model = os.path.basename(result["root"]).replace("_targeted_probe", "")
        for cond, row in result["conditions"].items():
            lines.append(
                f"| {model} | {cond} | "
                f"{fmt(row['scene_qsem_gate_enabled'].get('mean'))} | "
                f"{fmt(row['scene_qsem_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_qsem_disagree_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_qsem_margin_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_qsem_disagree_score'].get('mean'))} | "
                f"{fmt(row['scene_qsem_margin_score'].get('mean'))} | "
                f"{fmt(row['scene_qsem_disagree_threshold'].get('mean'))} | "
                f"{fmt(row['scene_qsem_margin_threshold'].get('mean'))} | "
                f"{fmt(row['scene_qsem_disagree_available'].get('mean'))} | "
                f"{fmt(row['scene_qsem_margin_available'].get('mean'))} |"
            )
    lines.append("")
    lines += [
        "## QSEM Top-K Source",
        "",
        "| model | condition | top100_n | top100_max | top100_disagree | top100_margin | top300_n | top300_max | top300_disagree | top300_margin |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        model = os.path.basename(result["root"]).replace("_targeted_probe", "")
        for cond, row in result["conditions"].items():
            lines.append(
                f"| {model} | {cond} | "
                f"{fmt(row['scene_qsem_top100_count'].get('mean'))} | "
                f"{fmt(row['scene_qsem_top100_max_score'].get('mean'))} | "
                f"{fmt(row['scene_qsem_top100_disagree'].get('mean'))} | "
                f"{fmt(row['scene_qsem_top100_margin'].get('mean'))} | "
                f"{fmt(row['scene_qsem_top300_count'].get('mean'))} | "
                f"{fmt(row['scene_qsem_top300_max_score'].get('mean'))} | "
                f"{fmt(row['scene_qsem_top300_disagree'].get('mean'))} | "
                f"{fmt(row['scene_qsem_top300_margin'].get('mean'))} |"
            )
    lines.append("")
    lines += [
        "## Guard Ratios",
        "",
        "| model | condition | guard_enabled | guard_mode | dual_mode | dual_mapper_thr | dual_bev_scale | img_thr_ok | img_thr | img_thr_scale | bev_thr_ok | bev_upper_ok | gate_pre | gate_post | img_damage_gate | img_damage_score | mapper_dual_gate | bev_damage_gate | bev_upper_gate | dual_damage_gate | dual_abstain | img_abstain | dual_abstain_score | img_abstain_score | dual_abstain_thr | img_abstain_thr | bev_rank_boost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        model = os.path.basename(result["root"]).replace("_targeted_probe", "")
        for cond, row in result["conditions"].items():
            lines.append(
                f"| {model} | {cond} | "
                f"{fmt(row['scene_img_damage_guard_enabled'].get('mean'))} | "
                f"{fmt(row['scene_img_damage_guard_mode_code'].get('mean'))} | "
                f"{fmt(row['scene_dual_gate_mode_code'].get('mean'))} | "
                f"{fmt(row['scene_dual_scene_mapper_threshold'].get('mean'))} | "
                f"{fmt(row['scene_dual_bev_threshold_scale'].get('mean'))} | "
                f"{fmt(row['scene_img_damage_threshold_available'].get('mean'))} | "
                f"{fmt(row['scene_img_damage_threshold'].get('mean'))} | "
                f"{fmt(row['scene_img_damage_threshold_scale'].get('mean'))} | "
                f"{fmt(row['scene_bev_damage_threshold_available'].get('mean'))} | "
                f"{fmt(row['scene_bev_damage_upper_threshold_available'].get('mean'))} | "
                f"{fmt(row['scene_gate_pre_guard_ratio'].get('mean'))} | "
                f"{fmt(row['scene_gate_post_guard_ratio'].get('mean'))} | "
                f"{fmt(row['scene_img_damage_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_img_damage_score'].get('mean'))} | "
                f"{fmt(row['scene_scene_mapper_dual_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_bev_damage_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_bev_damage_upper_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_dual_damage_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_dual_abstain_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_img_abstain_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_dual_abstain_score'].get('mean'))} | "
                f"{fmt(row['scene_img_abstain_score'].get('mean'))} | "
                f"{fmt(row['scene_dual_abstain_threshold'].get('mean'))} | "
                f"{fmt(row['scene_img_abstain_threshold'].get('mean'))} | "
                f"{fmt(row['scene_dual_bev_rank_boost'].get('mean'))} |"
            )
    lines.append("")
    lines += [
        "## Relative BEV/Fused Guard",
        "",
        "| model | condition | gate_pre | gate_post | fused_severe | bev>fused | bev_upper | dual_abstain | img_abstain | norm_ratio | fused_scale | dominance | bev_upper_scale | fused_ref_available | dual_available | img_available |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        model = os.path.basename(result["root"]).replace("_targeted_probe", "")
        for cond, row in result["conditions"].items():
            lines.append(
                f"| {model} | {cond} | "
                f"{fmt(row['scene_gate_pre_guard_ratio'].get('mean'))} | "
                f"{fmt(row['scene_gate_post_guard_ratio'].get('mean'))} | "
                f"{fmt(row['scene_fused_damage_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_bev_fused_dominance_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_bev_damage_upper_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_dual_abstain_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_img_abstain_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_bev_fused_normalized_ratio_mean'].get('mean'))} | "
                f"{fmt(row['scene_fused_damage_threshold_scale'].get('mean'))} | "
                f"{fmt(row['scene_bev_fused_dominance_ratio'].get('mean'))} | "
                f"{fmt(row['scene_bev_damage_upper_threshold_scale'].get('mean'))} | "
                f"{fmt(row['scene_bev_fused_reference_available'].get('mean'))} | "
                f"{fmt(row['scene_dual_abstain_available'].get('mean'))} | "
                f"{fmt(row['scene_img_abstain_available'].get('mean'))} |"
            )
    lines.append("")
    lines += [
        "## Context-Height Runtime Gate",
        "",
        "| model | condition | enabled | gate | context< | height< | v9bev< | context_score | height_score | v9bev_score | context_thr | height_thr | v9bev_thr | context_ok | height_ok | v9bev_ok |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        model = os.path.basename(result["root"]).replace("_targeted_probe", "")
        for cond, row in result["conditions"].items():
            lines.append(
                f"| {model} | {cond} | "
                f"{fmt(row['scene_contextheight_gate_enabled'].get('mean'))} | "
                f"{fmt(row['scene_contextheight_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_contextheight_context_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_contextheight_height_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_contextheight_v9_bev_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_contextheight_context_score'].get('mean'))} | "
                f"{fmt(row['scene_contextheight_height_score'].get('mean'))} | "
                f"{fmt(row['scene_contextheight_v9_bev_score'].get('mean'))} | "
                f"{fmt(row['scene_contextheight_context_threshold'].get('mean'))} | "
                f"{fmt(row['scene_contextheight_height_threshold'].get('mean'))} | "
                f"{fmt(row['scene_contextheight_v9_bev_threshold'].get('mean'))} | "
                f"{fmt(row['scene_contextheight_context_available'].get('mean'))} | "
                f"{fmt(row['scene_contextheight_height_available'].get('mean'))} | "
                f"{fmt(row['scene_contextheight_v9_bev_available'].get('mean'))} |"
            )
    lines.append("")
    lines += [
        "## Branch Split",
        "",
        "| model | condition | branch_aware | img_cand | bev_cand | source_bev_queries | selected_img | selected_bev | pred_source_field | added_source_top |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        model = os.path.basename(result["root"]).replace("_targeted_probe", "")
        for cond, row in result["conditions"].items():
            pred_presence = row.get("pred_supp_field_presence", {})
            source_top = ", ".join(f"{k}:{v}" for k, v in row.get("added_source_top", [])) or "N/A"
            lines.append(
                f"| {model} | {cond} | "
                f"{fmt(row['scene_branch_aware_enabled'].get('mean'))} | "
                f"{fmt(row['scene_img_candidate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_bev_candidate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_source_bev_query_ratio'].get('mean'))} | "
                f"{fmt(row['scene_selected_img_ratio'].get('mean'))} | "
                f"{fmt(row['scene_selected_bev_ratio'].get('mean'))} | "
                f"{fmt(pred_presence.get('supp_source_branch'))} | "
                f"{source_top} |"
            )
    lines.append("")
    lines += [
        "## Branch Candidate Sidecar",
        "",
        "| model | condition | rows | samples | valid | fused | bev | img | score_mean | score_p90 | truncated |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        model = os.path.basename(result["root"]).replace("_targeted_probe", "")
        for cond, row in result["conditions"].items():
            bc = row.get("branch_candidate", {})
            branches = bc.get("branch_counts", {})
            score = bc.get("score", {})
            lines.append(
                f"| {model} | {cond} | {bc.get('rows', 0)} | {bc.get('samples', 0)} | "
                f"{bc.get('valid', 0)} | {branches.get('fused', 0)} | "
                f"{branches.get('bev', 0)} | {branches.get('img', 0)} | "
                f"{fmt(score.get('mean'))} | {fmt(score.get('p90'))} | "
                f"{str(bool(bc.get('truncated'))).lower()} |"
            )
    lines.append("")
    lines += [
        "## Detailed Ratios",
        "",
        "| model | condition | gate_mean | mapper_score | mapper_gate | dyn_budget | eff_max_add | candidate_mean | branch_mapper | img_thr | bev_thr | img_util+ | bev_util+ | utility_ok_mean | utility_score_mean | slot_util | margin_on | margin | dual_margin | margin_reject | margin_pass | selected_margin | replace_slot_mean | added_mean | added_img_p90 | added_branch_p90 | added_util_p90 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        model = os.path.basename(result["root"]).replace("_targeted_probe", "")
        for cond, row in result["conditions"].items():
            lines.append(
                f"| {model} | {cond} | {fmt(row['scene_gate'].get('mean'))} | "
                f"{fmt(row['scene_mapper_score_mean'].get('mean'))} | "
                f"{fmt(row['scene_mapper_gate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_dynamic_budget_ratio'].get('mean'))} | "
                f"{fmt(row['scene_effective_max_add_queries'].get('mean'))} | "
                f"{fmt(row['scene_candidate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_branch_mapper_enabled'].get('mean'))} | "
                f"{fmt(row['scene_img_utility_threshold'].get('mean'))} | "
                f"{fmt(row['scene_bev_utility_threshold'].get('mean'))} | "
                f"{fmt(row['scene_img_utility_candidate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_bev_utility_candidate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_utility_candidate_ratio'].get('mean'))} | "
                f"{fmt(row['scene_utility_score_mean'].get('mean'))} | "
                f"{fmt(row['scene_fused_slot_utility_score_mean'].get('mean'))} | "
                f"{fmt(row['scene_marginal_enabled'].get('mean'))} | "
                f"{fmt(row['scene_marginal_margin'].get('mean'))} | "
                f"{fmt(row['scene_dual_marginal_margin'].get('mean'))} | "
                f"{fmt(row['scene_marginal_reject_ratio'].get('mean'))} | "
                f"{fmt(row['scene_marginal_pass_ratio'].get('mean'))} | "
                f"{fmt(row['scene_selected_margin_score_mean'].get('mean'))} | "
                f"{fmt(row['scene_replace_slot_ratio'].get('mean'))} | "
                f"{fmt(row['scene_added_ratio'].get('mean'))} | "
                f"{fmt(row['added_img_score'].get('p90'))} | "
                f"{fmt(row['added_branch_score'].get('p90'))} | "
                f"{fmt(row['added_utility_score'].get('p90'))} |"
            )
    lines.append("")
    lines += [
        "## Interpretation Notes",
        "",
        "- `supp_scene_telemetry_missing` means the sidecar has no supplement runtime fields at all. In practice this is usually a stale sidecar generated before the telemetry/runtime patch, or a config/runtime path that did not enter `damage_gated_soft_router` supplement admission.",
        "- When this appears, regenerate the candidate config and rerun the targeted probe with `EV_LOG_MODE=sidecar`; running this diagnostic alone only reads existing logs.",
        "- Valid UGSUP debugging requires `scene_enabled > 0` first. Only then are `gate+`, `cand+`, `util+`, `slots+`, and `scene_added` meaningful.",
        "",
    ]
    if any(result.get("warnings") for result in results):
        lines += ["## Warnings", ""]
        for result in results:
            model = os.path.basename(result["root"])
            for warning in result.get("warnings", []):
                lines.append(f"- `{model}`: {warning}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="*", help="Targeted probe dirs, manifests, or glob patterns.")
    parser.add_argument("--candidates", default="", help="Candidate names under --results-dir.")
    parser.add_argument("--results-dir", default="MultiCorrupt_results")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    roots = expand_roots(args)
    if not roots:
        raise SystemExit("No roots provided. Use roots/globs or --candidates.")
    results = []
    for root in roots:
        print(f"[UGSUP diag] analyzing: {root}")
        results.append(analyze_root(root))

    output_root = os.path.abspath(args.results_dir)
    os.makedirs(output_root, exist_ok=True)
    output_json = args.output_json or os.path.join(output_root, "ugsup_edl_diagnostic.json")
    output_md = args.output_md or os.path.join(output_root, "ugsup_edl_diagnostic.md")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    write_markdown(results, output_md)
    print(f"[UGSUP diag] wrote JSON: {output_json}")
    print(f"[UGSUP diag] wrote MD:   {output_md}")
    for result in results:
        model = os.path.basename(result["root"]).replace("_targeted_probe", "")
        for cond, row in result["conditions"].items():
            scene_presence = row.get("scene_supp_field_presence", {})
            pred_presence = row.get("pred_supp_field_presence", {})
            branch_candidate = row.get("branch_candidate", {})
            print(
                f"[UGSUP diag] {model} {cond}: "
                f"pred_added={row['pred_added_ratio']:.4f}, "
                f"scene_added={fmt(row['scene_added_ratio'].get('mean'))}, "
                f"gate+={fmt(row['scene_gate'].get('positive_ratio'))}, "
                f"cand+={fmt(row['scene_candidate_ratio'].get('positive_ratio'))}, "
                f"util+={fmt(row['scene_utility_candidate_ratio'].get('positive_ratio'))}, "
                f"slots+={fmt(row['scene_replace_slot_ratio'].get('positive_ratio'))}, "
                f"branch_aware={fmt(row['scene_branch_aware_enabled'].get('mean'))}, "
                f"selected_bev={fmt(row['scene_selected_bev_ratio'].get('mean'))}, "
                f"scene_supp_fields={fmt(scene_presence.get('scene_supp_enabled_runtime'))}, "
                f"pred_supp_fields={fmt(pred_presence.get('supp_added'))}, "
                f"pred_source_fields={fmt(pred_presence.get('supp_source_branch'))}, "
                f"branch_candidate_rows={branch_candidate.get('rows', 0)}, "
                f"bottleneck={row['bottleneck']}"
            )


if __name__ == "__main__":
    main()
