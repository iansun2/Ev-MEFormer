#!/usr/bin/env python3
"""Build train-only utility labels for bounded branch-candidate sidecars.

This tool labels fused/BEV/IMG candidate rows against nuScenes train GT using
center-distance proxy matching. It is an offline feasibility tool for
rule-based UMR v2 design; runtime policies must not read corruption labels,
severity, GT, or these output labels.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext


CLASS_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

FORBIDDEN_INPUT_MARKERS = (
    "nuscenes_infos_val",
    "multicorrupt_results",
    "targeted_probe",
    "corrupted/val",
    "corrupted_val",
    "corrupted-val",
    "val_corrupt",
    "val_clean",
    "validation",
)
ALLOWED_SPLITS = {"train_clean", "train_oc", "train", "clean"}
DIAGNOSTIC_ONLY_KEYS = (
    "corruption_type",
    "corruption_severity",
    "ev_split",
    "debug_raw_cls_unc",
    "debug_raw_center_unc",
    "debug_raw_height_unc",
    "debug_raw_combined_unc",
)
SAFE_FEATURE_KEYS = (
    "score",
    "distance_to_ego",
    "branch_idx",
    "candidate_rank",
    "branch_rank",
    "cls_weight",
    "center_weight",
    "height_weight",
    "non_fused_prob",
    "fused_keep",
    "damage_gate",
    "bev_damage",
    "img_damage",
    "dual_damage",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(value):
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def safe_int(value):
    try:
        if value is None:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def open_text(path):
    if path.endswith((".gz", ".gzip")):
        return gzip.open(path, "rt", encoding="utf-8")
    if path.endswith((".zst", ".zstd")):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("Reading zstd logs requires Python package 'zstandard'.") from exc
        raw = open(path, "rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw, closefd=True)
        import io
        return io.TextIOWrapper(reader, encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def open_output(path):
    if not path:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith((".gz", ".gzip")):
        return gzip.open(path, "wt", encoding="utf-8")
    return open(path, "w", encoding="utf-8")


def path_has_forbidden_marker(path):
    lower = str(path or "").lower().replace("\\", "/")
    markers = [m for m in FORBIDDEN_INPUT_MARKERS if m in lower]
    parts = [part for part in lower.split("/") if part]
    if "val" in parts:
        markers.append("val")
    return markers


def assert_formal_inputs(infos, log_root, allow_debug_val=False):
    markers = []
    for path in (infos, log_root):
        markers.extend(path_has_forbidden_marker(path))
    if markers and not allow_debug_val:
        raise SystemExit(
            "[BranchLabel][ERROR] refusing validation/evaluation-derived input: "
            f"{sorted(set(markers))}. Use --allow-debug-val only for throwaway diagnostics.")
    return sorted(set(markers))


def validate_row_metadata(row, expected_split="", allow_debug_val=False):
    markers = []
    for key in ("ev_split", "condition", "source_split"):
        markers.extend(path_has_forbidden_marker(row.get(key)))
    split = str(row.get("ev_split", "") or "").strip().lower()
    if split and split not in ALLOWED_SPLITS:
        markers.append(f"ev_split={split}")
    if expected_split:
        expected = str(expected_split).strip().lower()
        expected_aliases = {
            "train_clean": {"train_clean", "train", "clean"},
            "train_oc": {"train_oc"},
            "train": {"train", "train_clean", "clean"},
            "clean": {"clean", "train_clean", "train"},
        }.get(expected, {expected})
        if not split or split not in expected_aliases:
            markers.append(f"expected_split={expected},row_ev_split={split or '<missing>'}")
    if markers and not allow_debug_val:
        raise SystemExit(
            "[BranchLabel][ERROR] refusing non-train or mismatched branch-candidate row: "
            f"{sorted(set(markers))}. Use --allow-debug-val only for throwaway diagnostics.")
    return sorted(set(markers))


def find_branch_logs(path):
    if os.path.isfile(path):
        return [path]
    matches = []
    for root, _dirs, files in os.walk(path):
        for name in files:
            if name.startswith("ev_branch_candidates") and (
                    name.endswith(".jsonl")
                    or name.endswith(".jsonl.gz")
                    or name.endswith(".jsonl.zst")):
                matches.append(os.path.join(root, name))
    return sorted(matches)


def to_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def load_gt_by_token(infos_path, class_to_id):
    with open(infos_path, "rb") as f:
        payload = pickle.load(f)
    infos = payload.get("infos", payload) if isinstance(payload, dict) else payload
    gt_by_token = {}
    for info in infos:
        token = info.get("token") or info.get("sample_token")
        if token is None:
            continue
        boxes = to_list(info.get("gt_boxes", []))
        names = to_list(info.get("gt_names", []))
        labels = to_list(info.get("gt_labels_3d", []))
        valid = to_list(info.get("valid_flag", []))
        gts = []
        for idx, box in enumerate(boxes):
            if valid and idx < len(valid) and not bool(valid[idx]):
                continue
            if box is None or len(box) < 2:
                continue
            if labels and idx < len(labels):
                label = safe_int(labels[idx])
            else:
                name = str(names[idx]) if idx < len(names) else ""
                label = class_to_id.get(name)
            if label is None:
                continue
            label = int(label)
            gts.append({
                "gt_idx": idx,
                "label": label,
                "name": CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else str(label),
                "center": [
                    float(box[0]),
                    float(box[1]),
                    float(box[2]) if len(box) > 2 else 0.0,
                ],
            })
        gt_by_token[str(token)] = gts
    return gt_by_token


def center_dist_xy(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def best_match(center, pred_label, gts):
    any_best = None
    same_best = None
    for gt in gts:
        dist = center_dist_xy(center, gt["center"])
        item = (dist, gt)
        if any_best is None or dist < any_best[0]:
            any_best = item
        if int(gt["label"]) == int(pred_label):
            if same_best is None or dist < same_best[0]:
                same_best = item
    return any_best, same_best


def condition_name(row):
    split = str(row.get("ev_split", "") or "").strip().lower()
    corr = str(row.get("corruption_type", "clean") or "clean").lower()
    sev = row.get("corruption_severity", 0)
    try:
        sev_int = int(float(sev))
    except Exception:
        sev_int = 0
    if corr in ("", "none", "clean") or sev_int <= 0:
        if split and split not in ("clean", "none"):
            return split
        return "clean"
    return f"{corr}_s{sev_int}"


def iter_sample_groups(paths, progress_interval=1000000, max_records=0,
                       expected_split="", allow_debug_val=False):
    seen = 0
    start = time.time()
    for path in paths:
        current_token = None
        group = []
        with open_text(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                validate_row_metadata(
                    row, expected_split=expected_split, allow_debug_val=allow_debug_val)
                seen += 1
                if progress_interval and seen % progress_interval == 0:
                    elapsed = max(time.time() - start, 1e-6)
                    print(
                        f"[BranchLabel] read={seen} rows, "
                        f"{seen / elapsed:.1f} row/s, elapsed={elapsed / 60.0:.1f}m",
                        file=sys.stderr,
                        flush=True)
                token = str(row.get("sample_token", ""))
                if current_token is None:
                    current_token = token
                if token != current_token:
                    yield current_token, group
                    group = []
                    current_token = token
                group.append(row)
                if max_records and seen >= max_records:
                    if group:
                        yield current_token, group
                    return
        if group:
            yield current_token, group


def update_counter(counter, value):
    counter["n"] += 1
    counter["sum"] += float(value)


def counter_mean(counter):
    return counter["sum"] / max(counter["n"], 1)


def add_branch_stats(summary, condition, branch, label):
    item = summary["by_condition"][condition]["branches"][branch]
    item["rows"] += 1
    for key in (
            "candidate_useful",
            "near_any_gt",
            "cls_correct_near",
            "cls_error_near",
            "loc_fail",
            "fused_missing_same_gt",
            "fused_weak_same_gt",
            "better_loc_than_fused_same_gt",
            "img_supp_opportunity",
            "bev_supp_opportunity",
            "branch_supp_opportunity",
            "supp_opportunity"):
        item[key] += int(label["targets"].get(key, 0))
    for key in ("score", "same_class_center_error", "nearest_any_center_error"):
        value = label["match"].get(key) if key.endswith("error") else label.get(key)
        if value is not None:
            mean_key = f"{key}_mean"
            if not isinstance(item.get(mean_key), dict):
                item[mean_key] = {"n": 0, "sum": 0.0}
            update_counter(item[f"{key}_mean"], value)


def summarize_gt_opportunities(summary, condition, gt_best):
    cond = summary["by_condition"][condition]
    for _gt_idx, per_gt in gt_best.items():
        cond["gt_with_candidates"] += 1
        cond["gt_with_useful_candidates"] += 1
        best = min(
            (item for item in per_gt.values() if item is not None),
            key=lambda x: (x["dist"], -x["score"]),
            default=None)
        if best is None:
            continue
        cond["gt_best_branch"][best["branch"]] += 1
        fused = per_gt.get("fused")
        img = per_gt.get("img")
        bev = per_gt.get("bev")
        fused_weak = fused is None or fused["score"] <= cond["fused_replace_score_ceiling"]
        if img is not None and fused_weak:
            cond["img_supp_gt_opportunity"] += 1
        if bev is not None and fused_weak:
            cond["bev_supp_gt_opportunity"] += 1
        if img is not None and bev is not None and fused_weak:
            cond["dual_branch_gt_opportunity"] += 1


def label_group(rows, gts, args):
    gt_best = defaultdict(lambda: {"fused": None, "bev": None, "img": None})
    preliminary = []
    for row in rows:
        score = safe_float(row.get("score"))
        label = safe_int(row.get("label"))
        center = row.get("bbox_center")
        valid = bool(row.get("valid", True))
        if score is None or label is None or center is None or len(center) < 2 or not valid:
            continue
        any_best, same_best = best_match(center, label, gts)
        any_dist = any_best[0] if any_best is not None else None
        same_dist = same_best[0] if same_best is not None else None
        same_gt = same_best[1] if same_best is not None else None
        any_gt = any_best[1] if any_best is not None else None
        candidate_useful = int(same_dist is not None and same_dist <= args.same_class_threshold)
        near_any = int(any_dist is not None and any_dist <= args.near_any_threshold)
        cls_correct_near = int(
            near_any and any_gt is not None and int(any_gt["label"]) == int(label))
        cls_error_near = int(near_any and not cls_correct_near)
        loc_fail = int(same_dist is None or same_dist > args.loc_fail_threshold)
        branch = str(row.get("branch", "unknown"))
        matched_gt_idx = same_gt["gt_idx"] if same_gt else None
        if candidate_useful and matched_gt_idx is not None and branch in ("fused", "bev", "img"):
            slot = gt_best[matched_gt_idx][branch]
            candidate = {"branch": branch, "dist": same_dist, "score": score, "row": row}
            if slot is None or (same_dist, -score) < (slot["dist"], -slot["score"]):
                gt_best[matched_gt_idx][branch] = candidate
        preliminary.append({
            "row": row,
            "branch": branch,
            "score": score,
            "label": label,
            "condition": condition_name(row),
            "same_dist": same_dist,
            "any_dist": any_dist,
            "same_gt": same_gt,
            "any_gt": any_gt,
            "targets": {
                "candidate_useful": candidate_useful,
                "near_any_gt": near_any,
                "cls_correct_near": cls_correct_near,
                "cls_error_near": cls_error_near,
                "loc_fail": loc_fail,
            },
        })
    for item in preliminary:
        same_gt = item["same_gt"]
        fused_best = None
        if same_gt is not None:
            fused_best = gt_best.get(same_gt["gt_idx"], {}).get("fused")
        matched_useful_gt = bool(item["targets"]["candidate_useful"] and same_gt is not None)
        fused_missing = int(matched_useful_gt and fused_best is None)
        fused_weak = int(
            matched_useful_gt and
            (fused_best is None or fused_best["score"] <= args.fused_replace_score_ceiling))
        better_loc = int(
            item["targets"]["candidate_useful"] and
            fused_best is not None and
            item["same_dist"] + args.better_loc_margin < fused_best["dist"])
        img_supp_opportunity = int(
            item["branch"] == "img" and
            item["targets"]["candidate_useful"] and
            (fused_missing or fused_weak))
        bev_supp_opportunity = int(
            item["branch"] == "bev" and
            item["targets"]["candidate_useful"] and
            (fused_missing or fused_weak))
        branch_supp_opportunity = int(img_supp_opportunity or bev_supp_opportunity)
        item["targets"].update({
            "is_tp": item["targets"]["candidate_useful"],
            "fused_missing_same_gt": fused_missing,
            "fused_weak_same_gt": fused_weak,
            "better_loc_than_fused_same_gt": better_loc,
            "img_supp_opportunity": img_supp_opportunity,
            "bev_supp_opportunity": bev_supp_opportunity,
            "branch_supp_opportunity": branch_supp_opportunity,
            "supp_opportunity": branch_supp_opportunity,
        })
        item["fused_best"] = fused_best
    return preliminary, gt_best


def compact_label_record(item):
    row = item["row"]
    fused_best = item.get("fused_best")
    safe_features = {}
    for key in SAFE_FEATURE_KEYS:
        value = row.get(key)
        if key == "branch_idx":
            value = safe_int(value)
        else:
            value = safe_float(value)
        if value is not None:
            safe_features[key] = value
    branch = item["branch"]
    safe_features["branch_is_fused"] = 1.0 if branch == "fused" else 0.0
    safe_features["branch_is_bev"] = 1.0 if branch == "bev" else 0.0
    safe_features["branch_is_img"] = 1.0 if branch == "img" else 0.0
    diagnostics = {
        key: row.get(key)
        for key in DIAGNOSTIC_ONLY_KEYS
        if key in row
    }
    diagnostics["condition"] = item["condition"]
    for key in (
            "damage_state",
            "center_source",
            "admitted_by_current_policy",
            "action_label",
            "action_space",
            "selected_for_cls",
            "selected_for_center",
            "selected_for_height"):
        if key in row:
            diagnostics[key] = row.get(key)
    candidate = {
        "branch": branch,
        "branch_idx": safe_int(row.get("branch_idx")),
        "candidate_idx": safe_int(row.get("candidate_idx")),
        "candidate_rank": safe_int(row.get("candidate_rank")),
        "logging_rank": safe_int(row.get("logging_rank")),
        "branch_rank": safe_int(row.get("branch_rank")),
        "candidate_uid": safe_int(row.get("candidate_uid")),
        "task_id": safe_int(row.get("task_id")),
        "query_idx": safe_int(row.get("query_idx")),
        "local_label": safe_int(row.get("local_label")),
        "global_label": safe_int(row.get("global_label")),
        "label": item["label"],
        "label_name": row.get("label_name"),
        "score": item["score"],
        "bbox_center": row.get("bbox_center"),
        "distance_to_ego": safe_float(row.get("distance_to_ego")),
    }
    return {
        "sample_token": str(row.get("sample_token")),
        "condition": item["condition"],
        "branch": branch,
        "candidate": candidate,
        "candidate_uid": candidate["candidate_uid"],
        "query_idx": candidate["query_idx"],
        "label": item["label"],
        "label_name": row.get("label_name"),
        "score": item["score"],
        "distance_to_ego": safe_float(row.get("distance_to_ego")),
        "features": safe_features,
        "diagnostics": diagnostics,
        "targets": item["targets"],
        "match": {
            "same_class_center_error": item["same_dist"],
            "nearest_any_center_error": item["any_dist"],
            "matched_gt_idx": item["same_gt"]["gt_idx"] if item["same_gt"] else None,
            "nearest_any_gt_idx": item["any_gt"]["gt_idx"] if item["any_gt"] else None,
            "nearest_any_gt_label": item["any_gt"]["label"] if item["any_gt"] else None,
            "fused_best_same_error": fused_best["dist"] if fused_best else None,
            "fused_best_same_score": fused_best["score"] if fused_best else None,
        },
    }


def new_condition_bucket(args):
    return {
        "samples": 0,
        "rows": 0,
        "rows_labeled": 0,
        "gt_total": 0,
        "gt_with_candidates": 0,
        "gt_with_useful_candidates": 0,
        "img_supp_gt_opportunity": 0,
        "bev_supp_gt_opportunity": 0,
        "dual_branch_gt_opportunity": 0,
        "fused_replace_score_ceiling": float(args.fused_replace_score_ceiling),
        "gt_best_branch": Counter(),
        "branches": defaultdict(lambda: defaultdict(int)),
    }


def materialize_summary(summary):
    out = dict(summary)
    out["by_condition"] = {}
    for cond, bucket in sorted(summary["by_condition"].items()):
        branches = {}
        for branch, item in sorted(bucket["branches"].items()):
            rows = max(item.get("rows", 0), 1)
            branches[branch] = {}
            for key, value in sorted(item.items()):
                if key.endswith("_mean") and isinstance(value, dict):
                    branches[branch][key] = counter_mean(value)
                elif isinstance(value, int):
                    branches[branch][key] = value
                    if key not in ("rows",):
                        branches[branch][f"{key}_rate"] = value / rows
                else:
                    branches[branch][key] = value
        cond_out = {
            key: value
            for key, value in bucket.items()
            if key not in ("branches", "gt_best_branch")
        }
        cond_out["gt_best_branch"] = dict(bucket["gt_best_branch"])
        total_gt = max(cond_out.get("gt_with_useful_candidates", cond_out.get("gt_with_candidates", 0)), 1)
        cond_out["img_supp_gt_opportunity_rate"] = cond_out.get("img_supp_gt_opportunity", 0) / total_gt
        cond_out["bev_supp_gt_opportunity_rate"] = cond_out.get("bev_supp_gt_opportunity", 0) / total_gt
        cond_out["dual_branch_gt_opportunity_rate"] = cond_out.get("dual_branch_gt_opportunity", 0) / total_gt
        cond_out["branches"] = branches
        out["by_condition"][cond] = cond_out
    return out


def fmt(value):
    return "N/A" if value is None else f"{float(value):.4f}"


def write_markdown(summary, path):
    lines = ["# UMR Branch-Candidate Label Summary", ""]
    lines += [
        "## Condition Opportunities",
        "",
        "| condition | samples | labeled rows | gt total | gt useful cand. | best fused | best bev | best img | img supp gt | bev supp gt | dual gt |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cond, item in sorted(summary["by_condition"].items()):
        best = item.get("gt_best_branch", {})
        lines.append(
            f"| {cond} | {item.get('samples', 0)} | {item.get('rows_labeled', 0)} | "
            f"{item.get('gt_total', 0)} | {item.get('gt_with_useful_candidates', item.get('gt_with_candidates', 0))} | "
            f"{best.get('fused', 0)} | "
            f"{best.get('bev', 0)} | {best.get('img', 0)} | "
            f"{item.get('img_supp_gt_opportunity', 0)} ({fmt(item.get('img_supp_gt_opportunity_rate'))}) | "
            f"{item.get('bev_supp_gt_opportunity', 0)} ({fmt(item.get('bev_supp_gt_opportunity_rate'))}) | "
            f"{item.get('dual_branch_gt_opportunity', 0)} ({fmt(item.get('dual_branch_gt_opportunity_rate'))}) |"
        )
    lines += ["", "## Branch Utility Rates", ""]
    lines.append("| condition | branch | rows | useful | cls_error_near | loc_fail | img_supp | bev_supp | supp_opp | score_mean | same_err_mean |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cond, item in sorted(summary["by_condition"].items()):
        for branch, b in sorted(item.get("branches", {}).items()):
            lines.append(
                f"| {cond} | {branch} | {b.get('rows', 0)} | "
                f"{fmt(b.get('candidate_useful_rate'))} | "
                f"{fmt(b.get('cls_error_near_rate'))} | "
                f"{fmt(b.get('loc_fail_rate'))} | "
                f"{fmt(b.get('img_supp_opportunity_rate'))} | "
                f"{fmt(b.get('bev_supp_opportunity_rate'))} | "
                f"{fmt(b.get('supp_opportunity_rate'))} | "
                f"{fmt(b.get('score_mean'))} | "
                f"{fmt(b.get('same_class_center_error_mean'))} |"
            )
    lines += [
        "",
        "## Formal Notes",
        "",
        "- Labels are produced only from train/train-O.C logs and train infos.",
        "- Corruption/severity fields are diagnostic grouping metadata only; do not feed them to runtime mappers.",
        "- Matching is a center-distance proxy, not official nuScenes AP matching.",
        "- Raw `debug_raw_*_unc` values are diagnostic-only and are not emitted as formal features by default.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-log-root", required=True)
    parser.add_argument("--infos", default="data/nuscenes/nuscenes_infos_train.pkl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label-output", default="",
                        help="Optional JSONL/GZ row-label output. Omit for summary-only analysis.")
    parser.add_argument("--same-class-threshold", type=float, default=2.0)
    parser.add_argument("--near-any-threshold", type=float, default=2.0)
    parser.add_argument("--loc-fail-threshold", type=float, default=2.0)
    parser.add_argument("--fused-replace-score-ceiling", type=float, default=0.35)
    parser.add_argument("--better-loc-margin", type=float, default=0.10)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=1000000)
    parser.add_argument("--expected-split", default="",
                        help="Fail closed unless row ev_split matches this train split.")
    parser.add_argument("--allow-debug-val", action="store_true")
    args = parser.parse_args()

    blocked_markers = assert_formal_inputs(
        args.infos, args.branch_log_root, allow_debug_val=args.allow_debug_val)
    paths = find_branch_logs(args.branch_log_root)
    if not paths:
        raise FileNotFoundError(f"No branch-candidate logs found: {args.branch_log_root}")
    gt_by_token = load_gt_by_token(args.infos, CLASS_TO_ID)
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {
        "branch_log_root": args.branch_log_root,
        "infos": args.infos,
        "input_logs": paths,
        "input_log_sha256": {path: sha256_file(path) for path in paths},
        "counts": Counter(),
        "feature_presence": Counter(),
        "diagnostic_presence": Counter(),
        "by_condition": defaultdict(lambda: new_condition_bucket(args)),
        "thresholds": {
            "same_class_threshold": args.same_class_threshold,
            "near_any_threshold": args.near_any_threshold,
            "loc_fail_threshold": args.loc_fail_threshold,
            "fused_replace_score_ceiling": args.fused_replace_score_ceiling,
            "better_loc_margin": args.better_loc_margin,
        },
        "formal_calibration_safe": bool(not args.allow_debug_val and not blocked_markers),
        "debug_only": bool(args.allow_debug_val),
        "blocked_input_markers": blocked_markers,
        "expected_split": args.expected_split,
        "notes": [
            "Train/train-O.C branch-candidate labels only.",
            "Condition names are diagnostic grouping only, not runtime inputs.",
            "Branch-local raw uncertainties remain diagnostic-only by default.",
        ],
    }

    label_path = args.label_output
    if label_path and not os.path.isabs(label_path):
        label_path = os.path.join(args.out_dir, label_path)
    label_context = open_output(label_path) if label_path else nullcontext(None)
    label_content_digest = hashlib.sha256()
    with label_context as label_fp:
        for token, rows in iter_sample_groups(
                paths,
                progress_interval=args.progress_interval,
                max_records=args.max_records,
                expected_split=args.expected_split,
                allow_debug_val=args.allow_debug_val):
            summary["counts"]["samples_seen"] += 1
            summary["counts"]["rows_seen"] += len(rows)
            gts = gt_by_token.get(str(token), [])
            if not gts:
                summary["counts"]["samples_no_gt"] += 1
                continue
            labels, gt_best = label_group(rows, gts, args)
            condition = condition_name(rows[0]) if rows else "unknown"
            bucket = summary["by_condition"][condition]
            bucket["samples"] += 1
            bucket["rows"] += len(rows)
            bucket["rows_labeled"] += len(labels)
            bucket["gt_total"] += len(gts)
            summarize_gt_opportunities(summary, condition, gt_best)
            for item in labels:
                record = compact_label_record(item)
                add_branch_stats(summary, item["condition"], item["branch"], record)
                for key in record.get("features", {}):
                    summary["feature_presence"][key] += 1
                for key in record.get("diagnostics", {}):
                    summary["diagnostic_presence"][key] += 1
                if label_fp is not None:
                    label_line = json.dumps(record, separators=(",", ":")) + "\n"
                    label_fp.write(label_line)
                    label_content_digest.update(label_line.encode("utf-8"))
                    summary["counts"]["labels_written"] += 1

    final_summary = materialize_summary(summary)
    if label_path:
        final_summary["output_labels"] = label_path
        final_summary["output_labels_sha256"] = sha256_file(label_path)
        final_summary["output_labels_content_sha256"] = (
            label_content_digest.hexdigest())
        final_summary["output_label_records"] = int(
            summary["counts"]["labels_written"])
    summary_json = os.path.join(args.out_dir, "branch_candidate_labels_summary.json")
    summary_md = os.path.join(args.out_dir, "branch_candidate_labels_summary.md")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, sort_keys=True)
    write_markdown(final_summary, summary_md)
    print(f"[BranchLabel] wrote summary JSON: {summary_json}")
    print(f"[BranchLabel] wrote summary MD:   {summary_md}")
    if label_path:
        print(f"[BranchLabel] wrote row labels:   {label_path}")
    print(f"[BranchLabel] samples={summary['counts']['samples_seen']} rows={summary['counts']['rows_seen']}")


if __name__ == "__main__":
    main()
