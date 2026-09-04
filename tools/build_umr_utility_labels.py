#!/usr/bin/env python3
"""Build UMR utility labels from EV sidecar predictions and nuScenes train infos.

This is a Phase-1 feasibility tool for branch-wise utility calibration. It does
not use val/corrupted-val data and does not assume raw cross-branch uncertainty
is comparable. It labels final decoded predictions with train GT matching
targets so small offline risk/utility mappers can be tested first.
"""

import argparse
import gzip
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict


CLASS_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


FEATURE_KEYS = [
    "score",
    "distance_to_ego",
    "center_unc",
    "height_unc",
    "cls_unc",
    "combined_unc",
    # Safe route-derived signals. Avoid feeding raw uncertainty from multiple
    # branches into one mapper by default; that would reintroduce raw
    # cross-branch comparison through the learned weights.
    "v8_a4_route_score_bev",
    "v8_a4_route_score_img",
    "v8_bev_reliability",
    "v8_img_reliability",
    "v8_reliability_margin",
    "v9_a_bev",
    "v9_a_img",
    "v9_tail_prob_bev",
    "v9_tail_prob_img",
    "v9_route_score_bev",
    "v9_route_score_img",
    "v10_p_dmg_bev",
    "v10_p_dmg_img",
    "supp_scene_gate_ratio",
    "supp_candidate_ratio",
    "supp_added_ratio",
]

UNSAFE_FEATURE_KEYS = [
    "v7_ref_fused_center_unc",
    "v7_ref_bev_center_unc",
    "v7_ref_img_center_unc",
    "v8_a4_cls_unc_z_bev",
    "v8_a4_cls_unc_z_img",
    "v8_gscrg_raw_unc_fused",
    "v8_gscrg_raw_unc_bev",
    "v8_gscrg_raw_unc_img",
]

FORBIDDEN_INPUT_MARKERS = (
    "nuscenes_infos_val",
    "multicorrupt_results",
    "targeted_probe",
    "corrupted_val",
    "corrupted-val",
    "val_clean",
    "validation",
)
SPLIT_METADATA_KEYS = (
    "ev_split",
    "split",
    "dataset_split",
    "data_split",
    "reference_split",
)
FORBIDDEN_SPLIT_VALUES = {"val", "validation", "corrupted_val", "corrupted-val"}


def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value):
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def open_text(path):
    if path.endswith((".gz", ".gzip")):
        return gzip.open(path, "rt", encoding="utf-8")
    if path.endswith((".zst", ".zstd")):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError(
                "Reading zstd logs requires Python package 'zstandard'."
            ) from exc
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


def find_prediction_logs(path):
    if os.path.isfile(path):
        return [path]
    matches = []
    for root, _dirs, files in os.walk(path):
        for name in files:
            if (
                name.startswith("ev_prediction_light")
                or name.startswith("edl_prediction_light")
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


def find_meta_files(path):
    if os.path.isfile(path):
        root = os.path.dirname(path)
    else:
        root = path
    if not root or not os.path.isdir(root):
        return []
    matches = []
    for current, _dirs, files in os.walk(root):
        for name in files:
            lower = name.lower()
            if (
                lower.endswith(".json")
                and ("meta" in lower or "manifest" in lower or "summary" in lower)
            ):
                matches.append(os.path.join(current, name))
    return sorted(matches)


def metadata_forbidden_markers(value):
    markers = []
    if value is None:
        return markers
    text = str(value).strip().lower()
    if text in FORBIDDEN_SPLIT_VALUES:
        markers.append(f"split={text}")
    tokens = set(re.split(r"[_\-./\s]+", text))
    if "val" in tokens or "validation" in tokens:
        markers.append(f"split_like={text}")
    if "corrupted" in tokens and "val" in tokens:
        markers.append(f"corrupted_val_like={text}")
    return markers


def inspect_json_metadata(path, max_records=16):
    markers = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return markers
    stack = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if str(key) in SPLIT_METADATA_KEYS:
                    markers.extend(metadata_forbidden_markers(value))
                elif isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(item, list):
            stack.extend(item[:max_records])
    return markers


def inspect_prediction_metadata(log_paths, max_records_per_file=32):
    markers = []
    for path in log_paths:
        count = 0
        with open_text(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    pred = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in SPLIT_METADATA_KEYS:
                    if key in pred:
                        markers.extend(metadata_forbidden_markers(pred.get(key)))
                count += 1
                if count >= max_records_per_file:
                    break
    return markers


def assert_train_only_inputs(infos_path, pred_log_path, allow_debug_val=False):
    text = f"{infos_path} {pred_log_path}".lower()
    bad = [marker for marker in FORBIDDEN_INPUT_MARKERS if marker in text]
    tokens = path_tokens(infos_path) | path_tokens(pred_log_path)
    if "val" in tokens:
        bad.append("path_token=val")
    if "corrupted" in tokens and "val" in tokens:
        bad.append("path_token=corrupted-val")
    if bad and not allow_debug_val:
        raise ValueError(
            "Refusing possible val/corrupted-val input for thesis-safe calibration: "
            f"markers={bad}. Use train/train-O.C logs only. For throwaway diagnostics, "
            "rerun with --allow-debug-val and do not use the artifact as formal calibration."
        )
    base = os.path.basename(str(infos_path)).lower()
    if "train" not in base and not allow_debug_val:
        raise ValueError(
            f"Refusing infos file that does not look like train infos: {infos_path}. "
            "Expected nuscenes_infos_train.pkl for formal calibration."
        )
    return bad


def assert_train_only_metadata(pred_log_path, log_paths, allow_debug_val=False):
    bad = []
    for meta_path in find_meta_files(pred_log_path):
        for marker in inspect_json_metadata(meta_path):
            bad.append(f"{os.path.basename(meta_path)}:{marker}")
    for marker in inspect_prediction_metadata(log_paths):
        bad.append(f"prediction:{marker}")
    if bad and not allow_debug_val:
        raise ValueError(
            "Refusing sidecar metadata that looks like val/corrupted-val: "
            f"markers={bad[:12]}. Use train/train-O.C logs only. For throwaway "
            "diagnostics, rerun with --allow-debug-val and do not use the artifact "
            "as formal calibration."
        )
    return bad


def iter_predictions(paths):
    for path in paths:
        with open_text(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pred = json.loads(line)
                yield pred


def print_progress(prefix, count, start_time, interval):
    if not interval or count <= 0 or count % interval != 0:
        return
    elapsed = max(time.time() - start_time, 1e-6)
    rate = count / elapsed
    print(
        f"[UMR utility] {prefix}: {count} records, "
        f"{rate:.1f} rec/s, elapsed={elapsed/60.0:.1f}m",
        file=sys.stderr,
        flush=True,
    )


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
            gts.append({
                "gt_idx": idx,
                "label": int(label),
                "name": CLASS_NAMES[int(label)] if 0 <= int(label) < len(CLASS_NAMES) else str(label),
                "center": [float(box[0]), float(box[1]), float(box[2]) if len(box) > 2 else 0.0],
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


def extract_features(pred, extra_keys):
    out = {}
    for key in FEATURE_KEYS + list(extra_keys):
        if key in pred:
            value = safe_float(pred.get(key))
            if value is not None and math.isfinite(value):
                out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-log", required=True,
                        help="EV/EDL prediction-light JSONL file or directory.")
    parser.add_argument("--infos", default="data/nuscenes/nuscenes_infos_train.pkl",
                        help="nuScenes train infos PKL. Do not point this at val for formal calibration.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--same-class-threshold", type=float, default=2.0,
                        help="XY center-distance threshold for same-class TP utility.")
    parser.add_argument("--near-any-threshold", type=float, default=2.0,
                        help="XY center-distance threshold for class-error diagnosis.")
    parser.add_argument("--loc-fail-threshold", type=float, default=2.0)
    parser.add_argument("--score-min", type=float, default=0.0)
    parser.add_argument("--max-records", type=int, default=0,
                        help="Optional cap for quick smoke tests.")
    parser.add_argument("--progress-interval", type=int, default=100000,
                        help="Print progress every N input predictions. Use 0 to disable.")
    parser.add_argument("--extra-feature", action="append", default=[])
    parser.add_argument("--include-unsafe-cross-branch-features", action="store_true",
                        help="Debug only: include raw multi-branch uncertainty fields.")
    parser.add_argument("--allow-debug-val", action="store_true",
                        help="Debug only: allow val-like inputs and mark outputs as non-formal.")
    args = parser.parse_args()

    try:
        blocked_markers = assert_train_only_inputs(
            args.infos, args.pred_log, allow_debug_val=args.allow_debug_val)
    except ValueError as exc:
        raise SystemExit(f"[UMR utility][ERROR] {exc}") from None
    log_paths = find_prediction_logs(args.pred_log)
    if not log_paths:
        raise FileNotFoundError(f"No prediction-light logs found: {args.pred_log}")
    try:
        metadata_markers = assert_train_only_metadata(
            args.pred_log, log_paths, allow_debug_val=args.allow_debug_val)
    except ValueError as exc:
        raise SystemExit(f"[UMR utility][ERROR] {exc}") from None
    gt_by_token = load_gt_by_token(args.infos, CLASS_TO_ID)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    counts = Counter()
    by_condition = defaultdict(Counter)
    feature_presence = Counter()

    start_time = time.time()
    with open(args.output, "w", encoding="utf-8") as out:
        for pred in iter_predictions(log_paths):
            counts["seen"] += 1
            print_progress("label build", counts["seen"], start_time, args.progress_interval)
            if args.max_records and counts["written"] >= args.max_records:
                break
            score = safe_float(pred.get("score"))
            if score is None or score < args.score_min:
                counts["skip_score"] += 1
                continue
            token = pred.get("sample_token")
            center = pred.get("bbox_center")
            label = safe_int(pred.get("label"))
            if token is None or center is None or len(center) < 2 or label is None:
                counts["skip_missing_prediction_fields"] += 1
                continue
            gts = gt_by_token.get(str(token), [])
            if not gts:
                counts["no_gt"] += 1
                continue

            any_best, same_best = best_match(center, label, gts)
            any_dist = any_best[0] if any_best is not None else None
            same_dist = same_best[0] if same_best is not None else None
            same_gt = same_best[1] if same_best is not None else None
            any_gt = any_best[1] if any_best is not None else None

            is_tp = int(same_dist is not None and same_dist <= args.same_class_threshold)
            near_any = int(any_dist is not None and any_dist <= args.near_any_threshold)
            cls_correct_near = int(
                near_any and any_gt is not None and int(any_gt["label"]) == int(label)
            )
            cls_error_near = int(near_any and not cls_correct_near)
            loc_fail = int(same_dist is None or same_dist > args.loc_fail_threshold)

            condition = pred.get("corruption_type", "clean") or "clean"
            sev = pred.get("corruption_severity")
            if str(condition).lower() in ("", "none"):
                condition = "clean"
            if str(condition).lower() == "clean" and str(sev) in ("0", "0.0", "", "None", "none"):
                condition = "clean"
            elif sev not in (None, "", "None"):
                condition = f"{condition}_s{sev}"

            extra_features = list(args.extra_feature)
            if args.include_unsafe_cross_branch_features:
                extra_features.extend(UNSAFE_FEATURE_KEYS)
            features = extract_features(pred, extra_features)
            for key in features:
                feature_presence[key] += 1

            record = {
                "sample_token": str(token),
                "condition": condition,
                "box_idx": safe_int(pred.get("box_idx")),
                "label": label,
                "label_name": pred.get("label_name"),
                "score": score,
                "bbox_center": [safe_float(center[0]), safe_float(center[1]), safe_float(center[2]) if len(center) > 2 else None],
                "features": features,
                "targets": {
                    "is_tp": is_tp,
                    "near_any_gt": near_any,
                    "cls_correct_near": cls_correct_near,
                    "cls_error_near": cls_error_near,
                    "loc_fail": loc_fail,
                    "candidate_useful": is_tp,
                },
                "match": {
                    "same_class_center_error": same_dist,
                    "nearest_any_center_error": any_dist,
                    "matched_gt_idx": same_gt["gt_idx"] if same_gt else None,
                    "nearest_any_gt_idx": any_gt["gt_idx"] if any_gt else None,
                    "nearest_any_gt_label": any_gt["label"] if any_gt else None,
                },
            }
            out.write(json.dumps(record, separators=(",", ":")) + "\n")
            counts["written"] += 1
            by_condition[condition]["written"] += 1
            by_condition[condition]["is_tp"] += is_tp
            by_condition[condition]["cls_error_near"] += cls_error_near
            by_condition[condition]["loc_fail"] += loc_fail

    summary = {
        "input_logs": log_paths,
        "infos": args.infos,
        "output": args.output,
        "counts": dict(counts),
        "conditions": {k: dict(v) for k, v in sorted(by_condition.items())},
        "feature_presence": dict(feature_presence.most_common()),
        "thresholds": {
            "same_class_threshold": args.same_class_threshold,
            "near_any_threshold": args.near_any_threshold,
            "loc_fail_threshold": args.loc_fail_threshold,
            "score_min": args.score_min,
        },
        "formal_calibration_safe": bool(
            not args.allow_debug_val and
            not args.include_unsafe_cross_branch_features and
            not blocked_markers
        ),
        "debug_only": bool(args.allow_debug_val or args.include_unsafe_cross_branch_features),
        "blocked_input_markers": blocked_markers,
        "blocked_metadata_markers": metadata_markers,
        "unsafe_cross_branch_features_included": bool(args.include_unsafe_cross_branch_features),
        "notes": [
            "Use train/train-O.C labels only for formal calibration.",
            "Default features avoid raw multi-branch uncertainty fields.",
            "These labels are center-distance proxy labels, not official nuScenes AP matching.",
            "This Phase-1 tool labels final decoded predictions, not all raw branch candidates.",
        ],
    }

    summary_path = args.summary or os.path.splitext(args.output)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[UMR utility] wrote labels: {args.output}")
    print(f"[UMR utility] wrote summary: {summary_path}")
    print(f"[UMR utility] written={counts['written']} seen={counts['seen']}")


if __name__ == "__main__":
    main()
