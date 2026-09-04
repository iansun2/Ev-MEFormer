#!/usr/bin/env python3
"""Train lightweight offline UMR utility/risk mappers.

The mapper is intentionally simple and dependency-light. It is a Phase-1
feasibility check over labels produced by ``build_umr_utility_labels.py``.
It should be fit on train/train-O.C only.
"""

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict


DEFAULT_FEATURES = [
    "score",
    "distance_to_ego",
    "center_unc",
    "height_unc",
    "cls_unc",
    "combined_unc",
    # Safe route-derived signals. Do not include raw multi-branch uncertainty
    # fields by default, because a single mapper could then learn implicit
    # cross-branch raw uncertainty comparisons.
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
]

ACTION_BRANCH_FEATURES = [
    "score",
    "distance_to_ego",
    "branch_idx",
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
    "branch_is_fused",
    "branch_is_bev",
    "branch_is_img",
]

TARGET_CHOICES = [
    "candidate_useful",
    "is_tp",
    "cls_correct_near",
    "cls_error_near",
    "loc_fail",
    "near_any_gt",
    "fused_missing_same_gt",
    "fused_weak_same_gt",
    "better_loc_than_fused_same_gt",
    "img_supp_opportunity",
    "bev_supp_opportunity",
    "branch_supp_opportunity",
    "supp_opportunity",
]

FORBIDDEN_INPUT_MARKERS = (
    "multicorrupt_results",
    "targeted_probe",
    "nuscenes_infos_val",
    "corrupted-val",
    "corrupted_val",
    "corrupted/val",
    "val_corrupt",
    "val_clean",
    "validation",
)

UNSAFE_FEATURE_NAMES = {
    "v7_ref_fused_center_unc",
    "v7_ref_bev_center_unc",
    "v7_ref_img_center_unc",
    "v8_a4_cls_unc_z_bev",
    "v8_a4_cls_unc_z_img",
    "v8_gscrg_raw_unc_fused",
    "v8_gscrg_raw_unc_bev",
    "v8_gscrg_raw_unc_img",
}

FORBIDDEN_FEATURE_MARKERS = (
    "corruption",
    "severity",
    "condition",
    "gt",
    "ground_truth",
    "eval",
    "metric_summary",
)


def open_text(path):
    if path.endswith((".gz", ".gzip")):
        return gzip.open(path, "rt", encoding="utf-8")
    if path.endswith((".zst", ".zstd")):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError(
                "Reading zstd labels requires Python package 'zstandard'."
            ) from exc
        raw = open(path, "rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw, closefd=True)
        return io.TextIOWrapper(reader, encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def forbidden_path_markers(path):
    lower = str(path or "").lower().replace("\\", "/")
    markers = [marker for marker in FORBIDDEN_INPUT_MARKERS if marker in lower]
    parts = [part for part in lower.split("/") if part]
    if "val" in parts:
        markers.append("val")
    return markers


def forbidden_feature_markers(feature_name):
    lower = str(feature_name or "").lower()
    return [marker for marker in FORBIDDEN_FEATURE_MARKERS if marker in lower]


def infer_formal_source_kind(path):
    lower = str(path or "").lower().replace("\\", "/")
    parts = [part for part in lower.split("/") if part]
    if any(part in ("train", "train_clean", "train_oc") for part in parts):
        return "train_or_train_oc"
    if any(part.startswith("nuscenes_infos_train") for part in parts):
        return "train_or_train_oc"
    return "unknown"


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


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


def load_records(path, target, feature_names, max_records=0, seed=0, branch_filter=""):
    rng = random.Random(seed)
    rows = []
    counts = Counter()
    source_hash = hashlib.sha256()
    with open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            source_hash.update(line.encode("utf-8"))
            rec = json.loads(line)
            if branch_filter and str(rec.get("branch", "")).lower() != branch_filter:
                counts["skip_branch_filter"] += 1
                continue
            y = rec.get("targets", {}).get(target)
            if y is None:
                counts["skip_no_target"] += 1
                continue
            feats = rec.get("features", {})
            diagnostics = rec.get("diagnostics") or {}
            condition = rec.get("condition", diagnostics.get("condition", "unknown"))
            x = [safe_float(feats.get(name, rec.get(name, 0.0))) for name in feature_names]
            item = {
                "x": x,
                "y": int(y),
                "condition": condition,
            }
            counts["seen"] += 1
            counts[f"condition:{item['condition']}"] += 1
            if max_records and len(rows) >= max_records:
                j = rng.randint(0, counts["seen"] - 1)
                if j < max_records:
                    rows[j] = item
            else:
                rows.append(item)
    counts["loaded"] = len(rows)
    return rows, counts, source_hash.hexdigest()


def build_label_provenance(path, source_sha256, forbidden_markers, args):
    abs_path = os.path.abspath(path)
    real_path = os.path.realpath(path)
    try:
        stat = os.stat(path)
        size_bytes = int(stat.st_size)
        mtime_ns = int(stat.st_mtime_ns)
    except OSError:
        size_bytes = None
        mtime_ns = None
    formal_source_kind = infer_formal_source_kind(path)
    return {
        "labels_path": path,
        "labels_abspath": abs_path,
        "labels_realpath": real_path,
        "labels_size_bytes": size_bytes,
        "labels_mtime_ns": mtime_ns,
        "labels_content_sha256": source_sha256,
        "forbidden_input_markers": list(forbidden_markers),
        "formal_source_kind": formal_source_kind,
        "source_constraint": "train/train-O.C only; val/corrupted-val/eval labels are forbidden",
        "target": args.target,
        "feature_preset": args.feature_preset,
        "branch_filter": args.branch_filter,
        "max_records": args.max_records,
    }


def split_rows(rows, holdout_fraction, seed):
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * holdout_fraction))) if len(shuffled) > 1 else 0
    return shuffled[n_val:], shuffled[:n_val]


def standardize_fit(rows, n_features):
    means = []
    stds = []
    for j in range(n_features):
        vals = [row["x"][j] for row in rows]
        mean = sum(vals) / max(len(vals), 1)
        var = sum((v - mean) ** 2 for v in vals) / max(len(vals), 1)
        std = math.sqrt(var)
        if std < 1e-8:
            std = 1.0
        means.append(mean)
        stds.append(std)
    return means, stds


def standardize_row(x, means, stds):
    return [(v - m) / s for v, m, s in zip(x, means, stds)]


def train_logistic(rows, n_features, epochs, lr, l2, class_balance, means, stds,
                   progress_interval=10, target_name="target"):
    weights = [0.0] * n_features
    bias = 0.0
    pos = sum(row["y"] for row in rows)
    neg = len(rows) - pos
    pos_w = (neg / max(pos, 1)) if class_balance else 1.0
    start_time = time.time()
    for epoch in range(epochs):
        grad_w = [0.0] * n_features
        grad_b = 0.0
        norm = 0.0
        for row in rows:
            x = standardize_row(row["x"], means, stds)
            y = row["y"]
            pred = sigmoid(sum(w * v for w, v in zip(weights, x)) + bias)
            weight = pos_w if y == 1 else 1.0
            err = (pred - y) * weight
            for j, val in enumerate(x):
                grad_w[j] += err * val
            grad_b += err
            norm += weight
        norm = max(norm, 1.0)
        for j in range(n_features):
            grad = grad_w[j] / norm + l2 * weights[j]
            weights[j] -= lr * grad
        bias -= lr * (grad_b / norm)
        if progress_interval and (
            epoch == 0 or epoch + 1 == epochs or (epoch + 1) % progress_interval == 0
        ):
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                f"[UMR mapper] {target_name}: epoch {epoch + 1}/{epochs}, "
                f"elapsed={elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
    return weights, bias


def predict_row(row, weights, bias, means, stds):
    x = standardize_row(row["x"], means, stds)
    return sigmoid(sum(w * v for w, v in zip(weights, x)) + bias)


def auc_score(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = 0.0
    i = 0
    rank = 1
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i)) / 2.0
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        rank += (j - i + 1)
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate(rows, weights, bias, means, stds):
    labels = []
    scores = []
    by_condition = defaultdict(lambda: {"n": 0, "pos": 0, "brier": 0.0})
    for row in rows:
        y = int(row["y"])
        p = predict_row(row, weights, bias, means, stds)
        labels.append(y)
        scores.append(p)
        cond = row.get("condition", "unknown")
        by_condition[cond]["n"] += 1
        by_condition[cond]["pos"] += y
        by_condition[cond]["brier"] += (p - y) ** 2
    n = len(labels)
    brier = sum((p - y) ** 2 for p, y in zip(scores, labels)) / max(n, 1)
    eps = 1e-8
    logloss = -sum(
        y * math.log(max(min(p, 1 - eps), eps)) +
        (1 - y) * math.log(max(min(1 - p, 1 - eps), eps))
        for p, y in zip(scores, labels)
    ) / max(n, 1)
    acc = sum((1 if p >= 0.5 else 0) == y for p, y in zip(scores, labels)) / max(n, 1)
    auc = auc_score(labels, scores)
    condition_summary = {}
    for cond, item in sorted(by_condition.items()):
        condition_summary[cond] = {
            "n": item["n"],
            "positive_rate": item["pos"] / max(item["n"], 1),
            "brier": item["brier"] / max(item["n"], 1),
        }
    return {
        "n": n,
        "positive_rate": sum(labels) / max(n, 1),
        "auc": auc,
        "brier": brier,
        "logloss": logloss,
        "accuracy_at_0_5": acc,
        "by_condition": condition_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="candidate_useful",
                        choices=TARGET_CHOICES)
    parser.add_argument("--feature-preset", default="utility",
                        choices=["utility", "action_branch"],
                        help="Default feature list to use when --features is omitted.")
    parser.add_argument("--features", default="",
                        help="Comma-separated feature list. Overrides --feature-preset.")
    parser.add_argument("--branch-filter", default="", choices=["", "fused", "bev", "img"],
                        help="Optional branch row filter for branch-candidate labels.")
    parser.add_argument("--max-records", type=int, default=500000)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--no-class-balance", action="store_true")
    parser.add_argument("--allow-unsafe-cross-branch-features", action="store_true",
                        help="Debug only: allow raw multi-branch uncertainty features.")
    args = parser.parse_args()

    if args.features.strip():
        feature_names = [name.strip() for name in args.features.split(",") if name.strip()]
    elif args.feature_preset == "action_branch":
        feature_names = list(ACTION_BRANCH_FEATURES)
    else:
        feature_names = list(DEFAULT_FEATURES)
    unsafe_requested = sorted(set(feature_names) & UNSAFE_FEATURE_NAMES)
    forbidden_input_markers = sorted(set(forbidden_path_markers(args.labels)))
    forbidden_feature_hits = {
        name: forbidden_feature_markers(name)
        for name in feature_names
        if forbidden_feature_markers(name)
    }
    if forbidden_input_markers:
        raise SystemExit(
            "Refusing validation/evaluation-derived label path for formal mapper training: "
            f"{forbidden_input_markers}. Build action labels from train/train-O.C only."
        )
    if infer_formal_source_kind(args.labels) != "train_or_train_oc":
        raise SystemExit(
            "Refusing mapper training without an auditable train/train-O.C label path. "
            f"labels={args.labels}"
        )
    if forbidden_feature_hits:
        raise SystemExit(
            "Refusing GT/eval/corruption-derived feature names for formal mapper training: "
            f"{forbidden_feature_hits}."
        )
    if unsafe_requested and not args.allow_unsafe_cross_branch_features:
        raise SystemExit(
            "Refusing unsafe raw cross-branch uncertainty features by default: "
            f"{unsafe_requested}. Use route-derived or final-output features for formal "
            "calibration. For throwaway diagnostics, pass "
            "--allow-unsafe-cross-branch-features and mark the artifact debug-only."
        )
    rows, load_counts, source_sha256 = load_records(
        args.labels, args.target, feature_names, args.max_records, args.seed,
        branch_filter=args.branch_filter)
    if len(rows) < 10:
        raise RuntimeError(f"Too few rows for mapper training: {len(rows)}")
    train_rows, val_rows = split_rows(rows, args.holdout_fraction, args.seed)
    means, stds = standardize_fit(train_rows, len(feature_names))
    weights, bias = train_logistic(
        train_rows, len(feature_names), args.epochs, args.lr, args.l2,
        not args.no_class_balance, means, stds,
        progress_interval=args.progress_interval, target_name=args.target)

    payload = {
        "target": args.target,
        "features": feature_names,
        "normalization": {"mean": means, "std": stds},
        "model": {"type": "logistic_regression", "weights": weights, "bias": bias},
        "load_counts": dict(load_counts),
        "train_metrics": evaluate(train_rows, weights, bias, means, stds),
        "holdout_metrics": evaluate(val_rows, weights, bias, means, stds),
        "notes": [
            "Fit this only on train/train-O.C for thesis-safe calibration.",
            "This mapper may be used by rule-based runtime only when config provenance passes audit.",
            "Default feature set avoids raw multi-branch uncertainty comparison.",
        ],
        "feature_preset": args.feature_preset,
        "branch_filter": args.branch_filter,
        "formal_calibration_safe": bool(not unsafe_requested),
        "debug_only": bool(unsafe_requested),
        "unsafe_features": unsafe_requested,
        "forbidden_input_markers": forbidden_input_markers,
        "forbidden_feature_hits": forbidden_feature_hits,
        "label_provenance": build_label_provenance(
            args.labels, source_sha256, forbidden_input_markers, args),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    summary_md = os.path.splitext(args.output)[0] + ".md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# UMR Utility Mapper\n\n")
        f.write(f"- target: `{args.target}`\n")
        f.write(f"- branch_filter: `{args.branch_filter or 'all'}`\n")
        f.write(f"- rows loaded: `{len(rows)}`\n")
        f.write(f"- features: `{', '.join(feature_names)}`\n\n")
        f.write("## Label Provenance\n\n")
        provenance = payload["label_provenance"]
        f.write(f"- labels: `{provenance['labels_path']}`\n")
        f.write(f"- formal_source_kind: `{provenance['formal_source_kind']}`\n")
        f.write(f"- labels_content_sha256: `{provenance['labels_content_sha256']}`\n\n")
        f.write("## Holdout Metrics\n\n")
        for key, value in payload["holdout_metrics"].items():
            if key == "by_condition":
                continue
            f.write(f"- {key}: `{value}`\n")
        f.write("\n## Condition Summary\n\n")
        f.write("| condition | n | positive_rate | brier |\n")
        f.write("|---|---:|---:|---:|\n")
        for cond, item in payload["holdout_metrics"]["by_condition"].items():
            f.write(
                f"| {cond} | {item['n']} | {item['positive_rate']:.4f} | "
                f"{item['brier']:.4f} |\n"
            )
    print(f"[UMR mapper] wrote: {args.output}")
    print(f"[UMR mapper] wrote: {summary_md}")
    print(f"[UMR mapper] holdout: {payload['holdout_metrics']}")


if __name__ == "__main__":
    main()
