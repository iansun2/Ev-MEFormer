#!/usr/bin/env python3
"""Train a lightweight scene damage applicability mapper for UMR Phase 1."""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

from build_umr_scene_gate_labels import DEFAULT_SCENE_FEATURES, UNSAFE_SCENE_FEATURES


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


def load_records(path, target, feature_names, max_records=0, seed=0):
    rng = random.Random(seed)
    rows = []
    counts = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            y = rec.get("targets", {}).get(target)
            if y is None:
                counts["skip_no_target"] += 1
                continue
            weight = rec.get("weights", {}).get(target, 1.0)
            feats = rec.get("features", {})
            item = {
                "x": [safe_float(feats.get(name, rec.get(name, 0.0))) for name in feature_names],
                "y": min(max(safe_float(y), 0.0), 1.0),
                "weight": max(safe_float(weight, 1.0), 0.0),
                "condition": rec.get("condition", "unknown"),
                "policy": rec.get("label_policy", "unknown"),
            }
            if item["weight"] <= 0.0:
                counts["skip_zero_weight"] += 1
                continue
            counts["seen"] += 1
            if max_records and len(rows) >= max_records:
                j = rng.randint(0, counts["seen"] - 1)
                if j < max_records:
                    rows[j] = item
            else:
                rows.append(item)
    counts["loaded"] = len(rows)
    return rows, counts


def load_label_summary(path):
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def infer_label_summary_path(labels_path):
    root, _ext = os.path.splitext(labels_path)
    candidate = root + "_summary.json"
    return candidate if os.path.isfile(candidate) else ""


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


def train_logistic(rows, n_features, epochs, lr, l2, means, stds,
                   class_balance=True, progress_interval=10):
    weights = [0.0] * n_features
    bias = 0.0
    pos_mass = sum(row["weight"] * row["y"] for row in rows)
    neg_mass = sum(row["weight"] * (1.0 - row["y"]) for row in rows)
    pos_scale = (neg_mass / max(pos_mass, 1e-6)) if class_balance else 1.0
    start = time.time()
    for epoch in range(epochs):
        grad_w = [0.0] * n_features
        grad_b = 0.0
        norm = 0.0
        for row in rows:
            x = standardize_row(row["x"], means, stds)
            y = row["y"]
            p = sigmoid(sum(w * v for w, v in zip(weights, x)) + bias)
            sample_weight = row["weight"] * (pos_scale if y >= 0.5 else 1.0)
            err = (p - y) * sample_weight
            for j, val in enumerate(x):
                grad_w[j] += err * val
            grad_b += err
            norm += sample_weight
        norm = max(norm, 1.0)
        for j in range(n_features):
            weights[j] -= lr * (grad_w[j] / norm + l2 * weights[j])
        bias -= lr * (grad_b / norm)
        if progress_interval and (
            epoch == 0 or epoch + 1 == epochs or (epoch + 1) % progress_interval == 0
        ):
            elapsed = max(time.time() - start, 1e-6)
            print(
                f"[UMR scene mapper] epoch {epoch + 1}/{epochs}, elapsed={elapsed:.1f}s",
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


def percentile(values, q):
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    idx = min(max(int(round((len(vals) - 1) * q)), 0), len(vals) - 1)
    return vals[idx]


def evaluate(rows, weights, bias, means, stds, thresholds=None):
    thresholds = thresholds or {}
    labels = []
    scores = []
    by_condition = defaultdict(lambda: {"n": 0, "target_sum": 0.0, "weight_sum": 0.0, "score_sum": 0.0, "brier": 0.0})
    by_policy = defaultdict(lambda: {"n": 0, "score_sum": 0.0, "target_sum": 0.0})
    for row in rows:
        y = row["y"]
        p = predict_row(row, weights, bias, means, stds)
        labels.append(1 if y >= 0.5 else 0)
        scores.append(p)
        cond = row.get("condition", "unknown")
        by_condition[cond]["n"] += 1
        by_condition[cond]["target_sum"] += y
        by_condition[cond]["weight_sum"] += row["weight"]
        by_condition[cond]["score_sum"] += p
        by_condition[cond]["brier"] += (p - y) ** 2
        policy = row.get("policy", "unknown")
        by_policy[policy]["n"] += 1
        by_policy[policy]["score_sum"] += p
        by_policy[policy]["target_sum"] += y

    n = len(rows)
    brier = sum((p - row["y"]) ** 2 for p, row in zip(scores, rows)) / max(n, 1)
    eps = 1e-8
    logloss = -sum(
        row["y"] * math.log(max(min(p, 1 - eps), eps)) +
        (1 - row["y"]) * math.log(max(min(1 - p, 1 - eps), eps))
        for p, row in zip(scores, rows)
    ) / max(n, 1)
    cond_summary = {}
    for cond, item in sorted(by_condition.items()):
        cond_scores = [predict_row(row, weights, bias, means, stds) for row in rows if row.get("condition") == cond]
        trigger = {
            name: sum(1 for s in cond_scores if s >= thr) / max(len(cond_scores), 1)
            for name, thr in thresholds.items()
            if thr is not None
        }
        cond_summary[cond] = {
            "n": item["n"],
            "target_mean": item["target_sum"] / max(item["n"], 1),
            "score_mean": item["score_sum"] / max(item["n"], 1),
            "score_p50": percentile(cond_scores, 0.50),
            "score_p90": percentile(cond_scores, 0.90),
            "score_p99": percentile(cond_scores, 0.99),
            "brier": item["brier"] / max(item["n"], 1),
            "trigger_rate": trigger,
        }
    policy_summary = {
        policy: {
            "n": item["n"],
            "target_mean": item["target_sum"] / max(item["n"], 1),
            "score_mean": item["score_sum"] / max(item["n"], 1),
        }
        for policy, item in sorted(by_policy.items())
    }
    return {
        "n": n,
        "target_positive_rate_binary": sum(labels) / max(n, 1),
        "score_mean": sum(scores) / max(n, 1),
        "auc_binary_target_ge_0_5": auc_score(labels, scores),
        "brier_soft": brier,
        "logloss_soft": logloss,
        "by_condition": cond_summary,
        "by_policy": policy_summary,
    }


def threshold_suggestions(rows, weights, bias, means, stds):
    clean_scores = []
    for row in rows:
        if row.get("condition") == "clean":
            clean_scores.append(predict_row(row, weights, bias, means, stds))
    return {
        "clean_fpr_001": percentile(clean_scores, 0.999),
        "clean_fpr_005": percentile(clean_scores, 0.995),
        "clean_fpr_010": percentile(clean_scores, 0.990),
        "clean_fpr_020": percentile(clean_scores, 0.980),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-summary", default="")
    parser.add_argument("--allow-missing-label-summary", action="store_true")
    parser.add_argument("--target", default="scene_bev_damage_applicable")
    parser.add_argument("--features", default=",".join(DEFAULT_SCENE_FEATURES))
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--no-class-balance", action="store_true")
    parser.add_argument("--allow-unsafe-cross-branch-features", action="store_true")
    args = parser.parse_args()

    features = [name.strip() for name in args.features.split(",") if name.strip()]
    unsafe = sorted(set(features) & UNSAFE_SCENE_FEATURES)
    if unsafe and not args.allow_unsafe_cross_branch_features:
        raise SystemExit(f"Refusing unsafe scene features by default: {unsafe}")

    label_summary_path = args.label_summary or infer_label_summary_path(args.labels)
    label_summary_missing = not bool(label_summary_path)
    if label_summary_missing and not args.allow_missing_label_summary:
        raise RuntimeError(
            "Missing label summary for formal scene mapper training. "
            "Pass --label-summary from build_umr_scene_gate_labels.py, or use "
            "--allow-missing-label-summary for debug-only mapper output."
        )
    label_summary = load_label_summary(label_summary_path)
    if not label_summary and label_summary_path:
        label_summary_missing = True
        if not args.allow_missing_label_summary:
            raise RuntimeError(f"Could not read label summary: {label_summary_path}")
    label_bad_missing = label_summary.get("bad_missing_features", {}) if isinstance(label_summary, dict) else {}
    label_missing_rates = label_summary.get("missing_feature_rates", {}) if isinstance(label_summary, dict) else {}
    label_debug_only = bool(label_summary.get("debug_only", False)) if isinstance(label_summary, dict) else False
    label_formal_safe = bool(label_summary.get("formal_calibration_safe", False)) if isinstance(label_summary, dict) else False
    mapper_formal_safe = bool(
        not unsafe and
        not label_summary_missing and
        label_formal_safe and
        not label_debug_only and
        not label_bad_missing
    )

    rows, load_counts = load_records(args.labels, args.target, features, args.max_records, args.seed)
    if len(rows) < 10:
        raise RuntimeError(f"Too few rows for scene mapper training: {len(rows)}")
    train_rows, val_rows = split_rows(rows, args.holdout_fraction, args.seed)
    means, stds = standardize_fit(train_rows, len(features))
    weights, bias = train_logistic(
        train_rows, len(features), args.epochs, args.lr, args.l2,
        means, stds, class_balance=not args.no_class_balance,
        progress_interval=args.progress_interval)
    thresholds = threshold_suggestions(train_rows, weights, bias, means, stds)
    payload = {
        "target": args.target,
        "features": features,
        "normalization": {"mean": means, "std": stds},
        "model": {"type": "weighted_soft_logistic_regression", "weights": weights, "bias": bias},
        "thresholds": thresholds,
        "label_summary_path": label_summary_path,
        "label_summary_sha256": (
            sha256_file(label_summary_path) if label_summary_path else ""),
        "labels_path": args.labels,
        "labels_sha256": sha256_file(args.labels),
        "label_summary_missing": label_summary_missing,
        "feature_missing_rates": label_missing_rates,
        "bad_missing_features": label_bad_missing,
        "load_counts": dict(load_counts),
        "train_metrics": evaluate(train_rows, weights, bias, means, stds, thresholds),
        "holdout_metrics": evaluate(val_rows, weights, bias, means, stds, thresholds),
        "notes": [
            "Fit only on train/train-O.C scene logs.",
            "Targets are scene applicability labels; runtime must not read condition or severity.",
            "beams_s2 weak-positive support is encoded by soft target/weight in labels.",
        ],
        "formal_calibration_safe": mapper_formal_safe,
        "debug_only": bool(
            unsafe or
            label_summary_missing or
            label_debug_only or
            label_bad_missing or
            not label_formal_safe
        ),
        "unsafe_features": unsafe,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    md_path = os.path.splitext(args.output)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# UMR Scene Damage Mapper\n\n")
        f.write(f"- target: `{args.target}`\n")
        f.write(f"- rows loaded: `{len(rows)}`\n")
        f.write(f"- features: `{', '.join(features)}`\n\n")
        f.write("## Thresholds\n\n")
        for name, value in thresholds.items():
            f.write(f"- `{name}`: `{value}`\n")
        f.write("\n## Holdout Metrics\n\n")
        for key, value in payload["holdout_metrics"].items():
            if key in ("by_condition", "by_policy"):
                continue
            f.write(f"- {key}: `{value}`\n")
        f.write("\n## Holdout By Condition\n\n")
        f.write("| condition | n | target | score_mean | p90 | p99 | clean_fpr_005_trigger |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for cond, item in payload["holdout_metrics"]["by_condition"].items():
            trig = item.get("trigger_rate", {}).get("clean_fpr_005")
            f.write(
                f"| {cond} | {item['n']} | {item['target_mean']:.4f} | "
                f"{item['score_mean']:.4f} | {item['score_p90']:.4f} | "
                f"{item['score_p99']:.4f} | "
                f"{'N/A' if trig is None else f'{trig:.4f}'} |\n"
            )
    print(f"[UMR scene mapper] wrote: {args.output}")
    print(f"[UMR scene mapper] wrote: {md_path}")
    print(f"[UMR scene mapper] holdout: {payload['holdout_metrics']}")


if __name__ == "__main__":
    main()
