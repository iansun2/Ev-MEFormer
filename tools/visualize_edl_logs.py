#!/usr/bin/env python3
import argparse
import concurrent.futures
import csv
import gzip
import json
import math
import multiprocessing
import os
import random
import re
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


MODALITIES = ["fused", "bev", "img"]
MODALITY_COLORS = {
    "fused": "#1f77b4",
    "bev": "#2ca02c",
    "img": "#ff7f0e",
}
SOURCE_CATEGORIES = ["soft_blend", "fused", "bev", "img"]
SOURCE_COLORS = {
    "soft_blend": "#7f7f7f",
    "fused": "#1f77b4",
    "bev": "#2ca02c",
    "img": "#ff7f0e",
}
V7_DAMAGE_STATES = ["neutral", "lidar_failure", "image_failure", "dual_damage"]
V7_DAMAGE_COLORS = {
    "neutral": "#7f7f7f",
    "lidar_failure": "#1f77b4",
    "image_failure": "#ff7f0e",
    "dual_damage": "#d62728",
}
V7_CENTER_SOURCE_CATEGORIES = ["soft_blend", "bev_fallback", "img_fallback"]
V7_CENTER_SOURCE_COLORS = {
    "soft_blend": "#7f7f7f",
    "bev_fallback": "#2ca02c",
    "img_fallback": "#ff7f0e",
}
V8_DAMAGE_STATES = ["clean_like", "bev_damage", "img_damage", "dual_damage", "ambiguous"]
V8_DAMAGE_COLORS = {
    "clean_like": "#7f7f7f",
    "bev_damage": "#2ca02c",
    "img_damage": "#ff7f0e",
    "dual_damage": "#d62728",
    "ambiguous": "#9467bd",
}
V8_RELIABILITY_STATES = ["both_reliable", "bev_unreliable", "img_unreliable", "dual_unreliable", "ambiguous"]
V8_RELIABILITY_COLORS = {
    "both_reliable": "#7f7f7f",
    "bev_unreliable": "#2ca02c",
    "img_unreliable": "#ff7f0e",
    "dual_unreliable": "#d62728",
    "ambiguous": "#9467bd",
}
V8_SOURCE_CATEGORIES = ["fused", "bev", "img"]
V8_SOURCE_COLORS = {
    "fused": "#1f77b4",
    "bev": "#2ca02c",
    "img": "#ff7f0e",
}
BASELINE_NAME = "MEFormer R50"
BASELINE_CLEAN_NDS = 0.7096
BASELINE_RA_NDS = {
    "beamsreducing": 0.7162,
    "brightness": 0.9517,
    "dark": 0.9596,
    "fog": 0.7764,
    "missingcamera": 0.9552,
    "motionblur": 0.8470,
    "pointsreducing": 0.9014,
    "snow": 0.7843,
    "spatialmisalignment": 0.7469,
    "temporalmisalignment": 0.7836,
}

SCATTER_MAX_POINTS_PER_CONDITION = 4000
ANOMALY_HIST_BINS = np.linspace(-1.0, 10.0, 45)
DISTANCE_HIST_BINS = np.linspace(0.0, 100.0, 41)
_RNG = random.Random(0)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_manifest(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def open_text_reader(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    if path.endswith(".zst"):
        try:
            import io
            import zstandard as zstd
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                f"Cannot read zstd-compressed EDL log without zstandard: {path}"
            ) from exc
        raw_fp = open(path, "rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw_fp, closefd=True)
        return io.TextIOWrapper(reader, encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_jsonl(path):
    with open_text_reader(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def progress_iter(iterable, total=None, desc="Progress"):
    if tqdm is None:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
    )


def default_num_jobs():
    cpu = os.cpu_count() or 1
    return max(1, min(8, cpu))


def default_heartbeat_samples():
    return 250


class RunningStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0

    def add(self, value):
        if math.isnan(value):
            return
        self.count += 1
        self.total += float(value)

    def mean(self):
        if self.count == 0:
            return None
        return self.total / self.count


class HistogramAccumulator:
    def __init__(self, edges):
        self.edges = np.asarray(edges, dtype=float)
        self.counts = np.zeros(len(self.edges) - 1, dtype=np.int64)
        self.stats = RunningStats()

    def add(self, value):
        if math.isnan(value):
            return
        value = float(value)
        idx = int(np.searchsorted(self.edges, value, side="right") - 1)
        idx = max(0, min(idx, len(self.counts) - 1))
        self.counts[idx] += 1
        self.stats.add(value)

    def mean(self):
        return self.stats.mean()

    def total(self):
        return int(self.counts.sum())


class ReservoirSampler:
    def __init__(self, max_items):
        self.max_items = int(max_items)
        self.items = []
        self.seen = 0

    def add(self, item):
        self.seen += 1
        if len(self.items) < self.max_items:
            self.items.append(item)
            return
        idx = _RNG.randrange(self.seen)
        if idx < self.max_items:
            self.items[idx] = item

    def merge_items(self, items):
        for item in items:
            self.add(item)


def iter_json_array(path, chunk_size=1 << 20):
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buf = ""
        pos = 0
        started = False
        eof = False
        while True:
            if not eof and (len(buf) - pos) < (chunk_size // 2):
                more = f.read(chunk_size)
                if more:
                    buf = buf[pos:] + more
                    pos = 0
                else:
                    eof = True
                    buf = buf[pos:]
                    pos = 0

            while True:
                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if not started:
                    if pos >= len(buf):
                        break
                    if buf[pos] != "[":
                        raise ValueError(f"{path} is not a JSON array")
                    started = True
                    pos += 1
                    continue

                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if pos < len(buf) and buf[pos] == ",":
                    pos += 1
                    continue
                if pos < len(buf) and buf[pos] == "]":
                    return
                if pos >= len(buf):
                    break

                try:
                    obj, end = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    break
                yield obj
                pos = end

                if eof and pos >= len(buf):
                    return


def row_sidecar_paths(row):
    return {
        "scene": row.get("edl_scene_summary_jsonl"),
        "prediction": row.get("edl_prediction_light_jsonl"),
        "meta": row.get("edl_sidecar_meta_json"),
    }


def row_has_sidecar(row):
    paths = row_sidecar_paths(row)
    return (
        paths["scene"] and os.path.isfile(paths["scene"]) and
        paths["prediction"] and os.path.isfile(paths["prediction"])
    )


def row_log_size_mb(row):
    if row_has_sidecar(row):
        paths = row_sidecar_paths(row)
        size = 0
        for key in ("scene", "prediction", "meta"):
            path = paths.get(key)
            if path and os.path.isfile(path):
                size += os.path.getsize(path)
        return size / (1024.0 * 1024.0)
    log_json = row.get("edl_log_json")
    if log_json and os.path.isfile(log_json):
        return os.path.getsize(log_json) / (1024.0 * 1024.0)
    return 0.0


def parse_eval_log(path):
    pattern = re.compile(
        r"Corruption:\s*(\w+),\s*Severity:\s*(\d+),\s*NDS:\s*([\d.]+|N/A|ERROR),\s*mAP:\s*([\d.]+|N/A|ERROR)"
    )
    results = {}
    if not path or not os.path.isfile(path):
        return results
    with open(path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            try:
                results[(match.group(1), int(match.group(2)))] = (
                    float(match.group(3)),
                    float(match.group(4)),
                )
            except ValueError:
                continue
    return results


def extract_scene_weights(sample):
    weights = sample.get("scene_avg_weights")
    if isinstance(weights, dict):
        return {m: safe_float(weights.get(m, np.nan)) for m in MODALITIES}

    preds = sample.get("predictions", [])
    if not preds:
        return None
    accum_sum = {m: 0.0 for m in MODALITIES}
    accum_count = {m: 0 for m in MODALITIES}
    for pred in preds:
        pred_weights = pred.get("modality_weights") or pred.get("ensemble_info", {}).get("attn_weights")
        if not isinstance(pred_weights, dict):
            continue
        for m in MODALITIES:
            if m in pred_weights:
                value = safe_float(pred_weights[m])
                if not math.isnan(value):
                    accum_sum[m] += value
                    accum_count[m] += 1
    if not any(accum_count[m] for m in MODALITIES):
        return None
    return {
        m: (accum_sum[m] / accum_count[m]) if accum_count[m] else np.nan
        for m in MODALITIES
    }


def extract_scene_unc(sample):
    unc = sample.get("scene_avg_unc")
    if isinstance(unc, dict):
        return {m: safe_float(unc.get(m, np.nan)) for m in MODALITIES}

    preds = sample.get("predictions", [])
    if not preds:
        return None
    accum_sum = {m: 0.0 for m in MODALITIES}
    accum_count = {m: 0 for m in MODALITIES}
    for pred in preds:
        pred_unc = pred.get("per_modality_unc")
        if not isinstance(pred_unc, dict):
            continue
        for m in MODALITIES:
            if m in pred_unc:
                value = safe_float(pred_unc[m])
                if not math.isnan(value):
                    accum_sum[m] += value
                    accum_count[m] += 1
    if not any(accum_count[m] for m in MODALITIES):
        return None
    return {
        m: (accum_sum[m] / accum_count[m]) if accum_count[m] else np.nan
        for m in MODALITIES
    }


def compress_sample_for_qualitative(sample, split, condition, condition_with_sev):
    preds = sample.get("predictions", [])
    points = []
    for pred in preds:
        center = pred.get("bbox_center")
        if not isinstance(center, list) or len(center) < 2:
            continue
        x = safe_float(center[0])
        y = safe_float(center[1])
        if math.isnan(x) or math.isnan(y):
            continue
        points.append({
            "x": x,
            "y": y,
            "selected_modality": pred.get("selected_modality", "fused"),
            "combined_unc": safe_float(pred.get("combined_unc", pred.get("center_unc")), default=0.1),
        })
    return {
        "split": split,
        "condition": condition,
        "condition_with_sev": condition_with_sev,
        "sample_token": sample.get("sample_token", "unknown"),
        "pred_count": len(preds),
        "points": points,
    }


def _append_by_modality(store, condition, values):
    for modality in MODALITIES:
        val = values.get(modality, np.nan)
        if not math.isnan(val):
            store[condition][modality].append(val)


def _append_scalar(store, condition, value):
    if not math.isnan(value):
        store[condition].append(value)


def _partial_hist_payload(acc):
    return {
        "counts": acc.counts.tolist(),
        "edges": acc.edges.tolist(),
        "count": acc.stats.count,
        "total": acc.stats.total,
    }


def _merge_hist_payload(acc, payload):
    acc.counts += np.asarray(payload["counts"], dtype=np.int64)
    acc.stats.count += int(payload["count"])
    acc.stats.total += float(payload["total"])


def _init_partial_stores():
    return {
        "has_v6_schema": False,
        "has_v7_schema": False,
        "has_v8_schema": False,
        "has_v9_schema": False,
        "scene_weights": defaultdict(lambda: defaultdict(list)),
        "scene_uncs": defaultdict(lambda: defaultdict(list)),
        "selected_counts": defaultdict(lambda: defaultdict(int)),
        "entropy_values": defaultdict(list),
        "fused_weight_values": defaultdict(list),
        "distance_prior_top1": defaultdict(lambda: defaultdict(list)),
        "residual_override_values": defaultdict(list),
        "alpha_res_values": defaultdict(list),
        "tail_query_values": defaultdict(lambda: defaultdict(list)),
        "scatter_points": defaultdict(lambda: ReservoirSampler(SCATTER_MAX_POINTS_PER_CONDITION)),
        "v5_switch_stats": defaultdict(lambda: {"switched": 0, "total": 0}),
        "v5_fused_anomaly_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v5_switch_distance_hists": defaultdict(lambda: HistogramAccumulator(DISTANCE_HIST_BINS)),
        "v5_switch_class_counts": defaultdict(lambda: defaultdict(int)),
        "v5_fallback_level_counts": defaultdict(lambda: defaultdict(int)),
        "v5_selected_counts": defaultdict(lambda: defaultdict(int)),
        "v6_switch_stats": defaultdict(lambda: {"switched": 0, "total": 0}),
        "v6_gate_anomaly_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v6_switch_distance_hists": defaultdict(lambda: HistogramAccumulator(DISTANCE_HIST_BINS)),
        "v6_switch_class_counts": defaultdict(lambda: defaultdict(int)),
        "v6_fallback_level_counts": defaultdict(lambda: defaultdict(int)),
        "v6_candidate_counts": defaultdict(lambda: defaultdict(int)),
        "v6_center_source_counts": defaultdict(lambda: defaultdict(int)),
        "v6_height_source_counts": defaultdict(lambda: defaultdict(int)),
        "v6_gate_pass_anomaly_values": defaultdict(list),
        "v6_gate_pass_candidate_prob_values": defaultdict(list),
        "v6_gate_pass_candidate_margin_values": defaultdict(list),
        "v6_gate_pass_final_values": defaultdict(list),
        "v7_switch_stats": defaultdict(lambda: {"switched": 0, "total": 0}),
        "v7_bev_center_anomaly_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v7_img_center_anomaly_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v7_switch_distance_hists": defaultdict(lambda: HistogramAccumulator(DISTANCE_HIST_BINS)),
        "v7_switch_class_counts": defaultdict(lambda: defaultdict(int)),
        "v7_damage_state_counts": defaultdict(lambda: defaultdict(int)),
        "v7_center_source_counts": defaultdict(lambda: defaultdict(int)),
        "v8_fused_keep_values": defaultdict(list),
        "v8_center_fused_keep_values": defaultdict(list),
        "v8_damage_gate_values": defaultdict(list),
        "v8_high_damage_values": defaultdict(list),
        "v8_center_route_allowed_values": defaultdict(list),
        "v8_bev_damage_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v8_img_damage_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v8_dual_damage_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v8_context_damage_values": defaultdict(lambda: defaultdict(list)),
        "v8_a4_bev_damage_z_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v8_a4_img_damage_z_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v8_a4_cls_unc_z_bev_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v8_a4_cls_unc_z_img_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v8_a4_route_score_bev_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v8_a4_route_score_img_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v8_damage_state_counts": defaultdict(lambda: defaultdict(int)),
        "v8_reliability_state_counts": defaultdict(lambda: defaultdict(int)),
        "v8_selected_source_counts": defaultdict(lambda: defaultdict(int)),
        "v8_non_fused_choice_counts": defaultdict(lambda: defaultdict(int)),
        "v8_reliability_values": defaultdict(lambda: defaultdict(list)),
        "v8_reliability_margin_values": defaultdict(list),
        "v8_soft_weight_values": defaultdict(lambda: defaultdict(list)),
        "v8_output_delta_values": defaultdict(lambda: defaultdict(list)),
        "v8_domain_correct_values": defaultdict(list),
        "v9_a_bev_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v9_a_img_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v9_anchor_distance_bev_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v9_anchor_distance_img_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v9_tail_prob_bev_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v9_tail_prob_img_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v9_tail_excess_bev_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v9_tail_excess_img_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v9_q95_shift_bev_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v9_q95_shift_img_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v9_route_score_bev_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v9_route_score_img_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v9_velocity_l1_hists": defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS)),
        "v9_above_tau_values": defaultdict(lambda: defaultdict(list)),
        "v9_clean_like_values": defaultdict(list),
        "v9_scene_scalar_values": defaultdict(lambda: defaultdict(list)),
    }


def _finalize_partial(partial, representative_sample, meta):
    def nested_lists_to_dict(store):
        return {cond: {m: list(vals[m]) for m in MODALITIES} for cond, vals in store.items()}

    def nested_counts_to_dict(store):
        return {cond: {m: int(store[cond].get(m, 0)) for m in MODALITIES} for cond in store}

    return {
        "meta": meta,
        "has_v6_schema": bool(partial["has_v6_schema"]),
        "has_v7_schema": bool(partial["has_v7_schema"]),
        "has_v8_schema": bool(partial["has_v8_schema"]),
        "has_v9_schema": bool(partial["has_v9_schema"]),
        "scene_weights": nested_lists_to_dict(partial["scene_weights"]),
        "scene_uncs": nested_lists_to_dict(partial["scene_uncs"]),
        "selected_counts": nested_counts_to_dict(partial["selected_counts"]),
        "entropy_values": {cond: list(vals) for cond, vals in partial["entropy_values"].items()},
        "fused_weight_values": {cond: list(vals) for cond, vals in partial["fused_weight_values"].items()},
        "distance_prior_top1": nested_lists_to_dict(partial["distance_prior_top1"]),
        "residual_override_values": {cond: list(vals) for cond, vals in partial["residual_override_values"].items()},
        "alpha_res_values": {cond: list(vals) for cond, vals in partial["alpha_res_values"].items()},
        "tail_query_values": nested_lists_to_dict(partial["tail_query_values"]),
        "scatter_points": {
            cond: {"seen": sampler.seen, "items": list(sampler.items)}
            for cond, sampler in partial["scatter_points"].items()
        },
        "v5_switch_stats": {cond: dict(stats) for cond, stats in partial["v5_switch_stats"].items()},
        "v5_fused_anomaly_hists": {cond: _partial_hist_payload(hist) for cond, hist in partial["v5_fused_anomaly_hists"].items()},
        "v5_switch_distance_hists": {cond: _partial_hist_payload(hist) for cond, hist in partial["v5_switch_distance_hists"].items()},
        "v5_switch_class_counts": {cond: dict(counts) for cond, counts in partial["v5_switch_class_counts"].items()},
        "v5_fallback_level_counts": {cond: dict(counts) for cond, counts in partial["v5_fallback_level_counts"].items()},
        "v5_selected_counts": {cond: dict(counts) for cond, counts in partial["v5_selected_counts"].items()},
        "v6_switch_stats": {cond: dict(stats) for cond, stats in partial["v6_switch_stats"].items()},
        "v6_gate_anomaly_hists": {cond: _partial_hist_payload(hist) for cond, hist in partial["v6_gate_anomaly_hists"].items()},
        "v6_switch_distance_hists": {cond: _partial_hist_payload(hist) for cond, hist in partial["v6_switch_distance_hists"].items()},
        "v6_switch_class_counts": {cond: dict(counts) for cond, counts in partial["v6_switch_class_counts"].items()},
        "v6_fallback_level_counts": {cond: dict(counts) for cond, counts in partial["v6_fallback_level_counts"].items()},
        "v6_candidate_counts": {cond: dict(counts) for cond, counts in partial["v6_candidate_counts"].items()},
        "v6_center_source_counts": {cond: dict(counts) for cond, counts in partial["v6_center_source_counts"].items()},
        "v6_height_source_counts": {cond: dict(counts) for cond, counts in partial["v6_height_source_counts"].items()},
        "v6_gate_pass_anomaly_values": {cond: list(vals) for cond, vals in partial["v6_gate_pass_anomaly_values"].items()},
        "v6_gate_pass_candidate_prob_values": {cond: list(vals) for cond, vals in partial["v6_gate_pass_candidate_prob_values"].items()},
        "v6_gate_pass_candidate_margin_values": {cond: list(vals) for cond, vals in partial["v6_gate_pass_candidate_margin_values"].items()},
        "v6_gate_pass_final_values": {cond: list(vals) for cond, vals in partial["v6_gate_pass_final_values"].items()},
        "v7_switch_stats": {cond: dict(stats) for cond, stats in partial["v7_switch_stats"].items()},
        "v7_bev_center_anomaly_hists": {cond: _partial_hist_payload(hist) for cond, hist in partial["v7_bev_center_anomaly_hists"].items()},
        "v7_img_center_anomaly_hists": {cond: _partial_hist_payload(hist) for cond, hist in partial["v7_img_center_anomaly_hists"].items()},
        "v7_switch_distance_hists": {cond: _partial_hist_payload(hist) for cond, hist in partial["v7_switch_distance_hists"].items()},
        "v7_switch_class_counts": {cond: dict(counts) for cond, counts in partial["v7_switch_class_counts"].items()},
        "v7_damage_state_counts": {cond: dict(counts) for cond, counts in partial["v7_damage_state_counts"].items()},
        "v7_center_source_counts": {cond: dict(counts) for cond, counts in partial["v7_center_source_counts"].items()},
        "v8_fused_keep_values": {cond: list(vals) for cond, vals in partial["v8_fused_keep_values"].items()},
        "v8_center_fused_keep_values": {cond: list(vals) for cond, vals in partial["v8_center_fused_keep_values"].items()},
        "v8_damage_gate_values": {cond: list(vals) for cond, vals in partial["v8_damage_gate_values"].items()},
        "v8_high_damage_values": {cond: list(vals) for cond, vals in partial["v8_high_damage_values"].items()},
        "v8_center_route_allowed_values": {cond: list(vals) for cond, vals in partial["v8_center_route_allowed_values"].items()},
        "v8_bev_damage_hists": {cond: _partial_hist_payload(hist) for cond, hist in partial["v8_bev_damage_hists"].items()},
        "v8_img_damage_hists": {cond: _partial_hist_payload(hist) for cond, hist in partial["v8_img_damage_hists"].items()},
        "v8_dual_damage_hists": {cond: _partial_hist_payload(hist) for cond, hist in partial["v8_dual_damage_hists"].items()},
        "v8_context_damage_values": {
            cond: {m: list(vals[m]) for m in ("bev", "img")}
            for cond, vals in partial["v8_context_damage_values"].items()
        },
        "v8_a4_bev_damage_z_hists": {cond: _partial_hist_payload(hist) for cond, hist in partial["v8_a4_bev_damage_z_hists"].items()},
        "v8_a4_img_damage_z_hists": {cond: _partial_hist_payload(hist) for cond, hist in partial["v8_a4_img_damage_z_hists"].items()},
        "v8_a4_cls_unc_z_bev_hists": {cond: _partial_hist_payload(hist) for cond, hist in partial["v8_a4_cls_unc_z_bev_hists"].items()},
        "v8_a4_cls_unc_z_img_hists": {cond: _partial_hist_payload(hist) for cond, hist in partial["v8_a4_cls_unc_z_img_hists"].items()},
        "v8_a4_route_score_bev_hists": {cond: _partial_hist_payload(hist) for cond, hist in partial["v8_a4_route_score_bev_hists"].items()},
        "v8_a4_route_score_img_hists": {cond: _partial_hist_payload(hist) for cond, hist in partial["v8_a4_route_score_img_hists"].items()},
        "v8_damage_state_counts": {cond: dict(counts) for cond, counts in partial["v8_damage_state_counts"].items()},
        "v8_reliability_state_counts": {cond: dict(counts) for cond, counts in partial["v8_reliability_state_counts"].items()},
        "v8_selected_source_counts": {cond: dict(counts) for cond, counts in partial["v8_selected_source_counts"].items()},
        "v8_non_fused_choice_counts": {cond: dict(counts) for cond, counts in partial["v8_non_fused_choice_counts"].items()},
        "v8_reliability_values": {
            cond: {m: list(vals[m]) for m in ("bev", "img")}
            for cond, vals in partial["v8_reliability_values"].items()
        },
        "v8_reliability_margin_values": {cond: list(vals) for cond, vals in partial["v8_reliability_margin_values"].items()},
        "v8_soft_weight_values": {
            cond: {m: list(vals[m]) for m in MODALITIES}
            for cond, vals in partial["v8_soft_weight_values"].items()
        },
        "v8_output_delta_values": {
            cond: {m: list(vals[m]) for m in ("cls", "center", "height")}
            for cond, vals in partial["v8_output_delta_values"].items()
        },
        "v8_domain_correct_values": {cond: list(vals) for cond, vals in partial["v8_domain_correct_values"].items()},
        "v9_a_bev_hists": {cond: _partial_hist_payload(hist) for cond, hist in partial["v9_a_bev_hists"].items()},
        "v9_a_img_hists": {cond: _partial_hist_payload(hist) for cond, hist in partial["v9_a_img_hists"].items()},
        "v9_anchor_distance_bev_hists": {
            cond: _partial_hist_payload(hist) for cond, hist in partial["v9_anchor_distance_bev_hists"].items()
        },
        "v9_anchor_distance_img_hists": {
            cond: _partial_hist_payload(hist) for cond, hist in partial["v9_anchor_distance_img_hists"].items()
        },
        "v9_tail_prob_bev_hists": {
            cond: _partial_hist_payload(hist) for cond, hist in partial["v9_tail_prob_bev_hists"].items()
        },
        "v9_tail_prob_img_hists": {
            cond: _partial_hist_payload(hist) for cond, hist in partial["v9_tail_prob_img_hists"].items()
        },
        "v9_tail_excess_bev_hists": {
            cond: _partial_hist_payload(hist) for cond, hist in partial["v9_tail_excess_bev_hists"].items()
        },
        "v9_tail_excess_img_hists": {
            cond: _partial_hist_payload(hist) for cond, hist in partial["v9_tail_excess_img_hists"].items()
        },
        "v9_q95_shift_bev_hists": {
            cond: _partial_hist_payload(hist) for cond, hist in partial["v9_q95_shift_bev_hists"].items()
        },
        "v9_q95_shift_img_hists": {
            cond: _partial_hist_payload(hist) for cond, hist in partial["v9_q95_shift_img_hists"].items()
        },
        "v9_route_score_bev_hists": {
            cond: _partial_hist_payload(hist) for cond, hist in partial["v9_route_score_bev_hists"].items()
        },
        "v9_route_score_img_hists": {
            cond: _partial_hist_payload(hist) for cond, hist in partial["v9_route_score_img_hists"].items()
        },
        "v9_velocity_l1_hists": {
            cond: _partial_hist_payload(hist) for cond, hist in partial["v9_velocity_l1_hists"].items()
        },
        "v9_above_tau_values": {
            cond: {m: list(vals[m]) for m in ("bev", "img")}
            for cond, vals in partial["v9_above_tau_values"].items()
        },
        "v9_clean_like_values": {cond: list(vals) for cond, vals in partial["v9_clean_like_values"].items()},
        "v9_scene_scalar_values": {
            cond: {name: list(vals[name]) for name in vals}
            for cond, vals in partial["v9_scene_scalar_values"].items()
        },
        "representative_sample": representative_sample,
    }


def _aggregate_scene_record(partial, sample, condition, condition_with_sev):
    if any(key.startswith("scene_v6_") for key in sample.keys()):
        partial["has_v6_schema"] = True
    if any(key.startswith("scene_v7_") for key in sample.keys()):
        partial["has_v7_schema"] = True
    if any(key.startswith("scene_v8_") for key in sample.keys()):
        partial["has_v8_schema"] = True
    if any(key.startswith("scene_v9_") for key in sample.keys()):
        partial["has_v9_schema"] = True
    weights = extract_scene_weights(sample)
    if weights:
        _append_by_modality(partial["scene_weights"], condition, weights)
        _append_by_modality(partial["scene_weights"], condition_with_sev, weights)

    uncs = extract_scene_unc(sample)
    if uncs:
        _append_by_modality(partial["scene_uncs"], condition, uncs)
        _append_by_modality(partial["scene_uncs"], condition_with_sev, uncs)

    if "scene_selector_entropy" in sample:
        val = safe_float(sample["scene_selector_entropy"])
        _append_scalar(partial["entropy_values"], condition, val)
        _append_scalar(partial["entropy_values"], condition_with_sev, val)

    if "scene_fused_weight" in sample:
        val = safe_float(sample["scene_fused_weight"])
        _append_scalar(partial["fused_weight_values"], condition, val)
        _append_scalar(partial["fused_weight_values"], condition_with_sev, val)

    if "scene_distance_prior_top1_ratio" in sample:
        prior = sample["scene_distance_prior_top1_ratio"]
        if isinstance(prior, dict):
            payload = {m: safe_float(prior.get(m)) for m in MODALITIES}
            _append_by_modality(partial["distance_prior_top1"], condition, payload)
            _append_by_modality(partial["distance_prior_top1"], condition_with_sev, payload)

    if "scene_residual_override_ratio" in sample:
        val = safe_float(sample["scene_residual_override_ratio"])
        _append_scalar(partial["residual_override_values"], condition, val)
        _append_scalar(partial["residual_override_values"], condition_with_sev, val)

    if "scene_alpha_res_mean" in sample:
        val = safe_float(sample["scene_alpha_res_mean"])
        _append_scalar(partial["alpha_res_values"], condition, val)
        _append_scalar(partial["alpha_res_values"], condition_with_sev, val)

    if "scene_uncertainty_tail_query_ratio" in sample:
        tail = sample["scene_uncertainty_tail_query_ratio"]
        if isinstance(tail, dict):
            payload = {m: safe_float(tail.get(m)) for m in MODALITIES}
            _append_by_modality(partial["tail_query_values"], condition, payload)
            _append_by_modality(partial["tail_query_values"], condition_with_sev, payload)
    for sample_key, store_key in [
        ("scene_v6_gate_pass_anomaly_ratio", "v6_gate_pass_anomaly_values"),
        ("scene_v6_gate_pass_candidate_prob_ratio", "v6_gate_pass_candidate_prob_values"),
        ("scene_v6_gate_pass_candidate_margin_ratio", "v6_gate_pass_candidate_margin_values"),
        ("scene_v6_gate_pass_final_ratio", "v6_gate_pass_final_values"),
        ("scene_v8_fused_keep_mean", "v8_fused_keep_values"),
        ("scene_v8_center_fused_keep_mean", "v8_center_fused_keep_values"),
        ("scene_v8_damage_gate_mean", "v8_damage_gate_values"),
        ("scene_v8_high_damage_query_ratio", "v8_high_damage_values"),
        ("scene_v8_center_route_allowed_ratio", "v8_center_route_allowed_values"),
    ]:
        if sample_key in sample:
            val = safe_float(sample[sample_key])
            _append_scalar(partial[store_key], condition, val)
            _append_scalar(partial[store_key], condition_with_sev, val)
    for sample_key, store_key in [
        ("scene_v8_damage_state_ratio", "v8_damage_state_counts"),
        ("scene_v8_reliability_state_ratio", "v8_reliability_state_counts"),
        ("scene_v8_selected_source_ratio", "v8_selected_source_counts"),
    ]:
        payload = sample.get(sample_key)
        if isinstance(payload, dict):
            scale = 1000000
            for key, value in payload.items():
                val = safe_float(value)
                if not math.isnan(val):
                    partial[store_key][condition][key] += int(round(val * scale))
                    partial[store_key][condition_with_sev][key] += int(round(val * scale))
    for sample_key, store_key, categories in [
        ("scene_v8_soft_weight_ratio", "v8_soft_weight_values", MODALITIES),
        ("scene_v8_output_delta_cls_mean", "v8_output_delta_values", ("cls",)),
        ("scene_v8_output_delta_center_mean", "v8_output_delta_values", ("center",)),
        ("scene_v8_output_delta_height_mean", "v8_output_delta_values", ("height",)),
    ]:
        payload = sample.get(sample_key)
        if isinstance(payload, dict):
            for key in categories:
                val = safe_float(payload.get(key))
                if not math.isnan(val):
                    partial[store_key][condition][key].append(val)
                    partial[store_key][condition_with_sev][key].append(val)
        elif categories:
            val = safe_float(payload)
            if not math.isnan(val):
                key = categories[0]
                partial[store_key][condition][key].append(val)
                partial[store_key][condition_with_sev][key].append(val)
    for sample_key, modality in [
        ("scene_v8_bev_reliability_mean", "bev"),
        ("scene_v8_img_reliability_mean", "img"),
    ]:
        val = safe_float(sample.get(sample_key))
        if not math.isnan(val):
            partial["v8_reliability_values"][condition][modality].append(val)
            partial["v8_reliability_values"][condition_with_sev][modality].append(val)
    val = safe_float(sample.get("scene_v8_reliability_margin_mean"))
    if not math.isnan(val):
        partial["v8_reliability_margin_values"][condition].append(val)
        partial["v8_reliability_margin_values"][condition_with_sev].append(val)
    for sample_key, modality in [
        ("scene_v8_context_bev_damage_mean", "bev"),
        ("scene_v8_context_img_damage_mean", "img"),
    ]:
        val = safe_float(sample.get(sample_key))
        if not math.isnan(val):
            partial["v8_context_damage_values"][condition][modality].append(val)
            partial["v8_context_damage_values"][condition_with_sev][modality].append(val)
    for sample_key, name in [
        ("scene_v9_a_bev_mean", "a_bev"),
        ("scene_v9_a_img_mean", "a_img"),
        ("scene_v9_a_bev_above_tau_mean", "above_tau_bev"),
        ("scene_v9_a_img_above_tau_mean", "above_tau_img"),
        ("scene_v9_clean_like_mask_mean", "clean_like"),
        ("scene_v9_anchor_distance_bev_mean", "anchor_distance_bev"),
        ("scene_v9_anchor_distance_img_mean", "anchor_distance_img"),
        ("scene_v9_tail_prob_bev_mean", "tail_prob_bev"),
        ("scene_v9_tail_prob_img_mean", "tail_prob_img"),
        ("scene_v9_tail_excess_bev_mean", "tail_excess_bev"),
        ("scene_v9_tail_excess_img_mean", "tail_excess_img"),
        ("scene_v9_cvar_bev_mean", "cvar_bev"),
        ("scene_v9_cvar_img_mean", "cvar_img"),
        ("scene_v9_q95_shift_bev_mean", "q95_shift_bev"),
        ("scene_v9_q95_shift_img_mean", "q95_shift_img"),
        ("scene_v9_route_score_bev_mean", "route_score_bev"),
        ("scene_v9_route_score_img_mean", "route_score_img"),
        ("scene_v9_velocity_l1_mean", "velocity_l1"),
        ("scene_v9_fused_keep_unit_mean", "fused_keep_unit"),
        ("scene_v9_fused_keep_unit_q05", "fused_keep_unit_q05"),
        ("scene_v9_fused_keep_unit_q50", "fused_keep_unit_q50"),
        ("scene_v9_fused_keep_unit_q95", "fused_keep_unit_q95"),
        ("scene_v9_fused_keep_at_min_ratio", "fused_keep_at_min_ratio"),
        ("scene_v9_current_keep_min", "current_keep_min"),
        ("scene_v9_current_lambda_gate", "current_lambda_gate"),
        ("scene_v9_current_lambda_route", "current_lambda_route"),
        ("scene_v9_tau_bev_current", "tau_bev"),
        ("scene_v9_tau_img_current", "tau_img"),
        ("scene_v9_clean_distance_threshold_bev", "clean_distance_threshold_bev"),
        ("scene_v9_clean_distance_threshold_img", "clean_distance_threshold_img"),
        ("scene_v9_clean_q95_z_bev", "clean_q95_z_bev"),
        ("scene_v9_clean_q95_z_img", "clean_q95_z_img"),
        ("scene_v9_clean_anchor_frozen", "clean_anchor_frozen"),
        ("scene_v9_motionblur_img_bce_mean", "motionblur_img_bce"),
        ("scene_v9_beams_bev_bce_mean", "beams_bev_bce"),
        ("scene_v9_points_bev_bce_mean", "points_bev_bce"),
        ("scene_v9_dark_img_bce_mean", "dark_img_bce"),
        ("scene_v9_brightness_img_bce_mean", "brightness_img_bce"),
        ("scene_v9_missingcamera_img_bce_mean", "missingcamera_img_bce"),
        ("scene_v9_fog_bev_bce_mean", "fog_bev_bce"),
        ("scene_v9_fog_img_bce_mean", "fog_img_bce"),
    ]:
        val = safe_float(sample.get(sample_key))
        if not math.isnan(val):
            partial["v9_scene_scalar_values"][condition][name].append(val)
            partial["v9_scene_scalar_values"][condition_with_sev][name].append(val)


def _aggregate_prediction_record(partial, pred, condition, condition_with_sev):
    if any(key.startswith("v6_") for key in pred.keys()):
        partial["has_v6_schema"] = True
    if any(key in pred for key in ("v7_switch_triggered", "v7_damage_state", "v7_center_source")):
        partial["has_v7_schema"] = True
    if any(key.startswith("v8_") for key in pred.keys()):
        partial["has_v8_schema"] = True
    if any(key.startswith("v9_") for key in pred.keys()):
        partial["has_v9_schema"] = True
    selected = pred.get("selected_modality")
    if selected in MODALITIES:
        partial["selected_counts"][condition][selected] += 1
        partial["selected_counts"][condition_with_sev][selected] += 1

    dist = safe_float(pred.get("distance_to_ego"))
    unc = safe_float(pred.get("combined_unc", pred.get("center_unc")))
    if not math.isnan(dist) and not math.isnan(unc):
        partial["scatter_points"][condition].add((dist, unc))
        partial["scatter_points"][condition_with_sev].add((dist, unc))

    v5_switch = pred.get("v5_switch_triggered")
    if isinstance(v5_switch, bool):
        partial["v5_switch_stats"][condition]["total"] += 1
        partial["v5_switch_stats"][condition_with_sev]["total"] += 1
        if v5_switch:
            partial["v5_switch_stats"][condition]["switched"] += 1
            partial["v5_switch_stats"][condition_with_sev]["switched"] += 1

    v5_anom = safe_float(pred.get("fused_anomaly_score"))
    if not math.isnan(v5_anom):
        partial["v5_fused_anomaly_hists"][condition].add(v5_anom)
        partial["v5_fused_anomaly_hists"][condition_with_sev].add(v5_anom)

    if isinstance(v5_switch, bool) and v5_switch and not math.isnan(dist):
        partial["v5_switch_distance_hists"][condition].add(dist)
        partial["v5_switch_distance_hists"][condition_with_sev].add(dist)
        label_name = pred.get("label_name", "unknown")
        partial["v5_switch_class_counts"][condition][label_name] += 1
        partial["v5_switch_class_counts"][condition_with_sev][label_name] += 1
        selected_modality = pred.get("v5_selected_modality")
        if selected_modality in MODALITIES:
            partial["v5_selected_counts"][condition][selected_modality] += 1
            partial["v5_selected_counts"][condition_with_sev][selected_modality] += 1

    fallback_name = pred.get("v5_reference_fallback_name")
    if fallback_name:
        partial["v5_fallback_level_counts"][condition][fallback_name] += 1
        partial["v5_fallback_level_counts"][condition_with_sev][fallback_name] += 1

    v6_switch = pred.get("v6_switch_triggered")
    if isinstance(v6_switch, bool):
        partial["v6_switch_stats"][condition]["total"] += 1
        partial["v6_switch_stats"][condition_with_sev]["total"] += 1
        if v6_switch:
            partial["v6_switch_stats"][condition]["switched"] += 1
            partial["v6_switch_stats"][condition_with_sev]["switched"] += 1

    v6_anom = safe_float(pred.get("v6_gate_anomaly_score"))
    if not math.isnan(v6_anom):
        partial["v6_gate_anomaly_hists"][condition].add(v6_anom)
        partial["v6_gate_anomaly_hists"][condition_with_sev].add(v6_anom)

    v6_candidate = pred.get("v6_candidate_modality")
    if v6_candidate:
        partial["v6_candidate_counts"][condition][v6_candidate] += 1
        partial["v6_candidate_counts"][condition_with_sev][v6_candidate] += 1

    v6_center_source = pred.get("v6_center_source")
    if v6_center_source:
        partial["v6_center_source_counts"][condition][v6_center_source] += 1
        partial["v6_center_source_counts"][condition_with_sev][v6_center_source] += 1

    v6_height_source = pred.get("v6_height_source")
    if v6_height_source:
        partial["v6_height_source_counts"][condition][v6_height_source] += 1
        partial["v6_height_source_counts"][condition_with_sev][v6_height_source] += 1

    if isinstance(v6_switch, bool) and v6_switch and not math.isnan(dist):
        partial["v6_switch_distance_hists"][condition].add(dist)
        partial["v6_switch_distance_hists"][condition_with_sev].add(dist)
        label_name = pred.get("label_name", "unknown")
        partial["v6_switch_class_counts"][condition][label_name] += 1
        partial["v6_switch_class_counts"][condition_with_sev][label_name] += 1

    v6_fallback_name = pred.get("v6_reference_fallback_name")
    if v6_fallback_name:
        partial["v6_fallback_level_counts"][condition][v6_fallback_name] += 1
        partial["v6_fallback_level_counts"][condition_with_sev][v6_fallback_name] += 1

    v7_switch = pred.get("v7_switch_triggered")
    if isinstance(v7_switch, bool):
        partial["v7_switch_stats"][condition]["total"] += 1
        partial["v7_switch_stats"][condition_with_sev]["total"] += 1
        if v7_switch:
            partial["v7_switch_stats"][condition]["switched"] += 1
            partial["v7_switch_stats"][condition_with_sev]["switched"] += 1

    v7_bev_anom = safe_float(pred.get("v7_bev_center_anom"))
    if not math.isnan(v7_bev_anom):
        partial["v7_bev_center_anomaly_hists"][condition].add(v7_bev_anom)
        partial["v7_bev_center_anomaly_hists"][condition_with_sev].add(v7_bev_anom)
    v7_img_anom = safe_float(pred.get("v7_img_center_anom"))
    if not math.isnan(v7_img_anom):
        partial["v7_img_center_anomaly_hists"][condition].add(v7_img_anom)
        partial["v7_img_center_anomaly_hists"][condition_with_sev].add(v7_img_anom)

    v7_damage_state = pred.get("v7_damage_state")
    if v7_damage_state:
        partial["v7_damage_state_counts"][condition][v7_damage_state] += 1
        partial["v7_damage_state_counts"][condition_with_sev][v7_damage_state] += 1

    v7_center_source = pred.get("v7_center_source")
    if v7_center_source:
        partial["v7_center_source_counts"][condition][v7_center_source] += 1
        partial["v7_center_source_counts"][condition_with_sev][v7_center_source] += 1

    if isinstance(v7_switch, bool) and v7_switch and not math.isnan(dist):
        partial["v7_switch_distance_hists"][condition].add(dist)
        partial["v7_switch_distance_hists"][condition_with_sev].add(dist)
        label_name = pred.get("label_name", "unknown")
        partial["v7_switch_class_counts"][condition][label_name] += 1
        partial["v7_switch_class_counts"][condition_with_sev][label_name] += 1

    for key, store_key in [
        ("v8_fused_keep", "v8_fused_keep_values"),
        ("v8_center_fused_keep", "v8_center_fused_keep_values"),
        ("v8_damage_gate", "v8_damage_gate_values"),
        ("v8_center_route_allowed", "v8_center_route_allowed_values"),
    ]:
        val = safe_float(pred.get(key))
        if not math.isnan(val):
            partial[store_key][condition].append(val)
            partial[store_key][condition_with_sev].append(val)
    high_damage = pred.get("v8_high_damage_query")
    if isinstance(high_damage, bool):
        partial["v8_high_damage_values"][condition].append(float(high_damage))
        partial["v8_high_damage_values"][condition_with_sev].append(float(high_damage))
    for key, store_key in [
        ("v8_bev_damage", "v8_bev_damage_hists"),
        ("v8_img_damage", "v8_img_damage_hists"),
        ("v8_dual_damage", "v8_dual_damage_hists"),
        ("v8_a4_bev_damage_z", "v8_a4_bev_damage_z_hists"),
        ("v8_a4_img_damage_z", "v8_a4_img_damage_z_hists"),
        ("v8_a4_cls_unc_z_bev", "v8_a4_cls_unc_z_bev_hists"),
        ("v8_a4_cls_unc_z_img", "v8_a4_cls_unc_z_img_hists"),
        ("v8_a4_route_score_bev", "v8_a4_route_score_bev_hists"),
        ("v8_a4_route_score_img", "v8_a4_route_score_img_hists"),
    ]:
        val = safe_float(pred.get(key))
        if not math.isnan(val):
            partial[store_key][condition].add(val)
            partial[store_key][condition_with_sev].add(val)
    v8_state = pred.get("v8_damage_state")
    if v8_state:
        partial["v8_damage_state_counts"][condition][v8_state] += 1
        partial["v8_damage_state_counts"][condition_with_sev][v8_state] += 1
    v8_rel_state = pred.get("v8_reliability_state")
    if v8_rel_state:
        partial["v8_reliability_state_counts"][condition][v8_rel_state] += 1
        partial["v8_reliability_state_counts"][condition_with_sev][v8_rel_state] += 1
    v8_source = pred.get("v8_center_source")
    if v8_source:
        partial["v8_selected_source_counts"][condition][v8_source] += 1
        partial["v8_selected_source_counts"][condition_with_sev][v8_source] += 1
    v8_choice = pred.get("v8_non_fused_choice")
    if v8_choice:
        partial["v8_non_fused_choice_counts"][condition][v8_choice] += 1
        partial["v8_non_fused_choice_counts"][condition_with_sev][v8_choice] += 1
        lidar_domain = {"beamsreducing", "pointsreducing", "spatialmisalignment", "temporalmisalignment"}
        image_domain = {"brightness", "dark", "motionblur", "missingcamera"}
        if condition in lidar_domain:
            correct = float(v8_choice == "img")
            partial["v8_domain_correct_values"][condition].append(correct)
            partial["v8_domain_correct_values"][condition_with_sev].append(correct)
        elif condition in image_domain:
            correct = float(v8_choice == "bev")
            partial["v8_domain_correct_values"][condition].append(correct)
            partial["v8_domain_correct_values"][condition_with_sev].append(correct)
    for pred_key, modality in [
        ("v8_bev_reliability", "bev"),
        ("v8_img_reliability", "img"),
    ]:
        val = safe_float(pred.get(pred_key))
        if not math.isnan(val):
            partial["v8_reliability_values"][condition][modality].append(val)
            partial["v8_reliability_values"][condition_with_sev][modality].append(val)
    val = safe_float(pred.get("v8_reliability_margin"))
    if not math.isnan(val):
        partial["v8_reliability_margin_values"][condition].append(val)
        partial["v8_reliability_margin_values"][condition_with_sev].append(val)
    for pred_key, modality in [
        ("v8_context_bev_damage", "bev"),
        ("v8_context_img_damage", "img"),
    ]:
        val = safe_float(pred.get(pred_key))
        if not math.isnan(val):
            partial["v8_context_damage_values"][condition][modality].append(val)
            partial["v8_context_damage_values"][condition_with_sev][modality].append(val)
    for pred_key, modality in [
        ("v8_soft_weight_fused", "fused"),
        ("v8_soft_weight_bev", "bev"),
        ("v8_soft_weight_img", "img"),
    ]:
        val = safe_float(pred.get(pred_key))
        if not math.isnan(val):
            partial["v8_soft_weight_values"][condition][modality].append(val)
            partial["v8_soft_weight_values"][condition_with_sev][modality].append(val)
    for pred_key, name in [
        ("v8_output_delta_cls", "cls"),
        ("v8_output_delta_center", "center"),
        ("v8_output_delta_height", "height"),
    ]:
        val = safe_float(pred.get(pred_key))
        if not math.isnan(val):
            partial["v8_output_delta_values"][condition][name].append(val)
            partial["v8_output_delta_values"][condition_with_sev][name].append(val)
    for key, store_key in [
        ("v9_a_bev", "v9_a_bev_hists"),
        ("v9_a_img", "v9_a_img_hists"),
        ("v9_anchor_distance_bev", "v9_anchor_distance_bev_hists"),
        ("v9_anchor_distance_img", "v9_anchor_distance_img_hists"),
        ("v9_tail_prob_bev", "v9_tail_prob_bev_hists"),
        ("v9_tail_prob_img", "v9_tail_prob_img_hists"),
        ("v9_tail_excess_bev", "v9_tail_excess_bev_hists"),
        ("v9_tail_excess_img", "v9_tail_excess_img_hists"),
        ("v9_q95_shift_bev", "v9_q95_shift_bev_hists"),
        ("v9_q95_shift_img", "v9_q95_shift_img_hists"),
        ("v9_route_score_bev", "v9_route_score_bev_hists"),
        ("v9_route_score_img", "v9_route_score_img_hists"),
        ("v9_velocity_l1", "v9_velocity_l1_hists"),
    ]:
        val = safe_float(pred.get(key))
        if not math.isnan(val):
            partial[store_key][condition].add(val)
            partial[store_key][condition_with_sev].add(val)
    for key, modality in [
        ("v9_a_bev_above_tau", "bev"),
        ("v9_a_img_above_tau", "img"),
    ]:
        val = safe_float(pred.get(key))
        if not math.isnan(val):
            partial["v9_above_tau_values"][condition][modality].append(val)
            partial["v9_above_tau_values"][condition_with_sev][modality].append(val)
    val = safe_float(pred.get("v9_clean_like_mask"))
    if not math.isnan(val):
        partial["v9_clean_like_values"][condition].append(val)
        partial["v9_clean_like_values"][condition_with_sev].append(val)


def _aggregate_single_sidecar(row, progress_queue=None, heartbeat_samples=250):
    paths = row_sidecar_paths(row)
    if not row_has_sidecar(row):
        return None

    t0 = time.time()
    condition = row["corruption"]
    split = row["split"]
    severity = row.get("severity", "")
    condition_with_sev = "clean" if condition == "clean" else f"{condition}-s{severity}"
    partial = _init_partial_stores()
    sample_count = 0
    pred_count = 0
    last_report_preds = 0
    representative_sample = None
    current_token = None
    current_points = []
    current_pred_count = 0

    if progress_queue is not None:
        progress_queue.put({
            "kind": "start",
            "condition_with_sev": condition_with_sev,
            "log_size_mb": row_log_size_mb(row),
            "source": "sidecar",
        })

    for sample in iter_jsonl(paths["scene"]):
        sample_count += 1
        _aggregate_scene_record(partial, sample, condition, condition_with_sev)

    for pred in iter_jsonl(paths["prediction"]):
        pred_count += 1
        _aggregate_prediction_record(partial, pred, condition, condition_with_sev)

        sample_token = pred.get("sample_token", "unknown")
        if current_token is None:
            current_token = sample_token
        elif sample_token != current_token:
            sample_info = {
                "split": split,
                "condition": condition,
                "condition_with_sev": condition_with_sev,
                "sample_token": current_token,
                "pred_count": current_pred_count,
                "points": current_points,
            }
            if representative_sample is None or sample_info["pred_count"] > representative_sample["pred_count"]:
                representative_sample = sample_info
            current_token = sample_token
            current_points = []
            current_pred_count = 0

        current_pred_count += 1
        center = pred.get("bbox_center")
        if isinstance(center, list) and len(center) >= 2:
            x = safe_float(center[0])
            y = safe_float(center[1])
            if not math.isnan(x) and not math.isnan(y):
                current_points.append({
                    "x": x,
                    "y": y,
                    "selected_modality": pred.get("selected_modality", "fused"),
                    "combined_unc": safe_float(pred.get("combined_unc", pred.get("center_unc")), default=0.1),
                })

        if progress_queue is not None and (pred_count - last_report_preds) >= heartbeat_samples:
            progress_queue.put({
                "kind": "heartbeat",
                "condition_with_sev": condition_with_sev,
                "sample_count": sample_count,
                "pred_count": pred_count,
                "elapsed_sec": time.time() - t0,
                "source": "sidecar",
            })
            last_report_preds = pred_count

    if current_token is not None:
        sample_info = {
            "split": split,
            "condition": condition,
            "condition_with_sev": condition_with_sev,
            "sample_token": current_token,
            "pred_count": current_pred_count,
            "points": current_points,
        }
        if representative_sample is None or sample_info["pred_count"] > representative_sample["pred_count"]:
            representative_sample = sample_info

    meta = {
        "condition": condition,
        "condition_with_sev": condition_with_sev,
        "split": split,
        "severity": severity,
        "sample_count": sample_count,
        "pred_count": pred_count,
        "elapsed_sec": time.time() - t0,
        "log_json": paths["prediction"],
        "log_size_mb": row_log_size_mb(row),
        "source": "sidecar",
    }
    return _finalize_partial(partial, representative_sample, meta)


def _aggregate_single_log(row, progress_queue=None, heartbeat_samples=250):
    log_json = row.get("edl_log_json")
    if not log_json or not os.path.isfile(log_json):
        return None

    t0 = time.time()
    condition = row["corruption"]
    split = row["split"]
    severity = row.get("severity", "")
    condition_with_sev = "clean" if condition == "clean" else f"{condition}-s{severity}"

    partial = _init_partial_stores()
    representative_sample = None
    sample_count = 0
    pred_count = 0
    last_report_samples = 0

    if progress_queue is not None:
        progress_queue.put({
            "kind": "start",
            "condition_with_sev": condition_with_sev,
            "log_size_mb": (os.path.getsize(log_json) / (1024.0 * 1024.0)),
            "source": "raw_json",
        })

    is_jsonl_stream = log_json.endswith(".jsonl") or ".jsonl." in log_json
    source_iter = iter_jsonl(log_json) if is_jsonl_stream else iter_json_array(log_json)
    for sample in source_iter:
        sample_count += 1
        _aggregate_scene_record(partial, sample, condition, condition_with_sev)

        preds = sample.get("predictions", [])
        pred_count += len(preds)
        if representative_sample is None or len(preds) > representative_sample["pred_count"]:
            representative_sample = compress_sample_for_qualitative(sample, split, condition, condition_with_sev)

        if progress_queue is not None and (sample_count - last_report_samples) >= heartbeat_samples:
            progress_queue.put({
                "kind": "heartbeat",
                "condition_with_sev": condition_with_sev,
                "sample_count": sample_count,
                "pred_count": pred_count,
                "elapsed_sec": time.time() - t0,
                "source": "raw_json",
            })
            last_report_samples = sample_count

        for pred in preds:
            _aggregate_prediction_record(partial, pred, condition, condition_with_sev)

    meta = {
        "condition": condition,
        "condition_with_sev": condition_with_sev,
        "split": split,
        "severity": severity,
        "sample_count": sample_count,
        "pred_count": pred_count,
        "elapsed_sec": time.time() - t0,
        "log_json": log_json,
        "log_size_mb": (os.path.getsize(log_json) / (1024.0 * 1024.0)),
        "source": "raw_json",
    }
    return _finalize_partial(partial, representative_sample, meta)


def _aggregate_single_row(row, progress_queue=None, heartbeat_samples=250):
    if row_has_sidecar(row):
        return _aggregate_single_sidecar(row, progress_queue=progress_queue, heartbeat_samples=heartbeat_samples)
    return _aggregate_single_log(row, progress_queue=progress_queue, heartbeat_samples=heartbeat_samples)


def aggregate_by_condition(manifest_rows, jobs=1, heartbeat_samples=250):
    scene_weights = defaultdict(lambda: defaultdict(list))
    scene_uncs = defaultdict(lambda: defaultdict(list))
    selected_counts = defaultdict(lambda: defaultdict(int))
    entropy_values = defaultdict(list)
    fused_weight_values = defaultdict(list)
    distance_prior_top1 = defaultdict(lambda: defaultdict(list))
    residual_override_values = defaultdict(list)
    alpha_res_values = defaultdict(list)
    tail_query_values = defaultdict(lambda: defaultdict(list))
    scatter_points = defaultdict(lambda: ReservoirSampler(SCATTER_MAX_POINTS_PER_CONDITION))
    v5_switch_stats = defaultdict(lambda: {"switched": 0, "total": 0})
    v5_fused_anomaly_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v5_switch_distance_hists = defaultdict(lambda: HistogramAccumulator(DISTANCE_HIST_BINS))
    v5_switch_class_counts = defaultdict(lambda: defaultdict(int))
    v5_fallback_level_counts = defaultdict(lambda: defaultdict(int))
    v5_selected_counts = defaultdict(lambda: defaultdict(int))
    v6_switch_stats = defaultdict(lambda: {"switched": 0, "total": 0})
    v6_gate_anomaly_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v6_switch_distance_hists = defaultdict(lambda: HistogramAccumulator(DISTANCE_HIST_BINS))
    v6_switch_class_counts = defaultdict(lambda: defaultdict(int))
    v6_fallback_level_counts = defaultdict(lambda: defaultdict(int))
    v6_candidate_counts = defaultdict(lambda: defaultdict(int))
    v6_center_source_counts = defaultdict(lambda: defaultdict(int))
    v6_height_source_counts = defaultdict(lambda: defaultdict(int))
    v6_gate_pass_anomaly_values = defaultdict(list)
    v6_gate_pass_candidate_prob_values = defaultdict(list)
    v6_gate_pass_candidate_margin_values = defaultdict(list)
    v6_gate_pass_final_values = defaultdict(list)
    has_v6_schema = False
    v7_switch_stats = defaultdict(lambda: {"switched": 0, "total": 0})
    v7_bev_center_anomaly_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v7_img_center_anomaly_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v7_switch_distance_hists = defaultdict(lambda: HistogramAccumulator(DISTANCE_HIST_BINS))
    v7_switch_class_counts = defaultdict(lambda: defaultdict(int))
    v7_damage_state_counts = defaultdict(lambda: defaultdict(int))
    v7_center_source_counts = defaultdict(lambda: defaultdict(int))
    has_v7_schema = False
    v8_fused_keep_values = defaultdict(list)
    v8_center_fused_keep_values = defaultdict(list)
    v8_damage_gate_values = defaultdict(list)
    v8_high_damage_values = defaultdict(list)
    v8_center_route_allowed_values = defaultdict(list)
    v8_bev_damage_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v8_img_damage_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v8_dual_damage_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v8_context_damage_values = defaultdict(lambda: defaultdict(list))
    v8_a4_bev_damage_z_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v8_a4_img_damage_z_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v8_a4_cls_unc_z_bev_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v8_a4_cls_unc_z_img_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v8_a4_route_score_bev_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v8_a4_route_score_img_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v8_damage_state_counts = defaultdict(lambda: defaultdict(int))
    v8_reliability_state_counts = defaultdict(lambda: defaultdict(int))
    v8_selected_source_counts = defaultdict(lambda: defaultdict(int))
    v8_non_fused_choice_counts = defaultdict(lambda: defaultdict(int))
    v8_reliability_values = defaultdict(lambda: defaultdict(list))
    v8_reliability_margin_values = defaultdict(list)
    v8_soft_weight_values = defaultdict(lambda: defaultdict(list))
    v8_output_delta_values = defaultdict(lambda: defaultdict(list))
    v8_domain_correct_values = defaultdict(list)
    v9_a_bev_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v9_a_img_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v9_anchor_distance_bev_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v9_anchor_distance_img_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v9_tail_prob_bev_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v9_tail_prob_img_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v9_tail_excess_bev_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v9_tail_excess_img_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v9_q95_shift_bev_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v9_q95_shift_img_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v9_route_score_bev_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v9_route_score_img_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v9_velocity_l1_hists = defaultdict(lambda: HistogramAccumulator(ANOMALY_HIST_BINS))
    v9_above_tau_values = defaultdict(lambda: defaultdict(list))
    v9_clean_like_values = defaultdict(list)
    v9_scene_scalar_values = defaultdict(lambda: defaultdict(list))
    has_v8_schema = False
    has_v9_schema = False
    representative_samples = {}
    valid_rows = [
        row for row in manifest_rows
        if row_has_sidecar(row) or (row.get("edl_log_json") and os.path.isfile(row.get("edl_log_json")))
    ]
    valid_rows.sort(key=row_log_size_mb, reverse=True)

    def merge_partial(partial):
        if not partial:
            return
        nonlocal has_v6_schema, has_v7_schema, has_v8_schema, has_v9_schema
        has_v6_schema = has_v6_schema or bool(partial.get("has_v6_schema", False))
        has_v7_schema = has_v7_schema or bool(partial.get("has_v7_schema", False))
        has_v8_schema = has_v8_schema or bool(partial.get("has_v8_schema", False))
        has_v9_schema = has_v9_schema or bool(partial.get("has_v9_schema", False))

        for cond, modality_values in partial["scene_weights"].items():
            for modality, vals in modality_values.items():
                scene_weights[cond][modality].extend(vals)
        for cond, modality_values in partial["scene_uncs"].items():
            for modality, vals in modality_values.items():
                scene_uncs[cond][modality].extend(vals)
        for cond, modality_counts in partial["selected_counts"].items():
            for modality, count in modality_counts.items():
                selected_counts[cond][modality] += count
        for cond, vals in partial["entropy_values"].items():
            entropy_values[cond].extend(vals)
        for cond, vals in partial["fused_weight_values"].items():
            fused_weight_values[cond].extend(vals)
        for cond, modality_values in partial["distance_prior_top1"].items():
            for modality, vals in modality_values.items():
                distance_prior_top1[cond][modality].extend(vals)
        for cond, vals in partial["residual_override_values"].items():
            residual_override_values[cond].extend(vals)
        for cond, vals in partial["alpha_res_values"].items():
            alpha_res_values[cond].extend(vals)
        for cond, modality_values in partial["tail_query_values"].items():
            for modality, vals in modality_values.items():
                tail_query_values[cond][modality].extend(vals)
        for cond, payload in partial["scatter_points"].items():
            scatter_points[cond].merge_items(payload["items"])
        for cond, stats in partial["v5_switch_stats"].items():
            v5_switch_stats[cond]["switched"] += stats["switched"]
            v5_switch_stats[cond]["total"] += stats["total"]
        for cond, payload in partial["v5_fused_anomaly_hists"].items():
            _merge_hist_payload(v5_fused_anomaly_hists[cond], payload)
        for cond, payload in partial["v5_switch_distance_hists"].items():
            _merge_hist_payload(v5_switch_distance_hists[cond], payload)
        for cond, counts in partial["v5_switch_class_counts"].items():
            for class_name, count in counts.items():
                v5_switch_class_counts[cond][class_name] += count
        for cond, counts in partial["v5_fallback_level_counts"].items():
            for level, count in counts.items():
                v5_fallback_level_counts[cond][level] += count
        for cond, counts in partial["v5_selected_counts"].items():
            for modality, count in counts.items():
                v5_selected_counts[cond][modality] += count
        for cond, stats in partial["v6_switch_stats"].items():
            v6_switch_stats[cond]["switched"] += stats["switched"]
            v6_switch_stats[cond]["total"] += stats["total"]
        for cond, payload in partial["v6_gate_anomaly_hists"].items():
            _merge_hist_payload(v6_gate_anomaly_hists[cond], payload)
        for cond, payload in partial["v6_switch_distance_hists"].items():
            _merge_hist_payload(v6_switch_distance_hists[cond], payload)
        for cond, counts in partial["v6_switch_class_counts"].items():
            for class_name, count in counts.items():
                v6_switch_class_counts[cond][class_name] += count
        for cond, counts in partial["v6_fallback_level_counts"].items():
            for level, count in counts.items():
                v6_fallback_level_counts[cond][level] += count
        for cond, counts in partial["v6_candidate_counts"].items():
            for modality, count in counts.items():
                v6_candidate_counts[cond][modality] += count
        for cond, counts in partial["v6_center_source_counts"].items():
            for modality, count in counts.items():
                v6_center_source_counts[cond][modality] += count
        for cond, counts in partial["v6_height_source_counts"].items():
            for modality, count in counts.items():
                v6_height_source_counts[cond][modality] += count
        for cond, values in partial["v6_gate_pass_anomaly_values"].items():
            v6_gate_pass_anomaly_values[cond].extend(values)
        for cond, values in partial["v6_gate_pass_candidate_prob_values"].items():
            v6_gate_pass_candidate_prob_values[cond].extend(values)
        for cond, values in partial["v6_gate_pass_candidate_margin_values"].items():
            v6_gate_pass_candidate_margin_values[cond].extend(values)
        for cond, values in partial["v6_gate_pass_final_values"].items():
            v6_gate_pass_final_values[cond].extend(values)
        for cond, stats in partial["v7_switch_stats"].items():
            v7_switch_stats[cond]["switched"] += stats["switched"]
            v7_switch_stats[cond]["total"] += stats["total"]
        for cond, payload in partial["v7_bev_center_anomaly_hists"].items():
            _merge_hist_payload(v7_bev_center_anomaly_hists[cond], payload)
        for cond, payload in partial["v7_img_center_anomaly_hists"].items():
            _merge_hist_payload(v7_img_center_anomaly_hists[cond], payload)
        for cond, payload in partial["v7_switch_distance_hists"].items():
            _merge_hist_payload(v7_switch_distance_hists[cond], payload)
        for cond, counts in partial["v7_switch_class_counts"].items():
            for class_name, count in counts.items():
                v7_switch_class_counts[cond][class_name] += count
        for cond, counts in partial["v7_damage_state_counts"].items():
            for state, count in counts.items():
                v7_damage_state_counts[cond][state] += count
        for cond, counts in partial["v7_center_source_counts"].items():
            for source, count in counts.items():
                v7_center_source_counts[cond][source] += count
        for cond, vals in partial["v8_fused_keep_values"].items():
            v8_fused_keep_values[cond].extend(vals)
        for cond, vals in partial["v8_center_fused_keep_values"].items():
            v8_center_fused_keep_values[cond].extend(vals)
        for cond, vals in partial["v8_damage_gate_values"].items():
            v8_damage_gate_values[cond].extend(vals)
        for cond, vals in partial["v8_high_damage_values"].items():
            v8_high_damage_values[cond].extend(vals)
        for cond, vals in partial["v8_center_route_allowed_values"].items():
            v8_center_route_allowed_values[cond].extend(vals)
        for cond, payload in partial["v8_bev_damage_hists"].items():
            _merge_hist_payload(v8_bev_damage_hists[cond], payload)
        for cond, payload in partial["v8_img_damage_hists"].items():
            _merge_hist_payload(v8_img_damage_hists[cond], payload)
        for cond, payload in partial["v8_dual_damage_hists"].items():
            _merge_hist_payload(v8_dual_damage_hists[cond], payload)
        for cond, payload in partial.get("v8_a4_bev_damage_z_hists", {}).items():
            _merge_hist_payload(v8_a4_bev_damage_z_hists[cond], payload)
        for cond, payload in partial.get("v8_a4_img_damage_z_hists", {}).items():
            _merge_hist_payload(v8_a4_img_damage_z_hists[cond], payload)
        for cond, payload in partial.get("v8_a4_cls_unc_z_bev_hists", {}).items():
            _merge_hist_payload(v8_a4_cls_unc_z_bev_hists[cond], payload)
        for cond, payload in partial.get("v8_a4_cls_unc_z_img_hists", {}).items():
            _merge_hist_payload(v8_a4_cls_unc_z_img_hists[cond], payload)
        for cond, payload in partial.get("v8_a4_route_score_bev_hists", {}).items():
            _merge_hist_payload(v8_a4_route_score_bev_hists[cond], payload)
        for cond, payload in partial.get("v8_a4_route_score_img_hists", {}).items():
            _merge_hist_payload(v8_a4_route_score_img_hists[cond], payload)
        for cond, modality_values in partial["v8_context_damage_values"].items():
            for modality, vals in modality_values.items():
                v8_context_damage_values[cond][modality].extend(vals)
        for cond, counts in partial["v8_damage_state_counts"].items():
            for state, count in counts.items():
                v8_damage_state_counts[cond][state] += count
        for cond, counts in partial["v8_reliability_state_counts"].items():
            for state, count in counts.items():
                v8_reliability_state_counts[cond][state] += count
        for cond, counts in partial["v8_selected_source_counts"].items():
            for source, count in counts.items():
                v8_selected_source_counts[cond][source] += count
        for cond, counts in partial["v8_non_fused_choice_counts"].items():
            for source, count in counts.items():
                v8_non_fused_choice_counts[cond][source] += count
        for cond, modality_values in partial["v8_reliability_values"].items():
            for modality, vals in modality_values.items():
                v8_reliability_values[cond][modality].extend(vals)
        for cond, vals in partial["v8_reliability_margin_values"].items():
            v8_reliability_margin_values[cond].extend(vals)
        for cond, modality_values in partial["v8_soft_weight_values"].items():
            for modality, vals in modality_values.items():
                v8_soft_weight_values[cond][modality].extend(vals)
        for cond, delta_values in partial["v8_output_delta_values"].items():
            for name, vals in delta_values.items():
                v8_output_delta_values[cond][name].extend(vals)
        for cond, vals in partial["v8_domain_correct_values"].items():
            v8_domain_correct_values[cond].extend(vals)
        for key, store in [
            ("v9_a_bev_hists", v9_a_bev_hists),
            ("v9_a_img_hists", v9_a_img_hists),
            ("v9_anchor_distance_bev_hists", v9_anchor_distance_bev_hists),
            ("v9_anchor_distance_img_hists", v9_anchor_distance_img_hists),
            ("v9_tail_prob_bev_hists", v9_tail_prob_bev_hists),
            ("v9_tail_prob_img_hists", v9_tail_prob_img_hists),
            ("v9_tail_excess_bev_hists", v9_tail_excess_bev_hists),
            ("v9_tail_excess_img_hists", v9_tail_excess_img_hists),
            ("v9_q95_shift_bev_hists", v9_q95_shift_bev_hists),
            ("v9_q95_shift_img_hists", v9_q95_shift_img_hists),
            ("v9_route_score_bev_hists", v9_route_score_bev_hists),
            ("v9_route_score_img_hists", v9_route_score_img_hists),
            ("v9_velocity_l1_hists", v9_velocity_l1_hists),
        ]:
            for cond, payload in partial.get(key, {}).items():
                _merge_hist_payload(store[cond], payload)
        for cond, values in partial.get("v9_above_tau_values", {}).items():
            for modality, vals in values.items():
                v9_above_tau_values[cond][modality].extend(vals)
        for cond, vals in partial.get("v9_clean_like_values", {}).items():
            v9_clean_like_values[cond].extend(vals)
        for cond, values in partial.get("v9_scene_scalar_values", {}).items():
            for name, vals in values.items():
                v9_scene_scalar_values[cond][name].extend(vals)

        rep = partial.get("representative_sample")
        if rep:
            rep_key = rep["condition"]
            if rep_key not in representative_samples or rep["pred_count"] > representative_samples[rep_key]["pred_count"]:
                representative_samples[rep_key] = rep

    total_rows = len(valid_rows)
    print(
        f"[EDL Viz] Aggregating {total_rows} log files with jobs={jobs}, "
        f"heartbeat_samples={heartbeat_samples}",
        flush=True)
    if jobs <= 1:
        for row_idx, row in enumerate(progress_iter(valid_rows, total=total_rows, desc="EDL Viz files"), start=1):
            partial = _aggregate_single_row(row, progress_queue=None, heartbeat_samples=heartbeat_samples)
            merge_partial(partial)
            meta = partial["meta"]
            msg = (
                f"[EDL Viz] Done {row_idx}/{total_rows}: {meta['condition_with_sev']} | "
                f"samples={meta['sample_count']}, preds={meta['pred_count']}, "
                f"size={meta['log_size_mb']:.1f}MB, src={meta.get('source', 'unknown')}, t={meta['elapsed_sec']:.1f}s"
            )
            if tqdm is None:
                print(msg, flush=True)
            else:
                tqdm.write(msg, file=sys.stdout)
    else:
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        heartbeat_state = {}
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
                pending = {
                    executor.submit(_aggregate_single_row, row, progress_queue, heartbeat_samples): row
                    for row in valid_rows
                }
                done_bar = progress_iter(range(total_rows), total=total_rows, desc="EDL Viz files done")
                completed = 0

                def emit_heartbeat_messages():
                    updated = False
                    while True:
                        try:
                            item = progress_queue.get_nowait()
                        except Exception:
                            break
                        kind = item.get("kind")
                        condition_key = item.get("condition_with_sev", "unknown")
                        state = heartbeat_state.setdefault(condition_key, {})
                        state.update(item)
                        updated = True
                        if kind == "start":
                            msg = (
                                f"[EDL Viz] Start {condition_key} | "
                                f"size={item.get('log_size_mb', 0.0):.1f}MB, src={item.get('source', 'unknown')}"
                            )
                            if tqdm is None:
                                print(msg, flush=True)
                            else:
                                tqdm.write(msg, file=sys.stdout)
                    if updated and heartbeat_state:
                        ranked = sorted(
                            heartbeat_state.items(),
                            key=lambda kv: kv[1].get("sample_count", 0),
                            reverse=True,
                        )
                        summary = " | ".join(
                            f"{cond}:s={state.get('sample_count', 0)},p={state.get('pred_count', 0)},t={state.get('elapsed_sec', 0.0):.0f}s"
                            for cond, state in ranked[:3]
                        )
                        if tqdm is not None and summary:
                            done_bar.set_postfix_str(summary[:140], refresh=False)

                while pending:
                    done, not_done = concurrent.futures.wait(
                        pending.keys(),
                        timeout=0.5,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    emit_heartbeat_messages()
                    for future in done:
                        partial = future.result()
                        completed += 1
                        merge_partial(partial)
                        meta = partial["meta"]
                        done_bar.update(1)
                        heartbeat_state.pop(meta["condition_with_sev"], None)
                        msg = (
                            f"[EDL Viz] Done {completed}/{total_rows}: {meta['condition_with_sev']} | "
                            f"samples={meta['sample_count']}, preds={meta['pred_count']}, "
                            f"size={meta['log_size_mb']:.1f}MB, src={meta.get('source', 'unknown')}, t={meta['elapsed_sec']:.1f}s"
                        )
                        if tqdm is None:
                            print(msg, flush=True)
                        else:
                            tqdm.write(msg, file=sys.stdout)
                        pending.pop(future, None)
                emit_heartbeat_messages()
                if tqdm is not None:
                    done_bar.close()
        finally:
            manager.shutdown()

    return {
        "scene_weights": scene_weights,
        "scene_uncs": scene_uncs,
        "selected_counts": selected_counts,
        "entropy_values": entropy_values,
        "fused_weight_values": fused_weight_values,
        "distance_prior_top1": distance_prior_top1,
        "residual_override_values": residual_override_values,
        "alpha_res_values": alpha_res_values,
        "tail_query_values": tail_query_values,
        "scatter_points": {k: v.items for k, v in scatter_points.items()},
        "v5_switch_stats": v5_switch_stats,
        "v5_fused_anomaly_hists": v5_fused_anomaly_hists,
        "v5_switch_distance_hists": v5_switch_distance_hists,
        "v5_switch_class_counts": v5_switch_class_counts,
        "v5_fallback_level_counts": v5_fallback_level_counts,
        "v5_selected_counts": v5_selected_counts,
        "v6_switch_stats": v6_switch_stats,
        "v6_gate_anomaly_hists": v6_gate_anomaly_hists,
        "v6_switch_distance_hists": v6_switch_distance_hists,
        "v6_switch_class_counts": v6_switch_class_counts,
        "v6_fallback_level_counts": v6_fallback_level_counts,
        "v6_candidate_counts": v6_candidate_counts,
        "v6_center_source_counts": v6_center_source_counts,
        "v6_height_source_counts": v6_height_source_counts,
        "v6_gate_pass_anomaly_values": v6_gate_pass_anomaly_values,
        "v6_gate_pass_candidate_prob_values": v6_gate_pass_candidate_prob_values,
        "v6_gate_pass_candidate_margin_values": v6_gate_pass_candidate_margin_values,
        "v6_gate_pass_final_values": v6_gate_pass_final_values,
        "has_v6_schema": has_v6_schema,
        "v7_switch_stats": v7_switch_stats,
        "v7_bev_center_anomaly_hists": v7_bev_center_anomaly_hists,
        "v7_img_center_anomaly_hists": v7_img_center_anomaly_hists,
        "v7_switch_distance_hists": v7_switch_distance_hists,
        "v7_switch_class_counts": v7_switch_class_counts,
        "v7_damage_state_counts": v7_damage_state_counts,
        "v7_center_source_counts": v7_center_source_counts,
        "has_v7_schema": has_v7_schema,
        "v8_fused_keep_values": v8_fused_keep_values,
        "v8_center_fused_keep_values": v8_center_fused_keep_values,
        "v8_damage_gate_values": v8_damage_gate_values,
        "v8_high_damage_values": v8_high_damage_values,
        "v8_center_route_allowed_values": v8_center_route_allowed_values,
        "v8_bev_damage_hists": v8_bev_damage_hists,
        "v8_img_damage_hists": v8_img_damage_hists,
        "v8_dual_damage_hists": v8_dual_damage_hists,
        "v8_context_damage_values": v8_context_damage_values,
        "v8_a4_bev_damage_z_hists": v8_a4_bev_damage_z_hists,
        "v8_a4_img_damage_z_hists": v8_a4_img_damage_z_hists,
        "v8_a4_cls_unc_z_bev_hists": v8_a4_cls_unc_z_bev_hists,
        "v8_a4_cls_unc_z_img_hists": v8_a4_cls_unc_z_img_hists,
        "v8_a4_route_score_bev_hists": v8_a4_route_score_bev_hists,
        "v8_a4_route_score_img_hists": v8_a4_route_score_img_hists,
        "v8_damage_state_counts": v8_damage_state_counts,
        "v8_reliability_state_counts": v8_reliability_state_counts,
        "v8_selected_source_counts": v8_selected_source_counts,
        "v8_non_fused_choice_counts": v8_non_fused_choice_counts,
        "v8_reliability_values": v8_reliability_values,
        "v8_reliability_margin_values": v8_reliability_margin_values,
        "v8_soft_weight_values": v8_soft_weight_values,
        "v8_output_delta_values": v8_output_delta_values,
        "v8_domain_correct_values": v8_domain_correct_values,
        "has_v8_schema": has_v8_schema,
        "v9_a_bev_hists": v9_a_bev_hists,
        "v9_a_img_hists": v9_a_img_hists,
        "v9_anchor_distance_bev_hists": v9_anchor_distance_bev_hists,
        "v9_anchor_distance_img_hists": v9_anchor_distance_img_hists,
        "v9_tail_prob_bev_hists": v9_tail_prob_bev_hists,
        "v9_tail_prob_img_hists": v9_tail_prob_img_hists,
        "v9_tail_excess_bev_hists": v9_tail_excess_bev_hists,
        "v9_tail_excess_img_hists": v9_tail_excess_img_hists,
        "v9_q95_shift_bev_hists": v9_q95_shift_bev_hists,
        "v9_q95_shift_img_hists": v9_q95_shift_img_hists,
        "v9_route_score_bev_hists": v9_route_score_bev_hists,
        "v9_route_score_img_hists": v9_route_score_img_hists,
        "v9_velocity_l1_hists": v9_velocity_l1_hists,
        "v9_above_tau_values": v9_above_tau_values,
        "v9_clean_like_values": v9_clean_like_values,
        "v9_scene_scalar_values": v9_scene_scalar_values,
        "has_v9_schema": has_v9_schema,
        "representative_samples": representative_samples,
    }


def ordered_conditions(scene_weights):
    keys = list(scene_weights.keys())
    clean = [k for k in keys if k == "clean"]
    others = sorted(k for k in keys if k != "clean" and "-s" not in k)
    return clean + others


def ordered_conditions_with_severity(scene_weights, severity):
    keys = list(scene_weights.keys())
    clean = [k for k in keys if k == "clean"]
    suffix = f"-s{severity}"
    others = sorted(k for k in keys if k.endswith(suffix))
    return clean + others


def plot_stacked_weights(scene_weights, out_path):
    conditions = ordered_conditions(scene_weights)
    if not conditions:
        return

    x = np.arange(len(conditions))
    bottoms = np.zeros(len(conditions))

    plt.figure(figsize=(max(10, len(conditions) * 1.2), 5))
    for modality in MODALITIES:
        vals = [
            np.mean(scene_weights[cond][modality]) if scene_weights[cond][modality] else 0.0
            for cond in conditions
        ]
        plt.bar(x, vals, bottom=bottoms, label=modality, color=MODALITY_COLORS[modality])
        bottoms += np.array(vals)

    plt.xticks(x, conditions, rotation=30, ha="right")
    plt.ylabel("Average Modality Weight")
    plt.title("Modality Weights by Corruption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_selected_ratio(selected_counts, out_path):
    conditions = ordered_conditions(selected_counts)
    if not conditions:
        return

    x = np.arange(len(conditions))
    bottoms = np.zeros(len(conditions))
    plt.figure(figsize=(max(10, len(conditions) * 1.2), 5))

    for modality in MODALITIES:
        vals = []
        for cond in conditions:
            total = sum(selected_counts[cond][m] for m in MODALITIES)
            ratio = selected_counts[cond][modality] / total if total > 0 else 0.0
            vals.append(ratio)
        plt.bar(x, vals, bottom=bottoms, label=modality, color=MODALITY_COLORS[modality])
        bottoms += np.array(vals)

    plt.xticks(x, conditions, rotation=30, ha="right")
    plt.ylabel("Selected Modality Ratio")
    plt.title("Selected Modality by Corruption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_entropy_and_fused(entropy_values, fused_weight_values, out_path):
    conditions = ordered_conditions(entropy_values)
    if not conditions:
        return

    entropy_mean = [np.mean(entropy_values[c]) if entropy_values[c] else np.nan for c in conditions]
    fused_mean = [np.mean(fused_weight_values[c]) if fused_weight_values[c] else np.nan for c in conditions]

    x = np.arange(len(conditions))
    fig, ax1 = plt.subplots(figsize=(max(10, len(conditions) * 1.2), 5))
    ax1.bar(x, fused_mean, color="#4c78a8", alpha=0.75, label="Fused Weight")
    ax1.set_ylabel("Mean Fused Weight")
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=30, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, entropy_mean, color="#d62728", marker="o", linewidth=2, label="Selector Entropy")
    ax2.set_ylabel("Mean Selector Entropy")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title("Fused Weight and Selector Entropy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_uncertainty_histograms_for_conditions(scene_uncs, conditions, out_path, title_suffix=""):
    if not conditions:
        return

    n_rows = len(conditions)
    n_cols = len(MODALITIES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 2.2 * n_rows), squeeze=False)

    for r, condition in enumerate(conditions):
        for c, modality in enumerate(MODALITIES):
            ax = axes[r][c]
            vals = scene_uncs[condition][modality]
            if vals:
                ax.hist(
                    vals,
                    bins=min(20, max(5, len(vals) // 5)),
                    color=MODALITY_COLORS[modality],
                    alpha=0.85)
            ax.set_title(f"{condition} | {modality}")
            if c == 0:
                ax.set_ylabel("Count")
            if r == n_rows - 1:
                ax.set_xlabel("Scene Avg Uncertainty")

    title = "Per-Modality Uncertainty Distribution"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    plt.suptitle(title, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_uncertainty_histograms(scene_uncs, out_dir):
    for severity in (1, 2, 3):
        conditions = ordered_conditions_with_severity(scene_uncs, severity)
        if not conditions:
            continue
        _plot_uncertainty_histograms_for_conditions(
            scene_uncs,
            conditions,
            os.path.join(out_dir, f"uncertainty_histograms_s{severity}.png"),
            title_suffix=f"clean + severity {severity}")


def _switch_ratio(switch_stats, condition):
    total = switch_stats[condition]["total"]
    if total <= 0:
        return 0.0
    return switch_stats[condition]["switched"] / total


def plot_v5_switch_ratio(v5_switch_stats, out_path):
    conditions = ordered_conditions(v5_switch_stats)
    if not conditions:
        return
    vals = [
        _switch_ratio(v5_switch_stats, condition)
        for condition in conditions
    ]
    x = np.arange(len(conditions))
    plt.figure(figsize=(max(10, len(conditions) * 1.1), 5))
    plt.bar(x, vals, color="#4c78a8")
    plt.xticks(x, conditions, rotation=30, ha="right")
    plt.ylabel("Switch Ratio")
    plt.title("V5 Hard Switch Ratio by Corruption")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_v5_switch_ratio_heatmap(v5_switch_stats, out_path):
    corruptions = sorted({
        condition.split("-s")[0]
        for condition in v5_switch_stats.keys()
        if condition != "clean" and "-s" in condition
    })
    if not corruptions:
        return
    severities = [1, 2, 3]
    mat = np.full((len(corruptions), len(severities)), np.nan)
    for i, corr in enumerate(corruptions):
        for j, sev in enumerate(severities):
            key = f"{corr}-s{sev}"
            if key in v5_switch_stats and v5_switch_stats[key]["total"] > 0:
                mat[i, j] = _switch_ratio(v5_switch_stats, key)
    plt.figure(figsize=(8, max(5, len(corruptions) * 0.5)))
    im = plt.imshow(mat, aspect="auto", cmap="magma")
    plt.title("V5 Switch Ratio by Corruption / Severity")
    plt.xticks(range(len(severities)), [f"s{sev}" for sev in severities])
    plt.yticks(range(len(corruptions)), corruptions)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                plt.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_v5_anomaly_histograms(v5_fused_anomaly_hists, conditions, out_path, title_suffix=""):
    if not conditions:
        return
    fig, axes = plt.subplots(len(conditions), 1, figsize=(7, max(3, len(conditions) * 2.0)), squeeze=False)
    for row_idx, condition in enumerate(conditions):
        ax = axes[row_idx][0]
        hist = v5_fused_anomaly_hists[condition]
        if hist.total() > 0:
            ax.stairs(hist.counts, hist.edges, color="#d62728", linewidth=1.5, fill=True, alpha=0.35)
        ax.set_title(condition)
        ax.set_ylabel("Count")
        if row_idx == len(conditions) - 1:
            ax.set_xlabel("Fused Anomaly Score")
    title = "V5 Fused Anomaly Distribution"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    plt.suptitle(title, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_v5_anomaly_histograms(v5_fused_anomaly_hists, out_dir):
    for severity in (1, 2, 3):
        conditions = ordered_conditions_with_severity(v5_fused_anomaly_hists, severity)
        if not conditions:
            continue
        _plot_v5_anomaly_histograms(
            v5_fused_anomaly_hists,
            conditions,
            os.path.join(out_dir, f"v5_fused_anomaly_hist_s{severity}.png"),
            title_suffix=f"clean + severity {severity}")


def plot_v5_switch_distance_hist(v5_switch_distance_hists, out_path):
    conditions = ordered_conditions(v5_switch_distance_hists)
    conditions = [condition for condition in conditions if v5_switch_distance_hists[condition].total() > 0]
    if not conditions:
        return
    fig, axes = plt.subplots(len(conditions), 1, figsize=(7, max(3, len(conditions) * 2.0)), squeeze=False)
    for row_idx, condition in enumerate(conditions):
        ax = axes[row_idx][0]
        hist = v5_switch_distance_hists[condition]
        ax.stairs(hist.counts, hist.edges, color="#2ca02c", linewidth=1.5, fill=True, alpha=0.35)
        ax.set_title(condition)
        ax.set_ylabel("Count")
        if row_idx == len(conditions) - 1:
            ax.set_xlabel("Distance to Ego")
    plt.suptitle("V5 Switched Query Distance Distribution", y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_v5_switch_by_class(v5_switch_class_counts, out_path):
    conditions = ordered_conditions(v5_switch_class_counts)
    if not conditions:
        return
    class_names = sorted({
        class_name
        for condition in conditions
        for class_name in v5_switch_class_counts[condition].keys()
    })
    if not class_names:
        return
    mat = np.zeros((len(conditions), len(class_names)), dtype=float)
    for i, condition in enumerate(conditions):
        total = sum(v5_switch_class_counts[condition].values())
        for j, class_name in enumerate(class_names):
            count = v5_switch_class_counts[condition].get(class_name, 0)
            mat[i, j] = count / total if total > 0 else 0.0
    plt.figure(figsize=(max(8, len(class_names) * 0.6), max(4, len(conditions) * 0.45)))
    im = plt.imshow(mat, aspect="auto", cmap="viridis")
    plt.title("V5 Switched Query Class Ratio")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(conditions)), conditions)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_v5_reference_fallback_ratio(v5_fallback_level_counts, out_path):
    conditions = ordered_conditions(v5_fallback_level_counts)
    if not conditions:
        return
    levels = ["class_distance", "class_global", "global", "missing"]
    x = np.arange(len(conditions))
    bottoms = np.zeros(len(conditions))
    plt.figure(figsize=(max(10, len(conditions) * 1.1), 5))
    colors = {
        "class_distance": "#4c78a8",
        "class_global": "#f58518",
        "global": "#54a24b",
        "missing": "#e45756",
    }
    for level in levels:
        vals = []
        for condition in conditions:
            total = sum(v5_fallback_level_counts[condition].values())
            vals.append(v5_fallback_level_counts[condition].get(level, 0) / total if total > 0 else 0.0)
        plt.bar(x, vals, bottom=bottoms, color=colors[level], label=level)
        bottoms += np.array(vals)
    plt.xticks(x, conditions, rotation=30, ha="right")
    plt.ylabel("Reference Fallback Ratio")
    plt.title("V5 Reference Fallback Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_ratio_counts(counts, categories, colors, ylabel, title, out_path):
    conditions = ordered_conditions(counts)
    if not conditions:
        return
    x = np.arange(len(conditions))
    bottoms = np.zeros(len(conditions))
    plt.figure(figsize=(max(10, len(conditions) * 1.2), 5))
    for category in categories:
        vals = []
        for cond in conditions:
            total = sum(counts[cond].values())
            vals.append(counts[cond].get(category, 0) / total if total > 0 else 0.0)
        plt.bar(x, vals, bottom=bottoms, label=category, color=colors.get(category, "#7f7f7f"))
        bottoms += np.array(vals)
    plt.xticks(x, conditions, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_v6_switch_ratio(v6_switch_stats, out_path):
    conditions = ordered_conditions(v6_switch_stats)
    if not conditions:
        return
    vals = [_switch_ratio(v6_switch_stats, condition) for condition in conditions]
    x = np.arange(len(conditions))
    plt.figure(figsize=(max(10, len(conditions) * 1.1), 5))
    plt.bar(x, vals, color="#4c78a8")
    plt.xticks(x, conditions, rotation=30, ha="right")
    plt.ylabel("Switch Ratio")
    plt.title("V6 Switch Ratio by Corruption")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_v6_anomaly_histograms(v6_gate_anomaly_hists, conditions, out_path, title_suffix=""):
    if not conditions:
        return
    fig, axes = plt.subplots(len(conditions), 1, figsize=(7, max(3, len(conditions) * 2.0)), squeeze=False)
    for row_idx, condition in enumerate(conditions):
        ax = axes[row_idx][0]
        hist = v6_gate_anomaly_hists[condition]
        if hist.total() > 0:
            ax.stairs(hist.counts, hist.edges, color="#d62728", linewidth=1.5, fill=True, alpha=0.35)
        ax.set_title(condition)
        ax.set_ylabel("Count")
        if row_idx == len(conditions) - 1:
            ax.set_xlabel("V6 Gate Anomaly Score")
    title = "V6 Gate Anomaly Distribution"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    plt.suptitle(title, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_v6_anomaly_histograms(v6_gate_anomaly_hists, out_dir):
    for severity in (1, 2, 3):
        conditions = ordered_conditions_with_severity(v6_gate_anomaly_hists, severity)
        if not conditions:
            continue
        _plot_v6_anomaly_histograms(
            v6_gate_anomaly_hists,
            conditions,
            os.path.join(out_dir, f"v6_gate_anomaly_hist_s{severity}.png"),
            title_suffix=f"clean + severity {severity}")


def plot_v6_switch_distance_hist(v6_switch_distance_hists, out_path):
    conditions = ordered_conditions(v6_switch_distance_hists)
    conditions = [condition for condition in conditions if v6_switch_distance_hists[condition].total() > 0]
    if not conditions:
        return
    fig, axes = plt.subplots(len(conditions), 1, figsize=(7, max(3, len(conditions) * 2.0)), squeeze=False)
    for row_idx, condition in enumerate(conditions):
        ax = axes[row_idx][0]
        hist = v6_switch_distance_hists[condition]
        ax.stairs(hist.counts, hist.edges, color="#2ca02c", linewidth=1.5, fill=True, alpha=0.35)
        ax.set_title(condition)
        ax.set_ylabel("Count")
        if row_idx == len(conditions) - 1:
            ax.set_xlabel("Distance to Ego")
    plt.suptitle("V6 Switched Query Distance Distribution", y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_v6_switch_by_class(v6_switch_class_counts, out_path):
    conditions = ordered_conditions(v6_switch_class_counts)
    if not conditions:
        return
    class_names = sorted({
        class_name
        for condition in conditions
        for class_name in v6_switch_class_counts[condition].keys()
    })
    if not class_names:
        return
    mat = np.zeros((len(conditions), len(class_names)), dtype=float)
    for i, condition in enumerate(conditions):
        total = sum(v6_switch_class_counts[condition].values())
        for j, class_name in enumerate(class_names):
            count = v6_switch_class_counts[condition].get(class_name, 0)
            mat[i, j] = count / total if total > 0 else 0.0
    plt.figure(figsize=(max(8, len(class_names) * 0.6), max(4, len(conditions) * 0.45)))
    im = plt.imshow(mat, aspect="auto", cmap="viridis")
    plt.title("V6 Switched Query Class Ratio")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(conditions)), conditions)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_v6_reference_fallback_ratio(v6_fallback_level_counts, out_path):
    conditions = ordered_conditions(v6_fallback_level_counts)
    if not conditions:
        return
    levels = ["class_distance", "class_global", "global", "missing"]
    x = np.arange(len(conditions))
    bottoms = np.zeros(len(conditions))
    plt.figure(figsize=(max(10, len(conditions) * 1.1), 5))
    colors = {
        "class_distance": "#4c78a8",
        "class_global": "#f58518",
        "global": "#54a24b",
        "missing": "#e45756",
    }
    for level in levels:
        vals = []
        for condition in conditions:
            total = sum(v6_fallback_level_counts[condition].values())
            vals.append(v6_fallback_level_counts[condition].get(level, 0) / total if total > 0 else 0.0)
        plt.bar(x, vals, bottom=bottoms, color=colors[level], label=level)
        bottoms += np.array(vals)
    plt.xticks(x, conditions, rotation=30, ha="right")
    plt.ylabel("Reference Fallback Ratio")
    plt.title("V6 Reference Fallback Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def validate_v6_diagnostics(aggregates, strict_v6=False):
    has_v6_schema = bool(aggregates.get("has_v6_schema", False))
    if not (strict_v6 or has_v6_schema):
        return

    total_switch_rows = sum(
        stats.get("total", 0)
        for stats in aggregates["v6_switch_stats"].values()
    )
    total_candidate_rows = sum(
        sum(counts.values())
        for counts in aggregates["v6_candidate_counts"].values()
    )
    total_center_rows = sum(
        sum(counts.values())
        for counts in aggregates["v6_center_source_counts"].values()
    )
    total_height_rows = sum(
        sum(counts.values())
        for counts in aggregates["v6_height_source_counts"].values()
    )
    if (
        total_switch_rows == 0 and
        total_candidate_rows == 0 and
        total_center_rows == 0 and
        total_height_rows == 0
    ):
        raise RuntimeError(
            "Strict v6 diagnostics check failed: v6 schema was detected, but "
            "all switch/candidate/source aggregates are empty."
        )


def _plot_v7_anomaly_histograms(v7_center_anomaly_hists, conditions, out_path, xlabel, title_suffix=""):
    if not conditions:
        return
    fig, axes = plt.subplots(len(conditions), 1, figsize=(7, max(3, len(conditions) * 2.0)), squeeze=False)
    for row_idx, condition in enumerate(conditions):
        ax = axes[row_idx][0]
        hist = v7_center_anomaly_hists[condition]
        if hist.total() > 0:
            ax.stairs(hist.counts, hist.edges, color="#d62728", linewidth=1.5, fill=True, alpha=0.35)
        ax.set_title(condition)
        ax.set_ylabel("Count")
        if row_idx == len(conditions) - 1:
            ax.set_xlabel(xlabel)
    title = xlabel
    if title_suffix:
        title = f"{title} ({title_suffix})"
    plt.suptitle(title, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_v7_bev_center_anomaly_histograms(v7_bev_center_anomaly_hists, out_dir):
    for severity in (1, 2, 3):
        conditions = ordered_conditions_with_severity(v7_bev_center_anomaly_hists, severity)
        if not conditions:
            continue
        _plot_v7_anomaly_histograms(
            v7_bev_center_anomaly_hists,
            conditions,
            os.path.join(out_dir, f"v7_bev_center_anomaly_hist_s{severity}.png"),
            "V7 BEV Center Anomaly",
            title_suffix=f"clean + severity {severity}",
        )


def plot_v7_img_center_anomaly_histograms(v7_img_center_anomaly_hists, out_dir):
    for severity in (1, 2, 3):
        conditions = ordered_conditions_with_severity(v7_img_center_anomaly_hists, severity)
        if not conditions:
            continue
        _plot_v7_anomaly_histograms(
            v7_img_center_anomaly_hists,
            conditions,
            os.path.join(out_dir, f"v7_img_center_anomaly_hist_s{severity}.png"),
            "V7 IMG Center Anomaly",
            title_suffix=f"clean + severity {severity}",
        )


def plot_v7_switch_ratio(v7_switch_stats, out_path):
    conditions = ordered_conditions(v7_switch_stats)
    if not conditions:
        return
    vals = [_switch_ratio(v7_switch_stats, condition) for condition in conditions]
    x = np.arange(len(conditions))
    plt.figure(figsize=(max(10, len(conditions) * 1.1), 5))
    plt.bar(x, vals, color="#4c78a8")
    plt.xticks(x, conditions, rotation=30, ha="right")
    plt.ylabel("Switch Ratio")
    plt.title("V7 Switch Ratio by Corruption")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_v7_switch_distance_hist(v7_switch_distance_hists, out_path):
    conditions = ordered_conditions(v7_switch_distance_hists)
    conditions = [condition for condition in conditions if v7_switch_distance_hists[condition].total() > 0]
    if not conditions:
        return
    fig, axes = plt.subplots(len(conditions), 1, figsize=(7, max(3, len(conditions) * 2.0)), squeeze=False)
    for row_idx, condition in enumerate(conditions):
        ax = axes[row_idx][0]
        hist = v7_switch_distance_hists[condition]
        ax.stairs(hist.counts, hist.edges, color="#2ca02c", linewidth=1.5, fill=True, alpha=0.35)
        ax.set_title(condition)
        ax.set_ylabel("Count")
        if row_idx == len(conditions) - 1:
            ax.set_xlabel("Distance to Ego")
    plt.suptitle("V7 Switched Query Distance Distribution", y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_v7_switch_by_class(v7_switch_class_counts, out_path):
    conditions = ordered_conditions(v7_switch_class_counts)
    if not conditions:
        return
    class_names = sorted({
        class_name
        for condition in conditions
        for class_name in v7_switch_class_counts[condition].keys()
    })
    if not class_names:
        return
    mat = np.zeros((len(conditions), len(class_names)), dtype=float)
    for i, condition in enumerate(conditions):
        total = sum(v7_switch_class_counts[condition].values())
        for j, class_name in enumerate(class_names):
            count = v7_switch_class_counts[condition].get(class_name, 0)
            mat[i, j] = count / total if total > 0 else 0.0
    plt.figure(figsize=(max(8, len(class_names) * 0.6), max(4, len(conditions) * 0.45)))
    im = plt.imshow(mat, aspect="auto", cmap="viridis")
    plt.title("V7 Switched Query Class Ratio")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(conditions)), conditions)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_v8_gate_means(v8_fused_keep_values, v8_damage_gate_values, v8_high_damage_values, out_path):
    conditions = ordered_conditions(v8_fused_keep_values)
    if not conditions:
        return
    fused_keep = [
        np.mean(v8_fused_keep_values[c]) if v8_fused_keep_values[c] else np.nan
        for c in conditions
    ]
    damage_gate = [
        np.mean(v8_damage_gate_values[c]) if v8_damage_gate_values[c] else np.nan
        for c in conditions
    ]
    high_damage = [
        np.mean(v8_high_damage_values[c]) if v8_high_damage_values[c] else np.nan
        for c in conditions
    ]
    x = np.arange(len(conditions))
    width = 0.26
    plt.figure(figsize=(max(10, len(conditions) * 1.2), 5))
    plt.bar(x - width, fused_keep, width, label="fused_keep", color="#1f77b4")
    plt.bar(x, damage_gate, width, label="damage_gate", color="#d62728")
    plt.bar(x + width, high_damage, width, label="high_damage_query", color="#9467bd")
    plt.xticks(x, conditions, rotation=30, ha="right")
    plt.ylabel("Mean Query Ratio")
    plt.title("V8 Damage-Gated Router Response")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_v8_reliability_means(v8_reliability_values, v8_reliability_margin_values, out_path):
    conditions = ordered_conditions(v8_reliability_values)
    if not conditions:
        return
    bev_vals = [
        np.mean(v8_reliability_values[c]["bev"]) if v8_reliability_values[c]["bev"] else np.nan
        for c in conditions
    ]
    img_vals = [
        np.mean(v8_reliability_values[c]["img"]) if v8_reliability_values[c]["img"] else np.nan
        for c in conditions
    ]
    margin_vals = [
        np.mean(v8_reliability_margin_values[c]) if v8_reliability_margin_values[c] else np.nan
        for c in conditions
    ]
    x = np.arange(len(conditions))
    width = 0.26
    plt.figure(figsize=(max(10, len(conditions) * 1.2), 5))
    plt.bar(x - width, bev_vals, width, label="BEV reliability", color=MODALITY_COLORS["bev"])
    plt.bar(x, img_vals, width, label="IMG reliability", color=MODALITY_COLORS["img"])
    plt.bar(x + width, margin_vals, width, label="|BEV-IMG| margin", color="#9467bd")
    plt.xticks(x, conditions, rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Mean Reliability")
    plt.title("V8 Reliability by Corruption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_v8_soft_weight_means(v8_soft_weight_values, out_path):
    conditions = ordered_conditions(v8_soft_weight_values)
    if not conditions:
        return
    x = np.arange(len(conditions))
    width = 0.26
    plt.figure(figsize=(max(10, len(conditions) * 1.2), 5))
    for idx, modality in enumerate(MODALITIES):
        vals = [
            np.mean(v8_soft_weight_values[c][modality]) if v8_soft_weight_values[c][modality] else np.nan
            for c in conditions
        ]
        plt.bar(x + (idx - 1) * width, vals, width, label=modality, color=MODALITY_COLORS[modality])
    plt.xticks(x, conditions, rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Mean Soft Output Weight")
    plt.title("V8 Soft Source Weights by Corruption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_v8_output_delta_means(v8_output_delta_values, out_path):
    conditions = ordered_conditions(v8_output_delta_values)
    if not conditions:
        return
    keys = ["cls", "center", "height"]
    colors = {"cls": "#4c78a8", "center": "#f58518", "height": "#54a24b"}
    x = np.arange(len(conditions))
    width = 0.26
    plt.figure(figsize=(max(10, len(conditions) * 1.2), 5))
    for idx, key in enumerate(keys):
        vals = [
            np.mean(v8_output_delta_values[c][key]) if v8_output_delta_values[c][key] else np.nan
            for c in conditions
        ]
        plt.bar(x + (idx - 1) * width, vals, width, label=key, color=colors[key])
    plt.xticks(x, conditions, rotation=30, ha="right")
    plt.ylabel("Mean L1 Delta vs Fused")
    plt.title("V8 Output Delta by Corruption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_v8_domain_correct(v8_domain_correct_values, out_path):
    conditions = ordered_conditions(v8_domain_correct_values)
    if not conditions:
        return
    vals = [
        np.mean(v8_domain_correct_values[c]) if v8_domain_correct_values[c] else np.nan
        for c in conditions
    ]
    x = np.arange(len(conditions))
    plt.figure(figsize=(max(10, len(conditions) * 1.2), 5))
    plt.bar(x, vals, color="#4c78a8")
    plt.xticks(x, conditions, rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Safe-Modality Non-Fused Choice Ratio")
    plt.title("V8 Domain-Correct Route Ratio")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_v8_reliability_histograms(v8_reliability_values, out_dir):
    bins = np.linspace(0.0, 1.0, 41)
    for severity in (1, 2, 3):
        conditions = ordered_conditions_with_severity(v8_reliability_values, severity)
        if not conditions:
            continue
        fig, axes = plt.subplots(len(conditions), 1, figsize=(7, max(3, len(conditions) * 2.0)), squeeze=False)
        for row_idx, condition in enumerate(conditions):
            ax = axes[row_idx][0]
            for modality in ("bev", "img"):
                vals = v8_reliability_values[condition][modality]
                if vals:
                    ax.hist(
                        vals,
                        bins=bins,
                        alpha=0.4,
                        color=MODALITY_COLORS[modality],
                        label=modality)
            ax.set_xlim(0.0, 1.0)
            ax.set_title(condition)
            ax.set_ylabel("Count")
            if row_idx == 0:
                ax.legend()
            if row_idx == len(conditions) - 1:
                ax.set_xlabel("Reliability")
        plt.suptitle(f"V8 Reliability Histograms (clean + severity {severity})", y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"v8_reliability_hist_s{severity}.png"), dpi=180)
        plt.close()


def _plot_v8_damage_histograms(v8_damage_hists, conditions, out_path, xlabel, title_suffix=""):
    if not conditions:
        return
    fig, axes = plt.subplots(len(conditions), 1, figsize=(7, max(3, len(conditions) * 2.0)), squeeze=False)
    for row_idx, condition in enumerate(conditions):
        ax = axes[row_idx][0]
        hist = v8_damage_hists[condition]
        if hist.total() > 0:
            ax.stairs(hist.counts, hist.edges, color="#d62728", linewidth=1.5, fill=True, alpha=0.35)
        ax.set_title(condition)
        ax.set_ylabel("Count")
        if row_idx == len(conditions) - 1:
            ax.set_xlabel(xlabel)
    title = xlabel
    if title_suffix:
        title = f"{title} ({title_suffix})"
    plt.suptitle(title, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_v8_damage_histograms(v8_damage_hists, out_dir, name):
    for severity in (1, 2, 3):
        conditions = ordered_conditions_with_severity(v8_damage_hists, severity)
        if not conditions:
            continue
        _plot_v8_damage_histograms(
            v8_damage_hists,
            conditions,
            os.path.join(out_dir, f"v8_{name}_damage_hist_s{severity}.png"),
            f"V8 {name.upper()} Damage Score",
            title_suffix=f"clean + severity {severity}",
        )


def plot_v8_scalar_histograms(value_hists, out_dir, name, xlabel):
    for severity in (1, 2, 3):
        conditions = ordered_conditions_with_severity(value_hists, severity)
        if not conditions:
            continue
        _plot_v8_damage_histograms(
            value_hists,
            conditions,
            os.path.join(out_dir, f"v8_{name}_hist_s{severity}.png"),
            xlabel,
            title_suffix=f"clean + severity {severity}",
        )


def plot_v9_scalar_histograms(value_hists, out_dir, name, xlabel):
    for severity in (1, 2, 3):
        conditions = ordered_conditions_with_severity(value_hists, severity)
        if not conditions:
            continue
        _plot_v8_damage_histograms(
            value_hists,
            conditions,
            os.path.join(out_dir, f"v9_{name}_hist_s{severity}.png"),
            xlabel,
            title_suffix=f"clean + severity {severity}",
        )


def plot_v9_ratio_summary(v9_above_tau_values, v9_clean_like_values, out_path):
    conditions = ordered_conditions(v9_above_tau_values)
    if not conditions:
        conditions = ordered_conditions(v9_clean_like_values)
    if not conditions:
        return

    x = np.arange(len(conditions))
    width = 0.25
    bev_vals = [
        float(np.mean(v9_above_tau_values[c]["bev"])) if v9_above_tau_values[c]["bev"] else 0.0
        for c in conditions
    ]
    img_vals = [
        float(np.mean(v9_above_tau_values[c]["img"])) if v9_above_tau_values[c]["img"] else 0.0
        for c in conditions
    ]
    clean_like_vals = [
        float(np.mean(v9_clean_like_values[c])) if v9_clean_like_values[c] else 0.0
        for c in conditions
    ]

    plt.figure(figsize=(max(10, len(conditions) * 1.2), 5))
    plt.bar(x - width, bev_vals, width=width, label="a_bev > tau", color="#2ca02c")
    plt.bar(x, img_vals, width=width, label="a_img > tau", color="#ff7f0e")
    plt.bar(x + width, clean_like_vals, width=width, label="clean-like clamp", color="#7f7f7f")
    plt.xticks(x, conditions, rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Mean Ratio")
    plt.title("V9 Anomaly Tau and Clean-Like Clamp Ratios")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_v9_scene_scalar_evolution(v9_scene_scalar_values, out_path):
    conditions = ordered_conditions(v9_scene_scalar_values)
    if not conditions:
        return
    scalar_names = [
        "tau_bev",
        "tau_img",
        "clean_distance_threshold_bev",
        "clean_distance_threshold_img",
        "clean_q95_z_bev",
        "clean_q95_z_img",
    ]
    x = np.arange(len(conditions))
    plt.figure(figsize=(max(10, len(conditions) * 1.2), 5))
    for name in scalar_names:
        vals = [
            float(np.mean(v9_scene_scalar_values[c][name]))
            if v9_scene_scalar_values[c][name] else np.nan
            for c in conditions
        ]
        if np.all(np.isnan(vals)):
            continue
        plt.plot(x, vals, marker="o", linewidth=1.5, label=name)
    plt.xticks(x, conditions, rotation=30, ha="right")
    plt.ylabel("Mean Value")
    plt.title("V9 Online Tau / Clean Anchor Threshold Diagnostics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def validate_v7_diagnostics(aggregates, strict_v7=False):
    has_v7_schema = bool(aggregates.get("has_v7_schema", False))
    if not (strict_v7 or has_v7_schema):
        return

    total_switch_rows = sum(
        stats.get("total", 0)
        for stats in aggregates["v7_switch_stats"].values()
    )
    total_damage_rows = sum(
        sum(counts.values())
        for counts in aggregates["v7_damage_state_counts"].values()
    )
    total_center_rows = sum(
        sum(counts.values())
        for counts in aggregates["v7_center_source_counts"].values()
    )
    if (
        total_switch_rows == 0 and
        total_damage_rows == 0 and
        total_center_rows == 0
    ):
        raise RuntimeError(
            "Strict v7 diagnostics check failed: v7 schema was detected, but "
            "all switch/damage/source aggregates are empty."
        )


def validate_v9_diagnostics(aggregates, strict_v9=False):
    has_v9_schema = bool(aggregates.get("has_v9_schema", False))
    if not (strict_v9 or has_v9_schema):
        return

    total_anomaly_rows = (
        sum(hist.total() for hist in aggregates["v9_a_bev_hists"].values()) +
        sum(hist.total() for hist in aggregates["v9_a_img_hists"].values())
    )
    total_tau_rows = sum(
        len(vals["bev"]) + len(vals["img"])
        for vals in aggregates["v9_above_tau_values"].values()
    )
    total_scene_rows = sum(
        len(vals)
        for by_name in aggregates["v9_scene_scalar_values"].values()
        for vals in by_name.values()
    )
    if total_anomaly_rows == 0 and total_tau_rows == 0 and total_scene_rows == 0:
        raise RuntimeError(
            "Strict v9 diagnostics check failed: v9 schema was detected, but "
            "all anomaly/tau/scene aggregates are empty."
        )


def plot_distance_uncertainty_scatter(scatter_points, out_path, max_points_per_condition=1200):
    conditions = ordered_conditions(scatter_points)
    if not conditions:
        return

    palette = plt.cm.get_cmap("tab20", max(len(conditions), 3))
    plt.figure(figsize=(10, 6))
    for idx, condition in enumerate(conditions):
        pts = scatter_points[condition]
        if not pts:
            continue
        if len(pts) > max_points_per_condition:
            step = max(1, len(pts) // max_points_per_condition)
            pts = pts[::step]
        arr = np.array(pts)
        plt.scatter(arr[:, 0], arr[:, 1], s=8, alpha=0.25, color=palette(idx), label=condition)

    plt.xlabel("Distance to Ego")
    plt.ylabel("Combined / Center Uncertainty")
    plt.title("Distance vs Uncertainty")
    plt.legend(loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_distance_prior_vs_final_selection(distance_prior_top1, selected_counts, out_path):
    conditions = ordered_conditions(distance_prior_top1)
    if not conditions:
        return

    prior_fused = [
        np.mean(distance_prior_top1[c]["fused"]) if distance_prior_top1[c]["fused"] else np.nan
        for c in conditions
    ]
    final_fused = []
    for c in conditions:
        total = sum(selected_counts[c][m] for m in MODALITIES)
        final_fused.append(selected_counts[c]["fused"] / total if total > 0 else np.nan)

    x = np.arange(len(conditions))
    width = 0.38
    plt.figure(figsize=(max(10, len(conditions) * 1.2), 5))
    plt.bar(x - width / 2, prior_fused, width=width, label="Distance Prior Top1 (fused)", color="#9ecae1")
    plt.bar(x + width / 2, final_fused, width=width, label="Final Selected Top1 (fused)", color="#1f77b4")
    plt.xticks(x, conditions, rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Ratio")
    plt.title("Distance Prior vs Final Selection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_hard_query_residual_activation(alpha_res_values, residual_override_values, tail_query_values, out_path):
    conditions = ordered_conditions(alpha_res_values)
    if not conditions:
        return

    alpha_mean = [np.mean(alpha_res_values[c]) if alpha_res_values[c] else np.nan for c in conditions]
    override_mean = [np.mean(residual_override_values[c]) if residual_override_values[c] else np.nan for c in conditions]
    tail_mean = []
    for c in conditions:
        vals = []
        for m in MODALITIES:
            vals.extend(tail_query_values[c][m])
        tail_mean.append(float(np.mean(vals)) if vals else np.nan)

    x = np.arange(len(conditions))
    fig, ax1 = plt.subplots(figsize=(max(10, len(conditions) * 1.2), 5))
    ax1.bar(x, alpha_mean, color="#4c78a8", alpha=0.75, label="Alpha Residual")
    ax1.plot(x, override_mean, color="#d62728", marker="o", linewidth=2, label="Residual Override")
    ax1.set_ylabel("Alpha / Override")
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=30, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, tail_mean, color="#2ca02c", marker="s", linewidth=2, label="Tail Query Ratio")
    ax2.set_ylabel("Tail Query Ratio")
    ax2.set_ylim(0, 1.0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title("Hard-Query Residual Activation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_bev_qualitative(sample_info, out_path):
    points = sample_info.get("points", [])
    if not points:
        return

    xs, ys, colors, sizes = [], [], [], []
    for point in points:
        xs.append(point["x"])
        ys.append(point["y"])
        selected = point.get("selected_modality", "fused")
        colors.append(MODALITY_COLORS.get(selected, "#7f7f7f"))
        unc = safe_float(point.get("combined_unc"), default=0.1)
        sizes.append(max(20.0, 120.0 / (1.0 + unc * 20.0)))

    if not xs:
        return

    plt.figure(figsize=(7, 7))
    plt.scatter(xs, ys, c=colors, s=sizes, alpha=0.75, edgecolors="k", linewidths=0.25)
    plt.axhline(0, color="#bbbbbb", linewidth=0.8)
    plt.axvline(0, color="#bbbbbb", linewidth=0.8)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"BEV Qualitative: {sample_info['condition']} ({sample_info['sample_token'][:12]})")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=modality,
                   markerfacecolor=MODALITY_COLORS[modality], markersize=8)
        for modality in MODALITIES
    ]
    plt.legend(handles=handles, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_summary_json(aggregates, out_path):
    summary = {
        "conditions": {},
    }
    scene_weights = aggregates["scene_weights"]
    scene_uncs = aggregates["scene_uncs"]
    selected_counts = aggregates["selected_counts"]
    entropy_values = aggregates["entropy_values"]
    fused_weight_values = aggregates["fused_weight_values"]
    distance_prior_top1 = aggregates["distance_prior_top1"]
    residual_override_values = aggregates["residual_override_values"]
    alpha_res_values = aggregates["alpha_res_values"]
    tail_query_values = aggregates["tail_query_values"]
    v5_switch_stats = aggregates["v5_switch_stats"]
    v5_fused_anomaly_hists = aggregates["v5_fused_anomaly_hists"]
    v5_switch_distance_hists = aggregates["v5_switch_distance_hists"]
    v5_switch_class_counts = aggregates["v5_switch_class_counts"]
    v5_fallback_level_counts = aggregates["v5_fallback_level_counts"]
    v5_selected_counts = aggregates["v5_selected_counts"]
    v6_switch_stats = aggregates["v6_switch_stats"]
    v6_gate_anomaly_hists = aggregates["v6_gate_anomaly_hists"]
    v6_switch_distance_hists = aggregates["v6_switch_distance_hists"]
    v6_switch_class_counts = aggregates["v6_switch_class_counts"]
    v6_fallback_level_counts = aggregates["v6_fallback_level_counts"]
    v6_candidate_counts = aggregates["v6_candidate_counts"]
    v6_center_source_counts = aggregates["v6_center_source_counts"]
    v6_height_source_counts = aggregates["v6_height_source_counts"]
    v6_gate_pass_anomaly_values = aggregates["v6_gate_pass_anomaly_values"]
    v6_gate_pass_candidate_prob_values = aggregates["v6_gate_pass_candidate_prob_values"]
    v6_gate_pass_candidate_margin_values = aggregates["v6_gate_pass_candidate_margin_values"]
    v6_gate_pass_final_values = aggregates["v6_gate_pass_final_values"]
    v7_switch_stats = aggregates["v7_switch_stats"]
    v7_bev_center_anomaly_hists = aggregates["v7_bev_center_anomaly_hists"]
    v7_img_center_anomaly_hists = aggregates["v7_img_center_anomaly_hists"]
    v7_switch_distance_hists = aggregates["v7_switch_distance_hists"]
    v7_switch_class_counts = aggregates["v7_switch_class_counts"]
    v7_damage_state_counts = aggregates["v7_damage_state_counts"]
    v7_center_source_counts = aggregates["v7_center_source_counts"]
    v8_fused_keep_values = aggregates["v8_fused_keep_values"]
    v8_center_fused_keep_values = aggregates["v8_center_fused_keep_values"]
    v8_damage_gate_values = aggregates["v8_damage_gate_values"]
    v8_high_damage_values = aggregates["v8_high_damage_values"]
    v8_center_route_allowed_values = aggregates["v8_center_route_allowed_values"]
    v8_bev_damage_hists = aggregates["v8_bev_damage_hists"]
    v8_img_damage_hists = aggregates["v8_img_damage_hists"]
    v8_dual_damage_hists = aggregates["v8_dual_damage_hists"]
    v8_context_damage_values = aggregates["v8_context_damage_values"]
    v8_a4_bev_damage_z_hists = aggregates["v8_a4_bev_damage_z_hists"]
    v8_a4_img_damage_z_hists = aggregates["v8_a4_img_damage_z_hists"]
    v8_a4_cls_unc_z_bev_hists = aggregates["v8_a4_cls_unc_z_bev_hists"]
    v8_a4_cls_unc_z_img_hists = aggregates["v8_a4_cls_unc_z_img_hists"]
    v8_a4_route_score_bev_hists = aggregates["v8_a4_route_score_bev_hists"]
    v8_a4_route_score_img_hists = aggregates["v8_a4_route_score_img_hists"]
    v8_damage_state_counts = aggregates["v8_damage_state_counts"]
    v8_reliability_state_counts = aggregates["v8_reliability_state_counts"]
    v8_selected_source_counts = aggregates["v8_selected_source_counts"]
    v8_non_fused_choice_counts = aggregates["v8_non_fused_choice_counts"]
    v8_reliability_values = aggregates["v8_reliability_values"]
    v8_reliability_margin_values = aggregates["v8_reliability_margin_values"]
    v8_soft_weight_values = aggregates["v8_soft_weight_values"]
    v8_output_delta_values = aggregates["v8_output_delta_values"]
    v8_domain_correct_values = aggregates["v8_domain_correct_values"]
    v9_a_bev_hists = aggregates["v9_a_bev_hists"]
    v9_a_img_hists = aggregates["v9_a_img_hists"]
    v9_anchor_distance_bev_hists = aggregates["v9_anchor_distance_bev_hists"]
    v9_anchor_distance_img_hists = aggregates["v9_anchor_distance_img_hists"]
    v9_tail_prob_bev_hists = aggregates["v9_tail_prob_bev_hists"]
    v9_tail_prob_img_hists = aggregates["v9_tail_prob_img_hists"]
    v9_tail_excess_bev_hists = aggregates["v9_tail_excess_bev_hists"]
    v9_tail_excess_img_hists = aggregates["v9_tail_excess_img_hists"]
    v9_q95_shift_bev_hists = aggregates["v9_q95_shift_bev_hists"]
    v9_q95_shift_img_hists = aggregates["v9_q95_shift_img_hists"]
    v9_route_score_bev_hists = aggregates["v9_route_score_bev_hists"]
    v9_route_score_img_hists = aggregates["v9_route_score_img_hists"]
    v9_velocity_l1_hists = aggregates["v9_velocity_l1_hists"]
    v9_above_tau_values = aggregates["v9_above_tau_values"]
    v9_clean_like_values = aggregates["v9_clean_like_values"]
    v9_scene_scalar_values = aggregates["v9_scene_scalar_values"]

    for condition in ordered_conditions(scene_weights):
        total_selected = sum(selected_counts[condition][m] for m in MODALITIES)
        summary["conditions"][condition] = {
            "mean_modality_weights": {
                m: (float(np.mean(scene_weights[condition][m])) if scene_weights[condition][m] else None)
                for m in MODALITIES
            },
            "mean_modality_unc": {
                m: (float(np.mean(scene_uncs[condition][m])) if scene_uncs[condition][m] else None)
                for m in MODALITIES
            },
            "selected_modality_ratio": {
                m: (selected_counts[condition][m] / total_selected if total_selected > 0 else None)
                for m in MODALITIES
            },
            "mean_selector_entropy": float(np.mean(entropy_values[condition])) if entropy_values[condition] else None,
            "mean_fused_weight": float(np.mean(fused_weight_values[condition])) if fused_weight_values[condition] else None,
            "mean_distance_prior_top1_ratio": {
                m: (float(np.mean(distance_prior_top1[condition][m])) if distance_prior_top1[condition][m] else None)
                for m in MODALITIES
            },
            "mean_residual_override_ratio": float(np.mean(residual_override_values[condition])) if residual_override_values[condition] else None,
            "mean_alpha_res": float(np.mean(alpha_res_values[condition])) if alpha_res_values[condition] else None,
            "mean_uncertainty_tail_query_ratio": {
                m: (float(np.mean(tail_query_values[condition][m])) if tail_query_values[condition][m] else None)
                for m in MODALITIES
            },
            "mean_v5_switch_ratio": (
                _switch_ratio(v5_switch_stats, condition)
                if v5_switch_stats[condition]["total"] > 0 else None
            ),
            "mean_v5_fused_anomaly": v5_fused_anomaly_hists[condition].mean(),
            "mean_v5_switch_to_bev_ratio": (
                v5_selected_counts[condition].get("bev", 0) / max(sum(v5_selected_counts[condition].values()), 1)
                if v5_selected_counts[condition] else None
            ),
            "mean_v5_switch_to_img_ratio": (
                v5_selected_counts[condition].get("img", 0) / max(sum(v5_selected_counts[condition].values()), 1)
                if v5_selected_counts[condition] else None
            ),
            "mean_v5_reference_fallback_ratio": {
                level: (
                    v5_fallback_level_counts[condition].get(level, 0) / max(sum(v5_fallback_level_counts[condition].values()), 1)
                    if v5_fallback_level_counts[condition] else None
                )
                for level in ["class_distance", "class_global", "global", "missing"]
            },
            "v5_switch_ratio_by_class": dict(v5_switch_class_counts[condition]),
            "v5_switch_ratio_by_distance_bin": {
                "mean_switched_distance": (
                    v5_switch_distance_hists[condition].mean()
                )
            },
            "mean_v6_switch_ratio": (
                _switch_ratio(v6_switch_stats, condition)
                if v6_switch_stats[condition]["total"] > 0 else None
            ),
            "mean_v6_gate_anomaly": v6_gate_anomaly_hists[condition].mean(),
            "mean_v6_gate_pass_anomaly_ratio": (
                float(np.mean(v6_gate_pass_anomaly_values[condition]))
                if v6_gate_pass_anomaly_values[condition] else None
            ),
            "mean_v6_gate_pass_candidate_prob_ratio": (
                float(np.mean(v6_gate_pass_candidate_prob_values[condition]))
                if v6_gate_pass_candidate_prob_values[condition] else None
            ),
            "mean_v6_gate_pass_candidate_margin_ratio": (
                float(np.mean(v6_gate_pass_candidate_margin_values[condition]))
                if v6_gate_pass_candidate_margin_values[condition] else None
            ),
            "mean_v6_gate_pass_final_ratio": (
                float(np.mean(v6_gate_pass_final_values[condition]))
                if v6_gate_pass_final_values[condition] else None
            ),
            "mean_v6_candidate_modality_ratio": {
                modality: (
                    v6_candidate_counts[condition].get(modality, 0) /
                    max(sum(v6_candidate_counts[condition].values()), 1)
                    if v6_candidate_counts[condition] else None
                )
                for modality in ["bev", "img"]
            },
            "mean_v6_center_source_ratio": {
                source: (
                    v6_center_source_counts[condition].get(source, 0) /
                    max(sum(v6_center_source_counts[condition].values()), 1)
                    if v6_center_source_counts[condition] else None
                )
                for source in SOURCE_CATEGORIES
            },
            "mean_v6_height_source_ratio": {
                source: (
                    v6_height_source_counts[condition].get(source, 0) /
                    max(sum(v6_height_source_counts[condition].values()), 1)
                    if v6_height_source_counts[condition] else None
                )
                for source in SOURCE_CATEGORIES
            },
            "mean_v6_reference_fallback_ratio": {
                level: (
                    v6_fallback_level_counts[condition].get(level, 0) /
                    max(sum(v6_fallback_level_counts[condition].values()), 1)
                    if v6_fallback_level_counts[condition] else None
                )
                for level in ["class_distance", "class_global", "global", "missing"]
            },
            "v6_switch_ratio_by_class": dict(v6_switch_class_counts[condition]),
            "v6_switch_ratio_by_distance_bin": {
                "mean_switched_distance": (
                    v6_switch_distance_hists[condition].mean()
                )
            },
            "mean_v7_switch_ratio": (
                _switch_ratio(v7_switch_stats, condition)
                if v7_switch_stats[condition]["total"] > 0 else None
            ),
            "mean_v7_bev_center_anom": v7_bev_center_anomaly_hists[condition].mean(),
            "mean_v7_img_center_anom": v7_img_center_anomaly_hists[condition].mean(),
            "mean_v7_damage_state_ratio": {
                state: (
                    v7_damage_state_counts[condition].get(state, 0) /
                    max(sum(v7_damage_state_counts[condition].values()), 1)
                    if v7_damage_state_counts[condition] else None
                )
                for state in V7_DAMAGE_STATES
            },
            "mean_v7_center_source_ratio": {
                source: (
                    v7_center_source_counts[condition].get(source, 0) /
                    max(sum(v7_center_source_counts[condition].values()), 1)
                    if v7_center_source_counts[condition] else None
                )
                for source in V7_CENTER_SOURCE_CATEGORIES
            },
            "v7_switch_ratio_by_class": dict(v7_switch_class_counts[condition]),
            "v7_switch_ratio_by_distance_bin": {
                "mean_switched_distance": (
                    v7_switch_distance_hists[condition].mean()
                )
            },
            "mean_v8_fused_keep": (
                float(np.mean(v8_fused_keep_values[condition]))
                if v8_fused_keep_values[condition] else None
            ),
            "mean_v8_center_fused_keep": (
                float(np.mean(v8_center_fused_keep_values[condition]))
                if v8_center_fused_keep_values[condition] else None
            ),
            "mean_v8_damage_gate": (
                float(np.mean(v8_damage_gate_values[condition]))
                if v8_damage_gate_values[condition] else None
            ),
            "mean_v8_high_damage_query_ratio": (
                float(np.mean(v8_high_damage_values[condition]))
                if v8_high_damage_values[condition] else None
            ),
            "mean_v8_center_route_allowed_ratio": (
                float(np.mean(v8_center_route_allowed_values[condition]))
                if v8_center_route_allowed_values[condition] else None
            ),
            "mean_v8_bev_damage": v8_bev_damage_hists[condition].mean(),
            "mean_v8_img_damage": v8_img_damage_hists[condition].mean(),
            "mean_v8_dual_damage": v8_dual_damage_hists[condition].mean(),
            "mean_v8_context_damage": {
                modality: (
                    float(np.mean(v8_context_damage_values[condition][modality]))
                    if v8_context_damage_values[condition][modality] else None
                )
                for modality in ["bev", "img"]
            },
            "mean_v8_a4_damage_z": {
                "bev": v8_a4_bev_damage_z_hists[condition].mean(),
                "img": v8_a4_img_damage_z_hists[condition].mean(),
            },
            "mean_v8_a4_cls_unc_z": {
                "bev": v8_a4_cls_unc_z_bev_hists[condition].mean(),
                "img": v8_a4_cls_unc_z_img_hists[condition].mean(),
            },
            "mean_v8_a4_route_score": {
                "bev": v8_a4_route_score_bev_hists[condition].mean(),
                "img": v8_a4_route_score_img_hists[condition].mean(),
            },
            "mean_v8_damage_state_ratio": {
                state: (
                    v8_damage_state_counts[condition].get(state, 0) /
                    max(sum(v8_damage_state_counts[condition].values()), 1)
                    if v8_damage_state_counts[condition] else None
                )
                for state in V8_DAMAGE_STATES
            },
            "mean_v8_reliability": {
                modality: (
                    float(np.mean(v8_reliability_values[condition][modality]))
                    if v8_reliability_values[condition][modality] else None
                )
                for modality in ["bev", "img"]
            },
            "mean_v8_reliability_margin": (
                float(np.mean(v8_reliability_margin_values[condition]))
                if v8_reliability_margin_values[condition] else None
            ),
            "mean_v8_reliability_state_ratio": {
                state: (
                    v8_reliability_state_counts[condition].get(state, 0) /
                    max(sum(v8_reliability_state_counts[condition].values()), 1)
                    if v8_reliability_state_counts[condition] else None
                )
                for state in V8_RELIABILITY_STATES
            },
            "mean_v8_soft_weight_ratio": {
                modality: (
                    float(np.mean(v8_soft_weight_values[condition][modality]))
                    if v8_soft_weight_values[condition][modality] else None
                )
                for modality in MODALITIES
            },
            "mean_v8_output_delta": {
                name: (
                    float(np.mean(v8_output_delta_values[condition][name]))
                    if v8_output_delta_values[condition][name] else None
                )
                for name in ["cls", "center", "height"]
            },
            "mean_v8_domain_correct_route_ratio": (
                float(np.mean(v8_domain_correct_values[condition]))
                if v8_domain_correct_values[condition] else None
            ),
            "mean_v8_selected_source_ratio": {
                source: (
                    v8_selected_source_counts[condition].get(source, 0) /
                    max(sum(v8_selected_source_counts[condition].values()), 1)
                    if v8_selected_source_counts[condition] else None
                )
                for source in V8_SOURCE_CATEGORIES
            },
            "mean_v8_non_fused_choice_ratio": {
                source: (
                    v8_non_fused_choice_counts[condition].get(source, 0) /
                    max(sum(v8_non_fused_choice_counts[condition].values()), 1)
                    if v8_non_fused_choice_counts[condition] else None
                )
                for source in ["bev", "img"]
            },
            "mean_v9_anomaly": {
                "bev": v9_a_bev_hists[condition].mean(),
                "img": v9_a_img_hists[condition].mean(),
            },
            "mean_v9_above_tau_ratio": {
                modality: (
                    float(np.mean(v9_above_tau_values[condition][modality]))
                    if v9_above_tau_values[condition][modality] else None
                )
                for modality in ["bev", "img"]
            },
            "mean_v9_clean_like_ratio": (
                float(np.mean(v9_clean_like_values[condition]))
                if v9_clean_like_values[condition] else None
            ),
            "mean_v9_anchor_distance": {
                "bev": v9_anchor_distance_bev_hists[condition].mean(),
                "img": v9_anchor_distance_img_hists[condition].mean(),
            },
            "mean_v9_tail_prob": {
                "bev": v9_tail_prob_bev_hists[condition].mean(),
                "img": v9_tail_prob_img_hists[condition].mean(),
            },
            "mean_v9_tail_excess": {
                "bev": v9_tail_excess_bev_hists[condition].mean(),
                "img": v9_tail_excess_img_hists[condition].mean(),
            },
            "mean_v9_q95_shift": {
                "bev": v9_q95_shift_bev_hists[condition].mean(),
                "img": v9_q95_shift_img_hists[condition].mean(),
            },
            "mean_v9_route_score": {
                "bev": v9_route_score_bev_hists[condition].mean(),
                "img": v9_route_score_img_hists[condition].mean(),
            },
            "mean_v9_velocity_l1": v9_velocity_l1_hists[condition].mean(),
            "mean_v9_scene_scalar": {
                name: (
                    float(np.mean(vals)) if vals else None
                )
                for name, vals in v9_scene_scalar_values[condition].items()
            },
        }

    all_selected = defaultdict(int)
    all_fallback = defaultdict(int)
    all_class_counts = defaultdict(int)
    total_switched = 0
    total_switch_candidates = 0
    all_anomaly = RunningStats()
    all_switched_distance = RunningStats()
    all_v6_candidate = defaultdict(int)
    all_v6_center_source = defaultdict(int)
    all_v6_height_source = defaultdict(int)
    all_v6_fallback = defaultdict(int)
    all_v6_class_counts = defaultdict(int)
    total_v6_switched = 0
    total_v6_switch_candidates = 0
    all_v6_anomaly = RunningStats()
    all_v6_switched_distance = RunningStats()
    all_v6_gate_pass_anomaly = []
    all_v6_gate_pass_candidate_prob = []
    all_v6_gate_pass_candidate_margin = []
    all_v6_gate_pass_final = []
    all_v7_damage_state = defaultdict(int)
    all_v7_center_source = defaultdict(int)
    all_v7_class_counts = defaultdict(int)
    total_v7_switched = 0
    total_v7_switch_candidates = 0
    all_v7_bev_anomaly = RunningStats()
    all_v7_img_anomaly = RunningStats()
    all_v7_switched_distance = RunningStats()
    all_v8_damage_state = defaultdict(int)
    all_v8_reliability_state = defaultdict(int)
    all_v8_selected_source = defaultdict(int)
    all_v8_non_fused_choice = defaultdict(int)
    all_v8_fused_keep = []
    all_v8_center_fused_keep = []
    all_v8_damage_gate = []
    all_v8_high_damage = []
    all_v8_center_route_allowed = []
    all_v8_reliability = {"bev": [], "img": []}
    all_v8_context_damage = {"bev": [], "img": []}
    all_v8_reliability_margin = []
    all_v8_soft_weight = {m: [] for m in MODALITIES}
    all_v8_output_delta = {name: [] for name in ["cls", "center", "height"]}
    all_v8_domain_correct = []
    all_v8_bev_damage = RunningStats()
    all_v8_img_damage = RunningStats()
    all_v8_dual_damage = RunningStats()
    all_v8_a4_bev_damage_z = RunningStats()
    all_v8_a4_img_damage_z = RunningStats()
    all_v8_a4_cls_unc_z_bev = RunningStats()
    all_v8_a4_cls_unc_z_img = RunningStats()
    all_v8_a4_route_score_bev = RunningStats()
    all_v8_a4_route_score_img = RunningStats()
    all_v9_a_bev = RunningStats()
    all_v9_a_img = RunningStats()
    all_v9_anchor_distance_bev = RunningStats()
    all_v9_anchor_distance_img = RunningStats()
    all_v9_tail_prob_bev = RunningStats()
    all_v9_tail_prob_img = RunningStats()
    all_v9_tail_excess_bev = RunningStats()
    all_v9_tail_excess_img = RunningStats()
    all_v9_q95_shift_bev = RunningStats()
    all_v9_q95_shift_img = RunningStats()
    all_v9_route_score_bev = RunningStats()
    all_v9_route_score_img = RunningStats()
    all_v9_velocity_l1 = RunningStats()
    all_v9_above_tau = {"bev": [], "img": []}
    all_v9_clean_like = []
    all_v9_scene_scalar = defaultdict(list)
    for condition in v5_selected_counts:
        for modality, count in v5_selected_counts[condition].items():
            all_selected[modality] += count
    for condition in v5_fallback_level_counts:
        for level, count in v5_fallback_level_counts[condition].items():
            all_fallback[level] += count
    for condition in v5_switch_class_counts:
        for class_name, count in v5_switch_class_counts[condition].items():
            all_class_counts[class_name] += count
    for condition, stats in v5_switch_stats.items():
        if "-s" in condition or condition == "clean":
            total_switched += stats["switched"]
            total_switch_candidates += stats["total"]
    for condition, hist in v5_fused_anomaly_hists.items():
        if "-s" in condition or condition == "clean":
            if hist.stats.count:
                all_anomaly.count += hist.stats.count
                all_anomaly.total += hist.stats.total
    for condition, hist in v5_switch_distance_hists.items():
        if "-s" in condition or condition == "clean":
            if hist.stats.count:
                all_switched_distance.count += hist.stats.count
                all_switched_distance.total += hist.stats.total
    for condition in v6_candidate_counts:
        for modality, count in v6_candidate_counts[condition].items():
            all_v6_candidate[modality] += count
    for condition in v6_center_source_counts:
        for source, count in v6_center_source_counts[condition].items():
            all_v6_center_source[source] += count
    for condition in v6_height_source_counts:
        for source, count in v6_height_source_counts[condition].items():
            all_v6_height_source[source] += count
    for condition in v6_fallback_level_counts:
        for level, count in v6_fallback_level_counts[condition].items():
            all_v6_fallback[level] += count
    for condition, values in v6_gate_pass_anomaly_values.items():
        all_v6_gate_pass_anomaly.extend(values)
    for condition, values in v6_gate_pass_candidate_prob_values.items():
        all_v6_gate_pass_candidate_prob.extend(values)
    for condition, values in v6_gate_pass_candidate_margin_values.items():
        all_v6_gate_pass_candidate_margin.extend(values)
    for condition, values in v6_gate_pass_final_values.items():
        all_v6_gate_pass_final.extend(values)
    for condition in v6_switch_class_counts:
        for class_name, count in v6_switch_class_counts[condition].items():
            all_v6_class_counts[class_name] += count
    for condition, stats in v6_switch_stats.items():
        if "-s" in condition or condition == "clean":
            total_v6_switched += stats["switched"]
            total_v6_switch_candidates += stats["total"]
    for condition, hist in v6_gate_anomaly_hists.items():
        if "-s" in condition or condition == "clean":
            if hist.stats.count:
                all_v6_anomaly.count += hist.stats.count
                all_v6_anomaly.total += hist.stats.total
    for condition, hist in v6_switch_distance_hists.items():
        if "-s" in condition or condition == "clean":
            if hist.stats.count:
                all_v6_switched_distance.count += hist.stats.count
                all_v6_switched_distance.total += hist.stats.total
    for condition in v7_damage_state_counts:
        for state, count in v7_damage_state_counts[condition].items():
            all_v7_damage_state[state] += count
    for condition in v7_center_source_counts:
        for source, count in v7_center_source_counts[condition].items():
            all_v7_center_source[source] += count
    for condition in v7_switch_class_counts:
        for class_name, count in v7_switch_class_counts[condition].items():
            all_v7_class_counts[class_name] += count
    for condition, stats in v7_switch_stats.items():
        if "-s" in condition or condition == "clean":
            total_v7_switched += stats["switched"]
            total_v7_switch_candidates += stats["total"]
    for condition, hist in v7_bev_center_anomaly_hists.items():
        if "-s" in condition or condition == "clean":
            if hist.stats.count:
                all_v7_bev_anomaly.count += hist.stats.count
                all_v7_bev_anomaly.total += hist.stats.total
    for condition, hist in v7_img_center_anomaly_hists.items():
        if "-s" in condition or condition == "clean":
            if hist.stats.count:
                all_v7_img_anomaly.count += hist.stats.count
                all_v7_img_anomaly.total += hist.stats.total
    for condition, hist in v7_switch_distance_hists.items():
        if "-s" in condition or condition == "clean":
            if hist.stats.count:
                all_v7_switched_distance.count += hist.stats.count
                all_v7_switched_distance.total += hist.stats.total
    for condition in v8_damage_state_counts:
        for state, count in v8_damage_state_counts[condition].items():
            all_v8_damage_state[state] += count
    for condition in v8_reliability_state_counts:
        for state, count in v8_reliability_state_counts[condition].items():
            all_v8_reliability_state[state] += count
    for condition in v8_selected_source_counts:
        for source, count in v8_selected_source_counts[condition].items():
            all_v8_selected_source[source] += count
    for condition in v8_non_fused_choice_counts:
        for source, count in v8_non_fused_choice_counts[condition].items():
            all_v8_non_fused_choice[source] += count
    for values in v8_fused_keep_values.values():
        all_v8_fused_keep.extend(values)
    for values in v8_center_fused_keep_values.values():
        all_v8_center_fused_keep.extend(values)
    for values in v8_damage_gate_values.values():
        all_v8_damage_gate.extend(values)
    for values in v8_high_damage_values.values():
        all_v8_high_damage.extend(values)
    for values in v8_center_route_allowed_values.values():
        all_v8_center_route_allowed.extend(values)
    for modality_values in v8_reliability_values.values():
        for modality in ["bev", "img"]:
            all_v8_reliability[modality].extend(modality_values[modality])
    for modality_values in v8_context_damage_values.values():
        for modality in ["bev", "img"]:
            all_v8_context_damage[modality].extend(modality_values[modality])
    for values in v8_reliability_margin_values.values():
        all_v8_reliability_margin.extend(values)
    for modality_values in v8_soft_weight_values.values():
        for modality in MODALITIES:
            all_v8_soft_weight[modality].extend(modality_values[modality])
    for delta_values in v8_output_delta_values.values():
        for name in ["cls", "center", "height"]:
            all_v8_output_delta[name].extend(delta_values[name])
    for values in v8_domain_correct_values.values():
        all_v8_domain_correct.extend(values)
    for hist in v8_bev_damage_hists.values():
        if hist.stats.count:
            all_v8_bev_damage.count += hist.stats.count
            all_v8_bev_damage.total += hist.stats.total
    for hist in v8_img_damage_hists.values():
        if hist.stats.count:
            all_v8_img_damage.count += hist.stats.count
            all_v8_img_damage.total += hist.stats.total
    for hist in v8_dual_damage_hists.values():
        if hist.stats.count:
            all_v8_dual_damage.count += hist.stats.count
            all_v8_dual_damage.total += hist.stats.total
    for hist, acc in [
        (v8_a4_bev_damage_z_hists, all_v8_a4_bev_damage_z),
        (v8_a4_img_damage_z_hists, all_v8_a4_img_damage_z),
        (v8_a4_cls_unc_z_bev_hists, all_v8_a4_cls_unc_z_bev),
        (v8_a4_cls_unc_z_img_hists, all_v8_a4_cls_unc_z_img),
        (v8_a4_route_score_bev_hists, all_v8_a4_route_score_bev),
        (v8_a4_route_score_img_hists, all_v8_a4_route_score_img),
    ]:
        for condition_hist in hist.values():
            if condition_hist.stats.count:
                acc.count += condition_hist.stats.count
                acc.total += condition_hist.stats.total
    for hist, acc in [
        (v9_a_bev_hists, all_v9_a_bev),
        (v9_a_img_hists, all_v9_a_img),
        (v9_anchor_distance_bev_hists, all_v9_anchor_distance_bev),
        (v9_anchor_distance_img_hists, all_v9_anchor_distance_img),
        (v9_tail_prob_bev_hists, all_v9_tail_prob_bev),
        (v9_tail_prob_img_hists, all_v9_tail_prob_img),
        (v9_tail_excess_bev_hists, all_v9_tail_excess_bev),
        (v9_tail_excess_img_hists, all_v9_tail_excess_img),
        (v9_q95_shift_bev_hists, all_v9_q95_shift_bev),
        (v9_q95_shift_img_hists, all_v9_q95_shift_img),
        (v9_route_score_bev_hists, all_v9_route_score_bev),
        (v9_route_score_img_hists, all_v9_route_score_img),
        (v9_velocity_l1_hists, all_v9_velocity_l1),
    ]:
        for condition_hist in hist.values():
            if condition_hist.stats.count:
                acc.count += condition_hist.stats.count
                acc.total += condition_hist.stats.total
    for values in v9_above_tau_values.values():
        for modality in ["bev", "img"]:
            all_v9_above_tau[modality].extend(values[modality])
    for values in v9_clean_like_values.values():
        all_v9_clean_like.extend(values)
    for values in v9_scene_scalar_values.values():
        for name, vals in values.items():
            all_v9_scene_scalar[name].extend(vals)
    summary["v5_summary"] = {
        "mean_v5_switch_ratio": (total_switched / total_switch_candidates) if total_switch_candidates > 0 else None,
        "mean_v5_fused_anomaly": all_anomaly.mean(),
        "mean_v5_switch_to_bev_ratio": all_selected.get("bev", 0) / max(sum(all_selected.values()), 1),
        "mean_v5_switch_to_img_ratio": all_selected.get("img", 0) / max(sum(all_selected.values()), 1),
        "mean_v5_reference_fallback_ratio": {
            level: all_fallback.get(level, 0) / max(sum(all_fallback.values()), 1)
            for level in ["class_distance", "class_global", "global", "missing"]
        },
        "v5_switch_ratio_by_class": dict(all_class_counts),
        "v5_switch_ratio_by_distance_bin": {
            "mean_switched_distance": all_switched_distance.mean()
        },
    }
    summary["v6_summary"] = {
        "mean_v6_switch_ratio": (total_v6_switched / total_v6_switch_candidates) if total_v6_switch_candidates > 0 else None,
        "mean_v6_gate_anomaly": all_v6_anomaly.mean(),
        "mean_v6_gate_pass_anomaly_ratio": (
            float(np.mean(all_v6_gate_pass_anomaly)) if all_v6_gate_pass_anomaly else None
        ),
        "mean_v6_gate_pass_candidate_prob_ratio": (
            float(np.mean(all_v6_gate_pass_candidate_prob)) if all_v6_gate_pass_candidate_prob else None
        ),
        "mean_v6_gate_pass_candidate_margin_ratio": (
            float(np.mean(all_v6_gate_pass_candidate_margin)) if all_v6_gate_pass_candidate_margin else None
        ),
        "mean_v6_gate_pass_final_ratio": (
            float(np.mean(all_v6_gate_pass_final)) if all_v6_gate_pass_final else None
        ),
        "mean_v6_candidate_modality_ratio": {
            modality: all_v6_candidate.get(modality, 0) / max(sum(all_v6_candidate.values()), 1)
            for modality in ["bev", "img"]
        },
        "mean_v6_center_source_ratio": {
            source: all_v6_center_source.get(source, 0) / max(sum(all_v6_center_source.values()), 1)
            for source in SOURCE_CATEGORIES
        },
        "mean_v6_height_source_ratio": {
            source: all_v6_height_source.get(source, 0) / max(sum(all_v6_height_source.values()), 1)
            for source in SOURCE_CATEGORIES
        },
        "mean_v6_reference_fallback_ratio": {
            level: all_v6_fallback.get(level, 0) / max(sum(all_v6_fallback.values()), 1)
            for level in ["class_distance", "class_global", "global", "missing"]
        },
        "v6_switch_ratio_by_class": dict(all_v6_class_counts),
        "v6_switch_ratio_by_distance_bin": {
            "mean_switched_distance": all_v6_switched_distance.mean()
        },
    }
    summary["v7_summary"] = {
        "mean_v7_switch_ratio": (total_v7_switched / total_v7_switch_candidates) if total_v7_switch_candidates > 0 else None,
        "mean_v7_bev_center_anom": all_v7_bev_anomaly.mean(),
        "mean_v7_img_center_anom": all_v7_img_anomaly.mean(),
        "mean_v7_damage_state_ratio": {
            state: all_v7_damage_state.get(state, 0) / max(sum(all_v7_damage_state.values()), 1)
            for state in V7_DAMAGE_STATES
        },
        "mean_v7_center_source_ratio": {
            source: all_v7_center_source.get(source, 0) / max(sum(all_v7_center_source.values()), 1)
            for source in V7_CENTER_SOURCE_CATEGORIES
        },
        "v7_switch_ratio_by_class": dict(all_v7_class_counts),
        "v7_switch_ratio_by_distance_bin": {
            "mean_switched_distance": all_v7_switched_distance.mean()
        },
    }
    summary["v8_summary"] = {
        "mean_v8_fused_keep": float(np.mean(all_v8_fused_keep)) if all_v8_fused_keep else None,
        "mean_v8_center_fused_keep": (
            float(np.mean(all_v8_center_fused_keep)) if all_v8_center_fused_keep else None
        ),
        "mean_v8_damage_gate": float(np.mean(all_v8_damage_gate)) if all_v8_damage_gate else None,
        "mean_v8_high_damage_query_ratio": float(np.mean(all_v8_high_damage)) if all_v8_high_damage else None,
        "mean_v8_center_route_allowed_ratio": (
            float(np.mean(all_v8_center_route_allowed)) if all_v8_center_route_allowed else None
        ),
        "mean_v8_bev_damage": all_v8_bev_damage.mean(),
        "mean_v8_img_damage": all_v8_img_damage.mean(),
        "mean_v8_dual_damage": all_v8_dual_damage.mean(),
        "mean_v8_context_damage": {
            modality: (
                float(np.mean(all_v8_context_damage[modality]))
                if all_v8_context_damage[modality] else None
            )
            for modality in ["bev", "img"]
        },
        "mean_v8_a4_damage_z": {
            "bev": all_v8_a4_bev_damage_z.mean(),
            "img": all_v8_a4_img_damage_z.mean(),
        },
        "mean_v8_a4_cls_unc_z": {
            "bev": all_v8_a4_cls_unc_z_bev.mean(),
            "img": all_v8_a4_cls_unc_z_img.mean(),
        },
        "mean_v8_a4_route_score": {
            "bev": all_v8_a4_route_score_bev.mean(),
            "img": all_v8_a4_route_score_img.mean(),
        },
        "mean_v8_damage_state_ratio": {
            state: all_v8_damage_state.get(state, 0) / max(sum(all_v8_damage_state.values()), 1)
            for state in V8_DAMAGE_STATES
        },
        "mean_v8_reliability": {
            modality: float(np.mean(all_v8_reliability[modality])) if all_v8_reliability[modality] else None
            for modality in ["bev", "img"]
        },
        "mean_v8_reliability_margin": (
            float(np.mean(all_v8_reliability_margin)) if all_v8_reliability_margin else None
        ),
        "mean_v8_reliability_state_ratio": {
            state: all_v8_reliability_state.get(state, 0) / max(sum(all_v8_reliability_state.values()), 1)
            for state in V8_RELIABILITY_STATES
        },
        "mean_v8_soft_weight_ratio": {
            modality: float(np.mean(all_v8_soft_weight[modality])) if all_v8_soft_weight[modality] else None
            for modality in MODALITIES
        },
        "mean_v8_output_delta": {
            name: float(np.mean(all_v8_output_delta[name])) if all_v8_output_delta[name] else None
            for name in ["cls", "center", "height"]
        },
        "mean_v8_domain_correct_route_ratio": (
            float(np.mean(all_v8_domain_correct)) if all_v8_domain_correct else None
        ),
        "mean_v8_selected_source_ratio": {
            source: all_v8_selected_source.get(source, 0) / max(sum(all_v8_selected_source.values()), 1)
            for source in V8_SOURCE_CATEGORIES
        },
        "mean_v8_non_fused_choice_ratio": {
            source: all_v8_non_fused_choice.get(source, 0) / max(sum(all_v8_non_fused_choice.values()), 1)
            for source in ["bev", "img"]
        },
    }
    summary["v9_summary"] = {
        "mean_v9_anomaly": {
            "bev": all_v9_a_bev.mean(),
            "img": all_v9_a_img.mean(),
        },
        "mean_v9_above_tau_ratio": {
            modality: (
                float(np.mean(all_v9_above_tau[modality]))
                if all_v9_above_tau[modality] else None
            )
            for modality in ["bev", "img"]
        },
        "mean_v9_clean_like_ratio": (
            float(np.mean(all_v9_clean_like)) if all_v9_clean_like else None
        ),
        "mean_v9_anchor_distance": {
            "bev": all_v9_anchor_distance_bev.mean(),
            "img": all_v9_anchor_distance_img.mean(),
        },
        "mean_v9_tail_prob": {
            "bev": all_v9_tail_prob_bev.mean(),
            "img": all_v9_tail_prob_img.mean(),
        },
        "mean_v9_tail_excess": {
            "bev": all_v9_tail_excess_bev.mean(),
            "img": all_v9_tail_excess_img.mean(),
        },
        "mean_v9_q95_shift": {
            "bev": all_v9_q95_shift_bev.mean(),
            "img": all_v9_q95_shift_img.mean(),
        },
        "mean_v9_route_score": {
            "bev": all_v9_route_score_bev.mean(),
            "img": all_v9_route_score_img.mean(),
        },
        "mean_v9_velocity_l1": all_v9_velocity_l1.mean(),
        "mean_v9_scene_scalar": {
            name: (float(np.mean(vals)) if vals else None)
            for name, vals in all_v9_scene_scalar.items()
        },
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def compute_robustness_metrics(results, clean_nds, clean_map):
    all_corruptions = sorted(set(c for c, _ in results.keys()))
    ra_nds_per_c = {}
    ra_map_per_c = {}
    rra_nds_per_c = {}

    for corr in all_corruptions:
        nds_vals = []
        map_vals = []
        for sev in [1, 2, 3]:
            if (corr, sev) not in results:
                continue
            nds, m_ap = results[(corr, sev)]
            if clean_nds > 0:
                nds_vals.append(nds / clean_nds)
            if clean_map > 0:
                map_vals.append(m_ap / clean_map)
        if nds_vals:
            ra_nds_per_c[corr] = float(np.mean(nds_vals))
        if map_vals:
            ra_map_per_c[corr] = float(np.mean(map_vals))
        if corr in ra_nds_per_c and corr in BASELINE_RA_NDS and clean_nds > 0:
            rra_nds_per_c[corr] = (
                (ra_nds_per_c[corr] * clean_nds) / (BASELINE_RA_NDS[corr] * BASELINE_CLEAN_NDS) - 1.0
            ) * 100.0

    return {
        "baseline_name": BASELINE_NAME,
        "baseline_clean_nds": BASELINE_CLEAN_NDS,
        "baseline_ra_nds": BASELINE_RA_NDS,
        "ra_nds_per_c": ra_nds_per_c,
        "ra_map_per_c": ra_map_per_c,
        "rra_nds_per_c": rra_nds_per_c,
        "mra_nds": float(np.mean(list(ra_nds_per_c.values()))) if ra_nds_per_c else None,
        "mra_map": float(np.mean(list(ra_map_per_c.values()))) if ra_map_per_c else None,
        "mrra_nds": float(np.mean(list(rra_nds_per_c.values()))) if rra_nds_per_c else None,
    }


def plot_severity_heatmaps(results, out_path):
    corruptions = sorted(set(c for c, _ in results.keys()))
    if not corruptions:
        return
    severities = [1, 2, 3]
    nds_mat = np.full((len(corruptions), len(severities)), np.nan)
    map_mat = np.full((len(corruptions), len(severities)), np.nan)
    for i, corr in enumerate(corruptions):
        for j, sev in enumerate(severities):
            if (corr, sev) in results:
                nds_mat[i, j], map_mat[i, j] = results[(corr, sev)]

    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, len(corruptions) * 0.5)))
    for ax, mat, title in zip(axes, [nds_mat, map_mat], ["NDS", "mAP"]):
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_title(f"{title} by Corruption / Severity")
        ax.set_xticks(range(len(severities)))
        ax.set_xticklabels([f"s{sev}" for sev in severities])
        ax.set_yticks(range(len(corruptions)))
        ax.set_yticklabels(corruptions)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_ra_bars(metrics, out_path):
    corruptions = list(metrics["ra_nds_per_c"].keys())
    if not corruptions:
        return
    x = np.arange(len(corruptions))
    width = 0.36
    ra_nds = [metrics["ra_nds_per_c"][c] for c in corruptions]
    ra_map = [metrics["ra_map_per_c"].get(c, np.nan) for c in corruptions]

    plt.figure(figsize=(max(10, len(corruptions) * 1.1), 5))
    plt.bar(x - width / 2, ra_nds, width=width, label="RA-NDS", color="#4c78a8")
    plt.bar(x + width / 2, ra_map, width=width, label="RA-mAP", color="#f58518")
    plt.axhline(metrics["mra_nds"], color="#4c78a8", linestyle="--", linewidth=1.2, label="mRA-NDS")
    if metrics["mra_map"] is not None:
        plt.axhline(metrics["mra_map"], color="#f58518", linestyle="--", linewidth=1.2, label="mRA-mAP")
    plt.xticks(x, corruptions, rotation=30, ha="right")
    plt.ylabel("Resistance Ability")
    plt.title("RA by Corruption")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_rra_bars(metrics, out_path):
    corruptions = list(metrics["rra_nds_per_c"].keys())
    if not corruptions:
        return
    baseline_name = metrics.get("baseline_name", BASELINE_NAME)
    vals = [metrics["rra_nds_per_c"][c] for c in corruptions]
    x = np.arange(len(corruptions))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]

    plt.figure(figsize=(max(10, len(corruptions) * 1.1), 5))
    plt.bar(x, vals, color=colors)
    plt.axhline(0, color="black", linewidth=1.0)
    if metrics["mrra_nds"] is not None:
        plt.axhline(metrics["mrra_nds"], color="#9467bd", linestyle="--", linewidth=1.2, label=f"mRRA={metrics['mrra_nds']:.2f}%")
    plt.xticks(x, corruptions, rotation=30, ha="right")
    plt.ylabel("RRA-NDS (%)")
    plt.title(f"Relative Resistance Ability vs {baseline_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-log")
    parser.add_argument("--clean-nds", type=float)
    parser.add_argument("--clean-map", type=float)
    parser.add_argument("--jobs", type=int, default=default_num_jobs())
    parser.add_argument("--heartbeat-samples", type=int, default=default_heartbeat_samples())
    parser.add_argument("--strict-v6", action="store_true")
    parser.add_argument("--strict-v7", action="store_true")
    parser.add_argument("--strict-v9", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    summary_dir = os.path.join(args.output_dir, "summary")
    qualitative_dir = os.path.join(args.output_dir, "qualitative")
    ensure_dir(summary_dir)
    ensure_dir(qualitative_dir)

    manifest_rows = load_manifest(args.manifest)
    aggregates = aggregate_by_condition(
        manifest_rows,
        jobs=max(1, args.jobs),
        heartbeat_samples=max(1, args.heartbeat_samples),
    )
    validate_v6_diagnostics(aggregates, strict_v6=args.strict_v6)
    validate_v7_diagnostics(aggregates, strict_v7=args.strict_v7)
    validate_v9_diagnostics(aggregates, strict_v9=args.strict_v9)

    plot_stacked_weights(
        aggregates["scene_weights"],
        os.path.join(summary_dir, "modality_weights_by_corruption.png"))
    plot_selected_ratio(
        aggregates["selected_counts"],
        os.path.join(summary_dir, "selected_modality_by_corruption.png"))
    plot_entropy_and_fused(
        aggregates["entropy_values"],
        aggregates["fused_weight_values"],
        os.path.join(summary_dir, "fused_weight_and_entropy.png"))
    plot_uncertainty_histograms(
        aggregates["scene_uncs"],
        summary_dir)
    plot_distance_uncertainty_scatter(
        aggregates["scatter_points"],
        os.path.join(summary_dir, "distance_vs_uncertainty.png"))
    plot_distance_prior_vs_final_selection(
        aggregates["distance_prior_top1"],
        aggregates["selected_counts"],
        os.path.join(summary_dir, "distance_prior_vs_final_selection.png"))
    plot_hard_query_residual_activation(
        aggregates["alpha_res_values"],
        aggregates["residual_override_values"],
        aggregates["tail_query_values"],
        os.path.join(summary_dir, "hard_query_residual_activation.png"))
    plot_v5_switch_ratio(
        aggregates["v5_switch_stats"],
        os.path.join(summary_dir, "v5_switch_ratio_by_corruption.png"))
    plot_v5_switch_ratio_heatmap(
        aggregates["v5_switch_stats"],
        os.path.join(summary_dir, "v5_switch_ratio_by_corruption_severity.png"))
    plot_v5_anomaly_histograms(
        aggregates["v5_fused_anomaly_hists"],
        summary_dir)
    plot_v5_switch_distance_hist(
        aggregates["v5_switch_distance_hists"],
        os.path.join(summary_dir, "v5_switch_distance_hist.png"))
    plot_v5_switch_by_class(
        aggregates["v5_switch_class_counts"],
        os.path.join(summary_dir, "v5_switch_by_class.png"))
    plot_v5_reference_fallback_ratio(
        aggregates["v5_fallback_level_counts"],
        os.path.join(summary_dir, "v5_reference_fallback_ratio.png"))
    plot_v6_switch_ratio(
        aggregates["v6_switch_stats"],
        os.path.join(summary_dir, "v6_switch_ratio_by_corruption.png"))
    plot_v6_anomaly_histograms(
        aggregates["v6_gate_anomaly_hists"],
        summary_dir)
    plot_ratio_counts(
        aggregates["v6_candidate_counts"],
        ["bev", "img"],
        MODALITY_COLORS,
        "Candidate Modality Ratio",
        "V6 Candidate Modality by Corruption",
        os.path.join(summary_dir, "v6_candidate_modality_by_corruption.png"))
    plot_ratio_counts(
        aggregates["v6_center_source_counts"],
        SOURCE_CATEGORIES,
        SOURCE_COLORS,
        "Center Source Ratio",
        "V6 Center Source by Corruption",
        os.path.join(summary_dir, "v6_center_source_by_corruption.png"))
    plot_ratio_counts(
        aggregates["v6_height_source_counts"],
        SOURCE_CATEGORIES,
        SOURCE_COLORS,
        "Height Source Ratio",
        "V6 Height Source by Corruption",
        os.path.join(summary_dir, "v6_height_source_by_corruption.png"))
    plot_v6_switch_distance_hist(
        aggregates["v6_switch_distance_hists"],
        os.path.join(summary_dir, "v6_switch_distance_hist.png"))
    plot_v6_switch_by_class(
        aggregates["v6_switch_class_counts"],
        os.path.join(summary_dir, "v6_switch_by_class.png"))
    plot_v6_reference_fallback_ratio(
        aggregates["v6_fallback_level_counts"],
        os.path.join(summary_dir, "v6_reference_fallback_ratio.png"))
    plot_v7_switch_ratio(
        aggregates["v7_switch_stats"],
        os.path.join(summary_dir, "v7_switch_ratio_by_corruption.png"))
    plot_ratio_counts(
        aggregates["v7_center_source_counts"],
        V7_CENTER_SOURCE_CATEGORIES,
        V7_CENTER_SOURCE_COLORS,
        "Center Source Ratio",
        "V7 Center Source by Corruption",
        os.path.join(summary_dir, "v7_center_source_by_corruption.png"))
    plot_ratio_counts(
        aggregates["v7_damage_state_counts"],
        V7_DAMAGE_STATES,
        V7_DAMAGE_COLORS,
        "Damage State Ratio",
        "V7 Damage State by Corruption",
        os.path.join(summary_dir, "v7_damage_state_ratio.png"))
    plot_v7_bev_center_anomaly_histograms(
        aggregates["v7_bev_center_anomaly_hists"],
        summary_dir)
    plot_v7_img_center_anomaly_histograms(
        aggregates["v7_img_center_anomaly_hists"],
        summary_dir)
    plot_v7_switch_distance_hist(
        aggregates["v7_switch_distance_hists"],
        os.path.join(summary_dir, "v7_switch_distance_hist.png"))
    plot_v7_switch_by_class(
        aggregates["v7_switch_class_counts"],
        os.path.join(summary_dir, "v7_switch_by_class.png"))
    plot_v8_gate_means(
        aggregates["v8_fused_keep_values"],
        aggregates["v8_damage_gate_values"],
        aggregates["v8_high_damage_values"],
        os.path.join(summary_dir, "v8_gate_response_by_corruption.png"))
    plot_ratio_counts(
        aggregates["v8_damage_state_counts"],
        V8_DAMAGE_STATES,
        V8_DAMAGE_COLORS,
        "Damage State Ratio",
        "V8 Damage State by Corruption",
        os.path.join(summary_dir, "v8_damage_state_ratio.png"))
    plot_ratio_counts(
        aggregates["v8_reliability_state_counts"],
        V8_RELIABILITY_STATES,
        V8_RELIABILITY_COLORS,
        "Reliability State Ratio",
        "V8 Reliability State by Corruption",
        os.path.join(summary_dir, "v8_reliability_state_ratio.png"))
    plot_ratio_counts(
        aggregates["v8_selected_source_counts"],
        V8_SOURCE_CATEGORIES,
        V8_SOURCE_COLORS,
        "Selected Source Ratio",
        "V8 Selected Source by Corruption",
        os.path.join(summary_dir, "v8_selected_source_by_corruption.png"))
    plot_ratio_counts(
        aggregates["v8_non_fused_choice_counts"],
        ["bev", "img"],
        MODALITY_COLORS,
        "Non-Fused Choice Ratio",
        "V8 BEV/IMG Choice by Corruption",
        os.path.join(summary_dir, "v8_non_fused_choice_by_corruption.png"))
    plot_v8_reliability_means(
        aggregates["v8_reliability_values"],
        aggregates["v8_reliability_margin_values"],
        os.path.join(summary_dir, "v8_reliability_by_corruption.png"))
    plot_v8_soft_weight_means(
        aggregates["v8_soft_weight_values"],
        os.path.join(summary_dir, "v8_soft_weights_by_corruption.png"))
    plot_v8_output_delta_means(
        aggregates["v8_output_delta_values"],
        os.path.join(summary_dir, "v8_output_delta_by_corruption.png"))
    plot_v8_domain_correct(
        aggregates["v8_domain_correct_values"],
        os.path.join(summary_dir, "v8_domain_correct_route_by_corruption.png"))
    plot_v8_reliability_histograms(aggregates["v8_reliability_values"], summary_dir)
    plot_v8_damage_histograms(aggregates["v8_bev_damage_hists"], summary_dir, "bev")
    plot_v8_damage_histograms(aggregates["v8_img_damage_hists"], summary_dir, "img")
    plot_v8_damage_histograms(aggregates["v8_dual_damage_hists"], summary_dir, "dual")
    plot_v8_scalar_histograms(
        aggregates["v8_a4_bev_damage_z_hists"], summary_dir,
        "a4_bev_damage_z", "V8 A4 BEV Clean-Anchor Damage z")
    plot_v8_scalar_histograms(
        aggregates["v8_a4_img_damage_z_hists"], summary_dir,
        "a4_img_damage_z", "V8 A4 IMG Clean-Anchor Damage z")
    plot_v8_scalar_histograms(
        aggregates["v8_a4_cls_unc_z_bev_hists"], summary_dir,
        "a4_cls_unc_z_bev", "V8 A4 BEV cls_unc z")
    plot_v8_scalar_histograms(
        aggregates["v8_a4_cls_unc_z_img_hists"], summary_dir,
        "a4_cls_unc_z_img", "V8 A4 IMG cls_unc z")
    plot_v8_scalar_histograms(
        aggregates["v8_a4_route_score_bev_hists"], summary_dir,
        "a4_route_score_bev", "V8 A4 BEV Route Score")
    plot_v8_scalar_histograms(
        aggregates["v8_a4_route_score_img_hists"], summary_dir,
        "a4_route_score_img", "V8 A4 IMG Route Score")
    plot_v9_scalar_histograms(
        aggregates["v9_a_bev_hists"], summary_dir,
        "a_bev", "V9 BEV Anomaly Score")
    plot_v9_scalar_histograms(
        aggregates["v9_a_img_hists"], summary_dir,
        "a_img", "V9 IMG Anomaly Score")
    plot_v9_scalar_histograms(
        aggregates["v9_anchor_distance_bev_hists"], summary_dir,
        "anchor_distance_bev", "V9 BEV Positive RMS Anchor Distance")
    plot_v9_scalar_histograms(
        aggregates["v9_anchor_distance_img_hists"], summary_dir,
        "anchor_distance_img", "V9 IMG Positive RMS Anchor Distance")
    plot_v9_scalar_histograms(
        aggregates["v9_tail_excess_bev_hists"], summary_dir,
        "tail_excess_bev", "V9 BEV Per-Query Tail Excess")
    plot_v9_scalar_histograms(
        aggregates["v9_tail_excess_img_hists"], summary_dir,
        "tail_excess_img", "V9 IMG Per-Query Tail Excess")
    plot_v9_scalar_histograms(
        aggregates["v9_q95_shift_bev_hists"], summary_dir,
        "q95_shift_bev", "V9 BEV q95 Shift")
    plot_v9_scalar_histograms(
        aggregates["v9_q95_shift_img_hists"], summary_dir,
        "q95_shift_img", "V9 IMG q95 Shift")
    plot_v9_scalar_histograms(
        aggregates["v9_route_score_bev_hists"], summary_dir,
        "route_score_bev", "V9 BEV Route Score")
    plot_v9_scalar_histograms(
        aggregates["v9_route_score_img_hists"], summary_dir,
        "route_score_img", "V9 IMG Route Score")
    plot_v9_scalar_histograms(
        aggregates["v9_velocity_l1_hists"], summary_dir,
        "velocity_l1", "V9 Fused vs Soft-Mix Velocity L1")
    plot_v9_ratio_summary(
        aggregates["v9_above_tau_values"],
        aggregates["v9_clean_like_values"],
        os.path.join(summary_dir, "v9_anomaly_tau_and_clamp_by_corruption.png"))
    plot_v9_scene_scalar_evolution(
        aggregates["v9_scene_scalar_values"],
        os.path.join(summary_dir, "v9_clean_anchor_tau_evolution.png"))

    if args.eval_log and args.clean_nds is not None and args.clean_map is not None:
        results = parse_eval_log(args.eval_log)
        metrics = compute_robustness_metrics(results, args.clean_nds, args.clean_map)
        plot_severity_heatmaps(
            results,
            os.path.join(summary_dir, "nds_map_heatmaps.png"))
        plot_ra_bars(
            metrics,
            os.path.join(summary_dir, "ra_by_corruption.png"))
        plot_rra_bars(
            metrics,
            os.path.join(summary_dir, "rra_nds_by_corruption.png"))
        with open(os.path.join(summary_dir, "robustness_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    for condition, sample_info in aggregates["representative_samples"].items():
        out_path = os.path.join(qualitative_dir, f"bev_selected_modality_{condition}.png")
        plot_bev_qualitative(sample_info, out_path)

    write_summary_json(
        aggregates,
        os.path.join(summary_dir, "visualization_summary.json"))

    print(f"[EDL Viz] Saved summary plots to: {summary_dir}", flush=True)
    print(f"[EDL Viz] Saved qualitative plots to: {qualitative_dir}", flush=True)


if __name__ == "__main__":
    main()
