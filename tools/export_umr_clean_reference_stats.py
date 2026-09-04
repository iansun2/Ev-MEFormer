#!/usr/bin/env python3
"""Export clean-reference branch uncertainty stats for UMR CRG.

The output schema is intentionally compatible with the existing v7 reference
stats reader in ``EDLPMETransformer``. Inputs can be EV sidecar/light logs or
legacy EDL light logs, stored as plain JSONL, gzip, or zstd.
"""

import argparse
import glob
import gzip
import hashlib
import json
import math
import os
from collections import defaultdict


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
        return float(value)
    except (TypeError, ValueError):
        return None


def bucketize_distance(distance, bin_edges):
    for idx, edge in enumerate(bin_edges):
        if distance < edge:
            return idx
    return len(bin_edges)


def compute_stats(values):
    if not values:
        return None
    arr = sorted(float(value) for value in values)
    median = percentile(arr, 50)
    abs_dev = sorted(abs(value - median) for value in arr)
    mad = percentile(abs_dev, 50)
    return {
        "count": int(len(arr)),
        "median": median,
        "mad": mad,
        "p80": percentile(arr, 80),
        "p90": percentile(arr, 90),
        "p95": percentile(arr, 95),
        "p99": percentile(arr, 99),
        "p995": percentile(arr, 99.5),
    }


def percentile(sorted_values, q):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * (float(q) / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


def open_text(path):
    if path.endswith((".gz", ".gzip")):
        return gzip.open(path, "rt", encoding="utf-8")
    if path.endswith((".zst", ".zstd")):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError(
                "Reading zstd EV logs requires the Python package 'zstandard'. "
                "Install it or export reference logs with EV_LOG_COMPRESSION=none/gzip."
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


def find_jsonl_logs(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    patterns = [
        "ev_prediction_light*.jsonl",
        "ev_prediction_light*.jsonl.gz",
        "ev_prediction_light*.jsonl.zst",
        "edl_prediction_light.jsonl",
        "edl_prediction_light.jsonl.gz",
        "edl_prediction_light.jsonl.zst",
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(input_path, "**", pattern), recursive=True))
    return sorted(set(files))


def iter_jsonl_records(paths):
    for path in paths:
        with open_text(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield path, json.loads(line)


def scene_score(values, top_fraction=0.10):
    if not values:
        return None, None
    arr = [float(value) for value in values]
    mean_value = float(sum(arr) / len(arr))
    k = max(1, int(math.ceil(len(arr) * top_fraction)))
    top_values = sorted(arr, reverse=True)[:k]
    top_value = float(sum(top_values) / len(top_values))
    return mean_value, top_value


SCENE_SIGNAL_FIELDS = (
    "v8_a4_cls_unc_z_bev",
    "v8_a4_cls_unc_z_img",
    "v8_a4_bev_damage_z",
    "v8_a4_img_damage_z",
    "v8_gscrg_raw_unc_fused",
    "v8_gscrg_raw_unc_bev",
    "v8_gscrg_raw_unc_img",
    "v8_bev_damage",
    "v8_img_damage",
    "v9_tail_prob_bev",
    "v9_tail_prob_img",
    "v9_tail_excess_bev",
    "v9_tail_excess_img",
    "v10_p_dmg_bev",
    "v10_p_dmg_img",
    "v10_p_dmg_dual",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "--input-log",
        dest="input_log",
        required=True,
        help="Directory or JSONL file containing clean EV/EDL prediction-light logs.")
    parser.add_argument("--output", required=True, help="Output clean-reference stats JSON.")
    parser.add_argument("--distance-bins", nargs="*", type=float, default=[20.0, 40.0])
    args = parser.parse_args()

    if not os.path.exists(args.input_log):
        raise FileNotFoundError(args.input_log)
    log_paths = find_jsonl_logs(args.input_log)
    if not log_paths:
        raise FileNotFoundError(f"No prediction-light JSONL logs found under: {args.input_log}")

    class_distance = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    class_global = defaultdict(lambda: defaultdict(list))
    global_stats = defaultdict(list)
    scene_raw_acc = defaultdict(lambda: defaultdict(list))
    record_count = 0
    usable_count = 0

    for _path, pred in iter_jsonl_records(log_paths):
        record_count += 1
        label = pred.get("v7_ref_pred_class", pred.get("label"))
        if label is None:
            continue

        branch_payload = {
            "fused_center_unc": (
                safe_float(pred.get("v7_ref_fused_center_unc")),
                safe_float(pred.get("v7_ref_fused_distance", pred.get("distance_to_ego"))),
            ),
            "bev_center_unc": (
                safe_float(pred.get("v7_ref_bev_center_unc")),
                safe_float(pred.get("v7_ref_bev_distance", pred.get("distance_to_ego"))),
            ),
            "img_center_unc": (
                safe_float(pred.get("v7_ref_img_center_unc")),
                safe_float(pred.get("v7_ref_img_distance", pred.get("distance_to_ego"))),
            ),
        }

        class_key = str(int(label))
        has_any = False
        for stat_name, (value, distance) in branch_payload.items():
            if value is None or distance is None:
                continue
            branch_name = stat_name.replace("_center_unc", "")
            sample_token = pred.get("sample_token")
            if sample_token is not None:
                scene_raw_acc[str(sample_token)][f"{branch_name}_unc"].append(value)
            bin_idx = bucketize_distance(distance, args.distance_bins)
            bin_key = str(int(bin_idx))
            class_distance[class_key][bin_key][stat_name].append(value)
            class_global[class_key][stat_name].append(value)
            global_stats[stat_name].append(value)
            has_any = True

        sample_token = pred.get("sample_token")
        if sample_token is not None:
            for field_name in SCENE_SIGNAL_FIELDS:
                value = safe_float(pred.get(field_name))
                if value is not None:
                    scene_raw_acc[str(sample_token)][field_name].append(value)
                    global_stats[field_name].append(value)
                    has_any = True
        if has_any:
            usable_count += 1

    output = {
        "format": "umr_clean_reference_stats_v1",
        "distance_bins": list(args.distance_bins),
        "source_files": log_paths,
        "source_file_sha256": {path: sha256_file(path) for path in log_paths},
        "records": int(record_count),
        "usable_records": int(usable_count),
        "class_distance": {},
        "class_global": {},
        "global": {},
    }

    for class_key, bin_map in class_distance.items():
        output["class_distance"][class_key] = {}
        for bin_key, stat_map in bin_map.items():
            output["class_distance"][class_key][bin_key] = {}
            for stat_name, values in stat_map.items():
                stat = compute_stats(values)
                if stat is not None:
                    output["class_distance"][class_key][bin_key][stat_name] = stat

    for class_key, stat_map in class_global.items():
        output["class_global"][class_key] = {}
        for stat_name, values in stat_map.items():
            stat = compute_stats(values)
            if stat is not None:
                output["class_global"][class_key][stat_name] = stat

    for stat_name, values in global_stats.items():
        stat = compute_stats(values)
        if stat is not None:
            output["global"][stat_name] = stat

    scene_global = defaultdict(list)
    for _sample_token, sample_values in scene_raw_acc.items():
        bev_values = sample_values.get("bev_unc", [])
        img_values = sample_values.get("img_unc", [])
        fused_values = sample_values.get("fused_unc", [])
        n = min(len(bev_values), len(img_values))
        margin_values = [
            float(bev_values[idx]) - float(img_values[idx])
            for idx in range(n)
        ]
        for prefix, values in (
                ("bev", bev_values),
                ("img", img_values),
                ("fused", fused_values),
                ("margin", margin_values)):
            mean_value, top_value = scene_score(values)
            if mean_value is not None:
                scene_global[f"{prefix}_mean_unc"].append(mean_value)
                scene_global[f"{prefix}_top10_unc"].append(top_value)
        for field_name in SCENE_SIGNAL_FIELDS:
            values = sample_values.get(field_name, [])
            mean_value, top_value = scene_score(values)
            if mean_value is not None:
                scene_global[f"{field_name}_mean"].append(mean_value)
                scene_global[f"{field_name}_top10"].append(top_value)

    output["scene_global"] = {}
    for stat_name, values in scene_global.items():
        stat = compute_stats(values)
        if stat is not None:
            output["scene_global"][stat_name] = stat

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[UMR-CRG] logs={len(log_paths)} records={record_count} usable={usable_count}")
    print(f"[UMR-CRG] saved clean-reference stats: {args.output}")


if __name__ == "__main__":
    main()
