#!/usr/bin/env python3
"""Runtime visualization for finite IMG supplement decisions.

This tool is intentionally separate from runtime_visualize.py.  It reuses the
runtime data/model setup helpers, but renders the supplement mechanism with
decoded-box aligned source metadata.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parent
sys.path.insert(0, str(TOOL_DIR))
sys.path.insert(0, str(REPO_ROOT))

import runtime_visualize as rv  # noqa: E402


SOURCE_COLORS_RGB = {
    "fused_retained": (255, 234, 0),
    "img_supplement": (255, 23, 68),
    "unknown": (176, 176, 176),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Runtime visualization for finite IMG supplement decisions."
    )
    parser.add_argument("config", nargs="?", help="Config path. Optional with --pack/--candidate.")
    parser.add_argument("checkpoint", nargs="?", help="Checkpoint path. Optional with --pack/--candidate.")
    parser.add_argument("--pack", help="Runtime pack root with summaries/candidates.tsv.")
    parser.add_argument("--candidate", help="Candidate name inside --pack.")
    parser.add_argument("--model-label", default=None, help="Output label. Defaults to candidate/config name.")
    parser.add_argument("--out-dir", default=None, help="Output directory.")
    parser.add_argument("--clean-data-root", default="data/nuscenes", help="Clean nuScenes data root.")
    parser.add_argument(
        "--multicorrupt-root",
        default="/media/ting/Dataset/MultiCorrupt",
        help="MultiCorrupt root containing <corruption>/<severity>.",
    )
    parser.add_argument("--split", default="corrupt", help="Comma/space list: clean,corrupt.")
    parser.add_argument("--corruptions", default="beamsreducing", help="Comma/space corruption list.")
    parser.add_argument("--severities", default="3", help="Comma/space severity list.")
    parser.add_argument("--samples", type=int, default=8, help="Rendered samples per condition.")
    parser.add_argument("--start-index", type=int, default=0, help="Dataset index to start from.")
    parser.add_argument("--sample-indices", default="", help="Comma/space explicit dataset indices.")
    parser.add_argument("--sample-tokens", default="", help="Comma/space explicit nuScenes sample tokens.")
    parser.add_argument(
        "--selection",
        choices=("added", "gate", "first", "added_then_gate"),
        default="added_then_gate",
        help="Sample selection policy. added_then_gate falls back to gate-open then first samples.",
    )
    parser.add_argument("--scan-limit", type=int, default=0, help="Max scanned samples per pass; 0 means no cap.")
    parser.add_argument("--camera", default="CAM_FRONT", help="nuScenes camera key.")
    parser.add_argument(
        "--vis-points-source",
        choices=("raw", "keyframe", "pipeline"),
        default="raw",
        help="Point cloud used for visualization.",
    )
    parser.add_argument("--keyframe-time-eps", type=float, default=1e-4)
    parser.add_argument("--score-thr", type=float, default=0.30, help="Prediction score threshold.")
    parser.add_argument("--max-boxes", type=int, default=30, help="Max boxes rendered per sample.")
    parser.add_argument(
        "--duplicate-filter-thr",
        type=float,
        default=0.0,
        help=(
            "Visualization-only decoded BEV center-distance threshold for "
            "removing duplicate final boxes. 0 disables filtering."
        ),
    )
    parser.add_argument(
        "--duplicate-filter-policy",
        choices=("score_then_source", "source_then_score"),
        default="score_then_source",
        help=(
            "Policy for choosing which duplicate box remains. score_then_source "
            "keeps the higher final detection score first; source_then_score "
            "uses --duplicate-filter-source-priority first."
        ),
    )
    parser.add_argument(
        "--duplicate-filter-source-priority",
        default="fused_retained,img_supplement,unknown",
        help=(
            "Comma/space source priority used for duplicate filtering ties or "
            "source-first filtering. First source has highest priority."
        ),
    )
    parser.add_argument(
        "--duplicate-filter-class-agnostic",
        action="store_true",
        help="Remove duplicates across classes instead of only within the same class.",
    )
    parser.add_argument(
        "--duplicate-filter-all-sources",
        action="store_true",
        help=(
            "Remove duplicates across all source pairs, including same-source and "
            "unknown-source pairs. By default the visual filter only removes "
            "cross-source fused/img duplicate boxes."
        ),
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id.")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--fuse-conv-bn", action="store_true", help="Fuse conv/bn before inference.")
    parser.add_argument("--cfg-options", nargs="+", action=rv.MMCVDictAction, help="Config overrides.")
    parser.add_argument("--keep-assembled", action="store_true", help="Keep temporary assembled roots.")
    parser.add_argument("--assembly-dir", default=None, help="Directory for assembled nuScenes roots.")
    parser.add_argument(
        "--allow-external-assembly-dir",
        action="store_true",
        help="Allow --assembly-dir outside --out-dir.",
    )
    parser.add_argument("--depth-min", type=float, default=1.0)
    parser.add_argument("--depth-max", type=float, default=60.0)
    parser.add_argument("--point-radius", type=int, default=4)
    parser.add_argument("--point-alpha", type=float, default=0.75)
    parser.add_argument("--max-image-points", type=int, default=25000)
    parser.add_argument("--max-bev-points", type=int, default=80000)
    parser.add_argument("--bev-point-radius", type=int, default=1)
    parser.add_argument("--draw-labels", action="store_true", help="Draw source/score labels.")
    parser.add_argument("--no-bev-boxes", action="store_true", help="Do not draw prediction boxes in BEV.")
    parser.add_argument("--no-image-boxes", action="store_true", help="Do not draw prediction boxes on image.")
    parser.add_argument(
        "--source-color-overrides",
        default="",
        help="Comma/space list: fused_retained=#RRGGBB,img_supplement=#RRGGBB,unknown=#RRGGBB.",
    )
    parser.add_argument(
        "--bev-center-crop-to-image-height",
        action="store_true",
        help="Center-crop BEV height to match the camera image height.",
    )
    parser.add_argument(
        "--no-reliability-panel",
        action="store_true",
        help="Do not write the separate reliability/info panel image.",
    )
    parser.add_argument("--panel-width", type=int, default=720, help="Reliability panel width in pixels.")
    parser.add_argument("--allow-unaligned", action="store_true", help="Render even if aligned final source fields are absent.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve inputs and conditions without inference.")
    return parser.parse_args()


def parse_source_colors(text):
    colors = dict(SOURCE_COLORS_RGB)
    for item in rv.split_items(text):
        if "=" not in item:
            raise ValueError(f"--source-color-overrides expects source=#RRGGBB, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in colors:
            raise ValueError(f"Unknown source color key: {key}")
        colors[key] = rv.parse_rgb_color(value)
    return colors


def rgb_map_to_bgr(colors):
    return {key: rv.rgb_to_bgr(value) for key, value in colors.items()}


def extract_pred_with_edl(result):
    item = result[0] if isinstance(result, list) else result
    pts_bbox = item.get("pts_bbox", item)
    return (
        pts_bbox["boxes_3d"],
        pts_bbox["scores_3d"],
        pts_bbox["labels_3d"],
        pts_bbox.get("edl_info", {}) or {},
    )


def filter_predictions_with_edl(boxes, scores, labels, edl_info, score_thr, max_boxes):
    keep = scores > score_thr
    if int(keep.sum().item()) == 0:
        idx = rv.torch.nonzero(keep, as_tuple=False).flatten()
    else:
        idx = rv.torch.nonzero(keep, as_tuple=False).flatten()
        kept_scores = scores[idx]
        order = rv.torch.argsort(kept_scores, descending=True)
        if max_boxes > 0:
            order = order[:max_boxes]
        idx = idx[order]
    original_count = int(scores.shape[0])
    filtered = {}
    for key, value in edl_info.items():
        if isinstance(value, rv.torch.Tensor) and value.shape[:1] == (original_count,):
            filtered[key] = value[idx.to(value.device)]
        else:
            filtered[key] = value
    return boxes[idx], scores[idx], labels[idx], filtered, idx


def subset_predictions_with_edl(boxes, scores, labels, edl_info, keep_indices):
    original_count = int(scores.shape[0])
    idx = rv.torch.as_tensor(keep_indices, dtype=rv.torch.long, device=scores.device)
    filtered = {}
    for key, value in edl_info.items():
        if isinstance(value, rv.torch.Tensor) and value.shape[:1] == (original_count,):
            filtered[key] = value[idx.to(value.device)]
        else:
            filtered[key] = value
    return boxes[idx], scores[idx], labels[idx], filtered


def tensor_flat(edl_info, key, count, default=0.0):
    np_mod = rv.np
    if np_mod is None:
        import numpy as np_mod
    if count <= 0:
        return np_mod.zeros((0,), dtype=np_mod.float32)
    value = edl_info.get(key)
    if not isinstance(value, rv.torch.Tensor) or value.shape[:1] != (count,):
        return np_mod.full((count,), default, dtype=np_mod.float32)
    flat = value.detach().cpu().float()
    if flat.numel() == 0:
        return np_mod.full((count,), default, dtype=np_mod.float32)
    return flat.reshape(count, -1)[:, 0].numpy()


def tensor_int_flat(edl_info, key, count, default=-1):
    return tensor_flat(edl_info, key, count, float(default)).astype(rv.np.int64)


def tensor_scalar(edl_info, key, default=None):
    value = edl_info.get(key)
    if isinstance(value, rv.torch.Tensor):
        if value.numel() == 0:
            return default
        return float(value.detach().cpu().float().reshape(-1)[0].item())
    if isinstance(value, (int, float)):
        return float(value)
    return default


def source_names_from_edl(edl_info, count, allow_unaligned=False):
    aligned = tensor_flat(edl_info, "final_source_alignment_valid", count, 0.0)
    is_supp = tensor_flat(edl_info, "final_source_is_supplement", count, 0.0)
    branch = tensor_flat(edl_info, "final_source_branch", count, 0.0)
    aligned_min = float(aligned.min()) if aligned.size else 0.0
    if count and aligned_min <= 0.5 and not allow_unaligned:
        raise RuntimeError(
            "Aligned final source fields are missing. Rebuild with the current "
            "decode-index alignment patch, or pass --allow-unaligned for diagnostics."
        )
    names = []
    for ok, supp, branch_id in zip(aligned, is_supp, branch):
        if ok <= 0.5 and allow_unaligned:
            names.append("unknown")
        elif supp > 0.5 and int(round(float(branch_id))) == 2:
            names.append("img_supplement")
        elif supp > 0.5:
            names.append("unknown")
        else:
            names.append("fused_retained")
    return names


def summarize_sources(names):
    summary = {key: 0 for key in SOURCE_COLORS_RGB}
    for name in names:
        summary[name if name in summary else "unknown"] += 1
    return summary


def parse_source_priority(text):
    items = rv.split_items(text)
    if not items:
        items = ["fused_retained", "img_supplement", "unknown"]
    return {name: len(items) - rank for rank, name in enumerate(items)}


def duplicate_filter_rank(index, scores_np, source_names, priority, policy):
    source_rank = priority.get(source_names[index], 0)
    score = float(scores_np[index])
    if policy == "source_then_score":
        return source_rank, score
    return score, source_rank


def duplicate_filter_rank_note(policy):
    if policy == "source_then_score":
        return "ranking uses source priority, then final detection score"
    return "ranking uses final detection score, then source priority"


def visual_duplicate_filter_indices(args, boxes3d, scores, labels, source_names):
    count = len(source_names)
    base_stats = {
        "enabled": bool(args.duplicate_filter_thr and args.duplicate_filter_thr > 0.0),
        "threshold": float(args.duplicate_filter_thr),
        "policy": args.duplicate_filter_policy,
        "class_agnostic": bool(args.duplicate_filter_class_agnostic),
        "source_pair_scope": "all_sources" if args.duplicate_filter_all_sources else "fused_img_cross_source",
        "source_priority": rv.split_items(args.duplicate_filter_source_priority),
        "input_boxes": int(count),
        "kept_boxes": int(count),
        "removed_boxes": 0,
        "removed_by_source": {},
        "kept_by_source": summarize_sources(source_names),
        "removed_records": [],
        "rank_note": duplicate_filter_rank_note(args.duplicate_filter_policy),
        "note": "disabled",
    }
    if not base_stats["enabled"] or count <= 1:
        return list(range(count)), base_stats

    try:
        boxes_np = boxes3d.to("cpu").tensor.numpy() if hasattr(boxes3d, "to") else rv.np.asarray(boxes3d)
    except Exception as exc:
        base_stats["note"] = f"missing_box_tensor:{exc}"
        return list(range(count)), base_stats
    if boxes_np.ndim != 2 or boxes_np.shape[0] < count or boxes_np.shape[1] < 2:
        base_stats["note"] = "missing_box_centers"
        return list(range(count)), base_stats

    scores_np = scores.detach().cpu().float().numpy() if isinstance(scores, rv.torch.Tensor) else rv.np.asarray(scores)
    labels_np = labels.detach().cpu().long().numpy() if isinstance(labels, rv.torch.Tensor) else rv.np.asarray(labels)
    priority = parse_source_priority(args.duplicate_filter_source_priority)
    order = sorted(
        range(count),
        key=lambda i: duplicate_filter_rank(i, scores_np, source_names, priority, args.duplicate_filter_policy),
        reverse=True,
    )
    kept = []
    removed = []
    removed_by_source = {}
    threshold = float(args.duplicate_filter_thr)
    fused_img_pair = {"fused_retained", "img_supplement"}
    for idx in order:
        duplicate_of = None
        duplicate_dist = None
        for kept_idx in kept:
            if not args.duplicate_filter_all_sources:
                if {source_names[idx], source_names[kept_idx]} != fused_img_pair:
                    continue
            same_class = bool(labels_np[idx] == labels_np[kept_idx])
            if not args.duplicate_filter_class_agnostic and not same_class:
                continue
            dist = float(rv.np.hypot(
                float(boxes_np[idx, 0]) - float(boxes_np[kept_idx, 0]),
                float(boxes_np[idx, 1]) - float(boxes_np[kept_idx, 1]),
            ))
            if dist < threshold:
                duplicate_of = kept_idx
                duplicate_dist = dist
                break
        if duplicate_of is None:
            kept.append(idx)
            continue
        removed.append(idx)
        removed_by_source[source_names[idx]] = removed_by_source.get(source_names[idx], 0) + 1
        removed_record = {
            "removed_box_index": int(idx),
            "removed_source": source_names[idx],
            "removed_score": float(scores_np[idx]),
            "removed_class_id": int(labels_np[idx]),
            "kept_box_index": int(duplicate_of),
            "kept_source": source_names[duplicate_of],
            "kept_score": float(scores_np[duplicate_of]),
            "kept_class_id": int(labels_np[duplicate_of]),
            "distance": duplicate_dist,
        }
        base_stats["removed_records"].append(removed_record)

    keep_indices = sorted(kept)
    filtered_sources = [source_names[i] for i in keep_indices]
    base_stats.update({
        "kept_boxes": int(len(keep_indices)),
        "removed_boxes": int(count - len(keep_indices)),
        "removed_by_source": removed_by_source,
        "kept_by_source": summarize_sources(filtered_sources),
        "note": "posthoc_visualization_filter",
    })
    return keep_indices, base_stats


def box_geometry_records(boxes3d, count):
    empty = [
        {"bbox_center": None, "bbox_dims": None, "bbox_yaw": None}
        for _ in range(count)
    ]
    if boxes3d is None or count <= 0:
        return empty
    try:
        boxes_np = boxes3d.to("cpu").tensor.numpy() if hasattr(boxes3d, "to") else rv.np.asarray(boxes3d)
    except Exception:
        return empty
    if boxes_np.ndim != 2 or boxes_np.shape[0] < count or boxes_np.shape[1] < 3:
        return empty
    records = []
    for i in range(count):
        box = boxes_np[i]
        center = [float(v) for v in box[:3]]
        dims = [float(v) for v in box[3:6]] if box.shape[0] >= 6 else None
        yaw = float(box[6]) if box.shape[0] >= 7 else None
        records.append({
            "bbox_center": center,
            "bbox_dims": dims,
            "bbox_yaw": yaw,
        })
    return records


def reliability_records(edl_info, scores, labels, source_names, class_names, boxes3d=None):
    count = len(source_names)
    scores_np = scores.detach().cpu().float().numpy() if isinstance(scores, rv.torch.Tensor) else rv.np.asarray(scores)
    labels_np = labels.detach().cpu().long().numpy() if isinstance(labels, rv.torch.Tensor) else rv.np.asarray(labels)
    query_idx = tensor_int_flat(edl_info, "final_decode_query_idx", count, -1)
    task_idx = tensor_int_flat(edl_info, "final_decode_task_idx", count, -1)
    candidate_idx = tensor_int_flat(edl_info, "final_supp_candidate_query_idx", count, -1)
    img_score = tensor_flat(edl_info, "final_supp_img_score", count, float("nan"))
    fused_score = tensor_flat(edl_info, "final_supp_fused_score", count, float("nan"))
    util_score = tensor_flat(edl_info, "final_supp_utility_score", count, float("nan"))
    margin_score = tensor_flat(edl_info, "final_supp_marginal_utility_score", count, float("nan"))
    geometry = box_geometry_records(boxes3d, count)
    records = []
    for i, source in enumerate(source_names):
        cls_id = int(labels_np[i]) if i < len(labels_np) else -1
        cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        record = {
            "box_index": i,
            "source": source,
            "class_id": cls_id,
            "class_name": cls_name,
            "score": float(scores_np[i]) if i < len(scores_np) else None,
            "decode_task_idx": int(task_idx[i]),
            "decode_query_idx": int(query_idx[i]),
            "supp_candidate_query_idx": int(candidate_idx[i]),
            "supp_img_score": None if rv.np.isnan(img_score[i]) else float(img_score[i]),
            "supp_fused_score": None if rv.np.isnan(fused_score[i]) else float(fused_score[i]),
            "supp_utility_score": None if rv.np.isnan(util_score[i]) else float(util_score[i]),
            "supp_marginal_utility_score": None if rv.np.isnan(margin_score[i]) else float(margin_score[i]),
        }
        record.update(geometry[i])
        records.append(record)
    return records


def color_for_source(name, colors_bgr):
    return colors_bgr.get(name, colors_bgr["unknown"])


def draw_source_boxes_on_image(image, boxes3d, scores, source_names, lidar2img, colors_bgr, draw_labels=False):
    if boxes3d is None or len(boxes3d) == 0:
        return image
    boxes_cpu = boxes3d.to("cpu") if hasattr(boxes3d, "to") else boxes3d
    corners = boxes_cpu.corners.numpy()
    num = corners.shape[0]
    hom = rv.np.concatenate([corners.reshape(-1, 3), rv.np.ones((num * 8, 1), dtype=rv.np.float64)], axis=1)
    proj = hom @ rv.np.asarray(lidar2img, dtype=rv.np.float64).T
    depth = proj[:, 2].reshape(num, 8)
    xy = (proj[:, :2] / rv.np.maximum(proj[:, 2:3], 1e-6)).reshape(num, 8, 2)
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    out = image.copy()
    scores_np = scores.detach().cpu().numpy() if isinstance(scores, rv.torch.Tensor) else rv.np.asarray(scores)
    for i in range(num):
        if not rv.np.any(depth[i] > 1e-3):
            continue
        source = source_names[i] if i < len(source_names) else "unknown"
        color = color_for_source(source, colors_bgr)
        for a, b in edges:
            if depth[i, a] <= 1e-3 or depth[i, b] <= 1e-3:
                continue
            rv.clip_and_draw_line(out, xy[i, a], xy[i, b], color, 2)
        if draw_labels and i < len(scores_np):
            visible = xy[i][depth[i] > 1e-3]
            if len(visible):
                x, y = visible[:, 0].min(), visible[:, 1].min()
                text = f"{source.replace('_', '-')} {float(scores_np[i]):.2f}"
                rv.cv2.putText(
                    out, text, (int(max(0, x)), int(max(14, y))),
                    rv.cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, rv.cv2.LINE_AA)
    return out


def draw_source_bev_boxes(image, boxes3d, source_names, colors_bgr, pc_range):
    if boxes3d is None or len(boxes3d) == 0:
        return image
    out = image.copy()
    size = out.shape[1]
    boxes_np = boxes3d.to("cpu").tensor.numpy() if hasattr(boxes3d, "to") else rv.np.asarray(boxes3d)
    for i, box in enumerate(boxes_np):
        corners = rv.box_xy_corners(box)
        uv = rv.xy_to_bev_pixels(corners, pc_range, size).astype(rv.np.int32)
        source = source_names[i] if i < len(source_names) else "unknown"
        rv.cv2.polylines(out, [uv], isClosed=True, color=color_for_source(source, colors_bgr), thickness=2, lineType=rv.cv2.LINE_AA)
    return out


def pad_to_height(image, height):
    if image.shape[0] == height:
        return image
    canvas = rv.np.zeros((height, image.shape[1], 3), dtype=rv.np.uint8)
    y0 = (height - image.shape[0]) // 2
    canvas[y0:y0 + image.shape[0], :, :] = image
    return canvas


def format_float(value, digits=3):
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def draw_reliability_panel(
        min_height, width, condition, token, summary, edl_info, records, colors_bgr,
        duplicate_filter_stats=None):
    img_records = [r for r in records if r["source"] == "img_supplement"]
    metric_lines = 8
    source_lines = 3
    duplicate_lines = 0
    if duplicate_filter_stats and duplicate_filter_stats.get("enabled"):
        duplicate_lines = 5 + min(len(duplicate_filter_stats.get("removed_records") or []), 5)
    supplement_lines = max(1, min(len(img_records), 8) * 2)
    total_lines = 8 + source_lines + duplicate_lines + metric_lines + supplement_lines
    line_step = 34
    panel_height = max(int(min_height), 72 + total_lines * line_step)
    panel = rv.np.zeros((panel_height, width, 3), dtype=rv.np.uint8)
    y = 44

    def put(text, color=(230, 230, 230), scale=0.62, thickness=1, step=None):
        nonlocal y
        rv.cv2.putText(panel, str(text), (18, y), rv.cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, rv.cv2.LINE_AA)
        y += int(step if step is not None else line_step)

    put("Finite IMG Supplement", (255, 255, 255), 0.78, 2, step=40)
    put(f"condition: {condition}")
    put(f"token: {str(token)[:12]}")
    y += 14
    put("Final box sources", (255, 255, 255), 0.68, 2, step=38)
    for source in ("fused_retained", "img_supplement", "unknown"):
        color = colors_bgr[source]
        rv.cv2.rectangle(panel, (18, y - 18), (42, y + 6), color, thickness=-1)
        put(f"{source}: {summary.get(source, 0)}")
    if duplicate_filter_stats and duplicate_filter_stats.get("enabled"):
        y += 14
        put("Visual duplicate removal", (255, 255, 255), 0.68, 2, step=38)
        put(
            f"thr={format_float(duplicate_filter_stats.get('threshold'), 2)} "
            f"policy={duplicate_filter_stats.get('policy')}",
            (220, 220, 220), 0.55, step=30)
        put(
            f"removed={duplicate_filter_stats.get('removed_boxes', 0)} "
            f"kept={duplicate_filter_stats.get('kept_boxes', 0)}",
            (220, 220, 220), 0.55, step=30)
        removed_by_source = duplicate_filter_stats.get("removed_by_source") or {}
        put(f"removed_by_source={removed_by_source}", (190, 190, 190), 0.52, step=34)
        for rec in (duplicate_filter_stats.get("removed_records") or [])[:5]:
            put(
                f"drop #{rec['removed_box_index']} {rec['removed_source']} "
                f"near #{rec['kept_box_index']} {rec['kept_source']} "
                f"d={format_float(rec.get('distance'), 2)}",
                (190, 190, 190), 0.50, step=30)
    y += 14
    put("Scene telemetry", (255, 255, 255), 0.68, 2, step=38)
    for key, label in (
            ("supp_scene_gate_ratio", "gate"),
            ("supp_effective_max_add_queries", "budget"),
            ("supp_candidate_ratio", "candidate"),
            ("supp_utility_candidate_ratio", "utility"),
            ("supp_added_ratio", "predecode add"),
            ("supp_replace_slot_ratio", "replace slots"),
            ("supp_status_code", "status"),
            ("supp_noadd_reason_code", "noadd")):
        put(f"{label}: {format_float(tensor_scalar(edl_info, key))}")
    y += 14
    put("IMG supplement boxes", (255, 255, 255), 0.68, 2, step=38)
    if not img_records:
        put("none")
    for record in img_records[:8]:
        text = (
            f"#{record['box_index']} {record['class_name']} "
            f"s={format_float(record['score'], 2)} "
            f"img={format_float(record['supp_img_score'], 2)} "
            f"fused={format_float(record['supp_fused_score'], 2)}"
        )
        put(text, (220, 220, 220), 0.55, step=30)
        text = (
            f"  util={format_float(record['supp_utility_score'], 2)} "
            f"margin={format_float(record['supp_marginal_utility_score'], 2)} "
            f"q={record['decode_query_idx']} cand={record['supp_candidate_query_idx']}"
        )
        put(text, (190, 190, 190), 0.52, step=34)
    return panel


def compose_with_panel(camera_image, bev_image, panel=None):
    height = max(camera_image.shape[0], bev_image.shape[0], panel.shape[0] if panel is not None else 0)
    parts = [pad_to_height(camera_image, height), pad_to_height(bev_image, height)]
    if panel is not None:
        parts.append(pad_to_height(panel, height))
    return rv.np.concatenate(parts, axis=1)


def scene_gate_open(edl_info):
    gate = tensor_scalar(edl_info, "supp_scene_gate_ratio", 0.0)
    return gate is not None and gate > 0.0


def passes_selection(mode, source_summary, edl_info):
    if mode == "first":
        return True
    if mode == "added":
        return source_summary.get("img_supplement", 0) > 0
    if mode == "gate":
        return scene_gate_open(edl_info) or source_summary.get("img_supplement", 0) > 0
    raise ValueError(f"Unsupported selection mode: {mode}")


def selection_passes(selection):
    if selection == "added_then_gate":
        return ("added", "gate", "first")
    return (selection,)


def render_one(args, data, dataset, index, model, class_names, source_colors_bgr, pc_range, clean_root, assembled, cond_name, cond_out):
    info = dataset.data_infos[index]
    with rv.torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    boxes, scores, labels, edl_info = extract_pred_with_edl(result)
    boxes, scores, labels, edl_info, _ = filter_predictions_with_edl(
        boxes, scores, labels, edl_info, args.score_thr, args.max_boxes)
    source_names = source_names_from_edl(edl_info, len(scores), args.allow_unaligned)
    source_names_before_duplicate_filter = list(source_names)
    source_summary_before_duplicate_filter = summarize_sources(source_names_before_duplicate_filter)
    records_before_duplicate_filter = reliability_records(
        edl_info, scores, labels, source_names_before_duplicate_filter, class_names, boxes)
    keep_indices, duplicate_filter_stats = visual_duplicate_filter_indices(
        args, boxes, scores, labels, source_names_before_duplicate_filter)
    if len(keep_indices) != len(source_names_before_duplicate_filter):
        boxes, scores, labels, edl_info = subset_predictions_with_edl(
            boxes, scores, labels, edl_info, keep_indices)
        source_names = [source_names_before_duplicate_filter[i] for i in keep_indices]
    else:
        source_names = source_names_before_duplicate_filter
    source_summary = summarize_sources(source_names)

    cam_info = info["cams"][args.camera]
    image_path = rv.resolve_image_path(cam_info["data_path"], assembled)
    raw_img = rv.cv2.imread(str(image_path), rv.cv2.IMREAD_COLOR)
    if raw_img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    lidar2img = rv.raw_lidar2img_from_cam_info(cam_info)
    points, points_meta = rv.select_visualization_points(args, info, data, assembled)
    projected = rv.project_lidar_points_to_image(
        points, lidar2img, raw_img.shape, args.depth_min, args.depth_max, args.max_image_points)
    cam_vis = rv.draw_projected_points(
        raw_img, projected, args.depth_min, args.depth_max, args.point_radius, args.point_alpha)
    if not args.no_image_boxes:
        cam_vis = draw_source_boxes_on_image(
            cam_vis, boxes, scores, source_names, lidar2img, source_colors_bgr, args.draw_labels)

    bev_size = raw_img.shape[1]
    bev_vis = rv.draw_bev(
        points, None, labels, class_names, {"default": (255, 234, 0)}, pc_range, bev_size,
        args.max_bev_points, point_radius=args.bev_point_radius, draw_boxes=False)
    if not args.no_bev_boxes:
        bev_vis = draw_source_bev_boxes(bev_vis, boxes, source_names, source_colors_bgr, pc_range)
    bev_pre_crop_shape = list(bev_vis.shape)
    bev_crop_y0 = 0
    bev_crop_y1 = bev_vis.shape[0]
    bev_crop_applied = False
    if args.bev_center_crop_to_image_height:
        bev_vis, bev_crop_y0, bev_crop_y1, bev_crop_applied = rv.center_crop_height(bev_vis, raw_img.shape[0])

    token = info.get("token") or info.get("sample_idx") or rv.first_meta(data).get("sample_idx", str(index))
    records = reliability_records(edl_info, scores, labels, source_names, class_names, boxes)
    info_panel = None
    if not args.no_reliability_panel:
        info_panel = draw_reliability_panel(
            max(cam_vis.shape[0], bev_vis.shape[0]), args.panel_width, cond_name, token,
            source_summary, edl_info, records, source_colors_bgr, duplicate_filter_stats)
    composite = compose_with_panel(cam_vis, bev_vis, None)
    return {
        "composite": composite,
        "meta": {
            "sample_index": index,
            "sample_token": token,
            "condition": cond_name,
            "camera": args.camera,
            "image_path": str(image_path),
            "score_threshold": args.score_thr,
            "max_boxes": args.max_boxes,
            "rendered_boxes_before_duplicate_filter": int(len(source_names_before_duplicate_filter)),
            "rendered_boxes": int(len(boxes)),
            "source_summary_before_duplicate_filter": source_summary_before_duplicate_filter,
            "source_summary": source_summary,
            "source_records_before_duplicate_filter": records_before_duplicate_filter,
            "source_records": records,
            "visual_duplicate_filter": duplicate_filter_stats,
            "source_alignment_valid": bool(
                len(scores) == 0 or
                float(tensor_flat(edl_info, "final_source_alignment_valid", len(scores), 0.0).min()) > 0.5),
            "projected_points": int(len(projected)),
            "visualized_points_input": int(points.shape[0]),
            "image_shape": list(raw_img.shape),
            "bev_pre_crop_shape": bev_pre_crop_shape,
            "bev_shape": list(bev_vis.shape),
            "composite_shape": list(composite.shape),
            "info_panel_shape": list(info_panel.shape) if info_panel is not None else None,
            "info_panel_separate_image": info_panel is not None,
            "bev_center_crop_to_image_height": bool(args.bev_center_crop_to_image_height),
            "bev_crop_applied": bool(bev_crop_applied),
            "bev_crop_y0": int(bev_crop_y0),
            "bev_crop_y1": int(bev_crop_y1),
            "scene_supp_scene_gate_ratio": tensor_scalar(edl_info, "supp_scene_gate_ratio"),
            "scene_supp_effective_max_add_queries": tensor_scalar(edl_info, "supp_effective_max_add_queries"),
            "scene_supp_candidate_ratio": tensor_scalar(edl_info, "supp_candidate_ratio"),
            "scene_supp_utility_candidate_ratio": tensor_scalar(edl_info, "supp_utility_candidate_ratio"),
            "scene_supp_added_ratio": tensor_scalar(edl_info, "supp_added_ratio"),
            "scene_supp_replace_slot_ratio": tensor_scalar(edl_info, "supp_replace_slot_ratio"),
            "scene_supp_status_code": tensor_scalar(edl_info, "supp_status_code"),
            "scene_supp_noadd_reason_code": tensor_scalar(edl_info, "supp_noadd_reason_code"),
        },
        "info_panel": info_panel,
        "source_summary": source_summary,
        "selection_source_summary": source_summary_before_duplicate_filter,
        "edl_info": edl_info,
        "points_meta": points_meta,
    }


def should_consider_index(index, info, args, explicit_indices, explicit_tokens):
    token = info.get("token") or info.get("sample_idx")
    if explicit_indices and index not in explicit_indices:
        return False
    if explicit_tokens and token not in explicit_tokens:
        return False
    if not explicit_indices and not explicit_tokens and index < args.start_index:
        return False
    return True


def render_condition(args, base_cfg, model, class_names, source_colors_bgr, clean_root, source_root, condition, output_root, assembly_root):
    split, corruption, severity = condition
    cond_name = rv.condition_name(split, corruption, severity)
    assembled = rv.assemble_nuscenes_root(clean_root, source_root, assembly_root / rv.safe_label(cond_name), assembly_root)
    cfg = rv.build_condition_cfg(base_cfg, assembled, args.workers)
    dataset, data_loader = rv.build_condition_dataset(cfg, args.workers)
    pc_range = list(getattr(cfg, "point_cloud_range", [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]))
    cond_out = output_root / cond_name
    image_out = cond_out / "images"
    info_image_out = cond_out / "info_images"
    json_out = cond_out / "json"
    image_out.mkdir(parents=True, exist_ok=True)
    info_image_out.mkdir(parents=True, exist_ok=True)
    json_out.mkdir(parents=True, exist_ok=True)

    explicit_indices = {int(x) for x in rv.split_items(args.sample_indices)}
    explicit_tokens = set(rv.split_items(args.sample_tokens))
    rendered_indices = set()
    rows = []
    scanned_total = 0

    for pass_mode in selection_passes(args.selection):
        if len(rows) >= args.samples:
            break
        scanned_this_pass = 0
        for index, data in enumerate(data_loader):
            if len(rows) >= args.samples:
                break
            if index in rendered_indices:
                continue
            info = dataset.data_infos[index]
            if not should_consider_index(index, info, args, explicit_indices, explicit_tokens):
                continue
            scanned_this_pass += 1
            scanned_total += 1
            if args.scan_limit > 0 and scanned_this_pass > args.scan_limit:
                break
            try:
                rendered = render_one(
                    args, data, dataset, index, model, class_names, source_colors_bgr,
                    pc_range, clean_root, assembled, cond_name, cond_out)
            except RuntimeError as exc:
                if "Aligned final source fields are missing" in str(exc):
                    raise
                raise
            if not passes_selection(pass_mode, rendered["selection_source_summary"], rendered["edl_info"]):
                continue
            token = rendered["meta"]["sample_token"]
            stem = f"{len(rows):03d}_{index:06d}_{str(token)[:8]}_{cond_name}_{args.camera}_supp"
            out_png = image_out / f"{stem}.png"
            out_info_png = info_image_out / f"{stem}_info.png"
            out_json = json_out / f"{stem}.json"
            rv.cv2.imwrite(str(out_png), rendered["composite"])
            if rendered.get("info_panel") is not None:
                rv.cv2.imwrite(str(out_info_png), rendered["info_panel"])
            meta = dict(rendered["meta"])
            meta.update(rendered["points_meta"])
            meta.update({
                "split": split,
                "corruption": corruption,
                "severity": severity,
                "selection_pass": pass_mode,
                "output_png": str(out_png),
                "output_info_png": str(out_info_png) if rendered.get("info_panel") is not None else "",
            })
            out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            rows.append(meta)
            rendered_indices.add(index)
            if rendered.get("info_panel") is not None:
                print(f"[runtime-supp-vis] {cond_name}: wrote {out_png} and {out_info_png}")
            else:
                print(f"[runtime-supp-vis] {cond_name}: wrote {out_png}")

    if len(rows) < args.samples:
        print(
            f"[runtime-supp-vis] {cond_name}: rendered {len(rows)}/{args.samples} "
            f"samples after scanning {scanned_total}.")
    return rows


def build_conditions_with_roots(args, clean_root):
    conditions = []
    for split, corr, sev in rv.build_conditions(args):
        if split == "clean":
            source = clean_root
        else:
            source = Path(args.multicorrupt_root).resolve() / corr / str(sev)
            if not source.is_dir():
                raise FileNotFoundError(f"Missing MultiCorrupt condition root: {source}")
        conditions.append((split, corr, sev, source))
    return conditions


def main():
    args = parse_args()
    config, checkpoint, label = rv.resolve_config_checkpoint(args)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir or f"work_dirs/runtime_supplement_visualizations/{rv.safe_label(label)}_{timestamp}").resolve()
    assembly_root = Path(args.assembly_dir or out_root / "_assembled_nuscenes").resolve()
    if not args.allow_external_assembly_dir and not rv.is_relative_to(assembly_root, out_root):
        raise ValueError("--assembly-dir must be inside --out-dir unless --allow-external-assembly-dir is set.")
    out_root.mkdir(parents=True, exist_ok=True)
    assembly_root.mkdir(parents=True, exist_ok=True)

    clean_root = rv.resolve_user_path(args.clean_data_root).resolve()
    if not clean_root.is_dir():
        raise FileNotFoundError(f"Missing clean data root: {clean_root}")
    conditions = build_conditions_with_roots(args, clean_root)
    source_colors_rgb = parse_source_colors(args.source_color_overrides)
    json_root = out_root / "json"
    json_root.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "config": config,
        "checkpoint": checkpoint,
        "model_label": label,
        "out_dir": str(out_root),
        "clean_data_root": str(clean_root),
        "multicorrupt_root": str(Path(args.multicorrupt_root).resolve()),
        "conditions": [
            {"split": s, "corruption": c, "severity": v, "source_root": str(src)}
            for s, c, v, src in conditions
        ],
        "samples_per_condition": args.samples,
        "selection": args.selection,
        "score_thr": args.score_thr,
        "max_boxes": args.max_boxes,
        "camera": args.camera,
        "layout_version": "runtime_supplement_v1",
        "output_layout": {
            "run_json_dir": "json",
            "condition_image_dir": "<condition>/images",
            "condition_info_image_dir": "<condition>/info_images",
            "condition_json_dir": "<condition>/json",
        },
        "point_radius": args.point_radius,
        "point_alpha": args.point_alpha,
        "bev_point_radius": args.bev_point_radius,
        "visual_duplicate_filter": {
            "threshold": args.duplicate_filter_thr,
            "policy": args.duplicate_filter_policy,
            "source_priority": rv.split_items(args.duplicate_filter_source_priority),
            "class_agnostic": bool(args.duplicate_filter_class_agnostic),
            "source_pair_scope": "all_sources" if args.duplicate_filter_all_sources else "fused_img_cross_source",
            "rank_note": duplicate_filter_rank_note(args.duplicate_filter_policy),
            "note": "visualization-only; model outputs and metrics are unchanged",
        },
        "source_semantics": {
            "fused_retained": "decoded box whose aligned supplement flag is false",
            "img_supplement": "decoded box whose aligned supplement flag is true and final source branch is img",
        },
        "source_colors_rgb": {k: list(v) for k, v in source_colors_rgb.items()},
        "reliability_fields": [
            "final_supp_img_score",
            "final_supp_fused_score",
            "final_supp_utility_score",
            "final_supp_marginal_utility_score",
        ],
    }
    (json_root / "runtime_supplement_visualize_run.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8")
    print(json.dumps(run_meta, indent=2))
    if args.dry_run:
        print("[runtime-supp-vis] dry run complete; no inference executed.")
        return

    rv.ensure_runtime_imports()
    cfg = rv.load_config(config, args.cfg_options)
    rv.setup_multi_processes(cfg)
    if args.seed is not None:
        rv.set_random_seed(args.seed, deterministic=False)
    model = rv.build_runtime_model(cfg, checkpoint, args.gpu_id, args.fuse_conv_bn)
    class_names = list(getattr(model.module, "CLASSES", None) or getattr(cfg, "class_names", rv.DEFAULT_CLASSES))
    source_colors_bgr = rgb_map_to_bgr(source_colors_rgb)

    manifest_rows = []
    try:
        for split, corr, sev, source in conditions:
            rows = render_condition(
                args, cfg, model, class_names, source_colors_bgr, clean_root, source,
                (split, corr, sev), out_root, assembly_root)
            manifest_rows.extend(rows)
    finally:
        if not args.keep_assembled and not args.allow_external_assembly_dir and assembly_root.exists():
            shutil.rmtree(assembly_root)

    manifest_path = json_root / "supplement_render_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[runtime-supp-vis] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
