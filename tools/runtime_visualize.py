#!/usr/bin/env python3
"""Runtime nuScenes/MultiCorrupt front-camera and BEV visualizer.

This tool runs the model directly and renders thesis-style qualitative figures:
CAM_FRONT RGB with projected LiDAR points and predicted 3D boxes on the left,
and a black-background LiDAR BEV panel on the right.
"""

import argparse
import copy
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


DATA_SUBDIRS = ("samples", "sweeps", "v1.0-trainval", "maps", "lidarseg")
DEFAULT_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
THESIS_BOX_COLORS_RGB = {
    "car": (255, 234, 0),
    "pedestrian": (0, 229, 255),
    "truck": (255, 61, 0),
    "bus": (0, 230, 118),
    "motorcycle": (213, 0, 249),
    "bicycle": (41, 121, 255),
    "trailer": (255, 145, 0),
    "construction_vehicle": (174, 234, 0),
    "traffic_cone": (255, 23, 68),
    "barrier": (255, 255, 255),
}
NUSCENES_BOX_COLORS_RGB = {
    "car": (255, 158, 0),
    "pedestrian": (0, 0, 230),
    "truck": (255, 99, 71),
    "bus": (255, 69, 0),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "trailer": (255, 140, 0),
    "construction_vehicle": (233, 150, 70),
    "traffic_cone": (255, 255, 255),
    "barrier": (112, 128, 144),
}
MONO_BOX_COLORS_RGB = {"default": (255, 234, 0)}
BOX_COLOR_PALETTES_RGB = {
    "thesis": THESIS_BOX_COLORS_RGB,
    "nuscenes": NUSCENES_BOX_COLORS_RGB,
    "mono": MONO_BOX_COLORS_RGB,
}

cv2 = None
mmcv = None
np = None
torch = None
Config = None
MMDataParallel = None
fuse_conv_bn = None
load_checkpoint = None
wrap_fp16_model = None
set_random_seed = None
build_dataloader = None
build_dataset = None
build_model = None
setup_multi_processes = None
compat_cfg = None


class FallbackDictAction(argparse.Action):
    """Small fallback for --cfg-options when mmcv is not importable."""

    @staticmethod
    def _parse_value(value):
        lowered = value.lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        if lowered in ("none", "null"):
            return None
        try:
            return json.loads(value)
        except Exception:
            return value

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for item in values:
            if "=" not in item:
                parser.error(f"{option_string} expects key=value items, got: {item}")
            key, value = item.split("=", 1)
            options[key] = self._parse_value(value)
        setattr(namespace, self.dest, options)


try:
    from mmcv import DictAction as MMCVDictAction
except Exception:
    MMCVDictAction = FallbackDictAction


def ensure_runtime_imports():
    global cv2, mmcv, np, torch, Config, MMDataParallel
    global fuse_conv_bn, load_checkpoint, wrap_fp16_model, set_random_seed
    global build_dataloader, build_dataset, build_model, setup_multi_processes
    global compat_cfg

    if cv2 is not None:
        return

    import cv2 as _cv2
    import mmcv as _mmcv
    import numpy as _np
    import torch as _torch
    from mmcv import Config as _Config
    from mmcv.cnn import fuse_conv_bn as _fuse_conv_bn
    from mmcv.parallel import MMDataParallel as _MMDataParallel
    from mmcv.runner import (
        load_checkpoint as _load_checkpoint,
        wrap_fp16_model as _wrap_fp16_model,
    )
    from mmdet.apis import set_random_seed as _set_random_seed
    from mmdet3d.datasets import build_dataloader as _build_dataloader
    from mmdet3d.datasets import build_dataset as _build_dataset
    from mmdet3d.models import build_model as _build_model
    from mmdet3d.utils import setup_multi_processes as _setup_multi_processes

    try:
        from mmdet.utils import compat_cfg as _compat_cfg
    except ImportError:
        from mmdet3d.utils import compat_cfg as _compat_cfg

    try:
        import projects.mmdet3d_plugin  # noqa: F401
    except ImportError:
        pass

    cv2 = _cv2
    mmcv = _mmcv
    np = _np
    torch = _torch
    Config = _Config
    MMDataParallel = _MMDataParallel
    fuse_conv_bn = _fuse_conv_bn
    load_checkpoint = _load_checkpoint
    wrap_fp16_model = _wrap_fp16_model
    set_random_seed = _set_random_seed
    build_dataloader = _build_dataloader
    build_dataset = _build_dataset
    build_model = _build_model
    setup_multi_processes = _setup_multi_processes
    compat_cfg = _compat_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Runtime CAM_FRONT + BEV visualization for nuScenes/MultiCorrupt."
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
    parser.add_argument(
        "--split",
        default="corrupt",
        help="Comma/space list: clean,corrupt. Default: corrupt.",
    )
    parser.add_argument("--corruptions", default="beamsreducing", help="Comma/space corruption list.")
    parser.add_argument("--severities", default="3", help="Comma/space severity list.")
    parser.add_argument("--samples", type=int, default=8, help="Rendered samples per condition.")
    parser.add_argument("--start-index", type=int, default=0, help="Dataset index to start from.")
    parser.add_argument("--sample-indices", default="", help="Comma/space explicit dataset indices.")
    parser.add_argument("--sample-tokens", default="", help="Comma/space explicit nuScenes sample tokens.")
    parser.add_argument("--camera", default="CAM_FRONT", help="nuScenes camera key.")
    parser.add_argument(
        "--vis-points-source",
        choices=("raw", "keyframe", "pipeline"),
        default="raw",
        help=(
            "Point cloud used only for visualization. 'raw' reads the single "
            "keyframe LIDAR_TOP file, 'keyframe' filters the pipeline tensor to "
            "current-frame points, and 'pipeline' preserves the model input "
            "including multi-sweeps."
        ),
    )
    parser.add_argument(
        "--keyframe-time-eps",
        type=float,
        default=1e-4,
        help="Absolute time threshold for --vis-points-source keyframe.",
    )
    parser.add_argument("--score-thr", type=float, default=0.30, help="Prediction score threshold.")
    parser.add_argument("--max-boxes", type=int, default=30, help="Max boxes rendered per sample.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id.")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--fuse-conv-bn", action="store_true", help="Fuse conv/bn before inference.")
    parser.add_argument("--cfg-options", nargs="+", action=MMCVDictAction, help="Config overrides.")
    parser.add_argument("--keep-assembled", action="store_true", help="Keep temporary assembled roots.")
    parser.add_argument("--assembly-dir", default=None, help="Directory for assembled nuScenes roots.")
    parser.add_argument(
        "--allow-external-assembly-dir",
        action="store_true",
        help="Allow --assembly-dir outside --out-dir. Off by default for deletion safety.",
    )
    parser.add_argument("--depth-min", type=float, default=1.0, help="Min LiDAR projection depth/range.")
    parser.add_argument("--depth-max", type=float, default=60.0, help="Max depth/range for colormap.")
    parser.add_argument("--point-radius", type=int, default=4, help="Projected point radius in pixels.")
    parser.add_argument("--point-alpha", type=float, default=0.75, help="Projected point alpha.")
    parser.add_argument("--max-image-points", type=int, default=25000, help="Projected point cap.")
    parser.add_argument("--max-bev-points", type=int, default=80000, help="BEV point cap.")
    parser.add_argument("--bev-point-radius", type=int, default=1, help="BEV point radius.")
    parser.add_argument("--draw-labels", action="store_true", help="Draw class/score labels on image boxes.")
    parser.add_argument("--no-bev-boxes", action="store_true", help="Do not draw prediction boxes in BEV.")
    parser.add_argument("--no-image-boxes", action="store_true", help="Do not draw prediction boxes on image.")
    parser.add_argument(
        "--box-color-palette",
        choices=tuple(BOX_COLOR_PALETTES_RGB),
        default="nuscenes",
        help="Prediction box color palette. Default: nuscenes.",
    )
    parser.add_argument(
        "--box-color-overrides",
        default="",
        help=(
            "Comma/space list of class=#RRGGBB overrides, e.g. "
            "car=#FFEA00,pedestrian=#00E5FF,default=#FFEA00."
        ),
    )
    parser.add_argument("--show-front-marker", action="store_true", help="Draw Front text and arrow on BEV.")
    parser.add_argument(
        "--bev-center-crop-to-image-height",
        action="store_true",
        help="Center-crop BEV height to match the camera image height while keeping BEV width.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve inputs and conditions without inference.")
    return parser.parse_args()


def split_items(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw = []
        for item in value:
            raw.extend(split_items(item))
        return raw
    text = str(value).replace(",", " ")
    return [item.strip() for item in text.split() if item.strip()]


def normalize_class_name(name):
    return str(name).strip().lower().replace(" ", "_")


def parse_rgb_color(value):
    text = str(value).strip()
    named = {
        "white": "FFFFFF",
        "yellow": "FFEA00",
        "cyan": "00E5FF",
        "red": "FF1744",
        "green": "00E676",
        "blue": "2979FF",
        "magenta": "D500F9",
        "orange": "FF9100",
    }
    text = named.get(text.lower(), text)
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6:
        raise ValueError(f"Color must be #RRGGBB or RRGGBB, got: {value}")
    try:
        return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError as exc:
        raise ValueError(f"Invalid RGB color: {value}") from exc


def parse_box_color_overrides(text):
    overrides = {}
    for item in split_items(text):
        if "=" not in item:
            raise ValueError(f"--box-color-overrides expects class=#RRGGBB, got: {item}")
        key, value = item.split("=", 1)
        overrides[normalize_class_name(key)] = parse_rgb_color(value)
    return overrides


def build_box_color_map_rgb(palette_name, overrides_text, class_names):
    palette = BOX_COLOR_PALETTES_RGB[palette_name]
    colors = {normalize_class_name(k): tuple(v) for k, v in palette.items()}
    colors.update(parse_box_color_overrides(overrides_text))
    default = colors.get("default", THESIS_BOX_COLORS_RGB["car"])
    resolved = {normalize_class_name(cls): colors.get(normalize_class_name(cls), default) for cls in class_names}
    resolved["default"] = default
    return resolved


def rgb_to_bgr(rgb):
    r, g, b = rgb
    return int(b), int(g), int(r)


def color_for_label(label, class_names, box_colors_bgr):
    idx = int(label)
    cls = class_names[idx] if 0 <= idx < len(class_names) else "default"
    return box_colors_bgr.get(normalize_class_name(cls), box_colors_bgr.get("default", (0, 234, 255)))


def resolve_pack_candidate(pack_root, candidate):
    pack = resolve_user_path(pack_root)
    tsv = pack / "summaries" / "candidates.tsv"
    if not tsv.is_file():
        raise FileNotFoundError(f"Missing runtime pack candidates.tsv: {tsv}")
    with tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("name") == candidate:
                config = row.get("config")
                checkpoint = row.get("checkpoint")
                if not config or not checkpoint:
                    raise ValueError(f"Candidate row lacks config/checkpoint: {candidate}")
                return resolve_pack_path(config, pack), resolve_pack_path(checkpoint, pack)
    raise KeyError(f"Candidate not found in {tsv}: {candidate}")


def resolve_user_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return REPO_ROOT / path


def resolve_pack_path(path, pack_root):
    raw = Path(path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([pack_root / raw, REPO_ROOT / raw, Path.cwd() / raw])
        if len(raw.parts) >= 2 and raw.parts[0] == "configs":
            candidates.append(pack_root / raw)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0] if candidates else raw)


def resolve_config_checkpoint(args):
    config, checkpoint = args.config, args.checkpoint
    if args.pack or args.candidate:
        if not args.pack or not args.candidate:
            raise ValueError("--pack and --candidate must be provided together.")
        config, checkpoint = resolve_pack_candidate(args.pack, args.candidate)
    if not config or not checkpoint:
        raise ValueError("Provide CONFIG CHECKPOINT or --pack with --candidate.")
    config = str(Path(config))
    checkpoint = str(Path(checkpoint))
    if not Path(config).is_file():
        raise FileNotFoundError(f"Missing config: {config}")
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    label = args.model_label or args.candidate or Path(config).stem
    return config, checkpoint, label


def safe_label(text):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))


def safe_component(text, field):
    text = str(text)
    if text in ("", ".", "..") or "/" in text or "\\" in text:
        raise ValueError(f"Unsafe {field}: {text!r}")
    return text


def is_relative_to(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def is_lexically_relative_to(path, root):
    path = Path(os.path.abspath(os.fspath(path)))
    root = Path(os.path.abspath(os.fspath(root)))
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def assert_safe_child(path, root, label):
    path = Path(path).resolve()
    root = Path(root).resolve()
    if path == root or not is_relative_to(path, root):
        raise ValueError(f"Unsafe {label}: {path} is not a child of {root}")
    return path


def load_config(config_path, cfg_options=None):
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])
    if hasattr(cfg, "plugin") and cfg.plugin:
        import importlib

        if hasattr(cfg, "plugin_dir"):
            module_dir = cfg.plugin_dir
        else:
            module_dir = os.path.dirname(config_path)
        module_path = module_dir.replace("/", ".").strip(".")
        if module_path:
            importlib.import_module(module_path)
    cfg = compat_cfg(cfg)
    return cfg


def recursive_patch_paths(obj, clean_root, source_root, assembled_root):
    if isinstance(obj, dict):
        return {
            key: recursive_patch_paths(value, clean_root, source_root, assembled_root)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [recursive_patch_paths(value, clean_root, source_root, assembled_root) for value in obj]
    if isinstance(obj, tuple):
        return tuple(recursive_patch_paths(value, clean_root, source_root, assembled_root) for value in obj)
    if isinstance(obj, str):
        return patch_path_string(obj, clean_root, source_root, assembled_root)
    return obj


def patch_path_string(value, clean_root, source_root, assembled_root):
    subdir_markers = [f"{os.sep}{name}{os.sep}" for name in DATA_SUBDIRS]
    starts_with_subdir = any(value.startswith(f"{name}{os.sep}") for name in DATA_SUBDIRS)
    contains_marker = any(marker in value for marker in subdir_markers)
    if not (starts_with_subdir or contains_marker):
        return value

    raw_path = Path(value)
    if raw_path.is_absolute():
        for root in (clean_root, source_root):
            try:
                suffix = raw_path.resolve().relative_to(root.resolve())
                return str(assembled_root / suffix)
            except (ValueError, FileNotFoundError):
                pass
        parts = raw_path.parts
        for name in DATA_SUBDIRS:
            if name in parts:
                idx = parts.index(name)
                return str(assembled_root.joinpath(*parts[idx:]))
        return value

    parts = raw_path.parts
    for name in DATA_SUBDIRS:
        if parts and parts[0] == name:
            return str(assembled_root / raw_path)
        if name in parts:
            idx = parts.index(name)
            return str(assembled_root.joinpath(*parts[idx:]))
    return value


def patch_info_file(src_pkl, dst_pkl, clean_root, source_root, assembled_root):
    data = mmcv.load(str(src_pkl))
    patched = recursive_patch_paths(data, clean_root, source_root, assembled_root)
    mmcv.dump(patched, str(dst_pkl))


def symlink_force(src, dst):
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(Path(src).resolve(), target_is_directory=Path(src).is_dir())


def assemble_nuscenes_root(clean_root, source_root, assembled_root, assembly_base):
    clean_root = Path(clean_root).resolve()
    source_root = Path(source_root).resolve()
    assembled_root = assert_safe_child(assembled_root, assembly_base, "assembled condition root")
    if assembled_root.exists():
        shutil.rmtree(assembled_root)
    assembled_root.mkdir(parents=True, exist_ok=True)

    for subdir in DATA_SUBDIRS:
        source = source_root / subdir
        fallback = clean_root / subdir
        if source.exists() or source.is_symlink():
            symlink_force(source, assembled_root / subdir)
        elif fallback.exists() or fallback.is_symlink():
            symlink_force(fallback, assembled_root / subdir)

    for pkl in sorted(clean_root.glob("nuscenes_infos_*.pkl")):
        patch_info_file(pkl, assembled_root / pkl.name, clean_root, source_root, assembled_root)
    dbinfo = clean_root / "nuscenes_dbinfos_train.pkl"
    if dbinfo.exists():
        symlink_force(dbinfo, assembled_root / dbinfo.name)
    return assembled_root


def set_dataset_paths(dataset_cfg, data_root):
    data_root = str(Path(data_root).resolve())
    ann_file = str(Path(data_root) / "nuscenes_infos_val.pkl")

    def apply_recursive(cfg):
        if not isinstance(cfg, dict):
            return
        if "ann_file" in cfg or "pipeline" in cfg or cfg.get("type", "").endswith("NuScenesDataset"):
            cfg["data_root"] = data_root + os.sep
            cfg["ann_file"] = ann_file
            cfg["test_mode"] = True
        for key in ("dataset", "datasets"):
            child = cfg.get(key)
            if isinstance(child, dict):
                apply_recursive(child)
            elif isinstance(child, (list, tuple)):
                for item in child:
                    apply_recursive(item)

    if isinstance(dataset_cfg, (list, tuple)):
        for cfg in dataset_cfg:
            apply_recursive(cfg)
    else:
        apply_recursive(dataset_cfg)


def build_condition_cfg(base_cfg, data_root, workers):
    cfg = copy.deepcopy(base_cfg)
    set_dataset_paths(cfg.data.test, data_root)
    if "test_dataloader" not in cfg.data:
        cfg.data.test_dataloader = {}
    cfg.data.test_dataloader.samples_per_gpu = 1
    cfg.data.test_dataloader.workers_per_gpu = workers
    cfg.data.workers_per_gpu = workers
    return cfg


def build_runtime_model(cfg, checkpoint, gpu_id=0, fuse_conv_bn_flag=False):
    setup_multi_processes(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if cfg.get("optimizer_config", None) is not None and cfg.optimizer_config.get("type") == "CustomFp16OptimizerHook":
        wrap_fp16_model(model)
        for module_name, enabled in cfg.optimizer_config.get("custom_fp16", {}).items():
            model._modules[module_name].fp16_enabled = enabled
    ckpt = load_checkpoint(model, checkpoint, map_location="cpu")
    if fuse_conv_bn_flag:
        model = fuse_conv_bn(model)
    if "CLASSES" in ckpt.get("meta", {}):
        model.CLASSES = ckpt["meta"]["CLASSES"]
    model = MMDataParallel(model.cuda(gpu_id), device_ids=[gpu_id])
    model.eval()
    return model


def build_condition_dataset(cfg, workers):
    dataset = build_dataset(cfg.data.test)
    loader_cfg = dict(samples_per_gpu=1, workers_per_gpu=workers, dist=False, shuffle=False)
    loader_cfg.update(cfg.data.get("test_dataloader", {}))
    loader_cfg["samples_per_gpu"] = 1
    loader_cfg["workers_per_gpu"] = workers
    loader_cfg["dist"] = False
    loader_cfg["shuffle"] = False
    data_loader = build_dataloader(dataset, **loader_cfg)
    return dataset, data_loader


def flatten_leaves(obj):
    if hasattr(obj, "data"):
        obj = obj.data
    if isinstance(obj, (list, tuple)):
        out = []
        for item in obj:
            out.extend(flatten_leaves(item))
        return out
    return [obj]


def first_meta(data):
    metas = [item for item in flatten_leaves(data.get("img_metas")) if isinstance(item, dict)]
    if not metas:
        raise RuntimeError("Could not extract img_metas from dataloader batch.")
    return metas[0]


def first_points(data):
    leaves = flatten_leaves(data.get("points"))
    for item in leaves:
        if item is None:
            continue
        if hasattr(item, "tensor"):
            tensor = item.tensor
        else:
            tensor = item
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        if isinstance(tensor, np.ndarray):
            return tensor
    raise RuntimeError("Could not extract points from dataloader batch.")


def camera_index(info, camera):
    keys = list(info.get("cams", {}).keys())
    if camera not in info.get("cams", {}):
        raise KeyError(f"Camera {camera} not found. Available: {keys}")
    return keys.index(camera)


def raw_lidar2img_from_cam_info(cam_info):
    lidar2cam_r = np.linalg.inv(np.asarray(cam_info["sensor2lidar_rotation"], dtype=np.float64))
    lidar2cam_t = np.asarray(cam_info["sensor2lidar_translation"], dtype=np.float64) @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float64)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    intrinsic = np.asarray(cam_info["cam_intrinsic"], dtype=np.float64)
    viewpad = np.eye(4, dtype=np.float64)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    return viewpad @ lidar2cam_rt.T


def resolve_image_path(path, data_root):
    p = Path(path)
    if p.is_file():
        return p
    if not p.is_absolute():
        for root in (Path.cwd(), Path(data_root)):
            candidate = root / p
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Image path not found: {path}")


def resolve_data_file_path(path, data_root, label):
    data_root = Path(os.path.abspath(os.fspath(data_root)))
    p = Path(path)
    if p.is_file() and is_lexically_relative_to(p, data_root):
        return Path(os.path.abspath(os.fspath(p)))
    candidates = []
    parts = p.parts
    for name in DATA_SUBDIRS:
        if name in parts:
            suffix_parts = parts[parts.index(name):]
            if ".." not in suffix_parts:
                candidates.append(data_root.joinpath(*suffix_parts))
            break
    for candidate in candidates:
        candidate = Path(os.path.abspath(os.fspath(candidate)))
        if candidate.is_file() and is_lexically_relative_to(candidate, data_root):
            return candidate
    raise FileNotFoundError(f"{label} path not found: {path}")


def read_lidar_bin(path):
    points = np.fromfile(str(path), dtype=np.float32)
    if points.size % 5 != 0:
        raise ValueError(f"Expected nuScenes LIDAR .bin with 5 floats per point: {path}")
    return points.reshape(-1, 5)


def filter_keyframe_points(points, eps):
    if points.shape[1] < 5:
        return points, False
    mask = np.isfinite(points[:, 4]) & (np.abs(points[:, 4]) <= eps)
    return points[mask], True


def select_visualization_points(args, info, data, assembled):
    pipeline_points = first_points(data)
    if args.vis_points_source == "pipeline":
        return pipeline_points, {
            "visualized_points_source": "pipeline",
            "visualized_lidar_path": "",
            "pipeline_points_input": int(pipeline_points.shape[0]),
        }

    if args.vis_points_source == "keyframe":
        vis_points, time_dim_available = filter_keyframe_points(pipeline_points, args.keyframe_time_eps)
        if time_dim_available and len(vis_points) == 0:
            print("[runtime-vis][WARN] keyframe point filter returned 0 points.")
        return vis_points, {
            "visualized_points_source": "keyframe",
            "visualized_lidar_path": "",
            "pipeline_points_input": int(pipeline_points.shape[0]),
            "keyframe_time_eps": float(args.keyframe_time_eps),
            "keyframe_time_dim_available": bool(time_dim_available),
            "keyframe_points_filtered": int(vis_points.shape[0]),
        }

    lidar_path = info.get("lidar_path") or info.get("pts_filename")
    if not lidar_path:
        raise KeyError("Dataset info lacks lidar_path/pts_filename for raw visualization points.")
    resolved_lidar = resolve_data_file_path(lidar_path, assembled, "LiDAR")
    vis_points = read_lidar_bin(resolved_lidar)
    return vis_points, {
        "visualized_points_source": "raw",
        "visualized_lidar_info_path": str(lidar_path),
        "visualized_lidar_path": str(resolved_lidar),
        "visualized_lidar_realpath": str(resolved_lidar.resolve()),
        "pipeline_points_input": int(pipeline_points.shape[0]),
    }


def deterministic_subsample(array, limit):
    if limit <= 0 or len(array) <= limit:
        return array
    idx = np.linspace(0, len(array) - 1, num=limit, dtype=np.int64)
    return array[idx]


def project_lidar_points_to_image(points, lidar2img, image_shape, depth_min, depth_max, max_points):
    if points.shape[1] < 3:
        raise ValueError("Point array must have at least xyz columns.")
    pts = points[:, :3].astype(np.float64)
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    proj = hom @ np.asarray(lidar2img, dtype=np.float64).T
    depth = proj[:, 2]
    valid = np.isfinite(proj).all(axis=1) & (depth > max(depth_min, 1e-6))
    uv = np.zeros((pts.shape[0], 2), dtype=np.float64)
    uv[valid] = proj[valid, :2] / depth[valid, None]
    height, width = image_shape[:2]
    valid &= (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    uv = uv[valid]
    pts_valid = pts[valid]
    ranges = np.linalg.norm(pts_valid, axis=1)
    if depth_max > depth_min:
        valid_range = (ranges >= depth_min) & (ranges <= depth_max)
        uv = uv[valid_range]
        ranges = ranges[valid_range]
    packed = np.concatenate([uv, ranges[:, None]], axis=1) if len(uv) else np.zeros((0, 3))
    return deterministic_subsample(packed, max_points)


def draw_projected_points(image, projected, depth_min, depth_max, radius, alpha):
    if len(projected) == 0:
        return image
    overlay = image.copy()
    norm = (projected[:, 2] - depth_min) / max(depth_max - depth_min, 1e-6)
    norm = np.clip(norm, 0.0, 1.0)
    colors = cv2.applyColorMap((norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)
    for (u, v, _), color in zip(projected, colors[:, 0, :]):
        cv2.circle(overlay, (int(round(u)), int(round(v))), radius, color.tolist(), thickness=-1, lineType=cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)


def clip_and_draw_line(image, p0, p1, color, thickness):
    height, width = image.shape[:2]
    rect = (0, 0, width, height)
    ok, q0, q1 = cv2.clipLine(rect, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))))
    if ok:
        cv2.line(image, q0, q1, color, thickness, lineType=cv2.LINE_AA)


def draw_lidar_boxes_on_image(image, boxes3d, scores, labels, class_names, lidar2img, box_colors_bgr, draw_labels=False):
    if boxes3d is None or len(boxes3d) == 0:
        return image
    boxes_cpu = boxes3d.to("cpu") if hasattr(boxes3d, "to") else boxes3d
    corners = boxes_cpu.corners.numpy()
    num = corners.shape[0]
    hom = np.concatenate([corners.reshape(-1, 3), np.ones((num * 8, 1), dtype=np.float64)], axis=1)
    proj = hom @ np.asarray(lidar2img, dtype=np.float64).T
    depth = proj[:, 2].reshape(num, 8)
    xy = (proj[:, :2] / np.maximum(proj[:, 2:3], 1e-6)).reshape(num, 8, 2)
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    out = image.copy()
    scores_np = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
    labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    for i in range(num):
        if not np.any(depth[i] > 1e-3):
            continue
        color = color_for_label(labels_np[i], class_names, box_colors_bgr)
        for a, b in edges:
            if depth[i, a] <= 1e-3 or depth[i, b] <= 1e-3:
                continue
            clip_and_draw_line(out, xy[i, a], xy[i, b], color, 2)
        if draw_labels and i < len(scores_np):
            visible = xy[i][depth[i] > 1e-3]
            if len(visible):
                x, y = visible[:, 0].min(), visible[:, 1].min()
                cls = class_names[int(labels_np[i])] if int(labels_np[i]) < len(class_names) else str(int(labels_np[i]))
                text = f"{cls} {float(scores_np[i]):.2f}"
                cv2.putText(out, text, (int(max(0, x)), int(max(14, y))), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def box_xy_corners(box):
    # LiDARInstance3DBoxes layout is x, y, z, dx, dy, dz, yaw.
    # dx is the local forward/backward length and dy is the lateral width.
    x, y, length, width, yaw = box[0], box[1], box[3], box[4], box[6]
    c, s = np.cos(yaw), np.sin(yaw)
    corners = np.array([
        [length / 2, width / 2],
        [length / 2, -width / 2],
        [-length / 2, -width / 2],
        [-length / 2, width / 2],
    ])
    return corners @ np.array([[c, -s], [s, c]]).T + np.array([x, y])


def xy_to_bev_pixels(xy, pc_range, size):
    x_min, y_min, _, x_max, y_max, _ = pc_range
    x = xy[:, 0]
    y = xy[:, 1]
    u = (y_max - y) / max(y_max - y_min, 1e-6) * (size - 1)
    v = (x_max - x) / max(x_max - x_min, 1e-6) * (size - 1)
    return np.stack([u, v], axis=1)


def draw_bev(points, boxes3d, labels, class_names, box_colors_bgr, pc_range, size, max_points, point_radius=1, draw_boxes=True):
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pts = points[:, :3]
    mask = (
        (pts[:, 0] >= pc_range[0])
        & (pts[:, 0] <= pc_range[3])
        & (pts[:, 1] >= pc_range[1])
        & (pts[:, 1] <= pc_range[4])
    )
    pts = deterministic_subsample(pts[mask], max_points)
    if len(pts):
        uv = xy_to_bev_pixels(pts[:, :2], pc_range, size)
        for u, v in uv:
            cv2.circle(canvas, (int(round(u)), int(round(v))), point_radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    if draw_boxes and boxes3d is not None and len(boxes3d):
        boxes_np = boxes3d.to("cpu").tensor.numpy() if hasattr(boxes3d, "to") else np.asarray(boxes3d)
        labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
        for i, box in enumerate(boxes_np):
            corners = box_xy_corners(box)
            uv = xy_to_bev_pixels(corners, pc_range, size).astype(np.int32)
            color = color_for_label(labels_np[i], class_names, box_colors_bgr) if i < len(labels_np) else box_colors_bgr["default"]
            cv2.polylines(canvas, [uv], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    return canvas


def draw_front_marker(image):
    height, width = image.shape[:2]
    out = image.copy()
    bottom_margin = min(40, max(16, height // 18))
    arrow_len = min(100, max(30, height // 9))
    start = (width // 2, max(0, height - bottom_margin))
    end = (width // 2, max(8, start[1] - arrow_len))
    cv2.arrowedLine(out, start, end, (180, 180, 180), 3, tipLength=0.2)
    cv2.putText(
        out, "Front", (min(width - 80, width // 2 + 12), max(24, end[1] + 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA
    )
    return out


def center_crop_height(image, target_height):
    height = image.shape[0]
    if target_height <= 0 or target_height >= height:
        return image, 0, height, False
    y0 = (height - target_height) // 2
    y1 = y0 + target_height
    return image[y0:y1, :, :], y0, y1, True


def crop_visible_pc_range(pc_range, canvas_size, crop_y0, crop_y1, crop_applied):
    if not crop_applied or canvas_size <= 1:
        return [float(x) for x in pc_range]
    x_min, y_min, z_min, x_max, y_max, z_max = [float(x) for x in pc_range]
    span = x_max - x_min
    top_v = max(0, min(canvas_size - 1, crop_y0))
    bottom_v = max(0, min(canvas_size - 1, crop_y1 - 1))
    visible_x_max = x_max - (top_v / max(canvas_size - 1, 1)) * span
    visible_x_min = x_max - (bottom_v / max(canvas_size - 1, 1)) * span
    return [float(visible_x_min), y_min, z_min, float(visible_x_max), y_max, z_max]


def compose_panels(camera_image, bev_image):
    height = max(camera_image.shape[0], bev_image.shape[0])
    cam_canvas = np.zeros((height, camera_image.shape[1], 3), dtype=np.uint8)
    y0 = (height - camera_image.shape[0]) // 2
    cam_canvas[y0 : y0 + camera_image.shape[0], :, :] = camera_image
    if bev_image.shape[0] != height:
        bev_canvas = np.zeros((height, bev_image.shape[1], 3), dtype=np.uint8)
        y1 = (height - bev_image.shape[0]) // 2
        bev_canvas[y1 : y1 + bev_image.shape[0], :, :] = bev_image
    else:
        bev_canvas = bev_image
    return np.concatenate([cam_canvas, bev_canvas], axis=1)


def extract_pred(result):
    item = result[0] if isinstance(result, list) else result
    pts_bbox = item.get("pts_bbox", item)
    return pts_bbox["boxes_3d"], pts_bbox["scores_3d"], pts_bbox["labels_3d"]


def filter_predictions(boxes, scores, labels, score_thr, max_boxes):
    keep = scores > score_thr
    if int(keep.sum().item()) == 0:
        return boxes[keep], scores[keep], labels[keep]
    idx = torch.nonzero(keep, as_tuple=False).flatten()
    kept_scores = scores[idx]
    order = torch.argsort(kept_scores, descending=True)
    if max_boxes > 0:
        order = order[:max_boxes]
    idx = idx[order]
    return boxes[idx], scores[idx], labels[idx]


def condition_name(split, corruption, severity):
    if split == "clean":
        return "clean"
    return f"{corruption}_s{severity}"


def should_render_index(index, info, args, explicit_indices, explicit_tokens, rendered_count):
    token = info.get("token") or info.get("sample_idx")
    if explicit_indices and index not in explicit_indices:
        return False
    if explicit_tokens and token not in explicit_tokens:
        return False
    if not explicit_indices and not explicit_tokens and index < args.start_index:
        return False
    return rendered_count < args.samples


def render_condition(
    args, base_cfg, model, class_names, box_colors_rgb, box_colors_bgr,
    clean_root, source_root, condition, output_root, assembly_root
):
    split, corruption, severity = condition
    cond_name = condition_name(split, corruption, severity)
    cond_dir = safe_label(cond_name)
    assembled = assemble_nuscenes_root(clean_root, source_root, assembly_root / cond_dir, assembly_root)
    cfg = build_condition_cfg(base_cfg, assembled, args.workers)
    dataset, data_loader = build_condition_dataset(cfg, args.workers)
    cond_out = output_root / cond_name
    image_out = cond_out / "images"
    json_out = cond_out / "json"
    image_out.mkdir(parents=True, exist_ok=True)
    json_out.mkdir(parents=True, exist_ok=True)

    pc_range = list(getattr(cfg, "point_cloud_range", [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]))
    explicit_indices = {int(x) for x in split_items(args.sample_indices)}
    explicit_tokens = set(split_items(args.sample_tokens))
    rendered = 0
    rows = []

    for index, data in enumerate(data_loader):
        info = dataset.data_infos[index]
        if not should_render_index(index, info, args, explicit_indices, explicit_tokens, rendered):
            if rendered >= args.samples:
                break
            continue
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        boxes, scores, labels = extract_pred(result)
        boxes, scores, labels = filter_predictions(boxes, scores, labels, args.score_thr, args.max_boxes)

        cam_idx = camera_index(info, args.camera)
        cam_info = info["cams"][args.camera]
        image_path = resolve_image_path(cam_info["data_path"], assembled)
        raw_img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if raw_img is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        lidar2img = raw_lidar2img_from_cam_info(cam_info)
        points, points_meta = select_visualization_points(args, info, data, assembled)
        projected = project_lidar_points_to_image(
            points, lidar2img, raw_img.shape, args.depth_min, args.depth_max, args.max_image_points
        )
        cam_vis = draw_projected_points(
            raw_img, projected, args.depth_min, args.depth_max, args.point_radius, args.point_alpha
        )
        if not args.no_image_boxes:
            cam_vis = draw_lidar_boxes_on_image(
                cam_vis, boxes, scores, labels, class_names, lidar2img,
                box_colors_bgr, draw_labels=args.draw_labels
            )
        bev_size = raw_img.shape[1]
        bev_vis = draw_bev(
            points, boxes, labels, class_names, box_colors_bgr,
            pc_range, bev_size, args.max_bev_points,
            point_radius=args.bev_point_radius,
            draw_boxes=not args.no_bev_boxes,
        )
        bev_pre_crop_shape = list(bev_vis.shape)
        bev_crop_y0 = 0
        bev_crop_y1 = bev_vis.shape[0]
        bev_crop_applied = False
        if args.bev_center_crop_to_image_height:
            bev_vis, bev_crop_y0, bev_crop_y1, bev_crop_applied = center_crop_height(bev_vis, raw_img.shape[0])
        if args.show_front_marker:
            bev_vis = draw_front_marker(bev_vis)
        bev_visible_pc_range = crop_visible_pc_range(pc_range, bev_size, bev_crop_y0, bev_crop_y1, bev_crop_applied)
        composite = compose_panels(cam_vis, bev_vis)
        token = info.get("token") or info.get("sample_idx") or first_meta(data).get("sample_idx", str(index))
        stem = f"{rendered:03d}_{index:06d}_{str(token)[:8]}_{cond_name}_{args.camera}_bev"
        out_png = image_out / f"{stem}.png"
        out_json = json_out / f"{stem}.json"
        cv2.imwrite(str(out_png), composite)
        meta = {
            "sample_index": index,
            "render_rank": rendered,
            "sample_token": token,
            "condition": cond_name,
            "split": split,
            "corruption": corruption,
            "severity": severity,
            "camera": args.camera,
            "camera_index": cam_idx,
            "image_path": str(image_path),
            "score_threshold": args.score_thr,
            "max_boxes": args.max_boxes,
            "rendered_boxes": int(len(boxes)),
            "projected_points": int(len(projected)),
            "visualized_points_input": int(points.shape[0]),
            "bev_points_input": int(points.shape[0]),
            "image_shape": list(raw_img.shape),
            "bev_pre_crop_shape": bev_pre_crop_shape,
            "bev_shape": list(bev_vis.shape),
            "composite_shape": list(composite.shape),
            "bev_center_crop_to_image_height": bool(args.bev_center_crop_to_image_height),
            "bev_crop_applied": bool(bev_crop_applied),
            "bev_crop_y0": int(bev_crop_y0),
            "bev_crop_y1": int(bev_crop_y1),
            "bev_crop_box_xyxy_exclusive": [0, int(bev_crop_y0), int(bev_pre_crop_shape[1]), int(bev_crop_y1)],
            "bev_crop_removed_pixels": [
                int(bev_crop_y0),
                int(bev_pre_crop_shape[0] - bev_crop_y1),
            ],
            "bev_crop_target_shape": [int(raw_img.shape[0]), int(raw_img.shape[1]), int(raw_img.shape[2])],
            "point_cloud_range": [float(x) for x in pc_range],
            "bev_visible_pc_range": bev_visible_pc_range,
            "show_front_marker": bool(args.show_front_marker),
            "bev_front_marker_drawn": bool(args.show_front_marker),
            "bev_front_marker_draw_stage": "post_crop" if args.show_front_marker else "none",
            "bev_front_direction": "up",
            "box_color_palette": args.box_color_palette,
            "box_color_overrides": args.box_color_overrides,
            "box_colors_rgb": {k: list(v) for k, v in box_colors_rgb.items()},
            "output_png": str(out_png),
        }
        meta.update(points_meta)
        out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        rows.append(meta)
        rendered += 1
        print(f"[runtime-vis] {cond_name}: wrote {out_png}")
        if rendered >= args.samples:
            break
    return rows


def build_conditions(args):
    splits = split_items(args.split)
    corruptions = split_items(args.corruptions)
    severities = split_items(args.severities)
    conditions = []
    splits = [safe_component(item, "split") for item in splits]
    corruptions = [safe_component(item, "corruption") for item in corruptions]
    severities = [safe_component(item, "severity") for item in severities]
    if "clean" in splits:
        conditions.append(("clean", "clean", "0"))
    if "corrupt" in splits:
        for corr in corruptions:
            for sev in severities:
                conditions.append(("corrupt", corr, str(sev)))
    if not conditions:
        raise ValueError("--split must include clean and/or corrupt.")
    return conditions


def main():
    args = parse_args()
    config, checkpoint, label = resolve_config_checkpoint(args)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir or f"work_dirs/runtime_visualizations/{safe_label(label)}_{timestamp}").resolve()
    assembly_root = Path(args.assembly_dir or out_root / "_assembled_nuscenes").resolve()
    if not args.allow_external_assembly_dir and not is_relative_to(assembly_root, out_root):
        raise ValueError(
            f"--assembly-dir must be inside --out-dir unless "
            f"--allow-external-assembly-dir is set: {assembly_root} not under {out_root}"
        )
    out_root.mkdir(parents=True, exist_ok=True)
    assembly_root.mkdir(parents=True, exist_ok=True)

    clean_root = resolve_user_path(args.clean_data_root).resolve()
    if not clean_root.is_dir():
        raise FileNotFoundError(f"Missing clean data root: {clean_root}")

    conditions = []
    for split, corr, sev in build_conditions(args):
        if split == "clean":
            source = clean_root
        else:
            source = Path(args.multicorrupt_root).resolve() / corr / str(sev)
            if not source.is_dir():
                raise FileNotFoundError(f"Missing MultiCorrupt condition root: {source}")
        conditions.append((split, corr, sev, source))

    preview_box_colors_rgb = build_box_color_map_rgb(
        args.box_color_palette, args.box_color_overrides, DEFAULT_CLASSES
    )
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
        "score_thr": args.score_thr,
        "camera": args.camera,
        "visualization_layout_version": "front_bev_v2",
        "output_layout": {
            "run_json_dir": "json",
            "condition_image_dir": "<condition>/images",
            "condition_json_dir": "<condition>/json",
        },
        "bev_canvas_sizing": "square_from_camera_width",
        "compose_mode": "horizontal_center_pad",
        "bev_crop_mode": "center_to_image_height" if args.bev_center_crop_to_image_height else "none",
        "bev_crop_anchor": "center" if args.bev_center_crop_to_image_height else "none",
        "bev_crop_target": "camera_image" if args.bev_center_crop_to_image_height else "none",
        "bev_front_marker": "arrow_text" if args.show_front_marker else "none",
        "vis_points_source": args.vis_points_source,
        "keyframe_time_eps": args.keyframe_time_eps,
        "depth_min": args.depth_min,
        "depth_max": args.depth_max,
        "point_radius": args.point_radius,
        "point_alpha": args.point_alpha,
        "max_image_points": args.max_image_points,
        "max_bev_points": args.max_bev_points,
        "bev_point_radius": args.bev_point_radius,
        "draw_labels": bool(args.draw_labels),
        "draw_bev_boxes": not bool(args.no_bev_boxes),
        "draw_image_boxes": not bool(args.no_image_boxes),
        "box_color_palette": args.box_color_palette,
        "box_color_overrides": args.box_color_overrides,
        "box_color_class_reference": DEFAULT_CLASSES,
        "box_colors_rgb": {k: list(v) for k, v in preview_box_colors_rgb.items()},
        "show_front_marker": bool(args.show_front_marker),
        "bev_center_crop_to_image_height": bool(args.bev_center_crop_to_image_height),
        "max_boxes": args.max_boxes,
    }
    (json_root / "runtime_visualize_run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(json.dumps(run_meta, indent=2))
    if args.dry_run:
        print("[runtime-vis] dry run complete; no inference executed.")
        return

    ensure_runtime_imports()
    cfg = load_config(config, args.cfg_options)
    setup_multi_processes(cfg)
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=False)
    model = build_runtime_model(cfg, checkpoint, args.gpu_id, args.fuse_conv_bn)
    class_names = list(getattr(model.module, "CLASSES", None) or getattr(cfg, "class_names", DEFAULT_CLASSES))
    box_colors_rgb = build_box_color_map_rgb(args.box_color_palette, args.box_color_overrides, class_names)
    box_colors_bgr = {k: rgb_to_bgr(v) for k, v in box_colors_rgb.items()}
    run_meta.update({
        "box_color_class_reference": class_names,
        "box_colors_rgb": {k: list(v) for k, v in box_colors_rgb.items()},
    })
    (json_root / "runtime_visualize_run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    manifest_rows = []
    try:
        for split, corr, sev, source in conditions:
            rows = render_condition(
                args, cfg, model, class_names, box_colors_rgb, box_colors_bgr, clean_root, source,
                (split, corr, sev), out_root, assembly_root
            )
            manifest_rows.extend(rows)
    finally:
        if not args.keep_assembled and not args.allow_external_assembly_dir and assembly_root.exists():
            shutil.rmtree(assembly_root)

    manifest_path = json_root / "render_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[runtime-vis] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
