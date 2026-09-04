#!/usr/bin/env python3
"""Render clean-vs-corrupt MultiCorrupt examples for thesis figures.

This utility reads clean nuScenes metadata and resolves the matching corrupted
files from a MultiCorrupt root. It does not run model inference and it does not
mutate the dataset tree. Each output image is a two-panel composition:

  left:  clean sample
  right: corrupted sample

No text, legends, arrows, ego markers, or other labels are drawn inside the
image. A JSON manifest records paths and rendering modes for traceability.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DATA_SUBDIRS = ("samples", "sweeps", "v1.0-trainval", "maps", "lidarseg")

DEFAULT_CORRUPTIONS = (
    "beamsreducing",
    "brightness",
    "dark",
    "fog",
    "missingcamera",
    "motionblur",
    "pointsreducing",
    "snow",
    "spatialmisalignment",
    "temporalmisalignment",
)

# Rendering mode, not model/router domain. The mode decides which visual signal
# best communicates the corruption type in a clean-vs-corrupt figure.
DEFAULT_MODE_BY_CORRUPTION = {
    "beamsreducing": "bev",
    "pointsreducing": "bev",
    "brightness": "image",
    "dark": "image",
    "darkness": "image",
    "missingcamera": "image",
    "motionblur": "image",
    "fog": "overlay",
    "snow": "overlay",
    "spatialmisalignment": "overlay",
    "temporalmisalignment": "overlay",
}

CORRUPTION_ALIASES = {
    "darkness": "dark",
}

CORRUPT_METADATA_CORRUPTIONS = {"spatialmisalignment", "temporalmisalignment"}

RESOLVE_CACHE: Dict[Tuple[str, str, str, str], Optional[Path]] = {}
TABLE_CACHE: Dict[Tuple[str, str, str, str], Optional["NuScenesTables"]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate clean-vs-corrupt MultiCorrupt example images without "
            "running model inference."
        )
    )
    parser.add_argument(
        "--clean-root",
        default=Path("data/nuscenes"),
        type=Path,
        help="Clean nuScenes dataroot containing samples/ and v1.0-trainval/.",
    )
    parser.add_argument(
        "--multicorrupt-root",
        default=Path("/media/ting/Dataset/MultiCorrupt"),
        type=Path,
        help="MultiCorrupt root, usually <root>/<corruption>/<severity>/...",
    )
    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        help="nuScenes metadata version directory under --clean-root.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("work_dirs/multicorrupt_examples"),
        type=Path,
        help="Directory for output PNGs and manifest JSON.",
    )
    parser.add_argument(
        "--severity",
        default="3",
        help="Corruption severity to render. Usually 1, 2, or 3.",
    )
    parser.add_argument(
        "--corruptions",
        default=",".join(DEFAULT_CORRUPTIONS),
        help=(
            "Comma/space separated corruption names. Defaults to the ten "
            "MultiCorrupt types used by the project."
        ),
    )
    parser.add_argument(
        "--mode-map",
        default="",
        help=(
            "Optional comma/space separated overrides like "
            "motionblur=image,fog=overlay. Valid modes: bev,image,overlay."
        ),
    )
    parser.add_argument(
        "--camera",
        default="CAM_FRONT",
        help="Camera channel for image and overlay modes.",
    )
    parser.add_argument(
        "--lidar",
        default="LIDAR_TOP",
        help="LiDAR channel for BEV and overlay modes.",
    )
    parser.add_argument(
        "--sample-token",
        default=None,
        help="Optional nuScenes sample token. If omitted, samples are searched.",
    )
    parser.add_argument(
        "--sample-index",
        default=0,
        type=int,
        help="Starting sample index when searching.",
    )
    parser.add_argument(
        "--same-sample",
        action="store_true",
        help="Use one sample token for every corruption. Default searches per corruption.",
    )
    parser.add_argument(
        "--max-search-samples",
        default=0,
        type=int,
        help="Maximum samples to scan when searching. 0 means all samples.",
    )
    parser.add_argument(
        "--panel-width",
        default=960,
        type=int,
        help="Width of each clean/corrupt panel.",
    )
    parser.add_argument(
        "--panel-height",
        default=540,
        type=int,
        help="Height of each clean/corrupt panel.",
    )
    parser.add_argument(
        "--gap",
        default=0,
        type=int,
        help="Blank pixel gap between clean and corrupt panels.",
    )
    parser.add_argument(
        "--gap-color",
        default="0,0,0",
        help="BGR color for --gap, e.g. 0,0,0 or 255,255,255.",
    )
    parser.add_argument(
        "--bev-range",
        default=54.0,
        type=float,
        help="Symmetric BEV x/y range in meters.",
    )
    parser.add_argument(
        "--bev-point-radius",
        default=1,
        type=int,
        help="Point radius for BEV rendering.",
    )
    parser.add_argument(
        "--max-bev-points",
        default=180000,
        type=int,
        help="Maximum LiDAR points rendered in a BEV panel.",
    )
    parser.add_argument(
        "--depth-min",
        default=1.0,
        type=float,
        help="Minimum projected depth in meters.",
    )
    parser.add_argument(
        "--depth-max",
        default=60.0,
        type=float,
        help="Depth used as the far end of the projection colormap.",
    )
    parser.add_argument(
        "--max-image-points",
        default=35000,
        type=int,
        help="Maximum projected LiDAR points drawn on an image panel.",
    )
    parser.add_argument(
        "--image-point-radius",
        default=4,
        type=int,
        help="Point radius for projected LiDAR points.",
    )
    parser.add_argument(
        "--image-point-alpha",
        default=0.85,
        type=float,
        help="Alpha for projected LiDAR point overlay.",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help=(
            "Allow corrupted streams to fall back to clean files when missing. "
            "Warnings are written to the manifest. By default, the expected "
            "corrupted stream for each mode must exist."
        ),
    )
    parser.add_argument(
        "--min-missingcamera-image-diff",
        default=10.0,
        type=float,
        help=(
            "Minimum mean absolute pixel difference required when searching "
            "for missingcamera examples. Set 0 to disable this filter."
        ),
    )
    parser.add_argument(
        "--corrupt-metadata-policy",
        choices=("auto", "never", "always"),
        default="auto",
        help=(
            "When to load corrupted nuScenes metadata. auto loads it only for "
            "spatial/temporal misalignment, where calibration/pose metadata can "
            "change. never is most memory efficient; always is for debugging."
        ),
    )
    parser.add_argument(
        "--no-recursive-resolve",
        action="store_true",
        help="Disable recursive filename search under corruption/severity roots.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve samples and files, write manifest, but do not render PNGs.",
    )
    return parser.parse_args()


def require_deps() -> None:
    global np, cv2
    import numpy as np  # type: ignore
    import cv2  # type: ignore


def parse_list(value: str) -> List[str]:
    items: List[str] = []
    for raw in value.replace(",", " ").split():
        item = raw.strip()
        if item:
            items.append(CORRUPTION_ALIASES.get(item, item))
    return items


def parse_bgr(value: str) -> Tuple[int, int, int]:
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected B,G,R color triple, got: {value}")
    return tuple(max(0, min(255, v)) for v in parts)  # type: ignore[return-value]


def parse_mode_map(value: str) -> Dict[str, str]:
    mode_map = dict(DEFAULT_MODE_BY_CORRUPTION)
    if not value.strip():
        return mode_map
    for item in value.replace(",", " ").split():
        if "=" not in item:
            raise ValueError(f"Invalid --mode-map item: {item}")
        name, mode = [p.strip() for p in item.split("=", 1)]
        name = CORRUPTION_ALIASES.get(name, name)
        if mode not in {"bev", "image", "overlay"}:
            raise ValueError(f"Invalid render mode for {name}: {mode}")
        mode_map[name] = mode
    return mode_map


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class ResolvedFile:
    path: Path
    source: str
    fallback: bool = False


class NuScenesTables:
    def __init__(self, dataroot: Path, version: str):
        meta = dataroot / version
        if not meta.is_dir():
            raise FileNotFoundError(f"nuScenes metadata directory not found: {meta}")
        self.dataroot = dataroot
        self.version = version
        self.samples = read_json(meta / "sample.json")
        sample_data = read_json(meta / "sample_data.json")
        calibrated_sensor = read_json(meta / "calibrated_sensor.json")
        ego_pose = read_json(meta / "ego_pose.json")
        sensor = read_json(meta / "sensor.json")

        self.sample_by_token = {r["token"]: r for r in self.samples}
        self.sample_data_by_token = {r["token"]: r for r in sample_data}
        self.calib_by_token = {r["token"]: r for r in calibrated_sensor}
        self.ego_by_token = {r["token"]: r for r in ego_pose}
        self.sensor_by_token = {r["token"]: r for r in sensor}
        self.sample_channel: Dict[Tuple[str, str], dict] = {}
        for rec in sample_data:
            calib = self.calib_by_token[rec["calibrated_sensor_token"]]
            sensor_rec = self.sensor_by_token[calib["sensor_token"]]
            key = (rec["sample_token"], sensor_rec["channel"])
            if key not in self.sample_channel or rec.get("is_key_frame", False):
                self.sample_channel[key] = rec

    def get_sample(self, token: Optional[str], index: int) -> dict:
        if token:
            if token not in self.sample_by_token:
                raise KeyError(f"sample token not found: {token}")
            return self.sample_by_token[token]
        if not self.samples:
            raise RuntimeError("sample.json is empty.")
        index = max(0, min(index, len(self.samples) - 1))
        return self.samples[index]

    def get_sample_data(self, sample_token: str, channel: str) -> dict:
        key = (sample_token, channel)
        if key not in self.sample_channel:
            raise KeyError(
                f"sample_data not found for sample={sample_token}, channel={channel}"
            )
        return self.sample_channel[key]

    def full_path(self, sample_data_rec: dict) -> Path:
        return self.dataroot / sample_data_rec["filename"]


def q_to_rot(q: Sequence[float]):
    # nuScenes stores quaternions as [w, x, y, z].
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_points(points, rotation, translation):
    return rotation @ points + np.asarray(translation, dtype=np.float64).reshape(3, 1)


def inverse_transform_points(points, rotation, translation):
    return rotation.T @ (points - np.asarray(translation, dtype=np.float64).reshape(3, 1))


def condition_root_candidates(root: Path, corruption: str, severity: str) -> List[Path]:
    return [
        root / corruption / str(severity),
        root / corruption / f"severity_{severity}",
        root / corruption / f"s{severity}",
        root / corruption / f"sev{severity}",
        root / corruption / f"level_{severity}",
        root / f"{corruption}_{severity}",
        root / f"{corruption}_s{severity}",
        root / f"{corruption}_severity_{severity}",
    ]


def find_condition_root(root: Path, corruption: str, severity: str, version: str) -> Optional[Path]:
    for candidate in condition_root_candidates(root, corruption, severity):
        if (candidate / version).is_dir():
            return candidate
    fallback = root / corruption
    if (fallback / version).is_dir():
        return fallback
    return None


def get_corrupt_tables(
    clean_tables: NuScenesTables,
    args: argparse.Namespace,
    corruption: str,
) -> Optional[NuScenesTables]:
    key = (
        str(args.multicorrupt_root),
        corruption,
        str(args.severity),
        clean_tables.version,
    )
    if key in TABLE_CACHE:
        return TABLE_CACHE[key]
    condition_root = find_condition_root(
        args.multicorrupt_root,
        corruption,
        str(args.severity),
        clean_tables.version,
    )
    if condition_root is None:
        TABLE_CACHE[key] = None
        return None
    try:
        tables = NuScenesTables(condition_root, clean_tables.version)
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        TABLE_CACHE[key] = None
        return None
    TABLE_CACHE[key] = tables
    return tables


def should_use_corrupt_metadata(args: argparse.Namespace, corruption: str, mode: str) -> bool:
    policy = getattr(args, "corrupt_metadata_policy", "auto")
    if policy == "always":
        return True
    if policy == "never":
        return False
    return mode == "overlay" and corruption in CORRUPT_METADATA_CORRUPTIONS


def resolve_corrupt_file(
    root: Path,
    corruption: str,
    severity: str,
    rel_filename: str,
    recursive: bool = True,
) -> Optional[Path]:
    cache_key = (str(root), corruption, str(severity), rel_filename)
    if cache_key in RESOLVE_CACHE:
        return RESOLVE_CACHE[cache_key]

    condition_roots = condition_root_candidates(root, corruption, severity)
    candidates = [base / rel_filename for base in condition_roots]
    candidates.append(root / corruption / rel_filename)
    for path in candidates:
        if path.exists():
            RESOLVE_CACHE[cache_key] = path
            return path

    if recursive:
        filename = Path(rel_filename).name
        search_roots = [base for base in condition_roots if base.exists()]
        if not search_roots and (root / corruption).exists():
            search_roots = [root / corruption]
        for base in search_roots:
            matches = sorted(base.rglob(filename))
            if matches:
                RESOLVE_CACHE[cache_key] = matches[0]
                return matches[0]

    RESOLVE_CACHE[cache_key] = None
    return None


def load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def image_mean_abs_diff(a_path: Path, b_path: Path) -> float:
    a = load_image(a_path)
    b = load_image(b_path)
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return float(diff.mean())


def resize_cover(img, size: Tuple[int, int]):
    width, height = size
    h, w = img.shape[:2]
    scale = max(width / w, height / h)
    resized = cv2.resize(
        img,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    rh, rw = resized.shape[:2]
    y0 = max(0, (rh - height) // 2)
    x0 = max(0, (rw - width) // 2)
    return resized[y0 : y0 + height, x0 : x0 + width]


def load_lidar_points(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"LiDAR file not found: {path}")
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size % 5 != 0:
        raise ValueError(f"Unexpected LiDAR binary shape: {path}, values={raw.size}")
    return raw.reshape(-1, 5)[:, :3].T


def deterministic_subsample_columns(points, limit: int):
    if limit <= 0 or points.shape[1] <= limit:
        return points
    idx = np.linspace(0, points.shape[1] - 1, num=limit, dtype=np.int64)
    return points[:, idx]


def center_crop_height(image, target_height: int):
    height = image.shape[0]
    if target_height <= 0 or target_height >= height:
        return image
    y0 = (height - target_height) // 2
    return image[y0 : y0 + target_height, :, :]


def center_crop_width(image, target_width: int):
    width = image.shape[1]
    if target_width <= 0 or target_width >= width:
        return image
    x0 = (width - target_width) // 2
    return image[:, x0 : x0 + target_width, :]


def render_bev(
    lidar_path: Path,
    size: Tuple[int, int],
    bev_range: float,
    max_points: int,
    point_radius: int,
):
    points = load_lidar_points(lidar_path)
    points = deterministic_subsample_columns(points, max_points)
    width, height = size
    canvas_size = max(int(width), int(height))
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    x, y = points[0], points[1]
    mask = (x >= -bev_range) & (x <= bev_range) & (y >= -bev_range) & (y <= bev_range)
    x, y = x[mask], y[mask]
    if x.size == 0:
        return center_crop_width(center_crop_height(canvas, height), width)
    px = ((y + bev_range) / (2 * bev_range) * (canvas_size - 1)).astype(np.int32)
    py = ((bev_range - x) / (2 * bev_range) * (canvas_size - 1)).astype(np.int32)
    point_radius = max(1, int(point_radius))
    if point_radius <= 1:
        canvas[py, px] = (255, 255, 255)
    else:
        for u, v in zip(px, py):
            cv2.circle(canvas, (int(u), int(v)), point_radius, (255, 255, 255), -1)
    return center_crop_width(center_crop_height(canvas, height), width)


def project_lidar_to_camera(
    tables: NuScenesTables,
    lidar_rec: dict,
    cam_rec: dict,
    lidar_path: Path,
    image,
    depth_min: float,
    depth_max: float,
    max_points: int,
    point_radius: int,
    alpha: float,
):
    points = load_lidar_points(lidar_path)
    points = deterministic_subsample_columns(points, max_points)
    lidar_ranges = np.linalg.norm(points[:3, :], axis=0)

    lidar_calib = tables.calib_by_token[lidar_rec["calibrated_sensor_token"]]
    lidar_ego = tables.ego_by_token[lidar_rec["ego_pose_token"]]
    cam_calib = tables.calib_by_token[cam_rec["calibrated_sensor_token"]]
    cam_ego = tables.ego_by_token[cam_rec["ego_pose_token"]]

    points = transform_points(points, q_to_rot(lidar_calib["rotation"]), lidar_calib["translation"])
    points = transform_points(points, q_to_rot(lidar_ego["rotation"]), lidar_ego["translation"])
    points = inverse_transform_points(points, q_to_rot(cam_ego["rotation"]), cam_ego["translation"])
    points = inverse_transform_points(points, q_to_rot(cam_calib["rotation"]), cam_calib["translation"])

    depth = points[2, :]
    intrinsic = np.asarray(cam_calib["camera_intrinsic"], dtype=np.float64)
    uvw = intrinsic @ points
    uv = uvw[:2, :] / np.maximum(uvw[2:3, :], 1e-6)
    h, w = image.shape[:2]
    mask = (
        (depth > depth_min)
        & (depth < depth_max)
        & (uv[0] >= 0)
        & (uv[0] < w)
        & (uv[1] >= 0)
        & (uv[1] < h)
    )
    uv = uv[:, mask].T
    depth = depth[mask]
    ranges = lidar_ranges[mask]
    if depth_max > depth_min:
        range_mask = (ranges >= depth_min) & (ranges <= depth_max)
        uv = uv[range_mask]
        depth = depth[range_mask]
        ranges = ranges[range_mask]
    if uv.size == 0:
        return image.copy()

    range_norm = np.clip((ranges - depth_min) / max(depth_max - depth_min, 1e-6), 0, 1)
    colors = cv2.applyColorMap(
        (range_norm * 255).astype(np.uint8).reshape(-1, 1),
        cv2.COLORMAP_TURBO,
    )
    overlay = image.copy()
    point_radius = max(1, int(point_radius))
    for (u, v), color in zip(uv, colors[:, 0, :]):
        cv2.circle(
            overlay,
            (int(round(u)), int(round(v))),
            point_radius,
            color.tolist(),
            -1,
            cv2.LINE_AA,
        )
    alpha = max(0.0, min(1.0, float(alpha)))
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def compose_clean_corrupt(left, right, gap: int, gap_color: Tuple[int, int, int]):
    if left.shape[:2] != right.shape[:2]:
        raise ValueError(f"Panel sizes differ: {left.shape[:2]} vs {right.shape[:2]}")
    if gap <= 0:
        return np.concatenate([left, right], axis=1)
    spacer = np.full((left.shape[0], gap, 3), gap_color, dtype=np.uint8)
    return np.concatenate([left, spacer, right], axis=1)


def iter_search_samples(tables: NuScenesTables, args: argparse.Namespace) -> Iterable[dict]:
    start = max(0, args.sample_index)
    samples = tables.samples[start:] + tables.samples[:start]
    if args.max_search_samples and args.max_search_samples > 0:
        samples = samples[: args.max_search_samples]
    return samples


def resolve_corrupt_stream(
    clean_tables: NuScenesTables,
    clean_sample_data: dict,
    args: argparse.Namespace,
    corruption: str,
    expected_corrupt: bool,
    use_corrupt_metadata: bool,
) -> Tuple[Optional[ResolvedFile], Optional[dict], NuScenesTables, List[str]]:
    recursive = not args.no_recursive_resolve
    warnings: List[str] = []
    corrupt_tables = get_corrupt_tables(clean_tables, args, corruption) if use_corrupt_metadata else None
    stream_tables = clean_tables
    sample_data = clean_sample_data

    if corrupt_tables is not None:
        try:
            clean_calib = clean_tables.calib_by_token[clean_sample_data["calibrated_sensor_token"]]
            clean_sensor = clean_tables.sensor_by_token[clean_calib["sensor_token"]]
            sample_data = corrupt_tables.get_sample_data(
                clean_sample_data["sample_token"],
                clean_sensor["channel"],
            )
            stream_tables = corrupt_tables
            corrupt_path = corrupt_tables.full_path(sample_data)
            if corrupt_path.exists():
                return (
                    ResolvedFile(corrupt_path, "corrupt_metadata", False),
                    sample_data,
                    stream_tables,
                    warnings,
                )
            warnings.append(f"corrupt metadata path missing: {corrupt_path}")
        except KeyError as exc:
            warnings.append(f"corrupt metadata lookup failed: {exc}")

    rel_filename = sample_data["filename"]
    corrupt = resolve_corrupt_file(
        args.multicorrupt_root,
        corruption,
        str(args.severity),
        rel_filename,
        recursive=recursive,
    )
    if corrupt is not None:
        return ResolvedFile(corrupt, "corrupt_file", False), sample_data, stream_tables, warnings
    if expected_corrupt and not args.allow_fallback:
        warnings.append(f"missing expected corrupt stream: {corruption}/{args.severity}/{rel_filename}")
        return None, None, clean_tables, warnings
    clean = clean_tables.full_path(clean_sample_data)
    warnings.append(f"fallback to clean stream: {rel_filename}")
    return ResolvedFile(clean, "clean_fallback", True), clean_sample_data, clean_tables, warnings


def corruption_mode(corruption: str, mode_map: Dict[str, str]) -> str:
    if corruption not in mode_map:
        return "overlay"
    return mode_map[corruption]


def sample_has_required_files(
    tables: NuScenesTables,
    sample: dict,
    corruption: str,
    args: argparse.Namespace,
    mode_map: Dict[str, str],
) -> Tuple[bool, str]:
    token = sample["token"]
    try:
        cam = tables.get_sample_data(token, args.camera)
        lidar = tables.get_sample_data(token, args.lidar)
    except KeyError as exc:
        return False, str(exc)

    clean_img = tables.full_path(cam)
    clean_lidar = tables.full_path(lidar)
    if not clean_img.exists():
        return False, f"clean image missing: {clean_img}"
    if not clean_lidar.exists():
        return False, f"clean lidar missing: {clean_lidar}"

    mode = corruption_mode(corruption, mode_map)
    use_metadata = should_use_corrupt_metadata(args, corruption, mode)
    if mode in {"image", "overlay"}:
        expected = mode == "image" or (mode == "overlay" and not args.allow_fallback)
        corrupt_img, _, _, warnings = resolve_corrupt_stream(
            tables, cam, args, corruption, expected, use_metadata
        )
        if corrupt_img is None:
            return False, "; ".join(warnings)
        if (
            corruption == "missingcamera"
            and not args.dry_run
            and args.min_missingcamera_image_diff > 0
            and not corrupt_img.fallback
        ):
            diff = image_mean_abs_diff(clean_img, corrupt_img.path)
            if diff < args.min_missingcamera_image_diff:
                return (
                    False,
                    "missingcamera CAM_FRONT image difference too small: "
                    f"{diff:.3f} < {args.min_missingcamera_image_diff:.3f}",
                )
    if mode in {"bev", "overlay"}:
        expected = mode == "bev" or (mode == "overlay" and not args.allow_fallback)
        corrupt_lidar, _, _, warnings = resolve_corrupt_stream(
            tables, lidar, args, corruption, expected, use_metadata
        )
        if corrupt_lidar is None:
            return False, "; ".join(warnings)

    return True, "ok"


def find_valid_sample(
    tables: NuScenesTables,
    args: argparse.Namespace,
    mode_map: Dict[str, str],
) -> dict:
    if args.sample_token:
        return tables.get_sample(args.sample_token, args.sample_index)

    last_reason = None
    corruptions = parse_list(args.corruptions)
    for sample in iter_search_samples(tables, args):
        ok = True
        for corruption in corruptions:
            ok, reason = sample_has_required_files(tables, sample, corruption, args, mode_map)
            if not ok:
                last_reason = reason
                break
        if ok:
            return sample
    raise RuntimeError(
        "Could not find a sample valid for all requested corruptions. "
        "Try omitting --same-sample or passing --sample-token. "
        f"Last missing reason: {last_reason}"
    )


def find_valid_sample_for_corruption(
    tables: NuScenesTables,
    corruption: str,
    args: argparse.Namespace,
    mode_map: Dict[str, str],
) -> dict:
    if args.sample_token:
        sample = tables.get_sample(args.sample_token, args.sample_index)
        ok, reason = sample_has_required_files(tables, sample, corruption, args, mode_map)
        if not ok:
            raise RuntimeError(f"Requested --sample-token is not valid for {corruption}: {reason}")
        return sample

    last_reason = None
    for sample in iter_search_samples(tables, args):
        ok, reason = sample_has_required_files(tables, sample, corruption, args, mode_map)
        if ok:
            return sample
        last_reason = reason
    raise RuntimeError(
        f"Could not find valid sample for corruption={corruption}, severity={args.severity}. "
        f"Last missing reason: {last_reason}"
    )


def resolve_pair_metadata(
    tables: NuScenesTables,
    sample: dict,
    corruption: str,
    mode: str,
    args: argparse.Namespace,
) -> dict:
    token = sample["token"]
    cam_rec = tables.get_sample_data(token, args.camera)
    lidar_rec = tables.get_sample_data(token, args.lidar)
    clean_img_path = tables.full_path(cam_rec)
    clean_lidar_path = tables.full_path(lidar_rec)
    warnings: List[str] = []
    corrupt_img: Optional[ResolvedFile] = None
    corrupt_lidar: Optional[ResolvedFile] = None
    use_metadata = should_use_corrupt_metadata(args, corruption, mode)

    if mode in {"image", "overlay"}:
        expected = mode == "image" or (mode == "overlay" and not args.allow_fallback)
        corrupt_img, _, _, img_warnings = resolve_corrupt_stream(
            tables,
            cam_rec,
            args,
            corruption,
            expected_corrupt=expected,
            use_corrupt_metadata=use_metadata,
        )
        warnings.extend(img_warnings)
    if mode in {"bev", "overlay"}:
        expected = mode == "bev" or (mode == "overlay" and not args.allow_fallback)
        corrupt_lidar, _, _, lidar_warnings = resolve_corrupt_stream(
            tables,
            lidar_rec,
            args,
            corruption,
            expected_corrupt=expected,
            use_corrupt_metadata=use_metadata,
        )
        warnings.extend(lidar_warnings)

    return {
        "corruption": corruption,
        "severity": str(args.severity),
        "mode": mode,
        "sample_token": token,
        "camera": args.camera,
        "lidar": args.lidar,
        "clean_image": str(clean_img_path),
        "clean_lidar": str(clean_lidar_path),
        "corrupt_image": str(corrupt_img.path) if corrupt_img else None,
        "corrupt_image_source": corrupt_img.source if corrupt_img else None,
        "corrupt_lidar": str(corrupt_lidar.path) if corrupt_lidar else None,
        "corrupt_lidar_source": corrupt_lidar.source if corrupt_lidar else None,
        "warnings": warnings,
    }


def render_pair(
    tables: NuScenesTables,
    sample: dict,
    corruption: str,
    mode: str,
    args: argparse.Namespace,
) -> Tuple[object, dict]:
    token = sample["token"]
    cam_rec = tables.get_sample_data(token, args.camera)
    lidar_rec = tables.get_sample_data(token, args.lidar)
    clean_img_path = tables.full_path(cam_rec)
    clean_lidar_path = tables.full_path(lidar_rec)
    panel_size = (args.panel_width, args.panel_height)
    meta = resolve_pair_metadata(tables, sample, corruption, mode, args)
    corrupt_img: Optional[ResolvedFile] = None
    corrupt_lidar: Optional[ResolvedFile] = None
    corrupt_cam_rec = cam_rec
    corrupt_lidar_rec = lidar_rec
    corrupt_cam_tables = tables
    corrupt_lidar_tables = tables
    use_metadata = should_use_corrupt_metadata(args, corruption, mode)

    if mode in {"image", "overlay"}:
        expected = mode == "image" or (mode == "overlay" and not args.allow_fallback)
        corrupt_img, resolved_cam, corrupt_cam_tables, _ = resolve_corrupt_stream(
            tables,
            cam_rec,
            args,
            corruption,
            expected_corrupt=expected,
            use_corrupt_metadata=use_metadata,
        )
        if resolved_cam is not None:
            corrupt_cam_rec = resolved_cam
    if mode in {"bev", "overlay"}:
        expected = mode == "bev" or (mode == "overlay" and not args.allow_fallback)
        corrupt_lidar, resolved_lidar, corrupt_lidar_tables, _ = resolve_corrupt_stream(
            tables,
            lidar_rec,
            args,
            corruption,
            expected_corrupt=expected,
            use_corrupt_metadata=use_metadata,
        )
        if resolved_lidar is not None:
            corrupt_lidar_rec = resolved_lidar

    if mode == "bev":
        if corrupt_lidar is None:
            raise FileNotFoundError(f"Corrupt LiDAR file not found for {corruption}.")
        left = render_bev(
            clean_lidar_path,
            panel_size,
            args.bev_range,
            args.max_bev_points,
            args.bev_point_radius,
        )
        right = render_bev(
            corrupt_lidar.path,
            panel_size,
            args.bev_range,
            args.max_bev_points,
            args.bev_point_radius,
        )
    elif mode == "image":
        if corrupt_img is None:
            raise FileNotFoundError(f"Corrupt image file not found for {corruption}.")
        left = resize_cover(load_image(clean_img_path), panel_size)
        right = resize_cover(load_image(corrupt_img.path), panel_size)
    elif mode == "overlay":
        if corrupt_img is None or corrupt_lidar is None:
            raise FileNotFoundError(f"Corrupt image/lidar stream not found for {corruption}.")
        clean_img = load_image(clean_img_path)
        corrupt_img_data = load_image(corrupt_img.path)
        right_tables = corrupt_lidar_tables
        right_lidar_rec = corrupt_lidar_rec
        right_cam_rec = corrupt_cam_rec
        if corrupt_cam_tables is not corrupt_lidar_tables:
            meta.setdefault("warnings", []).append(
                "mixed camera/lidar metadata tables in overlay; using clean metadata for projection"
            )
            right_tables = tables
            right_lidar_rec = lidar_rec
            right_cam_rec = cam_rec
        left = project_lidar_to_camera(
            tables,
            lidar_rec,
            cam_rec,
            clean_lidar_path,
            clean_img,
            args.depth_min,
            args.depth_max,
            args.max_image_points,
            args.image_point_radius,
            args.image_point_alpha,
        )
        right = project_lidar_to_camera(
            right_tables,
            right_lidar_rec,
            right_cam_rec,
            corrupt_lidar.path,
            corrupt_img_data,
            args.depth_min,
            args.depth_max,
            args.max_image_points,
            args.image_point_radius,
            args.image_point_alpha,
        )
        left = resize_cover(left, panel_size)
        right = resize_cover(right, panel_size)
    else:
        raise ValueError(f"Unknown render mode: {mode}")

    image = compose_clean_corrupt(left, right, args.gap, parse_bgr(args.gap_color))
    return image, meta


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        require_deps()

    args.clean_root = args.clean_root.expanduser().resolve()
    args.multicorrupt_root = args.multicorrupt_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    corruptions = parse_list(args.corruptions)
    mode_map = parse_mode_map(args.mode_map)
    tables = NuScenesTables(args.clean_root, args.version)

    shared_sample = find_valid_sample(tables, args, mode_map) if args.same_sample else None
    manifest = {
        "clean_root": str(args.clean_root),
        "multicorrupt_root": str(args.multicorrupt_root),
        "version": args.version,
        "severity": str(args.severity),
        "same_sample": bool(args.same_sample),
        "dry_run": bool(args.dry_run),
        "corrupt_metadata_policy": args.corrupt_metadata_policy,
        "corruptions": corruptions,
        "outputs": [],
    }

    if shared_sample is not None:
        print(f"[multicorrupt-examples] shared sample: {shared_sample['token']}")
    else:
        print("[multicorrupt-examples] searching one valid sample per corruption")

    for corruption in corruptions:
        mode = corruption_mode(corruption, mode_map)
        sample = shared_sample or find_valid_sample_for_corruption(tables, corruption, args, mode_map)
        print(
            f"[multicorrupt-examples] {corruption} s{args.severity}: "
            f"mode={mode}, sample={sample['token']}"
        )
        if args.dry_run:
            meta = resolve_pair_metadata(tables, sample, corruption, mode, args)
            meta["output"] = None
        else:
            image, meta = render_pair(tables, sample, corruption, mode, args)
            out_path = args.out_dir / f"{safe_name(corruption)}_s{args.severity}_{mode}.png"
            if not cv2.imwrite(str(out_path), image):
                raise IOError(f"Failed to write image: {out_path}")
            meta["output"] = str(out_path)
        manifest["outputs"].append(meta)
        if meta["warnings"]:
            for warning in meta["warnings"]:
                print(f"[multicorrupt-examples][warn] {corruption}: {warning}")
        TABLE_CACHE.clear()
        gc.collect()

    manifest_path = args.out_dir / "render_multicorrupt_examples_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[multicorrupt-examples] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
