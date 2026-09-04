#!/usr/bin/env bash
# Start the Ev-MEFormer development container.
#
# The whole repository root is mounted at /Ev-MEFormer, because configs declare
# `plugin_dir = 'projects/mmdet3d_plugin/'` and tools/ resolve `projects/configs/...`
# relative to the repository root. Mounting a subdirectory breaks both.
#
# Dataset directories are supplied through environment variables; there are no
# defaults, because the right paths differ per machine. Each one is bind-mounted
# read-only at the SAME path inside the container: if data/nuscenes/ contains
# symlinks into a shared copy, those links store absolute host paths and dangle
# unless the source resolves identically on both sides.
#
# Usage:
#   NUSCENES_RAW=/path/to/nuscenes \
#   NUSCENES_META=/path/to/nuscenes_metadata \
#   MULTICORRUPT_ROOT=/path/to/MultiCorrupt \
#     bash scripts/start.sh
#
# Keeping those paths in an untracked wrapper (scripts/<name>.local.sh, which is
# gitignored) saves retyping them.
#
#   NUSCENES_RAW        samples, sweeps, maps, v1.0-trainval
#   NUSCENES_META       infos / dbinfos / gt_database from tools/create_data.sh
#   MULTICORRUPT_ROOT   MultiCorrupt validation set, needed only for corruption runs
#   WORK_DIRS           optional: put work_dirs/ on another disk; it is mounted at
#                       the same in-container path, so paths recorded in
#                       chain_manifest.json and runtime packs stay valid
#
# Unset variables are skipped. If data/nuscenes/ holds real files rather than
# symlinks to a shared copy, none of them are needed.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mounts=(-v "${REPO_ROOT}":/Ev-MEFormer)
for var in NUSCENES_RAW NUSCENES_META MULTICORRUPT_ROOT; do
  src="${!var:-}"
  [[ -n "$src" ]] || continue
  if [[ -d "$src" ]]; then
    mounts+=(-v "${src}:${src}:ro")
  else
    echo "[warn] $var is set but not a directory, skipping: $src" >&2
  fi
done
if [[ -n "${WORK_DIRS:-}" ]]; then
  mkdir -p "$WORK_DIRS"
  mounts+=(-v "${WORK_DIRS}:/Ev-MEFormer/work_dirs")
fi

docker run -it \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  "${mounts[@]}" \
  -w /Ev-MEFormer \
  --name ev-meformer \
  ev-meformer bash
