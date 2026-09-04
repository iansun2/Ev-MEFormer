#!/usr/bin/env python3
"""Register the three immutable UMR SGSUP evaluation configs in a manifest."""

import argparse
import ast
import hashlib
import json
import os
import tempfile

from mmcv import Config


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolved_sha256(path):
    payload = json.dumps(
        plain(dict(Config.fromfile(path))), sort_keys=True,
        separators=(",", ":"), default=repr).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def direct_base(path):
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "_base_"
                   for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                return ""
            value = value[0]
        return os.path.realpath(os.path.join(os.path.dirname(path), value))
    return ""


def atomic_write(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(prefix=".umr-runtime-configs-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--profile", choices=("vov_umr", "vov_v10", "r50_v10"),
        default="vov_umr")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--matched-base", required=True)
    parser.add_argument("--raw-control", required=True)
    parser.add_argument("--formal-final", required=True)
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (manifest.get("profile") != args.profile or
            (manifest.get("model_provenance") or {}).get("profile") != args.profile):
        raise SystemExit(
            "[UMR runtime register][ERROR] manifest/profile mismatch")
    base_real = os.path.realpath(args.base_config)
    base = Config.fromfile(args.base_config)
    records = {}
    inputs = (
        ("UMR_MATCHED_BASE",
         {"vov_umr": "matched_vov_umr_action_off",
          "vov_v10": "matched_vov_v10_action_off",
          "r50_v10": "matched_r50_v10_action_off"}[args.profile],
         args.matched_base),
        ("RULE_SGSUP_B_runtime", "unguarded_action_parity_control", args.raw_control),
        ("RULE_SGSUP_B_contextheight055_v9bev_dyn10_30",
         "formal_guarded_final", args.formal_final),
    )
    for name, role, path in inputs:
        if not os.path.isfile(path) or direct_base(path) != base_real:
            raise SystemExit(
                "[UMR runtime register][ERROR] invalid direct base: {}".format(path))
        cfg = Config.fromfile(path)
        ensemble = cfg.model.pts_bbox_head.ensemble
        expected_v10 = args.profile in ("vov_v10", "r50_v10")
        if bool(ensemble.get("v10_enable", False)) != expected_v10:
            raise SystemExit(
                "[UMR runtime register][ERROR] V10 state differs from profile")
        if name == "UMR_MATCHED_BASE":
            if plain(dict(cfg)) != plain(dict(base)):
                raise SystemExit(
                    "[UMR runtime register][ERROR] matched baseline changes base config")
            if bool(ensemble.get("supp_enable", False)):
                raise SystemExit(
                    "[UMR runtime register][ERROR] matched baseline enables supplement")
        else:
            if (ensemble.get("supp_enable") is not True or
                    ensemble.get("supp_mode") != "img_highscore" or
                    int(ensemble.get("supp_max_add_queries") or 0) != 90):
                raise SystemExit(
                    "[UMR runtime register][ERROR] supplement action contract mismatch")
            expected_guard = name.endswith("contextheight055_v9bev_dyn10_30")
            if bool(ensemble.get("supp_contextheight_gate_enable")) != expected_guard:
                raise SystemExit(
                    "[UMR runtime register][ERROR] guard role mismatch: {}".format(name))
        records[name] = {
            "role": role,
            "path": os.path.relpath(os.path.abspath(path), os.getcwd()),
            "sha256": sha256_file(path),
            "resolved_sha256": resolved_sha256(path),
        }
    manifest["runtime_configs"] = records
    atomic_write(args.manifest, manifest)
    print("[UMR runtime register] registered {} configs".format(len(records)))


if __name__ == "__main__":
    main()
