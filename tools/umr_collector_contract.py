#!/usr/bin/env python3
"""Canonical code fingerprint for formal UMR sidecar collection.

The runtime mappers are only reproducible when their train/train-O.C source
logs were produced by the same feature, logging, corruption, and collection
code.  Both ``tools/test.py`` and the provenance stamper import this module so
the collection-time and stamp-time contracts cannot drift independently.
"""

import hashlib
import json
import os


COLLECTOR_CODE_PATHS = (
    "projects/mmdet3d_plugin/models/utils/edl_pme_transformer.py",
    "projects/mmdet3d_plugin/models/utils/edl_uncertainty_logger.py",
    "projects/mmdet3d_plugin/models/dense_heads/edl_head.py",
    "projects/mmdet3d_plugin/datasets/pipelines/synthetic_corruptions.py",
    "tools/dist_test.sh",
    "tools/test.py",
    "tools/make_umr_phase1_log_configs.py",
    "tools/umr_utility_collect_phase1_full.sh",
    "tools/umr_collector_contract.py",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def collector_code_records(repo_root):
    records = []
    for relative_path in COLLECTOR_CODE_PATHS:
        path = os.path.join(repo_root, relative_path)
        if not os.path.isfile(path):
            raise RuntimeError(
                "formal UMR collector code is missing: {}".format(
                    relative_path))
        records.append({
            "path": relative_path,
            "sha256": sha256_file(path),
        })
    return records


def collector_code_fingerprint(records):
    canonical = json.dumps(
        list(records), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def current_collector_contract(repo_root):
    records = collector_code_records(repo_root)
    return {
        "format": "umr_collector_code_v1",
        "records": records,
        "fingerprint": collector_code_fingerprint(records),
    }
