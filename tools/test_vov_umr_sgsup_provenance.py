#!/usr/bin/env python3
"""Unit tests for the VoV SGSUP calibration-lineage closure."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest

from mmcv import Config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import stamp_umr_model_provenance as provenance
from tools import preflight_vov_umr_sgsup_b as portable_preflight
from tools.umr_collector_contract import current_collector_contract


def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


def write_json(path, payload):
    write(path, json.dumps(payload, sort_keys=True) + "\n")
    return path


class VoVSGSUPProvenanceTest(unittest.TestCase):

    def test_portable_preflight_rejects_legacy_source_without_collector_contract(self):
        expected = current_collector_contract(REPO_ROOT)
        valid = {
            "path": "ev_meta.rank000.json",
            "samples_per_gpu": 1,
            "collector_code": expected,
        }
        changed_fingerprint = dict(expected, fingerprint="0" * 64)
        changed_records = dict(expected, records=expected["records"][:-1])
        invalid_cases = {
            "legacy-missing": {
                "path": "ev_meta.rank000.json",
                "samples_per_gpu": 1,
            },
            "spg": dict(valid, samples_per_gpu=2),
            "fingerprint": dict(valid, collector_code=changed_fingerprint),
            "records": dict(valid, collector_code=changed_records),
        }

        for name, record in invalid_cases.items():
            with self.subTest(name=name):
                errors = portable_preflight.source_log_collector_errors(
                    record, expected)
                self.assertTrue(errors)

    def test_portable_preflight_accepts_current_collector_contract(self):
        expected = current_collector_contract(REPO_ROOT)
        record = {
            "path": "ev_meta.rank000.json",
            "samples_per_gpu": 1,
            "collector_code": expected,
        }

        self.assertEqual(
            portable_preflight.source_log_collector_errors(record, expected),
            [])

    def test_stamper_rejects_stale_or_incomplete_collector_metadata(self):
        expected = current_collector_contract(REPO_ROOT)
        valid = {
            "samples_per_gpu": 1,
            "collector_code_format": expected["format"],
            "collector_code_fingerprint": expected["fingerprint"],
            "collector_code_records": expected["records"],
        }
        invalid_cases = {
            "spg": dict(valid, samples_per_gpu=2),
            "missing": {"samples_per_gpu": 1},
            "fingerprint": dict(
                valid, collector_code_fingerprint="0" * 64),
            "records": dict(
                valid, collector_code_records=expected["records"][:-1]),
        }

        for name, extra in invalid_cases.items():
            with self.subTest(name=name), self.assertRaises(SystemExit):
                provenance.validate_collector_meta(
                    extra, expected, "ev_meta.rank000.json")

        provenance.validate_collector_meta(
            valid, expected, "ev_meta.rank000.json")

    def test_canonical_vov_and_r50_profiles_are_distinct(self):
        vov = provenance.config_identity(
            os.path.join(
                REPO_ROOT, "projects/configs/"
                "edl_pme_stage2_bbox_r3c_cls_voxel0075_vov_1600x640_cbgs_umr.py"),
            "vov_umr")
        r50 = provenance.config_identity(
            os.path.join(
                REPO_ROOT, "projects/configs/"
                "edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs_"
                "v10_router_selector_frozen.py"),
            "r50_v10")
        self.assertEqual(vov["backbone_type"], "VoVNet")
        self.assertFalse(vov["v10_enable"])
        self.assertEqual(r50["backbone_type"], "ResNet")
        self.assertTrue(r50["v10_enable"])
        frozen = Config.fromfile(os.path.join(
            REPO_ROOT, "projects/configs/"
            "edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs_"
            "v10_router_selector_frozen.py"))
        expected_thresholds = {
            "tau_margin": 1.588853454589843,
            "margin": 0.1,
            "tau_dmg": 0.18098025321960445,
            "T_oracle": 1.7976929346720378,
        }
        self.assertEqual(
            dict(frozen.model.pts_bbox_head.v10_oracle_thresholds),
            expected_thresholds)
        self.assertEqual(
            dict(frozen.model.pts_bbox_head.ensemble.v10_oracle_thresholds),
            expected_thresholds)
        self.assertEqual(
            frozen.model.pts_bbox_head.v10_matching_mode, "single_best")

    def test_profile_cross_use_is_rejected(self):
        r50_config = os.path.join(
            REPO_ROOT, "projects/configs/"
            "edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs_"
            "v10_router_selector_frozen.py")
        with self.assertRaises(SystemExit):
            provenance.config_identity(r50_config, "vov_umr")

    def test_r50_runtime_registry_accepts_v10_profile(self):
        root = self.temporary.name
        base = os.path.join(
            REPO_ROOT, "projects/configs/"
            "edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs_"
            "v10_router_selector_frozen.py")
        relative_base = os.path.relpath(base, root)
        matched = write(
            os.path.join(root, "UMR_MATCHED_BASE.py"),
            "_base_ = {!r}\n".format(relative_base))
        raw = write(
            os.path.join(root, "RULE_SGSUP_B_runtime.py"),
            "_base_ = {!r}\n"
            "model = dict(pts_bbox_head=dict(ensemble=dict("
            "supp_enable=True, supp_mode='img_highscore', "
            "supp_max_add_queries=90, "
            "supp_contextheight_gate_enable=False)))\n".format(relative_base))
        final = write(
            os.path.join(
                root, "RULE_SGSUP_B_contextheight055_v9bev_dyn10_30.py"),
            "_base_ = {!r}\n"
            "model = dict(pts_bbox_head=dict(ensemble=dict("
            "supp_enable=True, supp_mode='img_highscore', "
            "supp_max_add_queries=90, "
            "supp_contextheight_gate_enable=True)))\n".format(relative_base))
        manifest = write_json(os.path.join(root, "manifest.json"), {
            "profile": "r50_v10",
            "model_provenance": {"profile": "r50_v10"},
        })
        subprocess.run([
            sys.executable,
            os.path.join(
                REPO_ROOT, "tools/register_vov_umr_sgsup_runtime_configs.py"),
            "--profile", "r50_v10",
            "--manifest", manifest,
            "--base-config", base,
            "--matched-base", matched,
            "--raw-control", raw,
            "--formal-final", final,
        ], check=True, cwd=REPO_ROOT)
        with open(manifest, "r", encoding="utf-8") as handle:
            records = json.load(handle)["runtime_configs"]
        self.assertEqual(
            records["UMR_MATCHED_BASE"]["role"],
            "matched_r50_v10_action_off")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = self.temporary.name
        self.log_root = os.path.join(root, "ev_logs")
        self.train_infos = write(os.path.join(root, "train_infos.pkl"), "train")
        self.branch_source = write(
            os.path.join(self.log_root, "train_oc", "ev_branch_candidates.rank000.jsonl"),
            "{}\n")
        self.clean_scene = write(
            os.path.join(self.log_root, "train_clean", "ev_scene_summary.rank000.jsonl"),
            "{}\n")
        self.oc_scene = write(
            os.path.join(self.log_root, "train_oc", "ev_scene_summary.rank000.jsonl"),
            "{}\n")
        self.clean_prediction = write(
            os.path.join(self.log_root, "train_clean", "ev_prediction_light.rank000.jsonl"),
            "{}\n")

        calib = os.path.join(root, "calibration")
        self.branch_labels = write(
            os.path.join(calib, "branch", "branch_candidate_labels.jsonl"),
            '{"features":{},"targets":{"candidate_useful":1}}\n')
        branch_content_sha = provenance.jsonl_content_sha256(self.branch_labels)
        self.branch_summary = write_json(
            os.path.join(calib, "branch", "branch_candidate_labels_summary.json"), {
                "formal_calibration_safe": True,
                "debug_only": False,
                "expected_split": "train_oc",
                "infos": self.train_infos,
                "branch_log_root": os.path.join(self.log_root, "train_oc"),
                "input_logs": [self.branch_source],
                "input_log_sha256": {
                    self.branch_source: provenance.sha256_file(self.branch_source)},
                "output_labels": self.branch_labels,
                "output_labels_sha256": provenance.sha256_file(self.branch_labels),
                "output_labels_content_sha256": branch_content_sha,
                "output_label_records": 1,
                "counts": {"labels_written": 1},
            })
        self.scene_labels = write(
            os.path.join(calib, "scene", "scene_gate_labels.jsonl"), "{}\n")
        self.scene_summary = write_json(
            os.path.join(calib, "scene", "scene_gate_labels_summary.json"), {
                "scene_log": self.log_root,
                "log_paths": [self.clean_scene, self.oc_scene],
                "log_path_sha256": {
                    self.clean_scene: provenance.sha256_file(self.clean_scene),
                    self.oc_scene: provenance.sha256_file(self.oc_scene),
                },
                "output_labels": self.scene_labels,
                "output_labels_sha256": provenance.sha256_file(self.scene_labels),
                "written": 1,
                "formal_calibration_safe": True,
                "debug_only": False,
                "label_policy": {
                    "beams_s1": {"target": 0.0, "weight": 1.0},
                    "beams_s2": {"target": 0.35, "weight": 0.45},
                    "beams_s3": {"target": 1.0, "weight": 1.0},
                    "clean": {"target": 0.0, "weight": 1.5},
                    "fog_motionblur": {"target": 0.0, "weight": 2.0},
                    "other_corruptions": "negative",
                },
            })

        self.scene_mapper_path = os.path.join(
            calib, "scene", "mapper_scene_bev_damage_applicable.json")
        self.utility_mapper_path = os.path.join(
            calib, "action", "mapper_candidate_useful.json")
        self.reference_path = os.path.join(calib, "clean_reference_stats.json")
        self.scene_mapper = {
            "target": "scene_bev_damage_applicable",
            "features": provenance.EXPECTED_SCENE_FEATURES,
            "normalization": {
                "mean": [0.0] * len(provenance.EXPECTED_SCENE_FEATURES),
                "std": [1.0] * len(provenance.EXPECTED_SCENE_FEATURES)},
            "model": {"weights": [0.0] * len(provenance.EXPECTED_SCENE_FEATURES),
                      "bias": 0.0},
            "thresholds": {"clean_fpr_005": 0.5},
            "formal_calibration_safe": True, "debug_only": False,
            "unsafe_features": [],
            "label_summary_path": self.scene_summary,
            "labels_path": self.scene_labels,
            "labels_sha256": provenance.sha256_file(self.scene_labels),
            "label_summary_sha256": provenance.sha256_file(self.scene_summary),
        }
        self.utility_mapper = {
            "target": "candidate_useful", "feature_preset": "action_branch",
            "features": provenance.EXPECTED_UTILITY_FEATURES,
            "normalization": {
                "mean": [0.0] * len(provenance.EXPECTED_UTILITY_FEATURES),
                "std": [1.0] * len(provenance.EXPECTED_UTILITY_FEATURES)},
            "model": {"weights": [0.0] * len(provenance.EXPECTED_UTILITY_FEATURES),
                      "bias": 0.0},
            "formal_calibration_safe": True, "debug_only": False,
            "unsafe_features": [],
            "label_provenance": {
                "formal_source_kind": "train_or_train_oc",
                "labels_realpath": self.branch_labels,
                "labels_content_sha256": branch_content_sha,
            },
        }
        self.reference = {
            "format": "umr_clean_reference_stats_v1", "records": 1,
            "source_files": [self.clean_prediction],
            "source_file_sha256": {
                self.clean_prediction: provenance.sha256_file(self.clean_prediction)},
            "scene_global": {
                "bev_top10_unc": {"count": 1, "p99": 0.1},
                "img_top10_unc": {"count": 1, "p99": 0.1},
            },
        }
        self.payloads = [
            (self.scene_mapper_path, self.scene_mapper),
            (self.utility_mapper_path, self.utility_mapper),
            (self.reference_path, self.reference),
        ]
        self.args = argparse.Namespace(
            branch_labels=self.branch_labels,
            branch_label_summary=self.branch_summary,
            scene_labels=self.scene_labels,
            scene_label_summary=self.scene_summary,
            train_infos=self.train_infos,
            source_log_root=self.log_root,
        )
        self.log_meta = [
            {"formal_split": "train_clean", "source_files": {
                "scene_summary_path": {
                    "path": self.clean_scene,
                    "sha256": provenance.sha256_file(self.clean_scene)},
                "prediction_light_path": {
                    "path": self.clean_prediction,
                    "sha256": provenance.sha256_file(self.clean_prediction)},
            }},
            {"formal_split": "train_oc", "source_files": {
                "scene_summary_path": {
                    "path": self.oc_scene,
                    "sha256": provenance.sha256_file(self.oc_scene)},
                "branch_candidates_path": {
                    "path": self.branch_source,
                    "sha256": provenance.sha256_file(self.branch_source)},
            }},
        ]

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_current_source_chain_passes(self):
        for path, payload in self.payloads:
            provenance.validate_artifact(path, payload)
        result = provenance.validate_calibration_chain(
            self.args, self.payloads, self.log_meta)
        self.assertEqual(set(result), {
            "branch_labels", "branch_label_summary",
            "scene_labels", "scene_label_summary"})

    def test_copied_or_modified_branch_labels_fail(self):
        write(self.branch_labels, "{}\n")
        with self.assertRaises(SystemExit):
            provenance.validate_calibration_chain(
                self.args, self.payloads, self.log_meta)

    def test_empty_scene_labels_fail_even_if_hashes_are_rewritten(self):
        write(self.scene_labels, "")
        with open(self.scene_summary, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary["output_labels_sha256"] = provenance.sha256_file(self.scene_labels)
        summary["written"] = 0
        write_json(self.scene_summary, summary)
        self.scene_mapper["labels_sha256"] = provenance.sha256_file(self.scene_labels)
        self.scene_mapper["label_summary_sha256"] = provenance.sha256_file(
            self.scene_summary)
        with self.assertRaises(SystemExit):
            provenance.validate_calibration_chain(
                self.args, self.payloads, self.log_meta)

    def test_nonfinite_mapper_is_rejected(self):
        self.scene_mapper["model"]["weights"][0] = float("nan")
        with self.assertRaises(SystemExit):
            provenance.validate_artifact(
                self.scene_mapper_path, self.scene_mapper)


if __name__ == "__main__":
    unittest.main()
