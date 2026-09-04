#!/usr/bin/env python3
"""Tests for the SGSUP admission-time sidecar audit."""

import json
import os
import sys
import tempfile
import unittest

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.check_sgsup_admission_contract import audit_paths  # noqa: E402
from projects.mmdet3d_plugin.models.utils.edl_uncertainty_logger import (  # noqa: E402
    EDLUncertaintyLogger,
)


def valid_record():
    return {
        "sample_token": "sample-1",
        "ev_model_label": (
            "UMR_RULE_SGSUP_B_contextheight055_v9bev_dyn10_30"),
        "scene_supp_scene_mapper_configured": 1.0,
        "scene_supp_scene_mapper_feature_expected_count": 18.0,
        "scene_supp_scene_mapper_feature_available_count": 18.0,
        "scene_supp_scene_mapper_feature_finite_count": 18.0,
        "scene_supp_scene_mapper_feature_missing_count": 0.0,
        "scene_supp_scene_mapper_contract_ready": 1.0,
        "scene_supp_scene_mapper_q95_available": 1.0,
        "scene_supp_scene_mapper_q95_runtime_mean": 0.42,
        "scene_supp_scene_mapper_score_mean": 0.95,
        "scene_supp_scene_gate_pre_guard_ratio": 1.0,
        "scene_supp_scene_gate_post_guard_ratio": 0.0,
        "scene_supp_added_ratio": 0.0,
        "scene_supp_noadd_reason_code": 15.0,
    }


class SGSUPAdmissionContractTest(unittest.TestCase):

    @staticmethod
    def write_record(directory, record):
        path = os.path.join(directory, "scene.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return path

    def test_valid_contract_passes_without_gate_rate_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = audit_paths([
                self.write_record(directory, valid_record())])

        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["records"], 1)
        self.assertEqual(summary["mean_scene_gate_post_guard_ratio"], 0.0)

    def test_missing_q95_fails(self):
        record = valid_record()
        record["scene_supp_scene_mapper_feature_available_count"] = 17.0
        record["scene_supp_scene_mapper_feature_finite_count"] = 17.0
        record["scene_supp_scene_mapper_feature_missing_count"] = 1.0
        record["scene_supp_scene_mapper_contract_ready"] = 0.0
        record["scene_supp_scene_mapper_q95_available"] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            summary = audit_paths([self.write_record(directory, record)])

        self.assertEqual(summary["status"], "FAIL")
        self.assertTrue(any("available=17.0" in item
                            for item in summary["errors"]))

    def test_empty_malformed_missing_nonfinite_and_mixed_inputs_fail(self):
        cases = {}
        cases["empty"] = ""
        cases["malformed"] = "{not-json}\n"
        missing = valid_record()
        del missing["scene_supp_scene_mapper_score_mean"]
        cases["missing"] = json.dumps(missing) + "\n"
        nonfinite = valid_record()
        nonfinite["scene_supp_scene_mapper_score_mean"] = float("nan")
        cases["nonfinite"] = json.dumps(nonfinite) + "\n"
        invalid = valid_record()
        invalid["sample_token"] = "sample-2"
        invalid["scene_supp_scene_mapper_feature_available_count"] = 17.0
        cases["mixed"] = (
            json.dumps(valid_record()) + "\n" + json.dumps(invalid) + "\n")

        for name, payload in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "scene.jsonl")
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                self.assertEqual(audit_paths([path])["status"], "FAIL")

    def test_coverage_and_candidate_identity_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_record(directory, valid_record())
            insufficient = audit_paths([path], min_records=2)
            wrong_label = audit_paths(
                [path], expected_model_label="UMR_WRONG_CANDIDATE")

        self.assertEqual(insufficient["status"], "FAIL")
        self.assertEqual(wrong_label["status"], "FAIL")

    def test_supplement_off_and_positive_action_modes(self):
        off_record = {
            "sample_token": "sample-off",
            "ev_model_label": "UMR_TRAIN_UMR_MATCHED_BASE",
        }
        with tempfile.TemporaryDirectory() as directory:
            off_path = self.write_record(directory, off_record)
            off_summary = audit_paths(
                [off_path], expect_supplement_off=True,
                expected_model_label="UMR_TRAIN_UMR_MATCHED_BASE")
        self.assertEqual(off_summary["status"], "PASS")

        no_action = valid_record()
        no_action["scene_supp_added_ratio"] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_record(directory, no_action)
            summary = audit_paths([path], require_positive_action=True)
        self.assertEqual(summary["status"], "FAIL")

        action = valid_record()
        action["scene_supp_added_ratio"] = 0.1
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_record(directory, action)
            summary = audit_paths([path], require_positive_action=True)
        self.assertEqual(summary["status"], "PASS")

    def test_logger_publishes_admission_contract_to_scene_sidecar(self):
        record = valid_record()
        edl_info = {
            key[len("scene_"):]: value
            for key, value in record.items()
        }
        with tempfile.TemporaryDirectory() as directory:
            logger = EDLUncertaintyLogger(
                directory, ["car"], keep_full_json=False,
                prediction_light=False)
            logger.log_sample(
                "sample", torch.empty((0, 9)), torch.empty((0,)),
                torch.empty((0,), dtype=torch.long), edl_info)
            logger.save()
            summary = audit_paths([
                os.path.join(directory, "edl_scene_summary.jsonl")])

        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["records"], 1)


if __name__ == "__main__":
    unittest.main()
