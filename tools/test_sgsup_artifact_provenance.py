#!/usr/bin/env python3
"""Focused fail-closed tests for SGSUP runtime artifact provenance."""

import json
import inspect
import os
import sys
import tempfile
import types
import unittest

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from projects.mmdet3d_plugin.models.utils.edl_pme_transformer import EDLPMETransformer
from tools.build_umr_scene_gate_labels import DEFAULT_SCENE_FEATURES


FINGERPRINT = "a" * 64


def write_json(directory, name, payload):
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def scene_payload(fingerprint=FINGERPRINT):
    return {
        "target": "scene_bev_damage_applicable",
        "features": ["scene_v8_bev_damage_mean"],
        "model": {"weights": [1.0], "bias": 0.0},
        "normalization": {"mean": [0.0], "std": [1.0]},
        "formal_calibration_safe": True,
        "debug_only": False,
        "label_summary_path": "work_dirs/train/scene_summary.json",
        "label_summary_missing": False,
        "feature_missing_rates": {},
        "bad_missing_features": {},
        "unsafe_features": [],
        "model_provenance": {
            "fingerprint": fingerprint,
            "formal_splits": ["train_clean", "train_oc"],
            "source_log_root": "work_dirs/train/ev_logs",
            "train_infos": "data/nuscenes/nuscenes_infos_train.pkl",
        },
    }


def utility_payload(fingerprint=FINGERPRINT):
    return {
        "target": "candidate_useful",
        "features": ["score"],
        "model": {"weights": [1.0], "bias": 0.0},
        "normalization": {"mean": [0.0], "std": [1.0]},
        "formal_calibration_safe": True,
        "debug_only": False,
        "unsafe_features": [],
        "forbidden_input_markers": [],
        "label_provenance": {
            "formal_source_kind": "train_or_train_oc",
            "labels_content_sha256": "b" * 64,
            "forbidden_input_markers": [],
            "labels_path": "work_dirs/train/labels.jsonl",
        },
        "model_provenance": {"fingerprint": fingerprint},
    }


class SGSUPArtifactProvenanceTest(unittest.TestCase):

    def setUp(self):
        self.transformer = object.__new__(EDLPMETransformer)
        self.transformer.supp_expected_model_fingerprint = FINGERPRINT

    def test_scene_mapper_accepts_matching_model_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_json(directory, "scene.json", scene_payload())
            mapper = self.transformer._load_supp_scene_mapper(path)
        self.assertIsNotNone(mapper)
        self.assertEqual(mapper["target"], "scene_bev_damage_applicable")

    def test_scene_mapper_rejects_wrong_target_or_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            wrong_target = scene_payload()
            wrong_target["target"] = "corruption_classifier"
            target_path = write_json(directory, "scene-target.json", wrong_target)
            fingerprint_path = write_json(
                directory, "scene-fingerprint.json", scene_payload("c" * 64))
            self.assertIsNone(
                self.transformer._load_supp_scene_mapper(target_path))
            self.assertIsNone(
                self.transformer._load_supp_scene_mapper(fingerprint_path))

    def test_utility_mapper_rejects_model_fingerprint_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_json(
                directory, "utility.json", utility_payload("c" * 64))
            mapper = self.transformer._load_supp_utility_mapper(
                path, expected_target="candidate_useful")
        self.assertIsNone(mapper)

    def test_scene_mapper_missing_runtime_feature_fails_closed(self):
        self.transformer.supp_scene_mapper = {
            "features": ["scene_v8_bev_damage_mean"],
            "weights": [1.0],
            "bias": 10.0,
            "mean": [0.0],
            "std": [1.0],
        }
        self.transformer._supp_scene_missing_features_warned = False
        score = self.transformer._supp_scene_mapper_score(
            {}, {}, torch.zeros((2, 4), dtype=torch.float32))
        self.assertIsNone(score)

    def test_q95_alias_is_per_scene_and_available_before_scalar_telemetry(self):
        reference = torch.zeros((2, 4), dtype=torch.float32)
        raw = torch.tensor([
            [[0.10], [0.20], [0.30], [0.40]],
            [[0.60], [0.70], [0.80], [0.90]],
        ])
        value, available = self.transformer._scene_feature_tensor(
            "scene_v9_fused_keep_unit_q95",
            {
                "v9_fused_keep_unit": raw,
                # A late scalar with this name must never shadow the raw
                # admission-time query tensor.
                "v9_fused_keep_unit_q95": torch.tensor([9.0, 9.0]),
            }, {}, reference,
            return_available=True)

        self.assertTrue(available)
        torch.testing.assert_close(
            value, torch.quantile(raw.reshape(2, -1), 0.95, dim=1))

        changed = raw.clone()
        changed[1] = 0.0
        changed_value, changed_available = self.transformer._scene_feature_tensor(
            "scene_v9_fused_keep_unit_q95",
            {"v9_fused_keep_unit": changed}, {}, reference,
            return_available=True)
        self.assertTrue(changed_available)
        torch.testing.assert_close(changed_value[0], value[0])
        self.assertNotEqual(float(changed_value[1]), float(value[1]))

    def test_q95_alias_rejects_silent_source_substitutions(self):
        reference = torch.zeros((2, 4), dtype=torch.float32)
        invalid_sources = (
            torch.tensor([0.2, 0.8]),
            torch.zeros((3, 4, 1)),
            torch.empty((2, 0, 1)),
            torch.tensor([[[0.1], [float("nan")]], [[0.2], [0.3]]]),
            torch.tensor([[[0.1], [float("inf")]], [[0.2], [0.3]]]),
            0.5,
        )
        for source in invalid_sources:
            with self.subTest(source_type=type(source).__name__):
                value, available = self.transformer._scene_feature_tensor(
                    "scene_v9_fused_keep_unit_q95",
                    {"v9_fused_keep_unit": source}, {}, reference,
                    return_available=True)
                self.assertFalse(available)
                torch.testing.assert_close(value, torch.zeros(2))

    @staticmethod
    def _complete_scene_runtime(batch_size=2, num_queries=4):
        blended = {
            name[len("scene_"):]: torch.linspace(
                0.1, 0.2, batch_size, dtype=torch.float32)
            for name in DEFAULT_SCENE_FEATURES
        }
        blended["selector_fused_weight_mean"] = torch.linspace(
            0.8, 0.9, batch_size, dtype=torch.float32)
        blended["v8_output_delta_center"] = torch.arange(
            batch_size * num_queries, dtype=torch.float32,
        ).reshape(batch_size, num_queries, 1) / 10.0
        blended["v8_high_damage_query"] = torch.linspace(
            0.0, 1.0, batch_size * num_queries,
            dtype=torch.float32,
        ).reshape(batch_size, num_queries, 1)
        blended["v9_fused_keep_unit"] = torch.linspace(
            0.05, 0.95, batch_size * num_queries,
            dtype=torch.float32,
        ).reshape(batch_size, num_queries, 1)
        return blended

    def test_exact_ordered_18_feature_mapper_contract_is_ready(self):
        batch_size = 2
        reference = torch.zeros((batch_size, 4), dtype=torch.float32)
        self.transformer.supp_scene_mapper = {
            "features": list(DEFAULT_SCENE_FEATURES),
            "weights": [0.0] * len(DEFAULT_SCENE_FEATURES),
            "bias": 0.0,
            "mean": [0.0] * len(DEFAULT_SCENE_FEATURES),
            "std": [1.0] * len(DEFAULT_SCENE_FEATURES),
        }
        self.transformer._supp_scene_missing_features_warned = False
        score, diagnostics = self.transformer._supp_scene_mapper_score(
            self._complete_scene_runtime(batch_size), {}, reference,
            return_diagnostics=True)

        self.assertIsNotNone(score)
        torch.testing.assert_close(score, torch.full((batch_size,), 0.5))
        self.assertEqual(diagnostics["expected_count"], 18)
        self.assertEqual(diagnostics["available_count"], 18)
        self.assertEqual(diagnostics["finite_count"], 18)
        self.assertEqual(diagnostics["missing_features"], [])
        self.assertTrue(diagnostics["contract_ready"])
        self.assertTrue(diagnostics["q95_available"])
        expected_q95 = torch.quantile(
            self._complete_scene_runtime(batch_size)[
                "v9_fused_keep_unit"].reshape(batch_size, -1),
            0.95, dim=1)
        torch.testing.assert_close(
            diagnostics["q95_runtime"], expected_q95)

    def test_complete_mapper_fails_closed_when_q95_source_is_missing(self):
        reference = torch.zeros((2, 4), dtype=torch.float32)
        self.transformer.supp_scene_mapper = {
            "features": list(DEFAULT_SCENE_FEATURES),
            "weights": [0.0] * len(DEFAULT_SCENE_FEATURES),
            "bias": 10.0,
            "mean": [0.0] * len(DEFAULT_SCENE_FEATURES),
            "std": [1.0] * len(DEFAULT_SCENE_FEATURES),
        }
        self.transformer._supp_scene_missing_features_warned = False
        runtime = self._complete_scene_runtime()
        del runtime["v9_fused_keep_unit"]
        score, diagnostics = self.transformer._supp_scene_mapper_score(
            runtime, {}, reference, return_diagnostics=True)

        self.assertIsNone(score)
        self.assertFalse(diagnostics["contract_ready"])
        self.assertEqual(
            diagnostics["missing_features"],
            ["scene_v9_fused_keep_unit_q95"])

    def test_mapper_output_contract_rejects_silent_invalid_scores(self):
        reference = torch.zeros((2, 4), dtype=torch.float32)
        self.transformer.supp_scene_mapper = {
            "features": ["scene_v8_bev_damage_mean"],
            "weights": [0.0], "bias": 0.0,
            "mean": [0.0], "std": [1.0],
        }
        invalid_scores = (
            torch.tensor(0.5),
            torch.tensor([0.5]),
            torch.tensor([0.5, 0.5, 0.5]),
            torch.empty((0,)),
            torch.tensor([0.5, float("nan")]),
            torch.tensor([0.5, float("inf")]),
            0.5,
        )
        original = self.transformer._mapper_score_from_values
        try:
            for invalid in invalid_scores:
                with self.subTest(invalid_type=type(invalid).__name__):
                    self.transformer._supp_scene_missing_features_warned = False
                    self.transformer._mapper_score_from_values = (
                        lambda _mapper, _values, _reference,
                        invalid=invalid: invalid)
                    score, diagnostics = self.transformer._supp_scene_mapper_score(
                        {"v8_bev_damage_mean": torch.tensor([0.1, 0.2])},
                        {}, reference, return_diagnostics=True)
                    self.assertIsNone(score)
                    self.assertFalse(diagnostics["contract_ready"])
        finally:
            self.transformer._mapper_score_from_values = original

    @staticmethod
    def _admission_transformer():
        transformer = object.__new__(EDLPMETransformer)
        signature = inspect.signature(EDLPMETransformer.__init__)
        for name, parameter in signature.parameters.items():
            if name.startswith(("supp_", "scrg_")):
                setattr(transformer, name, parameter.default)
        transformer.supp_require_reference_stats = False
        transformer.supp_scene_mapper = {
            "features": list(DEFAULT_SCENE_FEATURES),
            "weights": [0.0] * len(DEFAULT_SCENE_FEATURES),
            "bias": 10.0,
            "mean": [0.0] * len(DEFAULT_SCENE_FEATURES),
            "std": [1.0] * len(DEFAULT_SCENE_FEATURES),
        }
        transformer.supp_utility_mapper = None
        transformer.supp_img_utility_mapper = None
        transformer.supp_bev_utility_mapper = None
        transformer.supp_fused_slot_utility_mapper = None
        transformer.supp_scene_mapper_threshold = 0.5
        transformer.supp_max_add_queries = 1
        transformer.supp_scene_budget_min_ratio = 0.0
        transformer.supp_scene_budget_max_ratio = 0.0
        transformer.supp_img_score_threshold = 0.02
        transformer.supp_fused_replace_score_ceiling = 0.35
        transformer.supp_nms_center_dist = 0.0
        transformer.supp_per_class_cap = 0
        transformer.supp_img_damage_guard_enable = False
        transformer.supp_contextheight_gate_enable = False
        transformer.supp_qsem_gate_enable = False
        transformer._supp_scene_missing_features_warned = False

        def scene_gate(_self, blended_outs, _v7_info):
            batch_size = blended_outs["cls_logits"].shape[1]
            reference = blended_outs["cls_logits"].new_zeros((batch_size,))
            return (
                torch.ones(batch_size, dtype=torch.bool),
                reference, reference, reference,
                float("nan"), float("nan"), float("nan"),
            )

        def head_weights(
                _self, _branch_name, _modalities, _v8_info,
                _blended_outs, _reference_tensor):
            return {}, []

        transformer._compute_scrg_scene_gate = types.MethodType(
            scene_gate, transformer)
        transformer._supp_runtime_branch_head_weights = types.MethodType(
            head_weights, transformer)
        return transformer

    @classmethod
    def _admission_inputs(cls, include_q95_source=True):
        batch_size = 1
        num_queries = 2
        num_classes = 2
        fused_cls = torch.full(
            (1, batch_size, num_queries, num_classes), -5.0)
        bev_cls = torch.full_like(fused_cls, -2.0)
        img_cls = torch.full_like(fused_cls, 5.0)
        fused_center = torch.zeros((1, batch_size, num_queries, 3))
        bev_center = torch.full_like(fused_center, 20.0)
        img_center = torch.full_like(fused_center, 10.0)
        blended = cls._complete_scene_runtime(batch_size, num_queries)
        blended.update({
            "cls_logits": fused_cls.clone(),
            "center": fused_center.clone(),
        })
        if not include_q95_source:
            del blended["v9_fused_keep_unit"]
        source_outs = {
            "cls_logits": torch.cat(
                [fused_cls, bev_cls, img_cls], dim=2),
            "center": torch.cat(
                [fused_center, bev_center, img_center], dim=2),
        }
        return blended, source_outs

    def test_full_admission_fails_closed_without_q95_source(self):
        transformer = self._admission_transformer()
        blended, source_outs = self._admission_inputs(
            include_q95_source=False)
        original_cls = blended["cls_logits"].clone()

        result = transformer._apply_img_supplement_admission(
            blended, source_outs, {}, ["fused", "bev", "img"],
            [2, 2, 2], v8_info={})

        self.assertEqual(result["supp_scene_mapper_contract_ready"], 0.0)
        self.assertEqual(result["supp_scene_mapper_q95_available"], 0.0)
        self.assertEqual(result["supp_added_ratio"], 0.0)
        self.assertEqual(result["supp_scene_gate_ratio"], 0.0)
        self.assertFalse(bool(result["supp_query_added"].any().item()))
        torch.testing.assert_close(result["cls_logits"], original_cls)

    def test_full_admission_can_add_with_complete_18_feature_contract(self):
        transformer = self._admission_transformer()
        blended, source_outs = self._admission_inputs(
            include_q95_source=True)
        original_cls = blended["cls_logits"].clone()

        result = transformer._apply_img_supplement_admission(
            blended, source_outs, {}, ["fused", "bev", "img"],
            [2, 2, 2], v8_info={})

        self.assertEqual(result["supp_scene_mapper_contract_ready"], 1.0)
        self.assertEqual(result["supp_scene_mapper_q95_available"], 1.0)
        self.assertGreater(result["supp_added_ratio"], 0.0)
        self.assertTrue(bool(result["supp_query_added"].any().item()))
        self.assertFalse(torch.equal(result["cls_logits"], original_cls))

    def test_reference_stats_fingerprint_mismatch_raises_when_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_json(
                directory,
                "clean_reference_stats.json",
                {
                    "format": "umr_clean_reference_stats_v1",
                    "model_provenance": {"fingerprint": "c" * 64},
                })
            self.transformer.v7_reference_stats_path = path
            self.transformer._v7_reference_stats_cache = None
            self.transformer._v7_reference_stats_cache_path = None
            self.transformer.supp_require_reference_stats = True
            with self.assertRaises(RuntimeError):
                self.transformer._load_v7_reference_stats(require_stats=True)


if __name__ == "__main__":
    unittest.main()
