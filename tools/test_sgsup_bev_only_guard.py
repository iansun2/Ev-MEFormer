#!/usr/bin/env python3
"""Focused tests for the SGSUP-B pure-BEV abstain guard."""

import inspect
import types
import unittest

import torch

from projects.mmdet3d_plugin.models.utils.edl_pme_transformer import EDLPMETransformer


class SGSUPBEVOnlyGuardTest(unittest.TestCase):

    def setUp(self):
        self.scene_gate = torch.tensor([True, True, True, False])
        self.bev_damage = torch.tensor([True, True, False, True])
        self.img_damage = torch.tensor([False, True, False, False])
        self.fused_damage = torch.tensor([False, False, True, False])
        self.bev_fused_dominance = torch.tensor([True, False, True, True])
        self.bev_upper_damage = torch.tensor([False, False, False, True])
        self.dual_abstain = torch.tensor([False, True, False, False])
        self.img_abstain = torch.tensor([False, False, True, False])

    def apply(self, enabled=True, mode=3, img_available=True, bev_available=True,
              fused_available=True, dominance_available=True):
        return EDLPMETransformer._supp_apply_scene_damage_guard(
            self.scene_gate,
            enabled,
            mode,
            self.img_damage,
            self.bev_damage,
            img_available,
            bev_available,
            fused_damage_gate=self.fused_damage,
            bev_fused_dominance_gate=self.bev_fused_dominance,
            bev_damage_upper_gate=self.bev_upper_damage,
            dual_abstain_gate=self.dual_abstain,
            img_abstain_gate=self.img_abstain,
            fused_threshold_available=fused_available,
            dominance_reference_available=dominance_available,
            bev_upper_threshold_available=bev_available,
            dual_abstain_available=True,
            img_abstain_available=True)

    def test_bev_only_mode_accepts_only_pure_bev_damage(self):
        torch.testing.assert_close(
            self.apply(), torch.tensor([True, False, False, False]))

    def test_bev_only_mode_fails_closed_without_either_threshold(self):
        torch.testing.assert_close(
            self.apply(img_available=False), torch.zeros(4, dtype=torch.bool))
        torch.testing.assert_close(
            self.apply(bev_available=False), torch.zeros(4, dtype=torch.bool))

    def test_disabled_guard_preserves_original_scene_gate(self):
        torch.testing.assert_close(self.apply(enabled=False), self.scene_gate)

    def test_existing_guard_modes_keep_their_contract(self):
        torch.testing.assert_close(
            self.apply(mode=1), torch.tensor([True, False, True, False]))
        torch.testing.assert_close(self.apply(mode=2), self.scene_gate)

    def test_relative_fused_mode_requires_dominance_and_healthy_fused_img(self):
        torch.testing.assert_close(
            self.apply(mode=4), torch.tensor([True, False, False, False]))

    def test_relative_fused_mode_fails_closed_without_reference_evidence(self):
        torch.testing.assert_close(
            self.apply(mode=4, fused_available=False),
            torch.zeros(4, dtype=torch.bool))
        torch.testing.assert_close(
            self.apply(mode=4, dominance_available=False),
            torch.zeros(4, dtype=torch.bool))

    def test_composite_pure_bev_mode_requires_bounded_bev_damage(self):
        torch.testing.assert_close(
            self.apply(mode=5), torch.tensor([True, False, False, False]))

    def test_composite_pure_bev_mode_fails_closed_without_required_evidence(self):
        torch.testing.assert_close(
            self.apply(mode=5, bev_available=False),
            torch.zeros(4, dtype=torch.bool))
        torch.testing.assert_close(
            self.apply(mode=5, fused_available=False),
            torch.zeros(4, dtype=torch.bool))
        torch.testing.assert_close(
            self.apply(mode=5, dominance_available=False),
            torch.zeros(4, dtype=torch.bool))

    def test_dual_abstain_mode_rejects_dual_img_or_fused_damage(self):
        torch.testing.assert_close(
            self.apply(mode=6), torch.tensor([True, False, False, False]))

    def test_dual_abstain_mode_fails_closed_without_required_evidence(self):
        torch.testing.assert_close(
            EDLPMETransformer._supp_apply_scene_damage_guard(
                self.scene_gate,
                True,
                6,
                self.img_damage,
                self.bev_damage,
                True,
                True,
                fused_damage_gate=self.fused_damage,
                bev_fused_dominance_gate=self.bev_fused_dominance,
                fused_threshold_available=True,
                dominance_reference_available=True,
                dual_abstain_available=False,
                img_abstain_available=True),
            torch.zeros(4, dtype=torch.bool))
        torch.testing.assert_close(
            EDLPMETransformer._supp_apply_scene_damage_guard(
                self.scene_gate,
                True,
                6,
                self.img_damage,
                self.bev_damage,
                True,
                True,
                fused_damage_gate=self.fused_damage,
                bev_fused_dominance_gate=self.bev_fused_dominance,
                dual_abstain_gate=self.dual_abstain,
                img_abstain_gate=self.img_abstain,
                fused_threshold_available=True,
                dominance_reference_available=False,
                dual_abstain_available=True,
                img_abstain_available=True),
            torch.zeros(4, dtype=torch.bool))

    def test_raw_modality_uncertainty_is_available_to_relative_guard(self):
        fused = torch.tensor([[[0.02], [0.04]]])
        bev = torch.tensor([[[0.03], [0.06]]])
        img = torch.tensor([[[0.01], [0.02]]])
        blended = EDLPMETransformer._publish_raw_modality_uncertainty(
            {}, {'fused': fused, 'bev': bev, 'img': img})

        for modality, expected in (
                ('fused', fused), ('bev', bev), ('img', img)):
            actual = EDLPMETransformer._scrg_tensor_value(
                f'{modality}_raw_modality_unc', blended, {})
            torch.testing.assert_close(actual, expected[..., 0])

    def test_raw_modality_uncertainty_is_published_before_supplement(self):
        source = inspect.getsource(EDLPMETransformer.forward)
        publish_pos = source.index(
            'self._publish_raw_modality_uncertainty')
        supplement_pos = source.index(
            'self._maybe_apply_img_supplement_admission')
        self.assertLess(publish_pos, supplement_pos)

    def test_supplement_off_matched_path_never_calls_admission(self):
        transformer = object.__new__(EDLPMETransformer)

        def forbidden_admission(_self, *_args, **_kwargs):
            raise AssertionError("matched supplement-off path called admission")

        transformer._apply_img_supplement_admission = types.MethodType(
            forbidden_admission, transformer)
        original_tensor = torch.tensor([1.0, 2.0])
        blended = {"bbox_output": original_tensor.clone()}
        original_keys = set(blended)

        result = transformer._maybe_apply_img_supplement_admission(
            False, blended, {}, {}, ["fused", "bev", "img"], [2, 2, 2])

        self.assertIs(result, blended)
        self.assertEqual(set(result), original_keys)
        torch.testing.assert_close(result["bbox_output"], original_tensor)
        self.assertFalse(any(key.startswith("supp_") for key in result))
        default = inspect.signature(
            EDLPMETransformer.__init__).parameters["supp_enable"].default
        self.assertIs(default, False)

    def test_scene_feature_tensor_resolves_dotted_nested_runtime_signal(self):
        transformer = object.__new__(EDLPMETransformer)
        reference = torch.zeros((2, 4), dtype=torch.float32)
        value, available = transformer._scene_feature_tensor(
            'distance_bin_abnormality_stats.height.bev.mid',
            {
                'distance_bin_abnormality_stats': {
                    'height': {
                        'bev': {
                            'mid': torch.tensor([0.42, 0.61]),
                        },
                    },
                },
            },
            {},
            reference,
            return_available=True)

        self.assertTrue(available)
        torch.testing.assert_close(value, torch.tensor([0.42, 0.61]))

    def test_scene_feature_tensor_resolves_selector_distance_bin_alias(self):
        transformer = object.__new__(EDLPMETransformer)
        reference = torch.zeros((2, 4), dtype=torch.float32)
        value, available = transformer._scene_feature_tensor(
            'distance_bin_abnormality_stats.height.bev.mid',
            {
                'selector_distance_bin_abnormality_stats': {
                    'height': {
                        'bev': {
                            'mid': torch.tensor([0.39, 0.57]),
                        },
                    },
                },
            },
            {},
            reference,
            return_available=True)

        self.assertTrue(available)
        torch.testing.assert_close(value, torch.tensor([0.39, 0.57]))

    def test_scene_feature_tensor_fails_closed_for_missing_dotted_signal(self):
        transformer = object.__new__(EDLPMETransformer)
        reference = torch.ones((2, 4), dtype=torch.float32)
        value, available = transformer._scene_feature_tensor(
            'distance_bin_abnormality_stats.height.bev.mid',
            {'distance_bin_abnormality_stats': {'height': {'img': {}}}},
            {},
            reference,
            return_available=True)

        self.assertFalse(available)
        torch.testing.assert_close(value, torch.zeros(2))

    def test_contextheight_gate_allows_context_when_height_missing(self):
        scene_gate = torch.tensor([True, True, True])
        context_gate = torch.tensor([True, False, True])
        height_gate = torch.tensor([False, False, False])
        v9_bev_gate = torch.tensor([True, True, False])

        actual = EDLPMETransformer._supp_apply_contextheight_gate(
            scene_gate,
            context_gate,
            height_gate,
            v9_bev_gate,
            context_available=True,
            height_available=False,
            v9_bev_available=True)

        torch.testing.assert_close(actual, torch.tensor([True, False, False]))

    def test_contextheight_gate_fails_closed_without_any_or_branch(self):
        scene_gate = torch.tensor([True, True])
        actual = EDLPMETransformer._supp_apply_contextheight_gate(
            scene_gate,
            torch.tensor([True, False]),
            torch.tensor([False, True]),
            torch.tensor([True, True]),
            context_available=False,
            height_available=False,
            v9_bev_available=True)

        torch.testing.assert_close(actual, torch.zeros(2, dtype=torch.bool))

    def test_contextheight_gate_fails_closed_without_v9_bev(self):
        scene_gate = torch.tensor([True, True])
        actual = EDLPMETransformer._supp_apply_contextheight_gate(
            scene_gate,
            torch.tensor([True, False]),
            torch.tensor([False, True]),
            torch.tensor([True, True]),
            context_available=True,
            height_available=True,
            v9_bev_available=False)

        torch.testing.assert_close(actual, torch.zeros(2, dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
