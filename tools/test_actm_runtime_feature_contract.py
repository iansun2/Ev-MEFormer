#!/usr/bin/env python3
"""Focused contract checks for ACT-M runtime mapper features."""

import unittest

import torch

from projects.mmdet3d_plugin.models.utils.edl_pme_transformer import EDLPMETransformer


class ACTMRuntimeFeatureContractTest(unittest.TestCase):

    def setUp(self):
        self.transformer = object.__new__(EDLPMETransformer)
        self.modalities = ['fused', 'bev', 'img']
        self.reference = torch.zeros(1, 2)
        self.v8_info = {
            'cls_weight_seq': torch.tensor(
                [[[[0.85, 0.10, 0.05], [0.86, 0.09, 0.05]]]]),
            'center_weight_seq': torch.tensor(
                [[[[0.95, 0.03, 0.02], [0.94, 0.04, 0.02]]]]),
            'height_weight_seq': torch.tensor(
                [[[[0.98, 0.01, 0.01], [0.97, 0.02, 0.01]]]]),
        }

    def test_extracts_exact_fused_head_weights(self):
        weights, missing = self.transformer._supp_runtime_branch_head_weights(
            'fused', self.modalities, self.v8_info, {}, self.reference)

        self.assertEqual(missing, [])
        torch.testing.assert_close(weights['cls_weight'], torch.tensor([[0.85, 0.86]]))
        torch.testing.assert_close(weights['center_weight'], torch.tensor([[0.95, 0.94]]))
        torch.testing.assert_close(weights['height_weight'], torch.tensor([[0.98, 0.97]]))

    def test_uses_requested_non_fused_modality_index(self):
        weights, missing = self.transformer._supp_runtime_branch_head_weights(
            'img', self.modalities, self.v8_info, {}, self.reference)

        self.assertEqual(missing, [])
        torch.testing.assert_close(weights['cls_weight'], torch.tensor([[0.05, 0.05]]))
        torch.testing.assert_close(weights['center_weight'], torch.tensor([[0.02, 0.02]]))
        torch.testing.assert_close(weights['height_weight'], torch.tensor([[0.01, 0.01]]))

    def test_reports_missing_contract_instead_of_shared_gate_fallback(self):
        weights, missing = self.transformer._supp_runtime_branch_head_weights(
            'fused', self.modalities, {}, {}, self.reference)

        self.assertEqual(missing, ['cls_weight', 'center_weight', 'height_weight'])
        for value in weights.values():
            torch.testing.assert_close(value, torch.zeros_like(self.reference))

    def test_accepts_explicit_v10_runtime_weight_fallback(self):
        runtime = {
            'v10_w_fused_cls': torch.tensor([[[0.85], [0.86]]]),
            'v10_w_fused_center': torch.tensor([[[0.95], [0.94]]]),
            'v10_w_fused_height': torch.tensor([[[0.98], [0.97]]]),
        }
        weights, missing = self.transformer._supp_runtime_branch_head_weights(
            'fused', self.modalities, {}, runtime, self.reference)

        self.assertEqual(missing, [])
        torch.testing.assert_close(weights['cls_weight'], torch.tensor([[0.85, 0.86]]))
        torch.testing.assert_close(weights['center_weight'], torch.tensor([[0.95, 0.94]]))
        torch.testing.assert_close(weights['height_weight'], torch.tensor([[0.98, 0.97]]))

    def test_rejects_malformed_head_weight_shapes(self):
        malformed = {
            'cls_weight_seq': torch.ones(3, 4),
            'center_weight_seq': torch.ones(2, 5, 3),
            'height_weight_seq': torch.ones(3, 2, 3),
        }
        _, missing = self.transformer._supp_runtime_branch_head_weights(
            'fused', self.modalities, malformed, {}, self.reference)

        self.assertEqual(missing, ['cls_weight', 'center_weight', 'height_weight'])


if __name__ == '__main__':
    unittest.main()
