import importlib.util
import os

import pytest
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH = os.path.join(
    REPO_ROOT, 'projects/mmdet3d_plugin/models/utils/v10_quality.py')
SPEC = importlib.util.spec_from_file_location('v10_quality_under_test', MODULE_PATH)
V10_QUALITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V10_QUALITY)


def _branch(closest_query):
    center = torch.tensor([[[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]]])
    center[0, closest_query] = torch.tensor([0.0, 0.0])
    return {
        'cls_logits': torch.zeros(1, 3, 1),
        'center': center,
        'height': torch.zeros(1, 3, 1),
    }


def _inputs():
    branches = {
        'fused': _branch(0),
        'bev': _branch(1),
        'img': _branch(2),
    }
    gt = {
        'gt_cls': torch.tensor([[0]]),
        'gt_box': torch.zeros(1, 1, 9),
    }
    return branches, gt


def test_union_matching_uses_all_branch_assignments():
    branches, gt = _inputs()
    result = V10_QUALITY.union_matching(
        branches, gt, matching_mode='union')
    assert result['matching_mode'] == 'union'
    assert result['pairs'].tolist() == [[0, 0, 0], [0, 1, 0], [0, 2, 0]]


def test_single_best_shares_fused_assignment():
    branches, gt = _inputs()
    result = V10_QUALITY.union_matching(
        branches, gt, matching_mode='single_best')
    assert result['matching_mode'] == 'single_best'
    assert result['pairs'].tolist() == [[0, 0, 0]]
    assert all(match.tolist() == [[0]] for match in result['matching'].values())


def test_unknown_matching_mode_fails_closed():
    branches, gt = _inputs()
    with pytest.raises(ValueError, match='Unsupported V10 matching_mode'):
        V10_QUALITY.union_matching(branches, gt, matching_mode='unknown')
