import contextlib
import io
import os
import sys
import unittest

import torch
from torch import nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import v9_rev20_preflight as preflight


class _DummyEnsemble(nn.Module):
    def __init__(self, omit=None):
        super().__init__()
        omit = set(omit or ())
        for name in (
                'distance_bin_edges_tensor',
                'v5_distance_bins_tensor',
                'v6_distance_bins_tensor',
                'v7_distance_bins_tensor'):
            if name not in omit:
                self.register_buffer(
                    name, torch.tensor([20.0, 40.0]), persistent=False)
        self.register_buffer('persistent_anchor', torch.ones(2), persistent=True)


class _DummyHead(nn.Module):
    def __init__(self, omit=None):
        super().__init__()
        self.ensemble = _DummyEnsemble(omit=omit)


class _DummyModel(nn.Module):
    def __init__(self, omit=None):
        super().__init__()
        self.pts_bbox_head = _DummyHead(omit=omit)
        self.proj = nn.Linear(2, 2)


def _checkpoint(model, include_compat=True):
    state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    if include_compat:
        buffers = dict(model.named_buffers())
        for name in preflight.MMCV_NONPERSISTENT_COMPAT_KEYS:
            if name in buffers:
                state[name] = buffers[name].detach().clone()
    return {'state_dict': state}


def _coverage_errors(model, checkpoint):
    with contextlib.redirect_stdout(io.StringIO()):
        return preflight._check_checkpoint_coverage(model, checkpoint)


class CheckpointCoverageTest(unittest.TestCase):
    def test_matching_nonpersistent_extras_pass(self):
        model = _DummyModel()
        self.assertEqual(_coverage_errors(model, _checkpoint(model)), [])

    def test_absent_nonpersistent_extras_pass(self):
        model = _DummyModel()
        self.assertEqual(
            _coverage_errors(model, _checkpoint(model, include_compat=False)), [])

    def test_module_prefix_normalization_passes(self):
        model = _DummyModel()
        checkpoint = _checkpoint(model)
        checkpoint['state_dict'] = {
            'module.' + name: value
            for name, value in checkpoint['state_dict'].items()
        }
        self.assertEqual(_coverage_errors(model, checkpoint), [])

    def test_nonpersistent_value_mismatch_fails(self):
        model = _DummyModel()
        checkpoint = _checkpoint(model)
        key = preflight.MMCV_NONPERSISTENT_COMPAT_KEYS[0]
        checkpoint['state_dict'][key] = torch.tensor([20.0, 41.0])
        errors = _coverage_errors(model, checkpoint)
        self.assertTrue(any('value mismatch' in item for item in errors))

    def test_nonpersistent_shape_mismatch_fails(self):
        model = _DummyModel()
        checkpoint = _checkpoint(model)
        key = preflight.MMCV_NONPERSISTENT_COMPAT_KEYS[0]
        checkpoint['state_dict'][key] = torch.tensor([20.0, 40.0, 60.0])
        errors = _coverage_errors(model, checkpoint)
        self.assertTrue(any('shape mismatch' in item for item in errors))

    def test_nonpersistent_dtype_mismatch_fails(self):
        model = _DummyModel()
        checkpoint = _checkpoint(model)
        key = preflight.MMCV_NONPERSISTENT_COMPAT_KEYS[0]
        checkpoint['state_dict'][key] = torch.tensor(
            [20.0, 40.0], dtype=torch.float64)
        errors = _coverage_errors(model, checkpoint)
        self.assertTrue(any('dtype mismatch' in item for item in errors))

    def test_nonpersistent_checkpoint_value_must_be_tensor(self):
        model = _DummyModel()
        checkpoint = _checkpoint(model)
        key = preflight.MMCV_NONPERSISTENT_COMPAT_KEYS[0]
        checkpoint['state_dict'][key] = [20.0, 40.0]
        errors = _coverage_errors(model, checkpoint)
        self.assertTrue(any('not a tensor' in item for item in errors))

    def test_registered_buffer_must_be_nonpersistent(self):
        model = _DummyModel()
        key = preflight.MMCV_NONPERSISTENT_COMPAT_KEYS[0]
        leaf = key.rsplit('.', 1)[1]
        owner = model.pts_bbox_head.ensemble
        owner._non_persistent_buffers_set.remove(leaf)
        error = preflight._validate_nonpersistent_compat_key(
            model, key, owner._buffers[leaf].clone())
        self.assertIn('not marked nonpersistent', error)

    def test_allowlisted_runtime_buffer_missing_fails(self):
        missing_leaf = 'distance_bin_edges_tensor'
        model = _DummyModel(omit=(missing_leaf,))
        checkpoint = _checkpoint(model)
        checkpoint['state_dict'][preflight.MMCV_NONPERSISTENT_COMPAT_KEYS[0]] = (
            torch.tensor([20.0, 40.0]))
        errors = _coverage_errors(model, checkpoint)
        self.assertTrue(any('not a registered buffer' in item for item in errors))

    def test_unknown_unexpected_key_fails(self):
        model = _DummyModel()
        checkpoint = _checkpoint(model)
        checkpoint['state_dict'][
            'pts_bbox_head.ensemble.unknown_distance_bins_tensor'] = (
                torch.tensor([20.0, 40.0]))
        errors = _coverage_errors(model, checkpoint)
        self.assertTrue(any('Unexpected checkpoint keys' in item for item in errors))

    def test_missing_persistent_key_fails(self):
        model = _DummyModel()
        checkpoint = _checkpoint(model)
        checkpoint['state_dict'].pop('proj.weight')
        errors = _coverage_errors(model, checkpoint)
        self.assertTrue(any('Missing checkpoint keys' in item for item in errors))

    def test_persistent_shape_mismatch_fails(self):
        model = _DummyModel()
        checkpoint = _checkpoint(model)
        checkpoint['state_dict']['proj.weight'] = torch.zeros(3, 2)
        errors = _coverage_errors(model, checkpoint)
        self.assertTrue(any('Tensor shape mismatches' in item for item in errors))

    def test_module_prefix_collision_fails(self):
        model = _DummyModel()
        checkpoint = _checkpoint(model, include_compat=False)
        checkpoint['state_dict']['module.proj.weight'] = (
            checkpoint['state_dict']['proj.weight'].clone())
        errors = _coverage_errors(model, checkpoint)
        self.assertTrue(any('collision' in item for item in errors))


class CheckpointEpochTest(unittest.TestCase):
    def test_integral_epoch_passes(self):
        self.assertEqual(
            preflight._check_checkpoint_epoch({'meta': {'epoch': 6}}, 6), [])

    def test_missing_epoch_fails(self):
        self.assertTrue(preflight._check_checkpoint_epoch({'meta': {}}, 6))

    def test_wrong_epoch_fails(self):
        self.assertTrue(
            preflight._check_checkpoint_epoch({'meta': {'epoch': 5}}, 6))

    def test_fractional_epoch_fails(self):
        self.assertTrue(
            preflight._check_checkpoint_epoch({'meta': {'epoch': 6.9}}, 6))

    def test_boolean_epoch_fails(self):
        self.assertTrue(
            preflight._check_checkpoint_epoch({'meta': {'epoch': True}}, 1))


if __name__ == '__main__':
    unittest.main()
