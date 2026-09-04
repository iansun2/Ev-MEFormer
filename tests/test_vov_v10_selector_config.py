import hashlib
import json
import os
import subprocess
import sys

from mmcv import Config
import pytest
import torch

from tools import stamp_umr_model_provenance as provenance
from tools import v10_preflight


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CONFIG = os.path.join(
    REPO_ROOT, 'projects/configs/',
    'edl_pme_stage2_bbox_r3c_cls_voxel0075_vov_1600x640_cbgs_'
    'v10_router_selector_placeholder.py')
BUILDER = os.path.join(REPO_ROOT, 'tools/build_vov_v10_selector_config.py')
PREFLIGHT = os.path.join(REPO_ROOT, 'tools/v10_preflight.py')
REGISTER = os.path.join(
    REPO_ROOT, 'tools/register_vov_umr_sgsup_runtime_configs.py')
CRITICAL_PATHS = (
    'projects/mmdet3d_plugin/models/utils/edl_pme_transformer.py',
    'projects/mmdet3d_plugin/models/utils/v10_quality.py',
    'projects/mmdet3d_plugin/models/dense_heads/edl_head.py',
    'projects/mmdet3d_plugin/datasets/pipelines/synthetic_corruptions.py',
    'tools/v10_quality_calibration.py',
    'tools/build_vov_v10_selector_config.py',
)
TRAIN_INFOS_SHA256 = (
    '6e07dac06f515722fa9dbb685dea40f39a561c2d06739d5bd7cb83a340855854')


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _calibration(checkpoint, *, dry_run=False):
    return {
        'format': 'v10_quality_calibration_v2',
        'formal_train_only': True,
        'tau_margin': 1.25,
        'margin': 0.10,
        'tau_dmg': 0.20,
        'T_oracle': 1.5,
        'model_provenance': {
            'base_config_sha256': _sha256(BASE_CONFIG),
            'checkpoint_sha256': _sha256(checkpoint),
            'checkpoint_epoch': 4,
            'train_infos_sha256': TRAIN_INFOS_SHA256,
            'critical_code': [
                {'path': path, 'sha256': _sha256(os.path.join(REPO_ROOT, path))}
                for path in CRITICAL_PATHS
            ],
        },
        '_diagnostics': {'dry_run': dry_run, 'total_pairs': 1234},
    }


def _run_builder(tmp_path, payload):
    checkpoint = tmp_path / 'epoch_4.pth'
    if not checkpoint.exists():
        torch.save({'meta': {'epoch': 4}, 'state_dict': {}}, checkpoint)
    calibration = tmp_path / 'calibration.json'
    calibration.write_text(json.dumps(payload), encoding='utf-8')
    output = tmp_path / 'vov_v10_calibrated.py'
    result = subprocess.run([
        sys.executable, BUILDER,
        '--base-config', BASE_CONFIG,
        '--calibration', str(calibration),
        '--source-checkpoint', str(checkpoint),
        '--output', str(output),
    ], cwd=REPO_ROOT, text=True, capture_output=True)
    return result, output, checkpoint


def test_builder_pins_calibrated_thresholds_and_r50_matching(tmp_path):
    checkpoint = tmp_path / 'epoch_4.pth'
    torch.save({'meta': {'epoch': 4}, 'state_dict': {}}, checkpoint)
    result, output, _ = _run_builder(
        tmp_path, _calibration(str(checkpoint)))
    assert result.returncode == 0, result.stderr
    cfg = Config.fromfile(str(output))
    assert cfg.model.img_backbone.type == 'VoVNet'
    assert cfg.model.pts_bbox_head.ensemble.v10_enable is True
    assert cfg.model.pts_bbox_head.v10_matching_mode == 'single_best'
    assert cfg.total_epochs == 6
    assert dict(cfg.model.pts_bbox_head.v10_oracle_thresholds) == {
        'tau_margin': 1.25, 'margin': 0.10,
        'tau_dmg': 0.20, 'T_oracle': 1.5,
    }
    assert (
        cfg.model.pts_bbox_head.v10_oracle_thresholds ==
        cfg.model.pts_bbox_head.ensemble.v10_oracle_thresholds)
    assert cfg.v10_calibration_provenance.formal_train_only is True
    assert os.path.realpath(os.path.join(
        os.path.dirname(str(output)), cfg.load_from)) == os.path.realpath(checkpoint)
    identity = provenance.config_identity(str(output), 'vov_v10')
    assert identity['backbone_type'] == 'VoVNet'
    assert identity['v10_enable'] is True
    assert identity['v10_matching_mode'] == 'single_best'
    with pytest.raises(SystemExit):
        provenance.config_identity(str(output), 'vov_umr')
    calibration = tmp_path / 'calibration.json'
    assert v10_preflight._check_vov_calibration_files(
        cfg, str(calibration), str(checkpoint)) == []
    preflight = subprocess.run([
        sys.executable, PREFLIGHT, str(output),
        '--profile', 'vov_v10', '--config-only',
        '--checkpoint', str(checkpoint),
        '--source-checkpoint', str(checkpoint),
        '--calibration', str(calibration),
    ], cwd=REPO_ROOT, text=True, capture_output=True)
    assert preflight.returncode == 0, preflight.stdout + preflight.stderr
    assert '[V10 Config Preflight] PASS' in preflight.stdout

    relative_base = os.path.relpath(str(output), str(tmp_path))
    matched = tmp_path / 'UMR_MATCHED_BASE.py'
    raw = tmp_path / 'RULE_SGSUP_B_runtime.py'
    final = tmp_path / 'RULE_SGSUP_B_contextheight055_v9bev_dyn10_30.py'
    matched.write_text("_base_ = {!r}\n".format(relative_base), encoding='utf-8')
    raw.write_text(
        "_base_ = {!r}\nmodel = dict(pts_bbox_head=dict(ensemble=dict("
        "supp_enable=True, supp_mode='img_highscore', "
        "supp_max_add_queries=90, "
        "supp_contextheight_gate_enable=False)))\n".format(relative_base),
        encoding='utf-8')
    final.write_text(
        "_base_ = {!r}\nmodel = dict(pts_bbox_head=dict(ensemble=dict("
        "supp_enable=True, supp_mode='img_highscore', "
        "supp_max_add_queries=90, "
        "supp_contextheight_gate_enable=True)))\n".format(relative_base),
        encoding='utf-8')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({
        'profile': 'vov_v10',
        'model_provenance': {'profile': 'vov_v10'},
    }), encoding='utf-8')
    registered = subprocess.run([
        sys.executable, REGISTER,
        '--profile', 'vov_v10', '--manifest', str(manifest),
        '--base-config', str(output), '--matched-base', str(matched),
        '--raw-control', str(raw), '--formal-final', str(final),
    ], cwd=REPO_ROOT, text=True, capture_output=True)
    assert registered.returncode == 0, registered.stdout + registered.stderr
    runtime_configs = json.loads(
        manifest.read_text(encoding='utf-8'))['runtime_configs']
    assert runtime_configs['UMR_MATCHED_BASE']['role'] == (
        'matched_vov_v10_action_off')


@pytest.mark.parametrize('mutation', ('dry_run', 'code_sha', 'checkpoint_epoch'))
def test_builder_rejects_nonformal_or_stale_calibration(tmp_path, mutation):
    checkpoint = tmp_path / 'epoch_4.pth'
    torch.save({'meta': {'epoch': 4}, 'state_dict': {}}, checkpoint)
    payload = _calibration(str(checkpoint))
    if mutation == 'dry_run':
        payload['_diagnostics']['dry_run'] = True
    elif mutation == 'code_sha':
        payload['model_provenance']['critical_code'][0]['sha256'] = '0' * 64
    else:
        payload['model_provenance']['checkpoint_epoch'] = 3
    result, output, _ = _run_builder(tmp_path, payload)
    assert result.returncode != 0
    assert not output.exists()
