# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import hashlib
import os
import sys
import warnings

# Configs resolve `plugin_dir` and `projects/...` imports relative to the
# repository root, but running `python tools/test.py` puts tools/ on sys.path
# instead. Add the root so the plugin is importable either way.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmcv
import mmdet
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

try:
    from umr_collector_contract import current_collector_contract
except ImportError:  # Support importing this file as tools.test in audits.
    from tools.umr_collector_contract import current_collector_contract

try:
    from mmdet.apis.test import collect_results_cpu, collect_results_gpu
except Exception:  # pragma: no cover - available in the runtime env
    try:
        from mmdet.apis import collect_results_cpu, collect_results_gpu
    except Exception:  # pragma: no cover
        collect_results_cpu = None
        collect_results_gpu = None

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == '':
        return int(default)
    return int(value)


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None or value == '':
        return float(default)
    return float(value)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
             ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function (deprecate), '
             'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank','--local-rank', type=int, default=0)
    # [EDL] Uncertainty logging
    parser.add_argument('--save-edl-log', action='store_true',
                        help='Save per-sample EDL uncertainty JSON log')
    parser.add_argument('--edl-log-dir', type=str, default=None,
                        help='Directory for EDL uncertainty log output')
    # [Ev-MEFormer] Streaming EUD/UMR logging for corruption evaluation.
    parser.add_argument('--save-ev-log', action='store_true',
                        help='Stream full Ev-MEFormer EUD/UMR logs during test')
    parser.add_argument('--ev-log-dir', type=str, default=None,
                        help='Directory for Ev-MEFormer streaming log output')
    parser.add_argument('--ev-log-mode', type=str, default='full_stream',
                        choices=['full_stream', 'sidecar'],
                        help='Ev-MEFormer log mode. full_stream stores full '
                             'sample and prediction streams; sidecar stores '
                             'summary/light streams only.')
    parser.add_argument('--ev-log-compression', type=str, default='gzip',
                        choices=['none', 'gzip', 'gz', 'zstd', 'zst'],
                        help='Compression for Ev-MEFormer JSONL streams')
    parser.add_argument('--ev-log-flush-interval', type=int, default=100,
                        help='Flush Ev-MEFormer streams every N samples. '
                             'Use 0 to rely on OS buffering.')
    parser.add_argument('--ev-log-prediction-full', action='store_true',
                        help='Also write a full per-prediction stream. The '
                             'full sample stream is already lossless; this '
                             'option trades disk/write cost for easier '
                             'per-prediction post-processing.')
    parser.add_argument('--ev-log-scene-only', action='store_true',
                        help='Write only scene-level EV sidecar records. This '
                             'keeps admission/gate telemetry and AP/NDS output '
                             'while skipping expensive per-prediction JSONL.')
    parser.add_argument('--ev-log-only', action='store_true',
                        help='When --save-ev-log is enabled, stream EV logs '
                             'and discard result objects after logging. This '
                             'is intended for train/train-O.C calibration log '
                             'collection where AP formatting/eval is not run.')
    parser.add_argument('--ev-log-branch-candidates', action='store_true',
                        default=_env_flag('EV_LOG_BRANCH_CANDIDATES', False),
                        help='Write bounded fused/BEV/IMG branch-candidate '
                             'sidecar rows for trainable router calibration. '
                             'Can also be enabled with EV_LOG_BRANCH_CANDIDATES=1.')
    parser.add_argument('--ev-log-query-semantic', action='store_true',
                        default=_env_flag('EV_LOG_QUERY_SEMANTIC', False),
                        help='Write query-level fused/BEV/IMG semantic summary '
                             'sidecar rows. Can also be enabled with '
                             'EV_LOG_QUERY_SEMANTIC=1.')
    parser.add_argument('--ev-query-semantic-topk', type=int,
                        default=_env_int('EV_QUERY_SEMANTIC_TOPK', 0),
                        help='Maximum query-semantic rows per sample after '
                             'ranking by max branch semantic score. Use 0 for all.')
    parser.add_argument('--ev-query-semantic-score-thr', type=float,
                        default=_env_float('EV_QUERY_SEMANTIC_SCORE_THR', 0.0),
                        help='Minimum max branch semantic score for query semantic rows.')
    parser.add_argument('--ev-branch-candidate-topk', type=int,
                        default=_env_int('EV_BRANCH_CANDIDATE_TOPK', 300),
                        help='Maximum branch-candidate rows per branch per sample. Use '
                             'EV_BRANCH_CANDIDATE_TOPK to set from bash.')
    parser.add_argument('--ev-branch-candidate-score-thr', type=float,
                        default=_env_float('EV_BRANCH_CANDIDATE_SCORE_THR', 0.0),
                        help='Minimum branch-candidate score to write.')
    parser.add_argument('--ev-branch-candidate-branches', type=str,
                        default=os.environ.get(
                            'EV_BRANCH_CANDIDATE_BRANCHES', 'fused,bev,img'),
                        help='Comma-separated branch list for branch-candidate logging.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def _collect_meta_dicts(obj):
    """Extract per-sample metadata dicts from MMDataContainer/list wrappers."""
    if hasattr(obj, 'data'):
        obj = obj.data
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, (list, tuple)):
        metas = []
        for item in obj:
            metas.extend(_collect_meta_dicts(item))
        return metas
    return []


def _sample_token_from_meta(meta, data_info, fallback):
    if isinstance(meta, dict):
        for key in ('sample_idx', 'token', 'sample_token'):
            if meta.get(key) is not None:
                return meta[key]
    if isinstance(data_info, dict):
        for key in ('token', 'sample_idx', 'sample_token'):
            if data_info.get(key) is not None:
                return data_info[key]
    return str(fallback)


def _dataset_info_by_token(dataset):
    data_infos = getattr(dataset, 'data_infos', None)
    if not isinstance(data_infos, list):
        return {}
    mapping = {}
    for idx, info in enumerate(data_infos):
        if not isinstance(info, dict):
            continue
        for key in ('token', 'sample_idx', 'sample_token'):
            value = info.get(key)
            if value is not None:
                mapping[str(value)] = info
    return mapping


def _merge_data_info(meta, token_to_info, fallback_info):
    info = {}
    if isinstance(fallback_info, dict):
        info.update(fallback_info)
    if isinstance(meta, dict):
        for key in ('sample_idx', 'token', 'sample_token'):
            value = meta.get(key)
            if value is not None and str(value) in token_to_info:
                info.update(token_to_info[str(value)])
                break
        info.update(meta)
    return info


def _strip_edl_info(result):
    """Drop heavy diagnostic tensors after streaming them to disk."""
    if isinstance(result, list):
        for item in result:
            _strip_edl_info(item)
        return result
    if not isinstance(result, dict):
        return result
    pts_bbox = result.get('pts_bbox', result)
    if isinstance(pts_bbox, dict):
        pts_bbox.pop('edl_info', None)
    return result


def _configure_ev_branch_candidate_logging(model, args):
    if not getattr(args, 'save_ev_log', False):
        return
    enabled = bool(getattr(args, 'ev_log_branch_candidates', False))
    branches = getattr(args, 'ev_branch_candidate_branches', 'fused,bev,img')
    configured = 0
    for module in model.modules():
        if not (hasattr(module, '_maybe_export_ev_branch_candidates') and
                hasattr(module, '_get_source_tensor_last')):
            continue
        setattr(module, 'ev_branch_candidate_log_enable', enabled)
        setattr(module, 'ev_branch_candidate_topk_per_sample',
                int(getattr(args, 'ev_branch_candidate_topk', 300) or 0))
        setattr(module, 'ev_branch_candidate_score_thr',
                float(getattr(args, 'ev_branch_candidate_score_thr', 0.0) or 0.0))
        setattr(module, 'ev_branch_candidate_branches', branches)
        configured += 1
    if enabled:
        if configured <= 0:
            raise RuntimeError(
                '--ev-log-branch-candidates was requested, but no PME exporter '
                'module was configured. Check that the loaded model uses '
                'EDLPMETransformer with branch-candidate export support.')
        print(
            "[EV BranchCandidate] enabled "
            f"(modules={configured}, topk={args.ev_branch_candidate_topk}, "
            f"score_thr={args.ev_branch_candidate_score_thr}, branches={branches})"
        )


def _configure_ev_query_semantic_logging(model, args):
    if not getattr(args, 'save_ev_log', False):
        return
    enabled = bool(getattr(args, 'ev_log_query_semantic', False))
    configured = 0
    for module in model.modules():
        if not (hasattr(module, '_publish_query_semantic_telemetry') and
                hasattr(module, '_get_single_source_tensor')):
            continue
        setattr(module, 'ev_query_semantic_log_enable', enabled)
        configured += 1
    if enabled:
        if configured <= 0:
            raise RuntimeError(
                '--ev-log-query-semantic was requested, but no PME semantic '
                'telemetry module was configured. Check that the loaded model '
                'uses EDLPMETransformer with query semantic telemetry support.')
        print(
            "[EV QuerySemantic] enabled "
            f"(modules={configured}, topk={args.ev_query_semantic_topk}, "
            f"score_thr={args.ev_query_semantic_score_thr})"
        )


def _build_ev_logger(args, cfg, dataset, rank=None, world_size=None):
    from projects.mmdet3d_plugin.models.utils.edl_uncertainty_logger import EDLUncertaintyLogger

    work_dir = getattr(cfg, 'work_dir', None)
    log_dir = args.ev_log_dir or args.edl_log_dir or (work_dir if work_dir else '.')
    sample_context = {
        'ev_split': os.environ.get('EV_SPLIT'),
        'corruption_type': os.environ.get('EV_CORRUPTION'),
        'corruption_severity': os.environ.get('EV_SEVERITY'),
        'ev_model_label': os.environ.get('EV_MODEL_LABEL'),
        'ev_log_mode': args.ev_log_mode,
        'ev_log_scene_only': str(bool(args.ev_log_scene_only)),
        'ev_log_branch_candidates': str(bool(args.ev_log_branch_candidates)),
        'ev_log_query_semantic': str(bool(args.ev_log_query_semantic)),
    }
    sample_context = {k: v for k, v in sample_context.items() if v not in (None, '')}
    record_source_hashes = _env_flag('EV_LOG_RECORD_SOURCE_HASHES', False)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    collector_contract = (
        current_collector_contract(repo_root) if record_source_hashes else {})
    test_loader_cfg = cfg.data.get('test_dataloader', {})
    resolved_samples_per_gpu = int(test_loader_cfg.get('samples_per_gpu', 1))
    extra_meta = {
        'config': args.config,
        'config_sha256': _sha256_file(args.config) if record_source_hashes else '',
        'checkpoint': args.checkpoint,
        'checkpoint_sha256': (
            _sha256_file(args.checkpoint) if record_source_hashes else ''),
        'dataset_size': len(dataset),
        'samples_per_gpu': resolved_samples_per_gpu,
        'collector_code_format': collector_contract.get('format', ''),
        'collector_code_fingerprint': collector_contract.get('fingerprint', ''),
        'collector_code_records': collector_contract.get('records', []),
        'branch_candidate_topk_per_branch': args.ev_branch_candidate_topk,
        'branch_candidate_score_thr': args.ev_branch_candidate_score_thr,
        'branch_candidate_branches': args.ev_branch_candidate_branches,
        'query_semantic_topk': args.ev_query_semantic_topk,
        'query_semantic_score_thr': args.ev_query_semantic_score_thr,
    }
    return EDLUncertaintyLogger(
        output_dir=log_dir,
        class_names=dataset.CLASSES,
        prefix='ev',
        rank=rank,
        world_size=world_size,
        compression=args.ev_log_compression,
        keep_full_json=False,
        stream_full=(args.ev_log_mode == 'full_stream' and not args.ev_log_scene_only),
        prediction_full_stream=(args.ev_log_prediction_full and not args.ev_log_scene_only),
        prediction_light=not args.ev_log_scene_only,
        flush_interval=args.ev_log_flush_interval,
        sample_context=sample_context,
        extra_meta=extra_meta,
        branch_candidates=args.ev_log_branch_candidates,
        branch_candidate_topk=args.ev_branch_candidate_topk,
        branch_candidate_score_thr=args.ev_branch_candidate_score_thr,
        branch_candidate_branches=args.ev_branch_candidate_branches,
        query_semantic=args.ev_log_query_semantic,
        query_semantic_topk=args.ev_query_semantic_topk,
        query_semantic_score_thr=args.ev_query_semantic_score_thr,
    )


def _log_ev_outputs(ev_logger, outputs, data, dataset, token_to_info, fallback_start_idx,
                    max_items=None):
    metas = _collect_meta_dicts(data.get('img_metas', []))
    if isinstance(outputs, dict):
        outputs = [outputs]
    if max_items is not None:
        outputs = outputs[:max(0, int(max_items))]
    for local_idx, output in enumerate(outputs):
        pts_bbox = output.get('pts_bbox', output)
        edl_info = pts_bbox.get('edl_info', {})
        meta = metas[local_idx] if local_idx < len(metas) else {}
        fallback_idx = fallback_start_idx + local_idx
        fallback_info = {}
        data_infos = getattr(dataset, 'data_infos', None)
        if isinstance(data_infos, list) and fallback_idx < len(data_infos):
            fallback_info = data_infos[fallback_idx]
        data_info = _merge_data_info(meta, token_to_info, fallback_info)
        sample_token = _sample_token_from_meta(meta, data_info, fallback_idx)
        bboxes = pts_bbox['boxes_3d']
        bboxes_tensor = bboxes.tensor if hasattr(bboxes, 'tensor') else bboxes
        ev_logger.log_sample(
            sample_token,
            bboxes_tensor,
            pts_bbox['scores_3d'],
            pts_bbox['labels_3d'],
            edl_info,
            data_info=data_info,
        )


def single_gpu_test_ev_stream(model, data_loader, args, cfg, show=False, out_dir=None):
    model.eval()
    dataset = data_loader.dataset
    ev_logger = _build_ev_logger(args, cfg, dataset, rank=0, world_size=1)
    token_to_info = _dataset_info_by_token(dataset)
    results = []
    seen = 0
    prog_bar = mmcv.ProgressBar(len(dataset))
    completed = False
    try:
        for data in data_loader:
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            if isinstance(result, dict):
                result = [result]
            _log_ev_outputs(ev_logger, result, data, dataset, token_to_info, seen)
            _strip_edl_info(result)
            if not args.ev_log_only:
                results.extend(result)
            seen += len(result)
            for _ in range(len(result)):
                prog_bar.update()
        completed = True
    finally:
        ev_logger.extra_meta["completed"] = completed
        ev_logger.save()
    return results


def multi_gpu_test_ev_stream(model, data_loader, args, cfg):
    if collect_results_cpu is None or collect_results_gpu is None:
        raise RuntimeError('mmdet.apis.test collect helpers are required for '
                           '--save-ev-log in distributed mode')
    model.eval()
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    ev_logger = _build_ev_logger(args, cfg, dataset, rank=rank, world_size=world_size)
    token_to_info = _dataset_info_by_token(dataset)
    results = []
    seen = 0
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
        progress_done = 0
    completed = False
    try:
        for data in data_loader:
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            if isinstance(result, dict):
                result = [result]
            # DistributedSampler pads the tail to make all ranks equally long.
            # Those padded samples are valid for metric collection because
            # MMDetection truncates gathered results later, but EV sidecar logs
            # are written independently per rank and would otherwise keep the
            # duplicate tail records. Phase-1 utility calibration expects a
            # one-pass train log, so skip padded tail batches for logging.
            real_count = sum(
                1 for local_idx in range(len(result))
                if rank + (seen + local_idx) * world_size < len(dataset))
            if real_count > 0:
                _log_ev_outputs(
                    ev_logger, result, data, dataset, token_to_info, seen,
                    max_items=real_count)
            _strip_edl_info(result)
            if not args.ev_log_only:
                results.extend(result)
            seen += len(result)
            if rank == 0:
                update_count = min(
                    len(result) * world_size,
                    max(0, len(dataset) - progress_done))
                for _ in range(update_count):
                    prog_bar.update()
                progress_done += update_count
        completed = True
    finally:
        ev_logger.extra_meta["completed"] = completed
        ev_logger.save()
    if args.ev_log_only:
        return []
    if args.gpu_collect:
        return collect_results_gpu(results, len(dataset))
    return collect_results_cpu(results, len(dataset), args.tmpdir)


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir or args.save_ev_log, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show", "--show-dir", or "--save-ev-log"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')
    if args.ev_log_only:
        if not args.save_ev_log:
            raise ValueError('--ev-log-only requires --save-ev-log')
        if args.out or args.eval or args.format_only or args.show or args.show_dir:
            raise ValueError(
                '--ev-log-only discards model outputs and cannot be combined '
                'with --out, --eval, --format-only, --show, or --show-dir')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if cfg.get('optimizer_config', None) is not None and cfg.optimizer_config['type'] == 'CustomFp16OptimizerHook':
        wrap_fp16_model(model)
        for module_name, v in cfg.optimizer_config['custom_fp16'].items():
            model._modules[module_name].fp16_enabled = v

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    _configure_ev_branch_candidate_logging(model, args)
    _configure_ev_query_semantic_logging(model, args)

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        if args.save_ev_log:
            outputs = single_gpu_test_ev_stream(
                model, data_loader, args, cfg, args.show, args.show_dir)
        else:
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        if args.save_ev_log:
            outputs = multi_gpu_test_ev_stream(model, data_loader, args, cfg)
        else:
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                     args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))
        
        # [EDL] Save per-sample uncertainty JSON log
        if args.save_edl_log and not args.save_ev_log:
            from projects.mmdet3d_plugin.models.utils.edl_uncertainty_logger import EDLUncertaintyLogger
            work_dir = getattr(cfg, 'work_dir', None)
            log_dir = args.edl_log_dir or (work_dir if work_dir else '.')
            edl_logger = EDLUncertaintyLogger(
                output_dir=log_dir,
                class_names=dataset.CLASSES
            )
            for idx, output in enumerate(outputs):
                # output format: {'pts_bbox': {'boxes_3d': ..., 'scores_3d': ..., 
                #                  'labels_3d': ..., 'edl_info': ...}}
                pts_bbox = output.get('pts_bbox', output)
                bboxes = pts_bbox['boxes_3d']
                scores = pts_bbox['scores_3d']
                labels = pts_bbox['labels_3d']
                edl_info = pts_bbox.get('edl_info', {})
                
                # Get sample token from dataset
                data_info = dataset.data_infos[idx]
                sample_token = data_info.get('token', data_info.get('sample_idx', str(idx)))
                
                # Extract bbox tensor
                bboxes_tensor = bboxes.tensor if hasattr(bboxes, 'tensor') else bboxes
                edl_logger.log_sample(
                    sample_token,
                    bboxes_tensor,
                    scores,
                    labels,
                    edl_info,
                    data_info=data_info,
                )
            
            edl_logger.save()
            edl_logger.print_summary()


if __name__ == '__main__':
    main()
