from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class EDLEpochHook(Hook):
    """
    Hook to update EDLHead's current_epoch for EDL loss scheduling.
    
    This enables:
    - EDL Warmup: Gradually increase NIG loss weight
    - KL Annealing: Control KL divergence regularization
    """
    
    def __init__(self,
                 v9_keep_min_schedule=None,
                 v9_gate_lambda_warmup=None,
                 v9_route_lambda_warmup=None,
                 selector_cls_lambda_warmup=None,
                 selector_box_lambda_warmup=None,
                 v10_cls_fused_keep_min_schedule=None,
                 v10_cls_fused_keep_min_schedule_conservative=None,
                 v10_route_kl_lambda_warmup=None):
        self.v9_keep_min_schedule = v9_keep_min_schedule or []
        self.v9_gate_lambda_warmup = v9_gate_lambda_warmup or None
        self.v9_route_lambda_warmup = v9_route_lambda_warmup or None
        self.selector_cls_lambda_warmup = selector_cls_lambda_warmup or None
        self.selector_box_lambda_warmup = selector_box_lambda_warmup or None
        self.v10_cls_fused_keep_min_schedule = v10_cls_fused_keep_min_schedule or []
        self.v10_cls_fused_keep_min_schedule_conservative = (
            v10_cls_fused_keep_min_schedule_conservative or [])
        self.v10_route_kl_lambda_warmup = v10_route_kl_lambda_warmup or None

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, 'module') else model

    def _get_head(self, runner):
        model = self._unwrap_model(runner.model)
        return getattr(model, 'pts_bbox_head', None)

    @staticmethod
    def _schedule_value(schedule, epoch, default=None):
        value = default
        for item in schedule:
            if isinstance(item, dict):
                start_epoch = int(item.get('epoch', item.get('start_epoch', 0)))
                item_value = item.get('value', default)
            else:
                start_epoch, item_value = item
                start_epoch = int(start_epoch)
            if epoch >= start_epoch:
                value = item_value
        return value

    @staticmethod
    def _warmup_value(cfg, runner, default):
        if not cfg or not cfg.get('enabled', True):
            return default
        start = float(cfg.get('start', default))
        end = float(cfg.get('end', default))
        warmup_epochs = max(float(cfg.get('warmup_epochs', 1.0)), 1e-6)
        if runner.epoch >= warmup_epochs:
            return end

        data_loader = getattr(runner, 'data_loader', None)
        iters_per_epoch = len(data_loader) if data_loader is not None else 1
        iters_per_epoch = max(float(iters_per_epoch), 1.0)
        inner_iter = float(getattr(runner, 'inner_iter', 0))
        progress = (float(runner.epoch) + inner_iter / iters_per_epoch) / warmup_epochs
        progress = min(max(progress, 0.0), 1.0)
        return start + (end - start) * progress

    @staticmethod
    def _publish_v9_schedule(
            head,
            keep_min=None,
            gate_lambda=None,
            route_lambda=None,
            selector_cls_lambda=None,
            selector_box_lambda=None):
        ensemble = getattr(head, 'ensemble', None)
        if keep_min is not None and ensemble is not None:
            ensemble.v8_fused_keep_min = float(keep_min)
            ensemble.v9_current_keep_min = float(keep_min)
        if gate_lambda is not None:
            head.lambda_v9_gate_domain = float(gate_lambda)
        if route_lambda is not None:
            head.lambda_v9_route_consistency = float(route_lambda)
        if selector_cls_lambda is not None:
            head.lambda_consistency_cls = float(selector_cls_lambda)
        if selector_box_lambda is not None:
            head.lambda_consistency_box = float(selector_box_lambda)
        if ensemble is not None:
            ensemble.v9_current_lambda_gate = float(getattr(head, 'lambda_v9_gate_domain', 0.0))
            ensemble.v9_current_lambda_route = float(getattr(head, 'lambda_v9_route_consistency', 0.0))
            ensemble.v9_current_lambda_selector_cls = float(getattr(head, 'lambda_consistency_cls', 0.0))
            ensemble.v9_current_lambda_selector_box = float(getattr(head, 'lambda_consistency_box', 0.0))

    @staticmethod
    def _publish_v10_schedule(head, cls_keep_min=None, route_kl_lambda=None):
        ensemble = getattr(head, 'ensemble', None)
        if cls_keep_min is not None and ensemble is not None:
            ensemble.v10_cls_fused_keep_min = float(cls_keep_min)
            ensemble.v10_current_cls_fused_keep_min = float(cls_keep_min)
        if route_kl_lambda is not None:
            head.lambda_v10_route_kl = float(route_kl_lambda)
        if ensemble is not None:
            ensemble.v10_current_lambda_route_kl = float(
                getattr(head, 'lambda_v10_route_kl', 0.0))

    def before_train_epoch(self, runner):
        """Update current_epoch at the start of each epoch."""
        head = self._get_head(runner)
        if head is not None:
            if hasattr(head, 'set_epoch'):
                head.set_epoch(runner.epoch)

            keep_min = self._schedule_value(
                self.v9_keep_min_schedule,
                runner.epoch,
                default=getattr(getattr(head, 'ensemble', None), 'v8_fused_keep_min', None))
            gate_lambda = self._warmup_value(
                self.v9_gate_lambda_warmup,
                runner,
                getattr(head, 'lambda_v9_gate_domain', 0.0))
            route_lambda = self._warmup_value(
                self.v9_route_lambda_warmup,
                runner,
                getattr(head, 'lambda_v9_route_consistency', 0.0))
            selector_cls_lambda = self._warmup_value(
                self.selector_cls_lambda_warmup,
                runner,
                getattr(head, 'lambda_consistency_cls', 0.0))
            selector_box_lambda = self._warmup_value(
                self.selector_box_lambda_warmup,
                runner,
                getattr(head, 'lambda_consistency_box', 0.0))
            if keep_min is not None:
                self._publish_v9_schedule(
                    head,
                    keep_min=keep_min,
                    gate_lambda=gate_lambda,
                    route_lambda=route_lambda,
                    selector_cls_lambda=selector_cls_lambda,
                    selector_box_lambda=selector_box_lambda)
            else:
                self._publish_v9_schedule(
                    head,
                    gate_lambda=gate_lambda,
                    route_lambda=route_lambda,
                    selector_cls_lambda=selector_cls_lambda,
                    selector_box_lambda=selector_box_lambda)

            ensemble = getattr(head, 'ensemble', None)
            if ensemble is not None:
                v10_schedule = (
                    self.v10_cls_fused_keep_min_schedule_conservative
                    if getattr(ensemble, 'v10_use_conservative_schedule', False)
                    else self.v10_cls_fused_keep_min_schedule
                )
                v10_keep_min = self._schedule_value(
                    v10_schedule,
                    runner.epoch,
                    default=getattr(ensemble, 'v10_cls_fused_keep_min', None))
            else:
                v10_keep_min = None
            v10_route_kl = self._warmup_value(
                self.v10_route_kl_lambda_warmup,
                runner,
                getattr(head, 'lambda_v10_route_kl', 0.0))
            self._publish_v10_schedule(
                head,
                cls_keep_min=v10_keep_min,
                route_kl_lambda=v10_route_kl)

            # Debug output (only on main process)
            if runner.rank == 0:
                warmup_epochs = getattr(head, 'warmup_epochs', 1.0)
                if runner.epoch < warmup_epochs:
                    warmup_factor = (runner.epoch + 1) / max((warmup_epochs + 1), 1.0)
                else:
                    warmup_factor = 1.0
                ensemble = getattr(head, 'ensemble', None)
                print(f"\n[EDLEpochHook] Epoch {runner.epoch + 1}")
                print(f"  EDL Warmup Factor: {warmup_factor:.4f}")
                print(f"  Warmup Epochs: {warmup_epochs}")
                if ensemble is not None and self.v9_keep_min_schedule:
                    print(f"  V9 keep_min: {getattr(ensemble, 'v8_fused_keep_min', float('nan')):.4f}")
                    print(
                        "  V9 lambdas:"
                        f" gate={getattr(head, 'lambda_v9_gate_domain', float('nan')):.4f},"
                        f" route={getattr(head, 'lambda_v9_route_consistency', float('nan')):.4f},"
                        f" sel_cls={getattr(head, 'lambda_consistency_cls', float('nan')):.4f},"
                        f" sel_box={getattr(head, 'lambda_consistency_box', float('nan')):.4f}")
                if ensemble is not None and self.v10_cls_fused_keep_min_schedule:
                    print(
                        "  V10:"
                        f" cls_keep_min={getattr(ensemble, 'v10_cls_fused_keep_min', float('nan')):.4f},"
                        f" route_kl={getattr(head, 'lambda_v10_route_kl', float('nan')):.4f},"
                        f" conservative={getattr(ensemble, 'v10_use_conservative_schedule', False)}")

    def before_train_iter(self, runner):
        head = self._get_head(runner)
        if head is None:
            return
        gate_lambda = self._warmup_value(
            self.v9_gate_lambda_warmup,
            runner,
            getattr(head, 'lambda_v9_gate_domain', 0.0))
        route_lambda = self._warmup_value(
            self.v9_route_lambda_warmup,
            runner,
            getattr(head, 'lambda_v9_route_consistency', 0.0))
        selector_cls_lambda = self._warmup_value(
            self.selector_cls_lambda_warmup,
            runner,
            getattr(head, 'lambda_consistency_cls', 0.0))
        selector_box_lambda = self._warmup_value(
            self.selector_box_lambda_warmup,
            runner,
            getattr(head, 'lambda_consistency_box', 0.0))
        self._publish_v9_schedule(
            head,
            gate_lambda=gate_lambda,
            route_lambda=route_lambda,
            selector_cls_lambda=selector_cls_lambda,
            selector_box_lambda=selector_box_lambda)
        v10_route_kl = self._warmup_value(
            self.v10_route_kl_lambda_warmup,
            runner,
            getattr(head, 'lambda_v10_route_kl', 0.0))
        self._publish_v10_schedule(head, route_kl_lambda=v10_route_kl)
