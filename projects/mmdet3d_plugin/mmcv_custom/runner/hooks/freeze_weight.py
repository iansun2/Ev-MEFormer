from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class FreezeWeight(Hook):
    def __init__(self, finetune_weight=None, prefix_names=None, freeze_bn=False):
        super().__init__()
        self.finetune_weight = finetune_weight if finetune_weight is not None else prefix_names
        self.finetune_weight = self.finetune_weight or []
        self.freeze_bn = bool(freeze_bn)

    def _is_trainable_name(self, name):
        return any(name.startswith(prefix) for prefix in self.finetune_weight)

    def _freeze_bn_modules(self, model):
        if not self.freeze_bn:
            return
        for module_name, module in model.named_modules():
            if self._is_trainable_name(module_name):
                continue
            if module.__class__.__name__.lower().find('batchnorm') >= 0:
                module.eval()

    def before_run(self, runner):
        if hasattr(runner.model, "module"):
            model = runner.model.module
        else:
            model = runner.model

        freezed = []
        not_freezed = []
        for name, p in model.named_parameters():
            flag = self._is_trainable_name(name) and p.requires_grad
            if flag:
                not_freezed.append(name)

            if not flag:
                p.requires_grad = False
                freezed.append(name)

        self._freeze_bn_modules(model)
        runner.logger.info(f"Freezed parameters: {', '.join(freezed)}")
        runner.logger.info(f"Learned parameters: {', '.join(not_freezed)}")

    def before_train_epoch(self, runner):
        if hasattr(runner.model, "module"):
            model = runner.model.module
        else:
            model = runner.model
        self._freeze_bn_modules(model)
