from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.optimizer import GradientCumulativeFp16OptimizerHook


@HOOKS.register_module()
class CustomFp16OptimizerHook(GradientCumulativeFp16OptimizerHook):
    """
    Custom Fp16OptimizerHook with gradient accumulation support.
    """

    def __init__(self,
                 update_interval=1,
                 custom_fp16={},
                 *args,
                 **kwargs):
        """
        Extends the base __init__.
        """
        super(CustomFp16OptimizerHook, self).__init__(*args, **kwargs)
        
        self.custom_fp16 = custom_fp16
        
        assert update_interval > 0, "update_interval must be greater than 0"
        self.update_interval = update_interval
        self.current_interval = 0

    def before_run(self, runner) -> None:
        """
        Keeps the original before_run behaviour.
        """
        super().before_run(runner)
        for module_name, v in self.custom_fp16.items():
            if hasattr(runner.model, 'module'):
                runner.model.module._modules[module_name].fp16_enabled = v
            else:
                runner.model._modules[module_name].fp16_enabled = v

    