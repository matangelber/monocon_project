import os.path as osp
import warnings
from math import inf

import mmcv
import torch.distributed as dist
from mmcv.runner import Hook
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

from mmdet.utils import get_root_logger

class BatchToStereoBatchHook(Hook):
    """Evaluation hook.
    """


    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 by_epoch=True,
                 save_best=None,
                 rule=None,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')
        if not interval > 0:
            raise ValueError(f'interval must be positive, but got {interval}')
        if start is not None and start < 0:
            warnings.warn(
                f'The evaluation start epoch {start} is smaller than 0, '
                f'use 0 instead', UserWarning)
            start = 0
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        self.start = start
        assert isinstance(save_best, str) or save_best is None
        self.save_best = save_best
        self.eval_kwargs = eval_kwargs
        self.initial_epoch_flag = True

        self.logger = get_root_logger()

        if self.save_best is not None:
            self._init_rule(rule, self.save_best)

    def _init_rule(self, rule, key_indicator):
        """Initialize rule, key_indicator, comparison_func, and best score.

        Args:
            rule (str | None): Comparison rule for best score.
            key_indicator (str | None): Key indicator to determine the
                comparison rule.
        """
        if rule not in self.rule_map and rule is not None:
            raise KeyError(f'rule must be greater, less or None, '
                           f'but got {rule}.')

        if rule is None:
            if key_indicator != 'auto':
                if any(key in key_indicator for key in self.greater_keys):
                    rule = 'greater'
                elif any(key in key_indicator for key in self.less_keys):
                    rule = 'less'
                else:
                    raise ValueError(f'Cannot infer the rule for key '
                                     f'{key_indicator}, thus a specific rule '
                                     f'must be specified.')
        self.rule = rule
        self.key_indicator = key_indicator
        if self.rule is not None:
            self.compare_func = self.rule_map[self.rule]

    def before_run(self, runner):
        if self.save_best is not None:
            if runner.meta is None:
                warnings.warn('runner.meta is None. Creating a empty one.')
                runner.meta = dict()
            runner.meta.setdefault('hook_msgs', dict())
