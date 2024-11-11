from mmcv.runner import HOOKS, Hook
from mmcv.parallel import collate, DataContainer
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from collections.abc import Sequence, Mapping
from functools import partial

def custom_collate_fn(batch, samples_per_gpu=1):
    """
    Custom collate function to separate `results_cam2` and `results_cam3`
    and apply the default `mmcv.collate` to the flattened batch.

    Args:
        batch (list[dict]): List of dictionaries, each containing 'results_cam2' and 'results_cam3'.

    Returns:
        list[dict]: Collated batch with each camera sample treated as a separate sample.
    """
    flattened_batch = []

    # Iterate through each sample in the batch and split it into individual camera samples
    for sample in batch:
        if 'results_cam2' in sample and 'results_cam3' in sample:
            flattened_batch.append(sample['results_cam2'])
            flattened_batch.append(sample['results_cam3'])
        else:
            raise KeyError("Expected each sample to contain 'results_cam2' and 'results_cam3' keys.")

    # Apply mmcv's collate function to the flattened batch
    return collate(flattened_batch, samples_per_gpu=samples_per_gpu)


# Register a custom hook
@HOOKS.register_module()
class CollateFnStereoHook(Hook):
    # """A custom hook to modify the `collate_fn` of the DataLoader before training starts."""
    #
    # def __init__(self, new_collate_fn):
    #     """
    #     Args:
    #         new_collate_fn (function): The custom `collate_fn` to use.
    #     """
    #     self.new_collate_fn = new_collate_fn

    def before_epoch(self, runner):
        """Modify the `collate_fn` of the dataloader before the run starts.

        Args:
            runner (obj:`EpochBasedRunner`): The runner for training/validation.
        """
        # Replace `collate_fn` in both train and val dataloaders
        runner.data_loader.collate_fn = partial(custom_collate_fn, samples_per_gpu=runner.data_loader.batch_size * 2)

    def before_epoch(self, runner):
        """Modify the `collate_fn` of the dataloader before the run starts.

        Args:
            runner (obj:`EpochBasedRunner`): The runner for training/validation.
        """
        # Replace `collate_fn` in both train and val dataloaders
        runner.data_loader.collate_fn = partial(custom_collate_fn, samples_per_gpu=runner.data_loader.batch_size * 2)
