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

# Example of a custom collate function
# def custom_collate_fn(batch, samples_per_gpu=1):
#     """Custom collate function example."""
#     reorder_batch = []
#     for d in batch:
#         reorder_batch += [d['results_cam2'], d['results_cam3']]
#     batch = reorder_batch
#     if not isinstance(batch, Sequence):
#         raise TypeError(f'{type(batch)} is not supported.')
#
#     if isinstance(batch[0], DataContainer):
#         stacked = []
#         if batch[0].cpu_only:
#             for i in range(0, len(batch), samples_per_gpu):
#                 stacked.append(
#                     [sample.data for sample in batch[i:i + samples_per_gpu]])
#             return DataContainer(
#                 stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
#         elif batch[0].stack:
#             for i in range(0, len(batch), samples_per_gpu):
#                 assert isinstance(batch[i].data, torch.Tensor)
#
#                 if batch[i].pad_dims is not None:
#                     ndim = batch[i].dim()
#                     assert ndim > batch[i].pad_dims
#                     max_shape = [0 for _ in range(batch[i].pad_dims)]
#                     for dim in range(1, batch[i].pad_dims + 1):
#                         max_shape[dim - 1] = batch[i].size(-dim)
#                     for sample in batch[i:i + samples_per_gpu]:
#                         for dim in range(0, ndim - batch[i].pad_dims):
#                             assert batch[i].size(dim) == sample.size(dim)
#                         for dim in range(1, batch[i].pad_dims + 1):
#                             max_shape[dim - 1] = max(max_shape[dim - 1],
#                                                      sample.size(-dim))
#                     padded_samples = []
#                     for sample in batch[i:i + samples_per_gpu]:
#                         pad = [0 for _ in range(batch[i].pad_dims * 2)]
#                         for dim in range(1, batch[i].pad_dims + 1):
#                             pad[2 * dim - 1] = max_shape[dim - 1] - sample.size(-dim)
#                         padded_samples.append(F.pad(sample.data, pad, value=sample.padding_value))
#                     stacked.append(default_collate(padded_samples))
#                 elif batch[i].pad_dims is None:
#                     stacked.append(default_collate([sample.data for sample in batch[i:i + samples_per_gpu]]))
#                 else:
#                     raise ValueError('pad_dims should be either None or integers (1-3)')
#
#         else:
#             for i in range(0, len(batch), samples_per_gpu):
#                 stacked.append([sample.data for sample in batch[i:i + samples_per_gpu]])
#         return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
#     elif isinstance(batch[0], Sequence):
#         transposed = zip(*batch)
#         return [custom_collate_fn(samples, samples_per_gpu) for samples in transposed]
#     elif isinstance(batch[0], Mapping):
#         return {key: custom_collate_fn([d[key] for d in batch], samples_per_gpu) for key in batch[0]}
#     else:
#         return default_collate(batch)



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

        # # Modify any validation/test dataloaders if available
        # if hasattr(runner, 'val_dataloader') and runner.val_dataloader is not None:
        #     runner.val_dataloader.collate_fn = self.new_collate_fn
        # if hasattr(runner, 'test_dataloader') and runner.test_dataloader is not None:
        #     runner.test_dataloader.collate_fn = self.new_collate_fn
        #
        # runner.logger.info(f"Custom collate_fn {self.new_collate_fn.__name__} has been set for all dataloaders.")
