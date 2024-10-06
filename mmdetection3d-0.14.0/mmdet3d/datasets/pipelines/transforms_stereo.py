from copy import deepcopy

import numpy as np
import torch
import warnings
import mmcv
from numpy import random
from mmcv import is_tuple_of
from mmcv.utils import build_from_cfg

from mmdet3d.core import VoxelGenerator
from mmdet3d.core.bbox import box_np_ops, CameraInstance3DBoxes
from mmdet3d.datasets.pipelines.loading import LoadImageFromFileMono3D, LoadAnnotations3DMonoCon
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import RandomFlip, Resize
from mmdet.datasets.pipelines.transforms import PhotoMetricDistortion, Normalize, Pad
from ..builder import OBJECTSAMPLERS
from .data_augment_utils import noise_per_object_v3_
from .transforms_3d import RandomFlipMonoCon, RandomShiftMonoCon
from mmdet3d.datasets.pipelines.formating import DefaultFormatBundle3D, Collect3D

@PIPELINES.register_module()
class LoadImageFromFileMono3DStereo(LoadImageFromFileMono3D):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in \
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        # super().__call__(results)
        # results['cam_intrinsic'] = results['img_info']['cam_intrinsic']
        # return results
        res2 = results
        res3 = deepcopy(res2)
        results = {'results_cam2': res2, 'results_cam3': res3}
        results['results_cam3']['img_info'] = results['results_cam3']['img_info_right']
        results['results_cam3']['ann_info'] = results['results_cam3']['ann_info_right']
        super().__call__(results['results_cam2'])
        super().__call__(results['results_cam3'])
        return results

@PIPELINES.register_module()
class LoadAnnotations3DMonoConStereo(LoadAnnotations3DMonoCon):

    def __call__(self, results):
        results['results_cam2'] = super().__call__(results['results_cam2'])
        results['results_cam3'] = super().__call__(results['results_cam3'])

        if self.with_2D_kpts:
            results['results_cam2'] = self._load_kpts_2d(results['results_cam2'])
            results['results_cam3'] = self._load_kpts_2d(results['results_cam3'])

        return results

@PIPELINES.register_module()
class PhotoMetricDistortionStereo(PhotoMetricDistortion):
    """Photometric distortion for stereo images.

    Applies the same transformations to both left and right images.
    """
    def transform_image(self, img, delta, alpha, mode, saturation, hue, swap_channels):
        """Transform a single image using given parameters."""
        # Apply random brightness
        img += delta

        # Apply random contrast if mode is 0
        if mode == 1 and alpha is not None:
            img *= alpha

        # Convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # Apply random saturation
        if saturation is not None:
            img[..., 1] *= saturation

        # Apply random hue
        if hue is not None:
            img[..., 0] += hue
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # Convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # Apply random contrast if mode is 1
        if mode == 0 and alpha is not None:
            img *= alpha

        # Randomly swap channels
        if swap_channels is not None:
            img = img[..., swap_channels]

        return img


    def __call__(self, results):
        """Call function to perform photometric distortion on stereo images.

        Args:
            results (dict): Result dict from loading pipeline. Expects 'img' to be a list with two images.

        Returns:
            dict: Result dict with stereo images distorted.
        """
        # Extract left and right images
        left_img = results['results_cam2']['img']
        right_img = results['results_cam3']['img']

        # Check if images have the expected dtype
        assert left_img.dtype == np.float32 and right_img.dtype == np.float32, \
            'PhotoMetricDistortionStereo needs the input images of dtype np.float32, '\
            'please set "to_float32=True" in "LoadImageFromFile" pipeline.'

        # Generate random parameters once for both images
        delta = random.uniform(-self.brightness_delta, self.brightness_delta)
        mode = random.randint(2)
        alpha = random.uniform(self.contrast_lower, self.contrast_upper) if random.randint(2) else None
        saturation = random.uniform(self.saturation_lower, self.saturation_upper) if random.randint(2) else None
        hue = random.uniform(-self.hue_delta, self.hue_delta) if random.randint(2) else None
        swap_channels = random.permutation(3)if random.randint(2) else None

        # Apply the same distortion parameters to both images
        left_img = self.transform_image(left_img, delta, alpha, mode, saturation, hue, swap_channels)
        right_img = self.transform_image(right_img, delta, alpha, mode,saturation, hue, swap_channels)

        # Update results dictionary
        results['results_cam2']['img'] = left_img
        results['results_cam3']['img'] = right_img
        return results


@PIPELINES.register_module()
class RandomFlipMonoConStereo(RandomFlipMonoCon):
    """Randomly flip two synchronized inputs, e.g., for stereo images, with the same parameters.

    This class extends RandomFlipMonoCon to apply the same flipping transformations
    to both images and their corresponding 3D bounding boxes.

    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): Probability of horizontal flip. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): Probability of vertical flip (not supported). Defaults to None.
    """

    def __init__(self, **kwargs):
        # Call the parent class initializer to handle the shared parameters.
        super(RandomFlipMonoConStereo, self).__init__(**kwargs)

    def __call__(self, results):
        """Apply flipping to two synchronized image results with the same parameters.

        Args:
            results (dict): The main results dict containing 'results_cam2' and 'results_cam3' as sub-results.

        Returns:
            dict: Updated results with flipped transformations applied to both cameras.
        """
        # Check for the presence of synchronized sub-results for cam2 and cam3
        if 'results_cam2' not in results or 'results_cam3' not in results:
            raise KeyError("Both 'results_cam2' and 'results_cam3' should be present in the input dict.")

        # Use the first camera results to decide the flip parameters
        results_cam2 = results['results_cam2']
        results_cam3 = results['results_cam3']

        # Apply the parent class __call__ to get the flip parameters for results_cam2
        results_cam2 = super(RandomFlipMonoConStereo, self).__call__(results_cam2)

        # Use the same flip parameters for results_cam3
        results_cam3['flip'] = results_cam2['flip']
        results_cam3['flip_direction'] = results_cam2['flip_direction']
        results_cam3['pcd_horizontal_flip'] = results_cam2['pcd_horizontal_flip']
        results_cam3['pcd_vertical_flip'] = results_cam2['pcd_vertical_flip']

        # Apply the flip transformation to results_cam3 using the same parameters
        if results_cam3['pcd_horizontal_flip']:
            self.random_flip_data_3d(results_cam3, 'horizontal')
            results_cam3['transformation_3d_flow'] = results_cam2.get('transformation_3d_flow', []).copy()

        # Update the main results dict
        results['results_cam2'] = results_cam2
        results['results_cam3'] = results_cam3

        return results

    def __repr__(self):
        """str: Return a string representation of the module."""
        repr_str = self.__class__.__name__ + f'(sync_2d={self.sync_2d}, flip_ratio={self.flip_ratio})'
        return repr_str


@PIPELINES.register_module()
class RandomShiftMonoConStereo(RandomShiftMonoCon):
    """Random Shift for Stereo Input.

    Inherits from RandomShiftMonoCon and applies the same shift parameters
    to both `results_cam2` and `results_cam3`.
    """

    def __call__(self, results):
        # Store the shift decision to ensure the same parameters are used
        if random.random() < self.shift_ratio:
            random_shift_x = random.randint(-self.max_shift_px,
                                            self.max_shift_px)
            random_shift_y = random.randint(-self.max_shift_px,
                                            self.max_shift_px)
            results['results_cam2'] = super(RandomShiftMonoConStereo, self).apply_shift(results['results_cam2'],
                                                                                        random_shift_x, random_shift_y)
            results['results_cam3'] = super(RandomShiftMonoConStereo, self).apply_shift(results['results_cam3'],
                                                                                            random_shift_x, random_shift_y)
        return results

@PIPELINES.register_module()
class NormalizeStereo(Normalize):
    def __call__(self, results):
        results['results_cam2'] = super().__call__(results['results_cam2'])
        results['results_cam3'] = super().__call__(results['results_cam3'])
        return results

@PIPELINES.register_module()
class PadStereo(Pad):
    def __call__(self, results):
        results['results_cam2'] = super().__call__(results['results_cam2'])
        results['results_cam3'] = super().__call__(results['results_cam3'])
        return results

@PIPELINES.register_module()
class DefaultFormatBundle3DStereo(DefaultFormatBundle3D):
    def __call__(self, results):
        results['results_cam2'] = super().__call__(results['results_cam2'])
        results['results_cam3'] = super().__call__(results['results_cam3'])
        return results

@PIPELINES.register_module()
class Collect3DStereo(Collect3D):
    def __call__(self, results):
        results['results_cam2'] = super().__call__(results['results_cam2'])
        results['results_cam3'] = super().__call__(results['results_cam3'])
        return results