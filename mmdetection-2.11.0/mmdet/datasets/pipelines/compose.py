import collections
from mmdet3d.core import show_3d_bbox
import glob
import os
from mmcv.utils import build_from_cfg
from copy import deepcopy
from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        if 'img_info' in data.keys():
            img_name = data['img_info']['file_name'].split('.')[0].split('/')[-1]
        else:
            img_name = data['results_cam2']['img_info']['file_name'].split('.')[0].split('/')[-1]
        if img_name == '000003':
            out_dir = '/home/matan/Projects/MonoCon/outputs/consistency_outputs/000003'
            img_dir_name = os.path.join(out_dir, f"img_")
            images_transforms = glob.glob(os.path.join(out_dir, img_dir_name) + '*')
            if len(images_transforms) == 0:
                img_dir_name = f"img_0"
            else:
                numbers = [int(s.split("_")[-1]) for s in images_transforms]
                new_num = max(numbers) + 1
                img_dir_name = f"img_{str(new_num)}"
            out_dir = os.path.join(out_dir, img_dir_name)

        for i, t in enumerate(self.transforms):
            if img_name == '000003':
                if 'results_cam2' in data.keys():
                    img_data = data['results_cam2']
                    if 'gt_bboxes_3d' in img_data.keys():
                        t_name = f"t_{str(i)}"+str(t.__class__).split("'")[-2].split(".")[-1]
                        if "Collect" not in t_name:
                            show_3d_bbox(img_data['img'],
                                         img_data['gt_bboxes_3d'],
                                         img_data['cam_intrinsic'],
                                         out_dir=out_dir,
                                         filename=t_name,
                                         img_metas=None,
                                         show=False,
                                         suffix="",
                                         bbox_type='gt')
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


@PIPELINES.register_module()
class ComposeForVisualization(Compose):
    def __call__(self,data):
        data_pipline = {}
        for t in self.transforms:
            data = t(data)
            data_pipline[t.__class__.__name__] = deepcopy(data)
            if data is None:
                return None
        return data_pipline