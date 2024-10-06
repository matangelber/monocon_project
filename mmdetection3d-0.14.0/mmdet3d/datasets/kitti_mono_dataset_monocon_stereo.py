import copy
import numpy as np

from copy import deepcopy
from mmdet.datasets import DATASETS
from ..core.bbox import Box3DMode, CameraInstance3DBoxes
from .kitti_mono_dataset import KittiMonoDataset

EPS = 1e-12
INF = 1e10

def transform_alpha_cam2_to_cam3(alpha_cam2, P2, P3, depth):
    """
    Transforms the alpha angle of an object from camera 2 to camera 3.

    Parameters:
    - alpha_cam2: The alpha angle in camera 2 (in radians)
    - P2: Projection matrix for camera 2 (3x4 numpy array)
    - P3: Projection matrix for camera 3 (3x4 numpy array)
    - depth: Depth of the object in the z-axis (in meters, relative to camera 2)

    Returns:
    - alpha_cam3: Transformed alpha angle in camera 3 (in radians)
    """
    # Extract the translation components from the projection matrices
    t2 = P2[0, 3] / P2[0, 0]  # Translation component for camera 2
    t3 = P3[0, 3] / P3[0, 0]  # Translation component for camera 3

    # Calculate the change in viewing angle (delta alpha) due to the shift
    delta_alpha = np.arctan((t3 - t2) / depth)

    # Adjust the alpha angle with respect to camera 3
    alpha_cam3 = alpha_cam2
    cond_in_range = np.vectorize(lambda x: (x > -np.pi) & (x < np.pi))
    alpha_cam3[cond_in_range(alpha_cam3)] = (alpha_cam3 + delta_alpha)[cond_in_range(alpha_cam3)]
    return alpha_cam3

def get_delta_x(P_source, P_dest):
    # Extract the translation components from the projection matrices
    t_source = P_source[0, 3] / P_source[0, 0]  # Translation component for camera 2
    t_dest = P_dest[0, 3] / P_dest[0, 0]  # Translation component for camera 3

    # Calculate the horizontal shift in pixels between camera 2 and camera 3
    delta_x = t_dest - t_source
    return delta_x


def transform_bbox_cam2_to_cam3(bbox, P2, P3):
    """
    Transforms a 2D bounding box from camera 2 to camera 3 in the KITTI dataset.

    Parameters:
    - bbox: List of the 2D bounding box in camera 2 [x_min, y_min, x_max, y_max]
    - P2: Projection matrix for camera 2 (3x4 numpy array)
    - P3: Projection matrix for camera 3 (3x4 numpy array)

    Returns:
    - bbox_cam3: Transformed bounding box in camera 3 [x_min_cam3, y_min, x_max_cam3, y_max]
    """
    # Extract the translation components from the projection matrices
    t2 = P2[0, 3] / P2[0, 0]  # Translation component for camera 2
    t3 = P3[0, 3] / P3[0, 0]  # Translation component for camera 3

    # Calculate the horizontal shift in pixels between camera 2 and camera 3
    delta_x = t3 - t2
    delta_x = get_delta_x(P2, P3)
    # Transform the bounding box from camera 2 to camera 3 by shifting the x-coordinates
    bbox_cam3 = bbox.copy()
    bbox_cam3[:, 0] += delta_x  # x_min_cam3 = x_min_cam2 + delta_x
    bbox_cam3[:, 2] += delta_x  # x_max_cam3 = x_max_cam2 + delta_x

    return bbox_cam3


@DATASETS.register_module()
class KittiMonoDatasetMonoConStereo(KittiMonoDataset):

    def __init__(self,
                 data_root,
                 info_file,
                 min_height=EPS,
                 min_depth=EPS,
                 max_depth=INF,
                 max_truncation=INF,
                 max_occlusion=INF,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            info_file=info_file,
            **kwargs)

        self.min_height = min_height
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_truncation = max_truncation
        self.max_occlusion = max_occlusion
        self.anno_infos_cam3 = self._get_anno_infos_cam3()
        self.data_infos_cam3 = self._get_data_infos_cam3()
        self.add_cam3_to_coco_annotations()

    def add_cam3_to_coco_annotations(self):
        cond_in_set = np.vectorize(lambda x: x in ['Car', 'Pedestrian', 'Cyclist'])
        for anno in self.anno_infos_cam3:
            image_id = anno['image']['image_idx']
            names = anno['annos']['name']
            names = names[cond_in_set(names)]
            bboxes_cam3 = anno['annos']['bbox']
            P0 = anno['calib']['P0']
            P2 = anno['calib']['P2']
            P3 = anno['calib']['P3']
            x_offset_from_cam0 = get_delta_x(P0, P3)
            x_offset_from_cam2 = get_delta_x(P2, P3)
            # coco_image_anno = self.coco.img_ann_map[image_id]
            for i, coco_anno in enumerate(self.coco.img_ann_map[image_id]):
                coco_anno['file_name_cam3'] = coco_anno['file_name'].replace('image_2', 'image_3')
                coco_anno['bbox_cam3'] = bboxes_cam3[i]
                coco_anno['bbox_cam3d_cam3'] = deepcopy(coco_anno['bbox_cam3d_cam0'])
                coco_anno['bbox_cam3d_cam3'][0] += x_offset_from_cam0
                coco_anno['center2d_cam3'] = deepcopy(coco_anno['center2d'])
                coco_anno['center2d_cam3'][0] = x_offset_from_cam2
                coco_anno['keypoints_cam3'] = deepcopy(coco_anno['keypoints'])
                coco_anno['keypoints_cam3'][::3] += x_offset_from_cam2

    def get_ann_info_cam3(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos_cam3[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        for ann in ann_info:
            cam2_keys_to_change = [k[:-5] for k in list(ann.keys()) if k.endswith('cam3')]
            for k in cam2_keys_to_change:
                ann[k] = ann[k + '_cam3']
        return self._parse_ann_info(self.data_infos_cam3[idx], ann_info)

    def _get_anno_infos_cam3(self):
        anno_infos = deepcopy(self.anno_infos)
        for ann_info in anno_infos:
            ann_info['image']['image_path'] = ann_info['image']['image_path'].replace('image_2', 'image_3')
            ann_info['annos']['bbox'] = transform_bbox_cam2_to_cam3(ann_info['annos']['bbox'],
                                                                    ann_info['calib']['P2'], ann_info['calib']['P3'])
            ann_info['annos']['alpha'] = transform_alpha_cam2_to_cam3(ann_info['annos']['alpha'],
                                                                    ann_info['calib']['P2'], ann_info['calib']['P3'],
                                                                     ann_info['annos']['dimensions'][:, 2])
        return anno_infos

    def _get_data_infos_cam3(self):
        data_infos = deepcopy(self.data_infos)
        cam_intrinsic_cam3_map = {ann_info['image']['image_path'].split('/')[-1] : ann_info['calib']['P3'] for ann_info in self.anno_infos}
        for data_info in data_infos:
            data_info['intrinsic_cam'] = cam_intrinsic_cam3_map[data_info['filename'].split('/')[-1]]
            data_info['filename'] = data_info['filename'].replace('image_2', 'image_3')
            data_info['file_name'] = data_info['filename']
        return data_infos

    def _add_cam3_to_coco_annotations(self):
        pass

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        add filtering mechanism based on occlusion, truncation or depth compared with its superclass

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_bboxes_cam3d = []
        centers2d = []
        depths = []
        gt_kpts_2d = []
        gt_kpts_valid_mask = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            x2 = min(img_info['width'], x1 + w)
            y2 = min(img_info['height'], y1 + h)
            x1 = max(0, x1)
            y1 = max(0, y1)
            bbox = [x1, y1, x2, y2]
            bbox_cam3d = np.array(ann['bbox_cam3d']).reshape(-1, )

            if ann.get('iscrowd', False) or ann['occluded'] > self.max_occlusion \
                    or ann['truncated'] > self.max_truncation or ann['center2d'][2] > self.max_depth or \
                    ann['center2d'][2] < self.min_depth or (y2 - y1) < self.min_height:
                gt_bboxes_ignore.append(bbox)
                continue

            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['category_id']])
            gt_masks_ann.append(ann.get('segmentation', None))
            bbox_cam3d[6] = -np.arctan2(bbox_cam3d[0],
                                        bbox_cam3d[2]) + bbox_cam3d[6]
            gt_bboxes_cam3d.append(bbox_cam3d)
            # 2.5D annotations in camera coordinates
            center2d = ann['center2d'][:2]
            depth = ann['center2d'][2]
            centers2d.append(center2d)
            depths.append(depth)

            # projected keypoints
            kpts_2d = np.array(ann['keypoints']).reshape(-1, 3)
            kpts_valid_mask = kpts_2d[:, 2].astype('int64')
            kpts_2d = kpts_2d[:, :2].astype('float32').reshape(-1)

            gt_kpts_2d.append(kpts_2d)
            gt_kpts_valid_mask.append(kpts_valid_mask)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_kpts_2d = np.array(gt_kpts_2d)
            gt_kpts_valid_mask = np.array(gt_kpts_valid_mask)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_kpts_2d = np.array([], dtype=np.float32)
            gt_kpts_valid_mask = np.array([], dtype=np.int64)

        if gt_bboxes_cam3d:
            gt_bboxes_cam3d = np.array(gt_bboxes_cam3d, dtype=np.float32)
            centers2d = np.array(centers2d, dtype=np.float32)
            depths = np.array(depths, dtype=np.float32)
        else:
            gt_bboxes_cam3d = np.zeros((0, self.bbox_code_size),
                                       dtype=np.float32)
            centers2d = np.zeros((0, 2), dtype=np.float32)
            depths = np.zeros((0), dtype=np.float32)

        gt_bboxes_cam3d = CameraInstance3DBoxes(
            gt_bboxes_cam3d,
            box_dim=gt_bboxes_cam3d.shape[-1],
            origin=(0.5, 0.5, 0.5))
        gt_labels_3d = copy.deepcopy(gt_labels)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_bboxes_3d=gt_bboxes_cam3d,
            gt_labels_3d=gt_labels_3d,
            centers2d=centers2d,
            depths=depths,
            gt_kpts_2d=gt_kpts_2d,
            gt_kpts_valid_mask=gt_kpts_valid_mask,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann


    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info_left = self.data_infos[idx]
        ann_info_left = self.get_ann_info(idx)
        img_info_right = self.data_infos_cam3[idx]
        ann_info_right = self.get_ann_info_cam3(idx)
        results = dict(img_info=img_info_left, ann_info=ann_info_left,
                       img_info_right=img_info_right, ann_info_right=ann_info_right)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)


    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data