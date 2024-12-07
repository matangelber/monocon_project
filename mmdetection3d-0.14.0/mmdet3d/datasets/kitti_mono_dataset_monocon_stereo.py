import copy
import numpy as np
import mmcv
import cv2
from mmcv.utils import print_log
from copy import deepcopy
from os import path as osp
from mmdet3d.core.visualizer.image_vis import draw_camera_bbox3d_on_img
from .kitti_mono_dataset import KittiMonoDataset
from mmdet.datasets import DATASETS
from ..core import show_multi_modality_result, show_bev_multi_modality_result, show_3d_gt, show_2d_gt, \
    concat_and_show_images, draw_keypoints
from ..core.bbox import CameraInstance3DBoxes, get_box_type, mono_cam_box2vis
from .utils import extract_result_dict, get_loading_pipeline
from create_data_tools_monocon.data_converter.kitti_converter import get_2d_boxes
from ..core import show_bev_stereo_multi_modality_result
from ..core.utils import get_delta_x_meters, get_delta_x_pixels

EPS = 1e-12
INF = 1e10

def transform_alpha_cam2_to_cam3(source_alpha, P_source, P_dest, depth):
    """
    Transforms the alpha angle of an object from camera 2 to camera 3.

    Parameters:
    - source_alpha: The alpha angle in camera 2 (in radians)
    - source_cam: Projection matrix for camera 2 (3x4 numpy array)
    - dest_cam: Projection matrix for camera 3 (3x4 numpy array)
    - depth: Depth of the object in the z-axis (in meters, relative to camera 2)

    Returns:
    - alpha_cam3: Transformed alpha angle in camera 3 (in radians)
    """
    # Extract the translation components from the projection matrices
    t_source = P_source[0, 3] / P_source[0, 0]  # Translation component for camera 2
    t_dest = P_dest[0, 3] / P_dest[0, 0]  # Translation component for camera 3

    # Calculate the change in viewing angle (delta alpha) due to the shift
    delta_alpha = np.arctan((t_dest - t_source) / depth)

    # Adjust the alpha angle with respect to camera 3
    alpha_cam3 = source_alpha
    cond_in_range = np.vectorize(lambda x: (x > -np.pi) & (x < np.pi))
    alpha_cam3[cond_in_range(alpha_cam3)] = (alpha_cam3 + delta_alpha)[cond_in_range(alpha_cam3)]
    return alpha_cam3

def transform_bbox_cam2_to_cam3(bbox, P2, P3, depth):
    """
    Transforms a 2D bounding box from camera 2 to camera 3 in the KITTI dataset.

    Parameters:
    - bbox: List of the 2D bounding box in camera 2 [x_min, y_min, x_max, y_max]
    - P2: Projection matrix for camera 2 (3x4 numpy array)
    - P3: Projection matrix for camera 3 (3x4 numpy array)

    Returns:
    - bbox_cam3: Transformed bounding box in camera 3 [x_min_cam3, y_min, x_max_cam3, y_max]
    """
    # Calculate the horizontal shift in pixels between camera 2 and camera 3
    delta_x = get_delta_x_pixels(P2, P3, depth)
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
            anno_coco_cam3 = [a for a in get_2d_boxes(anno, anno['annos']['occluded'], mono3d=True, proj_matrix='P3') if
                              a is not None]
            annos = anno['annos']
            names = annos['name']
            for k, v in annos.items():
                annos[k] = v[cond_in_set(names)]
            image_id = anno['image']['image_idx']
            names = annos['name']
            names = names[cond_in_set(names)]
            bboxes_cam3 = annos['bbox']
            wh = bboxes_cam3[:, 2:] - bboxes_cam3[:, :2]
            bboxes_cam3[:, 2:] = wh # in coco indices 3,4 are w, h instead of x2, y2 like in KITTI
            for i, coco_anno in enumerate(self.coco.img_ann_map[image_id]):
                coco_anno['file_name_cam3'] = coco_anno['file_name'].replace('image_2', 'image_3')
                keypoints = anno_coco_cam3[i]['keypoints']
                bbox = [min(keypoints[::3]), min(keypoints[1::3]), max(keypoints[::3]) - min(keypoints[::3]),
                        max(keypoints[1::3]) - min(keypoints[1::3])]
                coco_anno['bbox_cam3'] = bbox
                coco_anno['bbox_cam3d_cam3'] = anno_coco_cam3[i]['bbox_cam3d']
                coco_anno['center2d_cam3'] = anno_coco_cam3[i]['center2d']
                coco_anno['keypoints_cam3'] = keypoints
        return

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
                                                                    ann_info['calib']['P2'], ann_info['calib']['P3'],
                                                                     ann_info['annos']['location'][:, 2])
            ann_info['annos']['alpha'] = transform_alpha_cam2_to_cam3(ann_info['annos']['alpha'],
                                                                    ann_info['calib']['P2'], ann_info['calib']['P3'],
                                                                     ann_info['annos']['location'][:, 2])
        return anno_infos

    def _get_data_infos_cam3(self):
        data_infos = deepcopy(self.data_infos)
        cam_intrinsic_cam3_map = {ann_info['image']['image_path'].split('/')[-1] : ann_info['calib']['P3'] for ann_info in self.anno_infos}
        data_infos_cam3 = []
        for data_info in data_infos:
            data_info_cam3 = deepcopy(data_info)
            data_info_cam3['cam_intrinsic'] = cam_intrinsic_cam3_map[data_info_cam3['filename'].split('/')[-1]]
            data_info_cam3['filename'] = data_info_cam3['filename'].replace('image_2', 'image_3')
            data_info_cam3['file_name'] = data_info_cam3['filename']
            data_infos_cam3.append(data_info_cam3)
        return data_infos_cam3

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


    def prepare_test_img(self, idx):
        img_info_left = self.data_infos[idx]
        img_info_right = self.data_infos_cam3[idx]
        results = dict(img_info=img_info_left, img_info_right=img_info_right)
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


    def show(self, max_images_to_show=5, output_dir=None, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert output_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i in range(min(max_images_to_show, len(self.data_infos))):
            data_info = self.data_infos[i]
            img_path = data_info['file_name']
            file_name = osp.split(img_path)[-1].split('.')[0]
            img, cam_intrinsic = self._extract_data(i, pipeline,
                                                ['img', 'cam_intrinsic'])
            img_cam3, cam_intrinsic_cam3 = self._extract_data_cam3(i, pipeline,
                                                    ['img', 'cam_intrinsic'])

            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            # TODO: remove the hack of box from NuScenesMonoDataset
            gt_bboxes = mono_cam_box2vis(gt_bboxes) # local yaw -> global yaw
            gt_bboxes_cam3 = self.get_ann_info_cam3(i)['gt_bboxes_3d']
            # TODO: remove the hack of box from NuScenesMonoDataset
            gt_bboxes_cam3 = mono_cam_box2vis(gt_bboxes_cam3)  # local yaw -> global yaw
            show_img_left = show_3d_gt(
                img,
                gt_bboxes,
                cam_intrinsic,
                output_dir,
                file_name,
                show=show,
                suffix='left')
            show_img_right = show_3d_gt(
                img_cam3,
                gt_bboxes_cam3,
                cam_intrinsic_cam3,
                output_dir,
                file_name,
                show=show,
                suffix='right')
            concat_and_show_images(show_img_left, show_img_right, output_dir, file_name, show, suffix='gt_stereo')



    def show_keypoints(self, max_images_to_show=5, output_dir=None, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert output_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i in range(min(max_images_to_show, len(self.data_infos))):
            data_info = self.data_infos[i]
            img_path = data_info['file_name']
            file_name = osp.split(img_path)[-1].split('.')[0]
            img, cam_intrinsic = self._extract_data(i, pipeline,
                                                ['img', 'cam_intrinsic'])
            img_cam3, cam_intrinsic_cam3 = self._extract_data_cam3(i, pipeline,
                                                    ['img', 'cam_intrinsic'])
            # need to transpose channel to first dim
            # img = img.numpy().transpose(1, 2, 0)
            anno_info = self.get_ann_info(i)
            gt_bboxes = anno_info['gt_bboxes_3d']
            # TODO: remove the hack of box from NuScenesMonoDataset
            gt_bboxes = mono_cam_box2vis(gt_bboxes) # local yaw -> global yaw
            gt_kpts_2d = anno_info['gt_kpts_2d']
            anno_info_cam3 = self.get_ann_info_cam3(i)
            gt_bboxes_cam3 = anno_info_cam3['gt_bboxes_3d']
            # TODO: remove the hack of box from NuScenesMonoDataset
            gt_bboxes_cam3 = mono_cam_box2vis(gt_bboxes_cam3)  # local yaw -> global yaw
            gt_kpts_2d_cam3 = anno_info_cam3['gt_kpts_2d']
            show_img_left = show_3d_gt(
                img=img,
                gt_bboxes=gt_bboxes,
                proj_mat=cam_intrinsic,
                out_dir=None,
                filename=file_name,
                img_metas=None,
                show=False
            )
            show_img_right = show_3d_gt(
                img=img_cam3,
                gt_bboxes=gt_bboxes_cam3,
                proj_mat=cam_intrinsic_cam3,
                out_dir=None,
                filename=file_name,
                img_metas=None,
                show=False
            )

            show_img_left = draw_keypoints(
                show_img_left,
                gt_kpts_2d,
                output_dir,
                file_name,
                show=show,
                suffix='left')

            show_img_right = draw_keypoints(
                show_img_right,
                gt_kpts_2d_cam3,
                output_dir,
                file_name,
                show=show,
                suffix='right')
            concat_and_show_images(show_img_left, show_img_right, output_dir, file_name, show, suffix='keypoints_stereo')



    def show_stereo(self, max_images_to_show=5, output_dir=None, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert output_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i in range(min(max_images_to_show, len(self.data_infos))):
            data_info = self.data_infos[i]
            img_path = data_info['file_name']
            file_name = osp.split(img_path)[-1].split('.')[0]
            img, cam_intrinsic = self._extract_data(i, pipeline,
                                                ['img', 'cam_intrinsic'])
            img_cam3, cam_intrinsic_cam3 = self._extract_data_cam3(i, pipeline,
                                                    ['img', 'cam_intrinsic'])
            concat_and_show_images(img, img_cam3, output_dir, file_name, show, suffix='keypoints_stereo')



    def show_pipline_transform(self, max_images_to_show=5, output_dir=None, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert output_dir is not None, 'Expect out_dir, got none.'
        self.pipeline = self._get_pipeline_for_visualization(pipeline)
        for i in range(min(max_images_to_show, len(self.data_infos))):
            data_info = self.data_infos[i]
            img_path = data_info['file_name']
            file_name = osp.split(img_path)[-1].split('.')[0]
            results = self.prepare_train_img(i)
            for t_i, (t, res) in enumerate(results.items()):
                if 'Bundle' in t:
                    break
                suffix = str(t_i + 1) + '_' + t

                show_img_left = res['results_cam2']['img']
                show_img_right = res['results_cam3']['img']

                show_img_left = self.normalize_to_uint8(show_img_left)
                show_img_right = self.normalize_to_uint8(show_img_right)

                if 'gt_bboxes_3d' in res['results_cam2'].keys():
                    show_img_left = show_2d_gt(
                        img=show_img_left,
                        gt_bboxes=res['results_cam2']['gt_bboxes'],
                        out_dir=None,
                        filename=file_name,
                        show=False
                    )

                    show_img_right = show_2d_gt(
                        img=show_img_right,
                        gt_bboxes=res['results_cam3']['gt_bboxes'],
                        out_dir=None,
                        filename=file_name,
                        show=False
                    )

                    show_img_left = show_3d_gt(
                        img=show_img_left,
                        gt_bboxes=mono_cam_box2vis(res['results_cam2']['gt_bboxes_3d']),
                        proj_mat=res['results_cam2']['cam_intrinsic'],
                        out_dir=None,
                        filename=file_name,
                        show=False
                    )
                    show_img_right = show_3d_gt(
                        img=show_img_right,
                        gt_bboxes=mono_cam_box2vis(res['results_cam3']['gt_bboxes_3d']),
                        proj_mat=res['results_cam3']['cam_intrinsic'],
                        out_dir=None,
                        filename=file_name,
                        img_metas=None,
                        show=False
                    )

                    show_img_left = draw_keypoints(
                        show_img_left,
                        res['results_cam2']['gt_kpts_2d'],
                        output_dir,
                        file_name,
                        show=show,
                        suffix='left')

                    show_img_right = draw_keypoints(
                        show_img_right,
                        res['results_cam3']['gt_kpts_2d'],
                        output_dir,
                        file_name,
                        show=show,
                        suffix='right')
                show_img_left = self.add_text_to_image(show_img_left, f'Left - {str(t_i + 1)}: {t}')
                show_img_right = self.add_text_to_image(show_img_right,f'Right - {str(t_i + 1)}: {t}')


                concat_and_show_images(show_img_left, show_img_right, output_dir, file_name, show, suffix=suffix)



    def _extract_data_cam3(self, index, pipeline, key, load_annos=False):
        """Load data using input pipeline and extract data according to key.

        Args:
            index (int): Index for accessing the target data.
            pipeline (:obj:`Compose`): Composed data loading pipeline.
            key (str | list[str]): One single or a list of data key.
            load_annos (bool): Whether to load data annotations.
                If True, need to set self.test_mode as False before loading.

        Returns:
            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
                A single or a list of loaded data.
        """
        assert pipeline is not None, 'data loading pipeline is not provided'
        img_info = self.data_infos_cam3[index]
        input_dict = dict(img_info=img_info)

        if load_annos:
            ann_info = self.get_ann_info_cam3(index)
            input_dict.update(dict(ann_info=ann_info))

        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)

        # extract data items according to keys
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]

        return data

    def add_text_to_image(self, image, text):
        # Define the text, position, font, scale, color, and thickness
        position = (10, 20)  # (x, y) coordinates for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0)  # Red color in BGR format
        thickness = 1
        # Add text to the image
        cv2.putText(image, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
        return image

    def normalize_to_uint8(self, image):
        image = image - image.min()
        image = (255 * image / image.max()).astype(np.uint8)
        image =  np.ascontiguousarray(image)
        return image


    def show_bev_stereo(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i in range(len(self.data_infos)):
            result_left = results[2*i]
            result_right = results[2*i+1]
            if 'img_bbox' in result_left.keys():
                result_left = result_left['img_bbox']
            if 'img_bbox' in result_right.keys():
                result_right = result_right['img_bbox']
            data_info = self.data_infos[i]
            img_path = data_info['file_name']
            file_name = osp.split(img_path)[-1].split('.')[0]
            img, cam_intrinsic = self._extract_data(i, pipeline,
                                                ['img', 'cam_intrinsic'])
            # need to transpose channel to first dim
            # img = img.numpy().transpose(1, 2, 0)
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            pred_bboxes_left = result_left['boxes_3d']
            pred_bboxes_right = result_right['boxes_3d']
            # TODO: remove the hack of box from NuScenesMonoDataset
            gt_bboxes = mono_cam_box2vis(gt_bboxes)
            # pred_bboxes = mono_cam_box2vis(pred_bboxes)
            show_bev_stereo_multi_modality_result(
                img,
                gt_bboxes,
                pred_bboxes_left,
                pred_bboxes_right,
                cam_intrinsic,
                out_dir,
                file_name,
                box_mode='camera',
                show=show)


    def evaluate_stereo(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, pklfile_prefix)
        from mmdet3d.core.evaluation import kitti_eval
        gt_annos = [info['annos'] for info in self.anno_infos]
        gt_annos_right = [info['annos'] for info in self.anno_infos_cam3]
        gt_annos_all = [gt_annos[i // 2] if i % 2 == 0 else gt_annos_right[i // 2] for i in range(len(results))]
        if isinstance(result_files, dict):
            ap_dict = dict()
            for name, result_files_ in result_files.items():
                eval_types = ['bbox', 'bev', '3d']
                if '2d' in name:
                    eval_types = ['bbox']
                ap_result_str, ap_dict_ = kitti_eval(
                    gt_annos_all,
                    result_files_,
                    self.CLASSES,
                    eval_types=eval_types)
                for ap_type, ap in ap_dict_.items():
                    ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))

                print_log(
                    f'Results of {name}:\n' + ap_result_str, logger=logger)

        else:
            if metric == 'img_bbox2d':
                ap_result_str, ap_dict = kitti_eval(
                    gt_annos, result_files, self.CLASSES, eval_types=['bbox'])
            else:
                ap_result_str, ap_dict = kitti_eval(gt_annos, result_files,
                                                    self.CLASSES)
            print_log('\n' + ap_result_str, logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show:
            self.show(results, out_dir)
        return ap_dict



    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the \
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str | None): The prefix of pkl file.
            submission_prefix (str | None): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.anno_infos)
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.anno_infos[idx]
            sample_idx = info['image']['image_idx']
            image_shape = info['image']['image_shape'][:2]

            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            anno = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': []
            }
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                for box, box_lidar, bbox, score, label in zip(
                        box_preds, box_preds_lidar, box_2d_preds, scores,
                        label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(-np.arctan2(box[0], box[2]) + box[6])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }
                annos.append(anno)

            if submission_prefix is not None:
                curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                bbox[idx][3], dims[idx][1], dims[idx][2],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f)

            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print('Result is saved to %s' % out)

        return det_annos
    #
    # def bbox2result_kitti(self,
    #                       net_outputs,
    #                       class_names,
    #                       pklfile_prefix=None,
    #                       submission_prefix=None):
    #     """Convert 3D detection results to kitti format for evaluation and test
    #     submission.
    #
    #     Args:
    #         net_outputs (list[np.ndarray]): List of array storing the \
    #             inferenced bounding boxes and scores.
    #         class_names (list[String]): A list of class names.
    #         pklfile_prefix (str | None): The prefix of pkl file.
    #         submission_prefix (str | None): The prefix of submission file.
    #
    #     Returns:
    #         list[dict]: A list of dictionaries with the kitti format.
    #     """
    #     assert len(net_outputs) == len(self.anno_infos + self.anno_infos_cam3)
    #     if submission_prefix is not None:
    #         mmcv.mkdir_or_exist(submission_prefix)
    #
    #     det_annos = []
    #     print('\nConverting prediction to KITTI format')
    #     for idx, pred_dicts in enumerate(
    #             mmcv.track_iter_progress(net_outputs)):
    #         annos = []
    #         if idx % 2 == 0:
    #             info = self.anno_infos[idx // 2]
    #         else:
    #             info = self.anno_infos_cam3[idx // 2]
    #         sample_idx = info['image']['image_idx']
    #         image_shape = info['image']['image_shape'][:2]
    #
    #         box_dict = self.convert_valid_bboxes(pred_dicts, info)
    #         anno = {
    #             'name': [],
    #             'truncated': [],
    #             'occluded': [],
    #             'alpha': [],
    #             'bbox': [],
    #             'dimensions': [],
    #             'location': [],
    #             'rotation_y': [],
    #             'score': []
    #         }
    #         if len(box_dict['bbox']) > 0:
    #             box_2d_preds = box_dict['bbox']
    #             box_preds = box_dict['box3d_camera']
    #             scores = box_dict['scores']
    #             box_preds_lidar = box_dict['box3d_lidar']
    #             label_preds = box_dict['label_preds']
    #
    #             for box, box_lidar, bbox, score, label in zip(
    #                     box_preds, box_preds_lidar, box_2d_preds, scores,
    #                     label_preds):
    #                 bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
    #                 bbox[:2] = np.maximum(bbox[:2], [0, 0])
    #                 anno['name'].append(class_names[int(label)])
    #                 anno['truncated'].append(0.0)
    #                 anno['occluded'].append(0)
    #                 anno['alpha'].append(-np.arctan2(box[0], box[2]) + box[6])
    #                 anno['bbox'].append(bbox)
    #                 anno['dimensions'].append(box[3:6])
    #                 anno['location'].append(box[:3])
    #                 anno['rotation_y'].append(box[6])
    #                 anno['score'].append(score)
    #
    #             anno = {k: np.stack(v) for k, v in anno.items()}
    #             annos.append(anno)
    #
    #         else:
    #             anno = {
    #                 'name': np.array([]),
    #                 'truncated': np.array([]),
    #                 'occluded': np.array([]),
    #                 'alpha': np.array([]),
    #                 'bbox': np.zeros([0, 4]),
    #                 'dimensions': np.zeros([0, 3]),
    #                 'location': np.zeros([0, 3]),
    #                 'rotation_y': np.array([]),
    #                 'score': np.array([]),
    #             }
    #             annos.append(anno)
    #
    #         if submission_prefix is not None:
    #             curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
    #             with open(curr_file, 'w') as f:
    #                 bbox = anno['bbox']
    #                 loc = anno['location']
    #                 dims = anno['dimensions']  # lhw -> hwl
    #
    #                 for idx in range(len(bbox)):
    #                     print(
    #                         '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
    #                         '{:.4f} {:.4f} {:.4f} '
    #                         '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
    #                             anno['name'][idx], anno['alpha'][idx],
    #                             bbox[idx][0], bbox[idx][1], bbox[idx][2],
    #                             bbox[idx][3], dims[idx][1], dims[idx][2],
    #                             dims[idx][0], loc[idx][0], loc[idx][1],
    #                             loc[idx][2], anno['rotation_y'][idx],
    #                             anno['score'][idx]),
    #                         file=f)
    #
    #         annos[-1]['sample_idx'] = np.array(
    #             [sample_idx] * len(annos[-1]['score']), dtype=np.int64)
    #
    #         det_annos += annos
    #
    #     if pklfile_prefix is not None:
    #         if not pklfile_prefix.endswith(('.pkl', '.pickle')):
    #             out = f'{pklfile_prefix}.pkl'
    #         mmcv.dump(det_annos, out)
    #         print('Result is saved to %s' % out)
    #
    #     return det_annos


