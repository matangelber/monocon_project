import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils.gaussian_target import (gaussian_radius, gen_gaussian_target)
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                                transpose_and_gather_feat)
from mmdet3d.ops.attentive_norm import AttnBatchNorm2d
from mmdet3d.core.utils.transforms_3d import pts2Dto3D
from mmdet3d.datasets.kitti_mono_dataset_monocon_stereo import (get_delta_x_pixels, transform_alpha_cam2_to_cam3)
from mmdet3d.core.utils.transforms_3d import get_delta_x_pixels, get_delta_x_meters

INF = 1e8
EPS = 1e-12
PI = np.pi


@HEADS.register_module()
class MonoConHeadStereo(nn.Module):
    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 bbox3d_code_size=7,
                 num_kpt=9,
                 num_alpha_bins=12,
                 max_objs=30,
                 vector_regression_level=1,
                 pred_bbox2d=True,
                 loss_center_heatmap=None,
                 loss_wh=None,
                 loss_offset=None,
                 loss_center2kpt_offset=None,
                 loss_kpt_heatmap=None,
                 loss_kpt_heatmap_offset=None,
                 loss_dim=None,
                 loss_depth=None,
                 loss_alpha_cls=None,
                 loss_alpha_reg=None,
                 use_AN=True,
                 num_AN_affine=10,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(MonoConHeadStereo, self).__init__()
        assert bbox3d_code_size >= 7
        self.num_classes = num_classes
        self.bbox_code_size = bbox3d_code_size
        self.pred_bbox2d = pred_bbox2d
        self.max_objs = max_objs
        self.num_kpt = num_kpt
        self.num_alpha_bins = num_alpha_bins
        self.vector_regression_level = vector_regression_level

        self.use_AN = use_AN
        self.num_AN_affine = num_AN_affine
        self.norm = AttnBatchNorm2d if use_AN else nn.BatchNorm2d

        self.heatmap_head = self._build_head(in_channel, feat_channel, num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)
        self.center2kpt_offset_head = self._build_head(in_channel, feat_channel, self.num_kpt * 2)
        self.kpt_heatmap_head = self._build_head(in_channel, feat_channel, self.num_kpt)
        self.kpt_heatmap_offset_head = self._build_head(in_channel, feat_channel, 2)
        self.dim_head = self._build_head(in_channel, feat_channel, 3)
        self.depth_head = self._build_head(in_channel, feat_channel, 2)
        self._build_dir_head(in_channel, feat_channel)

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)
        self.loss_center2kpt_offset = build_loss(loss_center2kpt_offset)
        self.loss_kpt_heatmap = build_loss(loss_kpt_heatmap)
        self.loss_kpt_heatmap_offset = build_loss(loss_kpt_heatmap_offset)
        self.loss_dim = build_loss(loss_dim)
        if 'Aware' in loss_dim['type']:
            self.dim_aware_in_loss = True
        else:
            self.dim_aware_in_loss = False
        self.loss_depth = build_loss(loss_depth)
        self.loss_alpha_cls = build_loss(loss_alpha_cls)
        self.loss_alpha_reg = build_loss(loss_alpha_reg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def _build_head(self, in_channel, feat_channel, out_channel):
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            self._get_norm_layer(feat_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def _build_dir_head(self, in_channel, feat_channel):
        self.dir_feat = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            self._get_norm_layer(feat_channel),
            nn.ReLU(inplace=True),
        )
        self.dir_cls = nn.Sequential(nn.Conv2d(feat_channel, self.num_alpha_bins, kernel_size=1))
        self.dir_reg = nn.Sequential(nn.Conv2d(feat_channel, self.num_alpha_bins, kernel_size=1))

    def _get_norm_layer(self, feat_channel):
        return self.norm(feat_channel, momentum=0.03, eps=0.001) if not self.use_AN else \
            self.norm(feat_channel, self.num_AN_affine, momentum=0.03, eps=0.001)

    def init_weights(self):
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)  # -2.19
        self.kpt_heatmap_head[-1].bias.data.fill_(bias_init)
        for head in [self.wh_head, self.offset_head, self.center2kpt_offset_head, self.depth_head,
                     self.kpt_heatmap_offset_head, self.dim_head, self.dir_feat,
                     self.dir_cls, self.dir_reg]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        center_heatmap_pred = torch.clamp(center_heatmap_pred, min=1e-4, max=1 - 1e-4)
        kpt_heatmap_pred = self.kpt_heatmap_head(feat).sigmoid()
        kpt_heatmap_pred = torch.clamp(kpt_heatmap_pred, min=1e-4, max=1 - 1e-4)

        offset_pred = self.offset_head(feat)
        kpt_heatmap_offset_pred = self.kpt_heatmap_offset_head(feat)

        wh_pred = self.wh_head(feat)
        center2kpt_offset_pred = self.center2kpt_offset_head(feat)
        dim_pred = self.dim_head(feat)
        depth_pred = self.depth_head(feat)
        depth_pred[:, 0, :, :] = 1. / (depth_pred[:, 0, :, :].sigmoid() + EPS) - 1

        alpha_feat = self.dir_feat(feat)
        alpha_cls_pred = self.dir_cls(alpha_feat)
        alpha_offset_pred = self.dir_reg(alpha_feat)
        return center_heatmap_pred, wh_pred, offset_pred, center2kpt_offset_pred, kpt_heatmap_pred, \
               kpt_heatmap_offset_pred, dim_pred, alpha_cls_pred, alpha_offset_pred, depth_pred

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds', 'center2kpt_offset_preds',
                          'kpt_heatmap_preds', 'kpt_heatmap_offset_preds', 'dim_preds', 'alpha_cls_preds',
                          'alpha_offset_preds', 'depth_preds'))
    def loss(self,
             center_heatmap_preds,
             wh_preds,
             offset_preds,
             center2kpt_offset_preds,
             kpt_heatmap_preds,
             kpt_heatmap_offset_preds,
             dim_preds,
             alpha_cls_preds,
             alpha_offset_preds,
             depth_preds,
             gt_bboxes,
             gt_labels,
             gt_bboxes_3d,
             gt_labels_3d,
             centers2d,
             depths,
             gt_kpts_2d,
             gt_kpts_valid_mask,
             img_metas,
             attr_labels=None,
             proposal_cfg=None,
             gt_bboxes_ignore=None):

        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) \
               == len(center2kpt_offset_preds) == len(kpt_heatmap_preds) == len(kpt_heatmap_offset_preds) \
               == len(dim_preds) == len(alpha_cls_preds) == len(alpha_offset_preds) == 1

        center_heatmap_pred = center_heatmap_preds[0]
        wh_pred = wh_preds[0]
        offset_pred = offset_preds[0]
        center2kpt_offset_pred = center2kpt_offset_preds[0]
        kpt_heatmap_pred = kpt_heatmap_preds[0]
        kpt_heatmap_offset_pred = kpt_heatmap_offset_preds[0]
        dim_pred = dim_preds[0]
        alpha_cls_pred = alpha_cls_preds[0]
        alpha_offset_pred = alpha_offset_preds[0]
        depth_pred = depth_preds[0]
        batch_size = center_heatmap_pred.shape[0]

        target_result = self.get_targets(gt_bboxes, gt_labels,
                                         gt_bboxes_3d,
                                         centers2d,
                                         depths,
                                         gt_kpts_2d,
                                         gt_kpts_valid_mask,
                                         center_heatmap_pred.shape,
                                         img_metas[0]['pad_shape'],
                                         img_metas)

        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        center2kpt_offset_target = target_result['center2kpt_offset_target']
        dim_target = target_result['dim_target']
        depth_target = target_result['depth_target']
        alpha_cls_target = target_result['alpha_cls_target']
        alpha_offset_target = target_result['alpha_offset_target']
        kpt_heatmap_target = target_result['kpt_heatmap_target']
        kpt_heatmap_offset_target = target_result['kpt_heatmap_offset_target']

        indices = target_result['indices']
        indices_kpt = target_result['indices_kpt']

        mask_target = target_result['mask_target']
        mask_center2kpt_offset = target_result['mask_center2kpt_offset']
        mask_kpt_heatmap_offset = target_result['mask_kpt_heatmap_offset']

        # select desired preds and labels based on mask

        ###########################################################################
        indices_stereo = self.get_stereo_indices_2(gt_bboxes, gt_labels,
                             indices,
                             center_heatmap_pred,
                             depths,
                             img_metas)
        mask_stereo = indices_stereo != 0
        center_consistency_heatmap = self.get_center_heatmap(gt_bboxes, gt_labels,
                                                              gt_bboxes_3d,
                                                              center_heatmap_pred,
                                                              centers2d,
                                                              depths,
                                                              img_metas)
        corners_consistency_heatmap = self.get_corners_heatmap(gt_bboxes, gt_labels,
                                                               gt_bboxes_3d,
                                                               kpt_heatmap_pred,
                                                               gt_kpts_2d,
                                                               gt_kpts_valid_mask,
                                                               depths,
                                                               img_metas)

        # if len(gt_bboxes_3d[0]) == len(gt_bboxes_3d[1]) and len(gt_bboxes_3d[2]) == len(gt_bboxes_3d[3]):
        #         a = gt_bboxes_3d[0].tensor - gt_bboxes_3d[1].tensor
        #         b = gt_bboxes_3d[2].tensor - gt_bboxes_3d[3].tensor
        #         if a[:,1:-1].sum() != 0:
        #             print("HERE_A")
        #             print(a)
        #         if b[:,1:-1].sum() != 0:
        #             print("HERE_B")
        #             print(b)
        #         if len(gt_bboxes_3d[0]) != len(gt_bboxes_3d[1]):
        #             print(len(gt_bboxes_3d[0]), len(gt_bboxes_3d[1]))
        #         if len(gt_bboxes_3d[2]) != len(gt_bboxes_3d[3]):
        #             print(len(gt_bboxes_3d[2]), len(gt_bboxes_3d[3]))
        #         print(len(gt_bboxes_3d[0]), len(gt_bboxes_3d[1]))
        #         print(len(gt_bboxes_3d[2]), len(gt_bboxes_3d[3]))
        #         center_consistency_3d_points = self.decode_heatmap_to_3d_pts(gt_bboxes_3d,
        #                                   center_heatmap_pred,
        #                                   center2kpt_offset_pred,
        #                                   depth_pred,
        #                                   img_metas,
        #                                   k=30,
        #                                   kernel=3)
        #         loss_3d_points_consistency = self.smooth_l1_loss_between_consecutive_tensors(center_consistency_3d_points)
        # else:
        #     print("HERE_C")
        # center_consistency_3d_points = self.decode_heatmap_to_3d_pts(gt_bboxes_3d,
        #                                                              center_heatmap_pred,
        #                                                              center2kpt_offset_pred,
        #                                                              depth_pred,
        #                                                              img_metas,
        #                                                              k=30,
        #                                                              kernel=3)
        # loss_3d_points_consistency = self.smooth_l1_loss_between_consecutive_tensors(center_consistency_3d_points)

        #### FOR DEBUG: ###########################################################
        #### FOR DEBUG: ###########################################################
        #### FOR DEBUG: ###########################################################
        from mmdet3d.core import show_multi_modality_result, show_bev_multi_modality_result, show_3d_bbox, concat_and_show_images, draw_keypoints
        #
        def normalize_to_uint8(image):
            image = image - image.min()
            image = (255 * image / image.max()).astype(np.uint8)
            image = np.ascontiguousarray(image)
            return image

        def to_np(t):
            return t.clone().detach().cpu().numpy()

        if img_metas[0]['filename'][-5:] == "10.png":
            depth_pred_np = to_np(depth_pred)
            concat_and_show_images(normalize_to_uint8(depth_pred_np[0][0]), normalize_to_uint8(depth_pred_np[1][0]),
                                   out_dir="/home/matan/Projects/MonoCon/outputs/debug_loss",
                                   filename='depth_pred_0',
                                   show=False, suffix="")

        if img_metas[2]['filename'][-5:] == "10.png":
            depth_pred_np = to_np(depth_pred)
            concat_and_show_images(normalize_to_uint8(depth_pred_np[2][0]), normalize_to_uint8(depth_pred_np[3][0]),
                                   out_dir="/home/matan/Projects/MonoCon/outputs/debug_loss",
                                   filename='depth_pred_0',
                                   show=False, suffix="")
        # a_wh = to_np(wh_target)
        # a_dim = to_np(dim_target)
        # a_offset = to_np(offset_target)
        # a_depth = to_np(depth_target)
        # a_kpt_heatmap = to_np(kpt_heatmap_target)
        # a_center_heatmap = to_np(center_heatmap_target)
        # center_heatmap_pred_local_maxima = to_np(center_heatmap_pred_local_maxima)
        # a_center_consistency_heatmap = to_np(center_consistency_heatmap)
        # a_center_heatmap_pred = to_np(center_heatmap_pred)
        # a_mask = to_np(mask_target)
        # ### center_heatmap_target
        # concat_and_show_images(255 * a_center_heatmap[0][0], 255 * a_center_heatmap[1][0],
        #                        out_dir="/home/matan/Projects/MonoCon/outputs/debug_loss", filename='center_heatmap_0',
        #                        show=False, suffix="")
        # concat_and_show_images(255 * a_center_heatmap[2][0], 255 * a_center_heatmap[3][0],
        #                        out_dir="/home/matan/Projects/MonoCon/outputs/debug_loss", filename='center_heatmap_1',
        #                        show=False, suffix="")
        # concat_and_show_images(255 * np.abs((a_center_heatmap[0][0] - a_center_heatmap[1][0])),
        #                        255 * np.abs((a_center_heatmap[2][0] - a_center_heatmap[3][0])),
        #                        out_dir="/home/matan/Projects/MonoCon/outputs/debug_loss",
        #                        filename='center_heatmap_diff', show=False, suffix="")
        # ### kpts heatmap
        # concat_and_show_images(255 * a_kpt_heatmap[0].sum(0), 255 * a_kpt_heatmap[1].sum(0),
        #                        out_dir="/home/matan/Projects/MonoCon/outputs/debug_loss", filename='kpts_heatmap_0',
        #                        show=False, suffix="")
        # concat_and_show_images(255 * a_kpt_heatmap[2].sum(0), 255 * a_kpt_heatmap[3].sum(0),
        #                        out_dir="/home/matan/Projects/MonoCon/outputs/debug_loss", filename='kpts_heatmap_1',
        #                        show=False, suffix="")
        # concat_and_show_images(255 * np.abs((a_kpt_heatmap[0].sum(0) - a_kpt_heatmap[1].sum(0))),
        #                        255 * np.abs((a_kpt_heatmap[2].sum(0) - a_kpt_heatmap[3].sum(0))),
        #                        out_dir="/home/matan/Projects/MonoCon/outputs/debug_loss",
        #                        filename='kpts_heatmap_diff', show=False, suffix="")
        # ### center_heatmap_pred
        # concat_and_show_images(255 * a_center_heatmap_pred[0][0], 255 * a_center_heatmap_pred[1][0],
        #                        out_dir="/home/matan/Projects/MonoCon/outputs/debug_loss",
        #                        filename='center_heatmap_pred_0',
        #                        show=False, suffix="")
        # concat_and_show_images(255 * a_center_heatmap_pred[2][0], 255 * a_center_heatmap_pred[3][0],
        #                        out_dir="/home/matan/Projects/MonoCon/outputs/debug_loss",
        #                        filename='center_heatmap_pred_1',
        #                        show=False, suffix="")
        # concat_and_show_images(255 * np.abs((a_center_heatmap_pred[0][0] - a_center_heatmap_pred[1][0])),
        #                        255 * np.abs((a_center_heatmap_pred[2][0] - a_center_heatmap_pred[3][0])),
        #                        out_dir="/home/matan/Projects/MonoCon/outputs/debug_loss",
        #                        filename='center_heatmap_pred_diff', show=False, suffix="")
        # ### local maxima pred
        # concat_and_show_images(255 * center_heatmap_pred_local_maxima[0][0], 255 * center_heatmap_pred_local_maxima[1][0],
        #                        out_dir="/home/matan/Projects/MonoCon/outputs/debug_loss",
        #                        filename='center_heatmap_pred_local_maxima_0',
        #                        show=False, suffix="")
        # concat_and_show_images(255 * center_heatmap_pred_local_maxima[2][0], 255 * center_heatmap_pred_local_maxima[3][0],
        #                        out_dir="/home/matan/Projects/MonoCon/outputs/debug_loss",
        #                        filename='center_heatmap_pred_local_maxima_1',
        #                        show=False, suffix="")
        # ### center_heatmap consistency
        # concat_and_show_images(255 * a_center_consistency_heatmap[0][0], 255 * a_center_consistency_heatmap[1][0],
        #                        out_dir="/home/matan/Projects/MonoCon/outputs/debug_loss",
        #                        filename='center_consistency_heatmap_0',
        #                        show=False, suffix="")
        # concat_and_show_images(255 * a_center_consistency_heatmap[2][0], 255 * a_center_consistency_heatmap[3][0],
        #                        out_dir="/home/matan/Projects/MonoCon/outputs/debug_loss",
        #                        filename='center_consistency_heatmap_1',
        #                        show=False, suffix="")
        ###########################################################################
        ###########################################################################

        # 2d offset ToDo: project to correct camoffset_pred_stereo = self.extract_stereo_input_from_tensor(offset_pred, indices, mask_target)
        # offset_target_consistency_stereo = self.extract_stereo_input_and_switch_from_tensor(offset_pred, indices, mask_target)
        # offset_pred_stereo = self.extract_stereo_input_from_tensor(offset_pred, indices, mask_target)
        offset_pred = self.extract_input_from_tensor(offset_pred, indices, mask_target)
        # offset_target_stereo = self.extract_stereo_target_from_tensor(offset_target, mask_target)
        offset_target = self.extract_target_from_tensor(offset_target, mask_target)
        # 2d size
        # wh_target_consistency_stereo = self.extract_stereo_input_and_switch_from_tensor(wh_pred, indices, mask_target)
        # wh_pred_stereo = self.extract_stereo_input_from_tensor(wh_pred, indices, mask_target)
        wh_pred = self.extract_input_from_tensor(wh_pred, indices, mask_target)
        # wh_target_stereo = self.extract_stereo_target_from_tensor(wh_target, mask_target)
        wh_target = self.extract_target_from_tensor(wh_target, mask_target)
        # 3d dim
        dim_pred_consistency = self.extract_stereo_input_from_tensor(dim_pred, indices_stereo, mask_stereo)
        dim_pred_consistency = dim_pred_consistency.reshape(-1, dim_pred_consistency.size(-1))
        dim_target_consistency_stereo = self.extract_stereo_input_and_switch_from_tensor(dim_pred, indices_stereo, mask_stereo)
        dim_target_consistency_stereo = dim_target_consistency_stereo.reshape(-1, dim_target_consistency_stereo.size(-1))
        dim_pred = self.extract_input_from_tensor(dim_pred, indices, mask_target)
        dim_target = self.extract_target_from_tensor(dim_target, mask_target)
        # depth
        depth_pred_consistency = self.extract_stereo_input_from_tensor(depth_pred, indices_stereo, mask_stereo)
        depth_pred_consistency = depth_pred_consistency.reshape(-1, depth_pred_consistency.size(-1))
        depth_target_consistency_stereo = self.extract_stereo_input_and_switch_from_tensor(depth_pred, indices_stereo, mask_stereo)
        depth_target_consistency_stereo = depth_target_consistency_stereo.reshape(-1, depth_target_consistency_stereo.size(-1))
        depth_pred = self.extract_input_from_tensor(depth_pred, indices, mask_target)
        depth_target = self.extract_target_from_tensor(depth_target, mask_target)
        # alpha cls
        alpha_cls_pred = self.extract_input_from_tensor(alpha_cls_pred, indices, mask_target)
        alpha_cls_target = self.extract_target_from_tensor(alpha_cls_target, mask_target).type(torch.long)
        alpha_cls_onehot_target = alpha_cls_target.new_zeros([len(alpha_cls_target), self.num_alpha_bins]).scatter_(
            dim=1, index=alpha_cls_target.view(-1, 1), value=1)
        # alpha offset
        alpha_offset_pred = self.extract_input_from_tensor(alpha_offset_pred, indices, mask_target)
        alpha_offset_pred = torch.sum(alpha_offset_pred * alpha_cls_onehot_target, 1, keepdim=True)
        alpha_offset_target = self.extract_target_from_tensor(alpha_offset_target, mask_target)
        # center2kpt offset
        center2kpt_offset_pred = self.extract_input_from_tensor(center2kpt_offset_pred,
                                                                indices, mask_target)  # B * (num_kpt * 2)
        center2kpt_offset_target = self.extract_target_from_tensor(center2kpt_offset_target, mask_target)
        mask_center2kpt_offset = self.extract_target_from_tensor(mask_center2kpt_offset, mask_target)
        # kpt heatmap offset
        kpt_heatmap_offset_pred = transpose_and_gather_feat(kpt_heatmap_offset_pred, indices_kpt)
        kpt_heatmap_offset_pred = kpt_heatmap_offset_pred.reshape(batch_size, self.max_objs, self.num_kpt * 2)
        kpt_heatmap_offset_pred = kpt_heatmap_offset_pred[mask_target]
        kpt_heatmap_offset_target = kpt_heatmap_offset_target[mask_target]
        mask_kpt_heatmap_offset = self.extract_target_from_tensor(mask_kpt_heatmap_offset, mask_target)



        # calculate loss
        loss_center_heatmap = self.loss_center_heatmap(center_heatmap_pred, center_heatmap_target)
        loss_kpt_heatmap = self.loss_kpt_heatmap(kpt_heatmap_pred, kpt_heatmap_target)

        loss_wh = self.loss_wh(wh_pred, wh_target)
        loss_offset = self.loss_offset(offset_pred, offset_target)
        if self.dim_aware_in_loss:
            loss_dim = self.loss_dim(dim_pred, dim_target, dim_pred)
        else:
            loss_dim = self.loss_dim(dim_pred, dim_target)

        depth_pred, depth_log_variance = depth_pred[:, 0:1], depth_pred[:, 1:2]
        loss_depth = self.loss_depth(depth_pred, depth_target, depth_log_variance)

        center2kpt_offset_pred *= mask_center2kpt_offset
        loss_center2kpt_offset = self.loss_center2kpt_offset(center2kpt_offset_pred, center2kpt_offset_target,
                                                             avg_factor=(mask_center2kpt_offset.sum() + EPS))

        kpt_heatmap_offset_pred *= mask_kpt_heatmap_offset
        loss_kpt_heatmap_offset = self.loss_kpt_heatmap_offset(kpt_heatmap_offset_pred, kpt_heatmap_offset_target,
                                                               avg_factor=(mask_kpt_heatmap_offset.sum() + EPS))

        if mask_target.sum() > 0:
            loss_alpha_cls = self.loss_alpha_cls(alpha_cls_pred, alpha_cls_onehot_target)
        else:
            loss_alpha_cls = 0.0
        loss_alpha_reg = self.loss_alpha_reg(alpha_offset_pred, alpha_offset_target)


        ###############################################################
        #################### calculate stereo loss ####################
        ###############################################################

        # loss_wh_stereo = self.loss_wh(wh_pred, wh_pred_stereo)
        loss_center_heatmap_consistency = self.loss_center_heatmap(center_heatmap_pred, center_consistency_heatmap)
        loss_corners_heatmap_consistency = self.loss_kpt_heatmap(kpt_heatmap_pred, corners_consistency_heatmap)
        # loss_corners_heatmap_consistency = self.loss_kpt_heatmap(kpt_heatmap_pred[:,-1:,:,:], corners_consistency_heatmap[:,-1:,:,:])


        if self.dim_aware_in_loss:
            loss_dim_consistency = self.loss_dim(dim_pred_consistency, dim_target_consistency_stereo, dim_pred_consistency)
        else:
            loss_dim_consistency = self.loss_dim(dim_pred_consistency, dim_target_consistency_stereo)
        #
        depth_pred_consistency, depth_log_variance_consistency = depth_pred_consistency[:, 0:1], depth_pred_consistency[:, 1:2]
        depth_target_consistency_stereo, _ = depth_target_consistency_stereo[:, 0:1], depth_target_consistency_stereo[:,
                                                                                      1:2]
        loss_depth_consistency = self.loss_depth(depth_pred_consistency,
                                            depth_target_consistency_stereo,
                                            depth_log_variance_consistency)
        c = 0.5
        loss_dim_consistency *= c
        loss_depth_consistency *= c
        # loss_3d_points_consistency *= c
        loss_center_heatmap_consistency *= c
        loss_corners_heatmap_consistency *= 0.1
        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset,
            loss_dim=loss_dim,
            loss_center2kpt_offset=loss_center2kpt_offset,
            loss_kpt_heatmap=loss_kpt_heatmap,
            loss_kpt_heatmap_offset=loss_kpt_heatmap_offset,
            loss_alpha_cls=loss_alpha_cls,
            loss_alpha_reg=loss_alpha_reg,
            loss_depth=loss_depth,
            loss_dim_consistency=loss_dim_consistency,
            loss_depth_consistency=loss_depth_consistency,
            # loss_3d_points_consistency=loss_3d_points_consistency,
            loss_3d_heatmap_consistency=loss_center_heatmap_consistency,
            loss_corners_heatmap_consistency=loss_corners_heatmap_consistency

        )

    def get_k_local_maximas(self, center_heatmap_pred):
        center_heatmap_pred_local_maxima = get_local_maximum(
            center_heatmap_pred.clone().detach(), kernel=3)

        *batch_dets, ys, xs = get_topk_from_heatmap(
            center_heatmap_pred_local_maxima, k=30)
        batch_scores, batch_index, batch_topk_labels = batch_dets
        return batch_scores, ys, xs

    def get_center_heatmap(self, gt_bboxes, gt_labels,
                           gt_bboxes_3d,
                           center_heatmap_pred,
                           centers2d,
                           depths,
                           img_metas):
        img_h, img_w = img_metas[0]['pad_shape'][:2]
        bs, _, feat_h, feat_w = center_heatmap_pred.size()

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        calibs = [img_meta['cam_intrinsic'] for img_meta in img_metas]
        # objects as 2D center points
        center_heatmap_target = gt_bboxes[-1].new_zeros([bs, self.num_classes, feat_h, feat_w])
        scores, ys, xs = self.get_k_local_maximas(center_heatmap_pred)
        pred_centers = torch.cat([xs.unsqueeze(-1).float(), ys.unsqueeze(-1).float()], dim=-1)
        for batch_id in range(bs):
            dest_id = batch_id + (-1) ** batch_id
            gt_bbox = gt_bboxes[batch_id]
            if len(gt_bbox) < 1:
                continue
            gt_label = gt_labels[batch_id]
            gt_bbox_3d = gt_bboxes_3d[batch_id]
            if not isinstance(gt_bbox_3d, torch.Tensor):
                gt_bbox_3d = gt_bbox_3d.tensor.to(gt_bbox.device)
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)
            distances = torch.cdist(gt_centers, pred_centers[batch_id])
            closest_indices = torch.argmin(distances, dim=1)
            closest_pred_centers = pred_centers[batch_id][closest_indices]
            for j, ct in enumerate(closest_pred_centers):
                delta_x_pixels = get_delta_x_pixels(P_source=calibs[batch_id], P_dest=calibs[dest_id], depth=depths[batch_id][j])
                ctx_int, cty_int = ct.int()
                ctx_int += torch.round(delta_x_pixels * width_ratio).int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[dest_id, ind],
                                    [ctx_int, cty_int], radius)
        return center_heatmap_target


    def get_corners_heatmap(self, gt_bboxes, gt_labels,
                           gt_bboxes_3d,
                           kpt_heatmap_pred,
                           gt_kpts_2d,
                           gt_kpts_valid_mask,
                           depths,
                           img_metas):
        kpts_depths = [bbox_3d.corners[:, :, -1] if len(bbox_3d) > 0 else None for bbox_3d in gt_bboxes_3d]
        kpts_depths = [torch.cat([dk, d.clone().detach().cpu().unsqueeze(1)], dim=1) if dk is not None else None for
                       dk, d in
                       zip(kpts_depths, depths)]
        img_h, img_w = img_metas[0]['pad_shape'][:2]
        bs, kpts_num, feat_h, feat_w = kpt_heatmap_pred.size()

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        calibs = [img_meta['cam_intrinsic'] for img_meta in img_metas]
        # objects as 2D center points
        kpt_heatmap_target = gt_bboxes[-1].new_zeros([bs, kpts_num, feat_h, feat_w])
        kpts_local_maximas = [self.get_k_local_maximas(kpt_heatmap_pred[:, d, :, :].unsqueeze(1)) for d in range(self.num_kpt)]
        for k in range(self.num_kpt):
            # kpt = kpt_heatmap_pred[k]
            # kptx_int, kpty_int = kpt.int()
            # kptx, kpty = kpt
            # is_kpt_inside_image = (0 <= kptx_int < feat_w) and (0 <= kpty_int < feat_h)
            # if not is_kpt_inside_image:
            #     continue
            scores, ys, xs = kpts_local_maximas[k]
            pred_k_kpts = torch.cat([xs.unsqueeze(-1).float(), ys.unsqueeze(-1).float()], dim=-1)
            gt_kpts_2d = [g.reshape(-1,self.num_kpt,2) for g in gt_kpts_2d]
            for batch_id in range(bs):
                dest_id = batch_id + (-1) ** batch_id
                gt_bbox = gt_bboxes[batch_id]
                if len(gt_bbox) < 1:
                    continue

                # gt_label = gt_labels[batch_id]
                k_kpts = gt_kpts_2d[batch_id][:,k]
                # center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
                # center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
                # gt_centers = torch.cat((center_x, center_y), dim=1)
                distances = torch.cdist(k_kpts, pred_k_kpts[batch_id])
                closest_indices = torch.argmin(distances, dim=1)
                closest_pred_kpts = pred_k_kpts[batch_id][closest_indices]

                for j, ct in enumerate(closest_pred_kpts):
                    delta_x_pixels = get_delta_x_pixels(P_source=calibs[batch_id], P_dest=calibs[dest_id],
                                                        depth=kpts_depths[batch_id][j][k])
                    ctx_int, cty_int = ct.int()
                    ctx_int += torch.round(delta_x_pixels * width_ratio).int()
                    scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                    scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                    radius = gaussian_radius([scale_box_h, scale_box_w],
                                             min_overlap=0.3)
                    radius = max(0, int(radius))
                    gen_gaussian_target(kpt_heatmap_target[dest_id, k],
                                        [ctx_int, cty_int], radius)
        return kpt_heatmap_target

        #     for batch_id in range(bs):
        #         dest_id = batch_id + (-1) ** batch_id
        #         gt_bbox = gt_bboxes[batch_id]
        #         if len(gt_bbox) < 1:
        #             continue
        #         gt_kpt_2d_single = kpt_heatmap_pred[0]
        #
        #     gen_gaussian_target(kpt_heatmap_target[batch_id, k],
        #                         [kptx_int, kpty_int], radius)
        #
        #     kpt_index = kpty_int * feat_w + kptx_int
        #     indices_kpt[batch_id, j, k] = kpt_index
        #
        #     kpt_heatmap_offset_target[batch_id, j, k * 2] = kptx - kptx_int
        #     kpt_heatmap_offset_target[batch_id, j, k * 2 + 1] = kpty - kpty_int
        #     mask_kpt_heatmap_offset[batch_id, j, k * 2:k * 2 + 2] = 1
        #
        #     gt_label = gt_labels[batch_id]
        #     gt_bbox_3d = gt_bboxes_3d[batch_id]
        #     if not isinstance(gt_bbox_3d, torch.Tensor):
        #         gt_bbox_3d = gt_bbox_3d.tensor.to(gt_bbox.device)
        #     center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
        #     center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
        #     gt_centers = torch.cat((center_x, center_y), dim=1)
        #     distances = torch.cdist(gt_centers, pred_centers[batch_id])
        #     closest_indices = torch.argmin(distances, dim=1)
        #     closest_pred_centers = pred_centers[batch_id][closest_indices]
        #     for j, ct in enumerate(closest_pred_centers):
        #         delta_x_pixels = get_delta_x_pixels(P_source=calibs[batch_id], P_dest=calibs[dest_id], depth=depths[batch_id][j])
        #         ctx_int, cty_int = ct.int()
        #         ctx_int += torch.round(delta_x_pixels * width_ratio).int()
        #         ctx, cty = ct
        #         scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
        #         scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
        #         radius = gaussian_radius([scale_box_h, scale_box_w],
        #                                  min_overlap=0.3)
        #         radius = max(0, int(radius))
        #         ind = gt_label[j]
        #         gen_gaussian_target(kpt_heatmap_target[dest_id, ind],
        #                             [ctx_int, cty_int], radius)
        # return kpt_heatmap_target


    def get_stereo_indices(self, gt_bboxes, gt_labels,
                           gt_bboxes_3d,
                           center_heatmap_pred,
                           centers2d,
                           depths,
                           img_metas):
        img_h, img_w = img_metas[0]['pad_shape'][:2]
        bs, _, feat_h, feat_w = center_heatmap_pred.size()

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        calibs = [img_meta['cam_intrinsic'] for img_meta in img_metas]
        # objects as 2D center points
        center_heatmap_target = gt_bboxes[-1].new_zeros([bs, self.num_classes, feat_h, feat_w])
        scores, ys, xs = self.get_k_local_maximas(center_heatmap_pred)
        pred_centers = torch.cat([xs.unsqueeze(-1).float(), ys.unsqueeze(-1).float()], dim=-1)
        for batch_id in range(bs):
            dest_id = batch_id + (-1) ** batch_id
            gt_bbox = gt_bboxes[batch_id]
            if len(gt_bbox) < 1:
                continue
            gt_label = gt_labels[batch_id]
            gt_bbox_3d = gt_bboxes_3d[batch_id]
            if not isinstance(gt_bbox_3d, torch.Tensor):
                gt_bbox_3d = gt_bbox_3d.tensor.to(gt_bbox.device)
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)
            distances = torch.cdist(gt_centers, pred_centers[batch_id])
            closest_indices = torch.argmin(distances, dim=1)
            closest_pred_centers = pred_centers[batch_id][closest_indices]
            for j, ct in enumerate(closest_pred_centers):
                delta_x_pixels = get_delta_x_pixels(P_source=calibs[batch_id], P_dest=calibs[dest_id], depth=depths[batch_id][j])
                ctx_int, cty_int = ct.int()
                ctx_int += torch.round(delta_x_pixels * width_ratio).int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[dest_id, ind],
                                    [ctx_int, cty_int], radius)

        img_h, img_w = img_metas[0]['pad_shape'][:2]
        bs, _, feat_h, feat_w = center_heatmap_pred.size()
        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)
        gt_centers_batch = [torch.cat(
            ((gtbbx[:, [0]] + gtbbx[:, [2]]) * width_ratio / 2, (gtbbx[:, [1]] + gtbbx[:, [3]]) * height_ratio / 2),
            dim=1) for gtbbx in gt_bboxes]
        indices_stereo = gt_bboxes[-1].new_zeros([bs, self.max_objs]).type(torch.cuda.LongTensor)
        for batch_id, gt_centers in enumerate(gt_centers_batch):
            for j, ct in enumerate(gt_centers):
                delta_x_pixels = get_delta_x_pixels(P_source=calibs[batch_id], P_dest=calibs[dest_id],
                                                    depth=depths[batch_id][j])
                ctx_int, cty_int = ct.int()
                indices_stereo[batch_id, j] = cty_int * feat_w + ctx_int
        return center_heatmap_target

    import torch

    def match_stereo_objects(self, indices, centers, calibs, depths, width_ratio,  distance_threshold=100):
        """
        Matches objects between stereo pairs in a batch and filters unmatched objects.

        Parameters:
        - indices: Tensor of shape (batch_size, max_objs) containing object indices.
        - centers: Tensor of shape (batch_size, max_objs, 2) containing object center coordinates (x, y).
        - calibs: List of calibration matrices for each image in the batch.
        - depths: Tensor of shape (batch_size, max_objs) containing depth values for objects.
        - distance_threshold: Maximum allowed pixel distance to consider objects as matched.
        - feat_w: Feature map width, used for index computation.

        Returns:
        - updated_indices: Tensor of shape (batch_size, max_objs) with unmatched indices removed.
        """

        def match_points(points_n, points_m, distance_threshold):
            """
            Matches points from tensor N to tensor M based on minimal distance.
            Parameters:
            - points_n: Tensor of shape (N, 2), coordinates of N points.
            - points_m: Tensor of shape (M, 2), coordinates of M points.
            - distance_threshold: Maximum distance to consider points as matched.
            Returns:
            - matches: Tensor of shape (K, 2) where each row is [n_idx, m_idx].
                       K is the number of valid matches (K <= N).
            """
            # Compute pairwise distances
            distances = torch.cdist(points_n.clone().detach().cpu(), points_m.clone().detach().cpu(), p=2)  # Shape: (N, M)
            # Find the closest point in M for each point in N
            min_distances, min_indices = distances.min(dim=1)  # Shape: (N,), (N,)
            # Apply distance threshold
            valid_matches = min_distances <= distance_threshold
            # Build the matches
            matches = torch.stack(
                [torch.arange(points_n.size(0))[valid_matches], min_indices[valid_matches]],
                dim=1,
            )
            return matches

        bs, max_objs = indices.shape
        stereo_indices = torch.zeros_like(indices)

        for i in range(0, bs, 2):
            if i + 1 >= bs:  # Ensure there's a right image for the left image
                break

            left_indices = indices[i]
            right_indices = indices[i + 1]
            if left_indices[0].item() == 0 or right_indices[0].item() == 0:
                continue
            left_centers = centers[i]
            right_centers = centers[i + 1]
            left_depths = depths[i]
            right_depths = depths[i + 1]
            # Adjust centers in the left image to the right image frame
            adjusted_left_centers = left_centers.clone()

            for j, d in enumerate(left_depths):
                delta_x_pixels = get_delta_x_pixels(
                        P_source=calibs[i], P_dest=calibs[i + 1], depth=d
                    )
                adjusted_left_centers[j, 0] += delta_x_pixels * width_ratio
            match_centers_indices = match_points(adjusted_left_centers, right_centers, distance_threshold)
            for k ,match in enumerate(match_centers_indices):
                stereo_indices[i, k] = left_indices[match[0].item()]
                stereo_indices[i + 1, k] = right_indices[match[1].item()]
            # for j in range(max_objs):
            #     if left_indices[j] == 0:  # Skip empty entries
            #         continue
            #     delta_x_pixels = get_delta_x_pixels(
            #         P_source=calibs[i], P_dest=calibs[i + 1], depth=left_depths[j]
            #     )
            #     adjusted_left_centers[j, 0] += delta_x_pixels * width_ratio
            #
            # # Match objects between left and right images
            # matched_left_indices = torch.zeros_like(left_indices)
            # matched_right_indices = torch.zeros_like(right_indices)
            #
            # for j in range(max_objs):
            #     if left_indices[j] == 0:  # Skip empty entries
            #         break
            #
            #     left_center = adjusted_left_centers[j]
            #     min_distance = float("inf")
            #     match_idx = -1
            #
            #     for k in range(max_objs):
            #         if right_indices[k] == 0:  # Skip empty entries
            #             break
            #
            #         right_center = right_centers[k]
            #         distance = torch.norm(left_center - right_center, p=2).item()
            #
            #         if distance < distance_threshold and distance < min_distance:
            #             min_distance = distance
            #             match_
            #             match_idx = k
            #
            #     if match_idx != -1:
            #         matched_left_indices[j] = left_indices[j]
            #         matched_right_indices[match_idx] = right_indices[match_idx]
            #
            # # Update indices with matches only
            # updated_indices[i] = matched_left_indices
            # updated_indices[i + 1] = matched_right_indices
        a_indices = indices.clone().detach().cpu().numpy()
        a_stereo_indices = stereo_indices.clone().detach().cpu().numpy()
        return stereo_indices



    def get_stereo_indices_2(self, gt_bboxes, gt_labels,
                           indices,
                           center_heatmap_pred,
                           depths,
                           img_metas):
        calibs = [img_meta['cam_intrinsic'] for img_meta in img_metas]
        img_h, img_w = img_metas[0]['pad_shape'][:2]
        bs, _, feat_h, feat_w = center_heatmap_pred.size()

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)
        indices_copy = indices.clone().detach()
        gt_centers_batch = [torch.cat(
            ((gtbbx[:, [0]] + gtbbx[:, [2]]) * width_ratio / 2, (gtbbx[:, [1]] + gtbbx[:, [3]]) * height_ratio / 2),
            dim=1) for gtbbx in gt_bboxes]
        #
        # for batch_id in range(bs):
        #     for j, ind in enumerate(indices_copy):
        #         if ind == 0:
        #             break
        #         cty = ind / feat_w
        #         ctx = ind % feat_w
        stereo_indices = self.match_stereo_objects(indices, gt_centers_batch, calibs, depths, width_ratio,  distance_threshold=100)
        return stereo_indices

    def get_targets(self, gt_bboxes, gt_labels,
                    gt_bboxes_3d,
                    centers2d,
                    depths,
                    gt_kpts_2d,
                    gt_kpts_valid_mask,
                    feat_shape, img_shape,
                    img_metas):
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        calibs = []

        # objects as 2D center points
        center_heatmap_target = gt_bboxes[-1].new_zeros([bs, self.num_classes, feat_h, feat_w])

        # 2D attributes
        wh_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 2])
        offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 2])

        # 3D attributes
        dim_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 3])
        alpha_cls_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])
        alpha_offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])
        depth_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])

        # 2D-3D kpt heatmap and offset
        center2kpt_offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt * 2])
        kpt_heatmap_target = gt_bboxes[-1].new_zeros([bs, self.num_kpt, feat_h, feat_w])
        kpt_heatmap_offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt * 2])

        # indices
        indices = gt_bboxes[-1].new_zeros([bs, self.max_objs]).type(torch.cuda.LongTensor)
        indices_kpt = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt]).type(torch.cuda.LongTensor)

        # masks
        mask_target = gt_bboxes[-1].new_zeros([bs, self.max_objs])
        mask_center2kpt_offset = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt * 2])
        mask_kpt_heatmap_offset = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt * 2])
        for batch_id in range(bs):
            img_meta = img_metas[batch_id]
            cam_p2 = img_meta['cam_intrinsic']

            gt_bbox = gt_bboxes[batch_id]
            calibs.append(cam_p2)
            if len(gt_bbox) < 1:
                continue
            gt_label = gt_labels[batch_id]
            gt_bbox_3d = gt_bboxes_3d[batch_id]
            if not isinstance(gt_bbox_3d, torch.Tensor):
                gt_bbox_3d = gt_bbox_3d.tensor.to(gt_bbox.device)

            depth = depths[batch_id]

            gt_kpt_2d = gt_kpts_2d[batch_id]
            gt_kpt_valid_mask = gt_kpts_valid_mask[batch_id]

            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            gt_kpt_2d = gt_kpt_2d.reshape(-1, self.num_kpt, 2)
            gt_kpt_2d[:, :, 0] *= width_ratio
            gt_kpt_2d[:, :, 1] *= height_ratio

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio

                dim = gt_bbox_3d[j][3: 6]
                alpha = gt_bbox_3d[j][6]
                gt_kpt_2d_single = gt_kpt_2d[j]  # (9, 2)
                gt_kpt_valid_mask_single = gt_kpt_valid_mask[j]  # (9,)

                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                indices[batch_id, j] = cty_int * feat_w + ctx_int

                wh_target[batch_id, j, 0] = scale_box_w
                wh_target[batch_id, j, 1] = scale_box_h
                offset_target[batch_id, j, 0] = ctx - ctx_int
                offset_target[batch_id, j, 1] = cty - cty_int

                dim_target[batch_id, j] = dim
                depth_target[batch_id, j] = depth[j]

                alpha_cls_target[batch_id, j], alpha_offset_target[batch_id, j] = self.angle2class(alpha)

                mask_target[batch_id, j] = 1

                for k in range(self.num_kpt):
                    kpt = gt_kpt_2d_single[k]
                    kptx_int, kpty_int = kpt.int()
                    kptx, kpty = kpt
                    vis_level = gt_kpt_valid_mask_single[k]
                    if vis_level < self.vector_regression_level:
                        continue

                    center2kpt_offset_target[batch_id, j, k * 2] = kptx - ctx_int
                    center2kpt_offset_target[batch_id, j, k * 2 + 1] = kpty - cty_int
                    mask_center2kpt_offset[batch_id, j, k * 2:k * 2 + 2] = 1

                    is_kpt_inside_image = (0 <= kptx_int < feat_w) and (0 <= kpty_int < feat_h)
                    if not is_kpt_inside_image:
                        continue

                    gen_gaussian_target(kpt_heatmap_target[batch_id, k],
                                        [kptx_int, kpty_int], radius)

                    kpt_index = kpty_int * feat_w + kptx_int
                    indices_kpt[batch_id, j, k] = kpt_index

                    kpt_heatmap_offset_target[batch_id, j, k * 2] = kptx - kptx_int
                    kpt_heatmap_offset_target[batch_id, j, k * 2 + 1] = kpty - kpty_int
                    mask_kpt_heatmap_offset[batch_id, j, k * 2:k * 2 + 2] = 1

        indices_kpt = indices_kpt.reshape(bs, -1)
        mask_target = mask_target.type(torch.bool)

        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            center2kpt_offset_target=center2kpt_offset_target,
            dim_target=dim_target,
            depth_target=depth_target,
            alpha_cls_target=alpha_cls_target,
            alpha_offset_target=alpha_offset_target,
            kpt_heatmap_target=kpt_heatmap_target,
            kpt_heatmap_offset_target=kpt_heatmap_offset_target,
            indices=indices,
            indices_kpt=indices_kpt,
            mask_target=mask_target,
            mask_center2kpt_offset=mask_center2kpt_offset,
            mask_kpt_heatmap_offset=mask_kpt_heatmap_offset,
        )

        return target_result

    @staticmethod
    def extract_input_from_tensor(input, ind, mask):
        input = transpose_and_gather_feat(input, ind)
        return input[mask]

    @staticmethod
    def extract_stereo_input_from_tensor(input, ind, mask):
        input = transpose_and_gather_feat(input, ind)
        left_outputs = input[::2, :, :][mask[::2, :]]
        right_outputs = input[1::2,:,:][mask[1::2,:]]
        return torch.stack([left_outputs, right_outputs], dim=0)

    @staticmethod
    def extract_target_from_tensor(target, mask):
        return target[mask]

    @staticmethod
    def extract_stereo_target_from_tensor(target, mask):
        return torch.stack([target[::2,:,:][mask[::2,:]], target[1::2,:,:][mask[1::2,:]]], dim=0)

    @staticmethod
    def extract_stereo_input_and_switch_from_tensor(input, ind, mask):
        input = transpose_and_gather_feat(input, ind)
        batch_size = input.size(0)
        index_tensor = torch.stack((torch.arange(1, batch_size, 2), torch.arange(0, batch_size, 2)), dim=1).view(-1)
        target_stereo_input = input.clone().detach()
        target_stereo_input = target_stereo_input[index_tensor]
        return torch.stack(
            [target_stereo_input[::2, :, :][mask[::2, :]], target_stereo_input[1::2, :, :][mask[1::2, :]]], dim=0)

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class and residual. '''
        angle = angle % (2 * PI)
        assert (angle >= 0 and angle <= 2 * PI)
        angle_per_class = 2 * PI / float(self.num_alpha_bins)
        shifted_angle = (angle + angle_per_class / 2) % (2 * PI)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, cls, residual):
        ''' Inverse function to angle2class. '''
        angle_per_class = 2 * PI / float(self.num_alpha_bins)
        angle_center = cls * angle_per_class
        angle = angle_center + residual
        return angle

    def decode_alpha_multibin(self, alpha_cls, alpha_offset):
        alpha_score, cls = alpha_cls.max(dim=-1)
        cls = cls.unsqueeze(2)
        alpha_offset = alpha_offset.gather(2, cls)
        alpha = self.class2angle(cls, alpha_offset)

        alpha[alpha > PI] = alpha[alpha > PI] - 2 * PI
        alpha[alpha < -PI] = alpha[alpha < -PI] + 2 * PI
        return alpha

    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   center2kpt_offset_preds,
                   kpt_heatmap_preds,
                   kpt_heatmap_offset_preds,
                   dim_preds,
                   alpha_cls_preds,
                   alpha_offset_preds,
                   depth_preds,
                   img_metas,
                   rescale=False):

        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) \
               == len(center2kpt_offset_preds) == len(kpt_heatmap_preds) == len(kpt_heatmap_offset_preds) \
               == len(dim_preds) == len(alpha_cls_preds) == len(alpha_offset_preds) == 1
        scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]
        box_type_3d = img_metas[0]['box_type_3d']

        batch_det_bboxes, batch_det_bboxes_3d, batch_labels = self.decode_heatmap(
            center_heatmap_preds[0],
            wh_preds[0],
            offset_preds[0],
            center2kpt_offset_preds[0],
            kpt_heatmap_preds[0],
            kpt_heatmap_offset_preds[0],
            dim_preds[0],
            alpha_cls_preds[0],
            alpha_offset_preds[0],
            depth_preds[0],
            img_metas,
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel,
            thresh=self.test_cfg.thresh)

        if rescale:
            batch_det_bboxes[..., :4] /= batch_det_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        det_results = [
            [box_type_3d(batch_det_bboxes_3d,
                         box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5)),
             batch_det_bboxes[:, -1],
             batch_labels,
             batch_det_bboxes,
             ]
        ]
        return det_results

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       center2kpt_offset_pred,
                       kpt_heatmap_pred,
                       kpt_heatmap_offset_pred,
                       dim_pred,
                       alpha_cls_pred,
                       alpha_offset_pred,
                       depth_pred,
                       img_metas,
                       k=100,
                       kernel=3,
                       thresh=0.4):
        img_shape = img_metas[0]['pad_shape']
        camera_intrinsic = np.array([img_meta['cam_intrinsic'] for img_meta in img_metas])
        batch, cat, height, width = center_heatmap_pred.shape
        # assert batch == 1
        inp_h, inp_w, _ = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred.clone().detach(), kernel=kernel)

        *batch_dets, ys, xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = xs + offset[..., 0]
        topk_ys = ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)  # (b, k, 5)

        # decode 3D prediction
        dim = transpose_and_gather_feat(dim_pred, batch_index)
        alpha_cls = transpose_and_gather_feat(alpha_cls_pred, batch_index)
        alpha_offset = transpose_and_gather_feat(alpha_offset_pred, batch_index)
        depth_pred = transpose_and_gather_feat(depth_pred, batch_index)
        depth = depth_pred[:, :, 0:1]

        sigma = depth_pred[:, :, 1]
        sigma = torch.exp(-sigma)
        batch_bboxes[..., -1] *= sigma

        center2kpt_offset = transpose_and_gather_feat(center2kpt_offset_pred, batch_index)
        center2kpt_offset = center2kpt_offset.view(batch, k, self.num_kpt * 2)[..., -2:]
        center2kpt_offset[..., ::2] += xs.view(batch, k, 1).expand(batch, k, 1)
        center2kpt_offset[..., 1::2] += ys.view(batch, k, 1).expand(batch, k, 1)

        kpts = center2kpt_offset

        kpts[..., ::2] *= (inp_w / width)
        kpts[..., 1::2] *= (inp_h / height)

        # 1. decode alpha
        alpha = self.decode_alpha_multibin(alpha_cls, alpha_offset)  # (b, k, 1)

        # 1.5 get projected center
        center2d = kpts  # (b, k, 2)

        # 2. recover rotY
        rot_y = torch.stack([self.recover_rotation(k.unsqueeze(0), a.unsqueeze(0), ci) for (k, a, ci) in zip(kpts, alpha, camera_intrinsic)]).squeeze(1)  # (b, k, 3)

        # 2.5 recover box3d_center from center2d and depth
        P = center2d.new_tensor(camera_intrinsic)
        center3d = (depth * (center2d - torch.transpose(P[:, :2, 2:3], 1, 2)) - torch.transpose(P[:, :2, 3:], 1,                                                                                     2)) / P[:, :1, :1]        # 3. compose 3D box
        center3d = torch.cat([center3d, depth], dim=-1)
        batch_bboxes_3d = torch.cat([center3d, dim, rot_y], dim=-1)

        mask = batch_bboxes[..., -1] > thresh
        batch_bboxes = batch_bboxes[mask]
        batch_bboxes_3d = batch_bboxes_3d[mask]
        batch_topk_labels = batch_topk_labels[mask]

        return batch_bboxes, batch_bboxes_3d, batch_topk_labels

    def smooth_l1_loss_between_consecutive_tensors(self, tensor_list):
        """Calculates Smooth L1 loss between consecutive tensors in a list.

        Args:
            tensor_list: A list of tensors, each of shape (N, 3).

        Returns:
            The total Smooth L1 loss.
        """

        loss_fn = nn.SmoothL1Loss(reduction='sum')
        total_loss = 0.0

        for i in range(0, len(tensor_list) - 1, 2):
            t1, t2 = tensor_list[i], tensor_list[i + 1]
            if len(t1) == len(t2) and len(t1) != 0 :
                total_loss += loss_fn(t1[:,:2] / t1[:,-1:], t2[:,:2] / t2[:,-1:])

        return total_loss

    def decode_heatmap_to_3d_pts(self,
                                 gt_bboxes_3d,
                                 center_heatmap_pred,
                                 center2kpt_offset_pred,
                                 depth_pred,
                                 img_metas,
                                 k=100,
                                 kernel=3):

        img_shape = img_metas[0]['pad_shape']
        camera_intrinsic = np.array([img_meta['cam_intrinsic'] for img_meta in img_metas])
        batch, cat, height, width = center_heatmap_pred.shape
        inp_h, inp_w, _ = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred.clone().detach(), kernel=kernel)

        *batch_dets, ys, xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        # decode 3D prediction
        depth_pred = transpose_and_gather_feat(depth_pred, batch_index)
        depth = depth_pred[:, :, 0:1]

        center2kpt_offset = transpose_and_gather_feat(center2kpt_offset_pred, batch_index)
        center2kpt_offset = center2kpt_offset.view(batch, k, self.num_kpt * 2)[..., -2:]
        center2kpt_offset[..., ::2] += xs.view(batch, k, 1).expand(batch, k, 1)
        center2kpt_offset[..., 1::2] += ys.view(batch, k, 1).expand(batch, k, 1)

        kpts = center2kpt_offset

        kpts[..., ::2] *= (inp_w / width)
        kpts[..., 1::2] *= (inp_h / height)

        # 1.5 get projected center
        center2d = kpts  # (b, k, 2)

        # 2.5 recover box3d_center from center2d and depth
        P = center2d.new_tensor(camera_intrinsic)
        center3d = (depth * (center2d - torch.transpose(P[:, :2, 2:3], 1, 2)) - torch.transpose(P[:, :2, 3:], 1,                                                                                     2)) / P[:, :1, :1]        # 3. compose 3D box
        center3d = torch.cat([center3d, depth], dim=-1)
        bs, _, feat_h, feat_w = center_heatmap_pred.size()
        closest_pred_centers_batch = []
        for batch_id in range(batch):
            if len(gt_bboxes_3d[batch_id]) == 0:
                closest_pred_centers_batch.append(torch.empty((0, 3), device=center_heatmap_pred.device, dtype=center_heatmap_pred.dtype))
                continue
            dest_id = batch_id + (-1) ** batch_id
            gt_center_3d = gt_bboxes_3d[batch_id].center.to(center2d.device)
            distances = torch.cdist(gt_center_3d, center3d[batch_id])
            closest_indices = torch.argmin(distances, dim=1)
            closest_pred_centers = center3d[batch_id][closest_indices]
            closest_pred_centers_batch.append(closest_pred_centers)
        # print(len(closest_pred_centers_batch[0]), len(closest_pred_centers_batch[1]))
        # print(len(closest_pred_centers_batch[2]), len(closest_pred_centers_batch[3]))
        return closest_pred_centers_batch


    def recover_rotation(self, kpts, alpha, calib):
        device = kpts.device
        calib = torch.tensor(calib).type(torch.FloatTensor).to(device).unsqueeze(0)

        si = torch.zeros_like(kpts[:, :, 0:1]) + calib[:, 0:1, 0:1]
        rot_y = alpha + torch.atan2(kpts[:, :, 0:1] - calib[:, 0:1, 2:3], si)

        while (rot_y > PI).any():
            rot_y[rot_y > PI] = rot_y[rot_y > PI] - 2 * PI
        while (rot_y < -PI).any():
            rot_y[rot_y < -PI] = rot_y[rot_y < -PI] + 2 * PI

        return rot_y

    @staticmethod
    def _topk_channel(scores, K=40):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        return topk_scores, topk_inds, topk_ys, topk_xs

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      centers2d=None,
                      depths=None,
                      gt_kpts_2d=None,
                      gt_kpts_valid_mask=None,
                      attr_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x)
        assert gt_labels is not None
        assert attr_labels is None
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_bboxes_3d,
                              gt_labels_3d, centers2d, depths, gt_kpts_2d, gt_kpts_valid_mask,
                              img_metas, attr_labels)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        if proposal_cfg is None:
            return losses
        else:
            raise NotImplementedError
