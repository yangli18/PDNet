"""
This file contains specific functions for computing losses on the PDNET
file
"""
import torch
import os
import torch.distributed as dist
from torch.nn import functional as F
from torch import nn
from pdnet_core.layers.sigmoid_focal_loss import sigmoid_focal_loss_cuda, sigmoid_focal_loss_cpu
import numpy as np


INF1 = 100000000
INF2 = 200000000

class PDNetLossComputation(object):
    """
        This class computes the PDNet losses.
        """
    def __init__(self, cfg):
        self.num_gpus = get_num_gpus()

        self.grid_strides = cfg.MODEL.PDNET.GRID_STRIDES # (8, 16, 32, 64, 128)
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.PDNET.LOSS_GAMMA,
            cfg.MODEL.PDNET.LOSS_ALPHA
        )
        self.feat_levels = [np.log2(stride) for stride in self.grid_strides]

        # loss weights
        self.coarse_loss_weight = cfg.MODEL.PDNET.COARSE_LOSS_WEIGHT
        self.reg_loss_weight = cfg.MODEL.PDNET.REG_LOSS_WEIGHT

        # other loss configurations
        self.neg_iou_thres, self.pos_iou_thres = cfg.MODEL.PDNET.LOSS.IOU_THRES
        self.use_poor_matched_gt = cfg.MODEL.PDNET.LOSS.USE_POOR_MATCHED_GT
        self.poor_match_iou_thres = cfg.MODEL.PDNET.LOSS.POOR_MATCH_IOU_THRES

        self.iou_loss_type = cfg.MODEL.PDNET.LOSS.IOU_LOSS_TYPE
        self.iou_loss_discard_abnormal_bbox = cfg.MODEL.PDNET.LOSS.IOU_LOSS_DISCARD_ABNORMAL_BBOX
        self.bbox_iou_discard_abnormal_bbox = cfg.MODEL.PDNET.LOSS.BBOX_IOU_DISCARD_ABNORMAL_BBOX


    def __call__(self, grid_locations, cls_scores, coarse_boxes, reg_boxes, images, targets):

        N = len(targets)
        image_sizes = images.image_sizes
        self.prepare_strides_levels_grids(grid_locations)

        # cls_score_flattened: [N * (H1*W1 + H2*W2...)] * C
        # bboxes_flattened: [N * (H1*W1 + H2*W2...)] * 4
        cls_scores_flattened, coarse_boxes_flattened, reg_boxes_flattened \
            = self.concat_predictions(cls_scores, coarse_boxes, reg_boxes)

        ''' Target assignment '''
        coarse_labels, coarse_reg_targets = \
            self.assign_targets_for_coarse_boxes(targets, image_sizes)  # list[Tensor]

        coarse_labels = torch.cat(coarse_labels, dim=0) # [(N*num_grids)]
        coarse_reg_targets = torch.cat(coarse_reg_targets, dim=0) # [(N*num_grids) * 4]
        coarse_pos_inds = torch.nonzero(coarse_labels > 0).squeeze(1)
        total_coarse_pos_num = self.dist_reduce_sum(coarse_pos_inds.new_tensor([coarse_pos_inds.numel()])).item()
        self.avg_coarse_pos_num = total_coarse_pos_num / self.num_gpus

        labels, reg_targets = \
            self.assign_targets(targets, image_sizes, coarse_boxes_flattened.detach(), 
                                self.pos_iou_thres, self.neg_iou_thres)

        labels = torch.cat(labels, dim=0)
        labels[labels < -1] = -1
        pos_inds = torch.nonzero(labels > 0).squeeze(1)
        reg_targets = torch.cat(reg_targets, dim=0)
        total_pos_num = self.dist_reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        self.avg_pos_num = total_pos_num / self.num_gpus

        ''' Loss computation '''
        strides = self.strides.repeat(N, 1)
        coarse_losses = self.iou_loss(
            coarse_boxes_flattened[coarse_pos_inds] * strides[coarse_pos_inds],
            coarse_reg_targets[coarse_pos_inds],
            weight=None,
        )
        coarse_loss = coarse_losses.sum() / max(1, self.avg_coarse_pos_num) * self.coarse_loss_weight

        reg_losses = self.iou_loss(
            reg_boxes_flattened[pos_inds] * strides[pos_inds],
            reg_targets[pos_inds],
            weight=None,
        )
        reg_loss = reg_losses.sum() / max(1, self.avg_pos_num) * self.reg_loss_weight

        cls_losses = self.cls_loss_func(cls_scores_flattened, labels.int())
        cls_loss = cls_losses.sum() / max(1, self.avg_pos_num)

        losses = {
            "loss_cls": cls_loss,
            "loss_crs": coarse_loss,
            "loss_reg": reg_loss,
        }
        return losses


    def prepare_strides_levels_grids(self, grid_locations):
        strides = []
        levels = []
        for i, grid_location in enumerate(grid_locations):
            strides.append(grid_location.new_full((grid_location.size(0), 1), self.grid_strides[i]))
            levels.append(grid_location.new_full((grid_location.size(0), 1), self.feat_levels[i]).int())  # [3,4,5,6,7]

        self.strides = torch.cat(strides, dim=0)  # [#num_grids, 1]
        self.levels = torch.cat(levels, dim=0)
        self.grids = torch.cat(grid_locations, dim=0)  # [#grids, 2]


    def assign_targets_for_coarse_boxes(self, targets, image_sizes):
        with torch.no_grad():
            grid_labels = []
            reg_targets = []
            for targets_per_img, image_size in zip(targets, image_sizes):
                area = targets_per_img.area()
                assert (targets_per_img.mode == 'xyxy')
                bboxes_per_img = targets_per_img.bbox[:,(1, 0, 3, 2)]  # flip [x,y,x,y] to [y,x,y,x]. y is in front of x
                labels_per_img = targets_per_img.get_field("labels").long()
                reg_targets_per_img = bboxes_per_img[None, :, :].repeat(len(self.grids), 1, 1)

                centers = (bboxes_per_img[:, 0:2] + bboxes_per_img[:, 2:]) / 2  # [num_gtbox, 2]
                bboxes_hw = (bboxes_per_img[:, 2:] - bboxes_per_img[:, 0:2]).clamp(min=0) + 1  # [num_gtbox, 2]
                # ratios_hw = bboxes_hw[:, 0] / bboxes_hw[:, 1] # [num_gtbox], ratio: H/W

                proj_centers = centers[None, :, :] / self.strides[:, None, :]  # num_grids * num_gtbox * 2
                proj_hws = bboxes_hw[None, :, :] / self.strides[:, None, :]
                grid2tgt_yxdiff = self.grids[:, None] - proj_centers
                grid2tgt_norm_yxdiff = grid2tgt_yxdiff / proj_hws
                # grid2tgt_dist = torch.norm(grid2tgt_yxdiff, dim = 2)
                grid2tgt_norm_dist = torch.norm(grid2tgt_norm_yxdiff, dim=2)  # num_grids * num_gtbox

                # positive grid locations
                is_in_pos_grid = grid2tgt_yxdiff.abs().le(0.5).prod(dim=2).byte()  # num_grids * num_gtbox
                bboxes_level = assign_level(area)  # num_gtbox
                is_in_right_level = (bboxes_level[None, :] == self.levels)  # num_grids * num_gtbox
                is_pos = is_in_pos_grid & is_in_right_level  # num_grids * num_gtbox

                grid2tgt_norm_dist = grid2tgt_norm_dist.where(is_pos != 0, grid2tgt_norm_dist.new_tensor([INF2]))

                # choose the closest one as the matched gt  (num_grids * 1)
                grid_to_target_dist, grid_to_target_idx = grid2tgt_norm_dist.min(dim=1)

                # The label of ignored grids is set to -1
                labels_per_img = labels_per_img[grid_to_target_idx]
                labels_per_img[grid_to_target_dist == INF2] = 0  # negative
                labels_per_img[grid_to_target_dist == INF1] = -1  # ignored
                reg_targets_per_img = reg_targets_per_img[range(len(self.grids)), grid_to_target_idx]

                # valid grids (Non-padding image region)
                valid_wh = self.grids < ((self.grids.new_tensor(image_size)[None, :] / 32.).ceil() * 32 / self.strides)
                valid_grids = valid_wh[:, 0] & valid_wh[:, 1]
                labels_per_img[valid_grids == 0] = -1

                grid_labels.append(labels_per_img)
                reg_targets.append(reg_targets_per_img)

            return grid_labels, reg_targets


    def assign_targets(self, targets, image_sizes, bboxes_flattened, pos_iou_thres, neg_iou_thres):
        with torch.no_grad():
            N = len(targets)
            bboxes = bboxes_flattened.view(N, -1)

            labels = []
            reg_targets = []
            strides = self.strides.reshape(-1, 1)
            for bboxes_per_image, targets_per_image, image_size in zip(bboxes, targets, image_sizes):
                assert (targets_per_image.mode == 'xyxy')
                # convert [x,y,x,y] to [y,x,y,x]. y is in front of x
                gt_bboxes_per_img = targets_per_image.bbox[:, (1,0,3,2)]
                gt_labels_per_img = targets_per_image.get_field("labels").long()

                # TODO: set bboxes in the invalid region to zeros
                bboxes_per_image = bboxes_per_image.view(-1, 4) * strides
                ious = self.bbox_IoU(bboxes_per_image, gt_bboxes_per_img) # [#grids, #gt_boxes]

                assign_gt_inds = ious.new_full((ious.shape[0], ), -1, dtype=torch.long)
                labels_per_img = ious.new_full((ious.shape[0], ),  -1, dtype=torch.long)
                reg_targets_per_img = ious.new_zeros((ious.shape[0], 4))

                # assign postive or neg anchors
                max_iou_for_pred, gt_inds = ious.max(dim=1)
                pos = max_iou_for_pred >= pos_iou_thres
                neg = max_iou_for_pred < neg_iou_thres

                assign_gt_inds[pos] = gt_inds[pos] + 1
                assign_gt_inds[neg] = 0

                # low quality matches (assign the bboxes with highest IoU with each gt_bbox)
                max_iou_for_gt, _ = ious.max(dim=0)
                if self.poor_match_iou_thres:
                    matched_pairs = (ious == max_iou_for_gt[None, :]) & \
                                    (max_iou_for_gt[None, :] >= self.poor_match_iou_thres)
                else:
                    matched_pairs = ious == max_iou_for_gt[None, :]
                matched_ious = torch.where(matched_pairs, ious, ious.new_zeros(1, 1))
                pos2 = matched_pairs.nonzero()[:, 0]

                if self.use_poor_matched_gt:
                    matched_gt_inds = matched_ious.argmax(dim=1)
                    assign_gt_inds[pos2] = matched_gt_inds[pos2] + 1
                else:
                    assign_gt_inds[pos2] = gt_inds[pos2] + 1

                #
                pos_inds = torch.nonzero(assign_gt_inds>0).squeeze(1)
                pos_to_gt_inds = assign_gt_inds[pos_inds] - 1
                labels_per_img[neg] = 0
                labels_per_img[pos_inds] = gt_labels_per_img[pos_to_gt_inds]

                # valid grids (non-padding image region)
                valid_wh = self.grids < ((self.grids.new_tensor(image_size)[None, :] / 32.).ceil() * 32 / self.strides)
                valid_grids = valid_wh[:, (0,)] & valid_wh[:, (1,)]
                valid_grids = valid_grids.reshape(-1)
                labels_per_img[~valid_grids] = -2

                reg_targets_per_img[pos_inds, :] = gt_bboxes_per_img[pos_to_gt_inds, :]

                labels.append(labels_per_img)
                reg_targets.append(reg_targets_per_img)

            return labels, reg_targets


    def concat_predictions(self, cls_scores, coarse_bboxes, reg_bboxes):
        cls_scores_flattened = []
        coarse_bboxes_flattened = []
        reg_boxes_flattened = []
        for i in range(len(cls_scores)):
            N, AxC, H, W = cls_scores[i].shape
            Ax4 = coarse_bboxes[i].shape[1]
            A = Ax4 // 4
            C = AxC // A
            cls_scores_flattened.append( cls_scores[i].permute(0,2,3,1).reshape(N, -1 , A, C) )
            coarse_bboxes_flattened.append( coarse_bboxes[i].permute(0,2,3,1).reshape(N, -1, A, 4) )
            reg_boxes_flattened.append( reg_bboxes[i].permute(0,2,3,1).reshape(N, -1, A, 4) )

        cls_scores_flattened = torch.cat(cls_scores_flattened, dim=1).reshape(-1, C)
        coarse_bboxes_flattened = torch.cat(coarse_bboxes_flattened, dim=1).reshape(-1, 4)
        reg_boxes_flattened = torch.cat(reg_boxes_flattened, dim=1 ).reshape(-1,4)

        return cls_scores_flattened, coarse_bboxes_flattened, reg_boxes_flattened


    def bbox_IoU(self, pred_bboxes, gt_bboxes):
        '''
        :param pred_bboxes: Tensor[m, 4]
        :param gt_bboxes: Tensor[n, 4]
        :return:
        '''
        pred_areas = (pred_bboxes[:, 2:] - pred_bboxes[:, :2] + 1).prod(dim=1)
        gt_areas = (gt_bboxes[:, 2:] - gt_bboxes[:, :2] + 1).prod(dim=1)
        yx_tl = torch.max(pred_bboxes[:, None, :2], gt_bboxes[None, :, :2])
        yx_br = torch.min(pred_bboxes[:, None, 2:], gt_bboxes[None, :, 2:])
        intersect_areas = (yx_br - yx_tl + 1).clamp(min=0).prod(dim=2) # [m, n]
        iou = intersect_areas / (pred_areas[:, None] + gt_areas[None, :] - intersect_areas)
        if self.bbox_iou_discard_abnormal_bbox:
            flags = (pred_bboxes[:, :2] > pred_bboxes[:, 2:]).sum(dim=1) > 0
            iou[flags] = -INF2
        return iou


    def dist_reduce_sum(self, tensor):
        if self.num_gpus <= 1:
            return tensor
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor


    def iou_loss(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_right - target_left) * (target_bottom - target_top)
        pred_area = (pred_right - pred_left) * (pred_bottom - pred_top)

        w_intersect = torch.min(pred_right, target_right) - torch.max(pred_left, target_left)
        g_w_intersect = torch.max(pred_right, target_right) - torch.min(pred_left, target_left)
        h_intersect = torch.min(pred_bottom, target_bottom) - torch.max(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) - torch.min(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect.clamp(min=0) * h_intersect.clamp(min=0)
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion

        if self.iou_loss_discard_abnormal_bbox:
            flags = (pred_left > pred_right) | (pred_top > pred_bottom)
            ious[flags] = 0
            gious[flags] = 0

        if self.iou_loss_type == 'giou':
            losses = 1 - gious
        elif self.iou_loss_type == 'iou':
            losses = 1 - ious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return losses * weight
        else:
            return losses


def assign_level(area):
    # [0, 64) --> 3
    # [64, 128) --> 4
    # [128, 256) --> 5
    # [256, 512) --> 6
    # [512, -) --> 7
    lvl = (torch.log2(torch.sqrt(area)) - 2.0).int()
    lvl = torch.clamp(lvl, 3, 7)
    return lvl
    

class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        device = logits.device
        if logits.is_cuda:
            loss_func = sigmoid_focal_loss_cuda
        else:
            loss_func = sigmoid_focal_loss_cpu

        loss = loss_func(logits, targets, self.gamma, self.alpha)
        return loss


def get_num_gpus():
    return int(os.environ['WORLD_SIZE']) if "WORLD_SIZE" in os.environ else 1


def make_pdnet_loss_evaluator(cfg):
    loss_evaluator = PDNetLossComputation(cfg)
    return loss_evaluator