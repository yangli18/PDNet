import math
import torch
from torch import nn
from .inference import make_pdnet_postprocessor
from .loss import make_pdnet_loss_evaluator
import torch.nn.functional as F

from .pred_modules import PredDecoupleHead


class PDNetHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(PDNetHead, self).__init__()

        reg_conv_tower = []
        cls_conv_tower = []
        for i in range(cfg.MODEL.PDNET.NUM_CONVS):
            reg_conv_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False))
            reg_conv_tower.append(nn.GroupNorm(32, in_channels))
            reg_conv_tower.append(nn.ReLU(inplace=True))

            cls_conv_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False))
            cls_conv_tower.append(nn.GroupNorm(32, in_channels))
            cls_conv_tower.append(nn.ReLU(inplace=True))

        self.add_module('reg_conv_tower', nn.Sequential(*reg_conv_tower))
        self.add_module('cls_conv_tower', nn.Sequential(*cls_conv_tower))

        self.relu = nn.ReLU(inplace=True)
        self.pred_head = PredDecoupleHead(in_channels, cfg, cfg.MODEL.PDNET.PRED_HEAD)

        # Initialization
        det_head_modules = [self.reg_conv_tower, self.cls_conv_tower, self.pred_head]
        self.init_weights(cfg, det_head_modules)

    def init_weights(self, cfg, det_head_modules):
        for modules in det_head_modules:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if hasattr(l, 'bias') and l.bias is not None:
                        torch.nn.init.constant_(l.bias, 0)

        # bias_init
        prior_prob = cfg.MODEL.PDNET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if self.pred_head.cls_pred_collect.bias is not None:
            torch.nn.init.constant_(self.pred_head.cls_pred_collect.bias, bias_value)
        else: raise NotImplementedError

        if hasattr(self.pred_head, 'init_weights'):
            self.pred_head.init_weights()

    def forward(self, x, anchor_points, grid_centers):
        grid_locations = []
        feature_cls_ls = []
        feature_reg_ls = []
        anchor_point_ls = []

        for i, feature in enumerate(x):
            N, _, H, W  = feature.shape

            anchor_point = anchor_points[i][:, 0:H, 0:W].unsqueeze(0)
            feature_reg = self.reg_conv_tower(feature)
            feature_cls = self.cls_conv_tower(feature)

            feature_reg_ls.append(feature_reg)
            feature_cls_ls.append(feature_cls)
            anchor_point_ls.append(anchor_point)
            grid_locations.append(grid_centers[i][0:H, 0:W, :].reshape(H * W, -1))  # (H*W) * 2)

            # [N, #points * 2, H, W]
        cls_scores, coarse_boxes, reg_boxes = \
            self.pred_head(feature_reg_ls, feature_cls_ls, anchor_point_ls)

        return grid_locations, cls_scores, coarse_boxes, reg_boxes


class PDNetModule(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(PDNetModule, self).__init__()
        self.anchor_points, self.grid_centers = self.generate_anchor_points(cfg)
        self.head = PDNetHead(cfg, in_channels)

        # for train
        self.loss_evaluator = make_pdnet_loss_evaluator(cfg)

        # for test
        self.boxes_selector_test = make_pdnet_postprocessor(cfg)
        if cfg.MODEL.PDNET.MMDET_INFERENCE:
            from .mmdet_inference import make_pdnet_postprocessor as make_pdnet_mmdet_postprocessor
            self.boxes_selector_test = make_pdnet_mmdet_postprocessor(cfg)

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        grid_locations, cls_scores, coarse_boxes, reg_boxes = \
            self.head(features, self.anchor_points, self.grid_centers)

        if self.training:
            return self._forward_train(
                grid_locations, cls_scores, coarse_boxes, 
                reg_boxes, images, targets
            )
        else:
            return self._forward_test(
                grid_locations, cls_scores, 
                reg_boxes, images.image_sizes
            )

    def _forward_train(self, grid_locations, cls_scores, coarse_boxes, reg_boxes, images, targets):
        losses = self.loss_evaluator(
            grid_locations, cls_scores, coarse_boxes, reg_boxes, images, targets
        )
        return None, losses

    def _forward_test(self, grid_locations, cls_scores, reg_boxes, image_sizes):
        boxes = self.boxes_selector_test(
            cls_scores, reg_boxes, image_sizes
        )
        return boxes, {}

    def generate_anchor_points(self, cfg):
        anchor_points = []
        grid_centers = []
        point_strides = cfg.MODEL.PDNET.ANCHOR_POINT_STRIDES # [16, 32, 64, 128, 256]
        point_layouts = cfg.MODEL.PDNET.ANCHOR_POINT_LAYOUTS # [[-1, -1], [1, 1]]
        grid_strides = cfg.MODEL.PDNET.GRID_STRIDES # (8, 16, 32, 64, 128)

        max_feat_sizes = [[168, 168], [84, 84], [42, 42], [21, 21], [11, 11]]

        for i, feat_size in enumerate(max_feat_sizes):
            anchor_points_per_featmap = []
            step = point_strides[i] / grid_strides[i]

            y = torch.arange(0, feat_size[0], dtype=torch.float32)
            x = torch.arange(0, feat_size[1], dtype=torch.float32)
            ## 2 * H * W
            grid_center = torch.cat(
                (y.reshape(1,feat_size[0],1).expand(-1,-1,feat_size[1]),
                 x.reshape(1,1,feat_size[1]).expand(-1,feat_size[0],-1)),
                dim = 0
            )

            anchor_point_layouts = torch.tensor(point_layouts, dtype=torch.float32).reshape(1, -1, 2) * step
            anchor_points_per_featmap = (anchor_point_layouts.view(-1, 2, 1, 1) +
                                         grid_center[None]).view(-1, feat_size[0], feat_size[1])

            if torch.cuda.is_available():
                anchor_points_per_featmap = anchor_points_per_featmap.cuda()
                grid_center = grid_center.cuda()
            anchor_points.append(anchor_points_per_featmap)
            grid_centers.append(grid_center.permute(1,2,0)) # H * W * 2

        return anchor_points, grid_centers


def build_pdnet(cfg, in_channels):
    return PDNetModule(cfg, in_channels)