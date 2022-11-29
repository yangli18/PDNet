import math
import torch
from torch import nn
from torch.nn import functional as F
from pdnet_core.pred_collect_ops import build_pred_collect_module

class PredDecoupleHead(nn.Module):
    def __init__(self, in_channels, cfg, head_cfg):
        super(PredDecoupleHead, self).__init__()
        dyn_pt_pred_ch = head_cfg.DYNAMIC_POINT_PRED_CHANNELS
        self.dyn_pt_pred = nn.Conv2d(in_channels, dyn_pt_pred_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.dyn_pt_pred_splits = head_cfg.DYNAMIC_POINT_PRED_SPLITS

        self.bound_pt_num = head_cfg.BOUND_POINT_NUM
        self.bound_pt_loc_index = torch.tensor(head_cfg.BOUND_POINT_LOCATION_INDEX).cuda()
        self.limit_bound_pt_offset = head_cfg.LIMIT_BOUND_POINT_OFFSET
        self.limit_bound_pt_offset_ratio = head_cfg.LIMIT_BOUND_POINT_OFFSET_RATIO

        coords_convert_tensor = self.corners_to_cross_shaped_4point()
        self.box_to_bound_pt = torch.nn.Parameter(data=coords_convert_tensor, requires_grad=False)
        self.bound_pt_offset_index = torch.tensor(head_cfg.BOUND_POINT_OFFSET_INDEX).long().cuda()

        coords_convert_tensor = self.corners_to_3x3point()
        self.box_to_semantic_pt = torch.nn.Parameter(data=coords_convert_tensor, requires_grad=False)
        self.semantic_pt_offset_index = torch.tensor(head_cfg.SEMANTIC_POINT_OFFSET_INDEX).long().cuda()

        reg_map_out_ch = head_cfg.REG_MAP_OUT_CHANNELS
        self.reg_map_pred = nn.Conv2d(in_channels, reg_map_out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.reg_pred_collect = build_pred_collect_module(
            True, in_channels=reg_map_out_ch, point_num=self.bound_pt_num,
            batch_proc=False, channels_offset=True, bias=False
        )
        cls_map_out_ch = head_cfg.CLS_MAP_OUT_CHANNELS
        semantic_pt_num = head_cfg.SEMANTIC_POINT_NUM
        self.cls_map_pred = nn.Conv2d(in_channels, cls_map_out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.cls_pred_collect = build_pred_collect_module(
            False, in_channels=cls_map_out_ch, point_num=semantic_pt_num,
            batch_proc=False, channels_offset=True, bias=True
        )

    def corners_to_cross_shaped_4point(self):
        coords_convert_tensor = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0],
                                              [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5],
                                              [0, 0, 1, 0], [0, 0, 0, 1]]).view(6, 4, 1, 1)
        coords_convert_tensor = coords_convert_tensor[[0, 3, 2, 1, 4, 3, 2, 5]]
        return coords_convert_tensor

    def corners_to_3x3point(self):
        s = 1.0
        coords_convert_tensor = torch.tensor([[s, 0, 1 - s, 0], [0, s, 0, 1 - s],
                                              [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5],
                                              [1 - s, 0, s, 0], [0, 1 - s, 0, s]]).view(6, 4, 1, 1)
        coords_convert_tensor = coords_convert_tensor[[0, 1, 0, 3, 0, 5, 2, 1, 2, 3, 2, 5, 4, 1, 4, 3, 4, 5]]
        return coords_convert_tensor

    def forward(self, reg_feats, cls_feats, anchor_point_ls):
        reg_maps = [self.reg_map_pred(x) for x in reg_feats]
        cls_maps = [self.cls_map_pred(x) for x in cls_feats]

        cls_scores = []
        coarse_boxes = []
        reg_boxes = []
        for i, (reg_feat, anchor_point) in enumerate(zip(reg_feats, anchor_point_ls)):
            N, _, H, W = reg_feat.shape
            init_offset, bound_pt_offset, semantic_pt_offset, \
            reg_aggr_attn = self.dyn_pt_pred(reg_feat).split(self.dyn_pt_pred_splits, dim=1)
            coarse_box = anchor_point + init_offset  # [t, l, b, r]

            if self.limit_bound_pt_offset:
                bound_pt_offset = bound_pt_offset.view(N, 2, 2, -1, H, W)
                _wh_thres = (coarse_box[:, [3, 2]] - coarse_box[:, [1, 0]]) / 2 * self.limit_bound_pt_offset_ratio
                bound_pt_offset_extra = (bound_pt_offset - _wh_thres[:, None,:,None]).clamp(min=0) + \
                                        (bound_pt_offset + _wh_thres[:, None,:,None]).clamp(max=0)
                bound_pt_offset = (bound_pt_offset - bound_pt_offset_extra.detach()).view(N, -1, H, W)

            # Dynamic boundary point
            bound_pt = F.conv2d(coarse_box, self.box_to_bound_pt).detach()
            bound_pt = bound_pt.index_add_(1, self.bound_pt_offset_index, bound_pt_offset)
            bound_pt_loc = bound_pt.index_select(1, self.bound_pt_loc_index)

            # Regression prediction collection
            reg_val = self.reg_pred_collect(reg_maps[i], bound_pt)  # [N, #points*1, H, W]
            if i > 0:
                reg_aggr_attn = F.softmax(reg_aggr_attn.view(N, self.bound_pt_num, -1, H, W), dim=2)
                reg_val = reg_val.view(N, self.bound_pt_num, 1, -1, H, W)
                reg_val_low = self.reg_pred_collect(reg_maps[i-1], bound_pt * 2).view(N, self.bound_pt_num, 1, -1, H, W) * 0.5
                reg_val = (torch.cat([reg_val_low, reg_val], dim=2) * reg_aggr_attn[:,:,:,None]).sum(dim=2).view(N, -1, H, W)
            reg_box = (bound_pt_loc.detach() + reg_val).view(N, 4, -1, H, W).mean(dim=2)

            # Dynamic semantic point
            semantic_pt = F.conv2d(coarse_box, self.box_to_semantic_pt).detach()
            semantic_pt = semantic_pt.index_add_(1, self.semantic_pt_offset_index, semantic_pt_offset)

            # Classification prediction collection
            cls_score = self.cls_pred_collect(cls_maps[i], semantic_pt) # [N, #points*C, H, W]

            cls_scores.append(cls_score)
            coarse_boxes.append(coarse_box)
            reg_boxes.append(reg_box)

        return cls_scores, coarse_boxes, reg_boxes