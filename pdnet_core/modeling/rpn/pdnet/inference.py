import torch
from pdnet_core.structures.bounding_box import BoxList
from pdnet_core.structures.boxlist_ops import cat_boxlist
from pdnet_core.structures.boxlist_ops import boxlist_ml_nms
from pdnet_core.structures.boxlist_ops import remove_small_boxes


class PDNetPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the PDNet boxes.
    This is only used in the testing.
    """
    def __init__(
            self,
            pre_nms_thresh,
            pre_nms_top_n,
            nms_thresh,
            fpn_post_nms_top_n,
            min_size,
            num_classes,
            cfg
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
        """
        super(PDNetPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.strides = cfg.MODEL.PDNET.GRID_STRIDES

    def forward_for_single_feature_map(self, stride, cls, bbox, image_sizes):
        """
        Arguments:
            cls: tensor of size N, C, H, W
            bbox: tensor of size N, 4, H, W
        """
        N, C, H, W = cls.shape
        cls = cls.permute(0, 2, 3, 1).reshape(N, -1, C).sigmoid()   # [N, (H*W), C]
        bbox = bbox.permute(0, 2, 3, 1).reshape(N, -1, 4)   # [N, (H*W), 4]

        candidate_masks = cls > self.pre_nms_thresh # [N, (H*W), C]
        pre_nms_top_n = candidate_masks.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        results = []
        for cls_per_img, bbox_per_img, \
            candidate_masks_per_img, pre_nms_top_n_per_img, \
            img_size in zip(cls, bbox, candidate_masks, pre_nms_top_n, image_sizes):

            cls_per_img = cls_per_img[candidate_masks_per_img] # candidate_masks_per_img: (H*W) * C
            cls_per_img, inds_top_n = cls_per_img.topk(k=pre_nms_top_n_per_img, sorted=False)
            # get the locations of topk (per_pre_nms_top_n) class predictions from cls_per_img
            # (some predictions may correspond to the same bbox)
            candidate_inds_per_img = torch.nonzero(candidate_masks_per_img)[inds_top_n] # topk * 2
            bbox_loc_per_img = candidate_inds_per_img[:, 0]
            cls_label_per_img = candidate_inds_per_img[:, 1] + 1

            # decode the bbox
            detections = bbox_per_img[bbox_loc_per_img].view(-1, 2, 2)
            detections = torch.min(detections.view(-1, 2) * stride, detections.new_tensor(img_size).view(-1, 2))
            detections = detections.view(-1,4)[:,(1, 0, 3, 2)]  # yxyx to xyxy; convert to image coordinates

            detections[:, 2:] = torch.max(detections[:, :2], detections[:, 2:])  # x2y2 >= x1y1

            h, w = img_size
            boxlist = BoxList(detections, (w, h), mode="xyxy")
            boxlist.add_field("labels", cls_label_per_img)
            boxlist.add_field("scores", cls_per_img)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward(self, cls_scores, reg_boxes, image_sizes):
        """
        Arguments:
            cls_scores: list[tensor]
            reg_boxes: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []

        # process each feat level in turn
        for stride, bbox_cls, bbox_reg in zip(self.strides, cls_scores, reg_boxes):
            sampled_boxes.append(
                self.forward_for_single_feature_map(stride, bbox_cls, bbox_reg, image_sizes)
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists


    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def make_pdnet_postprocessor(config):
    pre_nms_thresh = config.MODEL.PDNET.INFERENCE_TH
    pre_nms_top_n = config.MODEL.PDNET.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.PDNET.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG

    box_selector = PDNetPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.PDNET.NUM_CLASSES,
        cfg=config
    )

    return box_selector