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

    def forward_for_single_image(self, box_cls, box_regression, img_size):
        h, w = img_size
        C, H, W = box_cls[0].shape

        mlvl_bboxes = []
        mlvl_scores = []

        for cls_score, bbox_pred, stride in zip(
                box_cls, box_regression, self.strides):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, C).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = self.pre_nms_top_n  # cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bboxes = torch.min(bbox_pred.reshape(-1, 2) * stride, bbox_pred.new_tensor(img_size).view(-1, 2))
            bboxes = bboxes.view(-1, 4)[:, (1, 0, 3, 2)]
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)

        bboxes = mlvl_bboxes[:, None].expand(-1, C, 4)
        valid_mask = mlvl_scores > self.pre_nms_thresh
        bboxes = bboxes[valid_mask]
        scores = mlvl_scores
        scores = scores[valid_mask]
        labels = torch.nonzero(valid_mask)[:, 1] + 1

        bboxes[:, 2:] = torch.max(bboxes[:, :2] + 1, bboxes[:, 2:]) # x2y2 >= x1y1 + 1

        boxlist = BoxList(bboxes, (int(w), int(h)), mode="xyxy")
        boxlist.add_field("labels", labels)
        boxlist.add_field("scores", scores)
        boxlist = boxlist.clip_to_image(remove_empty=False)
        # boxlist = remove_small_boxes(boxlist, self.min_size)
        # results.append(boxlist)
        return boxlist


    def forward(self, cls_scores, reg_boxes, image_sizes):
        """
        Arguments:
            cls_scores: list[tensor]
            reg_boxes: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        assert len(cls_scores) == len(reg_boxes)
        num_levels = len(cls_scores)

        sampled_boxes = []
        for img_id in range(len(cls_scores[0])):
            box_cls_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_reg_list = [
                reg_boxes[i][img_id].detach() for i in range(num_levels)
            ]
            det_bboxes = self.forward_for_single_image(
                box_cls_list, bbox_reg_list, image_sizes[img_id])
            sampled_boxes.append(det_bboxes)

        boxlists = sampled_boxes
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