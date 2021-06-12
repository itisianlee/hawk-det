import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_loss import AnchorLoss
from ..torchlib import calc_iou
from ..torchlib.transform import box_transform, lmk_transform

class MultiTask(AnchorLoss):
    def __init__(self, num_classes=2, overlap_threshold=0.3, pos_neg_ratio=1/3, variance=[0.1, 0.2]):
        super().__init__(num_classes, overlap_threshold, pos_neg_ratio, variance)
    
    def match(self, bboxes, labels, lmks, anchors, loc_t, conf_t, lmk_t, idx):
        # jaccard index
        overlaps = calc_iou(bboxes, anchors)  # (4, anchor_num)
        # (Bipartite Matching)
        # [1,num_objects] best anchor for each ground truth
        best_anchor_overlap, best_anchor_idx = overlaps.max(1, keepdim=True)
        print(best_anchor_overlap)

        # ignore hard gt
        valid_gt_idx = best_anchor_overlap[:, 0] >= 0.2
        best_anchor_idx_filter = best_anchor_idx[valid_gt_idx, :]
        if best_anchor_idx_filter.shape[0] <= 0:
            loc_t[idx] = 0
            conf_t[idx] = 0
            return

        # [1,num_priors] best ground truth for each anchor
        best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
        best_truth_idx.squeeze_(0)
        best_truth_overlap.squeeze_(0)
        best_anchor_idx.squeeze_(1)
        best_anchor_idx_filter.squeeze_(1)
        best_anchor_overlap.squeeze_(1)
        best_truth_overlap.index_fill_(0, best_anchor_idx_filter, 2)  # ensure best anchor
        # TODO refactor: index  best_anchor_idx with long tensor
        # ensure every gt matches with its anchor of max overlap
        for j in range(best_anchor_idx.size(0)):     # 判别此anchor是预测哪一个boxes
            best_truth_idx[best_anchor_idx[j]] = j
        matches = bboxes[best_truth_idx]            # Shape: [num_priors,4] 此处为每一个anchor对应的bbox取出来
        conf = labels[best_truth_idx]               # Shape: [num_priors]      此处为每一个anchor对应的label取出来
        conf[best_truth_overlap < self.overlap_threshold] = 0    # label as background   overlap<0.35的全部作为负样本
        loc = box_transform(matches, anchors, self.variance)

        matches_landm = lmks[best_truth_idx]
        landm = lmk_transform(matches_landm, anchors, self.variance)
        loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
        conf_t[idx] = conf  # [num_priors] top class label for each anchor
        lmk_t[idx] = landm

    def forward(self, predictions, anchors, annotations):
        # loc_preds, conf_preds, lmk_preds = predictions
        # batch_size = loc_preds.size(0)
        # anchor_num = anchors.size(0)

        # bboxes = annotations['bboxes']
        # lmks = annotations['landmarks'] 
        # labels  = annotations['labels']

        # # match priors (default boxes) and ground truth boxes
        # loc_t = torch.Tensor(batch_size, anchor_num, 4)
        # lmk_t = torch.Tensor(batch_size, anchor_num, 10)
        # conf_t = torch.LongTensor(batch_size, anchor_num)
        # for idx in range(batch_size):
        #     single_bboxes = bboxes[idx]
        #     single_lmks = lmks[idx]
        #     single_labels = labels[idx]
        #     self.match(single_bboxes, anchors, single_labels, single_lmks, 
        #           loc_t, 
        #           conf_t, 
        #           lmk_t, 
        #           idx)
