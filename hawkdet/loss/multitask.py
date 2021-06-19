import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_loss import AnchorLoss
from ..torchlib import calc_iou
from ..torchlib.transform import box_transform, lmk_transform, anchor_transform
from ..torchlib.functions import log_sum_exp


class MultiTask(AnchorLoss):
    def __init__(self, num_classes=2, overlap_threshold=0.35, neg_pos_ratio=7, variance=[0.1, 0.2]):
        super().__init__(num_classes, overlap_threshold, neg_pos_ratio, variance)
    
    def match(self, bboxes, labels, lmks, anchors, loc_t, conf_t, lmk_t, idx):
        # jaccard index
        overlaps = calc_iou(bboxes, anchors)  # (4, anchor_num)
        anchors = anchor_transform(anchors)
        # (Bipartite Matching)
        # [1,num_objects] best anchor for each ground truth
        best_anchor_overlap, best_anchor_idx = overlaps.max(1, keepdim=True)

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

    def forward(self, predictions, anchors, bboxes, labels, lmks):
        loc_preds, conf_preds, lmk_preds = predictions
        batch_size = loc_preds.size(0)
        anchor_num = anchors.size(0)

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(batch_size, anchor_num, 4).to(anchors)
        lmk_t = torch.Tensor(batch_size, anchor_num, 10).to(anchors)
        conf_t = torch.LongTensor(batch_size, anchor_num).to(anchors).to(dtype=torch.long)
        for idx in range(batch_size):
            single_bboxes = bboxes[idx]
            single_labels = labels[idx]
            single_lmks = lmks[idx]
            self.match(single_bboxes, single_labels, single_lmks, anchors, loc_t, conf_t, lmk_t, idx)
        
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos = conf_t > 0
        lmk_preds = lmk_preds[pos].view(-1, 10)
        lmk_t = lmk_t[pos].view(-1, 10)
        loss_lmk = F.smooth_l1_loss(lmk_preds, lmk_t, reduction='mean')

        pos = conf_t != 0
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        loc_preds = loc_preds[pos].view(-1, 4)
        loc_t = loc_t[pos].view(-1, 4)

        loss_l_ = F.smooth_l1_loss(loc_preds, loc_t, reduction='none').sum(dim=1)
        inf_mask = torch.logical_not(torch.isinf(loss_l_))
        loss_l = torch.sum(loss_l_.masked_select(inf_mask))

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_preds.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(batch_size, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(num_pos*self.neg_pos_ratio, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_preds)
        neg_idx = neg.unsqueeze(2).expand_as(conf_preds)
        conf_p = conf_preds[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c, loss_lmk