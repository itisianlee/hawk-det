import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_loss import AnchorLoss

class MultiTask(AnchorLoss):
    def __init__(self, num_classes=2, overlap_threshold=0.3, pos_neg_ratio=1/3, variance=[0.1, 0.2]):
        super.__init__(num_classes, overlap_threshold, pos_neg_ratio, variance)
    
    def match(self, bboxes, labels, lmks, anchors, loc_t, conf_t, lmk_t, idx):

    def forward(self, predictions, anchors, annotations):
        loc_preds, conf_preds, lmk_preds = predictions
        batch_size = loc_preds.size(0)
        anchor_num = anchors.size(0)

        bboxes = annotations['bboxes']
        lmks = annotations['landmarks'] 
        labels  = annotations['labels']

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(batch_size, anchor_num, 4)
        lmk_t = torch.Tensor(batch_size, anchor_num, 10)
        conf_t = torch.LongTensor(batch_size, anchor_num)
        for idx in range(batch_size):
            single_bboxes = bboxes[idx]
            single_lmks = lmks[idx]
            single_labels = labels[idx]
            match(self.threshold, 
                  single_bboxes, 
                  anchors,
                  self.variance, 
                  single_labels, 
                  single_lmks, 
                  loc_t, 
                  conf_t, 
                  lmk_t, 
                  idx)
