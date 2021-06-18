import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorLoss(nn.Module):
    def __init__(self, num_classes=2, overlap_threshold=0.3, neg_pos_ratio=3, variance=[0.1, 0.2]):
        super().__init__()
        self.num_classes = num_classes
        self.overlap_threshold = overlap_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.variance = variance

    def match(self):
        raise NotImplementedError