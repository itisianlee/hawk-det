import torch
import torch.nn as nn

from .bbox import BboxHead
from .cls import ClsHead
from .landmark import LandmarkHead
from ..build import head_registry


class Retinahead(nn.Module):
    def __init__(self, fpn_num, in_channels=512, num_anchors=3):
        super().__init__()
        self.bbox = nn.ModuleList([BboxHead(in_channels, num_anchors) for _ in range(fpn_num)])
        self.cls = nn.ModuleList([ClsHead(in_channels, num_anchors) for _ in range(fpn_num)])
        self.lmk = nn.ModuleList([LandmarkHead(in_channels, num_anchors) for _ in range(fpn_num)])

    def forward(self, fpn_out):
        bbox_out = torch.cat([self.bbox[i](feature) for i, feature in enumerate(fpn_out)], dim=1)
        cls_out = torch.cat([self.cls[i](feature) for i, feature in enumerate(fpn_out)], dim=1)
        lmk_out = torch.cat([self.lmk[i](feature) for i, feature in enumerate(fpn_out)], dim=1)
        return bbox_out, cls_out, lmk_out


@head_registry.register()
def retinahead(fpn_num, in_channels=512, num_anchors=3):
    return Retinahead(fpn_num, in_channels, num_anchors)