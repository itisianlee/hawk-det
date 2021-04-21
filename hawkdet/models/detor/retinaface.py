import torch
import torch.nn as nn

from ..build import detor_registry
from hawkdet.models.utils import IntermediateLayerGetter


class RetinaFace(nn.Module):
    def __init__(self, backbone, stem, head, backbone_return_layers):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super().__init__()
        # self.phase = phase
        self.backbone = IntermediateLayerGetter(backbone, backbone_return_layers)
        self.stem = stem
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.stem(x)
        x = self.head(x)
        return x


@detor_registry.register()
def retinaface(backbone, stem, head, backbone_return_layers):
    return RetinaFace(backbone, stem, head, backbone_return_layers)