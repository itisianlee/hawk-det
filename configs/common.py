from hawkdet.utils.yacs import CfgNode as CN

C = CN()
C.batch_size = 24
C.max_epoch = 100
C.num_workers = 4
C.lr = 1e-3
C.weight_decay = 5e-4
C.momentum = 0.9
C.num_classes = 2
C.min_sizes = [[16, 32], [64, 128], [256, 512]]
C.steps = [8, 16, 32]
C.variance = [0.1, 0.2]
C.loc_lambda = 2.0
C.clip = False
C.output_path = 'det-outputs'
C.checkpoint_every = 5
C.milestones = [30, 60, 90]  # epoch

C.Dataset = CN()
C.Dataset.name = 'WiderFace'
C.Dataset.image_size = (840, 840)
C.Dataset.label_file = '/Users/lijianwei04/Desktop/widerface/widerface/train/label.txt'
C.Dataset.image_mean = (104, 117, 123)  # bgr order

C.Detector = CN()
C.Detector.name = 'retinaface'
C.Detector.params = CN()
C.Detector.params.backbone_return_layers = ['layer2', 'layer3', 'layer4']
C.Detector.Backbone = CN()
C.Detector.Backbone.name = 'resnet50'
C.Detector.Backbone.params = CN()
C.Detector.Backbone.params.pretrained = False
C.Detector.Backbone.params.progress = True

C.Detector.Stem = CN()
C.Detector.Stem.name = 'fpn'
C.Detector.Stem.params = CN()
C.Detector.Stem.params.in_channels = 256
C.Detector.Stem.params.out_channels = 256
C.Detector.Stem.params.context_module = 'ssh'

C.Detector.Head = CN()
C.Detector.Head.name = 'retinahead'
C.Detector.Head.params = CN()
C.Detector.Head.params.fpn_num = 3
C.Detector.Head.params.in_channels = C.Detector.Stem.params.out_channels
C.Detector.Head.params.num_anchors = 2

config = C