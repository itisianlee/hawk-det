from hawkdet.utils.yacs import CfgNode as CN

C = CN()
C.batch_size = 32
C.max_epoch = 250
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
C.output_path = 'logs'
C.checkpoint_every = 5
C.milestones = [190, 220]  # epoch

C.Dataset = CN()
C.Dataset.name = 'WiderFace'
C.Dataset.image_size = (640, 640)
C.Dataset.label_file = '/root/paddlejob/workspace/hawk-det/widerface/train/label.txt'
C.Dataset.image_mean = (104, 117, 123)  # bgr order
# C.Dataset.image_mean = [104, 111, 120]  # bgr order
C.Dataset.image_std = [13.28474724, 12.9287223, 13.38387188]

C.Detector = CN()
C.Detector.name = 'retinaface'
C.Detector.params = CN()
C.Detector.params.backbone_return_layers = ['stage1', 'stage2', 'stage3']
C.Detector.Backbone = CN()
C.Detector.Backbone.name = 'mobilenetv1_0x25'
C.Detector.Backbone.params = CN()
C.Detector.Backbone.params.pretrained_model = 'model_zoo/mobilenetV1X0.25_pretrain.tar'
# C.Detector.Backbone.params.pretrained = True
C.Detector.Backbone.params.progress = True

C.Detector.Stem = CN()
C.Detector.Stem.name = 'fpn'
C.Detector.Stem.params = CN()
C.Detector.Stem.params.in_channels = 32
C.Detector.Stem.params.out_channels = 64
C.Detector.Stem.params.context_module = 'ssh'

C.Detector.Head = CN()
C.Detector.Head.name = 'retinahead'
C.Detector.Head.params = CN()
C.Detector.Head.params.fpn_num = 3
C.Detector.Head.params.in_channels = C.Detector.Stem.params.out_channels
C.Detector.Head.params.num_anchors = 2

config = C