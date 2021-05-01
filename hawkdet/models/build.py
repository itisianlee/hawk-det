from hawkdet.utils.registry import Registry

detor_registry = Registry('detor')
backbone_registry = Registry('backbone')
stem_registry = Registry('stem')
head_registry = Registry('head')


def build_detor(cfg):
    detector = detor_registry.get(cfg.name)
    backbone = backbone_registry.get(cfg.Backbone.name)(**cfg.Backbone.params)
    stem = stem_registry.get(cfg.Stem.name)(**cfg.Stem.params)
    head = head_registry.get(cfg.Head.name)(**cfg.Head.params)
    return detector(backbone, stem, head, **cfg.params)
