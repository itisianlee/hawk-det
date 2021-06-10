import torch


def box_transform(matched, anchors, variances):
    # dist b/t match center and anchor's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - anchors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * anchors[:, 2:])
    # match wh / anchor wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / anchors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_anchor,4]


def lmk_transform(matched, anchors, variances):
    # dist b/t match center and prior's center
    matched = torch.reshape(matched, (matched.size(0), 5, 2))
    priors_cx = anchors[:, 0].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_cy = anchors[:, 1].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_w = anchors[:, 2].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_h = anchors[:, 3].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    anchors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)
    g_cxcy = matched[:, :, :2] - anchors[:, :, :2]
    # encode variance
    g_cxcy /= (variances[0] * anchors[:, :, 2:])
    # g_cxcy /= anchors[:, :, 2:]
    g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
    # return target for smooth_l1_loss
    return g_cxcy


def box_invtransform(loc, anchors, variances):
    boxes = torch.cat((anchors[:, :2] + loc[:, :2] * variances[0] * anchors[:, 2:], 
                       anchors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def lmk_invtransform(pre, anchors, variances):
    landms = torch.cat((anchors[:, :2] + pre[:, :2] * variances[0] * anchors[:, 2:],
                        anchors[:, :2] + pre[:, 2:4] * variances[0] * anchors[:, 2:],
                        anchors[:, :2] + pre[:, 4:6] * variances[0] * anchors[:, 2:],
                        anchors[:, :2] + pre[:, 6:8] * variances[0] * anchors[:, 2:],
                        anchors[:, :2] + pre[:, 8:10] * variances[0] * anchors[:, 2:],
                        ), dim=1)
    return landms
