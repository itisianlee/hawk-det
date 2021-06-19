import cv2
import numpy as np

color_dict = {
    'r': (0, 0, 255),
    'b': (255, 0, 0),
    'g': (0, 255, 0)
}

def bbox_vis(img, bboxes, color='r', thickness=2, format='xywh'):
    assert format in ['xywh', 'xyxy']
    assert color in ['r', 'b', 'g']
    assert isinstance(bboxes, (np.ndarray, list))
    bboxes = np.array(bboxes)
    assert bboxes.ndim == 2 and bboxes.shape[1] == 4
    if format == 'xywh':
        bboxes[:, :2] -= bboxes[:, 2:] / 2
        bboxes[:, 2:] += bboxes[:, :2]
    bboxes = bboxes.astype(np.int)
    
    for b in bboxes:
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color_dict[color], thickness)


def lmk_vis(img, lmks, color='r', radius=5):
    assert color in ['r', 'b', 'g']
    lmks = np.array(lmks, dtype=np.int)
    lmks = np.reshape(lmks, (-1, 2))
    lmks = lmks[lmks[:, 0]>0]
    for l in lmks:
        cv2.circle(img, (l[0], l[1]), radius, color=color_dict[color], thickness=-1)