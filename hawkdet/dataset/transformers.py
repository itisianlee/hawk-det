import cv2
import numpy as np
import random

from ..lib.box_utils import matrix_iof


class RandomCrop:
    def __init__(self, image_size=(640, 640), iof_factor=1.0, min_face=16):
        self.image_size = image_size
        self.iof_factor = iof_factor  # iof(IoF(forgrand))
        self.min_face = min_face
        
    def __call__(self, img, bboxes, labels, lmks=None):
        img_h, img_w, _ = img.shape

        for _ in range(250):
            PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
            scale = random.choice(PRE_SCALES)
            short_side = min(img_h, img_w)
            side_len = int(scale * short_side)

            l = np.random.randint(0, img_w-side_len+1)
            t = np.random.randint(0, img_h-side_len+1)
            roi = np.array((l, t, l+side_len, t+side_len))

            value = matrix_iof(bboxes, roi[np.newaxis])
            flag = (value >= self.iof_factor)
            if not flag.any():
                continue

            centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
            bboxes_t = bboxes[mask].copy()
            labels_t = labels[mask].copy()
            lmks_t = lmks[mask].copy()
            lmks_t = lmks_t.reshape([-1, 5, 2])

            if bboxes_t.shape[0] == 0:
                continue
        
            img_t = img[roi[1]:roi[3], roi[0]:roi[2]]

            bboxes_t[:, :2] = np.maximum(bboxes_t[:, :2], roi[:2])
            bboxes_t[:, :2] -= roi[:2]
            bboxes_t[:, 2:] = np.minimum(bboxes_t[:, 2:], roi[2:])
            bboxes_t[:, 2:] -= roi[:2]

            # landm
            lmks_t[:, :, :2] = lmks_t[:, :, :2] - roi[:2]
            lmks_t[:, :, :2] = np.maximum(lmks_t[:, :, :2], np.array([0, 0]))
            lmks_t[:, :, :2] = np.minimum(lmks_t[:, :, :2], roi[2:] - roi[:2])
            lmks_t = lmks_t.reshape([-1, 10])

            # make sure that the cropped image contains at least one face > 16 pixel at training image scale
            b_w_t = (bboxes_t[:, 2] - bboxes_t[:, 0] + 1) / side_len * self.image_size[0]
            b_h_t = (bboxes_t[:, 3] - bboxes_t[:, 1] + 1) / side_len * self.image_size[1]
            mask = np.minimum(b_w_t, b_h_t) > self.min_face
            bboxes_t = bboxes_t[mask]
            labels_t = labels_t[mask]
            lmks_t = lmks_t[mask]

            if bboxes_t.shape[0] == 0:
                continue

            return img_t, bboxes_t, labels_t, lmks_t
        return img, bboxes, labels, lmks


class RandomDistort:
    def __call__(self, img, bboxes=None, labels=None, lmks=None):
        def _convert(image, alpha=1, beta=0):
            tmp = image.astype(float) * alpha + beta
            tmp[tmp < 0] = 0
            tmp[tmp > 255] = 255
            image[:] = tmp

        image = img.copy()

        if random.randrange(2):

            #brightness distortion
            if random.randrange(2):
                _convert(image, beta=random.uniform(-32, 32))

            #contrast distortion
            if random.randrange(2):
                _convert(image, alpha=random.uniform(0.5, 1.5))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            #saturation distortion
            if random.randrange(2):
                _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

            #hue distortion
            if random.randrange(2):
                tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
                tmp %= 180
                image[:, :, 0] = tmp

            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            #brightness distortion
            if random.randrange(2):
                _convert(image, beta=random.uniform(-32, 32))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            #saturation distortion
            if random.randrange(2):
                _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

            #hue distortion
            if random.randrange(2):
                tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
                tmp %= 180
                image[:, :, 0] = tmp

            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

            #contrast distortion
            if random.randrange(2):
                _convert(image, alpha=random.uniform(0.5, 1.5))

        return image, bboxes, labels, lmks


class Pad:
    def __init__(self, img_mean=[104, 111, 120]):
        self.img_mean = img_mean

    def __call__(self, img, bboxes=None, labels=None, lmks=None):
        height, width, _ = img.shape
        if height == width:
            return img, bboxes, labels, lmks
        long_side = max(width, height)
        image_t = np.empty((long_side, long_side, 3), dtype=img.dtype)
        image_t[:, :] = self.img_mean
        image_t[0:0 + height, 0:0 + width] = img
        return image_t, bboxes, labels, lmks


class RandomFlip:
    def __call__(self, img, bboxes=None, labels=None, lmks=None):
        _, width, _ = img.shape
        if random.randrange(2):
            img = img[:, ::-1]
            bboxes = bboxes.copy()
            bboxes[:, 0::2] = width - bboxes[:, 2::-2]

            # landm
            lmks = lmks.copy()
            lmks = lmks.reshape([-1, 5, 2])
            lmks[:, :, 0] = width - lmks[:, :, 0]
            tmp = lmks[:, 1, :].copy()
            lmks[:, 1, :] = lmks[:, 0, :]
            lmks[:, 0, :] = tmp
            tmp1 = lmks[:, 4, :].copy()
            lmks[:, 4, :] = lmks[:, 3, :]
            lmks[:, 3, :] = tmp1
            lmks = lmks.reshape([-1, 10])

        return img, bboxes, labels, lmks


class Resize:
    def __init__(self, image_size=(640, 640)):
        self.image_size = image_size

    def __call__(self, img, bboxes=None, labels=None, lmks=None):
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[random.randrange(5)]
        img = cv2.resize(img, self.image_size, interpolation=interp_method)
        img = img.astype(np.float32)
        return img, bboxes, labels, lmks


class Coord2Norm:
    def __call__(self, img, bboxes=None, labels=None, lmks=None):
        height, width, _ = img.shape
        bboxes[:, 0::2] /= width
        bboxes[:, 1::2] /= height

        lmks[:, 0::2] /= width
        lmks[:, 1::2] /= height
        return img, bboxes, labels, lmks


class ImageT:
    def __call__(self, img, bboxes=None, labels=None, lmks=None):
        img = img.transpose(2, 0, 1)
        # labels = np.expand_dims(labels, 1)

        return img, bboxes, labels, lmks


class Normalize:
    def __init__(self, image_mean, image_std):
        self.image_mean = image_mean
        self.image_std = image_std
    
    def __call__(self, img, bboxes=None, labels=None, lmks=None):
        img = (img - self.image_mean) / self.image_std
        return img, bboxes, labels, lmks


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes=None, labels=None, lmks=None):
        for t in self.transforms:
            img, bboxes, labels, lmks = t(img, bboxes, labels, lmks)
        return img, bboxes, labels, lmks


def build_transform(image_size, image_mean, image_std, iof_factor=1.0, min_face=16):
    transforms = Compose([
        RandomCrop(image_size, iof_factor, min_face), 
        RandomDistort(), Pad(image_mean), RandomFlip(), Normalize(image_mean, image_std),
        Coord2Norm(), Resize(image_size), ImageT()
    ])
    return transforms