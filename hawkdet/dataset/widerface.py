import cv2
import numpy as np
import torch

from torch.utils.data import Dataset


class WiderFace(Dataset):
    def __init__(self, txt_path, transforms=None):
        super().__init__()
        self.transforms = transforms
        self.imgs_path = []
        self.annos = []
        with open(txt_path) as f:
            lines = f.readlines()

        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.annos.append(np.array(labels_copy))
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                label = [float(x) for x in line.split(' ')]
                # format: xyxy
                anno = [
                    label[0], label[1], label[0] + label[2], label[1] + label[3],
                    label[4], label[5], label[7], label[8], label[10], label[11],
                    label[13], label[14], label[16], label[17], -1 if label[4]<0 else 1
                ]
                labels.append(anno)

        self.annos.append(np.array(labels))

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        anno = self.annos[index]
        assert anno.shape[0] > 0, 'length of annotation must be greater than 0'
        
        item = {
            'image': img,
            'bboxes': anno[:, :4],
            'labels': anno[:, -1],
            'landmarks': anno[:, 4:-1]
        }

        if self.transforms is not None:
            item = self.transforms(item)

        return item


def collater(batch_samples):
    bboxes = []
    labels = []
    lmks = []
    imgs = []
    for sample in batch_samples:
        single_img = sample['image']
        single_bboxes = sample['bboxes']
        single_labels = sample['labels']
        single_lmks = sample['landmarks']

        imgs.append(single_img)

        bboxes.append(torch.from_numpy(single_bboxes).float())
        labels.append(torch.from_numpy(single_lmks).float())
        lmks.append(torch.from_numpy(single_labels).float())
        

    return {'images': torch.from_numpy(np.array(imgs)), 'bboxes': bboxes, 'landmarks': lmks, 'labels': labels}