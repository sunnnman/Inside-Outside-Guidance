import os.path
from pathlib import Path
import json
import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image

class sstemDataset(data.Dataset):
    def __init__(self, dataset_path,
                 images_dir_name='stack1/raw/',masks_dir_name = 'stack1/mitochondria/' , json_dir='ssTEM.json'
                ,transform = None,**kwargs):
        super(sstemDataset, self).__init__(**kwargs)
        self.transform = transform
        self.dataset_path = Path(dataset_path)
        self._images_path = os.path.join(self.dataset_path,images_dir_name)
        self._masks_path = os.path.join(self.dataset_path,masks_dir_name)
        self.json_dir = os.path.join(self.dataset_path,json_dir)
        with open(self.json_dir, "r", encoding='utf-8') as jsonf:
            self.jsonData = json.load(jsonf)

    def __getitem__(self, index):
        # index += 1
        bbox_anno = self.jsonData['bbox'][index]
        image_name = self.jsonData['image'][index]
        gt_name = self.jsonData['gt'][index]

        _img = np.array(Image.open(self._images_path + image_name).convert('RGB')).astype(np.float32)  ###zsy open image imread

        _tmp = cv2.imread(self._masks_path + gt_name, -1)
        # if _tmp == None:
        #     print(self._masks_path + str(image_id) + '_' + str(index+1) + '_' + 'mask.png')

        _tmp = _tmp.astype(np.float32)
        _tmp_ = _tmp / 255
        # _tmp = cv2.imread(self._masks_path + str(image_id) + '_' + str(index) + '_' + 'mask.bmp').astype(np.int32)
        _void_pixels = (_tmp == 255)

        sample = {'image': _img, 'gt': _tmp, 'void_pixels': _void_pixels.astype(np.float32)}
        sample['meta'] = {'image': image_name,
                          'im_size': (_img.shape[0], _img.shape[1]),
                          'index': index,
                          'bbox': bbox_anno
                          }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return 388

class sstemDataset_(data.Dataset):
    def __init__(self, dataset_path,masks_dir_name = 'stack1/mitochondria/' ,transform = None,**kwargs):
        super(sstemDataset_, self).__init__(**kwargs)
        self.transform = transform
        self.dataset_path = Path(dataset_path)
        self._masks_path = os.path.join(self.dataset_path,masks_dir_name)
        self.gt_name = ['00.png','01.png','02.png','03.png','04.png','05.png','06.png','07.png','08.png','09.png','10.png',
                        '11.png','12.png','13.png','14.png','15.png','16.png','17.png','18.png','19.png',]

    def __getitem__(self, index):
        # index += 1

        _tmp = cv2.imread(self._masks_path + self.gt_name[index], -1)
        # if _tmp == None:
        #     print(self._masks_path + str(image_id) + '_' + str(index+1) + '_' + 'mask.png')

        _tmp = _tmp.astype(np.float32)
        sample = {}
        sample = {'gt': _tmp,
                          'name': self.gt_name[index],}

        return sample

    def __len__(self):
        return 19