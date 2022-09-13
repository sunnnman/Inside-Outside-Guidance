from pathlib import Path

import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image

class GrabCutDataset(data.Dataset):
    def __init__(self, dataset_path,
                 images_dir_name='data_GT', masks_dir_name='boundary_GT',
                transform = None,**kwargs):
        super(GrabCutDataset, self).__init__(**kwargs)

        self.transform = transform
        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.*')}

    def __getitem__(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split('.')[0]])
        image_name = image_name.split('.')[0]
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Read Image
        _img = np.array(Image.open(image_path).convert('RGB')).astype(np.float32)  ###zsy open image imread

        # Read Target object
        _tmp = (np.array(Image.open(mask_path,))).astype(np.float32)

        # berkeley读入的mask是三通道，这里取一个就行
        _tmp = _tmp[:,:,0]

        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        _void_pixels = (_tmp == 255)
        # _tmp[_void_pixels] = 0

        instances_mask[instances_mask == 128] = -1
        instances_mask[instances_mask > 128] = 1

        sample = {'image': _img,'gt': _tmp, 'void_pixels': _void_pixels.astype(np.float32) }
        sample['meta'] = {'image': image_name,
                          'im_size':(_img.shape[0], _img.shape[1])}
        # instances_ids = [1]
        #
        # instances_info = {
        #     x: {'ignore': False}
        #     for x in instances_ids
        # }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.dataset_samples)
