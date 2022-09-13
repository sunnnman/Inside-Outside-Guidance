from pycocotools.coco import COCO
import os
import sys
import tarfile
import numpy as np

import torch.utils.data as data
from PIL import Image
from six.moves import urllib
import json
from mypath import Path


class CocoSegmentation(data.Dataset):
    def __init__(self,
                 split,
                 year = '2014',
                 area_range=[],
                 only_pascal_categories=False,
                 without_pascal_categories=False,
                 one_mask_per_anno=False,
                 one_mask_per_class=False,
                 images_with_all_cstm_cats=None,
                 db_root=Path.db_root_dir('coco'),
                 # db_root=Path.db_root_dir('ctw1500'),
                 max_num_samples=None,
                 transform=None,
                 retname=True,
                 mini=False,
                 return_rle=False):

        self.split = split
        self.return_rle = return_rle
        self.root = os.path.join(db_root, split+year )
        # self.root = os.path.join(db_root, 'text_image' )
        # annFile = os.path.join(db_root, 'annotations', 'instances_'+split+year+'.json')
        annFile = os.path.join(db_root, 'annotations', 'instances_mini'+split+year+'.json')
        # annFile = os.path.join(db_root, 'annotations', 'stuff_val2017.json')
        # annFile = os.path.join(db_root, 'instances_test.json')
        # annFile = "/home/zhupengqi/datasets/instances_minival2014.json"

        self.coco = COCO(annFile)
        cat_ids = self.coco.getCatIds()
        print(cat_ids)
        # pascal_cat_name = self.coco.loadCats(cat_ids)
        #
        # pascal_cat_name = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'airplane',
        #                    'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train', 'bottle', 'chair',
        #                    'dining table', 'potted plant', 'couch', 'tv']
        # if only_pascal_categories:
        #     cat_ids = self.coco.getCatIds(catNms=pascal_cat_name)
        # elif without_pascal_categories:
        #     pascal_cat_ids = set(self.coco.getCatIds(catNms=pascal_cat_name))
        #     coco_cat_ids = set(self.coco.getCatIds())
        #     cat_ids = list(coco_cat_ids-pascal_cat_ids)
        # elif images_with_all_cstm_cats is not None:
        #     cat_ids = self.coco.getCatIds(catNms=images_with_all_cstm_cats)
        # else:
        #     cat_ids = []

        if mini:
            self.coco.imgs = {x: self.coco.imgs[x] for x in list(np.sort(self.coco.imgs.keys())[:5000])}

        if images_with_all_cstm_cats is not None:
            self.img_ids = self.coco.getImgIds(imgIds=self.coco.imgs.keys(), catIds=cat_ids)
        else:
            self.img_ids = self.coco.imgs.keys()

        self.ids = self.coco.getAnnIds(imgIds=self.img_ids, areaRng=area_range, catIds=cat_ids)
        self.transform = transform
        self.area_range = area_range
        self.only_pascal_categories = only_pascal_categories
        self.without_pascal_categories = without_pascal_categories
        self.cat_ids = cat_ids
        self.one_mask_per_anno = one_mask_per_anno
        self.one_mask_per_class = one_mask_per_class
        self.retname = retname

        if max_num_samples:
            if self.one_mask_per_anno or self.one_mask_per_class:
                self.img_ids = self.img_ids[:max_num_samples]
            else:
                self.ids = self.ids[:max_num_samples]

        # Display stats
        if self.one_mask_per_anno or self.one_mask_per_class:
            print("Number of images: {:d}".format(len(self.img_ids)))
        else:
            print('Number of images: {:d}\nNumber of objects: {:d}'.format(len(self.coco.imgs), len(self.ids)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        if self.one_mask_per_anno:
            img_id = self.img_ids[index]
            ids = coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)
            ann_meta = coco.loadAnns(ids)
            cat_id = self.cat_ids
        elif self.one_mask_per_class:
            img_id = self.img_ids[index]
            ann_meta = []
            for cat_id in self.cat_ids:
                ids = coco.getAnnIds(imgIds=img_id, catIds=cat_id)
                ann_meta.append(coco.loadAnns(ids))
            cat_id = self.cat_ids
        else:
            ann_meta = coco.loadAnns(int(self.ids[index]))
            # img_id = ann_meta[0]["image_id"]

            try:
                img_id = ann_meta[0]["image_id"]
                cat_id = ann_meta[0]['category_id']
            except Exception as e:
                # from ipdb import set_trace
                # set_trace()
                # continue
                print('此id为空')


        img_meta = coco.loadImgs(img_id)[0]
        path = img_meta['file_name']
        # path = path.split('/')[1]

        sample = {}
        if self.retname:
            sample['meta'] = {'image': str(path).split('.')[0],
                              'object': str(self.ids[index]),
                              'category': cat_id,
                              'im_size': (img_meta['height'], img_meta['width'])}

        if not self.return_rle:
            try:
                img = np.array(Image.open(os.path.join(self.root, path)).convert('RGB')).astype(np.float32)
                if self.one_mask_per_class:
                    target = np.zeros([img.shape[0], img.shape[1]])
                    for ii in range(len(cat_id)):
                        ann_meta_class = ann_meta[ii]
                        target_tmp = np.zeros([img.shape[0], img.shape[1]])
                        for ann in ann_meta_class:
                            target_tmp = np.logical_or(target_tmp, np.array(coco.annToMask(ann)))
                        target[target_tmp] = ii+1
                    target = target[:, :, np.newaxis]
                else:
                    target = np.zeros([img.shape[0], img.shape[1], 1])
                    for ann in ann_meta:
                        target = np.logical_or(target, np.array(coco.annToMask(ann).reshape([img.shape[0], img.shape[1], 1])))
                target = target.astype(np.float32)
            except:
                img = np.zeros((100, 100, 3))
                target = np.zeros((100, 100))
                print('Error reading image '+str(path)+' with object id '+str(self.ids[index]))

            sample['image'] = img
            sample['gt'] = target
            sample['void_pixels'] = target

            if self.transform is not None:
                sample = self.transform(sample)
        else:
            sample = {'segmentation': ann_meta}
        return sample

    def __len__(self):
        if self.one_mask_per_anno or self.one_mask_per_class:
            return len(self.img_ids)
        else:
            return len(self.ids)

    def __str__(self):
        return 'CocoSegmentation(split='+str(self.split)+', area_range='+str(self.area_range)+\
               ', without_pascal_categories='+str(self.without_pascal_categories)+\
               ', only_pascal_categories=' + str(self.only_pascal_categories)+')'
