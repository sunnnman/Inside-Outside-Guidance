import torch, cv2
import numpy.random as random
import numpy as np
import dataloaders.helpers as helpers
import scipy.misc as sm
from dataloaders.helpers import *

class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25), semseg=False):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales
        self.semseg = semseg

    def __call__(self, sample):

        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0])/2

            sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot = self.rots[random.randint(0, len(self.rots))]
            sc = self.scales[random.randint(0, len(self.scales))]

        for elem in sample.keys():
            if 'meta' in elem:
                continue

            tmp = sample[elem]

            h, w = tmp.shape[:2]
            center = (w / 2, h / 2)
            assert(center != 0)  # Strange behaviour warpAffine
            M = cv2.getRotationMatrix2D(center, rot, sc)

            if ((tmp == 0) | (tmp == 1)).all():
                flagval = cv2.INTER_NEAREST
            elif 'gt' in elem and self.semseg:
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC
            tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)

            sample[elem] = tmp

        return sample

    def __str__(self):
        return 'ScaleNRotate:(rot='+str(self.rots)+',scale='+str(self.scales)+')'


class FixedResize(object):
    """Resize the image and the ground truth to specified resolution.
    Args:
        resolutions (dict): the list of resolutions
    """
    def __init__(self, resolutions=None, flagvals=None):
        self.resolutions = resolutions
        self.flagvals = flagvals
        # self.flagvals.append()
        if self.flagvals is not None:
            assert(len(self.resolutions) == len(self.flagvals))

    def __call__(self, sample):

        # Fixed range of scales
        if self.resolutions is None:
            return sample

        elems = list(sample.keys())

        for elem in elems:

            if 'meta' in elem or 'bbox' in elem or ('extreme_points_coord' in elem and elem not in self.resolutions):
                continue
            if 'extreme_points_coord' in elem and elem in self.resolutions:
                bbox = sample['bbox']
                crop_size = np.array([bbox[3]-bbox[1]+1, bbox[4]-bbox[2]+1])
                res = np.array(self.resolutions[elem]).astype(np.float32)
                sample[elem] = np.round(sample[elem]*res/crop_size).astype(np.int)
                continue
            if elem in self.resolutions:
                if self.resolutions[elem] is None:
                    continue
                if elem == 'patch_size':
                    # print('crop_dev:', sample['crop_dev'])
                    continue
                if isinstance(sample[elem], list):
                    if sample[elem][0].ndim == 3:
                        output_size = np.append(self.resolutions[elem], [3, len(sample[elem])])
                    else:
                        output_size = np.append(self.resolutions[elem], len(sample[elem]))
                    tmp = sample[elem]
                    sample[elem] = np.zeros(output_size, dtype=np.float32)
                    for ii, crop in enumerate(tmp):
                        if self.flagvals is None:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem])
                        else:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem], flagval=self.flagvals[elem])
                else:
                    if self.flagvals is None:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem])
                    else:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem], flagval=self.flagvals[elem])
            else:
                del sample[elem]
                # if elem == 'gt':
                #     pass
                # else:
                #     del sample[elem]

        return sample

    def __str__(self):
        return 'FixedResize:'+str(self.resolutions)


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if 'meta' in elem:
                    continue
                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[elem] = tmp

        return sample

    def __str__(self):
        return 'RandomHorizontalFlip'


class IOGPoints(object):
    """
    Returns the IOG Points (top-left and bottom-right or top-right and bottom-left) in a given binary mask
    sigma: sigma of Gaussian to create a heatmap from a point
    pad_pixel: number of pixels fo the maximum perturbation
    elem: which element of the sample to choose as the binary mask
    """
    def __init__(self, sigma=10, elem='crop_gt',pad_pixel =10):
        self.sigma = sigma
        self.elem = elem
        self.pad_pixel =pad_pixel

    def __call__(self, sample):

        if sample[self.elem].ndim == 3:
            raise ValueError('IOGPoints not implemented for multiple object per image.')
        _target = sample[self.elem]

        targetshape=_target.shape
        if np.max(_target) == 0:
            sample['IOG_points'] = np.zeros([targetshape[0],targetshape[1],2], dtype=_target.dtype) #  TODO: handle one_mask_per_point case
        else:
            _points = helpers.iog_points(_target, self.pad_pixel)  # _point得到四个背景点，一个前景点的[]坐标位置
            sample['IOG_points'] = helpers.make_gt(_target, _points, sigma=self.sigma, one_mask_per_point=False)

        return sample

    def __str__(self):
        return 'IOGPoints:(sigma='+str(self.sigma)+', pad_pixel='+str(self.pad_pixel)+', elem='+str(self.elem)+')'


class ConcatInputs(object):

    def __init__(self, elems=('image', 'point')):
        self.elems = elems

    def __call__(self, sample):

        res = sample[self.elems[0]]

        for elem in self.elems[1:]:
            assert(sample[self.elems[0]].shape[:2] == sample[elem].shape[:2])

            # Check if third dimension is missing
            tmp = sample[elem]
            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            res = np.concatenate((res, tmp), axis=2)

        sample['concat'] = res
        return sample

    def __str__(self):
        return 'ExtremePoints:'+str(self.elems)

class CropFromMask_(object):
    """
        Returns image cropped in bounding box from a given mask
        """

    def __init__(self, crop_elems=('image', 'gt', 'void_pixels', 'dev'),
                 mask_elem='gt',
                 relax=0,
                 zero_pad=False):

        self.crop_elems = crop_elems
        self.mask_elem = mask_elem
        self.relax = relax
        self.zero_pad = zero_pad

    def __call__(self, sample):
        _target = sample[self.mask_elem]
        if _target.ndim == 2:
            _target = np.expand_dims(_target, axis=-1)
        for elem in self.crop_elems:
            if elem != 'dev':
                _img = sample[elem]
            _crop = []
            if self.mask_elem == elem:
                if _img.ndim == 2:
                    _img = np.expand_dims(_img, axis=-1)
                for k in range(0, _target.shape[-1]):
                    _tmp_img = _img[..., k]
                    _tmp_target = _target[..., k]
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_tmp_img.shape, dtype=_img.dtype))
                    else:
                        _crop.append(
                            helpers.crop_from_mask_(_tmp_img, _tmp_target, sample['meta']['bbox'], relax=self.relax, zero_pad=self.zero_pad))
            elif elem == 'dev':
                # print('计算regular,得出dev')
                for k in range(0, _target.shape[-1]):  # 如果有多个mask 应该就是target第三个维度 = mask数
                    if np.max(_target[..., k]) == 0:
                        _crop.append(1)
                    else:
                        regular = _target.sum() / (_target.shape[0] * _target.shape[1])
                        if regular > 0.7:
                            dev = random.randint(1, 2)
                        elif regular > 0.6:
                            dev = random.randint(1, 3)
                        # elif regular > 0.5:
                        #     dev = random.randint(3, 4)
                        # elif regular > 0.4:
                        #     dev = random.randint(4, 5)
                        # elif regular > 0.3:
                        #     dev = random.randint(4, 6)
                        else:
                            dev = random.randint(1, 3)
                        _crop.append(dev)

            else:
                for k in range(0, _target.shape[-1]):
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_img.shape, dtype=_img.dtype))
                    else:
                        _tmp_target = _target[..., k]
                        _crop.append(
                            helpers.crop_from_mask_(_img, _tmp_target, sample['meta']['bbox'], relax=self.relax, zero_pad=self.zero_pad))
            if len(_crop) == 1:
                sample['crop_' + elem] = _crop[0]
            else:
                sample['crop_' + elem] = _crop

        return sample


class CropFromMask(object):
    """
    Returns image cropped in bounding box from a given mask
    """
    def __init__(self, crop_elems=('image', 'gt','void_pixels', 'dev'),
                 mask_elem='gt',
                 relax=0,
                 zero_pad=False):

        self.crop_elems = crop_elems
        self.mask_elem = mask_elem
        self.relax = relax
        self.zero_pad = zero_pad

    def __call__(self, sample):
        _target = sample[self.mask_elem]

        if _target.ndim == 2:
            _target = np.expand_dims(_target, axis=-1)
        for elem in self.crop_elems:
            if elem != 'dev':
                _img = sample[elem]
            _crop = []
            if self.mask_elem == elem:
                if _img.ndim == 2:
                    _img = np.expand_dims(_img, axis=-1)
                for k in range(0, _target.shape[-1]):
                    _tmp_img = _img[..., k]
                    _tmp_target = _target[..., k]
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_tmp_img.shape, dtype=_img.dtype))
                    else:
                        _crop.append(helpers.crop_from_mask(_tmp_img, _tmp_target, relax=self.relax, zero_pad=self.zero_pad))
            elif elem == 'dev':

                # print('计算regular,得出dev')
                for k in range(0, _target.shape[-1]):  # 如果有多个mask 应该就是target第三个维度 = mask数
                    if np.max(_target[..., k]) == 0:
                        _crop.append(1)
                    else:
                        real_bbox = helpers.get_bbox(_target, pad=10, zero_pad=False)
                        # regular = _target.sum() / (_target.shape[0] * _target.shape[1])
                        # if regular > 0.6:
                        #     dev = random.randint(2, 3)
                        # elif regular > 0.3:
                        #     dev = random.randint(3, 5)
                        # else:
                        #     dev = random.randint(4, 6)
                        scale = ((real_bbox[2] - real_bbox[0]) * (real_bbox[3] - real_bbox[1])) / (_target.shape[0] * _target.shape[1])
                        crop_gt = _target[real_bbox[1]:real_bbox[3],real_bbox[0]:real_bbox[2]]
                        crop_gt = crop_gt.squeeze()

                        min_ = min((real_bbox[2] - real_bbox[0]), (real_bbox[3] - real_bbox[1]))
                        max_ = max((real_bbox[2] - real_bbox[0]), (real_bbox[3] - real_bbox[1]))
                        regular = crop_gt.sum() / (min_ * max_)

                        # if scale >= 0.5:
                        #     if max_ / min_ > 3:
                        #         dev = random.randint(3, 5)
                        #     elif regular >= 0.4:
                        #         dev = random.randint(5, 7)
                        #     else:
                        #         dev = random.randint(6, 8)
                        #
                        # elif scale < 0.5 and scale > 0.3:
                        #     if max_ / min_ > 3:
                        #         dev = random.randint(2, 4)
                        #     elif regular > 0.4:
                        #         dev = random.randint(4, 5)
                        #     else:
                        #         dev = random.randint(4, 6)
                        # elif scale <= 0.3 and scale >= 0.2:
                        #     if regular > 0.4:
                        #         dev = random.randint(3, 4)
                        #     else:
                        #         dev = random.randint(4, 5)
                        # else:
                        #     if regular > 0.4:
                        #         dev = random.randint(1, 2)
                        #     else:
                        #         dev = random.randint(2, 3)
                        # _crop.append(dev)

                        if scale >= 0.5:
                                dev = random.randint(6, 9)

                        elif scale < 0.5 and scale > 0.2:
                                dev = random.randint(4, 6)
                        else:
                            dev = random.randint(2, 3)

                        if regular > 0.7:
                            times = 1
                        elif regular > 0.3:
                            times = 1.2
                        else:
                            times = 1.5
                        dev = dev * times
                        _crop.append(dev)

            else:
                for k in range(0, _target.shape[-1]):
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_img.shape, dtype=_img.dtype))
                    else:
                        _tmp_target = _target[..., k]
                        _crop.append(helpers.crop_from_mask(_img, _tmp_target, relax=self.relax, zero_pad=self.zero_pad))
            if len(_crop) == 1:
                sample['crop_' + elem] = _crop[0]
            else:
                sample['crop_' + elem] = _crop

        return sample

    def __str__(self):
        return 'CropFromMask:(crop_elems='+str(self.crop_elems)+', mask_elem='+str(self.mask_elem)+\
               ', relax='+str(self.relax)+',zero_pad='+str(self.zero_pad)+')'

class Grids(object):
    """
    Returns the Grids (the grids which mask rate > screen) in a given binary mask
    sigma: sigma of Gaussian to create a heatmap from a point
    pad_pixel: number of pixels fo the maximum perturbation
    elem: which element of the sample to choose as the binary mask
    """
    def __init__(self, sigma=10, elem='crop_gt',pad_pixel =10, screen = 0.1):
        self.sigma = sigma
        self.elem = elem
        self.pad_pixel =pad_pixel
        self.screen = screen  # 过滤patch

    def __call__(self, sample):
        # 到这是切割好，但是还没重新调整过大小的sample
        if sample[self.elem].ndim == 3:
            raise ValueError('Grids not implemented for multiple object per image.')
        _target = sample[self.elem]
        targetshape = _target.shape
        if np.max(_target) == 0:
            sample['Grids'] = np.zeros([targetshape[0], targetshape[1], 2], dtype=_target.dtype)  # TODO: handle one_mask_per_point case
            sample['patch_size'] = [1,1]
        else:
            _grids,bg_points = helpers.grids(_target, sample['crop_dev'], sample['image'],screen=0.1,  pad_pixel=10,meta=sample['meta'])
            # _grids,bg_points = helpers.grids_(_target, sample['crop_dev'],sample['meta']['bbox'], sample['image'],screen=0.1,  pad_pixel=10)
            sample['Grids'] = helpers.make_gt_grids(_target, _grids, bg_points, sigma=self.sigma, one_mask_per_point=False,meta=sample['meta'],img_=sample['crop_image'],dev=sample['crop_dev'])
            # patch_size_x = int( (patch_size / targetshape[0]) * 512 ) + 1
            # patch_size_y = int( (patch_size / targetshape[1]) * 512 ) + 1
            # print('patch_size_x',patch_size_x)
            # print('patch_size_y',patch_size_y)

            # sample['patch_size'] = [patch_size_x, patch_size_y]


        return sample

    def __str__(self):
        return 'Grids:(screen='+str(self.screen)+', elem='+str(self.elem)+')'

class ToImage(object):
    """
    Return the given elements between 0 and 255
    """
    def __init__(self, norm_elem='image', custom_max=255.):
        self.norm_elem = norm_elem
        self.custom_max = custom_max

    def __call__(self, sample):
        if isinstance(self.norm_elem, tuple):
            for elem in self.norm_elem:
                tmp = sample[elem]
                sample[elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        else:
            tmp = sample[self.norm_elem]
            sample[self.norm_elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        return sample

    def __str__(self):
        return 'NormalizeImage'


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if elem != 'patch_size':
                if 'meta' in elem:
                    continue
                elif 'bbox' in elem:
                    tmp = sample[elem]
                    sample[elem] = torch.from_numpy(tmp)
                    continue

                tmp = sample[elem]

                if tmp.ndim == 2:
                    tmp = tmp[:, :, np.newaxis]

                # swap color axis because
                # numpy image: H x W x C
                # torch image: C X H X W
                tmp = tmp.transpose((2, 0, 1))
                sample[elem] = torch.from_numpy(tmp)

        return sample

    def __str__(self):
        return 'ToTensor'
