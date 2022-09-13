import os
import torch, cv2
import random
import numpy as np
from imageio import imsave

def tens2image(im):
    if im.size()[0] == 1:
        # tensor.detach().numpy()
        tmp = np.squeeze(im.detach().numpy(), axis=0)
    else:
        tmp = im.numpy()
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0))


def crop2fullmask(crop_mask, bbox, im=None,result_pre = None, im_size=None, zero_pad=False, relax=0, mask_relax=True,
                  interpolation=cv2.INTER_CUBIC, scikit=False):
    # if scikit:
    #     from skimage.transform import resize as sk_resize
    assert(not(im is None and im_size is None)), 'You have to provide an image or the image size  True'
    if im is None:
        im_si = im_size
    else:
        im_si = im.shape
    # Borers of image
    bounds = (0, 0, im_si[1] - 1, im_si[0] - 1)

    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    # Bounding box of initial mask
    bbox_init = (bbox[0] + relax,
                 bbox[1] + relax,
                 bbox[2] - relax,
                 bbox[3] - relax)

    if zero_pad:
        # Offsets for x and y
        offsets = (-bbox[0], -bbox[1])
    else:
#        assert((bbox == bbox_valid).all())
        offsets = (-bbox_valid[0], -bbox_valid[1])

    # Simple per element addition in the tuple
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

    if scikit:
        # crop_mask = sk_resize(crop_mask, (bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1), order=0, mode='constant').astype(crop_mask.dtype)
        pass
    else:
        crop_mask = cv2.resize(crop_mask, (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), interpolation=interpolation)
    result_ = np.zeros(im_si)
    result_[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1] = \
        crop_mask[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1]

    result = np.zeros(im_si)
    if mask_relax:
        result[bbox_init[1]:bbox_init[3]+1, bbox_init[0]:bbox_init[2]+1] = \
            result_[bbox_init[1]:bbox_init[3]+1, bbox_init[0]:bbox_init[2]+1]
    else:
        result = result_

    if result_pre is not None:
        result = result + result_pre
        result = np.where(result > 1, 1, result)

    return result

def crop2fullmask_(crop_mask, bbox, im=None, result_pre = None, im_size=None, zero_pad=False, relax=0, mask_relax=True,
                  interpolation=cv2.INTER_CUBIC, scikit=False):
    #(pred, bbox, gt, zero_pad=True, relax=0,mask_relax=False)
    if scikit:
        # from skimage.transform import resize as sk_resize
        pass
    assert(not(im is None and im_size is None)), 'You have to provide an image or the image size  True'
    if im is None:
        im_si = im_size
    else:
        im_si = im.shape
    # Borers of image
    bounds = (0, 0, im_si[1] - 1, im_si[0] - 1)
    bbox = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    crop_mask = cv2.resize(crop_mask, (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), interpolation=interpolation)


    # if result_pre is None:
    #     result_ = np.zeros(im_si)
    # else:
    #     result_ = result_pre
    result_ = np.zeros(im_si)

    result_[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1] = crop_mask

    if result_pre is not None:
        result = result_ + result_pre
        result = np.where(result > 1, 1, result)
    else:
        result = result_

    return result

def overlay_mask(im, ma, colors=None, alpha=0.5):
    assert np.max(im) <= 1.0
    if colors is None:
        colors = np.load(os.path.join(os.path.dirname(__file__), 'pascal_map.npy'))/255.
    else:
        colors = np.append([[0.,0.,0.]], colors, axis=0);

    if ma.ndim == 3:
        assert len(colors) >= ma.shape[0], 'Not enough colors'
    ma = ma.astype(np.bool)
    im = im.astype(np.float32)

    if ma.ndim == 2:
        fg = im * alpha+np.ones(im.shape) * (1 - alpha) * colors[1, :3]   # np.array([0,0,255])/255.0
    else:
        fg = []
        for n in range(ma.ndim):
            fg.append(im * alpha + np.ones(im.shape) * (1 - alpha) * colors[1+n, :3])
    # Whiten background
    bg = im.copy()
    if ma.ndim == 2:
        bg[ma == 0] = im[ma == 0]
        bg[ma == 1] = fg[ma == 1]
        total_ma = ma
    else:
        total_ma = np.zeros([ma.shape[1], ma.shape[2]])
        for n in range(ma.shape[0]):
            tmp_ma = ma[n, :, :]
            total_ma = np.logical_or(tmp_ma, total_ma)
            tmp_fg = fg[n]
            bg[tmp_ma == 1] = tmp_fg[tmp_ma == 1]
        bg[total_ma == 0] = im[total_ma == 0]

    # [-2:] is s trick to be compatible both with opencv 2 and 3
    contours = cv2.findContours(total_ma.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(bg, contours[0], -1, (0.0, 0.0, 0.0), 1)

    return bg
import PIL
def overlay_masks(im, masks, alpha=0.5):
    colors = np.load(os.path.join(os.path.dirname(__file__), 'pascal_map.npy'))/255.
    
    if isinstance(masks, np.ndarray):
        masks = [masks]

    assert len(colors) >= len(masks), 'Not enough colors'

    ov = im.copy()
    ov_black = im.copy()*0
    
    imgZero = np.zeros(np.array(masks, dtype = np.uint8).shape,np.uint8)
    im = im.astype(np.float32)
    total_ma = np.zeros([im.shape[0], im.shape[1]])
    i = 1
    for ma in masks:
        ma = ma.astype(np.bool)
        fg = im * alpha+np.ones(im.shape) * (1 - alpha) * colors[i, :3]   # np.array([0,0,255])/255.0
        i = i + 1
        ov[ma == 1] = fg[ma == 1]
        total_ma += ma

        # [-2:] is s trick to be compatible both with opencv 2 and 3
        contours = cv2.findContours(ma.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        cv2.drawContours(ov, contours[0], -1, (0.0, 0.0, 0.0), 1)
        cv2.drawContours(ov_black, contours[0], -1, (255, 255, 255), -1)#only draw a round
    ov[total_ma == 0] = im[total_ma == 0]

    return ov_black

from scipy import ndimage    
def getPositon(distance_transform):
    a = np.mat(distance_transform) 
    raw, column = a.shape# get the matrix of a raw and column
    _positon = np.argmax(a)# get the index of max in the a
    m, n = divmod(_positon, column)
    raw=m
    column=n
#    print "The raw is " ,m
#    print "The column is ",  n
#    print "The max of the a is ", a[m , n]
#    print(raw,column,a[m , n])
    return  raw,column

def iog_points(mask, pad_pixel=10):
    def find_point(id_x, id_y, ids):
        sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
        return [id_x[sel_id], id_y[sel_id]]

    inds_y, inds_x = np.where(mask > 0.5)   
    [h,w]=mask.shape
    left = find_point(inds_x, inds_y, np.where(inds_x <= np.min(inds_x))) 
    right = find_point(inds_x, inds_y, np.where(inds_x >= np.max(inds_x))) 
    top = find_point(inds_x, inds_y, np.where(inds_y <= np.min(inds_y))) 
    bottom = find_point(inds_x, inds_y, np.where(inds_y >= np.max(inds_y)))

    x_min=left[0]
    x_max=right[0]
    y_min=top[1]
    y_max=bottom[1]

    map_xor = (mask > 0.5)
    h,w = map_xor.shape
    map_xor_new = np.zeros((h+2,w+2))
    map_xor_new[1:(h+1),1:(w+1)] = map_xor[:,:]
    distance_transform=ndimage.distance_transform_edt(map_xor_new)
    distance_transform_back = distance_transform[1:(h+1),1:(w+1)]    
    raw,column=getPositon(distance_transform_back)
    center_point = [column,raw]

    left_top=[max(x_min-pad_pixel,0),  max(y_min-pad_pixel,0)]
    left_bottom=[max(x_min-pad_pixel ,0),     min(y_max+pad_pixel,h)]
    right_top=[min(x_max+pad_pixel,w),          max(y_min-pad_pixel,0)]
    righr_bottom=[min(x_max+pad_pixel ,w),    min(y_max+pad_pixel,h)]
    a=[center_point,left_top,left_bottom,right_top,righr_bottom]  

    return np.array(a)
    

def get_bbox(mask, points=None, pad=0, zero_pad=False):
    if points is not None:
        inds = np.flip(points.transpose(), axis=0)
    else:
        inds = np.where(mask > 0)

    if inds[0].shape[0] == 0:
        return None

    if zero_pad:
        x_min_bound = -np.inf
        y_min_bound = -np.inf
        x_max_bound = np.inf
        y_max_bound = np.inf
    else:
        x_min_bound = 0
        y_min_bound = 0
        x_max_bound = mask.shape[1] - 1
        y_max_bound = mask.shape[0] - 1

    x_min = max(inds[1].min() - pad, x_min_bound)
    y_min = max(inds[0].min() - pad, y_min_bound)
    x_max = min(inds[1].max() + pad, x_max_bound)
    y_max = min(inds[0].max() + pad, y_max_bound)

    return x_min, y_min, x_max, y_max


def crop_from_bbox(img, bbox, zero_pad=False):
    # Borders of image
    bounds = (0, 0, img.shape[1] - 1, img.shape[0] - 1)

    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    if zero_pad:
        # Initialize crop size (first 2 dimensions)
        crop = np.zeros((bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1), dtype=img.dtype)

        # Offsets for x and y
        offsets = (-bbox[0], -bbox[1])

    else:
        assert(bbox == bbox_valid)
        crop = np.zeros((bbox_valid[3] - bbox_valid[1] + 1, bbox_valid[2] - bbox_valid[0] + 1), dtype=img.dtype)
        offsets = (-bbox_valid[0], -bbox_valid[1])

    # Simple per element addition in the tuple
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

    img = np.squeeze(img)
    if img.ndim == 2:
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1] = \
            img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1]
    else:
        crop = np.tile(crop[:, :, np.newaxis], [1, 1, 3])  # Add 3 RGB Channels
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1, :] = \
            img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1, :]

    return crop


def fixed_resize(sample, resolution, flagval=None):

    if flagval is None:
        if ((sample == 0) | (sample == 1)).all():
            flagval = cv2.INTER_NEAREST
        else:
            flagval = cv2.INTER_CUBIC

    if isinstance(resolution, int):
        tmp = [resolution, resolution]
        tmp[np.argmax(sample.shape[:2])] = int(round(float(resolution)/np.min(sample.shape[:2])*np.max(sample.shape[:2])))
        resolution = tuple(tmp)

    if sample.ndim == 2 or (sample.ndim == 3 and sample.shape[2] == 3):
        sample = cv2.resize(sample, resolution[::-1], interpolation=flagval)
    else:
        tmp = sample
        sample = np.zeros(np.append(resolution, tmp.shape[2]), dtype=np.float32)
        for ii in range(sample.shape[2]):
            sample[:, :, ii] = cv2.resize(tmp[:, :, ii], resolution[::-1], interpolation=flagval)
    return sample


def crop_from_mask(img, mask, relax=0, zero_pad=False):
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, dsize=tuple(reversed(img.shape[:2])), interpolation=cv2.INTER_NEAREST)

    assert(mask.shape[:2] == img.shape[:2])
    bbox = get_bbox(mask, pad=relax, zero_pad=zero_pad)


    if bbox is None:
        return None

    crop = crop_from_bbox(img, bbox, zero_pad)

    return crop


def crop_from_mask_(img, mask,bbox_, relax=0, zero_pad=False):
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, dsize=tuple(reversed(img.shape[:2])), interpolation=cv2.INTER_NEAREST)

    assert(mask.shape[:2] == img.shape[:2])
    # bbox = get_bbox(mask, pad=relax, zero_pad=zero_pad)
    if zero_pad:
        x_min_bound = -np.inf
        y_min_bound = -np.inf
        x_max_bound = np.inf
        y_max_bound = np.inf
    else:
        x_min_bound = 0
        y_min_bound = 0
        x_max_bound = mask.shape[1] - 1
        y_max_bound = mask.shape[0] - 1

    x_min = max(bbox_[0] - relax, x_min_bound)
    y_min = max(bbox_[1] - relax, y_min_bound)
    x_max = min(bbox_[0]+bbox_[2] + relax, x_max_bound)
    y_max = min(bbox_[1]+bbox_[3] + relax, y_max_bound)

    bbox = x_min, y_min, x_max, y_max

    if bbox is None:
        return None

    crop = crop_from_bbox(img, bbox, zero_pad)

    return crop

def make_gaussian(size, sigma=10, center=None, d_type=np.float64):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(d_type)

def grids(mask, dev =1, img = 1 ,screen = 0.1, pad_pixel=10,meta=None):

    # 过滤出符合要求的grids
    def find_point(id_x, id_y, ids):
        sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
        return [id_x[sel_id], id_y[sel_id]]

    inds_y, inds_x = np.where(mask > 0.5)
    targer = np.where(mask > 0.5, 1, 0)
    guide = np.zeros_like(targer)
    [h,w]=mask.shape


    left = find_point(inds_x, inds_y, np.where(inds_x <= np.min(inds_x)))
    right = find_point(inds_x, inds_y, np.where(inds_x >= np.max(inds_x)))
    top = find_point(inds_x, inds_y, np.where(inds_y <= np.min(inds_y)))
    bottom = find_point(inds_x, inds_y, np.where(inds_y >= np.max(inds_y)))

    x_min=left[0]
    x_max=right[0]
    y_min=top[1]
    y_max=bottom[1]

    left_top = [max(x_min - pad_pixel, 0), max(y_min - pad_pixel, 0)]
    left_bottom = [max(x_min - pad_pixel, 0), min(y_max + pad_pixel, h)]
    right_top = [min(x_max + pad_pixel, w), max(y_min - pad_pixel, 0)]
    righr_bottom = [min(x_max + pad_pixel, w), min(y_max + pad_pixel, h)]
    a = [left_top,left_bottom,right_top,righr_bottom]

    # patch_size = int(min(h, w) / dev)
    patch_size = int(min(x_max - x_min, y_max - y_min) / dev)
    # patch_size = patch_size/2
    # if patch_size>60:
    #     patch_size = 20
    # elif h>30:
    #     patch_size = 10
    # print('patch_size:' + str(patch_size))
    # screen_3 = [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.3, 0.05]
    screen_3 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05]

    # 存在一个patch_size过大的问题，patch_size最大只给他截到100吧

    # if patch_size >= 70:
    #     patch_size = 70
    # elif patch_size >= 65:
    #     patch_size = 65
    # if patch_size >= 60:
    #     patch_size = 60
    # elif patch_size >= 55:
    #     patch_size = 55
    # if patch_size >= 50:
    #     patch_size = 50
    # elif patch_size >= 45:
    #     patch_size = 45
    # elif patch_size >= 40:
    #     patch_size = 40
    # elif patch_size >= 35:
    #     patch_size = 35
    # elif patch_size >= 30:
    if patch_size >= 30:
        patch_size = 30
    # elif patch_size >= 25:
    #     patch_size = 25
    elif patch_size >= 20:
        patch_size = 20
    # elif patch_size >= 15:
    #     patch_size = 15
    elif patch_size >= 10:
        patch_size = 10
    else:
        patch_size = 10

    # patch_size = patch_size - 20
    # if patch_size > 50:
    #     patch_size = 50
    # elif patch_size < 10:
    #     patch_size = 10
    # print( meta['image'] + '-' + meta['object'] + ' patch_size:' + str(patch_size))
    # print( meta['image'] + ' patch_size:' + str(patch_size))
    # patch_size = 30
    # patch_size = int(patch_size / 2)
    # screen_ = screen_1
    screen_ = screen_3
    np.random.seed(0)

    # 直接用bbox做指导
    # bbox_ = get_bbox(targer,pad=30)
    # guide[bbox_[1]:bbox_[3],bbox_[0]:bbox_[2]] = 1

    # guide_ = guide.astype(np.uint8)
    # cv2.imwrite("/home/zhupengqi/Inside-Outside-Guidance-master/results/Grid_pascal_5010_guassian7_sample_grabcut_bbox/guide/"+meta['image']+'.png',guide_)

    xmin = 0
    ymin = 0
    x_ = targer.shape[1] % patch_size
    y_ = targer.shape[0] % patch_size
    p = np.array([0, 1])
    for i in range(int(targer.shape[0] / patch_size)):
        for j in range(int(targer.shape[1] / patch_size)):
            screen = random.choice(screen_)
            x, y, w, h = xmin, ymin, patch_size, patch_size  # 矩形裁剪区域 (ymin:ymin+h, xmin:xmin+w) 的位置参数
            region = targer[y:y + h, x:x + w].copy()  # 切片获得裁剪后保留的图像区域
            prop = region.sum() / (w * h)
            if prop > screen:
                if prop == 1:
                    keep = np.random.choice([0, 1], p=p.ravel())
                    if keep:
                        guide[i * patch_size:i * patch_size + patch_size, j * patch_size:j * patch_size + patch_size] = 1
                else:
                    guide[i * patch_size:i * patch_size + patch_size, j * patch_size:j * patch_size + patch_size] = 1
            xmin += patch_size

        if x_ > 0:
            screen = random.choice(screen_)
            x, y, w, h = xmin, ymin, x_, patch_size
            region = targer[y:y + h, x:x + w].copy()  # 切片获得裁剪后保留的图像区域
            prop = region.sum() / (w * h)
            if prop > screen:
                keep = np.random.choice([0, 1], p=p.ravel())
                if keep:
                    guide[i * patch_size:i * patch_size + patch_size, j * patch_size:j * patch_size + patch_size] = 1
        ymin += patch_size
        xmin = 0

    if y_ > 0:
        i = int(mask.shape[0] / patch_size)
        xmin = 0
        ymin = i * patch_size
        for j in range(int(targer.shape[1] / patch_size)):
            screen = random.choice(screen_)
            x, y, w, h = xmin, ymin, patch_size, y_
            region = targer[y:y + h, x:x + w].copy()  # 切片获得裁剪后保留的图像区域
            prop = region.sum() / (w * h)
            if prop > screen:
                keep = np.random.choice([0, 1], p=p.ravel())
                if keep:
                    guide[i * patch_size:i * patch_size + patch_size, j * patch_size:j * patch_size + patch_size] = 1
            xmin += patch_size
        if x_ > 0:
            screen = random.choice(screen_)
            x, y, w, h = xmin, ymin, x_, y_
            region = targer[y:y + h, x:x + w].copy()  # 切片获得裁剪后保留的图像区域
            prop = region.sum() / (w * h)
            if prop > screen:
                keep = np.random.choice([0, 1], p=p.ravel())
                if keep:
                    guide[i * patch_size:i * patch_size + patch_size, j * patch_size:j * patch_size + patch_size] = 1

    return guide, np.array(a)

def grids_(mask):
    # 指导细化交互的grid_guide 根据初次反馈的pred作为mask，patch_size直接取到10，再次生成新一轮的grid_guide

    targer = np.where(mask > 0.5, 1, 0)
    guide = np.zeros_like(targer)

    patch_size = 10
    # screen_ = [0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]
    screen_ = [0.9, 0.9, 0.9, 0.8, 0.9, 0.9, 0.9]

    xmin = 0
    ymin = 0
    x_ = targer.shape[1] % patch_size
    y_ = targer.shape[0] % patch_size
    p = np.array([0, 1])
    for i in range(int(targer.shape[0] / patch_size)):
        for j in range(int(targer.shape[1] / patch_size)):
            screen = random.choice(screen_)
            x, y, w, h = xmin, ymin, patch_size, patch_size  # 矩形裁剪区域 (ymin:ymin+h, xmin:xmin+w) 的位置参数
            region = targer[y:y + h, x:x + w].copy()  # 切片获得裁剪后保留的图像区域
            prop = region.sum() / (w * h)
            if prop > screen:
                keep = np.random.choice([0, 1], p=p.ravel())
                if keep:
                    guide[i * patch_size:i * patch_size + patch_size, j * patch_size:j * patch_size + patch_size] = 1
            xmin += patch_size

        if x_ > 0:
            screen = random.choice(screen_)
            x, y, w, h = xmin, ymin, x_, patch_size
            region = targer[y:y + h, x:x + w].copy()  # 切片获得裁剪后保留的图像区域
            prop = region.sum() / (w * h)
            if prop > screen:
                keep = np.random.choice([0, 1], p=p.ravel())
                if keep:
                    guide[i * patch_size:i * patch_size + patch_size, j * patch_size:j * patch_size + patch_size] = 1
        ymin += patch_size
        xmin = 0

    if y_ > 0:
        i = int(mask.shape[0] / patch_size)
        xmin = 0
        ymin = i * patch_size
        for j in range(int(targer.shape[1] / patch_size)):
            screen = random.choice(screen_)
            x, y, w, h = xmin, ymin, patch_size, y_
            region = targer[y:y + h, x:x + w].copy()  # 切片获得裁剪后保留的图像区域
            prop = region.sum() / (w * h)
            if prop > screen:
                keep = np.random.choice([0, 1], p=p.ravel())
                if keep:
                    guide[i * patch_size:i * patch_size + patch_size, j * patch_size:j * patch_size + patch_size] = 1
            xmin += patch_size
        if x_ > 0:
            screen = random.choice(screen_)
            x, y, w, h = xmin, ymin, x_, y_
            region = targer[y:y + h, x:x + w].copy()  # 切片获得裁剪后保留的图像区域
            prop = region.sum() / (w * h)
            if prop > screen:
                keep = np.random.choice([0, 1], p=p.ravel())
                if keep:
                    guide[i * patch_size:i * patch_size + patch_size, j * patch_size:j * patch_size + patch_size] = 1
    return guide*255


def make_gt(img, labels, sigma=10, one_mask_per_point=False):
    """ Make the ground-truth for  landmark.
    img: the original color image
    labels: label with the Gaussian center(s) [[x0, y0],[x1, y1],...]
    sigma: sigma of the Gaussian.
    one_mask_per_point: masks for each point in different channels?
    """

    h, w = img.shape[:2]
    if labels is None:
        gt = make_gaussian((h, w), center=(h//2, w//2), sigma=sigma)
        gt_0 = np.zeros(shape=(h, w), dtype=np.float64)
        gt_0 = gt
        gt_1 = np.zeros(shape=(h, w), dtype=np.float64)
        
        gtout = np.zeros(shape=(h, w, 2))
        gtout[:, :, 0]=gt_0
        gtout[:, :, 1]=gt_1
        gtout = gtout.astype(dtype=img.dtype) #(0~1)        
        return gtout
    else:
        labels = np.array(labels)
        if labels.ndim == 1:
            labels = labels[np.newaxis]       
            gt_0 = np.zeros(shape=(h, w), dtype=np.float64)
            gt_1 = np.zeros(shape=(h, w), dtype=np.float64)
            gt_0 = np.maximum(gt_0, make_gaussian((h, w), center=labels[0, :], sigma=sigma))
           
        else:   
            gt_0 = np.zeros(shape=(h, w), dtype=np.float64)
            gt_1 = np.zeros(shape=(h, w), dtype=np.float64)
            for ii in range(1,labels.shape[0]):
                gt_1 = np.maximum(gt_1, make_gaussian((h, w), center=labels[ii, :], sigma=sigma))
            gt_0 = np.maximum(gt_0, make_gaussian((h, w), center=labels[0, :], sigma=sigma))
            
    gt = np.zeros(shape=(h, w, 2))
    gt[:, :, 0]=gt_0
    gt[:, :, 1]=gt_1

    gt = gt.astype(dtype=img.dtype) #(0~1)
    return gt  

def make_gt_grids(img, grids, bg_points, sigma=10, one_mask_per_point=False,meta=None,img_ = None,dev = None):
    '''Make the ground-truth for screened grids.

    Args:
        img:
        labels:
        sigma:
        one_mask_per_point:

    Returns:

    '''
    h, w = img.shape[:2]
    map_xor_new = np.zeros((h + 2, w + 2))
    map_xor_new[1:(h + 1), 1:(w + 1)] = grids[:, :]
    if grids.sum() == 0:
        gt = make_gaussian((h, w), center=(h // 2, w // 2), sigma=sigma)
        gt_0 = np.zeros(shape=(h, w), dtype=np.float64)
        gt_0 = gt
        gt_1 = np.zeros(shape=(h, w), dtype=np.float64)

        gtout = np.zeros(shape=(h, w, 2))
        gtout[:, :, 0] = gt_0
        gtout[:, :, 1] = gt_1
        gtout = gtout.astype(dtype=img.dtype)  # (0~1)
        return gtout
    else:
        # gt_0 = np.zeros(shape=(h, w), dtype=np.float64)
        gt_1 = np.zeros(shape=(h, w), dtype=np.float64)
        # 以下两句屏蔽负指导
        for ii in range(0, bg_points.shape[0]):
            gt_1 = np.maximum(gt_1, make_gaussian((h, w), center=bg_points[ii, :], sigma=sigma))
        # gt_0_temp = ndimage.distance_transform_edt(map_xor_new)
        # gt_0 = gt_0_temp[1:(h + 1), 1:(w + 1)]
        gt_0 = grids[:, :]
        gt_0 = np.where(gt_0 == 1, 255, 0)
        gt_0 = gt_0.astype(np.float32)
        gt_0 = cv2.GaussianBlur(gt_0, (7, 7), 0)
        # if dev > 3:
        #     gt_0 = cv2.GaussianBlur(gt_0, (7, 7), 0)
        # else:
        #     gt_0 = cv2.GaussianBlur(gt_0, (3, 3), 0)
        gt_0 = gt_0 / 255
    gt = np.zeros(shape=(h, w, 2))
    gt[:, :, 0] = gt_0
    gt[:, :, 1] = gt_1
    gt = gt.astype(dtype=img.dtype)  # (0~1)

    # result = gt_0 * 200
    # result_mask = np.zeros((img.shape[0], img.shape[1], 3))
    # result_mask[:, :, 1] = result
    # result_mask = result_mask.astype(np.uint8)
    # img_ = img_.astype(np.uint8)
    # result_final = cv2.addWeighted(img_, 1, result_mask, 0.9, 1)
    # path = "/home/zhupengqi/Inside-Outside-Guidance-master/results/Grid_coco_5010_guassian7_sample_ctw1500/guide/"
    # # # cv2.imwrite(path+meta['image']+'-'+meta['object']+'.png',result_final)
    # # # cv2.imwrite(path+meta['image']+'.png',result_final)
    # imsave(path+meta['image']+'-'+meta['object']+'.png',result_final)


    return gt


def cstm_normalize(im, max_value):
    """
    Normalize image to range 0 - max_value
    """
    imn = max_value*(im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn


def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key+':'+str(val)+'\n')
    log_file.close()
