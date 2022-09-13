import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.misc as sm
from mypath import Path
from networks.backbone import build_backbone
from networks.CoarseNet import CoarseNet
from networks.FineNet import FineNet
from dataloaders.helpers import *

affine_par = True
device = torch.device('cuda:0')
is_guide = False


def make_gaussian(size, sigma=10, center=None, d_type=np.float64):
    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]
    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(d_type)


def getPositon(distance_transform):
    a = np.mat(distance_transform)
    raw, column = a.shape  # get the matrix of a raw and column
    _positon = np.argmax(a)  # get the index of max in the a
    m, n = divmod(_positon, column)
    raw = m
    column = n
    return raw, column


def generate_distance_map(map_xor, points_center, points_bg, gt, pred):
# def generate_distance_map(map_xor, points_center, points_bg, gt):
    '''map_xor, points_center, points_bg, gt'''
    distance_transform = ndimage.distance_transform_edt(map_xor)
    occupy_list = [0,0,0,0,0]
    raw, column = getPositon(distance_transform)
    for i in range(10,60,10):
        size = int(i/2)
        area = map_xor[raw-size:raw+size,column-size:column+size]
        occupy = area.sum() / (i*i)
        if occupy != 1:
            index = int(i /10) - 1
            occupy_list[index] = occupy

    a = np.mat(occupy_list)
    _positon = np.argmax(a)
    sigma = int(((_positon+1) * 10) / 2)
    # print('sigma: ' + str(sigma))



    gt_0 = np.zeros(shape=(gt.shape[0], gt.shape[0]), dtype=np.float64)
    gt_0[column, raw] = 1
    # gt_0[raw, column] = 1
    map_center = np.sum(np.logical_and(gt_0, gt))
    map_bg = np.sum(np.logical_and(gt_0, 1 - gt))
    # sigma = 5
    # sigma = 10
    # sigma = 7
    new_click = np.zeros((512,512),np.uint8)
    new_click[raw-sigma:raw+sigma,column-sigma:column+sigma] = 1

    if map_center == 1:  # 如果是前景类

        # new_click = make_gaussian((gt.shape[0], gt.shape[0]), center=[column, raw], sigma=sigma)
        if points_center[raw, column] > 0:  # 如果本来就是grid指导
            new_click = new_click * 50
        else:
            new_click = new_click * 255

        # points_center = points_center * 255
        points_center = points_center + new_click
        # pred = pred + new_click
        # pred = np.where(pred > 255, 255, pred)
        min = np.min(points_center)
        max = np.max(points_center)
        points_center = (points_center - min) / (max - min)
        points_center = torch.from_numpy(points_center)
        points_center = points_center * 255
        # pred = 255 * np.maximum(points_center / 255,
        #                                  make_gaussian((gt.shape[0], gt.shape[0]), center=[column, raw], sigma=sigma*3))
    elif map_bg == 1:
        new_click = new_click * 255
        # points_bg = points_bg + new_click
        # points_bg = np.where(points_bg > 255, 255,points_bg)

        # points_bg = 255 * np.maximum(points_bg / 255,
        #                              make_gaussian((gt.shape[0], gt.shape[0]), center=[column, raw], sigma=sigma))
        # points_center = points_center - points_bg
        points_center = points_center - new_click
        # pred = pred - points_bg
        points_center = np.where(points_center < 0, 0, points_center)
    # attention = np.ones_like(points_center)
    # new_click = make_gaussian((gt.shape[0], gt.shape[0]), center=[column, raw], sigma=sigma)

    # if map_center == 1:  # 如果是前景类
    #     pred = pred + new_click
    #     attention = pred
    # elif map_bg == 1:
    #     pred = pred - new_click
    #     pred = np.where(pred < 0, 0, pred)
    #     points_bg = points_bg + new_click
    #     points_bg = np.where(points_bg > 255, 255, points_bg)
    # else:
    #     print('error')

    # if map_center == 1:  # 如果是前景类
    #     if points_center[raw, column] > 0:  # 如果本来就是grid指导
    #         new_click = new_click * 50
    #     else:
    #         new_click = new_click * 255
    #     points_center = points_center + new_click
    #     attention = points_center
    #     points_center = np.where(points_center > 255, 255, points_center)
    #
    # elif map_bg == 1:
    #     new_click = new_click * 255
    #     points_center = points_center - new_click
    #     points_center = np.where(points_center < 0, 0, points_center)
    #     points_bg = points_bg + new_click
    #     points_bg = np.where(points_bg > 255, 255, points_bg)
    # else:
    #     print('error')

    # if map_center == 1:
    #     min = np.min(attention)
    #     max = np.max(attention)
    #     attention = (attention - min) / (max - min)
    # points_center = grids_(pred)
    # points_center = grids_(points_center)
    # points_center = points_center * attention
    # points_bg = points_bg * attention
    pointsgt_new = np.zeros(shape=(gt.shape[0], gt.shape[0], 2))
    pointsgt_new[:, :, 0] = points_center
    # pointsgt_new[:, :, 0] = pred
    pointsgt_new[:, :, 1] = points_bg
    pointsgt_new = pointsgt_new.astype(dtype=np.uint8)
    pointsgt_new = pointsgt_new.transpose((2, 0, 1))
    pointsgt_new = pointsgt_new[np.newaxis, :, :, :]
    pointsgt_new = torch.from_numpy(pointsgt_new)
    return pointsgt_new




def iou_cal(pre, gts, extreme_points, mask_thres=0.5):
# def iou_cal(pre, gts, extreme_points, mask_thres=0.5):
    '''
    根据上一次预测的结果和上一次的guide map,计算iou生成这次的guide map
    Args:
        pre:上一次预测结果
        gts:ground ture
        extreme_points:上一次guide map
        mask_thres:

    Returns:iou 本次guide map

    '''
    iu_ave = 0
    distance_map_new = torch.zeros(extreme_points.shape)
    for jj in range(int(pre.shape[0])):
        pred = np.transpose(pre.cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
        pred = 1 / (1 + np.exp(-pred))
        pred = np.squeeze(pred)

        gts = gts.cpu()
        gt = tens2image(gts[jj, :, :, :])
        extreme_points = extreme_points.cpu()
        points_center = tens2image(extreme_points[jj, 0:1, :, :])
        points_bg = tens2image(extreme_points[jj, 1:2, :, :])
        gt = (gt > mask_thres)
        pred = (pred > mask_thres)
        pred_ = pred * 255
        map_and = np.logical_and(pred, gt)
        map_or = np.logical_or(pred, gt)
        map_xor = np.bitwise_xor(pred, gt)
        if np.sum(map_or) == 0:
            iu = 0
        else:
            iu = np.sum(map_and) / np.sum(map_or)
        iu_ave = iu_ave + iu
        distance_map_new[jj, :, :, :] = generate_distance_map(map_xor, points_center, points_bg, gt, pred_)
        # distance_map_new[jj, :, :, :] = generate_distance_map(map_xor, points_center, points_bg, gt)
    iu_ave = iu_ave / pre.shape[0]
    distance_map_new = distance_map_new.cuda()
    return iu_ave, distance_map_new


class PSPModule(nn.Module):
    """
    Pyramid Scene Parsing module
    """

    def __init__(self, in_features=2048, out_features=512, sizes=(1, 2, 3, 6), n_classes=1):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage_1(in_features, size) for size in sizes])
        self.bottleneck = self._make_stage_2(in_features * (len(sizes) // 4 + 1), out_features)
        self.relu = nn.ReLU()

    def _make_stage_1(self, in_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_features, in_features // 4, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(in_features // 4, affine=affine_par)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def _make_stage_2(self, in_features, out_features):
        conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features, affine=affine_par)
        relu = nn.ReLU(inplace=True)

        return nn.Sequential(conv, bn, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages]
        priors.append(feats)
        bottle = self.relu(self.bottleneck(torch.cat(priors, 1)))
        return bottle


class SegmentationNetwork(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21, nInputChannels=3,
                 sync_bn=True, freeze_bn=False):
        super(SegmentationNetwork, self).__init__()
        output_shape = 128
        channel_settings = [512, 1024, 512, 256]
        self.Coarse_net = CoarseNet(channel_settings, output_shape, num_classes)
        self.Fine_net = FineNet(channel_settings[-1], output_shape, num_classes)
        BatchNorm = nn.BatchNorm2d
        self.backbone = build_backbone(backbone, output_stride, BatchNorm, nInputChannels, pretrained=False)
        self.psp4 = PSPModule(in_features=2048, out_features=512, sizes=(1, 2, 3, 6), n_classes=256)
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
        # self.iog_points = nn.Sequential(nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1, bias=False),
        #                                 nn.BatchNorm2d(64),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
        #                                 nn.BatchNorm2d(128),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
        #                                 nn.BatchNorm2d(256),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
        #                                 nn.BatchNorm2d(256),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False),
        #                                 nn.BatchNorm2d(64),
        #                                 nn.ReLU())

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input_,gts,refinement_num_max):
    # def forward(self, input, IOG_points, gts, refinement_num_max):
        input = input_
        outlist = []
        img = input_[:,0:3,:,:]
        dismap = input_[:,3:5,:,:]

        for refinement_num in range(0, refinement_num_max):
            low_level_feat_4, low_level_feat_3, low_level_feat_2, low_level_feat_1 = self.backbone(input)
            low_level_feat_4 = self.psp4(low_level_feat_4)
            res_out = [low_level_feat_4, low_level_feat_3, low_level_feat_2, low_level_feat_1]
            coarse_fms, coarse_outs = self.Coarse_net(res_out)
            fine_out = self.Fine_net(coarse_fms)
            out_512 = F.upsample(fine_out, size=(512, 512), mode='bilinear', align_corners=True)
            iou_i, distance_map_new = iou_cal(out_512, gts, dismap)
            dismap = distance_map_new
            input = torch.cat((img,distance_map_new), dim = 1)
            # iou_i, distance_map_new = iou_cal(out_512, gts, distance_map_512)
            input = input.to(device)
            out = [coarse_outs[0], coarse_outs[1], coarse_outs[2], coarse_outs[3], fine_out, iou_i]
            outlist.append(out)
        return outlist  # 套了两层

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        # modules = [self.Coarse_net, self.Fine_net, self.psp4, self.upsample, self.iog_points]
        modules = [self.Coarse_net, self.Fine_net, self.psp4, self.upsample]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


def Network(nInputChannels=5, num_classes=1, backbone='resnet101', output_stride=16,
            sync_bn=None, freeze_bn=False, pretrained=False):
    model = SegmentationNetwork(nInputChannels=nInputChannels, num_classes=num_classes, backbone=backbone,
                                output_stride=output_stride, sync_bn=sync_bn, freeze_bn=freeze_bn)
    if pretrained:
        load_pth_name = Path.models_dir()
        pretrain_dict = torch.load(load_pth_name, map_location=lambda storage, loc: storage)
        conv1_weight_new = np.zeros((64, 5, 7, 7))
        conv1_weight_new[:, :3, :, :] = pretrain_dict['conv1.weight'].cpu().data
        pretrain_dict['conv1.weight'] = torch.from_numpy(conv1_weight_new)
        state_dict = model.state_dict()
        model_dict = state_dict
        for k, v in pretrain_dict.items():
            kk = 'backbone.' + k
            if kk in state_dict:
                model_dict[kk] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
    return model
