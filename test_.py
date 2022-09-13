from datetime import datetime

# import cv2.gapi
import scipy.misc as sm
from collections import OrderedDict
import glob
import numpy as np
import socket
from imageio import imsave
# PyTorch includes
import torch
import cv2
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# Custom includes
from dataloaders.combine_dbs import CombineDBs as combine_dbs
import dataloaders.pascal as pascal
import dataloaders.sbd as sbd
from dataloaders import custom_transforms as tr
from networks.loss import class_cross_entropy_loss
from dataloaders.helpers import *
from networks.mainnetwork import *
from dataloaders.grabcut import *
from dataloaders.sstem import *
from dataloaders.coco import *
from dataloaders.mscmr import *
from dataloaders.coco_city import *
# Set gpu_id to -1 to run in CPU mode, otherwise set the id of the corresponding gpu
gpu_id = 3
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU: {} '.format(gpu_id))

# Setting parameters
resume_epoch = 100  # test epoch
nInputChannels = 5  # Number of input channels (RGB + heatmap of IOG points)

# Results and model directories (a new directory is generated for every run)
save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
if resume_epoch == 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
else:
    run_id = 0
save_dir = os.path.join(save_dir_root, 'give_IOG_model_result')
# if not os.path.exists(os.path.join(save_dir, 'models')):
#     os.makedirs(os.path.join(save_dir, 'models'))

# Network definition
modelName = 'Grid_pascal_sbd_5010_guassian7_sample_'
net = Network(nInputChannels=nInputChannels,num_classes=1,
                backbone='resnet101',
                output_stride=16,
                sync_bn=None,
                freeze_bn=False)

# load pretrain_dict
# pretrain_dict = torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'))
# pretrain_dict = torch.load("/home/zhupengqi/Inside-Outside-Guidance-master/model/IOG_PASCAL_SBD.pth", map_location=device)
# pretrain_dict = torch.load("/home/zhupengqi/Inside-Outside-Guidance-master/runs/Grid_pascal_5010_guassian7_sample/models/Grid_pascal_5010_guassian7_sample__best_.pth", map_location=device)
pretrain_dict = torch.load("/home/zhupengqi/Inside-Outside-Guidance-master/runs/Grid_coco_5010_guassian7_sample/models/Grid_coco_5010_guassian7_sample_best_.pth", map_location=device)
# pretrain_dict = torch.load("/home/zhupengqi/Inside-Outside-Guidance-master/runs/Grid_pascal_sbd_5010_guassian7_sample_/models/Grid_pascal_5010_guassian7_sample__best_.pth", map_location=device)
result_save_dir = "/home/zhupengqi/Inside-Outside-Guidance-master/results/Grid_coco_5010_guassian7_sample_pascal/"

# print("Initializing weights from: {}".format(
#     os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
net.load_state_dict(pretrain_dict)
net.to(device)

# Generate result of the validation images
net.eval()
composed_transforms_ts = transforms.Compose([
    tr.CropFromMask(crop_elems=('image', 'gt','void_pixels'), relax=30, zero_pad=True),
    tr.FixedResize(resolutions={'gt': None, 'crop_image': (512, 512), 'crop_gt': (512, 512), 'crop_void_pixels': (512, 512)},flagvals={'gt':cv2.INTER_LINEAR,'crop_image':cv2.INTER_LINEAR,'crop_gt':cv2.INTER_LINEAR,'crop_void_pixels': cv2.INTER_LINEAR}),
    tr.IOGPoints(sigma=10, elem='crop_gt',pad_pixel=10),
    tr.ToImage(norm_elem='IOG_points'),
    tr.ConcatInputs(elems=('crop_image', 'IOG_points')),
    tr.ToTensor()])

composed_transforms_ts_grids = transforms.Compose([
        tr.CropFromMask(crop_elems=('image', 'gt', 'void_pixels', 'dev'), relax=30, zero_pad=True),
        tr.Grids(sigma=10, elem='crop_gt', pad_pixel=10),  # 根据resize后的crop获取是个负点击和一个正点击，并转为Gaussian map,拼接为2个通道
        tr.FixedResize(resolutions={'gt': None, 'crop_image': (512, 512), 'crop_gt': (512, 512), 'crop_void_pixels': (512, 512),
                                    'Grids': (512, 512)},
                       flagvals={'gt':cv2.INTER_LINEAR, 'crop_image': cv2.INTER_LINEAR, 'crop_gt': cv2.INTER_LINEAR,
                                 'crop_void_pixels': cv2.INTER_LINEAR, 'Grids': cv2.INTER_LINEAR}),
        tr.ToImage(norm_elem='Grids'),
        tr.ConcatInputs(elems=('crop_image', 'Grids')),
        tr.ToTensor()])

db_test = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts_grids, retname=True)
# db_test = CocoSegmentation_City(transform=composed_transforms_ts_grids, split='val')
# db_test = mscmrDataset(dataset_path='/home/zhupengqi/dataset/MSCMR/',transform=composed_transforms_ts_grids)
# db_test = sstemDataset(dataset_path='/home/zhupengqi/dataset/ssTEM/', transform=composed_transforms_ts_grids)
# db_test = CocoSegmentation(split='val', transform=composed_transforms_ts_grids)
# db_test = CocoSegmentation(split='val',year='2017', transform=composed_transforms_ts_grids)
# db_test = GrabCutDataset(dataset_path='/home/zhupengqi/dataset/GrabCut/', transform=composed_transforms_ts_grids)
# db_test = GrabCutDataset(dataset_path='/home/zhupengqi/dataset/Berkeley/', transform=composed_transforms_ts_grids)
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)


print('Testing Network')

def padding_bbox(bbox_,zero_pad = True,relax = 10):
    if zero_pad:
        x_min_bound = -np.inf
        y_min_bound = -np.inf
        x_max_bound = np.inf
        y_max_bound = np.inf

    x_min = max(int(bbox_[0]) - relax, x_min_bound)
    y_min = max(int(bbox_[1]) - relax, y_min_bound)
    x_max = min(int(bbox_[0])+int(bbox_[2]) + relax, x_max_bound)
    y_max = min(int(bbox_[1])+int(bbox_[3]) + relax, y_max_bound)
    return x_min, y_min, x_max, y_max


with torch.no_grad():
    pre_image = '00.tif'
    # result = None
    now = '0'
    for ii, sample_batched in enumerate(testloader):
        inputs, gts, metas = sample_batched['concat'], sample_batched['gt'], sample_batched['meta']
        # if now == metas['image'][0]:
        #     continue
        # else:
        #     img = cv2.imread(
        #         '/home/zhupengqi/Inside-Outside-Guidance-master/results/IOG_PASCAL_SBD_pascal/visible/JPEGImages/' +
        #         metas['image'][0] + '.jpg')
        #     cv2.imwrite(
        #         '/home/zhupengqi/Inside-Outside-Guidance-master/results/IOG_PASCAL_SBD_pascal/visible/visible/' +
        #         metas['image'][0] + '.jpg',img)
        #     now = metas['image'][0]

        # 注释从这开始
        crop_gt = sample_batched['crop_gt']
        crop_gt = crop_gt.to(torch.device('cpu'))
        inputs = inputs.to(device)
        coarse_outs1,coarse_outs2,coarse_outs3,coarse_outs4,fine_out = net.forward(inputs)

        outputs = fine_out.to(torch.device('cpu'))
        pred = np.transpose(outputs.data.numpy()[0, :, :, :], (1, 2, 0))
        pred = 1 / (1 + np.exp(-pred))
        pred = np.squeeze(pred)
        gt = tens2image(gts[0, :, :, :])
        # gt = tens2image(sample_batched['crop_gt'][0, :, :, :])

        bbox = get_bbox(gt, pad=30, zero_pad=True)
        # if bbox is None:
        #     print('this bbox is None: '+str(int(metas['image_id'][0])) + '_' + str(int(metas['index'][0])) + '_mask.png')
        #     continue

        result = crop2fullmask(pred, bbox, gt, zero_pad=True, relax=0,mask_relax=False)
        # result = np.where(result >= 0.2, 1, 0)
        # imsave(os.path.join(result_save_dir, metas['image'][0] + '-' + metas['object'][0] + '.png'), result)
        # # imsave(os.path.join(result_save_dir + 'mask/', metas['image'][0] + '.png'), result)
        # re = np.where(result > 0.2, 255, 0)
        # re= re.astype(np.uint8)
        # # cv2.imwrite(os.path.join(result_save_dir,  'temp.jpg'), re)
        # # tmp = cv2.imread(os.path.join(result_save_dir,  'temp.jpg'))
        # # gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        # contours, hierarchy = cv2.findContours(re, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # temp_image = np.zeros((re.shape[0],re.shape[1],3))
        # edge = cv2.drawContours(temp_image, contours, -1, (255, 255, 255), 2)
        # # edge = cv2.Laplacian(re, -1)
        # # edge = np.where(edge > 0, 255, 0)
        # # edge = np.where((result > 0.5) & (result < 0.7), 255, 0)
        #
        # # img = cv2.imread('/home/zhupengqi/Inside-Outside-Guidance-master/results/Grid_pascal_SBD_5010_guassian7_sample_sstem/raw/'+metas['image'][0])
        img = cv2.imread('/home/zhupengqi/Inside-Outside-Guidance-master/results/Grid_coco_5010_guassian7_sample_pascal/visible/JPEGImages/'+metas['image'][0]+'.jpg')
        # img = cv2.imread('/home/zhupengqi/Inside-Outside-Guidance-master/results/IOG_PASCAL_SBD_pascal/visible//JPEGImages/'+metas['image'][0]+'.jpg')
        # img = cv2.imread('/home/zhupengqi/Inside-Outside-Guidance-master/results/Grid_coco_5010_guassian7_sample_ctw1500/visible/text_image/'+metas['image'][0]+'.jpg')
        # img = cv2.imread('/home/datasets/coco/val2017/'+metas['image'][0]+'.jpg')

        # if pre_image == metas['image'][0]:
        #     img = cv2.imread(os.path.join(result_save_dir + '/visible/', metas['image'][0] + '.jpg'))
        # else:
        #     img = cv2.imread('/home/datasets/coco/val2014/'+metas['image'][0]+'.jpg')
        #     pre_image = metas['image'][0]

        # # img = cv2.imread('/home/zhupengqi/Inside-Outside-Guidance-master/results/IOG_PASCAL_SBD_mscmr/section_image/'+metas['image'][0])
        # # img = cv2.imread('/home/zhupengqi/Inside-Outside-Guidance-master/results/Grid_pascal_SBD_5010_guassian7_sample_cityscapes/visible/val_images/'+metas['image'][0]+'.png')
        # # img = cv2.imread('/home/zhupengqi/Inside-Outside-Guidance-master/results/Grid_pascal_5010_guassian7_sample_mscmr/visible/section_image/'+metas['gt'][0].split('_')[0]+'_'+metas['gt'][0].split('_')[1]+'_'+metas['gt'][0].split('_')[3]+'.png')
        result_ = np.where(result>=0.5,1,0)
        result_mask = np.zeros((gt.shape[0],gt.shape[1],3))

        re = result_.astype(np.uint8)
        edge = cv2.Laplacian(re, -1)
        edge = np.where(edge > 0, 255, 0)
        edge = edge.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=1)

        # ii = int(metas['gt'][0].split('.')[0][-1]) - 1
        i = ii % 4
        if i == 0:
            result_mask[:, :, 2] = result_ * 64
        elif i == 1:
            # result_mask[:, :, 0] = result_ * 29
            result_mask[:, :, 0] = result_ * 64
            result_mask[:, :, 2] = result_ * 26
        elif i == 2:
            result_mask[:, :, 1] = result_ * 64
        elif i == 3:
            result_mask[:, :, 1] = result_ * 64
            result_mask[:, :, 2] = result_ * 64

        result_mask = result_mask.astype(np.uint8)
        result_final = cv2.addWeighted(img, 0.7, result_mask, 1, 0)
        result_final = result_final.astype(np.uint8)
        # result_final = np.where(edge > 0, 1, result_final)
        result_final = np.where(result_[..., None], result_final, img)
        # #
        edge = np.expand_dims(edge, axis=-1)
        result_final = np.where(edge == 255, 0, result_final)
        #
        #
        # # imsave(os.path.join(result_save_dir + '/visible/', metas['image'][0] + '-' + metas['object'][0] + '.png'), result_final)
        # # cv2.imwrite('/home/zhupengqi/Inside-Outside-Guidance-master/results/IOG_PASCAL_SBD_mscmr/section_image/' +metas['image'][0], result_final)
        cv2.imwrite(os.path.join(result_save_dir + '/visible/JPEGImages/', metas['image'][0] + '.jpg'), result_final)
        # cv2.imwrite(os.path.join(result_save_dir + '/visible/', metas['image'][0] + '.jpg'), result_final)
        # 注释到这
        # cv2.imwrite(os.path.join(result_save_dir + '/visible/val_images/', metas['image'][0] + '.png'), result_final)
        # imsave(os.path.join(result_save_dir + '/mask/', metas['image'][0] + '-' + metas['object'][0] + '.png'), result)
        # imsave(os.path.join(result_save_dir, metas['image'][0] + '.png'), result)
        # imsave(os.path.join(result_save_dir, metas['gt'][0] + '.png'), result)
        # imsave(os.path.join(result_save_dir,
        #                     str(int(metas['image_id'][0])) + '_' + str(int(metas['index'][0])) + '_mask.png'), result)

        # bbox = metas['bbox']
        # bbox = padding_bbox(bbox)
        # pred = cv2.resize(pred,(bbox[2]-bbox[0],bbox[3]-bbox[1]))
        # result = crop2fullmask(pred, bbox, gt, zero_pad=True, relax=0, mask_relax=False)
        # # result = crop2fullmask(pred, bbox, gt, result, zero_pad=True, relax=0, mask_relax=False)
        # result_ = np.where(result > 0.8, 1, 0)
        # result_mask = np.zeros((gt.shape[0], gt.shape[1], 3))
        #
        # re = np.where(result > 0.8, 1, 0)
        # re= re.astype(np.uint8)
        # edge = cv2.Laplacian(re, -1)
        # edge = np.where(edge > 0, 255, 0)
        #
        # i = ii % 4
        # if i == 0:
        #     # result_mask[:, :, 2] = result_ * 139
        #     result_mask[:, :, 2] = result_ * 64
        # elif i == 1:
        #     # result_mask[:, :, 0] = result_ * 29
        #     # result_mask[:, :, 1] = result_ * 139
        #     result_mask[:, :, 1] = result_ * 64
        # elif i == 2:
        #     result_mask[:, :, 0] = result_ * 64
        #     # result_mask[:, :, 0] = result_ * 138
        #     # result_mask[:, :, 2] = result_ * 26
        # elif i == 3:
        #     # result_mask[:, :, 1] = result_ * 106
        #     # result_mask[:, :, 2] = result_ * 119
        #     result_mask[:, :, 1] = result_ * 64
        #     result_mask[:, :, 2] = result_ * 64
        #
        # result_mask = result_mask.astype(np.uint8)
        # result_final = cv2.addWeighted(img, 0.5, result_mask,1,0)
        # result_final = result_final.astype(np.uint8)
        # result_final = np.where(result_[..., None], result_final, img)
        #
        # edge = np.expand_dims(edge, axis=-1)
        # result_final = np.where(edge == 255, 0, result_final)
        #
        # cv2.imwrite(os.path.join(result_save_dir + '/raw/', metas['image'][0]),
        #             result_final)

        # if metas['image'][0] == now_image:
        #     result = crop2fullmask(pred, bbox,gt, result, zero_pad=True, relax=0, mask_relax=False)
        #     # result = crop2fullmask(pred, bbox, gt, result, zero_pad=True, relax=0, mask_relax=False)
        #     if ii == len(testloader) - 2:
        #         print(os.path.join(result_save_dir, '19.png'))
        #         imsave(os.path.join(result_save_dir, '19.png'), result)
        #     # result = crop2fullmask(pred, bbox, gt, zero_pad=True, relax=0, mask_relax=False)
        # else:
        #     print(os.path.join(result_save_dir, now_image[0:2] + '.png'))
        #     imsave(os.path.join(result_save_dir, now_image[0:2] + '.png'), result)
        #     result = None
        #     now_image = metas['image'][0]
        #     result = crop2fullmask(pred, bbox, gt, result, zero_pad=True, relax=0, mask_relax=False)





