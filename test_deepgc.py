from datetime import datetime
import scipy.misc as sm
from collections import OrderedDict
import glob
import numpy as np
import socket
from imageio import imsave
# PyTorch includes
import torch
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
from dataloaders.coco import *
from networks import deeplab_resnet as resnet
# Set gpu_id to -1 to run in CPU mode, otherwise set the id of the corresponding gpu
gpu_id = 5
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU: {} '.format(gpu_id))

# Setting parameters
resume_epoch = 100  # test epoch
nInputChannels = 5  # Number of input channels (RGB + heatmap of IOG points)
classifier = 'psp'  # Head classifier to use
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
modelName = 'Grid_deepgc_pascal_sample'
net = resnet.resnet101(1, pretrained=True, nInputChannels=nInputChannels, classifier=classifier)

# load pretrain_dict
# pretrain_dict = torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'))
pretrain_dict = torch.load("/home/zhupengqi/Inside-Outside-Guidance-master/runs/Grid_deepgc_pascal_sample/models/Grid_deepgc_pascal_sample_best_.pth")
result_save_dir = "/home/zhupengqi/Inside-Outside-Guidance-master/results/Grid_deepgc_pascal_sample_pascal/"

# print("Initializing weights from: {}".format(
#     os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
net.load_state_dict(pretrain_dict)
net.to(device)

# Generate result of the validation images
net.eval()
composed_transforms_ts = transforms.Compose([
    tr.CropFromMask(crop_elems=('image', 'gt','void_pixels'), relax=30, zero_pad=True),
    tr.FixedResize(resolutions={'gt': None, 'crop_image': (450, 450), 'crop_gt': (450, 450), 'crop_void_pixels': (450, 450)},flagvals={'gt':cv2.INTER_LINEAR,'crop_image':cv2.INTER_LINEAR,'crop_gt':cv2.INTER_LINEAR,'crop_void_pixels': cv2.INTER_LINEAR}),
    tr.IOGPoints(sigma=10, elem='crop_gt',pad_pixel=10),
    tr.ToImage(norm_elem='IOG_points'),
    tr.ConcatInputs(elems=('crop_image', 'IOG_points')),
    tr.ToTensor()])

composed_transforms_ts_grids = transforms.Compose([
        tr.CropFromMask(crop_elems=('image', 'gt', 'void_pixels', 'dev'), relax=30, zero_pad=True),
        tr.Grids(sigma=10, elem='crop_gt', pad_pixel=10),  # 根据resize后的crop获取是个负点击和一个正点击，并转为Gaussian map,拼接为2个通道
        tr.FixedResize(resolutions={'gt': None, 'crop_image': (450, 450), 'crop_gt': (450, 450), 'crop_void_pixels': (450, 450),
                                    'Grids': (450, 450)},
                       flagvals={'gt':cv2.INTER_LINEAR, 'crop_image': cv2.INTER_LINEAR, 'crop_gt': cv2.INTER_LINEAR,
                                 'crop_void_pixels': cv2.INTER_LINEAR, 'Grids': cv2.INTER_LINEAR}),
        tr.ToImage(norm_elem='Grids'),
        tr.ConcatInputs(elems=('crop_image', 'Grids')),
        tr.ToTensor()])

db_test = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts_grids, retname=True)
# db_test = GrabCutDataset(dataset_path='/home/datasets/GrabCut/', transform=composed_transforms_ts_grids)
# db_test = CocoSegmentation(split='val', transform=composed_transforms_ts_grids)
# db_test = CocoSegmentation(split='val', transform=composed_transforms_ts_grids)
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

print('Testing Network')


with torch.no_grad():
    for ii, sample_batched in enumerate(testloader):
        inputs, gts, metas = sample_batched['concat'], sample_batched['gt'], sample_batched['meta']
        # crop_gt = sample_batched['crop_gt']
        # crop_gt = crop_gt.to(torch.device('cpu'))
        inputs = inputs.to(device)
        fine_out = net.forward(inputs)

        outputs = fine_out.to(torch.device('cpu'))
        pred = np.transpose(outputs.data.numpy()[0, :, :, :], (1, 2, 0))
        pred = 1 / (1 + np.exp(-pred))
        pred = np.squeeze(pred)
        gt = tens2image(gts[0, :, :, :])
        # gt = tens2image(sample_batched['crop_gt'][0, :, :, :])

        bbox = get_bbox(gt, pad=30, zero_pad=True)
        if bbox is None:
            print('this bbox is None: '+str(int(metas['image_id'][0])) + '_' + str(int(metas['index'][0])) + '_mask.png')
            continue
        result = crop2fullmask(pred, bbox, gt, zero_pad=True, relax=0,mask_relax=False)

        # iou = get_iou(result, gt)
        # # iou = jaccard(gt,pred)
        # category = int(metas['category'][0])
        # iou_class_total[category] += iou
        # number_class_total[category] += 1
        #
        # iou_total = iou_total + iou
        # print('iou:', iou)
        # if ii % 10 == 0:
        #     print('------iou_avg:', iou_total / 10)
        #     iou_total = 0

        imsave(os.path.join(result_save_dir, metas['image'][0] + '-' + metas['object'][0] + '.png'), result)
        # imsave(os.path.join(result_save_dir, metas['image'][0] + '.png'), result)
        # imsave(os.path.join(result_save_dir,
                            # str(int(metas['image_id'][0])) + '_' + str(int(metas['index'][0])) + '_mask.png'), result)
    # for i in range(0, 21):
    #     if number_class_total[i] == 0:
    #         print('have no that class')
    #
    #     else:
    #         print('category ' + str(i) + ' IoU is：' + str(iou_class_total[i] / number_class_total[i]))
    #         miou += iou_class_total[i] / number_class_total[i]
    # print('total class iou is:' + miou)




