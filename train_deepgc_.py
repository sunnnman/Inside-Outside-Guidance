from datetime import datetime
import scipy.misc as sm
from collections import OrderedDict
import glob
import numpy as np
import socket
import timeit
import logging
import sys
# PyTorch includes
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloaders.coco import *
# Custom includes
from dataloaders.combine_dbs import CombineDBs as combine_dbs
import dataloaders.pascal as pascal
import dataloaders.sbd as sbd
from dataloaders import custom_transforms as tr
from dataloaders.helpers import *
from networks.loss import class_balanced_cross_entropy_loss
from networks.mainnetwork import *
from dataloaders.grabcut import *
from networks import deeplab_resnet as resnet
from torch.nn.functional import upsample

# Set gpu_id to -1 to run in CPU mode, otherwise set the id of the corresponding gpu
gpu_id = 4
device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU: {} '.format(gpu_id))

# Setting parameters
use_sbd = False  # train with SBD
nEpochs = 200  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
p = OrderedDict()  # Parameters to include in report
classifier = 'psp'  # Head classifier to use
p['trainBatch'] = 4  # Training batch size 5
snapshot = 20  # Store a model every snapshot epochs
nInputChannels = 5  # Number of input channels (RGB + heatmap of extreme points)
p['nAveGrad'] = 1  # Average the gradient of several iterations
p['lr'] = 1e-4  # Learning rate
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.9  # Momentum

modelName = 'Grid_deepgc_pascal_sample'


def setup_logger(name, save_dir, filename="log.txt", mode='w'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode=mode)  # 'a+' for add, 'w' for overwrite
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


running_loss_tr_min = 200000

model_save_dir = './runs/Grid_deepgc_pascal_sample'
log_dir = './runs/Grid_deepgc_pascal_sample/log'
logger = setup_logger("semantic_segmentation", log_dir, filename='{}_log.txt'.format(
    modelName))

# Results and model directories (a new directory is generated for every run)
save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
if resume_epoch == 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
else:
    run_id = 0
save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))
if not os.path.exists(os.path.join(save_dir, 'models')):
    os.makedirs(os.path.join(save_dir, 'models'))

# Network definition
# net = Network(nInputChannels=nInputChannels, num_classes=1,
#               backbone='resnet101',
#               output_stride=16,
#               sync_bn=None,
#               freeze_bn=False,
#               pretrained=True)

net = resnet.resnet101(1, pretrained=True, nInputChannels=nInputChannels, classifier=classifier)

if resume_epoch == 0:  # 这个是跑到一半如果中断什么的重新跑，net从mypath里加载的模型才是预训练模型
    print("Initializing from pretrained model")
else:
    # print("Initializing weights from: {}".format(
    #     os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
    # net.load_state_dict(
    #     torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
    #                map_location=lambda storage, loc: storage))
    print('Initializing weights from:/home/zhupengqi/Inside-Outside-Guidance-master/model/IOG_PASCAL.pth')
    net.load_state_dict(
        torch.load(
            "/home/zhupengqi/Inside-Outside-Guidance-master/runs/pretrained_IOG_pascal_0513/models/IOG_pascal_epoch-49.pth",
            map_location=lambda storage, loc: storage))

# train_params = [{'params': net.get_1x_lr_params(), 'lr': p['lr']},
#                 {'params': net.get_10x_lr_params(), 'lr': p['lr'] * 10}]
train_params = [{'params': resnet.get_1x_lr_params(net), 'lr': p['lr']},
                {'params': resnet.get_10x_lr_params(net), 'lr': p['lr'] * 10}]

net.to(device)

if resume_epoch != nEpochs:
    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())

    # Use the following optimizer
    optimizer = optim.SGD(train_params, lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
    p['optimizer'] = str(optimizer)

    # Preparation of the data loaders
    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
        tr.CropFromMask(crop_elems=('image', 'gt', 'void_pixels'), relax=30, zero_pad=True),
        tr.FixedResize(resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512), 'crop_void_pixels': (512, 512)},
                       flagvals={'crop_image': cv2.INTER_LINEAR, 'crop_gt': cv2.INTER_LINEAR,
                                 'crop_void_pixels': cv2.INTER_LINEAR}),
        tr.IOGPoints(sigma=10, elem='crop_gt', pad_pixel=10),
        tr.ToImage(norm_elem='IOG_points'),
        tr.ConcatInputs(elems=('crop_image', 'IOG_points')),
        tr.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        tr.CropFromMask(crop_elems=('image', 'gt', 'void_pixels'), relax=30, zero_pad=True),
        tr.FixedResize(resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512), 'crop_void_pixels': (512, 512)},
                       flagvals={'crop_image': cv2.INTER_LINEAR, 'crop_gt': cv2.INTER_LINEAR,
                                 'crop_void_pixels': cv2.INTER_LINEAR}),
        tr.IOGPoints(sigma=10, elem='crop_gt', pad_pixel=10),
        tr.ToImage(norm_elem='IOG_points'),
        tr.ConcatInputs(elems=('crop_image', 'IOG_points')),
        tr.ToTensor()])

    composed_transforms_tr_grids = transforms.Compose([
        tr.RandomHorizontalFlip(),  # 水平翻转
        tr.ScaleNRotate(rots=(-15, 15), scales=(.75, 1.25)),  # 随机旋转放大缩小
        tr.CropFromMask(crop_elems=('image', 'gt', 'void_pixels', 'dev'), relax=30, zero_pad=True),
        # 根据mask crop出image gt
        tr.Grids(sigma=10, elem='crop_gt', pad_pixel=10),  # 根据resize后的crop获取是个负点击和一个正点击，并转为Gaussian map,拼接为2个通道
        tr.FixedResize(resolutions={'crop_image': (450, 450), 'crop_gt': (450, 450), 'crop_void_pixels': (450, 450),
                                    'Grids': (450, 450)},
                       flagvals={'crop_image': cv2.INTER_LINEAR, 'crop_gt': cv2.INTER_LINEAR,
                                 'crop_void_pixels': cv2.INTER_LINEAR, 'Grids': cv2.INTER_LINEAR}),
        tr.ToImage(norm_elem='Grids'),  # 将原本的Gaussian
        tr.ConcatInputs(elems=('crop_image', 'Grids')),
        tr.ToTensor()])

    composed_transforms_ts_grids = transforms.Compose([
        tr.CropFromMask(crop_elems=('image', 'gt', 'void_pixels', 'dev'), relax=30, zero_pad=True),
        tr.Grids(sigma=10, elem='crop_gt', pad_pixel=10),  # 根据resize后的crop获取是个负点击和一个正点击，并转为Gaussian map,拼接为2个通道
        tr.FixedResize(resolutions={'crop_image': (450, 450), 'crop_gt': (450, 450), 'crop_void_pixels': (450, 450),
                                    'Grids': (450, 450)},
                       flagvals={'crop_image': cv2.INTER_LINEAR, 'crop_gt': cv2.INTER_LINEAR,
                                 'crop_void_pixels': cv2.INTER_LINEAR, 'Grids': cv2.INTER_LINEAR}),
        tr.ToImage(norm_elem='Grids'),
        tr.ConcatInputs(elems=('crop_image', 'Grids')),
        tr.ToTensor()])

    voc_train = pascal.VOCSegmentation(split='train', transform=composed_transforms_tr_grids)
    voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts_grids)

    # voc_train = CocoSegmentation(split='train', transform=composed_transforms_tr_grids)

    # voc_train = GrabCutDataset(dataset_path='/home/datasets/GrabCut/', transform=composed_transforms_tr)
    # voc_val = GrabCutDataset(dataset_path='/home/datasets/GrabCut/', transform=composed_transforms_ts_grids)

    if use_sbd:
        sbd = sbd.SBDSegmentation(split=['train', 'val'], transform=composed_transforms_tr_grids, retname=True)
        # db_train = combine_dbs([voc_train, sbd], excluded=[voc_val])
    else:
        db_train = voc_train

    p['dataset_train'] = str(db_train)
    p['transformations_train'] = [str(tran) for tran in composed_transforms_tr_grids.transforms]
    trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=2)


    def get_iou(pre, gt, mask_thres=0.5):
        iu_ave = 0
        gt = tens2image(gt[0, :, :, :])
        gt = (gt > mask_thres)
        pre = (pre > mask_thres)
        map_and = np.logical_and(pre, gt)
        map_or = np.logical_or(pre, gt)
        # map_xor = np.bitwise_xor(pred, gt)
        if np.sum(map_or) == 0:
            iu = 0
        else:
            iu = np.sum(map_and) / np.sum(map_or)

        return iu


    # Train variables
    num_img_tr = len(trainloader)
    running_loss_tr = 0.0
    aveGrad = 0
    print("Training Network")
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()
        epoch_loss = []
        net.train()
        iou = 0
        for ii, sample_batched in enumerate(trainloader):
            gts = sample_batched['crop_gt']
            inputs = sample_batched['concat']
            # void_pixels = sample_batched['crop_void_pixels']
            inputs.requires_grad_()
            inputs, gts = inputs.to(device), gts.to(device)
            output = net.forward(inputs)
            output = upsample(output, size=(450, 450), mode='bilinear', align_corners=True)

            # Compute the losses, side outputs and fuse
            loss = class_balanced_cross_entropy_loss(output, gts, size_average=True, batch_average=True)


            # 计算iou
            outputs = output.to(torch.device('cpu'))
            crop_gt = gts.to(torch.device('cpu'))
            pred = np.transpose(outputs.data.numpy()[0, :, :, :], (1, 2, 0))
            pred = 1 / (1 + np.exp(-pred))
            pred = np.squeeze(pred)
            iou += get_iou(pred, crop_gt)

            if ii % 10 == 0:
                # print('Epoch',epoch,'step',ii,'loss',loss)
                logger.info('Epoch:{:d} || step:{:d} || loss:{:.4f} || iou:{:.2f}'.format(epoch, ii, loss, iou / 10))
                iou = 0
            running_loss_tr += loss.item()

            # Print stuff
            if ii % num_img_tr == num_img_tr - 1 - p['trainBatch']:
                running_loss_tr = running_loss_tr / num_img_tr
                # print('[Epoch: %d, numImages: %5d]' % (epoch, ii*p['trainBatch']+inputs.data.shape[0]))
                # print('Loss: %f' % running_loss_tr)
                logger.info('[Epoch: %d, numImages: %5d]' % (epoch, ii * p['trainBatch'] + inputs.data.shape[0]))
                logger.info('Loss: %f' % running_loss_tr)
                # 保存当前最优模型
                if running_loss_tr_min > running_loss_tr:
                    running_loss_tr_min = running_loss_tr
                    torch.save(net.state_dict(), os.path.join(model_save_dir, 'models', modelName + '_best_' + '.pth'))
                running_loss_tr = 0
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")

            # Backward the averaged gradient
            loss /= p['nAveGrad']
            loss.backward()
            aveGrad += 1

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % p['nAveGrad'] == 0:
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

        # Save the model
        if (epoch % snapshot) == snapshot - 1 and epoch != 0:
            torch.save(net.state_dict(),
                       os.path.join(model_save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))
