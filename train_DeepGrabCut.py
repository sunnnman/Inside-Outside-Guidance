import socket
import timeit
from datetime import datetime
import os
import glob
from collections import OrderedDict
import logging
import sys

# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import upsample

# Tensorboard include
# from tensorboardX import SummaryWriter
from networks.mainnetwork import *
# Custom includes
from dataloaders.combine_dbs import CombineDBs as combine_dbs
from dataloaders import pascal, sbd
from networks import deeplab_resnet as resnet
# from layers.loss import class_balanced_cross_entropy_loss
from dataloaders import custom_transforms_ as tr
# from dataloaders.utils import generate_param_report
from networks.loss import *

gpu_id = 0
print('Using GPU: {} '.format(gpu_id))
# Setting parameters
use_sbd = False
nEpochs = 200  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
modelName = 'DeepGrabCut_IOGnet_pascal'

p = OrderedDict()  # Parameters to include in report
classifier = 'psp'  # Head classifier to use
p['trainBatch'] = 4  # Training batch size
testBatch = 5  # Testing batch size
useTest = True  # See evolution of the test set when training
nTestInterval = 10  # Run on test set every nTestInterval epochs
snapshot = 10  # Store a model every snapshot epochs
nInputChannels = 5  # Number of input channels (RGB + Distance Map of bounding box)
zero_pad_crop = True  # Insert zero padding when cropping the image
p['nAveGrad'] = 1  # Average the gradient of several iterations
p['lr'] = 1e-4  # Learning rate
p['wd'] = 5e-4  # Weight decay
p['momentum'] = 0.9  # Momentum

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

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
if not os.path.exists(os.path.join(save_dir, 'models')):
    os.makedirs(os.path.join(save_dir, 'models'))

# Network definition
modelName = 'deepgc_pascal'
# net = resnet.resnet101(1, pretrained=True, nInputChannels=nInputChannels, classifier=classifier)
net = Network(nInputChannels=nInputChannels, num_classes=1,
              backbone='resnet101',
              output_stride=16,
              sync_bn=None,
              freeze_bn=False,
              pretrained=True)


if resume_epoch == 0:
    print("Initializing from pretrained Deeplab-v2 model")
else:
    print("Initializing weights from: {}".format(
        os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
    net.load_state_dict(
        torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                   map_location=lambda storage, loc: storage)) # Load all tensors onto the CPU

train_params = [{'params': net.get_1x_lr_params(), 'lr': p['lr']},
                {'params': net.get_10x_lr_params(), 'lr': p['lr'] * 10}]

if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    net.cuda()

# if resume_epoch != nEpochs:
#     # Logging into Tensorboard
#     log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
#     writer = SummaryWriter(log_dir=log_dir)

    # Use the following optimizer
    optimizer = optim.SGD(train_params, lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
    p['optimizer'] = str(optimizer)

    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-15, 15), scales=(.75, 1.25)),
        tr.FixedResize(resolutions={'image': (450, 450), 'gt': (450, 450)}),
        tr.DistanceMap(v=0.15, elem='gt'),
        tr.ConcatInputs(elems=('image', 'distance_map')),
        tr.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        tr.FixedResize(resolutions={'image': (450, 450), 'gt': (450, 450)}),
        tr.DistanceMap(v=0.15, elem='gt'),
        tr.ConcatInputs(elems=('image', 'distance_map')),
        tr.ToTensor()])

    voc_train = pascal.VOCSegmentation(split='train', transform=composed_transforms_tr)
    voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts)

    if use_sbd:
        sbd_train = sbd.SBDSegmentation(split=['train', 'val'], transform=composed_transforms_tr, retname=True)
        db_train = combine_dbs([voc_train, sbd_train], excluded=[voc_val])
    else:
        db_train = voc_train

    trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=2)
    testloader = DataLoader(voc_val, batch_size=testBatch, shuffle=False, num_workers=2)

    # generate_param_report(os.path.join(save_dir, exp_name + '.txt'), p)

    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    running_loss_tr = 0.0
    running_loss_ts = 0.0
    aveGrad = 0
    print("Training Network")

    # Main Training and Testing Loop
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()

        net.train()
        for ii, sample_batched in enumerate(trainloader):

            inputs, gts = sample_batched['concat'], sample_batched['gt']

            # Forward-Backward of the mini-batch
            inputs, gts = Variable(inputs, requires_grad=True), Variable(gts)
            if gpu_id >= 0:
                inputs, gts = inputs.cuda(), gts.cuda()

            output = net.forward(inputs)
            output = upsample(output, size=(450, 450), mode='bilinear', align_corners=True)


            # Compute the losses, side outputs and fuse
            loss = class_balanced_cross_entropy_loss(output, gts, size_average=True, batch_average=True)
            running_loss_tr += loss.item()

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
            # if ii % num_img_tr == num_img_tr - 1:
            #     running_loss_tr = running_loss_tr / num_img_tr
            #     writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
            #     print('[Epoch: %d, numImages: %5d]' % (epoch, ii * p['trainBatch'] + inputs.data.shape[0]))
            #     print('Loss: %f' % running_loss_tr)
            #     running_loss_tr = 0
            #     stop_time = timeit.default_timer()
            #     print("Execution time: " + str(stop_time - start_time) + "\n")

            # Backward the averaged gradient
            loss /= p['nAveGrad']
            loss.backward()
            aveGrad += 1

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % p['nAveGrad'] == 0:
                # writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

        # Save the model
        if (epoch % snapshot) == snapshot - 1 and epoch != 0:
            torch.save(net.state_dict(), os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))

        # One testing epoch
        if useTest and epoch % nTestInterval == (nTestInterval - 1):
            net.eval()
            for ii, sample_batched in enumerate(testloader):
                inputs, gts = sample_batched['concat'], sample_batched['gt']

                # Forward pass of the mini-batch
                inputs, gts = Variable(inputs, requires_grad=True), Variable(gts)
                if gpu_id >= 0:
                    inputs, gts = inputs.cuda(), gts.cuda()

                with torch.no_grad():
                    output = net.forward(inputs)
                output = upsample(output, size=(450, 450), mode='bilinear', align_corners=True)

                # Compute the losses, side outputs and fuse
                loss = class_balanced_cross_entropy_loss(output, gts, size_average=True)
                running_loss_ts += loss.item()

                # Print stuff
                if ii % num_img_ts == num_img_ts - 1:
                    running_loss_ts = running_loss_ts / num_img_ts
                    print('[Epoch: %d, numImages: %5d]' % (epoch, ii * testBatch + inputs.data.shape[0]))
                    # writer.add_scalar('data/test_loss_epoch', running_loss_ts, epoch)
                    print('Loss: %f' % running_loss_ts)
                    running_loss_ts = 0

    # writer.close()
