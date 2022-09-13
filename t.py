# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 't.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import json

import cv2
import numpy as np
from imageio import imsave
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QRect
from networks import deeplab_resnet as resnet
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtWidgets import QFileDialog, QLabel, QGridLayout, QWidget
import os
import shutil
import torch
# from networks import refinementnetwork as rn
from networks.mainnetwork import *
# from networks.refinementnetwork import *
import random
import dataloaders.helpers as helpers

gpu_id = 4
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU: {} '.format(gpu_id))
net = Network(nInputChannels=5,num_classes=1,
                backbone='resnet101',
                output_stride=16,
                sync_bn=None,
                freeze_bn=False)
# net = resnet.resnet101(1, pretrained=True, nInputChannels=5, classifier='psp')
pretrain_dict = torch.load("/home/zhupengqi/Inside-Outside-Guidance-master/runs/Grid_pascal_5010_guassian7_sample/models/Grid_pascal_5010_guassian7_sample__best_.pth",map_location=device)
# pretrain_dict = torch.load("/home/zhupengqi/remote_project/Inside-Outside-Guidance-master/runs/Grids_pascal_max70_gaussian7_hw/models/Grids_pascal_max70_gaussian7_bbox_best_.pth",map_location=device)
# pretrain_dict = torch.load("/home/zhupengqi/remote_project/Inside-Outside-Guidance-master/model/Grid_deepgc_pascal_sample_best_.pth")
# pretrain_dict = torch.load("/home/zhupengqi/remote_project/Inside-Outside-Guidance-master/runs/Grid_pascal_max70_gaussian7_hw_80%_resnet50/models/Grid_pascal_max70_gaussian7_hw_80%_resnet50_best_.pth")
net.load_state_dict(pretrain_dict)
net.to(device)
net.eval()


click_select = False
click_cancel = False
correct_flag = False
object_num = 0
r_count = 0
c_count = 0

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1500, 1200)
        self.patch_size = 50
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(30, 20, 1024, 500))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setLineWidth(1)
        self.frame.setMidLineWidth(1)
        self.frame.setObjectName("frame")
        self.frame.move(30,30)
        self.layout = QGridLayout(self.frame)
        self.layout.setSpacing(5)
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(30, 1100, 1024, 120))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_2.setLineWidth(1)
        self.frame_2.setMidLineWidth(1)
        self.frame_2.setObjectName("frame_2")
        self.frame_2.move(30,600)
        self.pushButton = QtWidgets.QPushButton(self.frame_2)
        self.pushButton.setGeometry(QtCore.QRect(30, 10, 131, 30))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.butttonclicked)

        self.pushButton_done = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_done.setGeometry(QtCore.QRect(30, 45, 131, 30))
        self.pushButton_done.clicked.connect(self.donebuttonclicked)
        self.pushButton_done.setText('DONE')

        self.pushButton_clear = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_clear.setGeometry(QtCore.QRect(30, 85, 131, 30))
        self.pushButton_clear.clicked.connect(self.nextobjectbuttonclicked)
        self.pushButton_clear.setText('Next Object')

        self.pushButton_pre = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_pre.setGeometry(QtCore.QRect(800, 40, 69, 30))
        self.pushButton_pre.clicked.connect(self.prebutttonclicked)
        self.pushButton_pre.setText('PRE')

        self.pushButton_next = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_next.setGeometry(QtCore.QRect(880, 40, 69, 30))
        self.pushButton_next.clicked.connect(self.nextbutttonclicked)
        self.pushButton_next.setText('NEXT')

        # self.pushButton_next = QtWidgets.QPushButton(self.frame_2)
        # self.pushButton_next.setGeometry(QtCore.QRect(980, 40, 30, 30))
        # self.pushButton_next.clicked.connect(self.doneall_savejson)
        # self.pushButton_next.setText('OK')

        self.label_scale = QLabel(self.frame_2)
        self.horizontalSlider = QtWidgets.QSlider(self.frame_2)
        self.horizontalSlider.setGeometry(QtCore.QRect(240, 40, 521, 16))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalSlider.setMinimum(10)
        self.horizontalSlider.setMaximum(70)
        self.horizontalSlider.setValue(50)
        self.horizontalSlider.setSingleStep(10)
        self.horizontalSlider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksAbove)
        self.horizontalSlider.setTickInterval(10)
        self.horizontalSlider.valueChanged.connect(self.sliderValueChanged)

        MainWindow.setCentralWidget(self.centralwidget)
        # self.layout.setSpacing(0)
        self.frame.setLayout(self.layout)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.file_name_list = None
        self.h = 0
        self.w = 0
        self.index = 0

        # self.grids_record = {'record':[]}


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "LOAD"))

    def butttonclicked(self):
        # self.file_name_list = os.listdir('./click_image/')
        self.file_name_list = os.listdir('/home/datasets/GrabCut/data_GT/')
        self.loadfile()

    def prebutttonclicked(self):
        self.index -= 1
        self.label_scale.setText(str(self.index + 1))
        if self.index < 0:
            self.index = 0
            print('前面没有啦')
        self.cleargrids()
        global object_num
        object_num = 0
        self.loadfile()

    def nextbutttonclicked(self):
        self.index += 1
        self.label_scale.setText(str(self.index+1))
        if self.index > len(self.file_name_list)-1:
            self.index = len(self.file_name_list)-1
            print('已经是最后一张啦')
        self.cleargrids()
        global object_num
        object_num = 0
        self.loadfile()

    # def doneall_savejson(self):
    #     # 所有的完成，生成记录的json文件
    #     b =json.dumps(self.grids_record)
    #     f2 = open('./choose_record/record10.json', 'w')
    #     f2.write(b)
    #     f2.close()

    def donebuttonclicked(self):
        # 生成正guide
        guide_save = np.zeros((self.w,self.h))
        guide = np.zeros((self.w,self.h),dtype= np.uint8)
        # 记录grids的bbox
        # self.grids = []  # 清一下上次的值
        # count = 0
        # grid_list = []
        for i in range(self.layout.count()):
            grid = self.layout.itemAt(i).widget()
            selected = grid.selected
            bbox = grid.bbox
            # print(bbox)

            if selected:
                guide_save[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 255
                guide[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1
                # count += 1
                # grid_list.append(bbox)
            #     self.grids.append(bbox)

        cv2.imwrite('./guide.png',guide_save)
        # 生成负guide
        global grid_guide,crop_bbox
        # bg_points,crop_bbox= self.neg_points(mask=guide)
        bg_points = self.neg_points(mask=guide)
        grid_guide = helpers.make_gt_grids(guide_save, guide, bg_points, sigma= 10, one_mask_per_point=False)
        crop_bbox = helpers.get_bbox(grid_guide[:,:,0],pad=30,zero_pad=True)

        grid_guide_crop = helpers.crop_from_bbox(grid_guide,crop_bbox,True)
        img_crop = helpers.crop_from_bbox(self.img,crop_bbox,True)
        cv2.imwrite('./img_crop_.png',img_crop)

        # grid_guide_crop = grid_guide[crop_bbox[1]:crop_bbox[3],crop_bbox[0]:crop_bbox[2],:]
        # img_crop = self.img[crop_bbox[1]:crop_bbox[3],crop_bbox[0]:crop_bbox[2],:]


        # img_crop = cv2.resize(img_crop,(450,450))
        # grid_guide_crop = cv2.resize(grid_guide_crop, (450,450))

        img_crop = cv2.resize(img_crop,(512,512))
        grid_guide_crop = cv2.resize(grid_guide_crop,(512,512))

        tmp = grid_guide_crop
        grid_guide_crop = 255 * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)

        input = np.concatenate((img_crop,grid_guide_crop),axis=2)
        input = input.transpose((2,0,1)).astype(np.float32)

        input = torch.from_numpy(input)
        input = input.unsqueeze(0)
        input = input.to(device)
        grid_guide_crop = grid_guide_crop.transpose((2,0,1)).astype(np.float32)
        grid_guide_crop = torch.from_numpy(grid_guide_crop)
        grid_guide_crop = grid_guide_crop.unsqueeze(0)
        grid_guide_crop = grid_guide_crop.to(device)
        # 传入网络计算
        global out_list
        # out_list,fine_out = net.forward(input,grid_guide_crop,None)
        fine_out = net.forward(input)
        outputs = fine_out.to(torch.device('cpu'))
        pred = np.transpose(outputs.data.numpy()[0, :, :, :], (1, 2, 0))
        pred = 1 / (1 + np.exp(-pred))
        pred = np.squeeze(pred)
        # pred = np.where(pred > 0.5, 1, 0)
        # pred = pred.astype(np.uint8)

        pred = cv2.resize(pred,(crop_bbox[2]-crop_bbox[0],crop_bbox[3]-crop_bbox[1]))
        # pred = cv2.resize(pred,(grid_guide.shape[1],grid_guide.shape[0]))

        pred_result = helpers.crop2fullmask(pred, crop_bbox, guide_save, zero_pad=True, relax=0, mask_relax=False)
        global filename
        imsave('./results/hunman_scribble_grabcut/'+filename.split('.')[0]+'.png', pred_result)

        result_ = pred_result * 128
        # result_ = pred * 128
        mask = np.zeros((self.w,self.h,3))
        global object_num
        if object_num == 0:
            mask[:, :, 2] = result_
        elif object_num == 1:
            mask[:, :, 1] = result_
        elif object_num == 2:
            mask[:, :, 0] = result_
        elif object_num == 3:
            mask[:, :, 0] = result_
            mask[:, :, 1] = result_
        elif object_num == 4:
            mask[:, :, 0] = result_
            mask[:, :, 2] = result_
        elif object_num == 5:
            mask[:, :, 1] = result_
            mask[:, :, 2] = result_

        mask = mask.astype(np.uint8)
        cv2.imwrite('./mask.png', mask)

        print('pred success!')
        result = cv2.addWeighted(self.img,1, mask, 0.7, 1)
        cv2.imwrite('./result.png', result)

        self.cleargrids()

        self.label = ShowGrids(self.frame)
        self.layout.addWidget(self.label, 0, 0)
        self.label.setPixmap(QPixmap('./result.png'))

        #记录所选的grids
        # record = {
        #     'image':self.file_name_list[self.index],
        #     'grid_record':self.grids
        # }
        # self.grids_record['record'].append(record)
        global correct_flag
        correct_flag = True





    def cleargrids(self):
        for i in range(self.layout.count()):
            # print(self.layout.count())
            self.layout.itemAt(i).widget().deleteLater()


    def nextobjectbuttonclicked(self):
        # for i in range(self.layout.count()):
        #     grid = self.layout.itemAt(i).widget()
        #     grid.selected = False
        #     grid.setStyleSheet("QLabel{border:1px solid rgb(0, 0, 0);}")
        global object_num
        # self.donebuttonclicked()
        object_num += 1
        self.cleargrids()
        self.loadfile()


    def sliderValueChanged(self,new_value):
        global r_count
        global c_count
        self.patch_size = self.horizontalSlider.value()
        self.RemoveDir('./patch_image/')
        self.cleargrids()
        r_count = 0
        c_count = 0
        self.loadfile()

    # def keyPressEvent(self, event):
    #     print(event.key())
    #     if(event.key() == QtCore.Qt.Key_S):
    #         click = True

    def loadfile(self):
        print("load--file")
        print(self.patch_size)
        global r_count
        global c_count
        global object_num
        # if self.file is None:
        #     self.file, _ = QFileDialog.getOpenFileNames(self.centralwidget, '选择图片', 'E:\\', 'image files(*.jpg *.gif *.png)')
        # img = cv2.imread(self.file[0])
        global img
        if object_num > 0:
            img = cv2.imread('./result.png')
        else:
            # img = cv2.imread('./click_image/'+self.file_name_list[self.index])
            img = cv2.imread('/home/datasets/GrabCut/data_GT/'+self.file_name_list[self.index])
        global filename
        filename = self.file_name_list[self.index]
        self.img = img
        self.w = img.shape[0]
        self.h = img.shape[1]

        xmin = 0
        ymin = 0
        x_ = img.shape[1] % self.patch_size
        y_ = img.shape[0] % self.patch_size
        c_count = int(img.shape[1] / self.patch_size)
        r_count = int(img.shape[0] / self.patch_size)
        if x_ > 0:
            c_count += 1
        if y_ > 0:
            r_count += 1

        self.frame.resize(img.shape[1]+10,img.shape[0]+10)
        for i in range(int(img.shape[0] / self.patch_size)):
            for j in range(int(img.shape[1] / self.patch_size)):
                x, y, w, h = xmin, ymin, self.patch_size, self.patch_size  # 矩形裁剪区域 (ymin:ymin+h, xmin:xmin+w) 的位置参数
                imgCrop = img[y:y + h, x:x + w].copy()  # 切片获得裁剪后保留的图像区域
                cv2.imwrite('./patch_image/' + str(i) +'_'+ str(j) + '.jpg', imgCrop)
                xmin += self.patch_size
            if x_ > 0:
                x, y, w, h = xmin, ymin, x_, self.patch_size
                imgCrop = img[y:y + h, x:x + w].copy()  # 切片获得裁剪后保留的图像区域
                cv2.imwrite('./patch_image/' + str(i)  +'_'+  str(j + 1) + '.jpg', imgCrop)

            ymin += self.patch_size
            xmin = 0
        if y_ > 0:
            i = int(img.shape[0] / self.patch_size)
            xmin = 0
            ymin = i * self.patch_size

            for j in range(int(img.shape[1] / self.patch_size)):
                x, y, w, h = xmin, ymin, self.patch_size, y_
                imgCrop = img[y:y + h, x:x + w].copy()
                cv2.imwrite('./patch_image/' + str(i)  +'_'+  str(j) + '.jpg', imgCrop)
                xmin += self.patch_size
            if x_ > 0:
                x, y, w, h = xmin, ymin, x_, y_
                imgCrop = img[y:y + h, x:x + w].copy()
                cv2.imwrite('./patch_image/' + str(i)  +'_'+  str(j + 1) + '.jpg', imgCrop)

        for i in range(int(img.shape[0] / self.patch_size)):
            for j in range(int(img.shape[1] / self.patch_size)):
                # self.label = QLabel(self.frame)
                self.label = Grids(self.frame)
                # self.label.setFrameShape(QtWidgets.QFrame.Box)
                # self.label.setLineWidth(1)
                # self.label.setStyleSheet("QLabel{border:0.3px solid rgb(211, 211, 211);}")
                self.layout.addWidget(self.label,i,j)
                self.label.setPixmap(QPixmap('./patch_image/'+ str(i)  +'_'+  str(j) + '.jpg'))
                self.label.setScaledContents(True)
                # self.label.resize(74,74)
                # self.label.setMargin(1)
                self.label.bbox = [j*self.patch_size,i*self.patch_size,j*self.patch_size+self.patch_size,i*self.patch_size+self.patch_size]
                # self.label.linkActivated.connect(self.linkHovered)

            if x_ > 0:
                self.label = Grids(self.frame)
                # self.label.setStyleSheet("QLabel{border:0.3px solid rgb(211, 211, 211);}")
                self.layout.addWidget(self.label, i, j+1)
                self.label.setPixmap(QPixmap('./patch_image/' + str(i)  +'_'+  str(j+1) + '.jpg'))
                self.label.setScaledContents(True)
                self.label.bbox = [(j+1) * self.patch_size, i * self.patch_size, (j+1) * self.patch_size + x_,
                                   i * self.patch_size + self.patch_size]
                # self.label.linkHovered.connect(self.linkHovered)
                # self.label.linkActivated.connect(self.linkHovered)
        if y_ > 0:
            i = int(img.shape[0] / self.patch_size)
            for j in range(int(img.shape[1] / self.patch_size)):
                self.label = Grids(self.frame)
                # self.label.setStyleSheet("QLabel{border:1px solid rgb(211, 211, 211);}")
                self.layout.addWidget(self.label, i, j)
                self.label.setPixmap(QPixmap('./patch_image/' + str(i)  +'_'+  str(j) + '.jpg'))
                self.label.setScaledContents(True)
                self.label.bbox = [j * self.patch_size, i * self.patch_size, j * self.patch_size + self.patch_size,
                                   i * self.patch_size + y_]
            if x_ > 0:
                self.label = Grids(self.frame)
                # self.label.setStyleSheet("QLabel{border:1px solid rgb(211, 211, 211);}")
                self.layout.addWidget(self.label, i, j+1)
                self.label.setPixmap(QPixmap('./patch_image/' + str(i)  +'_'+  str(j+1) + '.jpg'))
                self.label.setScaledContents(True)
                self.label.bbox = [(j+1) * self.patch_size, i * self.patch_size, (1+j) * self.patch_size + x_,
                                   i * self.patch_size + y_]
        self.layout.setSpacing(0)
        # fname, _ = QFileDialog.getOpenFileNames(self.centralwidget, '选择图片', 'c:\\', 'image files(*.jpg *.gif *.png)')
        # self.label.setPixmap(QPixmap('./2007_001185.jpg'))

    def RemoveDir(self,filepath):
        '''
        如果文件夹不存在就创建，如果文件存在就清空！
        '''
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        else:
            shutil.rmtree(filepath)
            os.mkdir(filepath)

    def neg_points(self,mask,pad_pixel=10):
        def find_point(id_x, id_y, ids):
            sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
            return [id_x[sel_id], id_y[sel_id]]

        inds_y, inds_x = np.where(mask > 0.5)
        h = self.w
        w = self.h

        left = find_point(inds_x, inds_y, np.where(inds_x <= np.min(inds_x)))
        right = find_point(inds_x, inds_y, np.where(inds_x >= np.max(inds_x)))
        top = find_point(inds_x, inds_y, np.where(inds_y <= np.min(inds_y)))
        bottom = find_point(inds_x, inds_y, np.where(inds_y >= np.max(inds_y)))

        x_min = left[0]
        x_max = right[0]
        y_min = top[1]
        y_max = bottom[1]

        left_top = [max(x_min - pad_pixel, 0), max(y_min - pad_pixel, 0)]
        left_bottom = [max(x_min - pad_pixel, 0), min(y_max + pad_pixel, h)]
        right_top = [min(x_max + pad_pixel, w), max(y_min - pad_pixel, 0)]
        righr_bottom = [min(x_max + pad_pixel, w), min(y_max + pad_pixel, h)]
        a = [left_top, left_bottom, right_top, righr_bottom]

        x_min = max(left_top[0]-30,0)
        x_max = min(righr_bottom[0]+30,w)
        y_min = max(left_top[1]-30,0)
        y_max = min(righr_bottom[1]+30,h)
        b = [x_min, y_min, x_max, y_max]

        # return np.array(a),b
        return np.array(a)


class ShowGrids(QLabel):
    def __init__(self,parent):
        super(ShowGrids, self).__init__(parent)
        global correct_flag
        global grid_guide
        global img
        global out_list
        global crop_bbox


    def mousePressEvent(self, event):
        # global low_level_feat_4, low_level_feat_3, low_level_feat_2, low_level_feat_1
        if event.buttons() == QtCore.Qt.LeftButton:
            if correct_flag:
                # 校正正点击
                print('校正正点击')
                # point_pos = make_gaussian((self.parent().height(), self.parent().width()), center=[event.pos().x(), event.pos().y()], sigma=10)
                # guide = generate_distance_map_(event.pos().y(),event.pos().x(),grid_guide[:,:,0],grid_guide[:,:,1],grid_guide[:,:,0],True,crop_bbox)
                print(event.pos())
                pass

        if event.buttons() == QtCore.Qt.RightButton:
            if correct_flag:
                # 校正负点击
                print('校正负点击')
                # guide = generate_distance_map_(event.pos().y(),event.pos().x(),grid_guide[:,:,0],grid_guide[:,:,1],grid_guide[:,:,0],False,crop_bbox)


        # guide = guide.to(device)
        # fine_out = net.forward(None,guide,out_list)
        # outputs = fine_out.to(torch.device('cpu'))
        # pred = np.transpose(outputs.data.numpy()[0, :, :, :], (1, 2, 0))
        # pred = 1 / (1 + np.exp(-pred))
        #
        # pred = np.squeeze(pred)
        #
        #
        # # pred = cv2.resize(pred,(img.shape[1],img.shape[0]))
        #
        # pred_result = helpers.crop2fullmask(pred, crop_bbox, im_size=[img.shape[0],img.shape[1]], zero_pad=True, relax=0, mask_relax=False)
        # pred_result = np.where(pred_result > 0.9, 1, 0)
        # pred_result = pred_result * 255
        # mask = np.zeros((img.shape[0], img.shape[1], 3))
        #
        # global object_num
        # if object_num == 0:
        #     mask[:, :, 2] = pred_result
        # elif object_num == 1:
        #     mask[:, :, 1] = pred_result
        # elif object_num == 2:
        #     mask[:, :, 0] = pred_result
        # elif object_num == 3:
        #     mask[:, :, 0] = pred_result
        #     mask[:, :, 1] = pred_result
        # elif object_num == 4:
        #     mask[:, :, 0] = pred_result
        #     mask[:, :, 2] = pred_result
        # elif object_num == 5:
        #     mask[:, :, 1] = pred_result
        #     mask[:, :, 2] = pred_result
        #
        # mask = mask.astype(np.uint8)
        # cv2.imwrite('./mask_.png', mask)
        #
        # print('pred success!')
        # result = cv2.addWeighted(img, 1, mask, 0.5, 1)
        # cv2.imwrite('./result.png', result)
        #
        # self.setPixmap(QPixmap('./result.png'))
        # print('test')


class Grids(QLabel):
    def __init__(self,parent):
        super(Grids, self).__init__(parent)
        global click_select
        global click_cancel
        self.selected = False
        self.bbox = []

    def mousePressEvent(self, event):
        global click_select
        global click_cancel

        if event.buttons() == QtCore.Qt.LeftButton:
            print('左键')

            self.setStyleSheet("QLabel{border:2px solid rgb(0, 255, 0);}")
            self.selected = True
            if click_select:
                click_select = False
            else:
                click_select = True
                click_cancel = False

        if event.buttons() == QtCore.Qt.RightButton:
            print('右键')

            self.setStyleSheet("QLabel{border:none;}")
            self.selected = False
            if click_cancel:
                click_cancel = False
            else:
                click_cancel = True
                click_select = False

    def mouseDoubleClickEvent(self, event):
        print('双击填充')
        global click_select
        global click_cancel
        global r_count
        global c_count
        click_cancel = False
        click_select = False
        layout = self.parent().layout()
        # r = layout.rowCount()
        # c = layout.columnCount()
        selected_rc = np.zeros([r_count, c_count],np.uint8)
        print('r c is:' + str(r_count) + str(c_count))
        for i in range(r_count):
            for j in range(c_count):

                grid = layout.itemAtPosition(i, j).widget()
                if grid.selected:
                    selected_rc[i,j] = 1
        for i in range(r_count):
            for j in range(c_count):
                if selected_rc[i,j] == 0:
                    row_index = np.where(selected_rc[:, j] > 0)
                    col_index = np.where(selected_rc[i, :] > 0)
                    if len(row_index[0]) < 2 or len(col_index[0]) < 2:
                        continue
                    min_r = min(row_index[0])
                    max_r = max(row_index[0])
                    min_c = min(col_index[0])
                    max_c = max(col_index[0])
                    if min_r < i and max_r > i and min_c < j and max_c > j:
                        grid = layout.itemAtPosition(i, j).widget()
                        grid.selected = True
                        grid.setStyleSheet("QLabel{border:2px solid rgb(0, 255, 0);}")
                        selected_rc[i, j] = 1

        # inds = np.where(selected_rc > 0)


        # mask = np.zeros((layout.size()[0],layout.size()[1]), np.uint8)
        # mask = np.zeros((385,510), np.uint8)
        #
        # for i in range(layout.count()):
        #     grid = layout.itemAt(i).widget()
        #     selected = grid.selected
        #     bbox = grid.bbox
        #     if selected:
        #         mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 255
        #
        # _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # for cont in contours:
        #     # 取轮廓长度的1%为epsilon
        #     epsilon = 0.005 * cv2.arcLength(cont, True)
        #     # 预测多边形
        #     box = cv2.approxPolyDP(cont, 3, True)
        #     box_ = box.squeeze()
        #     new = []
        #     for i in range(box_.shape[0]):
        #         tmp = box_[i]
        #         t = [tmp[0], tmp[1]]
        #         new.append(t)
        #         poly = np.array(new)
        #         cv2.fillPoly(mask, [poly, ], (255, 255, 255))
        #         cv2.imwrite('./polygon.png', mask)
        #
        #     # img = cv2.polylines(mask, [box], True, (0, 0, 255), 10)
        print('test---')

    def enterEvent(self,event):
        if click_select:
            # self.setStyleSheet('border-width: 5px')
            self.setStyleSheet("QLabel{border:2px solid rgb(0, 255, 0);}")
            self.selected = True
            # self.setText('test')
        elif click_cancel:
            # self.setStyleSheet("QLabel{border:1px solid rgb(211,211,211);}")
            self.setStyleSheet("QLabel{border:none;}")
            self.selected = False
        else:
            print('还未开始选择')
            if not self.selected:
                self.setStyleSheet("QLabel{border:2px solid rgb(211,211,211);}")

    def leaveEvent(self, QEvent):
        if not click_select and not click_cancel:
            if not self.selected:
                self.setStyleSheet("QLabel{border:none;}")


    # def mouseMoveEvent(self, event):
    #     print(event.pos())

    def mouseReleaseEvent(self, event):
        # self.click = False
        pass
    def get_selected(self):
        return self.selected

    def get_bbox(self):
        return self.bbox

