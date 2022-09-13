import json
import pprint

import cv2 as cv
import numpy as np
import json
import os


def convertPolygonToMask(jsonfilePath):
    with open(jsonfilePath, "r", encoding='utf-8') as jsonf:
        jsonData = json.load(jsonf)
        maskSaveFolder = '/home/zhupengqi/dataset/cityscapes/datasets/cityscapes/train_mask'
        #图片中目标的数量 num=len(jsonData["shapes"])
        num = 0
        for annotation in jsonData["annotations"]:
            # img_h = jsonData["annotations"]['height']
            # img_w = jsonData["annotations"]['width']
            img_h = annotation['height']
            img_w = annotation['width']
            mask = np.zeros((img_h, img_w), np.uint8)
            # label = obj["label"]
            polygonPoints = annotation["segmentation"]
            polygonPoints = np.array(polygonPoints,np.int32)
            t = int(len(polygonPoints[0])/2)
            polygonPoints = polygonPoints.reshape((t, 2))

            # print("+" * 50, "\n", polygonPoints)
            # print(label)
            num+=1
            cv.drawContours(mask,[polygonPoints],-1,(1),-1)
            cv.imwrite(maskSaveFolder + annotation['image_id'] + '_' + annotation['id'] + "mask.png", mask)


if __name__ == "__main__":
    #main()
    jsonfilePath = r"/home/zhupengqi/dataset/cityscapes/datasets/cityscapes/train_.json"
    maskSaveFolder = r"./train_mask"
    convertPolygonToMask(jsonfilePath)
    # # 为了可视化把mask做一下阈值分割
    # _, th = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
    # cv.imshow("mask", th)
    # src = cv.imread(r"K:\deepImage\del\1.jpg")
    # cv.imwrite(maskSaveFolder + "\mask.png", mask)
    # cv.imshow("src", src)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

