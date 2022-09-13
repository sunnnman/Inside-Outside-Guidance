from albumentations.pytorch import ToTensorV2  # ToTensor已弃用
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
import glob
import pandas as pd
import numpy as np
import cv2

# Path to all data
DATA_PATH = "./kaggle_3m/"

# File path line length images for later sorting,用于排序的字符串范围
BASE_LEN = 56 # len(./kaggle_3m\TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_ <-!!!43.tif)
END_IMG_LEN = 4 # (.tif)
END_MASK_LEN = 9 # (_mask.tif)

# Raw data
data_map = []
# 对原始图像文件路径查看处理,统一放入data_map
for sub_dir_path in glob.glob(DATA_PATH + "*"):
    if os.path.isdir(sub_dir_path):
        dirname = sub_dir_path.split("/")[-1]
        for filename in os.listdir(sub_dir_path):
            image_path = sub_dir_path + "/" + filename
            data_map.extend([dirname, image_path])
    else:
        print("This is not a dir:", sub_dir_path)

df = pd.DataFrame({"dirname": data_map[::2],
                   "path": data_map[1::2]})


# Masks/Not masks
df_imgs = df[~df['path'].str.contains("mask")]
df_masks = df[df['path'].str.contains("mask")]
#print(len(df_imgs),len(df_masks))

# Data sorting
imgs = sorted(df_imgs["path"].values, key=lambda x : x[BASE_LEN:-END_IMG_LEN])
masks = sorted(df_masks["path"].values, key=lambda x : x[BASE_LEN:-END_MASK_LEN])

# Final dataframe 最终的数据文件结构
df = pd.DataFrame({"patient": df_imgs.dirname.values,
                       "image_path": imgs,
                   "mask_path": masks})



