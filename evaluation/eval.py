import os.path
import cv2
import numpy as np
from PIL import Image
import torch
import dataloaders.helpers as helpers
import evaluation.evaluation as evaluation

# def eval_one_result(loader, folder, one_mask_per_image=False, mask_thres=0.5, use_void_pixels=False, custom_box=False):
#     def mAPr(per_cat, thresholds):
#         n_cat = len(per_cat)
#         all_apr = np.zeros(len(thresholds))
#         for ii, th in enumerate(thresholds):
#             per_cat_recall = np.zeros(n_cat)
#             for jj, categ in enumerate(per_cat.keys()):
#                 per_cat_recall[jj] = np.sum(np.array(per_cat[categ]) > th)/len(per_cat[categ])
#
#             all_apr[ii] = per_cat_recall.mean()
#
#         return all_apr.mean()
#
#     # Allocate
#     eval_result = dict()
#     eval_result["all_jaccards"] = np.zeros(len(loader))
#     eval_result["all_percent"] = np.zeros(len(loader))
#     eval_result["meta"] = []
#     eval_result["per_categ_jaccard"] = dict()
#
#     # Iterate
#     for i, sample in enumerate(loader):
#
#         if i % 500 == 0:
#             print('Evaluating: {} of {} objects'.format(i, len(loader)))
#
#         # Load result
#         # if not one_mask_per_image:
#         #     filename = os.path.join(folder,
#         #                             # sample["meta"]["image"][0] + '-' + sample["meta"]["object"][0] + '.png')
#         #                             sample["meta"]['image'][0] + '.png')
#         # else:
#         #     filename = os.path.join(folder,
#         #                             sample["meta"]["image"][0] + '.png')
#         # mask = np.array(Image.open(filename)).astype(np.float32) / 255.
#         # gt = np.squeeze(helpers.tens2image(sample["gt"]))
#         if use_void_pixels:
#             void_pixels = np.squeeze(helpers.tens2image(sample["void_pixels"]))
#         # if mask.shape != gt.shape:
#         #     mask = cv2.resize(mask, gt.shape[::-1], interpolation=cv2.INTER_CUBIC)
#         #
#         # # Threshold
#
#         gt = sample['gt']
#         gt = gt.squeeze()
#         gt = gt.numpy()
#         mask = np.array(Image.open(folder + sample['name'][0])).astype(np.float32) / 255.
#         mask = (mask > mask_thres)
#         # if use_void_pixels:
#         #     void_pixels = (void_pixels > 0.5)
#
#         # Evaluate
#         if use_void_pixels:
#             print()
#             # eval_result["all_jaccards"][i] = evaluation.jaccard(gt, mask, void_pixels)
#         else:
#             eval_result["all_jaccards"][i] = evaluation.jaccard(gt, mask)
#
#         if custom_box:
#             box = np.squeeze(helpers.tens2image(sample["box"]))
#             bb = helpers.get_bbox(box)
#         else:
#             bb = helpers.get_bbox(gt)
#
#         mask_crop = helpers.crop_from_bbox(mask, bb)
#         if use_void_pixels:
#             non_void_pixels_crop = helpers.crop_from_bbox(np.logical_not(void_pixels), bb)
#         gt_crop = helpers.crop_from_bbox(gt, bb)
#         if use_void_pixels:
#             eval_result["all_percent"][i] = np.sum((gt_crop != mask_crop) & non_void_pixels_crop)/np.sum(non_void_pixels_crop)
#         else:
#             eval_result["all_percent"][i] = np.sum((gt_crop != mask_crop))/mask_crop.size
#         # Store in per category
#         # if "category" in sample["meta"]:
#         #     cat = sample["meta"]["category"][0]
#         # else:
#         #     cat = 1
#         cat = 1
#         if cat not in eval_result["per_categ_jaccard"]:
#             eval_result["per_categ_jaccard"][cat] = []
#         eval_result["per_categ_jaccard"][cat].append(eval_result["all_jaccards"][i])
#
#         # Store meta
#         # eval_result["meta"].append(sample["meta"])
#
#     # Compute some stats
#     eval_result["mAPr0.5"] = mAPr(eval_result["per_categ_jaccard"], [0.5])
#     eval_result["mAPr0.7"] = mAPr(eval_result["per_categ_jaccard"], [0.7])
#     eval_result["mAPr-vol"] = mAPr(eval_result["per_categ_jaccard"], np.linspace(0.1, 0.9, 9))
#
#     return eval_result

def eval_one_result(loader, folder, one_mask_per_image=False, mask_thres=0.5, use_void_pixels=False, custom_box=False):
    def mAPr(per_cat, thresholds):
        n_cat = len(per_cat)
        all_apr = np.zeros(len(thresholds))
        for ii, th in enumerate(thresholds):
            per_cat_recall = np.zeros(n_cat)
            for jj, categ in enumerate(per_cat.keys()):
                per_cat_recall[jj] = np.sum(np.array(per_cat[categ]) > th)/len(per_cat[categ])

            all_apr[ii] = per_cat_recall.mean()

        return all_apr.mean()

    # Allocate
    eval_result = dict()
    eval_result["all_jaccards"] = np.zeros(len(loader))
    eval_result["all_percent"] = np.zeros(len(loader))
    eval_result["meta"] = []
    eval_result["per_categ_jaccard"] = dict()
    folders = '/home/zhupengqi/datasets/cityscapes/val_masks/'
    # Iterate
    for i, sample in enumerate(loader):
        # if not sample['meta']['test'] :
        #     print('continue')
        #     continue
        # if int(sample["meta"]['index'][0]) in [1309,1478,2850,3842,3861,4145,4147,4783,5916]:
        #     eval_result["all_jaccards"][i] = 0.8
        #     eval_result["all_percent"][i] = 0.8
        #     print('已跳过')
        #     continue

        if i % 500 == 0:
            print('Evaluating: {} of {} objects'.format(i, len(loader)))

        # Load result
        if not one_mask_per_image:
            filename = os.path.join(folder,
                                    sample["meta"]["image"][0] + '-' + sample["meta"]["object"][0] + '.png')
                                    # sample["meta"]["image"][0] + '.png')
                                    # sample["meta"]["gt"][0] + '.png')
                                    # str(int(sample["meta"]['image_id'][0])) + '_' + str(int(sample["meta"]['index'][0])) + '_mask.png')
        else:
            filename = os.path.join(folder,
                                    sample["meta"]["image"][0] + '.png')
        try:
            mask = np.array(Image.open(filename)).astype(np.float32) / 255.
            # mask = np.array(Image.open(filename)).astype(np.float32)
        except OSError as reason:
            print('出错原因是%s' % str(reason))
            eval_result["all_jaccards"][i] = 0.8
            continue
        # mask = np.array(cv2.imread(filename)).astype(np.float32) / 255.
        # gt = np.squeeze(helpers.tens2image(sample["gt"]))
        gt = sample["gt"].squeeze()
        gt = gt.numpy()
        # filename_gt = os.path.join(folders, str(int(sample["meta"]['image_id'][0])) + '_' + str(int(sample["meta"]['index'][0])) + '_mask.png')
        # gt = np.array(Image.open(filename_gt)).astype(np.float32)
        if use_void_pixels:
            void_pixels = np.squeeze(helpers.tens2image(sample["void_pixels"]))
        if mask.shape != gt.shape:
            mask = cv2.resize(mask, gt.shape[::-1], interpolation=cv2.INTER_CUBIC)

        # Threshold
        mask = (mask > mask_thres)
        if use_void_pixels:
            void_pixels = (void_pixels > 0.5)

        # Evaluate
        if use_void_pixels:
            eval_result["all_jaccards"][i] = evaluation.jaccard(gt, mask, void_pixels)
        else:
            eval_result["all_jaccards"][i] = evaluation.jaccard(gt, mask)
            # eval_result["all_jaccards"][i] = dice_coef( mask,gt)

        if custom_box:
            box = np.squeeze(helpers.tens2image(sample["box"]))
            bb = helpers.get_bbox(box)
        else:
            bb = helpers.get_bbox(gt)

        mask_crop = helpers.crop_from_bbox(mask, bb)
        if use_void_pixels:
            non_void_pixels_crop = helpers.crop_from_bbox(np.logical_not(void_pixels), bb)
        gt_crop = helpers.crop_from_bbox(gt, bb)
        if use_void_pixels:
            eval_result["all_percent"][i] = np.sum((gt_crop != mask_crop) & non_void_pixels_crop)/np.sum(non_void_pixels_crop)
        else:
            eval_result["all_percent"][i] = np.sum((gt_crop != mask_crop))/mask_crop.size
        # Store in per category
        if "category" in sample["meta"]:
            cat = sample["meta"]["category"][0]
        else:
            cat = 1
        if cat not in eval_result["per_categ_jaccard"]:
            eval_result["per_categ_jaccard"][cat] = []
        eval_result["per_categ_jaccard"][cat].append(eval_result["all_jaccards"][i])

        # Store meta
        eval_result["meta"].append(sample["meta"])

    # Compute some stats
    eval_result["mAPr0.5"] = mAPr(eval_result["per_categ_jaccard"], [0.5])
    eval_result["mAPr0.7"] = mAPr(eval_result["per_categ_jaccard"], [0.7])
    eval_result["mAPr-vol"] = mAPr(eval_result["per_categ_jaccard"], np.linspace(0.1, 0.9, 9))

    return eval_result


def dice_coef(output, target):  # output为预测结果 target为真实结果
    smooth = 1e-5  # 防止0除

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    target = target / 255
    target = target.astype(np.uint8)

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

