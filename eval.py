import os.path

from torch.utils.data import DataLoader
from evaluation.eval import eval_one_result
from dataloaders.coco_city import *
import dataloaders.pascal as pascal
import dataloaders.sbd as sbd
from dataloaders.grabcut import *
from dataloaders.sstem import *
from dataloaders.coco import *
from dataloaders.mscmr import *
exp_root_dir = './results'

method_names = []
# method_names.append('hunman_scribble_grabcut')
method_names.append('Grid_sbd_5010_guassian7_sample_ctw1500')

if __name__ == '__main__':

    # Dataloader
    # dataset = pascal.VOCSegmentation(transform=None, retname=True)
    # dataset = sbd.SBDSegmentation(transform=None, retname=True)
    # dataset = CocoSegmentation_City(split='val', transform=None)
    # dataset = CocoSegmentation(split='val', year='2017', transform=None)
    # dataset = GrabCutDataset(dataset_path='/home/zhupengqi/dataset/GrabCut/', transform=None)
    # dataset = GrabCutDataset(dataset_path='/home/zhupengqi/dataset/Berkeley/', transform=None)
    # dataset = sstemDataset_(dataset_path='/home/zhupengqi/dataset/ssTEM/', transform=None)
    # dataset = mscmrDataset(dataset_path='/home/zhupengqi/dataset/MSCMR/',transform=None)
    dataset = CocoSegmentation(split='val', transform=None)
    dataloader = DataLoader(dataset, batch_size=1,shuffle=False,num_workers=0)

    # Iterate through all the different methods
    for method in method_names:
        # for ii in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        for ii in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            # results_folder = os.path.join(exp_root_dir, method, 'Results')
            # results_folder = os.path.join("/home/zhupengqi/Inside-Outside-Guidance-master/results/hunman_scribble_grabcut/")
            results_folder = os.path.join("/home/zhupengqi/Inside-Outside-Guidance-master/results/Grid_sbd_5010_guassian7_sample_ctw1500/")

            filename = os.path.join(exp_root_dir, 'eval_results', method.replace('/', '-') + '.txt')
            if not os.path.exists(os.path.join(exp_root_dir, 'eval_results')):
                os.makedirs(os.path.join(exp_root_dir, 'eval_results'))
    
            jaccards = eval_one_result(dataloader, results_folder, mask_thres=ii)
            val = jaccards["all_jaccards"].mean()
    
            # Show mean and store result
            print(ii)
            print("Result for {:<80}: {}".format(method, str.format("{0:.4f}", 100*val)))
            with open(filename, 'a+') as f:
                f.write(str(val)+'\n')

