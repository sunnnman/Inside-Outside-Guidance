
class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return '/home/zhupengqi/dataset/'  # folder that contains VOCdevkit/.

        if database == 'coco_cityscapes':
            return '/home/zhupengqi/dataset/cityscapes/datasets/cityscapes/'  # folder that contains VOCdevkit/.test

        elif database == 'sbd':
            return '/home/zhupengqi/dataset/benchmark/'  # folder with img/, inst/, cls/, etc.

        elif database == 'coco':
            return '/home/datasets/coco/'

        elif database == 'sstem':
            return '/home/datasets/ssTEM/'

        elif database == 'ctw1500':
            return '/home/zhupengqi/dataset/ctw1500/test/'

        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def models_dir():
        # return './model/IOG_PASCAL.pth'
        return './model/resnet101-5d3b4d8f.pth'
        #'resnet101-5d3b4d8f.pth' #resnet50-19c8e357.pth'
