import os
import cv2
import logging
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from model import BoxPolyp

class TrainData(Dataset):
    def __init__(self, datapath):
        self.datapath  = datapath
        self.samples   = os.listdir(self.datapath+'/image')
        self.colors    = np.load('../dataset/color.npy')
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(352, 352),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        name        = self.samples[idx]
        image       = cv2.imread(self.datapath+'/image/'+name)
        image       = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        mean1, std1 = image.mean(axis=(0,1), keepdims=True), image.std(axis=(0,1), keepdims=True)
        mean2, std2 = self.colors[np.random.randint(len(self.colors))]
        image       = np.uint8((image-mean1)/std1*std2+mean2)
        image       = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        mask        = np.float32(cv2.imread(self.datapath+'/mask/'+name, cv2.IMREAD_GRAYSCALE)>128)
        pair        = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask']

    def __len__(self):
        return len(self.samples)


class TestData(Dataset):
    def __init__(self, datapath):
        self.datapath         = datapath
        self.samples          = sorted(os.listdir(datapath+'/image'))
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(352, 352),
            ToTensorV2()
        ])

    def __getitem__(self, index):
        name   = self.samples[index]
        image  = cv2.imread(self.datapath+'/image/'+name)
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask   = np.float32(cv2.imread(self.datapath+'/mask/' +name, cv2.IMREAD_GRAYSCALE)>128)
        pair   = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask'], image, mask, name

    def __len__(self):
        return len(self.samples)


class LDPolypVideo(Dataset):
    def __init__(self, datapath):
        self.datapath  = datapath
        self.transform = A.Compose([
            A.Resize(352, 352),
            A.Normalize(),
            ToTensorV2()
        ])

        self.samples   = []
        for folder in os.listdir(datapath+'/bbox'):
            for name in os.listdir(datapath+'/bbox/'+folder):
                with open(datapath+'/bbox/'+folder+'/'+name, 'r') as f:
                    lines = f.readlines()
                    if int(lines[0])!=0:
                        self.samples.append(folder+'/'+name[:-4])

    def __getitem__(self, idx):
        ## read image
        name   = self.samples[idx]
        image  = cv2.imread(self.datapath+'/image/'+name+'.jpg')
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H,W,C  = image.shape
        mask   = np.zeros((H, W), dtype=np.float32)
        ## generate binary box mask
        with open(self.datapath+'/bbox/'+name+'.txt', 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                xmin, ymin, xmax, ymax     = line.strip().split(' ')
                xmin, ymin, xmax, ymax     = int(xmin), int(ymin), int(xmax), int(ymax)
                mask[ymin:ymax, xmin:xmax] = 1
        ## transform
        pair   = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask'], image, mask, name

    def __len__(self):
        return len(self.samples)



def getModel(cfg, backbone):
    '''
        input : backbone name
        output: model and logger
    '''    
    model = BoxPolyp(cfg).cuda()
    if cfg.mode=='train':
        model.train()
        model.backbone.eval()
        handle          = logging.FileHandler(backbone+'.log', mode='a', encoding='utf-8')
        formatter       = logging.Formatter(fmt='[%(asctime)s | %(message)s]', datefmt='%I:%M:%S')
        handle.setFormatter(formatter)
        logger          = logging.Logger(backbone+'_logger', level=logging.INFO)
        logger.addHandler(handle)
        return {'name':backbone, 'model':model, 'logger':logger, 'score':0}
    else:
        model.eval()
        return model