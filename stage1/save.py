import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import getModel, LDPolypVideo
from tqdm import tqdm

class Test(object):
    def __init__(self, cfg):
        self.cfg      = cfg
        self.loader   = DataLoader(dataset=LDPolypVideo(cfg.test_path), batch_size=1, shuffle=False, num_workers=6)  
        self.model    = getModel(cfg, backbone=cfg.backbone)

    def save(self):
        with torch.no_grad():
            for image, mask, image_o, mask_o, name in tqdm(self.loader):
                ## image to pred
                image_o, mask_o = image_o[0].numpy(), mask_o[0].numpy()
                H, W, C         = image_o.shape
                pred            = self.model(image.cuda().float())
                pred            = F.interpolate(pred, size=(H, W), mode='bilinear')
                pred            = (pred[0,0]>0).cpu().numpy()
                ## dice calculation
                inter, union    = (pred*mask_o).sum(), (pred+mask_o).sum()
                dice            = 2*inter/union
                if dice>0.7:
                    ## save pseudo mask
                    pseudo = np.stack([mask_o, pred, pred], axis=2)
                    os.makedirs(self.cfg.test_path+'/mask/'+name[0].split('/')[0], exist_ok=True)
                    cv2.imwrite(self.cfg.test_path+'/mask/'+name[0]+'.png', np.uint8(pseudo*255))


if __name__=='__main__':
    class Config:
        def __init__(self):
            self.mode           = 'test'
            self.backbone       = 'res2net50'
            self.snapshot       = './save/model/res2net50_???.pth'  # select the last model for inference
            self.test_path      = '../dataset/LDPolypVideo'
    Test(Config()).save()
