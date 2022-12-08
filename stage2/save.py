import os
import sys
import cv2
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import getModel, TestData
from tqdm import tqdm

class Test(object):
    def __init__(self, cfg):
        self.cfg      = cfg
        self.loader   = DataLoader(dataset=TestData(cfg.test_path), batch_size=1, shuffle=False, num_workers=6)  
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
                ## save prediction
                os.makedirs(self.cfg.save_path, exist_ok=True)
                cv2.imwrite(self.cfg.save_path+'/'+name[0]+'.png', np.uint8(pred*255))


if __name__=='__main__':
    class Config:
        def __init__(self, name, backbone, snapshot, save_path):
            self.mode           = 'test'
            self.test_path      = '../dataset/Polyp/test/'+name
            self.backbone       = backbone
            self.snapshot       = snapshot
            self.save_path      = save_path+'/'+name
    ## save predictions with res2net50 backbone
    for name in ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']:
        cfg = Config(name, 'res2net50', './save/model/res2net50_??.pth', '../evaluate/prediction/BoxPolyp-Res')  ## select the last res2net50 model
        Test(cfg).save()
    ## save predictions with pvt_v2_b2 backbone
    for name in ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']:
        cfg = Config(name, 'pvt_v2_b2', './save/model/pvt_v2_b2_??.pth', '../evaluate/prediction/BoxPolyp-Pvt') ## select the last pvt_v2_b2 model
        Test(cfg).save()