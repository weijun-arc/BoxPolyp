import os
import sys
import numpy as np
from datetime import datetime
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import getModel, TrainData, TestData

class Train:
    def __init__(self, cfg):
        ## data
        self.cfg        = cfg
        self.writer     = SummaryWriter(cfg.log_path)
        self.loader     = DataLoader(dataset=TrainData(cfg.train_path), batch_size=cfg.batch_size, shuffle=True, num_workers=6)  
        self.model      = getModel(cfg, cfg.backbone)
        self.optimizer  = torch.optim.AdamW(self.model['model'].parameters(), cfg.lr, weight_decay=1e-4)
        self.cosine_lr  = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.epoch, eta_min=1e-5)

    def forward(self):
        global_step = 0
        for epoch in range(1, self.cfg.epoch):
            ## training phase
            for i, (image, mask) in enumerate(self.loader):
                ## feed forward
                image, mask     = image.cuda(), mask.cuda().unsqueeze(1)
                size            = np.random.choice([256, 352, 448], p=[0.25, 0.5, 0.25])
                image, mask     = F.interpolate(image, size=size, mode='bilinear'), F.interpolate(mask, size=size, mode='bilinear')
                B,C,H,W         = image.size()
                pred            = self.model['model'](image)
                pred            = F.interpolate(pred, size=size, mode='bilinear')
                ## loss calculation
                loss_ce         = F.binary_cross_entropy_with_logits(pred, mask)
                pred            = torch.sigmoid(pred)
                inter           = (pred*mask).sum(dim=(1,2,3))
                union           = (pred+mask).sum(dim=(1,2,3))
                loss_dice       = 1-(2*inter/(union+1)).mean()
                loss            = loss_ce+loss_dice
                ## optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                ## logging
                global_step += 1
                self.writer.add_scalar('lr'  , self.optimizer.param_groups[0]['lr'], global_step=global_step)
                self.writer.add_scalars('loss', {'ce':loss_ce.item(), 'dice':loss_dice.item()}, global_step=global_step)
                if i % 20 == 0:
                    print('{} Epoch={:03d}/{:03d}, Step={:04d}/{:04d}, ce={:0.4f}, dice={:0.4f}'.format(datetime.now().time(), epoch, self.cfg.epoch, i, len(self.loader), loss_ce.item(), loss_dice.item()))
            
            ## evaluation phase
            self.model['model'].eval()
            with torch.no_grad():
                dice, cnt = 0, 0
                for name_data in ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']:
                    loader              = DataLoader(dataset=TestData(cfg.test_path+name_data), batch_size=1, shuffle=False, num_workers=6) 
                    for image, mask, image_o, mask_o, name in loader:
                        image_o, mask_o = image_o[0].numpy(), mask_o[0].numpy()
                        H, W, C         = image_o.shape
                        pred            = self.model['model'](image.cuda().float())
                        pred            = F.interpolate(pred, size=(H, W), mode='bilinear')
                        pred            = (pred[0, 0]>0).cpu().numpy()
                        inter, union    = (pred*mask_o).sum(), (pred+mask_o).sum()
                        dice           += (2*inter+1)/(union+1)
                        cnt            += 1
                if dice/cnt > self.model['score']:
                    self.model['score'] = dice/cnt
                    torch.save(self.model['model'].state_dict(), cfg.model_path+'/'+self.model['name']+'_'+str(epoch)+'.pth')
            self.model['model'].train()
            self.model['model'].backbone.eval()

            ## adjusting lr
            self.cosine_lr.step()


if __name__ == '__main__':
    class Config:
        def __init__(self):
            self.mode           = 'train'
            self.backbone       = 'res2net50'
            self.snapshot       = None
            self.epoch          = 128
            self.lr             = 1e-4
            self.batch_size     = 16

            self.train_path     = '../dataset/Polyp/train/'
            self.test_path      = '../dataset/Polyp/test/'
            self.save_path      = './save'
            self.model_path     = self.save_path+'/model' 
            self.log_path       = self.save_path+'/log' 
            os.makedirs(self.model_path, exist_ok=True)
            os.makedirs(self.log_path, exist_ok=True)

    cfg = Config()
    Train(cfg).forward()
