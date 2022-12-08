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
from utils import getModel, TrainData, TestData, bce_dice

import matplotlib.pyplot as plt

class Train:
    def __init__(self, cfg):
        ## data
        self.cfg        = cfg
        self.writer     = SummaryWriter(cfg.log_path)
        self.loader     = DataLoader(dataset=TrainData(cfg.train_path), batch_size=cfg.batch_size, shuffle=True, num_workers=6)  
        self.res2net50  = getModel(cfg, 'res2net50')
        self.pvt_v2_b2  = getModel(cfg, 'pvt_v2_b2')
        self.optimizer  = torch.optim.AdamW([{'params':self.res2net50['model'].parameters()}, {'params':self.pvt_v2_b2['model'].parameters()}], cfg.lr, weight_decay=1e-4)
        self.cosine_lr  = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.epoch, eta_min=1e-5)

    def forward(self):
        global_step = 0
        for epoch in range(1, self.cfg.epoch):
            ## training phase
            for i, (image, mask) in enumerate(self.loader):
                ## res2net
                image, mask                 = image.cuda(), mask.cuda()
                size1                       = np.random.choice([256, 352, 448], p=[0.25, 0.5, 0.25])
                image1                      = F.interpolate(image, size=size1, mode='bilinear')
                pred1                       = self.res2net50['model'](image1)
                pred1                       = F.interpolate(pred1, size=352, mode='bilinear')
                loss_ce1, loss_dice1, valid = bce_dice(pred1, mask)
                ## pvt_v2_b2
                size2                       = np.random.choice([256, 352, 448], p=[0.25, 0.5, 0.25])
                image2                      = F.interpolate(image, size=size2, mode='bilinear')
                pred2                       = self.pvt_v2_b2['model'](image2)
                pred2                       = F.interpolate(pred2, size=352, mode='bilinear')
                loss_ce2, loss_dice2, valid = bce_dice(pred2, mask)
                ## consistency
                loss_con                    = (torch.sigmoid(pred1)-torch.sigmoid(pred2))**2
                loss_con                    = (loss_con.squeeze()*(1-valid)).sum()/(1-valid).sum()
                ## optimization
                loss = loss_ce1 + loss_dice1 + loss_ce2 + loss_dice2 + loss_con
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                ## logging
                global_step += 1
                self.writer.add_scalar('lr'  , self.optimizer.param_groups[0]['lr'], global_step=global_step)
                self.writer.add_scalars('loss', {'ce_res':loss_ce1.item(), 'dice_res':loss_dice1.item(), 'ce_pvt':loss_ce2.item(), 'dice_pvt':loss_dice2.item(), 'contrast':loss_con.item()}, global_step=global_step)
                if i % 20 == 0:
                    print('{} Epoch={:03d}/{:03d}, Step={:04d}/{:04d}, ce-res={:0.4f}, dice-res={:0.4f}, ce-pvt={:0.4f}, dice-pvt={:0.4f}, loss_con={:0.4f}'.format(datetime.now().time(), epoch, self.cfg.epoch, i, len(self.loader), loss_ce1.item(), loss_dice1.item(), loss_ce2.item(), loss_dice2.item(), loss_con.item()))
            
            ## evaluation phase
            for model in [self.res2net50, self.pvt_v2_b2]:
                model['model'].eval()
                with torch.no_grad():
                    dice, cnt = 0, 0
                    for name_data in ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']:
                        loader              = DataLoader(dataset=TestData(cfg.test_path+name_data), batch_size=1, shuffle=False, num_workers=6) 
                        for image, mask, image_o, mask_o, name in loader:
                            image_o, mask_o = image_o[0].numpy(), mask_o[0].numpy()
                            H, W, C         = image_o.shape
                            pred            = model['model'](image.cuda().float())
                            pred            = F.interpolate(pred, size=(H, W), mode='bilinear')
                            pred            = (pred[0, 0]>0).cpu().numpy()
                            inter, union    = (pred*mask_o).sum(), (pred+mask_o).sum()
                            dice           += (2*inter+1)/(union+1)
                            cnt            += 1
                    if dice/cnt > model['score']:
                        model['score'] = dice/cnt
                        model['logger'].info('epoch=%d ########### dice=%.4f'%(epoch, dice/cnt))
                        torch.save(model['model'].state_dict(), cfg.model_path+'/'+model['name']+'_'+str(epoch)+'.pth')
                    else:
                        model['logger'].info('epoch=%d, dice=%.4f'%(epoch, dice/cnt))
                model['model'].train()
                model['model'].backbone.eval()

            ## adjusting lr
            self.cosine_lr.step()


if __name__ == '__main__':
    class Config:
        def __init__(self):
            self.mode           = 'train'
            self.backbone       = None
            self.snapshot       = None
            self.epoch          = 80
            self.lr             = 1e-4
            self.batch_size     = 16

            self.train_path     = ['../dataset/Polyp/train/', '../dataset/LDPolypVideo/']
            self.test_path      = '../dataset/Polyp/test/'
            self.save_path      = './save'
            self.model_path     = self.save_path+'/model' 
            self.log_path       = self.save_path+'/log' 
            os.makedirs(self.model_path, exist_ok=True)
            os.makedirs(self.log_path, exist_ok=True)

    cfg = Config()
    Train(cfg).forward()
