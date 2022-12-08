import torch
import torch.nn as nn
import torch.nn.functional as F
from network import pvt_v2_b2, Res2Net50, weight_init

class BoxPolyp(nn.Module):
    def __init__(self, args):
        super(BoxPolyp, self).__init__()
        if args.backbone=='res2net50':
            self.backbone = Res2Net50()
            channels      = [256, 512, 1024, 2048]
        if args.backbone=='pvt_v2_b2':
            self.backbone = pvt_v2_b2()
            channels      = [64, 128, 320, 512]

        self.args         = args
        self.linear1      = nn.Sequential(nn.Conv2d(channels[0], 32, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.linear2      = nn.Sequential(nn.Conv2d(channels[1], 32, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.linear3      = nn.Sequential(nn.Conv2d(channels[2], 32, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.linear4      = nn.Sequential(nn.Conv2d(channels[3], 32, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.linear       = nn.Conv2d(32, 1, 1)
        self.initialize()

    def forward(self, x):
        x1, x2, x3, x4    = self.backbone(x)
        x1, x2, x3, x4    = self.linear1(x1), self.linear2(x2), self.linear3(x3), self.linear4(x4)
        x3                = F.interpolate(x3, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x4                = F.interpolate(x4, size=x2.size()[2:], mode='bilinear', align_corners=True)
        pred              = self.linear(x2*x3*x4)
        return pred

    def initialize(self):
        if self.args.snapshot:
            self.load_state_dict(torch.load(self.args.snapshot))
        else:
            weight_init(self)