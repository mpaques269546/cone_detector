'''
FPN in PyTorch.
See the paper "Feature Pyramid Networks for Object Detection" for more details.
from https://github.com/Andy-zhujunwen/FPN-Semantic-segmentation/blob/master/FPN-Seg/model/FPN.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




class FPN(nn.Module):

    def __init__(self,  num_cls=2, activation=nn.Identity(), embed_dim=[96,192,384,768]):
        super(FPN, self).__init__()
        self.num_cls = num_cls
        self.activation = activation # try nn.Softmax(dim=1)
        if not isinstance(embed_dim, list):
            embed_dim = [embed_dim for i in range(4)]
    
        # semantic
        self.conv3 = nn.Conv2d(embed_dim[3], embed_dim[2], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(embed_dim[2], embed_dim[1], kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(embed_dim[1], embed_dim[0], kernel_size=3, stride=1, padding=1)
        self.conv0 = nn.Conv2d(embed_dim[0], embed_dim[0]//2, kernel_size=3, stride=1, padding=1)
        
        self.cls_head = nn.Conv2d( embed_dim[0]//2 , self.num_cls, kernel_size=1, stride=1, padding=0)
        
        # num_groups, num_channels: could be used with grad accum
        self.gn0 = nn.GroupNorm(embed_dim[0]//2, embed_dim[0]//2) 
        self.gn1 = nn.GroupNorm(embed_dim[0], embed_dim[0]) 
        self.gn2 = nn.GroupNorm(embed_dim[1], embed_dim[1])
        self.gn3 = nn.GroupNorm(embed_dim[2], embed_dim[2])

     
    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False) + y


    def forward(self, x, return_coarse_pred=False):
        # features from backbone
        s0 = x[0] #[B,D,H/4,W/4]
        s1 = x[1] #[B,D*2,H/8,W/8]
        s2 = x[2] #[B,D*4,H/16,W/16]
        s3 = x[3] #[B,D*8,H/32,W/32]

        [h, w] = s0.shape[-2:] # [_,_,H/4, W/4]
        
    
        # [B, 8*D, H//32, W//32]
        s3 = F.relu(self.gn3(self.conv3(s3)))
        s3 = F.relu(self.gn2(self.conv2(s3))) 
        s3 = F.relu(self.gn1(self.conv1(s3)))
        s3 = F.relu(self.gn0(self.conv0(s3)))
        s3 = self._upsample(s3, h, w)

        
        # [B, 4*D, H//16, W//16]
        s2 = F.relu(self.gn2(self.conv2(s2))) 
        s2 = F.relu(self.gn1(self.conv1(s2))) 
        s2 = F.relu(self.gn0(self.conv0(s2)))
        s2 = self._upsample(s2, h, w)
        
        # [B, 2*D, H//8, W//8]
        s1 = F.relu(self.gn1(self.conv1(s1)))
        s1 = F.relu(self.gn0(self.conv0(s1)))
        s1 = self._upsample(s1, h, w)
        
        # [B, D, H//4, W//4]
        s0 = F.relu(self.gn0(self.conv0(s0))) 
        
        # add
        s0 = self.cls_head(s0 + s1 + s2 + s3) # #[B, num_cls, H/4, W/4]
        
        
        if return_coarse_pred:
            return self.activation(s0)

        out = self._upsample(s0, 4 * h, 4 * w) ##[B,num_cls,H,W]
        # out: [B,C,H,W]

        return self.activation(out)




