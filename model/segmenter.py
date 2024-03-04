import torch
import torch.nn as nn
import math
import os
from torchvision import transforms
import torch.nn.functional as F

from .encoder import ResNet50, ViTAdapter
from .decoder import FPN, UperNet


class Segmenter(nn.Module):
    def __init__(
        self,
        image_size = 512,
        max_img_size=1536, 
        num_cls=2,
        activation=nn.Softmax(dim=1),
        pretrained_backbone_weights = None,
        pretrained_weights=None,
        key_encoder='encoder',
        key_decoder='decoder'):
        super().__init__()
        #self.encoder = ResNet50(num_classes=0, channels=3)
        #self.decoder =  FPN(num_cls=num_cls, activation=activation, embed_dim=[256,512,1024,2048])

        self.max_img_size = max_img_size
        self.image_size = image_size

        # vit adapter tiny + upernet
        self.encoder = ViTAdapter( pretrain_size=image_size, num_heads=3, conv_inplane=64, deform_num_heads=6,
                 patch_size= 16 , depth= 12, embed_dim= 192, drop_rate=0,
                 interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]], # 12 layers split in 4 blocks
                 cffn_ratio=0.25, deform_ratio=1.0, add_vit_feature=True, pretrained=None,
                 use_extra_extractor=True, with_cp=False, with_cls_token=False, qkv_bias=True,  window_attn=[False] * 12, window_size=[None] * 12)
        
        self.decoder = UperNet(in_channels=[192,192,192,192], hidden_size = 512, num_labels=num_cls, activation = activation )

        
        if pretrained_backbone_weights!=None:
            self.load(self.encoder, pretrained_weights)
        if pretrained_weights!=None:
            self.load(self.encoder, pretrained_weights, get_prefix=key_encoder+'.')
            self.load(self.decoder, pretrained_weights, get_prefix=key_decoder +'.')


    def forward_patch(self, patches):
        assert patches.ndim==4
        masks = []
        for b in range(patches.shape[0]):
            features = self.encoder.forward(patches[b].unsqueeze(0))
            mask = self.decoder.forward(features )
            del features
            masks.append(mask)
        del patches
        return torch.cat(masks, dim=0)
    
    def patchify(self, x, p=512):
        """
        x: (B, C, H, W)
        return: (B*n, C, p, p), n=H*W/p**2
        """
        assert x.dim()==4
        [B,C,H,W] = x.shape
        assert  H % p == 0 and W % p == 0
        h = H // p
        w = W // p
        n = h*w
        x = x.reshape(shape=(B, C, h, p, w, p))
        x = torch.einsum('bchpwq->bhwcpq', x)
        x = x.reshape(shape=(B*n, C, p, p))
        return x, h, w

    def unpatchify(self, x, B, h, w,  p=512):
        """
        x: [B*n, C, p, p]
        return: (B, C, H, W)
        """
        [Bn, C, p, p] = x.shape
        #n = int(Bn/B)
        #w = h = int(n**.5)
        x = x.reshape(shape=(B, h, w, C, p, p))
        x = torch.einsum('bhwcpq->bchpwq', x)
        x = x.reshape(shape=(B, C, h * p, w * p))
        return x

    
    def forward(self, x ):
        B0_, C0_, H0_, W0_  = x.shape
        
        if H0_>self.max_img_size or W0_>self.max_img_size:
            x = transforms.Resize( self.max_img_size)(x)
            B0, C0, H0, W0  = x.shape
        else:
            B0, C0, H0, W0 = B0_, C0_, H0_, W0_

        pad_H =   (H0//self.image_size + 1 ) * self.image_size -  H0
        pad_W = (W0//self.image_size +  1) * self.image_size - W0
        x = F.pad(x , ( 0, pad_W, 0, pad_H ) , "constant", 0)
                
        
        x, h, w  = self.patchify( x, p = self.image_size )
        x = self.forward_patch(x)
        x = self.unpatchify(x, B0, h, w,  p = self.image_size )
        x =  x[:,:,:H0, :W0]
        
        if H0_!=H0 or W0_!=W0:
            x = transforms.Resize( (H0_,W0_))(x)
        
        return x
            

    def load(self, model, pretrained_weights, get_prefix=''):
        param_dict = torch.load(pretrained_weights, map_location="cpu")
        print(param_dict.keys())
        if 'state_dict' in param_dict.keys():
            param_dict = param_dict['state_dict']
        if 'model' in param_dict.keys():
            param_dict = param_dict['model']
        #param_dict = {k.replace("batch_norm", "bn"): v for k, v in param_dict.items()}
        #param_dict = {k.replace("i_downsample", "downsample"): v for k, v in param_dict.items()}
        param_dict = {k.replace("gamma1", "gamma_1"): v for k, v in param_dict.items()}
        param_dict = {k.replace("gamma2", "gamma_2"): v for k, v in param_dict.items()}

        
        if len(get_prefix)>0:
            k_list = list(param_dict.keys())
            for k in k_list:
                if get_prefix not in k:
                    param_dict.pop(k)
            param_dict = {k.replace(get_prefix, ""): v for k, v in param_dict.items()}
        
        
        model_dict = model.state_dict()

        for k in param_dict.keys():
            if k in model_dict.keys():
                p1 = model_dict[k]
                p2 = param_dict[k]
                if p1.shape != p2.shape:
                    print(f'Resize {k} from shape {p2.shape} to {p1.shape}')
                    if p2.ndim==1 or p2.ndim==0:
                        p2 = p2.reshape(1,1,-1)
                        new_shape = p1.shape[0]
                    elif p2.ndim ==2:
                        p2 = p2.permute(1,0).unsqueeze(0)
                        new_shape = p1.shape[1]
                    elif p2.ndim==3:
                        p2 = p2.permute(0,2,1)
                        new_shape = p1.shape[1]
                    p2 = torch.nn.functional.interpolate(p2, size=new_shape, mode='linear')
                    print(p2.shape )
                    param_dict[k] = p2.permute(0,2,1).squeeze(1).squeeze(2)
        msg = model.load_state_dict(param_dict, strict= False )
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))




    def forward_old(self, x ):
        (B,C,H0,W0) = x.shape
        if H0!=self.image_size or W0!=self.image_size:
            print('resize')
            coeff = self.image_size / max([H0,W0])
            x = transforms.Resize(size= (int(coeff*H0), int(coeff*W0)))(x) # resize s.t. max_size = 512
            pad_H =   abs( x.shape[-2] - 512 )
            pad_W = abs( x.shape[-1] - 512 )
            x = transforms.Pad(padding =( pad_W, pad_H, 0, 0 ) , fill=0)(x) # padding ( left, top, right and bottom) # pad to have (512,512)
        features = self.encoder.forward(x)
        mask = self.decoder.forward(features )
        if H0!=self.image_size or W0!=self.image_size:
            mask = mask[:,:,pad_H:, pad_W:] # unpad
            mask = transforms.Resize(size = (H0,W0))(mask) #resize initial size
        return mask

