

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class RecallCrossEntropy(torch.nn.Module):
    def __init__(self,  ignore_index= -1):
        super(RecallCrossEntropy, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, prediction , target): 
        # input (batch,n_classes,H,W)
        # target (batch,H,W)
        [B,C,H,W] = prediction.shape
      
        pred = prediction.argmax(1)
        if target.ndim == 4 :
            target = target.argmax(1)
        idex = (pred != target).view(-1) 
        
        #calculate ground truth counts
        gt_counter = torch.ones((C,), device= prediction.device )
        gt_idx, gt_count = torch.unique(target,return_counts=True)
        
        # map ignored label to an exisiting one
        if len(gt_count)>0:
            gt_count[gt_idx==self.ignore_index] = gt_count[-1].clone()
        gt_idx[gt_idx==self.ignore_index] = 1 
        gt_counter[gt_idx.long()] = gt_count.float()
        
        #calculate false negative counts
        fn_counter = torch.ones((C,), device= prediction.device ) 
        fn = target.view(-1)[idex]
        fn_idx, fn_count = torch.unique(fn,return_counts=True)
        
        # map ignored label to an exisiting one
        if len(fn_count)>0:
            fn_count[fn_idx==self.ignore_index] = fn_count[-1].clone()
        fn_idx[fn_idx==self.ignore_index] = 1 
        fn_counter[fn_idx.long()] = fn_count.float()
        
        weight = fn_counter / gt_counter
        
        CE = F.cross_entropy(prediction , target, reduction='none',ignore_index=self.ignore_index)
        loss =  weight[target.long()] * CE
        return loss.mean()





class MultiLabelDiceLoss(nn.Module):
    def __init__(
        self,
        mode: str ='multilabel',
        smooth: float =1.,
        log_loss: bool = False,
        eps: float = 1e-7,):
        super(MultiLabelDiceLoss, self).__init__()
        self.mode = mode
        self.smooth = smooth
        self.log_loss = log_loss
        self.eps = eps
       
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == 'binary':
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

        if self.mode == "multiclass":
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            #y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
            #y_true = y_true.permute(0, 2, 1)  # N, C, H*W

        if self.mode == 'multilabel':
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
        
        scores = self.soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)
        
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores
        return loss.mean()
    
    def soft_dice_score(self,
        output: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 0.0,
        eps: float = 1e-7,
        dims=None,) -> torch.Tensor:
        assert output.size() == target.size()
        if dims is not None:
            intersection = torch.sum(output * target, dim=dims)
            cardinality = torch.sum(output + target, dim=dims)
        else:
            intersection = torch.sum(output * target)
            cardinality = torch.sum(output + target)
        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
        return dice_score
