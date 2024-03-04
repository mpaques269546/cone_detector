import os
import argparse
from pathlib import Path
import copy
import json 

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models


import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchmetrics
from torchmetrics import JaccardIndex
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import time


#Utils
import utils
from visualize import plot_results, plot_confusion_matrix, plot_tsne
from model import Segmenter
from dataset import MyTransform, MyDataset
from losses import RecallCrossEntropy, MultiLabelDiceLoss
from solver import Lion

import warnings
warnings.filterwarnings("ignore") #category=DeprecationWarning)



def segment(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ============ Data Loader ===============
    os.makedirs(args.output_dir, exist_ok = True)

    val_transform = MyTransform(hflip=False, vflip=False, rcrop=False, ccrop=False, colorjitter=False, size=args.image_size, normalize=True)
    train_transform = MyTransform(hflip=True, vflip=False, rcrop=False, ccrop=False, colorjitter=True, size=args.image_size, normalize=True)

    #infer_transform = MyTransform(hflip=False, vflip=False, crop=False,colorjitter=False, size=args.image_size ) #(512,512))
    dataset_val =   MyDataset(args.data_path+'val/', val_transform )
    dataset_train = MyDataset(args.data_path+'train/', train_transform  )
    dataset_infer = MyDataset(args.data_path+'test/', train_transform )
   
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu, #args.batch_size_per_gpu,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True) # ,sampler=sampler)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train) # drop last data to evenly dispatch data between GPUs
    train_loader = torch.utils.data.DataLoader(dataset_train,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler)

    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")
    num_cls = dataset_val.num_cls
    
    # ============ Models ===============

    segmenter = Segmenter(pretrained_weights=args.pretrained_weights, image_size=args.image_size, activation=nn.Identity())
    segmenter = segmenter.cuda()
    segmenter = nn.parallel.DistributedDataParallel(segmenter, device_ids=[args.gpu], find_unused_parameters=True) # find_unused_parameters=True


    # ============ Loss & Optimizers ===============
    if args.loss=='ce':
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.loss=='recall_ce':
        criterion = RecallCrossEntropy().cuda()
    elif args.loss=='dice':
        criterion = MultiLabelDiceLoss(mode='multiclass', smooth=1., log_loss= False, eps= 1e-7,).cuda()

    if args.optimizer=='adamw':
        optimizer = torch.optim.AdamW(segmenter.parameters(), lr=args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., weight_decay=args.weight_decay)
    elif args.optimizer=='lion':
        optimizer = Lion(segmenter.parameters(), lr=args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., weight_decay=args.weight_decay)
    elif args.optimizer=='sgd':
        optimizer = torch.optim.SGD(segmenter.parameters(), args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., momentum=0.9,weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)


    # ============ RUN ===============

    start_epoch = 0
    best_miou = 0
    

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train_stats = train(segmenter, optimizer, criterion, train_loader, epoch)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(segmenter, criterion, val_loader, epoch)
            print(f"Epoch {epoch} validation:  miou={test_stats['miou']:.2f}% ")
            inference(segmenter, dataset_infer, num_cls, epoch)
            
            if best_miou< test_stats['miou']:
                print('miou improved from {best_miou:.2f}% to {miou:.2f}%'.format(best_miou=best_miou, miou =test_stats['miou']) )
                best_miou = test_stats['miou']
                best_models = copy.deepcopy(segmenter)
                # save best models
                save_dict = {'model': best_models.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': args,}
                
                torch.save(save_dict, os.path.join(args.output_dir, 'models.pth'))
            else:
                print('no improvement')
            
            
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}

        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        plot_results(filename = args.output_dir+'log.txt', metrics_keys= ['test_miou'],  savename =args.output_dir + 'results.png')
    
        
        
    print(f"BEST MODEL: epoch {save_dict['epoch']:.0f}")
    test_stats = validate_network(best_models, criterion, val_loader, 10**4)   
    print(f"Validation: miou={test_stats['miou']:.2f}%")
    
    # kill the process group
    dist.destroy_process_group() 

def train(model, optimizer, criterion, loader, epoch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for batch_idx, (inp, target) in enumerate( metric_logger.log_every(loader, 50, header) ):
        # forward
        output = model.forward(inp.cuda(non_blocking=True) )
        loss = criterion.forward(output , target.squeeze(1).cuda(non_blocking=True))
        # compute the gradients
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item()) #.item(): copy from gpu to RAM
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def validate_network(model, criterion, loader, epoch):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    mean_iou = JaccardIndex(task='multiclass', num_classes=2, average=None, ignore_index=-1 ).cuda()
    
    for inp, target in metric_logger.log_every(loader, 50, header):
        #target: [B, C, H, W] or [B,H,W]
       
        # forward
        output = model.module.forward(inp.cuda(non_blocking=True))
        # output: [B, C, H, W]

        # loss
        loss = criterion.forward(output, target.squeeze(1).cuda(non_blocking=True))
        
                        
        #metrics
        if target.ndim==4:
            target = target.argmax(1)
        mean_iou.update(output.argmax(1) , target.long().cuda(non_blocking=True) )
        metric_logger.update(loss=loss.item())
        
    miou = mean_iou.compute().cpu() * 100
    metric_logger.meters['miou'].update( torch.mean(miou).item() )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def inference(model, dataset, num_cls, epoch):
    model.eval()
    palette = np.random.rand(num_cls,3)
    palette[0,:], palette[1,:] = np.array([1,1,1]), np.array([0,1,0])
    
    # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    inv_normalize = pth_transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
         std=[1/0.229, 1/0.224, 1/0.225])
    indexes = torch.randint(low=0, high=len(dataset), size=(10,))
    
    for i in indexes:
        fig_name = 'epoch_'+str(epoch)+'_img_'+str(int(i))+'.png'

        inp, target = dataset[i]

        t1 = time.time()
        output = model.module.forward(inp.unsqueeze(0).cuda(non_blocking=True) )
        # output size [b n h w]
        t2 = time.time()
        dt = round(t2-t1, 3)

        
        output = output.squeeze().cpu()
        [n_cls, w, h] = output.shape
        
        img = inv_normalize(inp)
        img = np.array(TF.to_pil_image(img))/255


        # Ground truth
        if len(target.shape)==3: #[C, H, W]
            target = target.argmax(0) ##[H, W]
        gt_mask = target.squeeze().numpy()
        gt_img = img.copy()
    
        # Prediction
        pred_mask = output.argmax(0).numpy()
        pred_img = img.copy()

        for n in range( n_cls):
            color = palette[n]
            gt = (gt_mask==n)*1
            pred = (pred_mask==n)*1
            for c in range(3):
                gt_img[:,:, c] = gt_img[:,:,c]*(1-gt) + gt_img[:,:,c]*gt*color[c]
                pred_img[:,:, c] = pred_img[:,:,c]*(1-pred) + pred_img[:,:,c]*pred*color[c]
        
        # add 1 to have same color range for the mask [0,1]
        pred_mask, gt_mask = pred_mask/num_cls, gt_mask/num_cls
        pred_mask[0,0], gt_mask[0,0]= 1,1
        pred_mask[-1,-1], gt_mask[-1,-1]= 0,0

        # plot
        fig = plt.figure(figsize=(10,10))

        ax = fig.add_subplot(2, 3, 1)
        ax.title.set_text('image')
        plt.imshow(img)
        #plt.axis('off')
        plt.tight_layout(pad=0., w_pad=0.5, h_pad=0.1)

        ax = fig.add_subplot(2, 3, 2)
        ax.title.set_text('GT image')
        plt.imshow(gt_img)
        plt.axis('off')
        plt.tight_layout(pad=0., w_pad=0.5, h_pad=0.1)

        ax = fig.add_subplot(2, 3, 3)
        ax.title.set_text('GT mask')
        plt.imshow( gt_mask)
        plt.tight_layout()
        plt.axis('off')

        ax = fig.add_subplot(2, 3, 4)
        ax.title.set_text('image')
        plt.imshow(img)
        #plt.axis('off')
        plt.tight_layout(pad=0., w_pad=0.5, h_pad=0.1)

        ax = fig.add_subplot(2, 3, 5)
        ax.title.set_text('Pred image')
        plt.imshow(pred_img)
        plt.axis('off')
        plt.tight_layout(pad=0., w_pad=0.5, h_pad=0.1)

        ax = fig.add_subplot(2, 3, 6)
        ax.title.set_text('Pred mask, dt='+str(dt)+'s')
        plt.imshow( pred_mask)
        plt.tight_layout()
        plt.axis('off')
                   
        # save
        plt.savefig(os.path.join(args.output_dir+fig_name))
        plt.close()
        
    print('inference imgs saved to {0}'.format(args.output_dir))
        







if __name__ == '__main__':
    parser = argparse.ArgumentParser('Segmentation')
    # model
    parser.add_argument("--image_size", default=512, type=int,  help="Resize image. [224, 320, 480]")
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    # misc
    parser.add_argument('--epochs', default=1000, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated. Use 0.01 for linear, 0.001 for vit""")
    parser.add_argument('--batch_size_per_gpu', default=4, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=50, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="output/", help='Path to save logs and checkpoints')
    # data
    parser.add_argument('--data_path', default='datasets/tree/', type=str)
    parser.add_argument('--loss', default='dice', type=str)
    parser.add_argument('--optimizer', default='adamw', type=str)
    
    parser.add_argument('--seed', default=0, type=int)
    # Weights constraints
    parser.add_argument('--weight_decay', default=0.001, type=float, help='weight decay value. If 0, no weight decay.')
   
    global args
    args = parser.parse_args()
    segment(args)


"""

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$((RANDOM + 10000))  train.py --data_path datasets/cone/ --pretrained_weights weights/resnet50_miil_21k.pth

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$((RANDOM + 10000))  train.py --data_path datasets/cone/ --pretrained_weights weights/upernet_augreg_adapter_tiny_512_160_ade20k.pth



"""




