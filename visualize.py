# plot

from sklearn.metrics import precision_score, accuracy_score, f1_score, ConfusionMatrixDisplay, confusion_matrix
import torch
from sklearn.manifold import TSNE

import pandas as pd
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import os

def txt_to_df(filename):
    with open(filename) as f:
        lines = []
        for line in f:
            line = line.strip()
            line = json.loads(line)
            lines.append( line )
    #keys = lines[0].keys()

    df = pd.DataFrame(lines)
    return df

def plot_results(filename, metrics_keys , savename):
    if os.path.isfile(filename):
        df = txt_to_df(filename)
    else:
        print(f'ERROR: no file {filename}')
        return None
        #df = pd.DataFrame({"epoch": [0], "train_loss": [0], "test_loss": [0], "test_acc": [0], "test_iou": [0], "test_prec":[0], "test_f1":[0] , "test_iou2":[0], "test_miou":[0] })
    if len(df)==0:
        #df = pd.DataFrame({"epoch": [0], "train_loss": [0], "test_loss": [0], "test_acc": [0], "test_iou": [0], "test_prec":[0], "test_f1":[0] , "test_iou2":[0], "test_miou":[0] })
        return None
    
    train_loss_keys = list(filter(lambda a: ('train_loss' in a), df.keys().tolist()))
    val_loss_keys = list(filter(lambda a: ('test_loss' in a), df.keys().tolist()))

    line = ['solid', 'dotted', 'dashed', 'dashdot', 'solid', 'dotted']
    val_color = ['blue', 'darkblue', 'cyan', 'purple', 'teal', 'navy', 'lightblue', 'blue', 'darkblue', 'cyan', 'purple', 'teal', 'navy']
    train_color = ['red', 'darkred', 'orange', 'magenta', 'lightcoral', 'red', 'darkred', 'orange', 'magenta']
    
    fig, ax1 = plt.subplots()
   
    # train_losses
    for i in range(len(train_loss_keys)):
        key = train_loss_keys[i]
        df_ = df.where(df[key]!='').dropna(how='all')
        loss = df_[key]
        epochs = df_['epoch']
        ax1.plot(epochs, loss, color=train_color[i], linestyle = line[i], linewidth='4' , label = key)
    for i in range(len(val_loss_keys)):
        key = val_loss_keys[i]
        df_ = df.where(df[key]!='').dropna(how='all')
        loss = df_[key]
        epochs = df_['epoch']
        ax1.plot(epochs, loss, color=val_color[i], linestyle = line[i], linewidth='4' , label = key)
    
    # metrics
    ax2 = ax1.twinx()
    for i in range(len(metrics_keys)):
        key = metrics_keys[i]
        df_ = df.where(df[key]!='').dropna(how='all')
        metrics = df_[key]
        epochs = df_['epoch']
        ax2.plot(epochs, metrics, color=val_color[i], linestyle = line[i], label = key)
    
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    ax1.set_xlabel('epochs', fontsize=14)
    ax1.set_ylabel("loss",fontsize=14)
    ax2.set_ylabel("metrics %",fontsize=14)
    plt.savefig(str(savename), facecolor='w')
    plt.clf()



'''
def plot_results(train_loss, val_loss, val_metrics, metrics_label, fig_name, path):
	# train loss: size (n_epochs, n_losses)
	# val_loss: size (n_epochs/val_freq, n_losses)
	# val_metrics: size (n_epochs/val_freq, n_labels, n_metrics)
	# metrics_label: size (n_metrics)
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    val_metrics = np.array(val_metrics)
    n_epochs = train_loss.shape[0]
    val_freq =  n_epochs//val_loss.shape[0]
    train_epochs = range(n_epochs)
    val_epochs = [i*val_freq for i in range(val_loss.shape[0])]
    val_color = ['lightblue', 'blue', 'darkblue', 'cyan']
    train_color = ['lightcoral', 'red', 'darkred', 'orange']
    line = ['solid', 'dotted', 'dashed', 'dashdot', 'solid', 'dotted']
    
    fig, ax1 = plt.subplots()
    for i in range(train_loss.shape[1]):
    	ax1.plot(train_epochs, train_loss[:,i], color=train_color[i], linestyle = line[0], label = 'train_loss'+str(i))
    
    for i in range( val_loss.shape[1]):
    	ax1.plot(val_epochs, val_loss[:, i], color=val_color[i], linestyle = line[0], label = 'val_loss'+str(i))
    
    ax2 = ax1.twinx()
    for i in range(val_metrics.shape[1]):# what label
    	for j in range( val_metrics.shape[2]): # what metrics
    		ax2.plot(val_epochs, val_metrics[:, i, j], color=val_color[i+1], linestyle = line[j+1], label = metrics_label[j]+str(i+1))
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    ax1.set_xlabel('epochs', fontsize=14)
    ax1.set_ylabel("loss",fontsize=14)
    ax2.set_ylabel("accuracy %",fontsize=14)
    #ax2.set_ylim([0, 100])
    plt.savefig(path + str(fig_name)+'.png', facecolor='w')
    plt.clf()
'''





def plot_confusion_matrix(pred, targ, labels, fig_name='', path="./",  ordinal=False):
    """in case of single label -> no pb: best pred class with true class
    in case of multi labe: approximation: n_best pred classes with n true classes
    """
    print(f"pred {pred[:10]}")
    print(f"targ {targ[:10]}")
    labels= np.array(labels)
    if ordinal:
        #pred_classes = torch.sum( (pred>0.5)*1 , dim=1) -1
        pred_classes = ordinal_decoding(pred)
        targ_classes = torch.sum( (targ>0.5)*1 , dim=1) -1
    else:
        pred_classes = pred.argmax(1)
        targ_classes = targ.argmax(1)
   
    pred_classes = pred_classes.long().squeeze().tolist()
    targ_classes = targ_classes.long().squeeze().tolist()
    print(f"pred_classes {pred_classes[:10]}")
    print(f"targ_classes {targ_classes[:10]}")

    pred_classes= labels[pred_classes]
    targ_classes= labels[targ_classes]
    print(f"labels pred_classes {pred_classes[:10]}")
    print(f"labels targ_classes {targ_classes[:10]}")
    
    fig, ax = plt.subplots(figsize=(5,5))
    ConfusionMatrixDisplay.from_predictions( targ_classes , pred_classes,
        normalize='true',ax=ax, xticks_rotation='vertical', colorbar=True, include_values=False)
    plt.tight_layout()
    plt.savefig(path+ 'confusion_'+fig_name,  facecolor='w')
    plt.close('all')

def ordinal_decoding(prediction):
    num_cls = prediction.shape[-1]
    samples = torch.zeros((num_cls, num_cls), device = prediction.device)
    for i in range(num_cls):
        samples[i,:i+1] = 1
    distance = torch.cdist(prediction , samples, p=2 ) # euclidean distance
    #distance = torch.cdist( (prediction>0.5)*1. , samples, p=0) # hamming distance
    pred = distance.argmin(-1)
    return pred 


def plot_tsne(features, targets, class_list, fig_name, path):
    print('start t-sne ...')
    tsne = TSNE(n_components= 2, random_state=123,init='pca', perplexity=50, 
        early_exaggeration= 15, learning_rate=200, metric='euclidean', n_jobs=4)
    z = tsne.fit_transform(features)
    fig = plt.figure(figsize=(8, 8))
    [n,c]=targets.shape
    np.random.seed(1)
    cmap = np.random.rand(c, 3)
    #cmap = plt.cm.get_cmap('hsv', c) ##cmap(i)
    ax = fig.add_subplot(1,1,1)
    ax.title.set_text('t-SNE')
    for i in range(c):
        name = class_list[i]
        idx = torch.where(targets[:, i]==1)[0]
        if len(idx)>0:
            ax.scatter( z[idx,0] , z[idx,1], s=8, color=cmap[i], label=name)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path+'tsne_representation_'+fig_name+'.png', facecolor='w')
    plt.clf()
    print('t-sne done!')

