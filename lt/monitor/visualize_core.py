#######!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:00:26 2018

@author: chemla
"""
import pdb, torch, numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import itertools
from utils.onehot import fromOneHot
from sklearn.metrics import confusion_matrix
import matplotlib.patches as mpatches


###############################################
##### 
##      Various utilies for plotting


def decudify(obj, scalar=False):
    if issubclass(type(obj), dict):
        return {k:decudify(v, scalar=scalar) for k,v in obj.items()}
    elif issubclass(type(obj), list):
        return [decudify(i, scalar=scalar) for i in obj]
    elif issubclass(type(obj), tuple):
        return tuple([decudify(i, scalar=scalar) for i in list(obj)])
    elif torch.is_tensor(obj):
        obj_cpu = obj.cpu()
        if scalar:
            obj_cpu = obj_cpu.detach().numpy()
        del obj
        return obj_cpu
    else:
        return obj

def merge_dicts(obj):
    if not (issubclass(type(obj),list) or issubclass(type(obj),tuple)) or len(obj) == 0 :
        return
    list_type = type(obj)
    if issubclass(type(obj[0]), dict):
        return {k:merge_dicts([ d[k] for d in obj ]) for k, v in obj[0].items()}
    elif issubclass(type(obj[0]), torch.Tensor):
        return torch.cat(obj, 0)
    elif issubclass(type(obj[0]), np.ndarray):
        return list_type(np.concatenate(obj, 0))
    elif issubclass(type(obj[0]), list) or issubclass(type(obj[0]), tuple) :
        return list_type([ merge_dicts( [l[i] for l in obj] ) for i in range(len(obj[0])) ])

def get_cmap(n, color_map='plasma'):
    return plt.cm.get_cmap(color_map, n)

def get_class_ids(dataset, task, ids=None):
    if ids is None:
        ids = np.arange(dataset.data.shape[0])
    metadata = np.array(dataset.metadata.get(task)[ids])
    if metadata is None:
        raise Exception('please give a valid metadata task')
    n_classes = list(set(metadata))
    ids = []
    for i in n_classes:
        ids.append(np.where(metadata==i)[0])
    return ids, n_classes

def get_divs(n):
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
       primfac.append(n)
    primfac = np.array(primfac)
    return int(np.prod(primfac[0::2])), int(np.prod(primfac[1::2]))


def get_balanced_ids(metadata, n_points=None):
    items = list(set(metadata))
    class_ids = []
    n_ids = metadata.shape[0] // len(items) if n_points is None else int(n_points) // len(items)
    for i, item in enumerate(items):
        class_ids.append(np.where(metadata==item)[0])
        if len(class_ids[-1]) < (n_ids // len(items)):
            n_ids = len(class_ids[-1])
    class_ids = [list(i[:n_ids]) for i in class_ids]
    ids = reduce(lambda x,y: x+y, class_ids)
    return ids


# Plots two distributions side by side, depending of the data type
def plot_comparison(data, synth, var=None):
    n_rows, n_columns = get_divs(data.shape[0])
    fig = plt.figure()
    if len(data.shape) == 2:
        axes = fig.subplots(n_rows, n_columns)
        for i in range(n_rows):
            for j in range(n_columns):
                axes[i,j].plot(data[i*n_columns+j])
                axes[i,j].plot(synth[i*n_columns+j], linewidth=0.5)
                if not var is None:
                    axes[i,j].bar(range(var.shape[1]), var[i*n_columns+j], align='edge', alpha=0.4)
                plt.tick_params(axis='y',  which='both',  bottom='off')
    elif len(data.shape) == 3:
        axes = fig.subplots(n_rows, 2*n_columns)
        for i in range(n_rows):
            for j in range(n_columns):
                if axes.ndim == 1:
                    axes = axes[np.newaxis, :]
                axes[i,2*j].imshow(data[i*n_columns+j], aspect='auto')
                axes[i,2*j].set_title('data')
                axes[i,2*j+1].imshow(synth[i*n_columns+j], aspect='auto')
                axes[i,2*j+1].set_title('reconstruction')
    return fig, axes
                
def plot_latent_path(zs, data, synth, reduction=None):
    fig = plt.figure()
    grid = plt.GridSpec(1, 4, hspace=0.2, wspace=0.2)
        
    if not reduction is None:
        zs = reduction.transform(zs)
        
    gradient = get_cmap(zs.shape[0])
    if zs.shape[1] == 2:
        ax = fig.add_subplot(grid[:2])
        ax.plot(zs[:,0],zs[:,1], c=gradient(np.arange(zs.shape[0])))
    elif zs.shape[1] == 3:
        ax = fig.add_subplot(grid[:2], projection='3d', xmargin=1)
        for i in range(zs.shape[0]-1):
            ax.plot([zs[i,0], zs[i+1, 0]], [zs[i,1], zs[i+1, 1]], [zs[i,2], zs[i+1, 2]], c=gradient(i))

    ax = fig.add_subplot(grid[2])
    ax.imshow(data, aspect='auto')
    ax = fig.add_subplot(grid[3])
    ax.imshow(synth, aspect='auto')
    return fig, fig.axes

def plot(current_z, *args, **kwargs):
    if current_z.shape[1] == 2:
        fig, ax = plot_2d(current_z, *args, **kwargs)
    elif current_z.shape[1] == 3:
        fig, ax = plot_3d(current_z, *args, **kwargs)
    else:
        fig, ax = plot_3d(current_z[:, :3], *args, **kwargs)
    return fig, ax


def plot_2d(current_z, meta=None, var=None, classes=None, class_ids=None, class_names=None, cmap='plasma', centroids=None, legend=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if meta is None:
        meta = np.zeros((current_z.shape[0]))
        cmap = get_cmap(0)
    else:
        cmap = get_cmap(len(classes))

    current_alpha = 0.06 if (centroids and not meta is None) else 0.9
    current_var = var if not var is None else np.ones(current_z.shape[0])
    # plot
    if current_z.shape[1]==2:
        ax.scatter(current_z[:, 0], current_z[:,1], c=cmap(meta), alpha = current_alpha, s=current_var)
    else:
        ax.scatter(current_z[:, 0], current_z[:,1], c=cmap(meta), alpha = current_alpha, s=current_var)
    # make centroids
    if centroids and not meta is None:
        for i, cid in enumerate(class_ids):
            centroid = np.mean(current_z[cid], axis=0)
            ax.scatter(centroid[0], centroid[1], s = 30, c=cmap(classes[i]))
            ax.text(centroid[0], centroid[1], class_names[i], color=cmap(classes[i]), fontsize=10)
    # make legends   
    if legend and not meta is None:
        handles = []
        for cl in classes:
            patch = mpatches.Patch(color=cmap(cl), label=class_names[cl])
            handles.append(patch)
        fig.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
    return fig, ax


def plot_3d(current_z, meta=None, var=None, classes=None, class_ids=None, class_names=None, cmap='plasma', centroids=None, legend=True):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    if meta is None:
        meta = np.zeros((current_z.shape[0]))
        cmap = get_cmap(0)
    else:
        cmap = get_cmap(len(classes))

    current_alpha = 0.06 if (centroids and not meta is None) else 1.0
    current_var = var if not var is None else np.ones(current_z.shape[0])
    # plot
    if current_z.shape[1]==2:
        ax.scatter(current_z[:, 0], current_z[:,1], np.zeros_like(current_z[:,0]), c=cmap(meta), alpha = current_alpha, s=current_var)
    else:
        ax.scatter(current_z[:, 0], current_z[:,1], current_z[:, 2], c=cmap(meta), alpha = current_alpha, s=current_var)
    # make centroids
    if centroids and not meta is None:
        for i, cid in enumerate(class_ids):
            centroid = np.mean(current_z[cid], axis=0)
            ax.scatter(centroid[0], centroid[1], centroid[2], s = 30, c=cmap(classes[i]))
            ax.text(centroid[0], centroid[1], centroid[2], class_names[i], color=cmap(classes[i]), fontsize=10)
    # make legends   
    if legend and not meta is None:
        handles = []
        for cl in classes:
            patch = mpatches.Patch(color=cmap(cl), label=class_names[cl])
            handles.append(patch)
        fig.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
    return fig, ax




##############################
######   confusion matirices

def plot_confusion_matrix(confusion_matrices, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not issubclass(type(confusion_matrices), list):
        confusion_matrices = [confusion_matrices]
    if not issubclass(type(classes), list):
        classes = [classes]

    fig = plt.figure(figsize=(16,8))
    
    for i, cm in enumerate(confusion_matrices):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
            
        print(cm)
        
        ax = fig.add_subplot(1, len(confusion_matrices), i+1)
        cmap = get_cmap(max(list(classes[i].keys())), 'Blues')
        img = ax.imshow(cm, cmap=cmap)
        
        ax.set_title(title)
#        plt.colorbar(img, ax=ax)
        tick_marks = np.arange(len(classes[i]))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels([classes[i][n] for n in range(max(list(classes[i].keys())))], rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels([classes[i][n] for n in range(max(list(classes[i].keys())))])

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        
    return fig


def make_confusion_matrix(false_labels, true_labels, classes):
    if not issubclass(type(false_labels), list):
        false_labels = [false_labels]
    if not issubclass(type(false_labels), list):
        true_labels = [true_labels]
    if not issubclass(type(classes), list):
        classes = [classes]
        
    assert len(false_labels) == len(true_labels) == len(classes)      
    cnf_matrices = []
        
    for i, fl in enumerate(false_labels):
        tl = true_labels[i]
        if fl.ndim == 2:
            fl = fromOneHot(fl)
        if tl.ndim == 2:
            tl = fromOneHot(tl)
        cnf_matrices.append(confusion_matrix(fl, tl))
        
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    fig = plot_confusion_matrix(cnf_matrices, classes=classes, title='Confusion matrix, without normalization')
    return fig
    


