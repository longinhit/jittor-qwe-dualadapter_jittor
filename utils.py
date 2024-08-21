import os
import jclip as clip
import json
import jittor as jt
import jittor.nn as nn
from jittor import Module
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys

import cv2
import glob
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE


def cls_acc(output, target, topk=1):
    pred = output.topk(5, 1, True, True)[1].transpose() # int32 <class 'jittor.jittor_core.Var'> 
    correct = pred.equal(target.view(1, -1).expand_as(pred)) # bool <class 'jittor.jittor_core.Var'>
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) # <class 'float'>
    acc = 100 * acc / target.shape[0]
    return acc


def load_text_feature(cfg):
    save_path = cfg['cache_dir'] + "/text_weights_template.pkl"
    clip_weights_template = jt.load(save_path)
    save_path = cfg['cache_dir'] + "/text_weights_cupl.pkl"
    clip_weights_cupl = jt.load(save_path)
    save_path = cfg['cache_dir'] + "/text_weights_negative_template.pkl"
    clip_weights_negative = jt.load(save_path)
    return clip_weights_template, clip_weights_cupl, clip_weights_negative


def load_few_shot_feature(cfg):
    cache_keys = jt.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pkl")
    cache_values = jt.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pkl")
    return cache_keys, cache_values


def loda_val_test_feature(cfg, split):
    features = jt.load(cfg['cache_dir'] + "/" + split + "_f.pkl")
    labels = jt.load(cfg['cache_dir'] + "/" + split + "_l.pkl")
    return features, labels


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)


def accuracy(shot_logits, cache_values, topk=(1,)):
    target = cache_values.topk(max(topk), 1, True, True)[1].squeeze()
    pred = shot_logits.topk(max(topk), 1, True, True)[1].squeeze()
    idx = (target != pred)
    return idx


def kl_loss(logit1, logit2):
    p = nn.log_softmax(logit1, dim=1)
    q = nn.softmax(logit2, dim=1) + 1e-8
    # print(p)
    # print(q)
    kl_div = nn.KLDivLoss(reduction='none')
    kl = kl_div(p, q).sum(dim=1)
    # print(kl.shape)
    # print(kl)
    return kl.mean()


class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.0):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def execute(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = nn.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * nn.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()
    

def save_checkpoint(model, save_path, max_accuracy):

    mode_save_dict = {k: v for k, v in model.state_dict().items() if not k.startswith('clip_model') } #clip_model模型不需要存储
    
    save_state = {
        'model': mode_save_dict, 
        'max_accuracy': max_accuracy,
        'beta': model.beta, 
        'alpha': model.alpha, 
        'beta2': model.beta2, 
        'alpha2': model.alpha2, 
        'lam': model.lam 
    }
    jt.save(save_state, save_path)  
    print(f"{save_path} saved !!!")


def load_checkpoint(model, model_path):
    print(model_path)
    checkpoint = jt.load(model_path)
    msg = model.load_state_dict(checkpoint['model']) # update
    model.beta, model.alpha, model.beta2, model.alpha2, model.lam = checkpoint['beta'], checkpoint['alpha'], checkpoint['beta2'], checkpoint['alpha2'], checkpoint['lam'] 
    max_accuracy = checkpoint['max_accuracy']
    print(f"=> loaded successfully '{model_path}', max_accuracy :{max_accuracy}")
    del checkpoint
    jt.gc()