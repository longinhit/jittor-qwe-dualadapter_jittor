import os
import random
import argparse
import yaml
import numpy as np
from tqdm import tqdm

import jittor as jt
from jittor import nn
from jittor import transform
from jittor.dataset import DataLoader

from datasets import build_dataset
from datasets.utils import build_data_loader
import jclip as clip
from utils import *

from PIL import Image

if jt.has_cuda:
    jt.flags.use_cuda = 1 # Enable CUDA in Jittor

def extract_few_shot_feature(cfg, clip_model, train_loader_cache):

    cache_keys = []
    cache_values = []
    with jt.no_grad():
        # Data augmentation for the cache model
        for augment_idx in range(cfg['augment_epoch']):
            train_features = []
            print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
            for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                images = images.to(jt.flags.use_cuda)
                image_features = clip_model.encode_image(images)
                train_features.append(image_features)
                if augment_idx == 0:
                    target = target.to(jt.flags.use_cuda)
                    cache_values.append(target)
            cache_keys.append(jt.concat(train_features, dim=0).unsqueeze(0))

    cache_keys = jt.concat(cache_keys, dim=0).mean(dim=0)
    cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0)
    cache_values = nn.one_hot(jt.concat(cache_values, dim=0)) # <class 'jittor.jittor_core.Var'>
    print(f"extract_few_shot_feature, cache_values:{cache_values.shape} cache_keys: {cache_keys.shape}")
    jt.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pkl")
    jt.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pkl")
    return


# For positive prompts and template
def extract_val_test_feature(cfg, split, clip_model, loader):
    features, labels = [], []
    with jt.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.to(jt.flags.use_cuda), target.to(jt.flags.use_cuda)
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)
    features, labels = jt.concat(features), jt.concat(labels)
    jt.save(features, cfg['cache_dir'] + "/" + split + "_f.pkl")
    jt.save(labels, cfg['cache_dir'] + "/" + split + "_l.pkl")
    return


# For positive prompts
def extract_text_feature(cfg, classnames, prompt_path, clip_model, template):
    f = open(prompt_path)
    prompts = json.load(f)
    with jt.no_grad():
        clip_weights = []
        for i, classname in enumerate(classnames):
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            
            # positive template
            texts = [t.format(classname) for t in template]
        
            texts_token = clip.tokenize(texts, truncate=True).to(jt.flags.use_cuda)
            class_embeddings = clip_model.encode_text(texts_token)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
            
        clip_weights = jt.stack(clip_weights, dim=1).to(jt.flags.use_cuda)
    jt.save(clip_weights, cfg['cache_dir'] + "/text_weights_template.pkl")
    text_weights_template = clip_weights
    with jt.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            
            # positive prompts 
            texts = prompts[classname]
        
            texts_token = clip.tokenize(texts, truncate=True).to(jt.flags.use_cuda)
            class_embeddings = clip_model.encode_text(texts_token)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = jt.stack(clip_weights, dim=1).to(jt.flags.use_cuda)
        print(clip_weights.shape)
    jt.save(clip_weights, cfg['cache_dir'] + "/text_weights_cupl.pkl")
    return


# For negative prompts
def extract_text_feature2(cfg, classnames, prompt_path, clip_model, template):
    f = open(prompt_path)
    prompts = json.load(f)
    with jt.no_grad():
        clip_weights = []
        clip_weights_all = []
        for i, classname in enumerate(classnames):
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            
            # negative template
            texts = [t.format(classname) for t in template]
        
            texts_token = clip.tokenize(texts, truncate=True).to(jt.flags.use_cuda)
            class_embeddings = clip_model.encode_text(texts_token)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
            
        clip_weights = jt.stack(clip_weights, dim=1).to(jt.flags.use_cuda)
    jt.save(clip_weights, cfg['cache_dir'] + "/text_weights_negative_template.pkl")
    return

    
if __name__ == '__main__':
    
    clip_model, preprocess = clip.load("./jclip/ViT-B-32.pkl") 
    clip_model.eval()

    dataset = 'jittor_dataset'
    k_shot = [4]
    cfg = yaml.load(open('configs/{}.yaml'.format(dataset), 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    for k in k_shot:
        
        random.seed(1)
        jt.set_seed(1)
        
        cfg['shots'] = k
        dataset = build_dataset(dataset, cfg['root_path'], k)

        val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

        train_tranform = transform.Compose([
            transform.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=Image.BICUBIC),
            transform.RandomHorizontalFlip(p=0.5),
            transform.ToTensor(),
            transform.ImageNormalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
        train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)

        # Construct the cache model by few-shot training set
        print("\nConstructing cache model by few-shot visual features and labels.")
        extract_few_shot_feature(cfg, clip_model, train_loader_cache)

        # Extract val/test features
        print("\nLoading visual features and labels from val and test set.")
        extract_val_test_feature(cfg, "val", clip_model, val_loader)
        extract_val_test_feature(cfg, "test", clip_model, test_loader)
                
        extract_text_feature(cfg, dataset.classnames, dataset.cupl_path, clip_model, dataset.template)
        extract_text_feature2(cfg, dataset.classnames, dataset.cupl_path, clip_model, dataset.negative_template)