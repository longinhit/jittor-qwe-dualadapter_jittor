import os
import random
import argparse
import yaml
from tqdm import tqdm
from PIL import Image

import jittor as jt
from jittor import nn
from jittor import transform
from jittor import optim
from jittor import lr_scheduler
from jittor.dataset import DataLoader

from datasets.utils import read_image 
from datasets import build_dataset
from datasets.utils import build_data_loader, Datum
import jclip as clip
from utils import *

from CustomCLIP import PositiveAdapter, NegativeAdapter, CustomCLIP

if jt.has_cuda:
    jt.flags.use_cuda = 1 # Enable CUDA in Jittor

def cdist(x1, x2, p):
    x1_norm = (x1 ** 2).sum(dim=1).unsqueeze(1)
    x2_norm = (x2 ** 2).sum(dim=1).unsqueeze(0)
    dist = x1_norm + x2_norm - p * jt.matmul(x1, jt.transpose(x2, 0, 1))
    return jt.sqrt(jt.clamp(dist, min_v=0.0))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', dest='shot', type=int, default=4, help='shots number')
    parser.add_argument('--config', dest='config', default='configs/jittor_dataset.yaml', help='settings of DualAdapter in yaml format')
    parser.add_argument('--test', dest='test', help='test dataset')
    args = parser.parse_args()
    return args
    

def get_test_loader(cfg, clip_model, preprocess):
    test_data_dir = 'data/TestSetA/'
    image_list =[]
    for file_name in os.listdir(test_data_dir):
        impath = os.path.join(test_data_dir, file_name)
        item = Datum(
            impath=impath,
            label=-1,  # 竞赛测试集没有label
            classname='unknown' #  竞赛测试集没有classname
        )
        image_list.append(item)
    test_loader = build_data_loader(data_source=image_list, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    return test_loader,image_list

def extract_image_features(cfg, clip_model, loader):
    features = []
    with jt.no_grad():
        for i, (images, _) in enumerate(tqdm(loader)):
            images = images.to(jt.flags.use_cuda)
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
    features = jt.concat(features)
    print(f"extract_test_feature features size: {features.size()}")
    return features
    
def generate_jittor_result(cfg, clip_model, preprocess, customCLIPmodel): 

    test_loader, image_list = get_test_loader(cfg, clip_model, preprocess) 
    image_features = extract_image_features(cfg, clip_model, test_loader)

    customCLIPmodel.eval()
    print("\n-------- Evaluating on the test set. --------")
    with jt.no_grad():
        output, _, _ = customCLIPmodel(image_features)
        with open('./data/result.txt', 'w') as save_file:
            preds = output.topk(5, 1, True, True)[1]
            for index in range(preds.size(0)):
                image_file = image_list[index].impath
                image_file = image_file[image_file.rfind('/')+1:]
                top5_idx = np.asarray(preds[index].cpu())
                save_file.write(image_file + ' ' + ' '.join(str(idx) for idx in top5_idx) + '\n')


def DualAdapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights_template, clip_weights_cupl, clip_weights_negative, clip_model, train_loader_F, preprocess, classnames):
    
    feat_dim, cate_num = clip_weights_template.shape # cate_num=374
    cache_values = cache_values.reshape(cate_num, -1, cate_num) # cache_values: (1024, 374) <class 'numpy.ndarray'> -> 
    cache_keys = cache_keys.transpose().reshape(cate_num, cfg['shots'], feat_dim).reshape(cate_num, -1, feat_dim)
    
    cache_keys, cache_values = cache_keys.reshape(-1, feat_dim), cache_values.reshape(-1, cate_num)
    print("**** cache_keys shape: {:}. ****\n".format(cache_keys.shape))
    print("**** cache_values shape: {:}. ****\n".format(cache_values.shape))
    customCLIPmodel = CustomCLIP(cfg, clip_weights_template, clip_weights_cupl, clip_weights_negative, clip_model, cache_keys, cache_values, classnames)

    optimizer = jt.optim.AdamW([
            {'params': customCLIPmodel.positive_adapter.parameters(), 'lr': cfg['lr'], 'eps': cfg['eps'], 'weight_decay': 1e-1},
            {'params': customCLIPmodel.negative_adapter.parameters(), 'lr': cfg['lr'] * 5, 'eps': cfg['eps'], 'weight_decay': 1e-1}],
            lr=cfg['lr'],  # 这里提供一个全局的学习率
            eps=cfg['eps'])  

    scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F)) # 余弦退火学习率调度策略
    Loss = SmoothCrossEntropy()

    best_acc, best_epoch = 0.0, 0
    feat_num = cfg['training_feat_num'] # feat_num
    for train_idx in range(cfg['train_epoch']):
        # Train
        customCLIPmodel.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        loss1_list = []
        loss2_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.to(jt.flags.use_cuda), target.to(jt.flags.use_cuda) # float32; int32
            with jt.no_grad():
                image_features = clip_model.encode_image(images) 
                image_features /= image_features.norm(dim=-1, keepdim=True)

            final_logits, new_clip_weights, consistency_loss = customCLIPmodel(image_features) # float16/float16/float32
            loss1 = Loss(final_logits, target)
            loss2 = adaptive_reranking_loss(image_features, new_clip_weights.transpose(), target)
            
            loss = loss1 + loss2 + 8 * consistency_loss # + loss3 #+ loss4

            acc = cls_acc(final_logits, target)
            correct_samples += acc / 100 * len(final_logits)
            all_samples += len(final_logits)
            loss_list.append(loss.item())
            loss1_list.append(loss1.item())
            loss2_list.append(loss2.item())

            optimizer.zero_grad()
            optimizer.backward(loss)

            optimizer.step()
            scheduler.step()
            
        current_lr = optimizer.lr # scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}, CE Loss: {:.4f}, ReRank Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list), sum(loss1_list)/len(loss1_list), sum(loss2_list)/len(loss2_list)))

        # Eval
        customCLIPmodel.eval()
        with jt.no_grad():
            final_logits,_,_ = customCLIPmodel(val_features)
        acc = cls_acc(final_logits, val_labels)
        
        print("**** DualAdapter's test accuracy: {:.2f}. ****".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            save_checkpoint(customCLIPmodel, cfg['cache_dir'] + f"/model-best_acc-{str(cfg['shots'])}-shots.pth", acc)

    print(f"**** DualAdapter's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Evaluating on the test set. --------")
    with jt.no_grad():
        final_logits, _, _ = customCLIPmodel(test_features)
        acc = cls_acc(final_logits, test_labels)
    print("**** DualAdapter's final test accuracy: {:.2f}. ****\n".format(acc)) 

    generate_jittor_result(cfg, clip_model, preprocess, customCLIPmodel)


def adaptive_reranking_loss(
    visual_features: jt.Var,
    class_prototypes: jt.Var,
    labels: jt.Var,
    scale: float = 4.0,
    knn: int = 3,
    **_: jt.Var,
) -> jt.Var:

    N = visual_features.shape[0]
    C = class_prototypes.shape[0]
    knn = min(knn, C)
    
    distances = cdist(visual_features.float(), class_prototypes.float(), p=2) # 重写

    sorted_distances, sorted_indices = jt.sort(
        distances, dim=1, descending=False)
    anchor = (
        ((visual_features - class_prototypes[labels]) ** 2).sum(-1).sqrt().unsqueeze(1)
    )
    sorted_distances = sorted_distances[:, :knn]

    pos_cla_proto = class_prototypes[labels].unsqueeze(1)
    all_cls = class_prototypes[sorted_indices[:, :knn]]
    margins = (1.0 - (all_cls * pos_cla_proto).sum(-1)) / scale

    loss = jt.maximum(
        anchor + margins - sorted_distances,
        jt.zeros(N, knn), # jt.zeros(N, knn).to(visual_features.device)
    )

    return loss.mean()


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg['shots'] = args.shot
    print(cfg['shots'])

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    print(cfg)

    # CLIP
    clip_model, preprocess = clip.load("./jclip/ViT-B-32.pkl") # 改为绝对路径
    clip_model.eval()

    # Prepare dataset
    random.seed(cfg['seed'])
    jt.set_seed(cfg['seed'])

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights_template, clip_weights_cupl, clip_weights_negative = load_text_feature(cfg) # # <class 'numpy.ndarray'> float32
    
    # 将加载的NumPy数组转换为Jittor张量
    clip_weights_template = jt.array(clip_weights_template) # <class 'jittor.jittor_core.Var'>
    clip_weights_cupl = jt.array(clip_weights_cupl)
    clip_weights_negative = jt.array(clip_weights_negative)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = load_few_shot_feature(cfg) 

    cache_keys = jt.array(cache_keys)
    cache_values = jt.array(cache_values)
    
    val_features = jt.array(jt.load(cache_dir + "/" + "val" + "_f.pkl"))
    val_labels = jt.array(jt.load(cache_dir + "/" + "val" + "_l.pkl"))
    test_features = jt.array(jt.load(cache_dir + "/" + "test" + "_f.pkl"))
    test_labels = jt.array( jt.load(cache_dir + "/" + "test" + "_l.pkl"))

    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])
    train_tranform = transform.Compose([
        transform.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=Image.BICUBIC),
        transform.RandomHorizontalFlip(p=0.5),
        transform.ToTensor(),
        transform.ImageNormalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)
    DualAdapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights_template, clip_weights_cupl, clip_weights_negative, clip_model, train_loader_F, preprocess, dataset.classnames)


if __name__ == '__main__':
    main()
