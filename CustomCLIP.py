import jittor as jt
from jittor import init
from jittor import nn, Module
from collections import OrderedDict

import jclip as clip

def generate_pseudo_negative_cache(cfg, cache_keys):
    print('Generating pseudo negative cache...')
    num_shot = cfg['shots']
    num_class = cache_keys.shape[0] // cfg['shots']
    feat_dim = cache_keys.shape[-1]

    # Reshaping the cache keys
    cache_keys = cache_keys.reshape(num_class, num_shot, feat_dim)
    
    # Initializing negative cache keys
    negative_cache_keys = jt.zeros((num_class, num_shot, feat_dim))
    if jt.flags.use_cuda:
        negative_cache_keys = negative_cache_keys.to(jt.flags.use_cuda)
    filtered = 1
    num_negative = num_class - filtered

    # Precompute mean cache keys for each class
    mean_cache_keys = cache_keys.mean(dim=1)
    mean_cache_keys = jt.normalize(mean_cache_keys, dim=1)

    # Compute all cosine similarities in a vectorized manner
    similarity_matrix = mean_cache_keys @ mean_cache_keys.transpose()

    # Get indices of the classes with lowest similarity
    _, negative_indices = jt.topk(similarity_matrix, k=num_negative, largest=False, dim=1)

    # Calculate negative cache keys
    for i in range(num_class):
        selected_cache_keys = cache_keys[negative_indices[i, :], :, :]
        negative_cache_keys[i, :, :] = jt.mean(selected_cache_keys, dim=0)

    # Reshape and normalize
    negative_cache_keys = negative_cache_keys.reshape(-1, feat_dim)
    negative_cache_keys = jt.normalize(negative_cache_keys, dim=1)
    
    return negative_cache_keys


def generate_soft_label(cfg, cache_keys, cache_values, temperature=0.1):
    num_shot = cfg['shots']
    num_class = cache_keys.shape[0] // cfg['shots']
    feat_dim = cache_keys.shape[-1]

    # Reshaping the cache keys and values
    cache_keys = cache_keys.reshape(num_class, num_shot, feat_dim)
    cache_values = cache_values.reshape(num_class, num_shot, num_class)
    
    soft_cache_values = jt.zeros((num_class, num_shot, num_class))

    if jt.flags.use_cuda:
        soft_cache_values = soft_cache_values.to(jt.flags.use_cuda)

    for i in range(num_class):
        keys = cache_keys[i, :, :]
        values = cache_values[i, :, :]
        cos_sim = keys @ keys.transpose()
        sum_sim = cos_sim.sum(dim=1) - 1
        avg_sim = sum_sim / (num_shot - 1)
        confidence = nn.softmax(avg_sim / temperature, dim=0)
        soft_cache_values[i, :, :] = values * confidence.unsqueeze(1) * num_shot
    
    soft_cache_values = soft_cache_values.reshape(-1, num_class)

    return soft_cache_values # soft_cache_values.float16()


class CosineSimilarity(nn.Module):
    def __init__(self, dim=1, eps=1e-7):
        super(CosineSimilarity, self).__init__()
        self.dim = dim
        self.eps = eps

    def execute(self, x1, x2):
        w12 = jt.sum(x1 * x2, dim=self.dim)
        w1 = jt.norm(x1, p=2, dim=self.dim)
        w2 = jt.norm(x2, p=2, dim=self.dim)
        return w12 / (w1 * w2).clamp(min_v=self.eps)


class PositiveAdapter(nn.Module):
    def __init__(self, cfg, feat_dim, cate_num):
        super(PositiveAdapter, self).__init__()
        self.shots = cfg['shots']
        self.feat_dim, self.cate_num = feat_dim, cate_num
        self.feat_num = cfg['training_feat_num']
        
        self.res_template = nn.Parameter(jt.zeros([self.cate_num, cfg['training_feat_num']]).to(jt.flags.use_cuda), requires_grad=True)
        self.res_cupl = nn.Parameter(jt.zeros([self.cate_num, cfg['training_feat_num']]).to(jt.flags.use_cuda), requires_grad=True)
        self.res_keys = nn.Parameter(jt.zeros([self.cate_num, cfg['training_feat_num']]).to(jt.flags.use_cuda), requires_grad=True)

        
    def execute(self, cache_keys, negative_cache_keys, clip_weights_template, clip_weights_cupl, clip_weights_negative, cache_values):
        new_cache_keys = cache_keys.clone()
        new_cache_keys = new_cache_keys.reshape(-1, self.feat_dim)
        new_cache_keys = new_cache_keys + self.res_keys.unsqueeze(1).repeat(1, self.shots, 1).reshape(-1, self.feat_num)
        new_cache_values = cache_values
        
        new_negative_cache_keys = negative_cache_keys.clone()
        new_negative_cache_keys = new_negative_cache_keys.reshape(-1, self.feat_dim)

        res_text_template = self.res_template.transpose()
        res_text_cupl = self.res_cupl.transpose()
        new_clip_weights_template = clip_weights_template.clone()
        new_clip_weights_template = clip_weights_template + res_text_template
        new_clip_weights_cupl = clip_weights_cupl.clone()
        new_clip_weights_cupl = clip_weights_cupl + res_text_cupl
        new_clip_weights_negative = clip_weights_negative.clone()
        
        # Normalize
        new_clip_weights_template = jt.normalize(new_clip_weights_template, dim=0)
        new_clip_weights_cupl = jt.normalize(new_clip_weights_cupl, dim=0)
        new_clip_weights_negative = jt.normalize(new_clip_weights_negative, dim=0)
        new_cache_keys = jt.normalize(new_cache_keys, dim=1)
        new_negative_cache_keys = jt.normalize(new_negative_cache_keys, dim=1)
        
        return new_cache_keys, new_negative_cache_keys, new_clip_weights_template, new_clip_weights_cupl, new_clip_weights_negative, new_cache_values
    

class NegativeAdapter(nn.Module):
    def __init__(self, cfg, feat_dim, cate_num):
        super(NegativeAdapter, self).__init__()
        self.shots = cfg['shots']
        self.feat_dim, self.cate_num = feat_dim, cate_num
        self.feat_num = cfg['training_feat_num']
        
        self.res_negative = nn.Parameter(jt.zeros([self.cate_num, cfg['training_feat_num']]).to(jt.flags.use_cuda), requires_grad=True)
        self.res_keys2 = nn.Parameter(jt.zeros([self.cate_num, cfg['training_feat_num']]).to(jt.flags.use_cuda), requires_grad=True)

        
    def execute(self, cache_keys, negative_cache_keys, clip_weights_template, clip_weights_cupl, clip_weights_negative, cache_values):
        new_cache_keys = cache_keys.clone()
        new_cache_keys = new_cache_keys.reshape(-1, self.feat_dim)
        new_cache_values = cache_values
        
        new_negative_cache_keys = negative_cache_keys.clone()
        new_negative_cache_keys = new_negative_cache_keys.reshape(-1, self.feat_dim)
        new_negative_cache_keys = new_negative_cache_keys + self.res_keys2.unsqueeze(1).repeat(1, self.shots, 1).reshape(-1, self.feat_num)

        res_text_negative = self.res_negative.transpose()
        new_clip_weights_template = clip_weights_template.clone()
        new_clip_weights_cupl = clip_weights_cupl.clone()
        new_clip_weights_negative = clip_weights_negative.clone()
        new_clip_weights_negative = clip_weights_negative + res_text_negative
        # Normalize
        new_clip_weights_template = jt.normalize(new_clip_weights_template, dim=0)
        new_clip_weights_cupl = jt.normalize(new_clip_weights_cupl, dim=0)
        new_clip_weights_negative = jt.normalize(new_clip_weights_negative, dim=0)
        new_cache_keys = jt.normalize(new_cache_keys, dim=1)
        new_negative_cache_keys = jt.normalize(new_negative_cache_keys, dim=1)
        
        return new_cache_keys, new_negative_cache_keys, new_clip_weights_template, new_clip_weights_cupl, new_clip_weights_negative, new_cache_values


class CustomCLIP(nn.Module):
    def __init__(self, cfg, clip_weights_template, clip_weights_cupl, clip_weights_negative, clip_model, cache_keys, cache_values, classnames):
        super().__init__()
        self.beta, self.alpha, self.beta2, self.alpha2, self.lam = cfg['init_beta'], cfg['init_alpha'], cfg['init_beta2'], cfg['init_alpha2'], cfg['init_lambda']
        self.feat_dim, self.cate_num = clip_weights_template.shape
        
        self.clip_weights_template = clip_weights_template 
        self.clip_weights_cupl = clip_weights_cupl
        self.clip_weights_negative = clip_weights_negative 
        self.clip_model = clip_model
        self.cache_keys = cache_keys 
        self.cache_values = cache_values
        self.cfg = cfg
        
        self.cos = CosineSimilarity(dim=1, eps=1e-7)

        vecs = cache_keys.float().clone() # [1496,512,]  vecs[labels == 1].shape = [1500,] other = [4,]
        labels_tuple = jt.argmax(cache_values.float().clone(), dim=1) # [2,1496,] <class 'tuple'> (jt.Var([106 106 106 ... 102 102 102], dtype=int32), jt.Var([1. 1. 1. ... 1. 1. 1.], dtype=float32))
        labels = jt.array(labels_tuple[0]) # Convert the tuple: jt.Var([106 106 106 ... 102 102 102], dtype=int32) to a Jittor tensor [1496,]
        
        # 计算 mus
        mus = jt.concat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(clip_weights_cupl.shape[1])]) # [374,512,]

        # KS Estimator.  
        center_vecs = jt.concat([vecs[labels == i] - jt.unsqueeze(mus[i], 0) for i in range(clip_weights_cupl.shape[1])])
        # 手动计算协方差矩阵，代替cov函数。
        center_vecs_mean = center_vecs.mean(dim=0, keepdims=True)
        centered_center_vecs = center_vecs - center_vecs_mean
        cov_matrix = (centered_center_vecs.transpose(1, 0) @ centered_center_vecs) / (center_vecs.shape[0] - 1)
        # 手动计算矩阵的迹（trace），代替trace函数
        trace_cov = cov_matrix.diag().sum()
        cov_inv = center_vecs.shape[1] * jt.linalg.pinv((center_vecs.shape[0] - 1) * cov_matrix + trace_cov * jt.init.eye(center_vecs.shape[1]).to(jt.flags.use_cuda))

        ps = jt.ones(clip_weights_cupl.shape[1]).to(jt.flags.use_cuda) * 1. / clip_weights_cupl.shape[1]
        self.W = jt.linalg.einsum('nd, dc -> cn', mus, cov_inv)
        self.b = ps.log() - jt.linalg.einsum('nd, dc, nc -> n', mus, cov_inv, mus) / 2

        self.negative_cache_keys = generate_pseudo_negative_cache(cfg, self.cache_keys)
        self.soft_cache_values = generate_soft_label(cfg, cache_keys, self.cache_values)
        
        self.positive_adapter = PositiveAdapter(cfg, self.feat_dim, self.cate_num) # cuda
        self.negative_adapter = NegativeAdapter(cfg, self.feat_dim, self.cate_num) # cuda

   
    def execute(self, image_features):

        positive_adapter_output = self.positive_adapter(self.cache_keys, self.negative_cache_keys, self.clip_weights_template, self.clip_weights_cupl, self.clip_weights_negative, self.cache_values)
        new_cache_keys, new_negative_cache_keys, new_clip_weights_template, new_clip_weights_cupl, new_clip_weights_negative, R_FW = self.negative_adapter(*positive_adapter_output)

        # Positive
        R_fF = image_features @ new_cache_keys.transpose() # float16 [256,1496,] @矩阵乘法 
        Aff = ((-1) * (self.beta - self.beta * (R_fF))).exp() # float32 -> float16
        cache_logits = Aff @ self.soft_cache_values # float16
        new_clip_weights = 0.45 * new_clip_weights_template + 0.55 * new_clip_weights_cupl
        R_fW = 100. * (image_features @ new_clip_weights) 

        # Negative
        R_fF2 = (1 - image_features @ new_negative_cache_keys.transpose())
        Aff2 = ((-1) * (self.beta - self.beta * (R_fF2))).exp() # float32 -> float16
        cache_logits2 = Aff2 @ self.cache_values
        R_fW2 = 100. * (1 - image_features @ new_clip_weights_negative) * 0.15 # to scale
        
        text_distance_template = 1 - self.cos(new_clip_weights, self.clip_weights_template)
        text_distance_cupl = 1 - self.cos(new_clip_weights, self.clip_weights_cupl)
        consistency_loss = (0.45 * text_distance_template + (1 - 0.45) * text_distance_cupl).mean()

        ape_logits = R_fW + cache_logits * self.alpha
        ape_logits2 = R_fW2 + cache_logits2 * self.alpha
        final_logits = self.lam * ape_logits + (1 - self.lam) * ape_logits2

        return final_logits, new_clip_weights, consistency_loss
    
    def updateHyperparameters(self,beta, alpha, beta2, alpha2, lam ):
        self.beta, self.alpha, self.beta2, self.alpha2, self.lam = beta, alpha, beta2, alpha2, lam 