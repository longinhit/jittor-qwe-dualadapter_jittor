| 第四届计图挑战赛开源项目

# Jittor 开放域少样本视觉分类赛题
| 赛题一：开放域少样本视觉分类赛题

结果排名(2024年8月01日)

![结果排名(2024年7月19日)](https://s3.bmp.ovh/imgs/2024/08/01/1094b154d74aa2c5.png)

## 简介

本项目基于CLIP预训练模型。

通过积极推理，学习样本是什么，同时通过消极推理，学习样本不是什么。若仅通过积极推理，难以区分相似类别，但是合并消极推理得出的间接结论，就可以消除部分不正确类别。在一般分类任务中，这种双学习结合的方法，正选择和负排除相辅相成，有效地提高了整体分类精度。

主要方法框架如图：
![方法框架](https://s3.bmp.ovh/imgs/2024/07/19/1193d59b3bad34cb.png)
![方法框架](https://s3.bmp.ovh/imgs/2024/07/19/85ab28a40a1ec699.png)

同时，模型中通过增加概率学方法，进一步提升分类精度。采用高斯判别分析（GDA）算法，应用于CLIP的下游分类任务。GDA假设不同类别的特征遵循具有相同协方差的高斯分布，通过贝叶斯公式计算分类概率。
## 安装 

本项目可在 1 张 A800 上运行，训练时间约为 1 小时。

#### 运行环境
- ubuntu 20.04 LTS
- python >= 3.9
- jittor >= 1.3.0

#### 安装依赖
执行以下命令安装 python 依赖
```
pip install -r requirements.txt
```

#### 预训练模型
预训练模型Vit-B-32.pkl，可从openAI网站下载(https://huggingface.co/immich-app/ViT-B-32__openai)，下载后用下面的代码将权重转换成jittor能够使用的权重，命名为ViT-B-32.pkl放在jclip文件夹下。

```python
import torch
import jittor as jt
clip = torch.load('ViT-B-32.pt').state_dict()

for k in clip.keys():
    clip[k] = clip[k].float().cpu()
jt.save(clip, 'ViT-B-32.pkl')
```

## 数据预处理

将数据下载解压到 `<root>/data` 下。 数据目录结构如下

```
data/
    TestSetA
        image_0001.jpg
        image_0002.jpg
        ...
    TrainSet
        Animal
            Bear
                1.jpg
                2.jpg
                ...
            Bee
                ...
        Caltech-101
            ...
        Food-101
            ...
        Thu-dog
            ...
```


## 训练&验证

先提取数据特征：
```
python extract_features.py
```

之后进行训练
```
python main.py
```
main文件中合并了训练和测试的全部内容


最后结果保存在data/result.txt

## 致谢

此项目基于论文 *Negative Yields Positive: Unified Dual-Path Adapter for Vision-Language Models* 实现。

<!-- ## 注意事项

点击项目的“设置”，在Description一栏中添加项目描述，需要包含“jittor”字样。同时在Topics中需要添加jittor。

![image-20220419164035639](https://s3.bmp.ovh/imgs/2022/04/19/6a3aa627eab5f159.png) -->