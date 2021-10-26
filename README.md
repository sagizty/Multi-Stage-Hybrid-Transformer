![Screen Shot 2021-09-07 at 3 02 37 PM](https://user-images.githubusercontent.com/50575108/132299311-44f49a0f-8199-45be-a0f9-ad5f7fa9a9fd.png)
![Screen Shot 2021-09-07 at 3 02 50 PM](https://user-images.githubusercontent.com/50575108/132299317-e64df48d-6878-4b9e-93a0-1cc3c5ac9f21.png)


# 胰腺癌任务。pancreatic-cancer-diagnosis-tansformer

病理学图像数据，
特点：尺寸大分辨率高，染色风格不同导致底色不同，且缩放倍数不一定，数据稀少
训练和测试都会很好，但是一到新的数据表现会差很多

人工判定的规则明确：形状+分布聚合规律

# CV+细胞学综述

arXiv:2105.11277v1 


# 目前做了Transformer的实验，阶段性结论如下：
1.对学习率敏感，在e-5数量级有最好表现
2.对尺寸敏感，学习率挑对了然后大一点效果反而比尺寸小的时候好（resNet50在1000尺寸上好于224.ViT在384尺寸上好于224.）
3.表现结果基本上与ResNet50类似

# 数据集 721
dataroot = '/data/pancreatic-cancer-project/ZTY_dataset'

数据集划分
*********************************N*************************************
N类按照0.7：0.2：0.1的比例划分完成，一共1852张图片
训练集/data/pancreatic-cancer-project/ZTY_dataset/train/N：1297张
验证集/data/pancreatic-cancer-project/ZTY_dataset/val/N：370张
测试集/data/pancreatic-cancer-project/ZTY_dataset/test/N：185张
*********************************P*************************************
P类按照0.7：0.2：0.1的比例划分完成，一共1240张图片
训练集/data/pancreatic-cancer-project/ZTY_dataset/train/P：869张
验证集/data/pancreatic-cancer-project/ZTY_dataset/val/P：248张
测试集/data/pancreatic-cancer-project/ZTY_dataset/test/P：123张

# 数据集 712
dataroot = '/data/pancreatic-cancer-project/712_dataset'

7:1:2 训练：验证：测试
*********************************N*************************************
N类按照0.7：0.1：0.2的比例划分完成，一共1852张图片
训练集/data/pancreatic-cancer-project/712_dataset/train/N：1297张
验证集/data/pancreatic-cancer-project/712_dataset/val/N：185张
测试集/data/pancreatic-cancer-project/712_dataset/test/N：370张
*********************************P*************************************
P类按照0.7：0.1：0.2的比例划分完成，一共1240张图片
训练集/data/pancreatic-cancer-project/712_dataset/train/P：869张
验证集/data/pancreatic-cancer-project/712_dataset/val/P：123张
测试集/data/pancreatic-cancer-project/712_dataset/test/P：248张


数据预处理在data processing

# CVPR2021 | 视觉 Transformer 的可视化

https://blog.csdn.net/zandaoguang/article/details/114558404



# 涉及的内容

SAM

https://mp.weixin.qq.com/s/1dFGbcVd9ZYNDyykwu7ALg

常用的一阶优化器(如SGD,Adam)只寻求最小化训练损失。它们通常会忽略与泛化相关的高阶信息，如曲率。然而，深度神经网络的损失具有高度非凸性，在评估时容易达到接近0的训练误差，但泛化误差较高，更谈不上在测试集具有不同分布时的鲁棒性。

由于对视觉数据缺乏归纳偏差ViTs和MLPs放大了一阶优化器的这种缺陷，导致过度急剧的损失scene和较差的泛化性能，如前一节所示。假设平滑收敛时的损失scene可以显著提高那些无卷积架构的泛化能力，那么最近提出的锐度感知最小化(SAM)可以很好的避免锐度最小值。

Pytorch版本

https://blog.csdn.net/u011984148/article/details/114957773



DPT 可变形的分patch



相对位置编码


Seesaw loss


# 注意力模块
SimAM 无参数注意力模块



# 应该需要看一下的类似任务

基于深度学习的乳腺转移瘤识别（Deep Learning for Identifying Metastatic Breast Cancer）
https://blog.csdn.net/u014593748/article/details/78200173







# 7月7日 关于胰腺癌模型的一个思路点说明

首先，我们希望模型具有鲁棒性，需要对于不同染色条件下的结果，以及不同人之间的样本都能有稳定表现。
因此，在训练与测试中需要对于不同染色/取样条件进行模拟。
染色颜色与对比度等等之前已经通过pytorch的transform实现了。

今天的实验中我发现一个问题，如果旋转的话可能会有黑边，在cam上是会有影响的。有些判断阳性时cam关注了黑边（但也不是全部）


测试集不采用数据增强，因此，如果在训练集对黑边进行避免（即采用中心裁剪的方式，取中间700尺寸的方形之后resize到384），会导致训练集模型看的实际数据尺寸（缩放倍率）与测试集不同。尽管模型看到的数据尺寸都是384，但是放大倍率不同。（目前新做了实验）

此时，结果会显著下降且波动变大。（推测：不对测试集数据进行修改，模型看到的细胞大小不同，视野大小不同等等。存在一个特征适应的问题）


# idea
这也因此使我反思，我们的模型希望是能够对于一定缩放倍率的图像都能够有范化性。那么它会需要有更好的全局信息处理能力，也因此这可能是transformer的一个优势。

最有可能的保持性能与范化性并且解释性合适的一个idea就是采用融合模型（前几层采用ResNet，后几层采用Transformer）
此外就是之前说的训练逻辑希望能够有更好的鲁棒性。


目标：预训练一个能够在下游病理学任务表现较好的深度神经网络模型，从而应对病理学图像中的很多问题：

问题：
细胞重叠
红细胞等噪声
不同类别之间差异相对CV来说很轻微
受外部条件干扰较大（缩放相同时视野也可能不同，染色条件影响整体域分布）

调研到的解决思路：
多路径实现（logo）+注意力机制辅助特征融合
SFA尺度特征增强，AFF自适应特征融合

结合Transformer提出自己的优化
修改实现多尺度的cnn与transformer结合，利用注意力模块辅助positional encoding，从而实现一个小参数量的logo
一开始先摸索几个baseline，后面再去多优化，一开始主要是先要在这个领域抛出这个概念

可行性上的隐患：
这样的话注意力可视化可能有难度，主要是我经验不足，需要多学习
需要预训练

学习率敏感问题，不同层的特性导致其对学习敏感程度不同，可能需要不同层用不同学习率，或者交替学习率训练

基线已经很好，不知道能不能超过（最大的问题）



实现的想法：
多学习一下如何构建，计划先把vit的注意力可视化与融合模型的修改精通一下（大约1周）同时也顺便调参跑几个sota来做对比

下一步工作的想法：
以及后面我多看看怎么想办法去提高这个数据范化性的训练方法/模型结构。

多看相关文章+多做几个模型的实验。+学一下如何实现Transformer和融合模型的注意力可视化。+SAM训练（争取）
假期我主要是看文章+多做几个模型的实验。+学一下如何实现Transformer和融合模型的注意力可视化。+SAM训练（争取）
此外，我借了一本《细胞生物学》的书，计划假期学习一下，先获得一些基础的知识。

8月初回来后，我计划先具体的做注意力可视化和SAM训练，测试imagenet的重头训练（已经下载了数据集了，但是好像做预训练很有技巧，还需要学习），以及构建几个融合模型。




# 采集数据之后的想法和体验

影响因素：

非常受人工采集的影响，采集时的原则是“尽量”和肉眼看见的差不多

软件白平衡校准设置，白平衡采用白色的区域（尽量不要有颜色）校准

受周围组织/细胞影响，会导致颜色白平衡不好调整。有的时候通过白平衡颜色变了，但是细胞看得清晰了。阳性/阴性都有颜色不同的。

亮度是调的，并且没有定量指标能够看。往往是看着比较暗，白平衡也无法很好的增加对比度，于是就调亮一点

人工遍历区域来找细胞，“合适的”区域（细胞个数太少，细胞与红细胞对比度过低……）

软件缩放比率（0627及之后都是0.18，之前不知道）

因为调焦的原因，同一张照片可能只有部分区域清晰（很少）


临床意义：

比如说有些片染色出来目标几乎没有，对于当场分析就没意义。还有就是踩到有的就要多分析几个，没有的就多检查一下/重复测试几个





# 阶段性方案与结果（仅有ResNet部分采用迁移学习的参数）

在0929 与0527数据集上

hybrid1：ResNet50 接 Encoder （ViT、CoAtNet）

Training complete in 159m 23s
Best epoch idx:  44
Best epoch train Acc: 95.844875
Best epoch val Acc: 94.155844
Best epoch val PR:
negative precision: 94.6524  recall: 95.6757
positive precision: 93.3884  recall: 91.8699

hybrid2：ResNet+4个Decoder，采用每个stage的feature map用focus模块进行编码之后作为cross attention的q和k输入decoder，decoder主体串联采用级联方式，第一级输入为最后一个stage的feature map经分patch编码之后

Training complete in 319m 50s
Best epoch idx:  141
Best epoch train Acc: 97.830102
Best epoch val Acc: 95.454545
Best epoch val PR:
negative precision: 97.2376  recall: 95.1351
positive precision: 92.9134  recall: 95.9350


Epoch:  test 
Loss: 0.1343  Acc: 95.9547
negative precision: 98.0501  recall: 95.1351
negative TP: 352
positive precision: 93.0502  recall: 97.1774
positive TP: 241



# 简单介绍idea  版本： 8月18日 19：46


我们的motivation在于希望不同stage卷积的注意力特征带来的对细节的关注能够帮助全局建模的transformer在不同stage上更好的注意到局部特性。具体将通过focus模块引导decoder的cross attention模块，从而在卷积模块考虑了局部特征并增强泛化性的同时，帮助transformer模块在全局建模中获得优势。

我们注意到，卷积网络在病理学图像的研究中被广泛应用，由于卷积运算具有的inductive bias他们通常具有较好的泛化性并在下游任务的迁移中表现良好。从计算特性上来说，采用卷积神经网络的模型能够较好的关注局部特征，但在全局特征建模上由于其感受野交互的限制需要依托于层级结构，这样的设计使得其在关注全局特征时局部特征的重要性被削弱了。在rose数据中，全局的细胞分布与细胞间相对位置关系十分重要（引用医学内容..）同时癌症与否的确认也与细胞的细致结构紧密相关（引用医学内容…）。因此全局特征的建模能够更好的适配本任务。

新兴的transformer模型在一系列近期cv文章中表现优异。由于其计算采用全局注意力的特征，transformer模型具有全局建模上更加优异的性能，对于不同位置的目标区域能够更好的综合考虑。为了结合两大类网络的优势，一系列计算机视觉领域的文章进行了融合模型的探究，与该领域不同的是，医学图像领域有限且昂贵的数据难以支撑纯transformer模型实现足够具有泛化性的建模。针对病理学图像分类任务相关研究较少，如何利用有限的数据，针对该数据的特征与领域存在的问题，有效的建立融合模型从而在分类中性能优异并具有泛化性十分具有挑战性。

为了获得不同stage的特征，以unet为代表的skip connection策略取得了良好效果，在分割任务中被大量采用，然而分割任务的像素级预测对于快速高效且低维度的分类任务来说过于复杂且冗余。我们希望能够在不同stage获得一个较小的feature来辅助后续transformer注意力建模从而以较低成本达到合理的优化。早期的融合模型可以追溯到首篇vision transformer的文章vit中，而后续的一系列文章均为优化特征交互付出了大量努力（介绍分stage交互，logo融合，平行交互融合的模型）
