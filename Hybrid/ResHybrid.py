"""
融合模型构建    版本： 10月07日 14：10

张天翊


ResNet结构
stem模块实现数据标准化输入
4个stage进行feature extraction。每个stage采用了Bottleneck思路构建的Conv Block与Identity Block
此外还有对接的mlp结构实现分类任务。（若只是做feature extraction，可以后面换成其他的对应的需求）


Stage内的block结构：
Conv Block：输入和输出的维度（通道数和size）是不一样的，所以不能连续串联，它的作用是改变网络的维度；
Identity Block：输入维度和输出维度（通道数和size）相同，可以串联，用于加深网络，从而提高表现。

他们都是由通用的Bottleneck block来构造
Bottleneck block由卷积路线+残差路线2个路线构成，数据的通道数由inplane变为midplane，最后再变为outplane。
卷积路线提取不同深度的特征从而实现全局感知。残差路线本质是在Bottleneck block内部的conv之间实现跳接从而减小深度网络的过拟合


我的理解：
相当于说conv block在训练的时候起的作用会大一点，迈的步子大一点，identity block走的步子小，但是走的小心谨慎
非常有意思你看几种resnet的结构你就可以看到，加深加的是identity block，conv block相当于是起每个阶段的带头转换的功能

特征图：
# input = 3, 384, 384
torch.Size([1, 256, 96, 96])
torch.Size([1, 512, 48, 48])
torch.Size([1, 1024, 24, 24])
torch.Size([1, 2048, 12, 12])
torch.Size([1, 1000])


# input = 3, 224, 224
torch.Size([1, 256, 56, 56])
torch.Size([1, 512, 28, 28])
torch.Size([1, 1024, 14, 14])
torch.Size([1, 2048, 7, 7])
torch.Size([1, 1000])
参考：

讲解
https://www.bilibili.com/video/BV1T7411T7wa?from=search&seid=2163932670658847290
https://www.bilibili.com/video/BV14E411H7Uw/?spm_id_from=333.788.recommend_more_video.-1

https://zhuanlan.zhihu.com/p/353235794

参考代码
https://note.youdao.com/ynoteshare1/index.html?id=5a7dbe1a71713c317062ddeedd97d98e&type=note


"""
import torch
from torch import nn
from functools import partial
from torchsummary import summary
import os
from Hybrid import Transformer_blocks


# ResNet最核心的模块构建器
class Bottleneck_block_constructor(nn.Module):
    """
    Bottleneck Block的各个plane值：
    inplane：输出block的之前的通道数
    midplane：在block中间处理的时候的通道数（这个值是输出维度的1/4）
    outplane = midplane*self.extention：输出的通道维度数

    stride: 本步骤（conv/identity）打算进行处理的步长设置

    downsample：若为conv block，传入将通道数进行变化的conv层，把通道数由inplane卷为outplane
    identity block则没有这个问题，因为该模块的 inplane=outplane
    """

    # 每个stage中维度拓展的倍数：outplane相对midplane的放大倍数
    extention = 4

    # 定义初始化的网络和参数
    def __init__(self, inplane, midplane, stride, downsample=None):
        super(Bottleneck_block_constructor, self).__init__()
        # 计算输出通道维度
        outplane = midplane * self.extention

        # 只在这里操作步长，其余卷积目标是小感受区域的信息
        self.conv1 = nn.Conv2d(inplane, midplane, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(midplane)

        self.conv2 = nn.Conv2d(midplane, midplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(midplane)

        self.conv3 = nn.Conv2d(midplane, outplane, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(midplane * self.extention)

        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        # 卷积操作forward pass，标准的卷，标，激过程（cbr）
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))

        # 残差信息是否直连（如果时Identity block就是直连；如果是Conv Block就需要对参差边进行卷积，改变通道数和size使得它和outplane一致）
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            # 参差数据直接传过来
            residual = x

        # 此时通道数一致了，将参差部分和卷积部分相加。
        out += residual

        # 最后再进行激活
        out = self.relu(out)

        # 我的理解：其实绝大部分被激活的信息来自residential路线，这样也因此学习得比较慢可是不容易过拟合?

        return out


# 网络构建器
class Hybrid_backbone_4(nn.Module):

    # 初始化网络结构和参数
    def __init__(self, block_constructor, bottleneck_channels_setting=None, identity_layers_setting=None,
                 stage_stride_setting=None, fc_num_classes=None, feature_idx=None):

        if bottleneck_channels_setting is None:
            bottleneck_channels_setting = [64, 128, 256, 512]
        if identity_layers_setting is None:
            identity_layers_setting = [3, 4, 6, 3]
        if stage_stride_setting is None:
            stage_stride_setting = [1, 2, 2, 2]

        # self.inplane为当前的feature map(stem的网络层)的通道数
        self.inplane = 64
        self.fc_num_classes = fc_num_classes
        self.feature_idx = feature_idx

        super(Hybrid_backbone_4, self).__init__()  # 这个递归写法是为了拿到自己这个class里面的其他函数进来

        # 关于模块结构组的构建器
        self.block_constructor = block_constructor  # Bottleneck_block_constructor
        # 每个stage中Bottleneck Block的中间维度，输入维度取决于上一层
        self.bcs = bottleneck_channels_setting  # [64, 128, 256, 512]
        # 每个stage的conv block后跟着的identity block个数
        self.ils = identity_layers_setting  # [3, 4, 6, 3]
        # 每个stage的conv block的步长设置
        self.sss = stage_stride_setting  # [1, 2, 2, 2]

        # stem的网络层
        # 将RGB图片的通道数卷为inplane
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        # ResNet写法：构建每个stage，ResNet写死是4个stage，名字是layer1-4不然无法传入权重
        self.layer1 = self.make_stage_layer(self.block_constructor, self.bcs[0], self.ils[0], self.sss[0])
        self.layer2 = self.make_stage_layer(self.block_constructor, self.bcs[1], self.ils[1], self.sss[1])
        self.layer3 = self.make_stage_layer(self.block_constructor, self.bcs[2], self.ils[2], self.sss[2])
        self.layer4 = self.make_stage_layer(self.block_constructor, self.bcs[3], self.ils[3], self.sss[3])

        # 后续的网络
        if self.fc_num_classes is not None:
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(512 * self.block_constructor.extention, fc_num_classes)

    def forward(self, x):
        # 定义构建的模型中的数据传递方法

        # stem部分:conv+bn+relu+maxpool
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        stem_out = self.maxpool(out)

        # Resnet block实现4个stage
        stage1_out = self.layer1(stem_out)
        stage2_out = self.layer2(stage1_out)
        stage3_out = self.layer3(stage2_out)
        stage4_out = self.layer4(stage3_out)

        if self.fc_num_classes is not None:
            # 对接mlp来做分类
            fc_out = self.avgpool(stage4_out)
            fc_out = torch.flatten(fc_out, 1)
            fc_out = self.fc(fc_out)

        if self.feature_idx == 'stages':
            if self.fc_num_classes is not None:
                return stage1_out, stage2_out, stage3_out, stage4_out, fc_out
            else:
                return stage1_out, stage2_out, stage3_out, stage4_out
        elif self.feature_idx == 'features':
            if self.fc_num_classes is not None:
                return stem_out, stage1_out, stage2_out, stage3_out, stage4_out, fc_out
            else:
                return stem_out, stage1_out, stage2_out, stage3_out, stage4_out
        else:  # self.feature_idx is None
            if self.fc_num_classes is not None:
                return fc_out
            else:
                return stage4_out

    def make_stage_layer(self, block_constractor, midplane, block_num, stride=1):
        """
        block:模块构建器
        midplane：每个模块中间运算的维度，一般等于输出维度/4
        block_num：重复次数
        stride：Conv Block的步长
        """

        block_list = []

        # 先计算要不要加downsample模块
        outplane = midplane * block_constractor.extention  # extention存储在block_constractor里面

        if stride != 1 or self.inplane != outplane:
            # 若步长变了，则需要残差也重新采样。 若输入输出通道不同，残差信息也需要进行对应尺寸变化的卷积
            downsample = nn.Sequential(
                nn.Conv2d(self.inplane, outplane, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(midplane * block_constractor.extention)
            )  # 注意这里不需要激活，因为我们要保留原始残差信息。后续与conv信息叠加后再激活
        else:
            downsample = None

        # 每个stage都是1个改变采样的 Conv Block 加多个加深网络的 Identity Block 组成的

        # Conv Block
        conv_block = block_constractor(self.inplane, midplane, stride=stride, downsample=downsample)
        block_list.append(conv_block)

        # 更新网络下一步stage的输入通道要求（同时也是内部Identity Block的输入通道要求）
        self.inplane = outplane

        # Identity Block
        for i in range(1, block_num):
            block_list.append(block_constractor(self.inplane, midplane, stride=1, downsample=None))

        return nn.Sequential(*block_list)  # pytorch对模块进行堆叠组装后返回


"""
测试
resnet50 = Hybrid_backbone_3(block_constructor=Bottleneck_block_constructor,
                           bottleneck_channels_setting=[64, 128, 256, 512],
                           identity_layers_setting=[3, 4, 6, 3],
                           stage_stride_setting=[1, 2, 2, 2],
                           fc_num_classes=1000,
                           feature_idx='stages')

x = torch.randn(1, 3, 224, 224)
stage1_out, stage2_out, stage3_out, stage4_out, out = resnet50(x)
print(stage1_out.shape)
print(stage2_out.shape)
print(stage3_out.shape)
print(stage4_out.shape)
print(out.shape)


# 224
torch.Size([1, 256, 56, 56])
torch.Size([1, 512, 28, 28])
torch.Size([1, 1024, 14, 14])
torch.Size([1, 2048, 7, 7])
torch.Size([1, 1000])

# 384
torch.Size([1, 256, 96, 96])
torch.Size([1, 512, 48, 48])
torch.Size([1, 1024, 24, 24])
torch.Size([1, 2048, 12, 12])
torch.Size([1, 1000])
"""


# 网络构建器_3stages
class Hybrid_backbone_3(nn.Module):

    # 初始化网络结构和参数
    def __init__(self, block_constructor, bottleneck_channels_setting=None, identity_layers_setting=None,
                 stage_stride_setting=None, fc_num_classes=None, feature_idx=None):

        if bottleneck_channels_setting is None:
            bottleneck_channels_setting = [64, 128, 256]
        if identity_layers_setting is None:
            identity_layers_setting = [3, 4, 6]
        if stage_stride_setting is None:
            stage_stride_setting = [1, 2, 2]

        # self.inplane为当前的feature map(stem的网络层)的通道数
        self.inplane = 64
        self.fc_num_classes = fc_num_classes
        self.feature_idx = feature_idx

        super(Hybrid_backbone_3, self).__init__()  # 这个递归写法是为了拿到自己这个class里面的其他函数进来

        # 关于模块结构组的构建器
        self.block_constructor = block_constructor  # Bottleneck_block_constructor
        # 每个stage中Bottleneck Block的中间维度，输入维度取决于上一层
        self.bcs = bottleneck_channels_setting  # [64, 128, 256]
        # 每个stage的conv block后跟着的identity block个数
        self.ils = identity_layers_setting  # [3, 4, 6]
        # 每个stage的conv block的步长设置
        self.sss = stage_stride_setting  # [1, 2, 2]

        # stem的网络层
        # 将RGB图片的通道数卷为inplane
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        # ResNet写法：构建每个stage，ResNet写死是4个stage，名字是layer1-4不然无法传入权重
        self.layer1 = self.make_stage_layer(self.block_constructor, self.bcs[0], self.ils[0], self.sss[0])
        self.layer2 = self.make_stage_layer(self.block_constructor, self.bcs[1], self.ils[1], self.sss[1])
        self.layer3 = self.make_stage_layer(self.block_constructor, self.bcs[2], self.ils[2], self.sss[2])

        # 后续的网络
        if self.fc_num_classes is not None:
            self.avgpool = nn.AvgPool2d(24)  # 224-14  384-24
            self.fc = nn.Linear(self.bcs[-1] * self.block_constructor.extention, fc_num_classes)

    def forward(self, x):
        # 定义构建的模型中的数据传递方法

        # stem部分:conv+bn+relu+maxpool
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        stem_out = self.maxpool(out)

        # Resnet block实现4个stage
        stage1_out = self.layer1(stem_out)
        stage2_out = self.layer2(stage1_out)
        stage3_out = self.layer3(stage2_out)

        if self.fc_num_classes is not None:
            # 对接mlp来做分类
            fc_out = self.avgpool(stage3_out)
            fc_out = torch.flatten(fc_out, 1)
            fc_out = self.fc(fc_out)

        if self.feature_idx == 'stages':
            if self.fc_num_classes is not None:
                return stage1_out, stage2_out, stage3_out, fc_out
            else:
                return stage1_out, stage2_out, stage3_out
        elif self.feature_idx == 'features':
            if self.fc_num_classes is not None:
                return stem_out, stage1_out, stage2_out, stage3_out, fc_out
            else:
                return stem_out, stage1_out, stage2_out, stage3_out
        else:  # self.feature_idx is None
            if self.fc_num_classes is not None:
                return fc_out
            else:
                return stage3_out

    def make_stage_layer(self, block_constractor, midplane, block_num, stride=1):
        """
        block:模块构建器
        midplane：每个模块中间运算的维度，一般等于输出维度/4
        block_num：重复次数
        stride：Conv Block的步长
        """

        block_list = []

        # 先计算要不要加downsample模块
        outplane = midplane * block_constractor.extention  # extention存储在block_constractor里面

        if stride != 1 or self.inplane != outplane:
            # 若步长变了，则需要残差也重新采样。 若输入输出通道不同，残差信息也需要进行对应尺寸变化的卷积
            downsample = nn.Sequential(
                nn.Conv2d(self.inplane, outplane, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(midplane * block_constractor.extention)
            )  # 注意这里不需要激活，因为我们要保留原始残差信息。后续与conv信息叠加后再激活
        else:
            downsample = None

        # 每个stage都是1个改变采样的 Conv Block 加多个加深网络的 Identity Block 组成的

        # Conv Block
        conv_block = block_constractor(self.inplane, midplane, stride=stride, downsample=downsample)
        block_list.append(conv_block)

        # 更新网络下一步stage的输入通道要求（同时也是内部Identity Block的输入通道要求）
        self.inplane = outplane

        # Identity Block
        for i in range(1, block_num):
            block_list.append(block_constractor(self.inplane, midplane, stride=1, downsample=None))

        return nn.Sequential(*block_list)  # pytorch对模块进行堆叠组装后返回


"""
测试
resnet50 = Hybrid_backbone_3(block_constructor=Bottleneck_block_constructor,
                             bottleneck_channels_setting=[64, 128, 256, 512],
                             identity_layers_setting=[3, 4, 6, 3],
                             stage_stride_setting=[1, 2, 2, 2],
                             fc_num_classes=1000,
                             feature_idx='stages')

x = torch.randn(1, 3, 224, 224)
stage1_out, stage2_out, stage3_out, stage4_out, out = resnet50(x)
print(stage1_out.shape)
print(stage2_out.shape)
print(stage3_out.shape)
print(stage4_out.shape)
print(out.shape)


# 224
torch.Size([1, 256, 56, 56])
torch.Size([1, 512, 28, 28])
torch.Size([1, 1024, 14, 14])
torch.Size([1, 1000])

# 384
torch.Size([1, 256, 96, 96])
torch.Size([1, 512, 48, 48])
torch.Size([1, 1024, 24, 24])
torch.Size([1, 1000])
"""


def Hybrid_a(backbone, img_size=224, patch_size=1, in_chans=3, num_classes=1000, embed_dim=768, depth=8,
             num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, drop_rate=0., attn_drop_rate=0.,
             drop_path_rate=0., norm_layer=None, act_layer=None):
    embed_layer = partial(Transformer_blocks.Hybrid_feature_map_Embed, backbone=backbone)

    Hybrid_model = Transformer_blocks.VisionTransformer(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                                                        num_heads, mlp_ratio, qkv_bias, representation_size,
                                                        drop_rate, attn_drop_rate, drop_path_rate, embed_layer,
                                                        norm_layer, act_layer)

    return Hybrid_model


def create_model(model_idx, edge_size, pretrained=True, num_classes=2, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., use_cls_token=True, use_pos_embedding=True, use_att_module='SimAM'):
    # use_att_module in ['SimAM', 'CBAM', 'SE']  different attention module we applied in the ablation study

    if pretrained:
        from torchvision import models
        backbone_weights = models.resnet50(pretrained=True).state_dict()
        # True 是预训练好的Resnet50模型，False是随机初始化参数的模型
    else:
        backbone_weights = None

    if model_idx[0:11] == 'Hybrid1_224' and edge_size == 224:  # ablation study： no focus depth=8 edge_size == 224
        backbone = Hybrid_backbone_4(block_constructor=Bottleneck_block_constructor,
                                     bottleneck_channels_setting=[64, 128, 256, 512],
                                     identity_layers_setting=[3, 4, 6, 3],
                                     stage_stride_setting=[1, 2, 2, 2],
                                     fc_num_classes=None,
                                     feature_idx=None)

        if pretrained:
            try:
                backbone.load_state_dict(backbone_weights, False)
            except:
                print("backbone not loaded")
            else:
                print("backbone loaded")

        model = Hybrid_a(backbone, img_size=edge_size, patch_size=1, in_chans=3, num_classes=num_classes, embed_dim=768,
                         depth=8, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                         norm_layer=None, act_layer=None)

    elif model_idx[0:11] == 'Hybrid1_384' and edge_size == 384:  # ablation study： no focus depth=8 edge_size == 384
        backbone = Hybrid_backbone_4(block_constructor=Bottleneck_block_constructor,
                                     bottleneck_channels_setting=[64, 128, 256, 512],
                                     identity_layers_setting=[3, 4, 6, 3],
                                     stage_stride_setting=[1, 2, 2, 2],
                                     fc_num_classes=None,
                                     feature_idx=None)

        if pretrained:
            try:
                backbone.load_state_dict(backbone_weights, False)
            except:
                print("backbone not loaded")
            else:
                print("backbone loaded")

        model = Hybrid_a(backbone, img_size=edge_size, patch_size=1, in_chans=3, num_classes=num_classes, embed_dim=768,
                         depth=8, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                         norm_layer=None, act_layer=None)

    elif model_idx[0:11] == 'Hybrid2_224' and edge_size == 224:  # Proposed model ablation study： edge_size==224
        backbone = Hybrid_backbone_4(block_constructor=Bottleneck_block_constructor,
                                     bottleneck_channels_setting=[64, 128, 256, 512],
                                     identity_layers_setting=[3, 4, 6, 3],
                                     stage_stride_setting=[1, 2, 2, 2],
                                     fc_num_classes=None,
                                     feature_idx='stages')
        if pretrained:
            try:
                backbone.load_state_dict(backbone_weights, False)
            except:
                print("backbone not loaded")
            else:
                print("backbone loaded")

        model = Transformer_blocks.Stage_wise_hybrid_Transformer(backbone, num_classes=num_classes,
                                                                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                                 drop_path_rate=drop_path_rate,
                                                                 use_cls_token=use_cls_token,
                                                                 use_pos_embedding=use_pos_embedding,
                                                                 use_att_module=use_att_module,
                                                                 stage_size=(56, 28, 14, 7),
                                                                 stage_dim=[256, 512, 1024, 2048])

    elif model_idx[0:11] == 'Hybrid2_384' and edge_size == 384:  # Proposed model 384 !!!
        backbone = Hybrid_backbone_4(block_constructor=Bottleneck_block_constructor,
                                     bottleneck_channels_setting=[64, 128, 256, 512],
                                     identity_layers_setting=[3, 4, 6, 3],
                                     stage_stride_setting=[1, 2, 2, 2],
                                     fc_num_classes=None,
                                     feature_idx='stages')
        if pretrained:
            try:
                backbone.load_state_dict(backbone_weights, False)
            except:
                print("backbone not loaded")
            else:
                print("backbone loaded")

        model = Transformer_blocks.Stage_wise_hybrid_Transformer(backbone, num_classes=num_classes,
                                                                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                                 drop_path_rate=drop_path_rate,
                                                                 use_cls_token=use_cls_token,
                                                                 use_pos_embedding=use_pos_embedding,
                                                                 use_att_module=use_att_module,
                                                                 stage_size=(96, 48, 24, 12),
                                                                 stage_dim=[256, 512, 1024, 2048])

    elif model_idx[0:11] == 'Hybrid3_224' and edge_size == 224:  # Proposed model ablation study： edge_size==224
        backbone = Hybrid_backbone_3(block_constructor=Bottleneck_block_constructor,
                                     bottleneck_channels_setting=[64, 128, 256],
                                     identity_layers_setting=[3, 4, 6],
                                     stage_stride_setting=[1, 2, 2],
                                     fc_num_classes=None,
                                     feature_idx='stages')
        if pretrained:
            try:
                backbone.load_state_dict(backbone_weights, False)
            except:
                print("backbone not loaded")
            else:
                print("backbone loaded")

        model = Transformer_blocks.Stage_wise_hybrid_Transformer(backbone, num_classes=num_classes,
                                                                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                                 drop_path_rate=drop_path_rate,
                                                                 use_cls_token=use_cls_token,
                                                                 use_pos_embedding=use_pos_embedding,
                                                                 use_att_module=use_att_module,
                                                                 stage_size=(56, 28, 14),
                                                                 stage_dim=[256, 512, 1024])

    elif model_idx[0:11] == 'Hybrid3_384' and edge_size == 384:  # Proposed model 384 !!!
        backbone = Hybrid_backbone_3(block_constructor=Bottleneck_block_constructor,
                                     bottleneck_channels_setting=[64, 128, 256],
                                     identity_layers_setting=[3, 4, 6],
                                     stage_stride_setting=[1, 2, 2],
                                     fc_num_classes=None,
                                     feature_idx='stages')
        if pretrained:
            try:
                backbone.load_state_dict(backbone_weights, False)
            except:
                print("backbone not loaded")
            else:
                print("backbone loaded")

        model = Transformer_blocks.Stage_wise_hybrid_Transformer(backbone, num_classes=num_classes,
                                                                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                                 drop_path_rate=drop_path_rate,
                                                                 use_cls_token=use_cls_token,
                                                                 use_pos_embedding=use_pos_embedding,
                                                                 use_att_module=use_att_module,
                                                                 stage_size=(96, 48, 24),
                                                                 stage_dim=[256, 512, 1024])

    else:
        print('not a valid hybrid model')
        return -1

    return model


'''
    elif model_idx[0:11] == 'Hybridx_224' and edge_size == 224:  # TODO next version model: Parallel_hybrid_Transformer
        backbone = Hybrid_backbone(block_constructor=Bottleneck_block_constructor,
                                   bottleneck_channels_setting=[64, 128, 256, 512],
                                   identity_layers_setting=[3, 4, 6, 3],
                                   stage_stride_setting=[1, 2, 2, 2],
                                   fc_num_classes=None,
                                   feature_idx='features')
        if pretrained:
            try:
                backbone.load_state_dict(backbone_weights, False)
            except:
                print("backbone not loaded")
            else:
                print("backbone loaded")

        model = Transformer_blocks.Parallel_hybrid_Transformer(backbone, num_classes=num_classes, use_cls_token=1,
                                                               stage_size=(56, 28, 14, 7),
                                                               stage_dim=[256, 512, 1024, 2048])

    elif model_idx[0:11] == 'Hybridx_384' and edge_size == 384:
        backbone = Hybrid_backbone(block_constructor=Bottleneck_block_constructor,
                                   bottleneck_channels_setting=[64, 128, 256, 512],
                                   identity_layers_setting=[3, 4, 6, 3],
                                   stage_stride_setting=[1, 2, 2, 2],
                                   fc_num_classes=None,
                                   feature_idx='features')
        if pretrained:
            try:
                backbone.load_state_dict(backbone_weights, False)
            except:
                print("backbone not loaded")
            else:
                print("backbone loaded")

        model = Transformer_blocks.Parallel_hybrid_Transformer(backbone, num_classes=num_classes, use_cls_token=1,
                                                               stage_size=(96, 48, 24, 12),
                                                               stage_dim=[256, 512, 1024, 2048])
        # TODO next version model: Parallel_hybrid_Transformer
'''

# ResNet 迁移学习 后续可以自己搭建其他网络，然后从这个倒入了参数的模型中拆层过去

# backbone.load_state_dict(torch.load(r'C:\Users\admin\Desktop\ResNet50_224_weights.pth'), False)
"""
# Hybrid1测试
backbone = Hybrid_backbone_4(block_constructor=Bottleneck_block_constructor,
                               bottleneck_channels_setting=[64, 128, 256, 512],
                               identity_layers_setting=[3, 4, 6, 3],
                               stage_stride_setting=[1, 2, 2, 2],
                               fc_num_classes=None,
                               feature_idx=None)

model = Hybrid(backbone, img_size=224, patch_size=1, in_chans=3, num_classes=1000, embed_dim=768,
                       depth=6, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, drop_rate=0.,
                       attn_drop_rate=0., drop_path_rate=0., norm_layer=None, act_layer=None)

x = torch.randn(7, 3, 224, 224)
x = model(x)
print(x.shape)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只让程序看到物理卡号为card_no的卡（注意：no标号从0开始）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 这个是为了之后走双卡
model.to(device)
summary(model, input_size=(3, 224, 224))


# Hybrid2测试224
backbone = Hybrid_backbone_4(block_constructor=Bottleneck_block_constructor,
                           bottleneck_channels_setting=[64, 128, 256, 512],
                           identity_layers_setting=[3, 4, 6, 3],
                           stage_stride_setting=[1, 2, 2, 2],
                           fc_num_classes=None,
                           feature_idx='stages')

model = Transformer_blocks.Stage_wise_hybrid_Transformer(backbone, num_tokens=1)
x = torch.randn(7, 3, 224, 224)
x = model(x)
print(x.shape)


# Hybrid2测试384
backbone = Hybrid_backbone_4(block_constructor=Bottleneck_block_constructor,
                           bottleneck_channels_setting=[64, 128, 256, 512],
                           identity_layers_setting=[3, 4, 6, 3],
                           stage_stride_setting=[1, 2, 2, 2],
                           fc_num_classes=None,
                           feature_idx='stages')

model = Transformer_blocks.Stage_wise_hybrid_Transformer(backbone, num_tokens=1, stage_size=(96, 48, 24, 12),
                                                         stage_dim=[256, 512, 1024, 2048])
x = torch.randn(6, 3, 384, 384)
x = model(x)
print(x.shape)


# Hybridx测试
backbone = Hybrid_backbone_4(block_constructor=Bottleneck_block_constructor,
                           bottleneck_channels_setting=[64, 128, 256, 512],
                           identity_layers_setting=[3, 4, 6, 3],
                           stage_stride_setting=[1, 2, 2, 2],
                           fc_num_classes=None,
                           feature_idx='features')

model = Transformer_blocks.Parallel_hybrid_Transformer(backbone, num_tokens=1, stage_size=(96, 48, 24, 12),
                                                       stage_dim=[256, 512, 1024, 2048])
x = torch.randn(6, 3, 384, 384)
x = model(x)
print(x.shape)



# 函数测试
model=create_model(model_idx='Hybrid1_384_505_b8', edge_size=384, pretrained=True, num_classes=2)
x = torch.randn(6, 3, 384, 384)
x = model(x)
print(x.shape)


# 函数测试
model=create_model(model_idx='Hybrid2_384_505_b8', edge_size=384, pretrained=True, num_classes=2,
                   drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                   use_cls_token=True, use_pos_embedding=True)
x = torch.randn(6, 3, 384, 384)
x = model(x)
print(x.shape)

# 函数测试
model=create_model(model_idx='Hybrid3_384_505_b8', edge_size=384, pretrained=True, num_classes=2,
                   drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                   use_cls_token=True, use_pos_embedding=True)
x = torch.randn(6, 3, 384, 384)
x = model(x)
print(x.shape)
"""
