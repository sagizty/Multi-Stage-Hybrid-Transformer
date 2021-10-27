"""
Models  ver： OCT 27th 20：00 official release

by the authors, check our github page:
https://github.com/sagizty/Multi-Stage-Hybrid-Transformer


ResNet stages' feature map

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

ref
https://note.youdao.com/ynoteshare1/index.html?id=5a7dbe1a71713c317062ddeedd97d98e&type=note
"""
import torch
from torch import nn
from functools import partial
from torchsummary import summary
import os
from Hybrid import Transformer_blocks


# ResNet Bottleneck_block_constructor
class Bottleneck_block_constructor(nn.Module):

    extention = 4

    # 定义初始化的网络和参数
    def __init__(self, inplane, midplane, stride, downsample=None):
        super(Bottleneck_block_constructor, self).__init__()

        outplane = midplane * self.extention

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

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out += residual

        out = self.relu(out)

        return out


# Hybrid_backbone of ResNets
class Hybrid_backbone_4(nn.Module):

    def __init__(self, block_constructor, bottleneck_channels_setting=None, identity_layers_setting=None,
                 stage_stride_setting=None, fc_num_classes=None, feature_idx=None):

        if bottleneck_channels_setting is None:
            bottleneck_channels_setting = [64, 128, 256, 512]
        if identity_layers_setting is None:
            identity_layers_setting = [3, 4, 6, 3]
        if stage_stride_setting is None:
            stage_stride_setting = [1, 2, 2, 2]

        self.inplane = 64
        self.fc_num_classes = fc_num_classes
        self.feature_idx = feature_idx

        super(Hybrid_backbone_4, self).__init__()

        self.block_constructor = block_constructor  # Bottleneck_block_constructor
        self.bcs = bottleneck_channels_setting  # [64, 128, 256, 512]
        self.ils = identity_layers_setting  # [3, 4, 6, 3]
        self.sss = stage_stride_setting  # [1, 2, 2, 2]

        # stem
        # alter the RGB pic chanel to match inplane
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        # ResNet stages
        self.layer1 = self.make_stage_layer(self.block_constructor, self.bcs[0], self.ils[0], self.sss[0])
        self.layer2 = self.make_stage_layer(self.block_constructor, self.bcs[1], self.ils[1], self.sss[1])
        self.layer3 = self.make_stage_layer(self.block_constructor, self.bcs[2], self.ils[2], self.sss[2])
        self.layer4 = self.make_stage_layer(self.block_constructor, self.bcs[3], self.ils[3], self.sss[3])

        # cls head
        if self.fc_num_classes is not None:
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(512 * self.block_constructor.extention, fc_num_classes)

    def forward(self, x):

        # stem
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        stem_out = self.maxpool(out)

        # Resnet block of 4 stages
        stage1_out = self.layer1(stem_out)
        stage2_out = self.layer2(stage1_out)
        stage3_out = self.layer3(stage2_out)
        stage4_out = self.layer4(stage3_out)

        if self.fc_num_classes is not None:
            # connect to cls head mlp if asked
            fc_out = self.avgpool(stage4_out)
            fc_out = torch.flatten(fc_out, 1)
            fc_out = self.fc(fc_out)

        # get what we need for different usage
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
        block:
        midplane：usually = output chanel/4
        block_num：
        stride：stride of the ResNet Conv Block
        """

        block_list = []

        outplane = midplane * block_constractor.extention  # extention

        if stride != 1 or self.inplane != outplane:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplane, outplane, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(midplane * block_constractor.extention)
            )
        else:
            downsample = None

        # Conv Block
        conv_block = block_constractor(self.inplane, midplane, stride=stride, downsample=downsample)
        block_list.append(conv_block)

        self.inplane = outplane

        # Identity Block
        for i in range(1, block_num):
            block_list.append(block_constractor(self.inplane, midplane, stride=1, downsample=None))

        return nn.Sequential(*block_list)  # stack blocks


class Hybrid_backbone_3(nn.Module):  # 3 stages version

    def __init__(self, block_constructor, bottleneck_channels_setting=None, identity_layers_setting=None,
                 stage_stride_setting=None, fc_num_classes=None, feature_idx=None):

        if bottleneck_channels_setting is None:
            bottleneck_channels_setting = [64, 128, 256]
        if identity_layers_setting is None:
            identity_layers_setting = [3, 4, 6]
        if stage_stride_setting is None:
            stage_stride_setting = [1, 2, 2]

        self.inplane = 64
        self.fc_num_classes = fc_num_classes
        self.feature_idx = feature_idx

        super(Hybrid_backbone_3, self).__init__()

        self.block_constructor = block_constructor  # Bottleneck_block_constructor
        self.bcs = bottleneck_channels_setting  # [64, 128, 256]
        self.ils = identity_layers_setting  # [3, 4, 6]
        self.sss = stage_stride_setting  # [1, 2, 2]

        # stem
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        # ResNet 3 stages
        self.layer1 = self.make_stage_layer(self.block_constructor, self.bcs[0], self.ils[0], self.sss[0])
        self.layer2 = self.make_stage_layer(self.block_constructor, self.bcs[1], self.ils[1], self.sss[1])
        self.layer3 = self.make_stage_layer(self.block_constructor, self.bcs[2], self.ils[2], self.sss[2])

        if self.fc_num_classes is not None:
            self.avgpool = nn.AvgPool2d(24)  # 224-14  384-24
            self.fc = nn.Linear(self.bcs[-1] * self.block_constructor.extention, fc_num_classes)

    def forward(self, x):
        # stem:conv+bn+relu+maxpool
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        stem_out = self.maxpool(out)

        # Resnet 3 stages
        stage1_out = self.layer1(stem_out)
        stage2_out = self.layer2(stage1_out)
        stage3_out = self.layer3(stage2_out)

        if self.fc_num_classes is not None:
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
        block:
        midplane:
        block_num:
        stride:
        """

        block_list = []

        outplane = midplane * block_constractor.extention  # extention

        if stride != 1 or self.inplane != outplane:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplane, outplane, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(midplane * block_constractor.extention)
            )
        else:
            downsample = None

        # Conv Block
        conv_block = block_constractor(self.inplane, midplane, stride=stride, downsample=downsample)
        block_list.append(conv_block)

        self.inplane = outplane

        # Identity Block
        for i in range(1, block_num):
            block_list.append(block_constractor(self.inplane, midplane, stride=1, downsample=None))

        return nn.Sequential(*block_list)


def Hybrid_a(backbone, img_size=224, patch_size=1, in_chans=3, num_classes=1000, embed_dim=768, depth=8,
             num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, drop_rate=0., attn_drop_rate=0.,
             drop_path_rate=0., norm_layer=None, act_layer=None):
    # directly stack CNNs and Transformer blocks
    embed_layer = partial(Transformer_blocks.Hybrid_feature_map_Embed, backbone=backbone)

    Hybrid_model = Transformer_blocks.VisionTransformer(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                                                        num_heads, mlp_ratio, qkv_bias, representation_size,
                                                        drop_rate, attn_drop_rate, drop_path_rate, embed_layer,
                                                        norm_layer, act_layer)

    return Hybrid_model


def create_model(model_idx, edge_size, pretrained=True, num_classes=2, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., use_cls_token=True, use_pos_embedding=True, use_att_module='SimAM'):
    """
    get one of MSHT models

    :param model_idx: the model we are going to use. by the format of Model_size_other_info
    :param edge_size: the input edge size of the dataloder
    :param pretrained: The backbone CNN is initiate randomly or by its official Pretrained models
    :param num_classes: classification required number of your dataset

    :param drop_rate: The dropout layer's probility of proposed models
    :param attn_drop_rate: The dropout layer(right after the MHSA block or MHGA block)'s probility of proposed models
    :param drop_path_rate: The probility of stochastic depth

    :param use_cls_token: To use the class token
    :param use_pos_embedding: To use the positional enbedding
    :param use_att_module: To use which attention module in the FGD Focus block
    # use_att_module in ['SimAM', 'CBAM', 'SE']  different attention module we applied in the ablation study

    :return: prepared model
    """

    if pretrained:
        from torchvision import models
        backbone_weights = models.resnet50(pretrained=True).state_dict()
        # True for pretrained Resnet50 model, False will randomly initiate
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
