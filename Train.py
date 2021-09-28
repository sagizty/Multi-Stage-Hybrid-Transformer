"""
训练框架函数  版本： 9月21日 15：17

Backbone：ViT/ResNet50 实现分类任务的迁移学习
数据集:
image folder 格式就行
"""

from __future__ import print_function, division
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchsummary import summary
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import shutil
from tensorboardX import SummaryWriter
from PIL import Image


def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def setup_seed(seed):  # 设置随机数种子
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def imshow(inp, title=None):  # 与data_transforms对应
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    '''
    # 因为 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    '''
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# 获取模型
def get_model(num_classes=1000, edge_size=224, model_idx=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0):
    if model_idx[0:6] == 'ResNet':
        if model_idx[0:8] == 'ResNet34':
            model = models.resnet34(pretrained=True)  # True 是预训练好的Resnet34模型，False是随机初始化参数的模型
        elif model_idx[0:8] == 'ResNet50':
            model = models.resnet50(pretrained=True)  # True 是预训练好的Resnet50模型，False是随机初始化参数的模型
        elif model_idx[0:9] == 'ResNet101':
            model = models.resnet101(pretrained=True)  # True 是预训练好的Resnet101模型，False是随机初始化参数的模型
        else:
            print('this model is not defined in get model')
            return -1
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    # TODO 测试完记得删除？
    elif model_idx[0:4] == 'test':
        if model_idx[0:5] == 'test0':  # ResNet+Encoder/Decoder融合模型384  EG:Hybrid2_224_501
            from Hybrid import ResHybrid

            model_idx = 'Hybrid2_384'  # TODO 测试完记得删除

            model = ResHybrid.create_model(model_idx, edge_size, pretrained=True, num_classes=num_classes,
                                           drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0)
        elif model_idx[0:5] == 'test1':  # ResNet+Encoder/Decoder融合模型384  EG:Hybrid2_224_501
            from Hybrid import ResHybrid

            model_idx = 'Hybrid2_384'  # TODO 测试完记得删除

            model = ResHybrid.create_model(model_idx, edge_size, pretrained=True, num_classes=num_classes,
                                           drop_rate=0.3, attn_drop_rate=0.2, drop_path_rate=0.1)
        elif model_idx[0:5] == 'test2':  # ResNet+Encoder/Decoder融合模型384  EG:Hybrid2_224_501
            from Hybrid import ResHybrid

            model_idx = 'Hybrid2_384'  # TODO 测试完记得删除

            model = ResHybrid.create_model(model_idx, edge_size, pretrained=True, num_classes=num_classes,
                                           drop_rate=0.2, attn_drop_rate=0.2, drop_path_rate=0.2)

        elif model_idx[0:5] == 'test3':  # ResNet+Encoder/Decoder融合模型384  EG:Hybrid2_224_501
            from Hybrid import ResHybrid

            model_idx = 'Hybrid2_384'  # TODO 测试完记得删除

            model = ResHybrid.create_model(model_idx, edge_size, pretrained=True, num_classes=num_classes,
                                           drop_rate=0.1, attn_drop_rate=0.2, drop_path_rate=0.2)

        elif model_idx[0:5] == 'test4':  # ResNet+Encoder/Decoder融合模型384  EG:Hybrid2_224_501
            from Hybrid import ResHybrid

            model_idx = 'Hybrid2_384'  # TODO 测试完记得删除

            model = ResHybrid.create_model(model_idx, edge_size, pretrained=True, num_classes=num_classes,
                                           drop_rate=0.3, attn_drop_rate=0.3, drop_path_rate=0.3)
        elif model_idx[0:5] == 'test5':  # ResNet+Encoder/Decoder融合模型384  EG:Hybrid2_224_501
            from Hybrid import ResHybrid

            model_idx = 'Hybrid2_384'  # TODO 测试完记得删除

            model = ResHybrid.create_model(model_idx, edge_size, pretrained=True, num_classes=num_classes,
                                           drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.2)

        elif model_idx[0:5] == 'test6':  # ResNet+Encoder/Decoder融合模型384  EG:Hybrid2_224_501
            from Hybrid import ResHybrid

            model_idx = 'Hybrid2_384'  # TODO 测试完记得删除

            model = ResHybrid.create_model(model_idx, edge_size, pretrained=True, num_classes=num_classes,
                                           drop_rate=0.2, attn_drop_rate=0.2, drop_path_rate=0.3)

        elif model_idx[0:5] == 'test7':  # ResNet+Encoder/Decoder融合模型384  EG:Hybrid2_224_501
            from Hybrid import ResHybrid

            model_idx = 'Hybrid2_384'  # TODO 测试完记得删除

            model = ResHybrid.create_model(model_idx, edge_size, pretrained=True, num_classes=num_classes,
                                           drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.3)
        elif model_idx[0:5] == 'test8':  # ResNet+Encoder/Decoder融合模型384  EG:Hybrid2_224_501
            from Hybrid import ResHybrid

            model_idx = 'Hybrid2_384'  # TODO 测试完记得删除

            model = ResHybrid.create_model(model_idx, edge_size, pretrained=True, num_classes=num_classes,
                                           drop_rate=0.1, attn_drop_rate=0.2, drop_path_rate=0.3)

        elif model_idx[0:5] == 'test9':  # ResNet+Encoder/Decoder融合模型384  EG:Hybrid2_224_501
            from Hybrid import ResHybrid

            model_idx = 'Hybrid2_384'  # TODO 测试完记得删除

            model = ResHybrid.create_model(model_idx, edge_size, pretrained=True, num_classes=num_classes,
                                           drop_rate=0.2, attn_drop_rate=0.1, drop_path_rate=0.3)
        elif model_idx[0:5] == 'testa':  # ResNet+Encoder/Decoder融合模型384  EG:Hybrid2_224_501
            from Hybrid import ResHybrid

            model_idx = 'Hybrid2_384'  # TODO 测试完记得删除

            model = ResHybrid.create_model(model_idx, edge_size, pretrained=True, num_classes=num_classes,
                                           drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.05)
        elif model_idx[0:5] == 'testb':  # ResNet+Encoder/Decoder融合模型384  EG:Hybrid2_224_501
            from Hybrid import ResHybrid

            model_idx = 'Hybrid2_384'  # TODO 测试完记得删除

            model = ResHybrid.create_model(model_idx, edge_size, pretrained=True, num_classes=num_classes,
                                           drop_rate=0.13, attn_drop_rate=0.22, drop_path_rate=0.11)
        elif model_idx[0:5] == 'testc':  # ResNet+Encoder/Decoder融合模型384  EG:Hybrid2_224_501
            from Hybrid import ResHybrid

            model_idx = 'Hybrid2_384'  # TODO 测试完记得删除

            model = ResHybrid.create_model(model_idx, edge_size, pretrained=True, num_classes=num_classes,
                                           drop_rate=0.08, attn_drop_rate=0.12, drop_path_rate=0.21)
        elif model_idx[0:5] == 'testd':  # ResNet+Encoder/Decoder融合模型384  EG:Hybrid2_224_501
            from Hybrid import ResHybrid

            model_idx = 'Hybrid2_384'  # TODO 测试完记得删除

            model = ResHybrid.create_model(model_idx, edge_size, pretrained=True, num_classes=num_classes,
                                           drop_rate=0.1, attn_drop_rate=0.12, drop_path_rate=0.15)
        elif model_idx[0:5] == 'teste':  # ResNet+Encoder/Decoder融合模型384  EG:Hybrid2_224_501
            from Hybrid import ResHybrid

            model_idx = 'Hybrid2_384'  # TODO 测试完记得删除

            model = ResHybrid.create_model(model_idx, edge_size, pretrained=True, num_classes=num_classes,
                                           drop_rate=0.13, attn_drop_rate=0.12, drop_path_rate=0.05)
        elif model_idx[0:5] == 'testf':  # ResNet+Encoder/Decoder融合模型384  EG:Hybrid2_224_501
            from Hybrid import ResHybrid

            model_idx = 'Hybrid2_384'  # TODO 测试完记得删除

            model = ResHybrid.create_model(model_idx, edge_size, pretrained=True, num_classes=num_classes,
                                           drop_rate=0.1, attn_drop_rate=0.12, drop_path_rate=0.3)
        elif model_idx[0:5] == 'testg':  # ResNet+Encoder/Decoder融合模型384  EG:Hybrid2_224_501
            from Hybrid import ResHybrid

            model_idx = 'Hybrid2_384'  # TODO 测试完记得删除

            model = ResHybrid.create_model(model_idx, edge_size, pretrained=True, num_classes=num_classes,
                                           drop_rate=0.1, attn_drop_rate=0.15, drop_path_rate=0.3)

        elif model_idx[0:5] == 'testh':  # ResNet+Encoder/Decoder融合模型384  EG:Hybrid2_224_501
            from Hybrid import ResHybrid

            model_idx = 'Hybrid2_384'  # TODO 测试完记得删除

            model = ResHybrid.create_model(model_idx, edge_size, pretrained=True, num_classes=num_classes,
                                           drop_rate=0.1, attn_drop_rate=0.17, drop_path_rate=0.3)

    elif model_idx[0:6] == 'Hybrid':  # ResNet+Encoder/Decoder融合模型384  EG:Hybrid2_224_501
        from Hybrid import ResHybrid

        model = ResHybrid.create_model(model_idx, edge_size, pretrained=True, num_classes=num_classes,
                                       drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                       drop_path_rate=drop_path_rate)

    elif model_idx[0:7] == 'ViT_224':
        # ViT 迁移学习224
        import timm
        from pprint import pprint
        model_names = timm.list_models('*vit*')
        pprint(model_names)
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    elif model_idx[0:7] == 'ViT_384':
        # ViT 迁移学习384
        import timm
        from pprint import pprint
        model_names = timm.list_models('*vit*')
        pprint(model_names)
        model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=num_classes)

    elif model_idx[0:7] == 'bot_256':  # BoT修改模型 edge=256
        import timm
        from pprint import pprint
        model_names = timm.list_models('*bot*')
        pprint(model_names)
        # ['botnet26t_256', 'botnet50ts_256', 'eca_botnext26ts_256']
        edge_size = 256  # !!!!!!!
        model = timm.create_model('botnet26t_256', pretrained=True, num_classes=num_classes)

    elif model_idx[0:3] == 'vgg':
        # vgg16_bn 迁移学习
        import timm
        from pprint import pprint
        model_names = timm.list_models('*vgg*')
        pprint(model_names)
        if model_idx[0:8] == 'vgg16_bn':
            model = timm.create_model('vgg16_bn', pretrained=True, num_classes=num_classes)
        elif model_idx[0:5] == 'vgg16':
            model = timm.create_model('vgg16', pretrained=True, num_classes=num_classes)
        elif model_idx[0:8] == 'vgg19_bn':
            model = timm.create_model('vgg19_bn', pretrained=True, num_classes=num_classes)
        elif model_idx[0:5] == 'vgg19':
            model = timm.create_model('vgg19', pretrained=True, num_classes=num_classes)

    elif model_idx[0:8] == 'xception':  # xception 迁移学习 lr=501
        import timm
        from pprint import pprint
        model_names = timm.list_models('*xception*')
        pprint(model_names)
        model = timm.create_model('xception', pretrained=True, num_classes=num_classes)

    elif model_idx[0:10] == 'swin_b_384':  # swin_b修改模型384
        import timm
        from pprint import pprint
        model_names = timm.list_models('*swin*')
        pprint(model_names)  # swin_base_patch4_window12_384  swin_base_patch4_window12_384_in22k
        model = timm.create_model('swin_base_patch4_window12_384', pretrained=True, num_classes=num_classes)

    elif model_idx[0:11] == 'DenseNet169':  # Densenet169 TODO 死活有bug
        import timm
        from pprint import pprint
        model_names = timm.list_models('*densenet*')
        pprint(model_names)
        model = timm.create_model('densenet169', pretrained=True, num_classes=num_classes)

    elif model_idx[0:14] == 'ResN50_ViT_384':  # ResNet+ViT融合模型384
        import timm
        from pprint import pprint
        model_names = timm.list_models('*vit_base_resnet*')
        pprint(model_names)
        model = timm.create_model('vit_base_resnet50_384', pretrained=True, num_classes=num_classes)

    elif model_idx[0:14] == 'ResN50_ViT_224':  # ResNet+ViT融合模型224
        import timm
        from pprint import pprint
        model_names = timm.list_models('*vit_base_resnet*')
        pprint(model_names)
        model = timm.create_model('vit_base_r50_s16_224', pretrained=True, num_classes=num_classes)

    elif model_idx[0:14] == 'efficientnet_b':  # efficientnet_b3,4 迁移学习lr=401
        import timm
        from pprint import pprint
        model_names = timm.list_models('*efficientnet*')
        pprint(model_names)
        model = timm.create_model(model_idx[0:15], pretrained=True, num_classes=num_classes)

    elif model_idx[0:16] == 'cross_former_224':
        from cross_former_models import crossformer
        backbone = crossformer.CrossFormer(img_size=224,
                                           patch_size=[4, 8, 16, 32],
                                           in_chans=3,
                                           num_classes=0,  # backbone 模式
                                           embed_dim=96,
                                           depths=[2, 2, 18, 2],
                                           num_heads=[3, 6, 12, 24],
                                           group_size=[7, 7, 7, 7],
                                           mlp_ratio=4.,
                                           qkv_bias=True,
                                           qk_scale=None,
                                           drop_rate=0.0,
                                           drop_path_rate=0.3,
                                           ape=False,
                                           patch_norm=True,
                                           use_checkpoint=False,
                                           merge_size=[[2, 4], [2, 4], [2, 4]], )

        save_model_path = './saved_models/crossformer-b.pth'  # todo
        # r'C:\Users\admin\Desktop\saved_models\crossformer-b.pth'
        # /home/ZTY/saved_models/crossformer-b.pth
        backbone.load_state_dict(torch.load(save_model_path)['model'], False)
        model = crossformer.cross_former_cls_head_warp(backbone, num_classes)

        return model

    # TODO ConViT levit有问题, 需要维修
    elif model_idx in ['ConViT_501', 'ConViT_502']:  # ConViT 迁移学习
        import timm
        from pprint import pprint
        model_names = timm.list_models('*convit*')
        pprint(model_names)
        model = timm.create_model('convit_base', pretrained=True, num_classes=num_classes)

    else:
        print("没有定义该模型！！")
        return -1

    try:
        img = torch.randn(1, 3, edge_size, edge_size)
        preds = model(img)  # (1, 1000)
        print('模型测试：', preds)
    except:
        print("模型有问题！！")
        return -1
    else:
        print('model is ready now!')
        return model


# Grad CAM部分：CNN注意力可视化
def check_grad_CAM(model, dataloader, class_names, check_index=1, num_images=3, device='cpu', skip_batch=10,
                   pic_name='test', draw_path='/home/ZTY/imaging_results', writer=None):
    '''
    检查num_images个图片在每个类别上的cam，每行有每个类别的图，行数=num_images，为检查的图片数量
    :param model:输入模型
    :param dataloader:输入数据dataloader
    :param class_names:分类的类别名字
    :param num_images:需要检验的原图数量,此数量需要小于batchsize
    :param device:cpu/gpu
    :param pic_name:输出图片的名字
    :param draw_path:输出图片的文件夹
    :param writer:输出图片上传到tensorboard服务器
    :return:
    '''
    from utils.grad_CAM import get_last_conv_name, GradCAM, gen_cam

    # Get a batch of training data
    dataloader = iter(dataloader)
    for i in range(check_index * skip_batch):
        inputs, classes = next(dataloader)

    # 预测测试
    was_training = model.training
    model.eval()

    inputs = inputs.to(device)
    labels = classes.to(device)

    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    # 先确定最后一个卷积层名字
    layer_name = get_last_conv_name(model)
    grad_cam = GradCAM(model, layer_name)  # 生成grad cam调取器，包括注册hook等

    images_so_far = 0
    plt.figure()

    for j in range(inputs.size()[0]):

        for cls_idx in range(len(class_names)):
            images_so_far += 1
            ax = plt.subplot(num_images, len(class_names), images_so_far)
            ax.axis('off')
            ax.set_title('GT:{} Pr:{} CAM on {}'.format(class_names[int(labels[j])], class_names[preds[j]],
                                                        class_names[cls_idx]))
            # 基于输入数据 和希望检查的类id，建立对应的cam mask
            mask = grad_cam(inputs[j], cls_idx)
            # 调取原图
            check_image = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
            # 转为叠加图cam，与热力图heatmap保存
            cam, heatmap = gen_cam(check_image, mask)

            plt.imshow(cam)  # 接收一张图像，只是画出该图，并不会立刻显示出来。
            plt.pause(0.001)  # pause a bit so that plots are updated

            if images_so_far == num_images * len(class_names):
                picpath = draw_path + '/' + pic_name + '.jpg'
                if not os.path.exists(draw_path):
                    os.makedirs(draw_path)

                '''
                myfig = plt.gcf()  # get current image
                myfig.savefig(picpath, dpi=1000)
                '''
                plt.savefig(picpath, dpi=1000)
                plt.show()

                grad_cam.remove_handlers()  # 删除注册的hook
                model.train(mode=was_training)

                if writer is not None:  # 用这个方式读取保存的图片到tensorboard上面
                    image_PIL = Image.open(picpath)
                    img = np.array(image_PIL)
                    writer.add_image(pic_name, img, 1, dataformats='HWC')

                return

    grad_cam.remove_handlers()  # 删除注册的hook
    model.train(mode=was_training)


# Transformer的注意力可视化模块
def check_attention(model, data_input):  # 目前实验了ViT-pytorch的模型可以用，其他不确定 TODO
    from vit_pytorch.recorder import Recorder
    v = Recorder(model)

    # forward pass now returns predictions and the attention maps
    img = torch.randn(1, 3, 256, 256)
    preds, attns = v(data_input)

    # there is one extra patch due to the CLS token
    # attns size is  (1, 6, 16, 65, 65) - (batch x layers x heads x patch x patch)

    v = v.eject()  # wrapper is discarded and original ViT instance is returned


# 训练部分

def better_performance(temp_acc, temp_vac, best_acc, best_vac):  # 迭代过程中选用更好的结果

    if temp_vac >= best_vac and temp_acc >= best_acc:
        return True
    elif temp_vac > best_vac:
        return True
    else:
        return False


def train_model(model, dataloaders, criterion, optimizer, class_names, dataset_sizes, model_idx=None, num_epochs=25,
                intake_epochs=30, check_minibatch=100, scheduler=None, device=None,
                draw_path='/home/ZTY/imaging_results', enable_attention_check=False, enable_visualize_check=False,
                enable_sam=False, writer=None):
    # scheduler is an LR scheduler object from torch.optim.lr_scheduler.
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    # 用来保存最好的模型参数
    best_model_wts = copy.deepcopy(model.state_dict())  # deepcopy 防止copy的是内存地址，这里因为目标比较大，用这个保证摘下来
    # 初始化log字典
    json_log = {}

    # 初始化最好的表现
    best_acc = 0.0
    best_vac = 0.0
    temp_acc = 0.0
    temp_vac = 0.0
    best_epoch_idx = 1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # 记录实验json log
        json_log[str(epoch + 1)] = {}

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:  # 采用这个写法来综合写train与val过程

            index = 0
            model_time = time.time()

            # 初始化计数字典
            log_dict = {}
            for cls_idx in range(len(class_names)):
                log_dict[class_names[cls_idx]] = {'tp': 0.0, 'tn': 0.0, 'fp': 0.0, 'fn': 0.0}  # json只接受float

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # 初始化记录表现
            running_loss = 0.0
            log_running_loss = 0.0
            running_corrects = 0

            check_dataloaders = copy.deepcopy(dataloaders)

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:  # 不同任务段用不同dataloader的数据
                inputs = inputs.to(device)
                # print('inputs[0]',type(inputs[0]))

                labels = labels.to(device)

                # zero the parameter gradients
                if not enable_sam:
                    optimizer.zero_grad()

                # forward
                # 要track grad if only in train！包一个 with torch.set_grad_enabled(phase == 'train'):不然就True就行
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # preds是最大值出现的位置，相当于是类别id
                    loss = criterion(outputs, labels)  # loss是基于输出的vector与onehot label做loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if enable_sam:
                            loss.backward()
                            # first forward-backward pass
                            optimizer.first_step(zero_grad=True)

                            # second forward-backward pass
                            loss2 = criterion(model(inputs), labels)  # 为了计算图重新算model(inputs)，不能直接用之前的outputs
                            loss2.backward()  # make sure to do a full forward pass
                            optimizer.second_step(zero_grad=True)
                        else:
                            loss.backward()
                            optimizer.step()

                # 统计表现总和
                log_running_loss += loss.item()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Compute precision and recall for each class.
                for cls_idx in range(len(class_names)):
                    tp = np.dot((labels.cpu().data == cls_idx).numpy().astype(int),
                                (preds == cls_idx).cpu().numpy().astype(int))
                    tn = np.dot((labels.cpu().data != cls_idx).numpy().astype(int),
                                (preds != cls_idx).cpu().numpy().astype(int))

                    fp = np.sum((preds == cls_idx).cpu().numpy()) - tp

                    fn = np.sum((labels.cpu().data == cls_idx).numpy()) - tp

                    # log_dict[cls_idx] = {'tp': 0.0, 'tn': 0.0, 'fp': 0.0, 'fn': 0.0}  # json只接受float
                    log_dict[class_names[cls_idx]]['tp'] += tp
                    log_dict[class_names[cls_idx]]['tn'] += tn
                    log_dict[class_names[cls_idx]]['fp'] += fp
                    log_dict[class_names[cls_idx]]['fn'] += fn

                # 记录内容给tensorboard
                if writer is not None:
                    # ...log the running loss
                    writer.add_scalar(phase + ' minibatch loss',
                                      float(loss.item()),
                                      epoch * len(dataloaders[phase]) + index)
                    writer.add_scalar(phase + ' minibatch ACC',
                                      float(torch.sum(preds == labels.data) / inputs.size(0)),
                                      epoch * len(dataloaders[phase]) + index)

                # 画图检测效果
                if index % check_minibatch == check_minibatch - 1:
                    model_time = time.time() - model_time

                    check_index = index // check_minibatch + 1

                    epoch_idx = epoch + 1
                    print('Epoch:', epoch_idx, '   ', phase, 'index of ' + str(check_minibatch) + ' minibatch:',
                          check_index, '     time used:', model_time)

                    print('minibatch AVG loss:', float(log_running_loss) / check_minibatch)

                    if enable_visualize_check:
                        visualize_check(model, check_dataloaders[phase], class_names, check_index, num_images=3,
                                        device=device,
                                        pic_name='Visual_' + phase + '_E_' + str(epoch_idx) + '_I_' + str(index + 1),
                                        skip_batch=check_minibatch, draw_path=draw_path, writer=writer)

                    if enable_attention_check == 'CAM':
                        check_grad_CAM(model, check_dataloaders[phase], class_names, check_index, num_images=2,
                                       device=device,
                                       pic_name='GradCAM_' + phase + '_E_' + str(epoch_idx) + '_I_' + str(index + 1),
                                       skip_batch=check_minibatch, draw_path=draw_path, writer=writer)
                    else:
                        pass

                    model_time = time.time()
                    log_running_loss = 0.0

                index += 1

            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()

            # 记录输出本轮情况
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase] * 100
            print('\nEpoch: {}  {} \nLoss: {:.4f}  Acc: {:.4f}'.format(epoch + 1, phase, epoch_loss, epoch_acc))
            # 记录内容给tensorboard
            if writer is not None:
                # ...log the running loss
                writer.add_scalar(phase + ' loss',
                                  float(epoch_loss),
                                  epoch + 1)
                writer.add_scalar(phase + ' ACC',
                                  float(epoch_acc),
                                  epoch + 1)

            for cls_idx in range(len(class_names)):
                tp = log_dict[class_names[cls_idx]]['tp']
                tn = log_dict[class_names[cls_idx]]['tn']
                fp = log_dict[class_names[cls_idx]]['fp']
                fn = log_dict[class_names[cls_idx]]['fn']
                tp_plus_fp = tp + fp
                tp_plus_fn = tp + fn
                fp_plus_tn = fp + tn
                fn_plus_tn = fn + tn

                # precision
                if tp_plus_fp == 0:
                    precision = 0
                else:
                    precision = float(tp) / tp_plus_fp * 100
                # recall
                if tp_plus_fn == 0:
                    recall = 0
                else:
                    recall = float(tp) / tp_plus_fn * 100

                # TPR (sensitivity)
                TPR = recall

                # TNR (specificity)
                # FPR
                if fp_plus_tn == 0:
                    TNR = 0
                    FPR = 0
                else:
                    TNR = tn / fp_plus_tn * 100
                    FPR = fp / fp_plus_tn * 100

                # NPV
                if fn_plus_tn == 0:
                    NPV = 0
                else:
                    NPV = tn / fn_plus_tn * 100

                print('{} precision: {:.4f}  recall: {:.4f}'.format(class_names[cls_idx], precision, recall))
                print('{} sensitivity: {:.4f}  specificity: {:.4f}'.format(class_names[cls_idx], TPR, TNR))
                print('{} FPR: {:.4f}  NPV: {:.4f}'.format(class_names[cls_idx], FPR, NPV))
                print('{} TP: {}'.format(class_names[cls_idx], tp))
                print('{} TN: {}'.format(class_names[cls_idx], tn))
                print('{} FP: {}'.format(class_names[cls_idx], fp))
                print('{} FN: {}'.format(class_names[cls_idx], fn))
                # 记录内容给tensorboard
                if writer is not None:
                    # ...log the running loss
                    writer.add_scalar(phase + '   ' + class_names[cls_idx] + ' precision',
                                      precision,
                                      epoch + 1)
                    writer.add_scalar(phase + '   ' + class_names[cls_idx] + ' recall',
                                      recall,
                                      epoch + 1)
                # 记录实验json log
                json_log[str(epoch + 1)][phase] = log_dict

            if phase == 'val':
                temp_vac = epoch_acc
            else:
                temp_acc = epoch_acc  # 假设这里是train的时候，记得记录

            # deep copy the model，如果本epoch为止比之前都表现更好才刷新参数记录
            # 目的是在epoch很多的时候，表现开始下降了，那么拿中间的就行
            if phase == 'val' and better_performance(temp_acc, temp_vac, best_acc, best_vac) and epoch >= intake_epochs:
                # TODO 需要定义"更好" 模型在足够收敛epoch>=30之后才有意义
                best_epoch_idx = epoch + 1
                best_acc = temp_acc
                best_vac = temp_vac
                best_model_wts = copy.deepcopy(model.state_dict())
                best_log_dic = log_dict

            print('\n')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch idx: ', best_epoch_idx)
    print('Best epoch train Acc: {:4f}'.format(best_acc))
    print('Best epoch val Acc: {:4f}'.format(best_vac))
    for cls_idx in range(len(class_names)):
        tp = best_log_dic[class_names[cls_idx]]['tp']
        tn = best_log_dic[class_names[cls_idx]]['tn']
        fp = best_log_dic[class_names[cls_idx]]['fp']
        fn = best_log_dic[class_names[cls_idx]]['fn']
        tp_plus_fp = tp + fp
        tp_plus_fn = tp + fn
        fp_plus_tn = fp + tn
        fn_plus_tn = fn + tn

        # precision
        if tp_plus_fp == 0:
            precision = 0
        else:
            precision = float(tp) / tp_plus_fp * 100
        # recall
        if tp_plus_fn == 0:
            recall = 0
        else:
            recall = float(tp) / tp_plus_fn * 100

        # TPR (sensitivity)
        TPR = recall

        # TNR (specificity)
        # FPR
        if fp_plus_tn == 0:
            TNR = 0
            FPR = 0
        else:
            TNR = tn / fp_plus_tn * 100
            FPR = fp / fp_plus_tn * 100

        # NPV
        if fn_plus_tn == 0:
            NPV = 0
        else:
            NPV = tn / fn_plus_tn * 100

        print('{} precision: {:.4f}  recall: {:.4f}'.format(class_names[cls_idx], precision, recall))
        print('{} sensitivity: {:.4f}  specificity: {:.4f}'.format(class_names[cls_idx], TPR, TNR))
        print('{} FPR: {:.4f}  NPV: {:.4f}'.format(class_names[cls_idx], FPR, NPV))

    # 结束记录内容给tensorboard
    if writer is not None:
        writer.close()

    # load best model weights as final model training result 这也是一种避免过拟合的方法
    model.load_state_dict(best_model_wts)
    # 保存json_log  indent=2 更加美观
    json.dump(json_log, open(os.path.join(draw_path, model_idx + '_log.json') , 'w'), ensure_ascii=False, indent=2)
    return model


def visualize_check(model, dataloader, class_names, check_index=1, num_images=9, device='cpu', pic_name='test',
                    skip_batch=10, draw_path='/home/ZTY/imaging_results', writer=None):  # 预测测试
    '''
    对num_images个图片进行检查，每行放3个图
    :param model:输入模型
    :param dataloader:输入数据dataloader
    :param class_names:分类的类别名字
    :param num_images:需要检验的原图数量,此数量需要小于batchsize
    :param device:cpu/gpu
    :param pic_name:输出图片的名字
    :param draw_path:输出图片的文件夹
    :param writer:输出图片上传到tensorboard服务器
    :return:
    '''
    was_training = model.training
    model.eval()

    images_so_far = 0
    plt.figure()

    with torch.no_grad():

        dataloader = iter(dataloader)
        for i in range(check_index * skip_batch):
            inputs, classes = next(dataloader)

        inputs = inputs.to(device)
        labels = classes.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images // 3, 3, images_so_far)
            ax.axis('off')
            ax.set_title('Pred: {} True: {}'.format(class_names[preds[j]], class_names[int(labels[j])]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                picpath = draw_path + '/' + pic_name + '.jpg'
                if not os.path.exists(draw_path):
                    os.makedirs(draw_path)

                '''
                myfig = plt.gcf()  # get current image
                myfig.savefig(picpath, dpi=1000)
                '''
                plt.savefig(picpath, dpi=1000)
                plt.show()

                model.train(mode=was_training)

                if writer is not None:  # 用这个方式读取保存的图片到tensorboard上面
                    image_PIL = Image.open(picpath)
                    img = np.array(image_PIL)
                    writer.add_image(pic_name, img, 1, dataformats='HWC')

                return

        model.train(mode=was_training)


def main(args):
    if args.paint:
        # 使用Agg模式，不在本地画图
        import matplotlib
        matplotlib.use('Agg')

    enable_notify = args.enable_notify  # True
    enable_tensorboard = args.enable_tensorboard  # True
    enable_attention_check = args.enable_attention_check  # False   'CAM' 'SAA'
    enable_visualize_check = args.enable_visualize_check  # False

    enable_sam = args.enable_sam  # False

    Pre_Trained_model_path = args.Pre_Trained_model_path  # None

    gpu_idx = args.gpu_idx  # GPU idx start with0, -1 to use multipel GPU

    # model info
    model_idx = args.model_idx  # 'Hybrid2_384_507_b8'  # 'ViT_384_505_b8'

    drop_rate = args.drop_rate
    attn_drop_rate = args.attn_drop_rate
    drop_path_rate = args.drop_path_rate

    # 任务类别数量
    num_classes = args.num_classes  # 2
    # 边长
    edge_size = args.edge_size  # 224 384 1000

    # batch info
    batch_size = args.batch_size  # 8
    num_workers = args.num_workers  # main training num_workers 4

    num_epochs = args.num_epochs  # 50
    intake_epochs = args.intake_epochs  # 30

    lr = args.lr  # 0.000007
    lrf = args.lrf  # 0.0

    opt_name = args.opt_name  # 'Adam'

    # 路径 配置
    draw_root = args.draw_root  # r'C:\Users\admin\Desktop\runs'
    model_path = args.model_path  # r'C:\Users\admin\Desktop\saved_models'
    dataroot = args.dataroot  # r'C:\Users\admin\Desktop\ZTY_dataset1'

    # k5_dataset\fold_1  # 随机5折交叉验证
    # f5_dataset\fold_1  # 分层5折交叉验证

    '''
    # 服务器配置
    draw_root = '/home/ZTY/runs' 
    model_path = '/home/ZTY/saved_models'
    dataroot = '/data/pancreatic-cancer-project/712_dataset'
    '''

    # 模型记录  ViT_384_401_b8 ResNet50_384_401 swin_b_501 ResN50_ViT_501 ResN50_ViT_501_s
    # Hybrid2_384_401  cross_former_224_501

    # ViT lr = 501  edge_size=384 迁移学习 'ViT_f_1'
    # ResNet lr = 401  edge_size=384 迁移学习 'ResNet50_f_1'

    if enable_notify:
        import notifyemail as notify

        notify.Reboost(mail_host='smtp.163.com', mail_user='tum9598@163.com', mail_pass='EXVGQACCXPUIUQAE',
                       default_reciving_list=['904639643@qq.com', 'pytorch1225@163.com'],
                       log_root_path='log', max_log_cnt=5)

        if enable_tensorboard:
            notify.add_text('过程上传到tensorboard')
        else:
            notify.add_text('过程没有上传到tensorboard')

        notify.add_text('  ')

        notify.add_text('模型编号 ' + str(model_idx))
        notify.add_text('  ')

        notify.add_text('GPU idx: ' + str(gpu_idx))
        notify.add_text('  ')

        notify.add_text('任务类别数量 ' + str(num_classes))
        notify.add_text('边长 ' + str(edge_size))
        notify.add_text('batch_size ' + str(batch_size))
        notify.add_text('num_epochs ' + str(num_epochs))
        notify.add_text('lr ' + str(lr))
        notify.add_text('opt_name ' + str(opt_name))
        notify.add_text('enable_sam ' + str(enable_sam))
        notify.send_log()

    print("*********************************{}*************************************".format('setting'))
    print(args)

    draw_path = os.path.join(draw_root, 'PC_' + model_idx)
    save_model_path = os.path.join(model_path, 'PC_' + model_idx + '.pth')

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if os.path.exists(draw_path):
        del_file(draw_path)  # 每次开始的时候都先清空一次
    else:
        os.makedirs(draw_path)

    # 调取tensorboard服务器
    if enable_tensorboard:
        writer = SummaryWriter(draw_path)
    else:
        writer = None
    # nohup tensorboard --logdir=/home/ZTY/runs --host=10.201.10.16 --port=7777 &
    # tensorboard --logdir=C:\Users\admin\Desktop\runs --host=192.168.1.139 --port=7777

    # 根据边长选择合适的data transform
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation((0, 180)),
            transforms.CenterCrop(700),  # 旋转之后选取中间区域（避免黑边）
            transforms.Resize(edge_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
            # 色相饱和度对比度明度的相关的处理H S L，随即灰度化
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(700),
            transforms.Resize(edge_size),
            transforms.ToTensor()
        ]),
    }

    datasets = {x: torchvision.datasets.ImageFolder(os.path.join(dataroot, x), data_transforms[x]) for x in
                ['train', 'val']}  # 两个数据集合并装载

    dataloaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True,
                                                        num_workers=num_workers),  # 调小了num_workers，之前是4
                   'val': torch.utils.data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False,
                                                      num_workers=num_workers // 4 + 1)  # 4
                   }

    class_names = ['negative', 'positive'][0:num_classes]  # A G E B
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}  # 数据数量

    if gpu_idx == -1:
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            gpu_use = gpu_idx
        else:
            print('we dont have more GPU idx here, try to use gpu_idx=0')
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只让程序看到物理卡号为gpu_idx的卡（注意：no标号从0开始）
                gpu_use = 0
            except:
                print("GPU distributing ERRO occur use CPU instead")

    else:
        # Decide which device we want to run on
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)  # 只让程序看到物理卡号为gpu_idx的卡（注意：no标号从0开始）
            gpu_use = gpu_idx
        except:
            print('we dont have that GPU idx here, try to use gpu_idx=0')
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只让程序看到物理卡号为gpu_idx的卡（注意：no标号从0开始）
                gpu_use = 0
            except:
                print("GPU distributing ERRO occur use CPU instead")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 这个是为了之后走双卡

    # 调取模型
    if Pre_Trained_model_path is not None:
        if os.path.exists(Pre_Trained_model_path):
            pretrain_model = get_model(1000, edge_size, model_idx, drop_rate, attn_drop_rate, drop_path_rate)
            pretrain_model.load_state_dict(torch.load(Pre_Trained_model_path), False)
            # 离谱!!!，为什么strict=False不能直接改类别  TODO
            num_features = pretrain_model.num_features
            pretrain_model.head = nn.Linear(num_features, num_classes)
            model = pretrain_model
            print('pretrain model loaded')
            # torch.save(model_t.state_dict(), r'C:\Users\admin\Desktop\Hybrid2_384_PreTrain_000_checkpoint19.pth')
        else:
            print('Pre_Trained_model_path:' + Pre_Trained_model_path, ' is NOT avaliable!!!!\n')
            print('we ignore this with a new start up')
    else:
        # 调取模型
        model = get_model(num_classes, edge_size, model_idx, drop_rate, attn_drop_rate, drop_path_rate)

    print('GPU:', gpu_use)

    if gpu_use == -1:
        model = nn.DataParallel(model)

    model.to(device)

    try:
        summary(model, input_size=(3, edge_size, edge_size))  # to device 之后安排, 输出模型结构
    except:
        pass

    print("model :", model_idx)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.00005, momentum=0.9)
    # Every step_size epochs, Decay LR by multipling a factor of gamma
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # Adam
    # optimizer = optim.Adam(model_ft.parameters(), lr=0.0000001, weight_decay=0.01)
    # scheduler = None

    # SGD
    # optimizer = optim.SGD(model_ft.parameters(), lr=0.00002, momentum=0.8)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    if opt_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.005)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 15 0.1  default SGD StepLR scheduler
    elif opt_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = None

    if enable_sam:
        from utils.sam import SAM

        if opt_name == 'SGD':
            base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
            optimizer = SAM(model.parameters(), base_optimizer, lr=lr, momentum=0.8)
            scheduler = None
        elif opt_name == 'Adam':
            base_optimizer = torch.optim.Adam  # define an optimizer for the "sharpness-aware" update
            optimizer = SAM(model.parameters(), base_optimizer, lr=lr, weight_decay=0.01)

    if lrf > 0:  # use cosine learning rate schedule
        import math
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * (1 - lrf) + lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # train
    model_ft = train_model(model, dataloaders, criterion, optimizer, class_names, dataset_sizes, model_idx=model_idx,
                           num_epochs=num_epochs, intake_epochs=intake_epochs, check_minibatch=400 // batch_size,
                           scheduler=scheduler, device=device, draw_path=draw_path,
                           enable_attention_check=enable_attention_check,
                           enable_visualize_check=enable_visualize_check, enable_sam=enable_sam, writer=writer)
    # 保存模型(多卡训练的也保存为单卡模型）
    if gpu_use == -1:
        torch.save(model_ft.module.state_dict(), save_model_path)
        print('model trained by multi-GPUs has its single GPU copy saved at ', save_model_path)
    else:
        torch.save(model_ft.state_dict(), save_model_path)
        print('model trained by GPU (idx:' + str(gpu_use) + ') has been saved at ', save_model_path)


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Model Name or index
    parser.add_argument('--model_idx', default='Hybrid2_384_507_testsample', type=str, help='Model Name or index')
    # drop_rate, attn_drop_rate, drop_path_rate
    parser.add_argument('--drop_rate', default=0.0, type=float, help='dropout rate , default 0.0')
    parser.add_argument('--attn_drop_rate', default=0.0, type=float, help='dropout rate Aftter Attention, default 0.0')
    parser.add_argument('--drop_path_rate', default=0.0, type=float, help='drop path for stochastic depth, default 0.0')

    # Enviroment parameters
    parser.add_argument('--gpu_idx', default=-1, type=int,
                        help='use a single GPU with its index, -1 to use multiple GPU')

    # Path parameters
    parser.add_argument('--dataroot', default='/data/pancreatic-cancer-diagnosis-tansformer/ZTY_dataset2',
                        help='path to dataset')
    parser.add_argument('--model_path', default='/home/pancreatic-cancer-diagnosis-tansformer/saved_models',
                        help='path to save model state-dict')
    parser.add_argument('--draw_root', default='/home/pancreatic-cancer-diagnosis-tansformer/runs', help='path to draw and save tensorboard output')

    # Help tool parameters
    parser.add_argument('--paint', action='store_true', help='paint in front desk')  # matplotlib.use('Agg')
    parser.add_argument('--enable_notify', action='store_true', help='enable notify to send email')
    # check tool parameters
    parser.add_argument('--enable_tensorboard', action='store_true', help='enable tensorboard to save status')
    # enable_attention_check = False  # 'CAM' 'SAA'
    parser.add_argument('--enable_attention_check', default=None, type=str, help='check and save attention map')
    parser.add_argument('--enable_visualize_check', action='store_true', help='check and save pics')

    # Training status parameters
    parser.add_argument('--enable_sam', action='store_true', help='use SAM strategy in training')
    # '/home/ZTY/pancreatic-cancer-diagnosis-tansformer/saved_models/PC_Hybrid2_384_PreTrain_000.pth'
    parser.add_argument('--Pre_Trained_model_path', default=None, type=str,
                        help='Finetuning a trained model in this dataset')

    # Dataset based parameters
    parser.add_argument('--num_classes', default=2, type=int, help='classification number')
    parser.add_argument('--edge_size', default=384, type=int, help='edge size of input image')  # 224 256 384 1000
    parser.add_argument('--num_workers', default=8, type=int, help='use CPU num_workers , default 4')

    # Training seting parameters
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size default 8')
    parser.add_argument('--num_epochs', default=50, type=int, help='training epochs')
    parser.add_argument('--intake_epochs', default=0, type=int, help='only save model at epochs after intake_epochs')
    parser.add_argument('--lr', default=0.000007, type=float, help='learing rate')
    parser.add_argument('--lrf', type=float, default=0.0,
                        help='learing rate decay rate, default 0(not enabled), suggest 0.1 and lr=0.00005')
    parser.add_argument('--opt_name', default='Adam', type=str, help='optimizer name Adam or SGD')

    return parser


if __name__ == '__main__':
    # 设置随机数种子
    setup_seed(517)

    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
