"""
Training script  ver： OCT 26th 21：00 official release

dataset structure: ImageNet
image folder dataset is used.
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


# Tools
def del_file(filepath):
    """
    clear all items within a folder
    :param filepath: folder path
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def setup_seed(seed):  # setting up the random seed
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def imshow(inp, title=None):  # Imshow for Tensor
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    '''
    # if required: Alter the transform 
    # because transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
def get_model(num_classes=1000, edge_size=224, model_idx=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
              pretrained_backbone=True, use_cls_token=True, use_pos_embedding=True, use_att_module='SimAM'):
    """
    :param num_classes: classification required number of your dataset
    :param edge_size: the input edge size of the dataloder
    :param model_idx: the model we are going to use. by the format of Model_size_other_info

    :param drop_rate: The dropout layer's probility of proposed models
    :param attn_drop_rate: The dropout layer(right after the MHSA block or MHGA block)'s probility of proposed models
    :param drop_path_rate: The probility of stochastic depth

    :param pretrained_backbone: The backbone CNN is initiate randomly or by its official Pretrained models

    :param use_cls_token: To use the class token
    :param use_pos_embedding: To use the positional enbedding
    :param use_att_module: To use which attention module in the FGD Focus block

    :return: prepared model
    """
    if model_idx[0:3] == 'ViT':
        # Transfer learning for ViT
        import timm
        from pprint import pprint
        model_names = timm.list_models('*vit*')
        pprint(model_names)
        if edge_size == 224:
            model = timm.create_model('vit_base_patch16_224', pretrained=pretrained_backbone, num_classes=num_classes)
        elif edge_size == 384:
            model = timm.create_model('vit_base_patch16_384', pretrained=pretrained_backbone, num_classes=num_classes)
        else:
            pass

    elif model_idx[0:3] == 'vgg':
        # Transfer learning for vgg16_bn
        import timm
        from pprint import pprint
        model_names = timm.list_models('*vgg*')
        pprint(model_names)
        if model_idx[0:8] == 'vgg16_bn':
            model = timm.create_model('vgg16_bn', pretrained=pretrained_backbone, num_classes=num_classes)
        elif model_idx[0:5] == 'vgg16':
            model = timm.create_model('vgg16', pretrained=pretrained_backbone, num_classes=num_classes)
        elif model_idx[0:8] == 'vgg19_bn':
            model = timm.create_model('vgg19_bn', pretrained=pretrained_backbone, num_classes=num_classes)
        elif model_idx[0:5] == 'vgg19':
            model = timm.create_model('vgg19', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:4] == 'deit':  # Transfer learning for DeiT
        import timm
        from pprint import pprint
        model_names = timm.list_models('*deit*')
        pprint(model_names)
        if edge_size == 384:
            model = timm.create_model('deit_base_patch16_384', pretrained=pretrained_backbone, num_classes=2)
        elif edge_size == 224:
            model = timm.create_model('deit_base_patch16_224', pretrained=pretrained_backbone, num_classes=2)
        else:
            pass

    elif model_idx[0:5] == 'twins':  # Transfer learning for twins
        import timm
        from pprint import pprint

        model_names = timm.list_models('*twins*')
        pprint(model_names)
        model = timm.create_model('twins_pcpvt_base', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:6] == 'convit' and edge_size == 224:  # Transfer learning for ConViT
        import timm
        from pprint import pprint

        model_names = timm.list_models('*convit*')
        pprint(model_names)
        model = timm.create_model('convit_base', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:6] == 'ResNet':  # Transfer learning for the ResNets
        if model_idx[0:8] == 'ResNet34':
            model = models.resnet34(pretrained=pretrained_backbone)
        elif model_idx[0:8] == 'ResNet50':
            model = models.resnet50(pretrained=pretrained_backbone)
        elif model_idx[0:9] == 'ResNet101':
            model = models.resnet101(pretrained=pretrained_backbone)
        else:
            print('this model is not defined in get model')
            return -1
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_idx[0:6] == 'Hybrid':  # ours: MSHT
        from Hybrid import ResHybrid
        # NOTICE: HERE 'pretrained' controls only The backbone CNN is initiate randomly
        # or by its official Pretrained models
        model = ResHybrid.create_model(model_idx, edge_size, pretrained=pretrained_backbone, num_classes=num_classes,
                                       drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                       drop_path_rate=drop_path_rate, use_cls_token=use_cls_token,
                                       use_pos_embedding=use_pos_embedding, use_att_module=use_att_module)

    elif model_idx[0:7] == 'bot_256' and edge_size == 256:  # Model: BoT
        import timm
        from pprint import pprint
        model_names = timm.list_models('*bot*')
        pprint(model_names)
        # NOTICE: we find no weight for BoT in timm
        # ['botnet26t_256', 'botnet50ts_256', 'eca_botnext26ts_256']
        model = timm.create_model('botnet26t_256', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:8] == 'densenet':  # Transfer learning for densenet
        import timm
        from pprint import pprint

        model_names = timm.list_models('*densenet*')
        pprint(model_names)
        model = timm.create_model('densenet121', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:8] == 'xception':  # Transfer learning for Xception
        import timm
        from pprint import pprint
        model_names = timm.list_models('*xception*')
        pprint(model_names)
        model = timm.create_model('xception', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:9] == 'visformer' and edge_size == 224:  # Transfer learning for Visformer
        import timm
        from pprint import pprint
        model_names = timm.list_models('*visformer*')
        pprint(model_names)
        model = timm.create_model('visformer_small', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:10] == 'swin_b_384':  # Transfer learning for Swin Transformer (swin_b_384)
        import timm
        from pprint import pprint
        model_names = timm.list_models('*swin*')
        pprint(model_names)  # swin_base_patch4_window12_384  swin_base_patch4_window12_384_in22k
        model = timm.create_model('swin_base_patch4_window12_384', pretrained=pretrained_backbone,
                                  num_classes=num_classes)

    elif model_idx[0:11] == 'mobilenetv3':  # Transfer learning for mobilenetv3
        import timm
        from pprint import pprint
        model_names = timm.list_models('*mobilenet*')
        pprint(model_names)
        model = timm.create_model('mobilenetv3_large_100', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:11] == 'inceptionv3':  # Transfer learning for Inception v3
        import timm
        from pprint import pprint
        model_names = timm.list_models('*inception*')
        pprint(model_names)
        model = timm.create_model('inception_v3', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:14] == 'efficientnet_b':  # Transfer learning for efficientnet_b3,4
        import timm
        from pprint import pprint
        model_names = timm.list_models('*efficientnet*')
        pprint(model_names)
        model = timm.create_model(model_idx[0:15], pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:16] == 'cross_former_224':  # Transfer learning for crossformer base
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
        if pretrained_backbone:
            save_model_path = '../saved_models/crossformer-b.pth'  # todo model is downloaded at the path
            # downloaded from official model state at https://github.com/cheerss/CrossFormer
            backbone.load_state_dict(torch.load(save_model_path)['model'], False)
        model = crossformer.cross_former_cls_head_warp(backbone, num_classes)

        return model

    else:
        print("The model is not difined in the script！！")
        return -1

    try:
        img = torch.randn(1, 3, edge_size, edge_size)
        preds = model(img)  # (1, class_number)
        print('test model output：', preds)
    except:
        print("Problem exist in the model defining process！！")
        return -1
    else:
        print('model is ready now!')
        return model


# Grad CAM part：Visualize of CNN+Transformer attention area
def cls_token_s12_transform(tensor, height=12, width=12):  # based on pytorch_grad_cam
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def cls_token_s24_transform(tensor, height=24, width=24):  # based on pytorch_grad_cam
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def no_cls_token_s12_transform(tensor, height=12, width=12):  # based on pytorch_grad_cam
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def choose_cam_by_model(model, model_idx, edge_size, use_cuda=True):
    """
    :param model: model object
    :param model_idx: model idx for the getting pre-setted layer and size
    :param edge_size: image size for the getting pre-setted layer and size
    """
    from pytorch_grad_cam import GradCAM

    # reshape_transform  todo
    # check class: target_category = None
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.

    if model_idx[0:3] == 'ViT' and edge_size == 384:
        target_layers = [model.blocks[-1].norm1]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                           reshape_transform=cls_token_s24_transform)

    elif model_idx[0:4] == 'deit' and edge_size == 384:
        target_layers = [model.blocks[-1].norm1]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                           reshape_transform=cls_token_s24_transform)

    elif model_idx[0:7] == 'Hybrid1' and edge_size == 384:
        target_layers = [model.blocks[-1].norm1]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                           reshape_transform=cls_token_s12_transform)

    elif model_idx[0:10] == 'ResN50_ViT' and edge_size == 384:
        target_layers = [model.blocks[-1].norm1]  # model.layer4[-1]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                           reshape_transform=cls_token_s24_transform)

    elif model_idx[0:6] == 'ResNet' and edge_size == 384:
        target_layers = [model.layer4[-1]]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda, reshape_transform=None)  # CNN: None

    elif model_idx[0:7] == 'Hybrid2' and edge_size == 384:
        target_layers = [model.dec4.norm1]

        if 'CLS' in model_idx.split('_') and 'No' in model_idx.split('_'):
            grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                               reshape_transform=no_cls_token_s12_transform)

        else:
            grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                               reshape_transform=cls_token_s12_transform)

    elif model_idx[0:7] == 'Hybrid3' and edge_size == 384:
        target_layers = [model.dec3.norm1]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                           reshape_transform=cls_token_s24_transform)

    else:
        print('ERRO in model_idx')
        return -1

    return grad_cam


def check_SAA(model, model_idx, edge_size, dataloader, class_names, check_index=1, num_images=2, device='cpu',
              skip_batch=10, pic_name='test', draw_path='/home/ZTY/imaging_results', writer=None):
    """
    check num_images of images and visual the models's attention area
    output a pic with 2 column and rows of num_images

    :param model: model object
    :param model_idx: model idx for the getting pre-setted layer and size
    :param edge_size: image size for the getting pre-setted layer and size

    :param dataloader: checking dataloader
    :param class_names: The name of classes for painting
    :param num_images: how many image u want to check, should SMALLER THAN the batchsize
    :param device: cpu/gpu object
    :param skip_batch: number of skip over minibatch
    :param pic_name: name of the output pic
    :param draw_path: path folder for output pic
    :param writer: attach the pic to the tensorboard backend

    :return: None
    """
    from pytorch_grad_cam.utils import show_cam_on_image

    # Get a batch of training data
    dataloader = iter(dataloader)
    for i in range(check_index * skip_batch):  # skip the tested batchs
        inputs, classes = next(dataloader)

    # test model
    was_training = model.training
    model.eval()

    inputs = inputs.to(device)
    labels = classes.to(device)

    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    grad_cam = choose_cam_by_model(model, model_idx, edge_size)  # 使用选择的方式

    images_so_far = 0
    plt.figure()

    for j in range(inputs.size()[0]):

        for type in ['ori', 'tar']:
            images_so_far += 1
            if type == 'ori':
                ax = plt.subplot(num_images, 2, images_so_far)
                ax.axis('off')
                ax.set_title('Ground Truth:{}'.format(class_names[int(labels[j])]))
                imshow(inputs.cpu().data[j])
                plt.pause(0.001)  # pause a bit so that plots are updated

            else:
                ax = plt.subplot(num_images, 2, images_so_far)
                ax.axis('off')
                ax.set_title('Predict:{}'.format(class_names[preds[j]]))
                # focus on the specific target class to create grayscale_cam
                # grayscale_cam is generate on batch
                grayscale_cam = grad_cam(inputs, target_category=None, eigen_smooth=False, aug_smooth=False)

                # get a cv2 encoding image from dataloder by inputs[j].cpu().numpy().transpose((1, 2, 0))
                cam_img = show_cam_on_image(inputs[j].cpu().numpy().transpose((1, 2, 0)), grayscale_cam[j])

                plt.imshow(cam_img)
                plt.pause(0.001)  # pause a bit so that plots are updated

            if images_so_far == num_images * 2:
                picpath = draw_path + '/' + pic_name + '.jpg'
                if not os.path.exists(draw_path):
                    os.makedirs(draw_path)

                plt.savefig(picpath, dpi=1000)
                plt.show()

                model.train(mode=was_training)
                if writer is not None:  # attach the pic to the tensorboard backend if avilable
                    image_PIL = Image.open(picpath)
                    img = np.array(image_PIL)
                    writer.add_image(pic_name, img, 1, dataformats='HWC')

                plt.cla()
                plt.close("all")
                return

    model.train(mode=was_training)


def visualize_check(model, dataloader, class_names, check_index=1, num_images=9, device='cpu', pic_name='test',
                    skip_batch=10, draw_path='/home/ZTY/imaging_results', writer=None):  # 预测测试
    """
    check num_images of images and visual them
    output a pic with 3 column and rows of num_images//3

    :param model: model object
    :param dataloader: checking dataloader
    :param class_names: The name of classes for painting
    :param num_images: how many image u want to check, should SMALLER THAN the batchsize
    :param device: cpu/gpu object
    :param skip_batch: number of skip over minibatch
    :param pic_name: name of the output pic
    :param draw_path: path folder for output pic
    :param writer: attach the pic to the tensorboard backend

    :return:  None

    """
    was_training = model.training
    model.eval()

    images_so_far = 0
    plt.figure()

    with torch.no_grad():

        dataloader = iter(dataloader)
        for i in range(check_index * skip_batch):  # skip the tested batchs
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

                if writer is not None:  # attach the pic to the tensorboard backend if avilable
                    image_PIL = Image.open(picpath)
                    img = np.array(image_PIL)
                    writer.add_image(pic_name, img, 1, dataformats='HWC')

                plt.cla()
                plt.close("all")
                return

        model.train(mode=was_training)


# Training Script
def better_performance(temp_acc, temp_vac, best_acc, best_vac):  # determin which epoch have the best model

    if temp_vac >= best_vac and temp_acc >= best_acc:
        return True
    elif temp_vac > best_vac:
        return True
    else:
        return False


def train_model(model, dataloaders, criterion, optimizer, class_names, dataset_sizes, edge_size=384,
                model_idx=None, num_epochs=25, intake_epochs=0, check_minibatch=100, scheduler=None, device=None,
                draw_path='/home/MSHT/results', enable_attention_check=False, enable_visualize_check=False,
                enable_sam=False, writer=None):
    """
    Training iteration

    :param model: model object
    :param dataloaders: 2 dataloader(train and val) dict
    :param criterion: loss func obj
    :param optimizer: optimizer obj

    :param class_names: The name of classes for priting
    :param dataset_sizes: size of datasets
    :param edge_size: image size for the input image

    :param model_idx: model idx for the getting pre-setted model

    :param num_epochs: total training epochs
    :param intake_epochs: number of skip over epochs when choosing the best model
    :param check_minibatch: number of skip over minibatch in calculating the criteria's results etc.
    :param scheduler: scheduler is an LR scheduler object from torch.optim.lr_scheduler.

    :param device: cpu/gpu object
    :param draw_path: path folder for output pic

    :param enable_attention_check: use attention_check to show the pics of models' attention areas
    :param enable_visualize_check: use visualize_check to show the pics
    :param enable_sam: use SAM training strategy

    :param writer: attach the records to the tensorboard backend
    """

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    # for saving the best model state dict
    best_model_wts = copy.deepcopy(model.state_dict())  # deepcopy
    # initial an empty dict
    json_log = {}

    # initial best performance
    best_acc = 0.0
    best_vac = 0.0
    temp_acc = 0.0
    temp_vac = 0.0
    best_epoch_idx = 1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # record json log, initially empty
        json_log[str(epoch + 1)] = {}

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:  # alternatively train/val

            index = 0
            model_time = time.time()

            # initiate the empty log dict
            log_dict = {}
            for cls_idx in range(len(class_names)):
                # only float type is allowed in json
                log_dict[class_names[cls_idx]] = {'tp': 0.0, 'tn': 0.0, 'fp': 0.0, 'fn': 0.0}

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # criterias, initially empty
            running_loss = 0.0
            log_running_loss = 0.0
            running_corrects = 0

            check_dataloaders = copy.deepcopy(dataloaders)

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:  # use different dataloder in different phase
                inputs = inputs.to(device)
                # print('inputs[0]',type(inputs[0]))

                labels = labels.to(device)

                # zero the parameter gradients
                if not enable_sam:
                    optimizer.zero_grad()

                # forward
                # track grad if only in train!
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if enable_sam:
                            loss.backward()
                            # first forward-backward pass
                            optimizer.first_step(zero_grad=True)

                            # second forward-backward pass
                            loss2 = criterion(model(inputs), labels)  # SAM need another model(inputs)
                            loss2.backward()  # make sure to do a full forward pass when using SAM
                            optimizer.second_step(zero_grad=True)
                        else:
                            loss.backward()
                            optimizer.step()

                # log criterias: update
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

                    # log_dict[cls_idx] = {'tp': 0.0, 'tn': 0.0, 'fp': 0.0, 'fn': 0.0}  # notice is float inside
                    log_dict[class_names[cls_idx]]['tp'] += tp
                    log_dict[class_names[cls_idx]]['tn'] += tn
                    log_dict[class_names[cls_idx]]['fp'] += fp
                    log_dict[class_names[cls_idx]]['fn'] += fn

                # attach the records to the tensorboard backend
                if writer is not None:
                    # ...log the running loss
                    writer.add_scalar(phase + ' minibatch loss',
                                      float(loss.item()),
                                      epoch * len(dataloaders[phase]) + index)
                    writer.add_scalar(phase + ' minibatch ACC',
                                      float(torch.sum(preds == labels.data) / inputs.size(0)),
                                      epoch * len(dataloaders[phase]) + index)

                # at the checking time now
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

                    if enable_attention_check:
                        try:
                            check_SAA(model, model_idx, edge_size, check_dataloaders[phase], class_names, check_index,
                                      num_images=1, device=device,
                                      pic_name='GradCAM_' + str(epoch_idx) + '_I_' + str(index + 1),
                                      skip_batch=check_minibatch, draw_path=draw_path, writer=writer)
                        except:
                            print('model:', model_idx, ' with edge_size', edge_size, 'is not supported yet')
                    else:
                        pass

                    model_time = time.time()
                    log_running_loss = 0.0

                index += 1

            if phase == 'train':
                if scheduler is not None:  # lr scheduler: update
                    scheduler.step()

            # log criterias: print
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase] * 100
            print('\nEpoch: {}  {} \nLoss: {:.4f}  Acc: {:.4f}'.format(epoch + 1, phase, epoch_loss, epoch_acc))

            # attach the records to the tensorboard backend
            if writer is not None:
                # ...log the running loss
                writer.add_scalar(phase + ' loss',
                                  float(epoch_loss),
                                  epoch + 1)
                writer.add_scalar(phase + ' ACC',
                                  float(epoch_acc),
                                  epoch + 1)

            # calculating the confusion matrix
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
                # attach the records to the tensorboard backend
                if writer is not None:
                    # ...log the running loss
                    writer.add_scalar(phase + '   ' + class_names[cls_idx] + ' precision',
                                      precision,
                                      epoch + 1)
                    writer.add_scalar(phase + '   ' + class_names[cls_idx] + ' recall',
                                      recall,
                                      epoch + 1)
            # json log: update
            json_log[str(epoch + 1)][phase] = log_dict

            if phase == 'val':
                temp_vac = epoch_acc
            else:
                temp_acc = epoch_acc  # not useful actually

            # deep copy the model
            if phase == 'val' and better_performance(temp_acc, temp_vac, best_acc, best_vac) and epoch >= intake_epochs:
                # TODO what is better? we now use the wildly used method only
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

    # attach the records to the tensorboard backend
    if writer is not None:
        writer.close()

    # load best model weights as final model training result
    model.load_state_dict(best_model_wts)
    # save json_log  indent=2 for better view
    json.dump(json_log, open(os.path.join(draw_path, model_idx + '_log.json'), 'w'), ensure_ascii=False, indent=2)
    return model


def main(args):
    if args.paint:
        # use Agg kernal, not painting in the front-desk
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
    model_idx = args.model_idx  # the model we are going to use. by the format of Model_size_other_info
    # structural parameter
    drop_rate = args.drop_rate
    attn_drop_rate = args.attn_drop_rate
    drop_path_rate = args.drop_path_rate
    use_cls_token = False if args.cls_token_off else True
    use_pos_embedding = False if args.pos_embedding_off else True
    use_att_module = None if args.att_module == 'None' else args.att_module

    # pretrained_backbone
    pretrained_backbone = False if args.backbone_PT_off else True

    # classification required number of your dataset
    num_classes = args.num_classes  # 2
    # image size for the input image
    edge_size = args.edge_size  # 224 384 1000

    # batch info
    batch_size = args.batch_size  # 8
    num_workers = args.num_workers  # main training num_workers 4

    num_epochs = args.num_epochs  # 50
    intake_epochs = args.intake_epochs  # 0
    check_minibatch = args.check_minibatch if args.check_minibatch is not None else 400 // batch_size

    lr = args.lr  # 0.000007
    lrf = args.lrf  # 0.0

    opt_name = args.opt_name  # 'Adam'

    # PATH info
    draw_root = args.draw_root
    model_path = args.model_path
    dataroot = args.dataroot

    if enable_notify:  # use notifyemail to send the record to somewhere
        import notifyemail as notify

        notify.Reboost(mail_host='smtp.163.com', mail_user='tum9598@163.com', mail_pass='EXVGQACCXPUIUQAE',
                       default_reciving_list=['904639643@qq.com', 'pytorch1225@163.com'],
                       log_root_path='log', max_log_cnt=5)

        if enable_tensorboard:
            notify.add_text('update to the tensorboard')
        else:
            notify.add_text('not update to the tensorboard')

        notify.add_text('  ')

        notify.add_text('model idx ' + str(model_idx))
        notify.add_text('  ')

        notify.add_text('GPU idx: ' + str(gpu_idx))
        notify.add_text('  ')

        notify.add_text('cls number ' + str(num_classes))
        notify.add_text('edge size ' + str(edge_size))
        notify.add_text('batch_size ' + str(batch_size))
        notify.add_text('num_epochs ' + str(num_epochs))
        notify.add_text('lr ' + str(lr))
        notify.add_text('opt_name ' + str(opt_name))
        notify.add_text('enable_sam ' + str(enable_sam))
        notify.send_log()

    print("*********************************{}*************************************".format('setting'))
    print(args)

    draw_path = os.path.join(draw_root, 'PC_' + model_idx)  # PC is for the pancreatic cancer
    save_model_path = os.path.join(model_path, 'PC_' + model_idx + '.pth')

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if os.path.exists(draw_path):
        del_file(draw_path)  # clear the output folder, NOTICE this may be DANGEROUS
    else:
        os.makedirs(draw_path)

    # start tensorboard backend
    if enable_tensorboard:
        writer = SummaryWriter(draw_path)
    else:
        writer = None
    # if u run locally
    # nohup tensorboard --logdir=/home/MSHT/runs --host=0.0.0.0 --port=7777 &
    # tensorboard --logdir=/home/ZTY/runs --host=0.0.0.0 --port=7777

    # Data Augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation((0, 180)),
            transforms.CenterCrop(700),  # center area for classification
            transforms.Resize(edge_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
            # HSL shift operation
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(700),
            transforms.Resize(edge_size),
            transforms.ToTensor()
        ]),
    }

    datasets = {x: torchvision.datasets.ImageFolder(os.path.join(dataroot, x), data_transforms[x]) for x in
                ['train', 'val']}  # 2 dataset obj is prepared here and combine together

    dataloaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True,
                                                        num_workers=num_workers),  # colab suggest 2
                   'val': torch.utils.data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False,
                                                      num_workers=num_workers // 4 + 1)
                   }

    class_names = ['negative', 'positive'][0:num_classes]  # A G E B
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}  # size of each dataset

    if gpu_idx == -1:  # use all cards
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            gpu_use = gpu_idx
        else:
            print('we dont have more GPU idx here, try to use gpu_idx=0')
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # setting k for: only card idx k is sighted for this code
                gpu_use = 0
            except:
                print("GPU distributing ERRO occur use CPU instead")

    else:
        # Decide which device we want to run on
        try:
            # setting k for: only card idx k is sighted for this code
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
            gpu_use = gpu_idx
        except:
            print('we dont have that GPU idx here, try to use gpu_idx=0')
            try:
                # setting 0 for: only card idx 0 is sighted for this code
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                gpu_use = 0
            except:
                print("GPU distributing ERRO occur use CPU instead")

    # device enviorment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get model
    if Pre_Trained_model_path is not None:
        if os.path.exists(Pre_Trained_model_path):
            pretrain_model = get_model(1000, edge_size, model_idx, drop_rate, attn_drop_rate, drop_path_rate,
                                       False, use_cls_token, use_pos_embedding, use_att_module)
            pretrain_model.load_state_dict(torch.load(Pre_Trained_model_path), False)
            # Fixme: there is a bug, why strict=False cannot work in(when) setting the cls number to others
            num_features = pretrain_model.num_features
            pretrain_model.head = nn.Linear(num_features, num_classes)
            model = pretrain_model
            print('pretrain model loaded')

        else:
            print('Pre_Trained_model_path:' + Pre_Trained_model_path, ' is NOT avaliable!!!!\n')
            print('we ignore this with a new start up')
    else:
        # get model: randomly initiate model, except the backbone CNN(when pretrained_backbone is True)
        model = get_model(num_classes, edge_size, model_idx, drop_rate, attn_drop_rate, drop_path_rate,
                          pretrained_backbone, use_cls_token, use_pos_embedding, use_att_module)

    print('GPU:', gpu_use)

    if gpu_use == -1:
        model = nn.DataParallel(model)

    model.to(device)

    try:
        summary(model, input_size=(3, edge_size, edge_size))  # should be after .to(device)
    except:
        pass

    print("model :", model_idx)

    criterion = nn.CrossEntropyLoss()

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
        # cosine Scheduler by https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * (1 - lrf) + lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # train
    model_ft = train_model(model, dataloaders, criterion, optimizer, class_names, dataset_sizes,
                           edge_size=edge_size, model_idx=model_idx, num_epochs=num_epochs,
                           intake_epochs=intake_epochs, check_minibatch=check_minibatch,
                           scheduler=scheduler, device=device, draw_path=draw_path,
                           enable_attention_check=enable_attention_check,
                           enable_visualize_check=enable_visualize_check, enable_sam=enable_sam, writer=writer)
    # save model if its a multi-GPU model, save as a single GPU one too
    if gpu_use == -1:
        torch.save(model_ft.module.state_dict(), save_model_path)
        print('model trained by multi-GPUs has its single GPU copy saved at ', save_model_path)
    else:
        torch.save(model_ft.state_dict(), save_model_path)
        print('model trained by GPU (idx:' + str(gpu_use) + ') has been saved at ', save_model_path)


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Model Name or index
    parser.add_argument('--model_idx', default='Hybrid2_384_401_testsample', type=str, help='Model Name or index')
    # drop_rate, attn_drop_rate, drop_path_rate
    parser.add_argument('--drop_rate', default=0.0, type=float, help='dropout rate , default 0.0')
    parser.add_argument('--attn_drop_rate', default=0.0, type=float, help='dropout rate Aftter Attention, default 0.0')
    parser.add_argument('--drop_path_rate', default=0.0, type=float, help='drop path for stochastic depth, default 0.0')

    # Abalation Studies
    parser.add_argument('--cls_token_off', action='store_true', help='use cls_token in model structure')
    parser.add_argument('--pos_embedding_off', action='store_true', help='use pos_embedding in model structure')
    # 'SimAM', 'CBAM', 'SE' 'None'
    parser.add_argument('--att_module', default='SimAM', type=str, help='use which att_module in model structure')

    # backbone_PT_off  by default is false, in default setting the backbone weight is required
    parser.add_argument('--backbone_PT_off', action='store_true', help='use a freash backbone weight in training')

    # Enviroment parameters
    parser.add_argument('--gpu_idx', default=-1, type=int,
                        help='use a single GPU with its index, -1 to use multiple GPU')

    # Path parameters
    parser.add_argument('--dataroot', default='/data/pancreatic-cancer-project/ZTY_dataset2',
                        help='path to dataset')
    parser.add_argument('--model_path', default='/home/pancreatic-cancer-project/saved_models',
                        help='path to save model state-dict')
    parser.add_argument('--draw_root', default='/home/pancreatic-cancer-project/runs',
                        help='path to draw and save tensorboard output')

    # Help tool parameters
    parser.add_argument('--paint', action='store_false', help='paint in front desk')  # matplotlib.use('Agg')
    parser.add_argument('--enable_notify', action='store_true', help='enable notify to send email')
    # check tool parameters
    parser.add_argument('--enable_tensorboard', action='store_true', help='enable tensorboard to save status')

    parser.add_argument('--enable_attention_check', action='store_true', help='check and save attention map')
    parser.add_argument('--enable_visualize_check', action='store_true', help='check and save pics')

    # Training status parameters
    parser.add_argument('--enable_sam', action='store_true', help='use SAM strategy in training')
    # '/home/ZTY/pancreatic-cancer-project/saved_models/PC_Hybrid2_384_PreTrain_000.pth'
    parser.add_argument('--Pre_Trained_model_path', default=None, type=str,
                        help='Finetuning a trained model in this dataset')

    # Dataset based parameters
    parser.add_argument('--num_classes', default=2, type=int, help='classification number')
    parser.add_argument('--edge_size', default=384, type=int, help='edge size of input image')  # 224 256 384 1000
    parser.add_argument('--num_workers', default=2, type=int, help='use CPU num_workers , default 2 for colab')

    # Training seting parameters
    parser.add_argument('--batch_size', default=8, type=int, help='Training batch_size default 8')

    # check_minibatch for painting pics
    parser.add_argument('--check_minibatch', default=None, type=int, help='check batch_size')

    parser.add_argument('--num_epochs', default=50, type=int, help='training epochs')
    parser.add_argument('--intake_epochs', default=0, type=int, help='only save model at epochs after intake_epochs')
    parser.add_argument('--lr', default=0.00001, type=float, help='learing rate')
    parser.add_argument('--lrf', type=float, default=0.0,
                        help='learing rate decay rate, default 0(not enabled), suggest 0.1 and lr=0.00005')
    parser.add_argument('--opt_name', default='Adam', type=str, help='optimizer name Adam or SGD')

    return parser


if __name__ == '__main__':
    # setting up the random seed
    setup_seed(517)

    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
