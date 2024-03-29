"""
attention visulization config  ver： Oct 18th 18：30
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


def softmax(x):
    """Compute the softmax in a numerically stable way."""
    sof = nn.Softmax()
    return sof(x)


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


# Grad CAM part：Visualize of CNN+Transformer attention area
def cls_token_s12_transform(tensor, height=12, width=12):  # based on pytorch_grad_cam
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def cls_token_s14_transform(tensor, height=14, width=14):  # based on pytorch_grad_cam
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


def swinT_transform_224(tensor, height=7, width=7):  # 224 7
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def swinT_transform_384(tensor, height=12, width=12):  # 384 12
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def choose_cam_by_model(model, model_idx, edge_size, use_cuda=True, MSHT_CAM_check='decoder_4'):
    """
    :param model: model object
    :param model_idx: model idx for the getting pre-setted layer and size
    :param edge_size: image size for the getting pre-setted layer and size
    """
    from pytorch_grad_cam import GradCAM

    # reshape_transform  todo conformer 224！！
    # check class: target_category = None
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.

    if model_idx[0:3] == 'ViT' or model_idx[0:4] == 'deit':
        target_layers = [model.blocks[-1].norm1]
        if edge_size == 384:
            grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                               reshape_transform=cls_token_s24_transform)
        elif edge_size == 224:
            grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                               reshape_transform=cls_token_s14_transform)
        else:
            print('ERRO in ViT/DeiT edge size')
            return -1

    elif model_idx[0:3] == 'vgg':
        target_layers = [model.features[-1]]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda, reshape_transform=None)

    elif model_idx[0:6] == 'swin_b':
        target_layers = [model.layers[-1].blocks[-1].norm1]
        if edge_size == 384:
            grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                               reshape_transform=swinT_transform_384)
        elif edge_size == 224:
            grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                               reshape_transform=swinT_transform_224)
        else:
            print('ERRO in Swin Transformer edge size')
            return -1

    elif model_idx[0:6] == 'ResNet':
        target_layers = [model.layer4[-1]]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda, reshape_transform=None)  # CNN: None

    elif model_idx[0:7] == 'Hybrid1' and edge_size == 384:
        target_layers = [model.blocks[-1].norm1]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                           reshape_transform=cls_token_s12_transform)

    elif model_idx[0:7] == 'Hybrid2' and edge_size == 384:
        if MSHT_CAM_check == 'decoder_1':
            target_layers = [model.dec1.norm1]

            if 'CLS' in model_idx.split('_') and 'No' in model_idx.split('_'):
                grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                                   reshape_transform=no_cls_token_s12_transform)

            else:
                grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                                   reshape_transform=cls_token_s12_transform)

        elif MSHT_CAM_check == 'decoder_2':
            target_layers = [model.dec2.norm1]

            if 'CLS' in model_idx.split('_') and 'No' in model_idx.split('_'):
                grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                                   reshape_transform=no_cls_token_s12_transform)

            else:
                grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                                   reshape_transform=cls_token_s12_transform)

        elif MSHT_CAM_check == 'decoder_3':
            target_layers = [model.dec3.norm1]

            if 'CLS' in model_idx.split('_') and 'No' in model_idx.split('_'):
                grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                                   reshape_transform=no_cls_token_s12_transform)

            else:
                grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                                   reshape_transform=cls_token_s12_transform)

        elif MSHT_CAM_check == 'decoder_4':
            target_layers = [model.dec4.norm1]

            if 'CLS' in model_idx.split('_') and 'No' in model_idx.split('_'):
                grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                                   reshape_transform=no_cls_token_s12_transform)

            else:
                grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                                   reshape_transform=cls_token_s12_transform)

        elif MSHT_CAM_check == 'encoder_1':
            target_layers = [model.backbone.layer1[-1]]
            grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                               reshape_transform=None)

        elif MSHT_CAM_check == 'encoder_2':
            target_layers = [model.backbone.layer2[-1]]
            grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                               reshape_transform=None)

        elif MSHT_CAM_check == 'encoder_3':
            target_layers = [model.backbone.layer3[-1]]
            grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                               reshape_transform=None)

        elif MSHT_CAM_check == 'encoder_4':
            target_layers = [model.backbone.layer4[-1]]
            grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                               reshape_transform=None)
        else:
            print('ERRO in MSHT_CAM_check')
            return -1

    elif model_idx[0:7] == 'Hybrid3' and edge_size == 384:
        target_layers = [model.dec3.norm1]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                           reshape_transform=cls_token_s24_transform)

    elif model_idx[0:9] == 'mobilenet':
        target_layers = [model.blocks[-1]]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda, reshape_transform=None)  # CNN: None

    elif model_idx[0:10] == 'ResN50_ViT' and edge_size == 384:
        target_layers = [model.blocks[-1].norm1]  # model.layer4[-1]
        if edge_size == 384:
            grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                               reshape_transform=cls_token_s24_transform)
        elif edge_size == 224:
            grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                               reshape_transform=cls_token_s14_transform)
        else:
            print('ERRO in ResN50_ViT edge size')
            return -1

    elif model_idx[0:12] == 'efficientnet':
        target_layers = [model.conv_head]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda, reshape_transform=None)  # CNN: None


    else:
        print('ERRO in model_idx')
        return -1

    return grad_cam


def check_SAA(model, model_idx, edge_size, dataloader, class_names, check_index=1, num_images=2, device='cpu',
              skip_batch=10, pic_name='test', draw_path='../imaging_results', check_all=True,
              MSHT_CAM_check='decoder_4', writer=None):
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
    :param check_all: choose the type of checking CAM : by default False to be only on the predicted type'
                    True to be on all types

    :param MSHT_CAM_check: which layer's attention you want to see with MSHT ? default: decoder_4
                          4 encoders and 4 decoders are ok to check (encoder_1 to decoder_4)

    :param writer: attach the pic to the tensorboard backend

    :return: None
    """
    from pytorch_grad_cam.utils import show_cam_on_image

    # Get a batch of training data (using these iter method is for other skill and funcs)
    dataloader = iter(dataloader)
    for i in range(check_index * skip_batch):  # skip the tested batchs
        inputs, classes = next(dataloader)

    # choose checking type: false to be only on the predicted type'; true to be on all types
    if check_all:
        checking_type = ['ori', ]
        checking_type.extend([cls for cls in range(len(class_names))])
    else:
        checking_type = ['ori', 'tar']

    # test model
    was_training = model.training
    model.eval()

    inputs = inputs.to(device)
    labels = classes.to(device)

    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    grad_cam = choose_cam_by_model(model, model_idx, edge_size, MSHT_CAM_check=MSHT_CAM_check)  # choose model

    images_so_far = 0
    plt.figure()

    for j in range(inputs.size()[0]):

        for type in checking_type:
            images_so_far += 1
            if type == 'ori':
                ax = plt.subplot(num_images, len(checking_type), images_so_far)
                ax.axis('off')
                ax.set_title('Ground Truth:{}'.format(class_names[int(labels[j])]))
                imshow(inputs.cpu().data[j])
                plt.pause(0.001)  # pause a bit so that plots are updated

            else:
                ax = plt.subplot(num_images, len(checking_type), images_so_far)
                ax.axis('off')
                if type == 'tar':
                    ax.set_title('Predict: {}'.format(class_names[preds[j]]))
                    # focus on the specific target class to create grayscale_cam
                    # grayscale_cam is generate on batch
                    grayscale_cam = grad_cam(inputs, target_category=None, eigen_smooth=False, aug_smooth=False)
                else:
                    # pseudo confidence by softmax
                    ax.set_title('{:.1%} {}'.format(softmax(outputs[j])[int(type)], class_names[int(type)]))
                    # focus on the specific target class to create grayscale_cam
                    # grayscale_cam is generate on batch
                    grayscale_cam = grad_cam(inputs, target_category=int(type), eigen_smooth=False, aug_smooth=False)

                # get a cv2 encoding image from dataloder by inputs[j].cpu().numpy().transpose((1, 2, 0))
                cam_img = show_cam_on_image(inputs[j].cpu().numpy().transpose((1, 2, 0)), grayscale_cam[j])

                plt.imshow(cam_img)
                plt.pause(0.001)  # pause a bit so that plots are updated

            if images_so_far == num_images * len(checking_type):  # complete when the pics is enough
                picpath = os.path.join(draw_path, pic_name + '.jpg')
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
                    skip_batch=10, draw_path='/home/ZTY/imaging_results', writer=None):  # visual check
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
