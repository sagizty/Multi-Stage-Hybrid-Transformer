"""
attention visulization config  ver： Nov 1st 20：00 official release
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


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


def swinT_transform_384(tensor, height=12, width=12):  # 224 7
    result = tensor.reshape(tensor.size(0),height, width, tensor.size(2))

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

    elif model_idx[0:3] == 'vgg':
        target_layers = [model.features[-1]]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda, reshape_transform=None)

    elif model_idx[0:4] == 'deit' and edge_size == 384:
        target_layers = [model.blocks[-1].norm1]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                           reshape_transform=cls_token_s24_transform)

    elif model_idx[0:6] == 'swin_b' and edge_size == 384:
        target_layers = [model.layers[-1].blocks[-1].norm1]  # model.layer4[-1]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                           reshape_transform=swinT_transform_384)

    elif model_idx[0:7] == 'Hybrid1' and edge_size == 384:
        target_layers = [model.blocks[-1].norm1]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                           reshape_transform=cls_token_s12_transform)

    elif model_idx[0:10] == 'ResN50_ViT' and edge_size == 384:
        target_layers = [model.blocks[-1].norm1]  # model.layer4[-1]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                           reshape_transform=cls_token_s24_transform)

    elif model_idx[0:6] == 'ResNet':
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

    grad_cam = choose_cam_by_model(model, model_idx, edge_size)  # choose model

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