"""
测试模型训练效果  版本： 9月21日 15：17
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os
import shutil
from tensorboardX import SummaryWriter
from PIL import Image
import argparse
import json
import Train


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


def visualize_check(model, dataloader, class_names, check_index=1, num_images=9, device='cpu', skip_batch=10,
                    pic_name='test', draw_path='/home/ZTY/imaging_results', writer=None):  # 预测测试
    '''
    对num_images个图片进行检查，每行放3个图
    :param model:输入模型
    :param dataloader:输入数据dataloader
    :param class_names:分类的类别名字
    :param num_images:需要检验的原图数量,此数量需要小于batchsize
    :param device:cpu/gpu
    :param skip_batch:跳过多少个minibatcht检查一次
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

                # plt.savefig(picpath)
                # plt.imshow(np.transpose(img_list[-1],(1,2,0)))
                # plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
                plt.show()
                plt.savefig(picpath, dpi=1000)

                model.train(mode=was_training)

                if writer is not None:  # 用这个方式读取保存的图片到tensorboard上面
                    image_PIL = Image.open(picpath)
                    img = np.array(image_PIL)
                    writer.add_image(pic_name, img, 1, dataformats='HWC')

                return

        model.train(mode=was_training)


def check_grad_CAM(model, dataloader, class_names, check_index=1, num_images=3, device='cpu', skip_batch=10,
                   pic_name='test', draw_path='/home/ZTY/imaging_results', writer=None):
    """
    检查num_images个图片在每个类别上的cam，每行有每个类别的图，行数=num_images，为检查的图片数量
    :param model:输入模型
    :param dataloader:输入数据dataloader
    :param class_names:分类的类别名字
    :param num_images:需要检验的原图数量,此数量需要小于batchsize
    :param device:cpu/gpu
    :param skip_batch:跳过多少个minibatcht检查一次
    :param pic_name:输出图片的名字
    :param draw_path:输出图片的文件夹
    :param writer:输出图片上传到tensorboard服务器
    :return:
    """
    from utils import grad_CAM

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
    layer_name = grad_CAM.get_last_conv_name(model)
    grad_cam = grad_CAM.GradCAM(model, layer_name)  # 生成grad cam调取器，包括注册hook等

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
            cam, heatmap = grad_CAM.gen_cam(check_image, mask)

            plt.imshow(cam)
            plt.pause(0.001)  # pause a bit so that plots are updated

            if images_so_far == num_images * len(class_names):
                picpath = draw_path + '/' + pic_name + '.jpg'
                if not os.path.exists(draw_path):
                    os.makedirs(draw_path)

                plt.show()
                plt.savefig(picpath, dpi=1000)

                grad_cam.remove_handlers()  # 删除注册的hook
                model.train(mode=was_training)

                if writer is not None:  # 用这个方式读取保存的图片到tensorboard上面
                    image_PIL = Image.open(picpath)
                    img = np.array(image_PIL)
                    writer.add_image(pic_name, img, 1, dataformats='HWC')

                return

    grad_cam.remove_handlers()  # 删除注册的hook
    model.train(mode=was_training)


def check_SAA(model, dataloader, class_names, check_index=1, num_images=3, device='cpu', skip_batch=10,
              pic_name='test', draw_path='/home/ZTY/imaging_results', writer=None):
    '''
    TODO
    还在施工！！！！！！！！！！
    检查num_images个图片在每个类别上的self-attention activation，每行有每个类别的图，行数=num_images，为检查的图片数量
    :param model:输入模型
    :param dataloader:输入数据dataloader
    :param class_names:分类的类别名字
    :param num_images:需要检验的原图数量,此数量需要小于batchsize
    :param device:cpu/gpu
    :param skip_batch:跳过多少个minibatcht检查一次
    :param pic_name:输出图片的名字
    :param draw_path:输出图片的文件夹
    :param writer:输出图片上传到tensorboard服务器
    :return:
    '''
    # Get a batch of training data
    dataloader = iter(dataloader)
    for i in range(check_index * skip_batch):
        inputs, classes = next(dataloader)

    # 预测测试
    was_training = model.training
    model.eval()

    from vit_pytorch.recorder import Recorder
    v = Recorder(model)

    inputs = inputs.to(device)
    labels = classes.to(device)

    preds, attns = v(inputs)
    # attns is (1, 6, 16, 65, 65) - (batch x layers x heads x patch x patch)
    print(preds.shape)
    print(attns.shape)

    '''
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
            plt.imshow(cam)
            plt.pause(0.001)  # pause a bit so that plots are updated
            if images_so_far == num_images * len(class_names):
                picpath = draw_path + '/' + pic_name + '.jpg'
                if not os.path.exists(draw_path):
                    os.makedirs(draw_path)
                plt.show()
                plt.savefig(picpath, dpi=1000)
                model = v.eject()  # 删除注册的hook
                model.train(mode=was_training)
                if writer is not None:  # 用这个方式读取保存的图片到tensorboard上面
                    image_PIL = Image.open(picpath)
                    img = np.array(image_PIL)
                    writer.add_image(pic_name, img, 1, dataformats='HWC')
                return
    '''

    model = v.eject()  # 删除注册的hook
    model.train(mode=was_training)


def test_model(model, test_dataloader, criterion, class_names, test_dataset_size, test_model_idx, check_minibatch=100,
               device=None, draw_path='/home/ZTY/imaging_results', enable_attention_check=None,
               enable_visualize_check=True,
               writer=None):
    # scheduler is an LR scheduler object from torch.optim.lr_scheduler.
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    print('Epoch: Test')
    print('-' * 10)

    phase = 'test'
    index = 0
    model_time = time.time()

    # 初始化log字典
    json_log = {}
    json_log['test'] = {}

    # 初始化计数字典
    log_dict = {}
    for cls_idx in range(len(class_names)):
        log_dict[class_names[cls_idx]] = {'tp': 0.0, 'tn': 0.0, 'fp': 0.0, 'fn': 0.0}

    model.eval()  # Set model to evaluate mode

    # 初始化记录表现
    running_loss = 0.0
    log_running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in test_dataloader:  # 不同任务段用不同dataloader的数据
        inputs = inputs.to(device)
        # print('inputs[0]',type(inputs[0]))

        labels = labels.to(device)

        # zero the parameter gradients训练的话才需要，测试不需要
        # optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)  # preds是最大值出现的位置，相当于是类别id
        loss = criterion(outputs, labels)  # loss是基于输出的vector与onehot label做loss

        # 统计表现总和
        log_running_loss += loss.item()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        # Compute recision and recall for each class.
        for cls_idx in range(len(class_names)):
            tp = np.dot((labels.cpu().data == cls_idx).numpy().astype(int),
                        (preds == cls_idx).cpu().numpy().astype(int))
            tn = np.dot((labels.cpu().data != cls_idx).numpy().astype(int),
                        (preds != cls_idx).cpu().numpy().astype(int))

            fp = np.sum((preds == cls_idx).cpu().numpy()) - tp

            fn = np.sum((labels.cpu().data == cls_idx).numpy()) - tp

            # log_dict[cls_idx] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
            log_dict[class_names[cls_idx]]['tp'] += tp
            log_dict[class_names[cls_idx]]['tn'] += tn
            log_dict[class_names[cls_idx]]['fp'] += fp
            log_dict[class_names[cls_idx]]['fn'] += fn

        # 记录内容给tensorboard
        if writer is not None:
            # ...log the running loss
            writer.add_scalar(phase + ' minibatch loss',
                              float(loss.item()),
                              index)
            writer.add_scalar(phase + ' minibatch ACC',
                              float(torch.sum(preds == labels.data) / inputs.size(0)),
                              index)

        # 画图检测效果
        if index % check_minibatch == check_minibatch - 1:
            model_time = time.time() - model_time

            check_index = index // check_minibatch + 1

            epoch_idx = 'test'
            print('Epoch:', epoch_idx, '   ', phase, 'index of ' + str(check_minibatch) + ' minibatch:',
                  check_index, '     time used:', model_time)

            print('minibatch AVG loss:', float(log_running_loss) / check_minibatch)

            if enable_attention_check == 'CAM':
                check_grad_CAM(model, test_dataloader, class_names, check_index, num_images=2, device=device,
                               pic_name='GradCAM_' + str(epoch_idx) + '_I_' + str(index + 1),
                               skip_batch=check_minibatch, draw_path=draw_path, writer=writer)

            elif enable_attention_check == 'SAA':
                check_SAA(model, test_dataloader, class_names, check_index, num_images=2, device=device,
                          pic_name='GradCAM_' + str(epoch_idx) + '_I_' + str(index + 1),
                          skip_batch=check_minibatch, draw_path=draw_path, writer=writer)
                break  # 调试专用终止符，正式时候需要去掉bresk TODO

            else:
                pass

            if enable_visualize_check:
                visualize_check(model, test_dataloader, class_names, check_index, num_images=9, device=device,
                                pic_name='Visual_' + str(epoch_idx) + '_I_' + str(index + 1),
                                skip_batch=check_minibatch, draw_path=draw_path, writer=writer)

            model_time = time.time()
            log_running_loss = 0.0

        index += 1

    # 记录输出本轮情况
    epoch_loss = running_loss / test_dataset_size
    epoch_acc = running_corrects.double() / test_dataset_size * 100
    print('\nEpoch:  {} \nLoss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

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

        # 记录实验json log
        json_log['test'][phase] = log_dict

    print('\n')

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # 记录内容给tensorboard
    if writer is not None:
        writer.close()

    # 保存json_log  indent=2 更加美观
    json.dump(json_log, open(os.path.join(draw_path, test_model_idx + '_log.json'), 'w'), ensure_ascii=False, indent=2)

    return model


def main(args):
    if args.paint:
        # 使用Agg模式，不在本地画图
        import matplotlib
        matplotlib.use('Agg')

    gpu_idx = args.gpu_idx  # GPU idx start with0, -1 to use multipel GPU

    enable_notify = args.enable_notify  # False
    enable_tensorboard = args.enable_tensorboard  # False
    # Vision Transformer enable_attention_check = 'SAA'
    # 纯Transformer结构的attention我还没做好，CNN/Hybrid使用 enable_attention_check = 'CAM'
    enable_attention_check = args.enable_attention_check  # False   'CAM' 'SAA'
    enable_visualize_check = args.enable_visualize_check  # False

    model_idx = args.model_idx  # 'Hybrid2_384_507_b8'  # 'ViT_384_505_b8'

    # 路径 配置
    draw_root = args.draw_root  # r'C:\Users\admin\Desktop\runs'
    model_path = args.model_path  # r'C:\Users\admin\Desktop\saved_models'
    dataroot = args.dataroot  # r'C:\Users\admin\Desktop\ZTY_dataset1'

    # 服务器
    # draw_path = '/home/ZTY/runs'
    # model_path = '/home/ZTY/saved_models'
    test_model_idx = 'PC_' + model_idx + '_test'
    draw_path = os.path.join(draw_root, test_model_idx)
    save_model_path = os.path.join(model_path, 'PC_' + model_idx + '.pth')
    # 验证数据
    test_dataroot = os.path.join(dataroot, 'test')

    # 任务类别数量
    num_classes = args.num_classes
    class_names = ['negative', 'positive'][0:num_classes]  # A G E B
    # 边长
    edge_size = args.edge_size  # 1000 224 384

    # 设置
    batch_size = args.batch_size  # 10
    criterion = nn.CrossEntropyLoss()

    if enable_notify:
        import notifyemail as notify

        notify.Reboost(mail_host='smtp.163.com', mail_user='tum9598@163.com', mail_pass='EXVGQACCXPUIUQAE',
                       default_reciving_list=['904639643@qq.com', 'pytorch1225@163.com'],
                       log_root_path='log', max_log_cnt=5)

        if enable_tensorboard:
            notify.add_text('进行PC验证测试 模型编号：' + str(model_idx) + '.  上传到tensorboard')
        else:
            notify.add_text('进行PC验证测试 模型编号：' + str(model_idx) + '.  不上传到tensorboard')
        notify.add_text('边长 edge_size =' + str(edge_size))
        notify.add_text('测试 batch_size =' + str(batch_size))
        notify.send_log()

    # 模型
    model = Train.get_model(num_classes, edge_size, model_idx)

    try:
        model.load_state_dict(torch.load(save_model_path))
        print("model loaded")
        print("model :", model_idx)
    except:
        try:
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(save_model_path), False)
            print("DataParallel model loaded")
        except:
            print("model loading erro!!")
            return -1

    if gpu_idx == -1:
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
        else:
            print('we dont have more GPU idx here, try to use gpu_idx=0')
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只让程序看到物理卡号为gpu_idx的卡（注意：no标号从0开始）
            except:
                print("GPU distributing ERRO occur use CPU instead")

    else:
        # Decide which device we want to run on
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)  # 只让程序看到物理卡号为gpu_idx的卡（注意：no标号从0开始）
        except:
            print('we dont have that GPU idx here, try to use gpu_idx=0')
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只让程序看到物理卡号为gpu_idx的卡（注意：no标号从0开始）
            except:
                print("GPU distributing ERRO occur use CPU instead")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 单卡验证

    # 检查跳过
    check_minibatch = 80 // batch_size

    model.to(device)

    if os.path.exists(draw_path):
        del_file(draw_path)  # 每次开始的时候都先清空一次
    else:
        os.makedirs(draw_path)

    # 调取tensorboard服务器
    if enable_tensorboard:
        writer = SummaryWriter(draw_path)
    else:
        writer = None

    # tensorboard --logdir=C:\Users\admin\Desktop\runs --host=192.168.1.139 --port=7777
    # nohup tensorboard --logdir=/home/ZTY/runs --host=10.201.10.16 --port=7777 &

    print("*********************************{}*************************************".format('setting'))
    print(args)

    # 加载数据
    '''
    # 之前的数据增强
    data_transforms = {
        'train': transforms.Compose([  # 建议先做像素级别的变化，之后做空间级别的变化
            transforms.RandomRotation((0, 180)),
            transforms.Resize(edge_size),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(edge_size),
            transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
            # 色相饱和度对比度明度的相关的处理H S L，随即灰度化
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Resize(edge_size),
            transforms.CenterCrop(edge_size),
            transforms.ToTensor()
        ]),
    }
    '''
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
    # 注意，这里的预处理参数是按照预训练的配置来设置的，这样迁移学习才能有更明显的效果
    # 因此之后可以在tensor这里最后试一下加上 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    test_datasets = torchvision.datasets.ImageFolder(test_dataroot, data_transforms['val'])  # test数据使用val的数据transform
    test_dataset_size = len(test_datasets)
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False, num_workers=1)

    test_model(model, test_dataloader, criterion, class_names, test_dataset_size, test_model_idx=test_model_idx,
               check_minibatch=check_minibatch, device=device, draw_path=draw_path,
               enable_attention_check=enable_attention_check,
               enable_visualize_check=enable_visualize_check, writer=writer)


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Model Name or index
    parser.add_argument('--model_idx', default='Hybrid2_384_507_b8_e50_2', type=str, help='Model Name or index')

    # Enviroment parameters
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='use a single GPU with its index, -1 to use multiple GPU')

    # Path parameters
    parser.add_argument('--dataroot', default=r'/data/pancreatic-cancer-diagnosis-tansformer/ZTY_dataset2',
                        help='path to dataset')
    parser.add_argument('--model_path', default=r'/home/pancreatic-cancer-diagnosis-tansformer/saved_models',
                        help='path to save model state-dict')
    parser.add_argument('--draw_root', default=r'/home/pancreatic-cancer-diagnosis-tansformer/runs',
                        help='path to draw and save tensorboard output')

    # Help tool parameters
    parser.add_argument('--paint', action='store_true', help='paint in front desk')  # matplotlib.use('Agg')
    parser.add_argument('--enable_notify', action='store_true', help='enable notify to send email')
    # check tool parameters
    parser.add_argument('--enable_tensorboard', action='store_true', help='enable tensorboard to save status')
    # enable_attention_check = False  # 'CAM' 'SAA'
    parser.add_argument('--enable_attention_check', default=None, type=str, help='check and save attention map')
    parser.add_argument('--enable_visualize_check', action='store_true', help='check and save pics')

    # Dataset based parameters
    parser.add_argument('--num_classes', default=2, type=int, help='classification number')
    parser.add_argument('--edge_size', default=384, type=int, help='edge size of input image')  # 224 256 384 1000
    # Training seting parameters
    parser.add_argument('--batch_size', default=1, type=int, help='check batch_size')

    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
