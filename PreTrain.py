"""
imagenet Pre-Training script  ver： DEC 20st 14:00 official release

model: Hybrid2_384_PreTrain
dataset: ImageNet-1k
"""

from __future__ import print_function, division
import os
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
import time

import copy
import shutil
from tensorboardX import SummaryWriter

from utils.visual_usage import *
from utils.tools import setup_seed, del_file
from Hybrid.getmodel import get_model


# Training Script
def better_performance(temp_acc, temp_vac, best_acc, best_vac):  # determin which epoch have the best model

    if temp_vac >= best_vac and temp_acc >= best_acc:
        return True
    elif temp_vac > best_vac:
        return True
    else:
        return False


def train_model_PT(model, dataloaders, criterion, optimizer, class_names, dataset_sizes, num_epochs=300,
                   intake_epochs=0, check_minibatch=10, scheduler=None, device=None,
                   draw_path='../imaging_results',
                   model_idx=None, gpu_use=-1, checkpoint_gap=0, checkpoint_path=None, load_checkpoint_path=None,
                   enable_attention_check=False, enable_visualize_check=False, enable_sam=False, writer=None):
    # scheduler is an LR scheduler object from torch.optim.lr_scheduler.
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # save_checkpoint status
    if checkpoint_gap > 0 and checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        if checkpoint_gap > 0 and os.path.exists(checkpoint_path):
            save_checkpoint = True
        else:
            save_checkpoint = False
    else:
        save_checkpoint = False

    print("save_checkpoint status:", str(save_checkpoint))

    # TODO load_checkpoint_path when we want to restart at a certain stage

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

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:  # alternatively train/val

            index = 0
            model_time = time.time()

            # record json log, initially empty
            json_log[str(epoch + 1)] = {}

            # initiate
            log_dict = {}
            for cls_idx in range(len(class_names)):
                log_dict[cls_idx] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # initiate
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
                            loss2 = criterion(model(inputs), labels)
                            loss2.backward()  # make sure to do a full forward pass
                            optimizer.second_step(zero_grad=True)
                        else:
                            loss.backward()
                            optimizer.step()

                # log
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

                    # log_dict[cls_idx] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
                    log_dict[cls_idx]['tp'] += tp
                    log_dict[cls_idx]['tn'] += tn
                    log_dict[cls_idx]['fp'] += fp
                    log_dict[cls_idx]['fn'] += fn

                # tensorboard
                if writer is not None:
                    # ...log the running loss
                    writer.add_scalar(phase + ' minibatch loss',
                                      float(loss.item()),
                                      epoch * len(dataloaders[phase]) + index)
                    writer.add_scalar(phase + ' minibatch ACC',
                                      float(torch.sum(preds == labels.data) / inputs.size(0)),
                                      epoch * len(dataloaders[phase]) + index)

                # check
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
                            check_SAA(model, model_idx, 384, check_dataloaders[phase], class_names, check_index,
                                      num_images=1, device=device,
                                      pic_name='GradCAM_' + str(epoch_idx) + '_I_' + str(index + 1),
                                      skip_batch=check_minibatch, draw_path=draw_path, writer=writer)
                        except:
                            print('model:', model_idx, ' with edge_size', 384, 'is not supported yet')
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
                tp = log_dict[cls_idx]['tp']
                tn = log_dict[cls_idx]['tn']
                fp = log_dict[cls_idx]['fp']
                fn = log_dict[cls_idx]['fn']
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

                '''
                # too much for imagenet-1k
                print('{} precision: {:.4f}  recall: {:.4f}'.format(class_names[cls_idx], precision, recall))
                print('{} sensitivity: {:.4f}  specificity: {:.4f}'.format(class_names[cls_idx], TPR, TNR))
                print('{} FPR: {:.4f}  NPV: {:.4f}'.format(class_names[cls_idx], FPR, NPV))
                # tensorboard
                if writer is not None:
                    # ...log the running loss
                    writer.add_scalar(phase + '   ' + class_names[cls_idx] + ' precision',
                                      precision,
                                      epoch + 1)
                    writer.add_scalar(phase + '   ' + class_names[cls_idx] + ' recall',
                                      recall,
                                      epoch + 1)
                '''
            # json log: update
            json_log[str(epoch + 1)][phase] = log_dict

            if phase == 'val':
                temp_vac = epoch_acc
            else:
                temp_acc = epoch_acc  # not useful actually

            # deep copy the model
            if phase == 'val' and better_performance(temp_acc, temp_vac, best_acc, best_vac) and epoch >= intake_epochs:
                best_epoch_idx = epoch + 1
                best_acc = temp_acc
                best_vac = temp_vac
                best_model_wts = copy.deepcopy(model.state_dict())
                best_log_dic = log_dict

            print('\n')

        print()

        # save_checkpoint
        if save_checkpoint and epoch % checkpoint_gap == checkpoint_gap - 1:
            print('save checkpoint of epoch_idx:', epoch)
            save_checkpoint_path = os.path.join(checkpoint_path, model_idx + "_checkpoint_at_epochidx_"
                                                + str(epoch) + '.pth')

            if gpu_use == -1:
                state = {'model': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            else:
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}

            torch.save(state, save_checkpoint_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch idx: ', best_epoch_idx)
    print('Best epoch train Acc: {:4f}'.format(best_acc))
    print('Best epoch val Acc: {:4f}'.format(best_vac))
    for cls_idx in range(len(class_names)):
        tp = best_log_dic[cls_idx]['tp']
        tn = best_log_dic[cls_idx]['tn']
        fp = best_log_dic[cls_idx]['fp']
        fn = best_log_dic[cls_idx]['fn']
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
    '''
    # too much for imagenet-1k
    for cls_idx in range(len(class_names)):
        print('{} precision: {:.4f}  recall: {:.4f}'.format(class_names[cls_idx], precision, recall))
        print('{} sensitivity: {:.4f}  specificity: {:.4f}'.format(class_names[cls_idx], TPR, TNR))
        print('{} FPR: {:.4f}  NPV: {:.4f}'.format(class_names[cls_idx], FPR, NPV))
    '''

    # tensorboard
    if writer is not None:
        writer.close()

    # load best model weights as final model training result
    model.load_state_dict(best_model_wts)
    # save json_log  indent=2 for better view
    json.dump(json_log, open(os.path.join(draw_path, model_idx + '_log.json'), 'w'), ensure_ascii=False, indent=2)
    return model


def main(args):
    if args.paint is False:
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

    drop_rate = args.drop_rate
    attn_drop_rate = args.attn_drop_rate
    drop_path_rate = args.drop_path_rate
    pretrained_backbone = False if args.pretrained_backbone_off else True

    # ImageNet-1k
    num_classes = args.num_classes  # 1000
    # edge size of 384
    edge_size = args.edge_size  # 384

    # batch info
    batch_size = args.batch_size  # 8
    num_workers = args.num_workers  # main training num_workers 70

    num_epochs = args.num_epochs  # 50
    intake_epochs = args.intake_epochs  # 30

    lr = args.lr  # 0.000007
    lrf = args.lrf  # 0.0

    opt_name = args.opt_name  # 'Adam'

    # PATH
    draw_root = args.draw_root
    model_path = args.model_path
    dataroot = args.dataroot

    # checkpoint
    checkpoint_gap = args.checkpoint_gap

    if enable_notify:
        import notifyemail as notify

        notify.Reboost(mail_host='smtp.163.com', mail_user='tum9598@163.com', mail_pass='EXVGQACCXPUIUQAE',
                       default_reciving_list=['tum9598@163.com'],  # change here if u want to use notify
                       log_root_path='log', max_log_cnt=5)

        if enable_tensorboard:
            notify.add_text('update to the tensorboard')
        else:
            notify.add_text('not update to the tensorboard')

        notify.add_text('  ')

        notify.add_text('models idx ' + str(model_idx))
        notify.add_text('  ')

        notify.add_text('GPU idx: ' + str(gpu_idx))
        notify.add_text('  ')

        notify.add_text('classes number ' + str(num_classes))
        notify.add_text('edge size ' + str(edge_size))
        notify.add_text('batch_size ' + str(batch_size))
        notify.add_text('num_epochs ' + str(num_epochs))
        notify.add_text('lr ' + str(lr))
        notify.add_text('opt_name ' + str(opt_name))
        notify.add_text('enable_sam ' + str(enable_sam))
        notify.send_log()

    print("*********************************{}*************************************".format('setting'))
    print(args)

    draw_path = os.path.join(draw_root, 'PT_' + model_idx)
    save_model_path = os.path.join(model_path, 'PT_' + model_idx + '.pth')

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if os.path.exists(draw_path):
        del_file(draw_path)  # clear the output folder, NOTICE this may be DANGEROUS
    else:
        os.makedirs(draw_path)

    if enable_tensorboard:
        writer = SummaryWriter(draw_path)
    else:
        writer = None
    # if u run locally
    # nohup tensorboard --logdir=/home/MSHT/runs --host=0.0.0.0 --port=7777 &
    # tensorboard --logdir=/home/ZTY/runs --host=0.0.0.0 --port=7777

    # Data Augmentation
    data_transforms = {
        "train": transforms.Compose([transforms.RandomResizedCrop((edge_size, edge_size)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((edge_size, edge_size)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])}

    datasets = {x: torchvision.datasets.ImageFolder(os.path.join(dataroot, x), data_transforms[x]) for x in
                ['train', 'val']}  # get Train and Val imageNet dataset

    dataloaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True,
                                                        num_workers=num_workers),
                   'val': torch.utils.data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False,
                                                      num_workers=num_workers // 4 + 1)  # 4
                   }

    class_names = [d.name for d in os.scandir(os.path.join(dataroot, 'train')) if d.is_dir()]
    if len(class_names) == num_classes:
        print("class_names:", class_names)
    else:
        print('classfication number of the model mismatch the dataset requirement')
        return -1
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    # get model
    model = get_model(num_classes, edge_size, model_idx, drop_rate, attn_drop_rate, drop_path_rate, pretrained_backbone)

    if Pre_Trained_model_path is not None:
        if os.path.exists(Pre_Trained_model_path):
            model.load_state_dict(torch.load(Pre_Trained_model_path), False)
        else:
            print('Pre_Trained_model_path:' + Pre_Trained_model_path, ' is NOT avaliable!!!!\n')
            print('we ignore this with a new start up')

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if gpu_use == -1:
        model = nn.DataParallel(model)

    model.to(device)

    # check model summary
    summary(model, input_size=(3, edge_size, edge_size))  # after to device

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
            # scheduler = None
        elif opt_name == 'Adam':
            base_optimizer = torch.optim.Adam  # define an optimizer for the "sharpness-aware" update
            optimizer = SAM(model.parameters(), base_optimizer, lr=lr, weight_decay=0.01)

    if lrf > 0:  # use cosine learning rate schedule
        import math
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * (1 - lrf) + lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # train
    model_ft = train_model_PT(model, dataloaders, criterion, optimizer, class_names, dataset_sizes,
                              num_epochs=num_epochs, intake_epochs=intake_epochs, check_minibatch=40000 // batch_size,
                              scheduler=scheduler, device=device, draw_path=draw_path, checkpoint_gap=checkpoint_gap,
                              model_idx=model_idx, checkpoint_path=model_path, gpu_use=gpu_use,
                              enable_attention_check=enable_attention_check,
                              enable_visualize_check=enable_visualize_check,
                              enable_sam=enable_sam, writer=writer)

    # save model if its a multi-GPU model, save as a single GPU one too
    if gpu_use == -1:
        torch.save(model_ft.module.state_dict(), save_model_path)
        print('model trained by multi-GPUs has its single GPU copy saved at ', save_model_path)
    else:
        torch.save(model_ft.state_dict(), save_model_path)
        print('model trained by GPU (idx:' + str(gpu_use) + ') has been saved at ', save_model_path)


def get_args_parser():
    parser = argparse.ArgumentParser(description='Supervised ImageNet Training')

    # Model Name or index
    parser.add_argument('--model_idx', default='conformer_PT_sample', type=str, help='Model Name or index')
    # drop_rate, attn_drop_rate, drop_path_rate
    parser.add_argument('--drop_rate', default=0.0, type=float, help='dropout rate , default 0.0')
    parser.add_argument('--attn_drop_rate', default=0.0, type=float, help='dropout rate Aftter Attention, default 0.0')
    parser.add_argument('--drop_path_rate', default=0.0, type=float, help='drop path for stochastic depth, default 0.0')
    parser.add_argument('--pretrained_backbone_off', action='store_true', help='disable pretrained backbone weight')

    # Enviroment parameters
    parser.add_argument('--gpu_idx', default=-1, type=int,
                        help='use a single GPU with its index, -1 to use multiple GPU')

    # Path parameters
    parser.add_argument('--dataroot', default='/data/imagenet_1k', help='path to dataset')
    parser.add_argument('--model_path', default='/home/pancreatic-cancer-project/saved_models',
                        help='path to save model state-dict')
    parser.add_argument('--draw_root', default='/home/pancreatic-cancer-project/runs',
                        help='path to draw and save tensorboard output')

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
    parser.add_argument('--Pre_Trained_model_path', default=None, type=str,
                        help='Finetuning a trained model in this dataset')

    # Dataset based parameters
    parser.add_argument('--num_classes', default=1000, type=int, help='classification number')
    parser.add_argument('--edge_size', default=384, type=int, help='edge size of input image')  # 224 256 384 1000
    parser.add_argument('--num_workers', default=70, type=int, help='use CPU num_workers , default 4')
    # num_workers: 16-server 104 3090-server 21  batch_size: 16-server 210 3090-server 42
    # Training seting parameters
    parser.add_argument('--batch_size', default=210, type=int, help='batch_size default 8')
    parser.add_argument('--num_epochs', default=300, type=int, help='training epochs')
    parser.add_argument('--intake_epochs', default=0, type=int, help='only save model at epochs after intake_epochs')
    parser.add_argument('--lr', default=0.00001, type=float, help='learing rate')
    parser.add_argument('--lrf', type=float, default=0.1,
                        help='learing rate decay rate, default 0(not enabled), suggest 0.1 and lr=0.00005')
    parser.add_argument('--opt_name', default='Adam', type=str, help='optimizer name Adam or SGD')

    # checkpoint info
    parser.add_argument('--checkpoint_gap', default=10, type=int, help='save check point epoch gap, default 10')

    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

'''
Experiment record 0: MSHT
PreTrain on 3*A6000 GPU memory 256G, video memory 3*48G CPU 104 kernal Xeon
lr=0.00001 
lrf=0.1
num_epochs=300 (145 is the best, at last)
batch_size=210 
num_worker=70
checkpoint_gap=10
edge_size=384


USAGE suggestion like:

nohup python PreTrain.py --model_idx Hybrid2_384_PreTrain --enable_tensorboard --enable_notify 
--num_epochs 300 --batch_size 210 --lrf 0.1 --num_workers 70 --edge_size 384 &
nohup tensorboard --logdir=/home/pancreatic-cancer-project/runs --host=172.28.230.21 --port=7777 > tensor_board.out 2>&1 &

'''
