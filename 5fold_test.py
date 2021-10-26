"""
5 fold Testing script  ver： OCT 26th 21：00 official release
"""

from __future__ import print_function, division

import os
import argparse
import json
import torch
import numpy as np
import torch.nn as nn
import torchvision
from torchvision import models, transforms
import time
from tensorboardX import SummaryWriter

from utils.visual_usage import *
from utils.tools import get_model, setup_seed, del_file

from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy import interp


def get_all_5_fold_models_idxs(model_path, k=5):
    """
    auto check the path to grab group-uped models

    the model we are going to use. by the format of Model_size_other_info
    each model name SHOULD end with‘_k？’, like 'Hybriod2_384_401_lf25_b8_k1' for the fold 1 model

    :param model_path: the path of saved models, code will detect the model groups matching the rule
    :param k: the fold num, code will check if the model group is enough matching the fold num k

    :return: prepared model idx groups
    """
    model_names_dict = {}
    for model_name in os.listdir(model_path):
        if model_name[0:-7] not in model_names_dict:
            model_names_dict[model_name[0:-7]] = 1
        else:
            model_names_dict[model_name[0:-7]] += 1

    print(model_names_dict)

    legal_k_fold_list = []
    for model_name in model_names_dict:
        if model_names_dict[model_name] == k:
            legal_k_fold_list.append(model_name)

    return legal_k_fold_list


def plot_roc(roc_auc_list, imagename='ROC_cruve.png', draw_root='/home/pancreatic-cancer-project/imaging_results'):
    if not os.path.exists(draw_root):
        os.makedirs(draw_root)
    img_path = os.path.join(draw_root, imagename)
    plt.figure()  # get a new figure

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    for i in range(len(roc_auc_list)):
        tprs.append(interp(mean_fpr, roc_auc_list[i][0], roc_auc_list[i][1]))
        roc_auc = roc_auc_list[i][2]
        aucs.append(roc_auc)
        plt.plot(roc_auc_list[i][0], roc_auc_list[i][1], lw=1.5, alpha=0.6,
                 label='Fold %d (AUC = %0.3f)' % (i, roc_auc))
        plt.rcParams['font.size'] = 8
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc=auc(mean_fpr,mean_tpr)# calculating the average AUC
    mean_auc = np.mean(aucs)
    # std_auc=np.std(tprs)
    std_auc = np.std(aucs)
    print('Mean AUC = %0.3f ± %0.3f' % (mean_auc, std_auc))

    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean AUC = %0.3f $\pm$ %0.3f' % (mean_auc, std_auc), lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_tpr,tprs_lower,tprs_upper,color='gray',alpha=.2,label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', font1)
    plt.ylabel('True Positive Rate', font1)
    plt.title('Receiver operating characteristic curves of test set', font1)
    plt.legend(loc='lower right', fontsize=8)

    plt.savefig(img_path, dpi=1000)
    plt.show()
    plt.close()

    return img_path


class soft_max_layer(nn.Module):
    def __init__(self):
        super(soft_max_layer, self).__init__()
        self.soft_max = nn.Softmax()

    def forward(self, x):
        return self.soft_max(x)


def group_test_model(model, test_dataloader, criterion, class_names, test_dataset_size, model_idx, edge_size,
                     check_minibatch=100,
                     device=None, draw_path='/home/ZTY/imaging_results', enable_attention_check=False,
                     enable_visualize_check=True,
                     writer=None):
    """
    Testing iteration for 5 fold validating(AUC saved)

    :param model: model object
    :param test_dataloader: the test_dataloader obj
    :param criterion: loss func obj
    :param class_names: The name of classes for priting
    :param test_dataset_size: size of datasets

    :param model_idx: model idx for the getting trained model
    :param edge_size: image size for the input image
    :param check_minibatch: number of skip over minibatch in calculating the criteria's results etc.

    :param device: cpu/gpu object
    :param draw_path: path folder for output pic
    :param enable_attention_check: use attention_check to show the pics of models' attention areas
    :param enable_visualize_check: use visualize_check to show the pics

    :param writer: attach the records to the tensorboard backend
    """
    test_model_idx = 'PC_' + model_idx + '_test'

    # scheduler is an LR scheduler object from torch.optim.lr_scheduler.
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    soft_max_la = soft_max_layer()

    since = time.time()

    print('Epoch: Test')
    print('-' * 10)

    phase = 'test'
    index = 0
    model_time = time.time()

    # initiate the empty json dict
    json_log = {}
    json_log['test'] = {}

    # initiate the empty log dict
    log_dict = {}
    for cls_idx in range(len(class_names)):
        log_dict[class_names[cls_idx]] = {'tp': 0.0, 'tn': 0.0, 'fp': 0.0, 'fn': 0.0}

    model.eval()  # Set model to evaluate mode

    # initiate the running loss
    running_loss = 0.0
    log_running_loss = 0.0
    running_corrects = 0

    # list for AUC
    score_list = []
    label_list = []
    # Iterate over data.
    for inputs, labels in test_dataloader:  # use different dataloder in different phase
        inputs = inputs.to(device)
        # print('inputs[0]',type(inputs[0]))

        labels = labels.to(device)

        # zero the parameter gradients only need in training
        # optimizer.zero_grad()

        # forward
        outputs = model(inputs)

        outputs2 = soft_max_la(outputs)

        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # calculate the score_list（softmax confidence）and label_list（ground truth）
        score_list.extend(outputs2[:, 1].detach().cpu().numpy())  # put on cpu
        # score_list.extend(outputs[:, 1].detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())

        # running loss
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

        # write to tensorboard
        if writer is not None:
            # ...log the running loss
            writer.add_scalar(phase + ' minibatch loss',
                              float(loss.item()),
                              index)
            writer.add_scalar(phase + ' minibatch ACC',
                              float(torch.sum(preds == labels.data) / inputs.size(0)),
                              index)

        # at the checking time now
        if index % check_minibatch == check_minibatch - 1:
            model_time = time.time() - model_time

            check_index = index // check_minibatch + 1

            epoch_idx = 'test'
            print('Epoch:', epoch_idx, '   ', phase, 'index of ' + str(check_minibatch) + ' minibatch:',
                  check_index, '     time used:', model_time)

            print('minibatch AVG loss:', float(log_running_loss) / check_minibatch)

            if enable_attention_check:
                try:
                    check_SAA(model, model_idx, edge_size, test_dataloader, class_names, check_index,
                              num_images=1, device=device,
                              pic_name='GradCAM_' + str(epoch_idx) + '_I_' + str(index + 1),
                              skip_batch=check_minibatch, draw_path=draw_path, writer=writer)
                except:
                    print('model:', model_idx, ' with edge_size', edge_size, 'is not supported yet')
            else:
                pass

            if enable_visualize_check:
                visualize_check(model, test_dataloader, class_names, check_index, num_images=6, device=device,
                                pic_name='Visual_' + str(epoch_idx) + '_I_' + str(index + 1),
                                skip_batch=check_minibatch, draw_path=draw_path, writer=writer)

            model_time = time.time()
            log_running_loss = 0.0

        index += 1

    # log criterias: print
    epoch_loss = running_loss / test_dataset_size
    epoch_acc = running_corrects.double() / test_dataset_size * 100
    print('\nEpoch:  {} \nLoss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    # json log: update
    json_log['test'][phase] = log_dict

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

        # AUC
        fpr, tpr, thresholds = roc_curve(label_list, score_list)
        roc_auc = auc(fpr, tpr)
        print('{} precision: {:.4f}  recall: {:.4f}'.format(class_names[cls_idx], precision, recall))
        print('{} sensitivity: {:.4f}  specificity: {:.4f}'.format(class_names[cls_idx], TPR, TNR))
        print('{} FPR: {:.4f}  NPV: {:.4f}'.format(class_names[cls_idx], FPR, NPV))
        print('{} TP: {}'.format(class_names[cls_idx], tp))
        print('{} TN: {}'.format(class_names[cls_idx], tn))
        print('{} FP: {}'.format(class_names[cls_idx], fp))
        print('{} FN: {}'.format(class_names[cls_idx], fn))
        print("AUC:{:.4f}".format(roc_auc_score(label_list, score_list)))
        print("AUC:{:.4f}".format(roc_auc))

    print('\n')

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # write to tensorboard
    if writer is not None:
        writer.close()

    # save json_log  indent=2 for better view
    json.dump(json_log, open(os.path.join(draw_path, test_model_idx + '_log.json'), 'w'), ensure_ascii=False, indent=2)

    return model, fpr, tpr, roc_auc


def test_5_fold_model_group(model_idx_group, args):
    if model_idx_group[0:3] == 'PC_':  # check name capbility
        model_idx_group = model_idx_group[3:]

    gpu_idx = args.gpu_idx  # GPU idx start with0, -1 to use multipel GPU

    # PATH
    draw_root = args.draw_root
    model_path = args.model_path
    dataroot = args.dataroot
    # choose the test dataset
    test_dataroot = os.path.join(dataroot, 'test')
    # dataset info
    num_classes = args.num_classes
    class_names = ['negative', 'positive'][0:num_classes]  # A G E B

    enable_notify = args.enable_notify  # False
    if enable_notify:
        import notifyemail as notify

    enable_tensorboard = args.enable_tensorboard  # False
    # Vision Transformer enable_attention_check = 'SAA'
    # 纯Transformer结构的attention我还没做好，CNN/Hybrid使用 enable_attention_check = 'CAM'
    enable_attention_check = args.enable_attention_check  # False   'CAM' 'SAA'
    enable_visualize_check = args.enable_visualize_check  # False

    # validating setting
    batch_size = args.batch_size  # 10
    criterion = nn.CrossEntropyLoss()

    # skip minibatch
    check_minibatch = args.check_minibatch if args.check_minibatch is not None else 80 // batch_size

    # structural parameter
    drop_rate = args.drop_rate
    attn_drop_rate = args.attn_drop_rate
    drop_path_rate = args.drop_path_rate

    use_cls_token = False if 'CLS' in model_idx_group.split('_') and 'No' in model_idx_group.split('_') else True
    use_pos_embedding = False if 'Pos' in model_idx_group.split('_') and 'No' in model_idx_group.split('_') else True
    if 'ATT' in model_idx_group.split('_') and 'No' in model_idx_group.split('_'):
        use_att_module = None
    elif 'CBAM' in model_idx_group.split('_'):
        use_att_module = 'CBAM'
    elif 'SE' in model_idx_group.split('_'):
        use_att_module = 'SE'
    else:
        use_att_module = 'SimAM'

    # ROC curve
    roc_auc_list = []  # plot_auc

    # 5fold test and calculation of ROC AUC
    for k_fold in range(1, 6):
        model_idx_act = model_idx_group + '_k' + str(int(k_fold))

        # default 384 read in name  of model idx
        edge_size = 224 if '224' in model_idx_group.split('_') else 384

        test_model_idx = 'PC_' + model_idx_act + '_test'
        draw_path = os.path.join(draw_root, test_model_idx)
        save_model_path = os.path.join(model_path, 'PC_' + model_idx_act + '.pth')

        if enable_notify:
            if enable_tensorboard:
                notify.add_text('testing model_idx: ' + str(model_idx_act) + '. update to the tensorboard')
            else:
                notify.add_text('testing model_idx: ' + str(model_idx_act) + '.  not update to the tensorboard')
            notify.add_text('edge_size =' + str(edge_size))
            notify.add_text('batch_size =' + str(batch_size))

        # get model
        pretrained_backbone = False  # model is trained already, pretrained backbone weight is useless here
        model = get_model(num_classes, edge_size, model_idx_act, drop_rate, attn_drop_rate, drop_path_rate,
                          pretrained_backbone, use_cls_token, use_pos_embedding, use_att_module)

        try:
            model.load_state_dict(torch.load(save_model_path))
            print("model loaded")
            print("model :", model_idx_act)
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
                    # setting 0 for: only card idx 0 is sighted for this code
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                except:
                    print("GPU distributing ERRO occur use CPU instead")

        else:
            # Decide which device we want to run on
            try:
                # setting k for: only card idx k is sighted for this code
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
            except:
                print('we dont have that GPU idx here, try to use gpu_idx=0')
                try:
                    # setting 0 for: only card idx 0 is sighted for this code
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                except:
                    print("GPU distributing ERRO occur use CPU instead")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # single card for test

        model.to(device)

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

        # Data Augmentation is not used in validating or testing
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation((0, 180)),
                transforms.CenterCrop(700),
                transforms.Resize(edge_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
                transforms.ToTensor()
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop(700),
                transforms.Resize(edge_size),
                transforms.ToTensor()
            ]),
        }
        # test setting is the same as the validate dataset's setting
        test_datasets = torchvision.datasets.ImageFolder(test_dataroot, data_transforms['val'])

        test_dataset_size = len(test_datasets)
        test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False,
                                                      num_workers=1)

        net, fpr, tpr, roc_auc = group_test_model(model, test_dataloader, criterion, class_names, test_dataset_size,
                                                  model_idx=model_idx_act, edge_size=edge_size,
                                                  check_minibatch=check_minibatch, device=device, draw_path=draw_path,
                                                  enable_attention_check=enable_attention_check,
                                                  enable_visualize_check=enable_visualize_check, writer=writer)
        roc_auc_list.append([fpr, tpr, roc_auc])

    imagename = 'Test_' + model_idx_group + '_ROC_cruve.png'
    img_path = plot_roc(roc_auc_list, imagename, draw_root=draw_root)

    if enable_notify:
        notify.add_file(img_path)


def main(args):
    if args.paint:
        # agg
        import matplotlib
        matplotlib.use('Agg')

    model_idx_groups = args.model_idx_groups
    enable_notify = args.enable_notify  # False
    model_path = args.model_path

    if enable_notify:
        import notifyemail as notify

        notify.Reboost(mail_host='smtp.163.com', mail_user='tum9598@163.com', mail_pass='EXVGQACCXPUIUQAE',
                       default_reciving_list=['pytorch1225@163.com'],
                       log_root_path='log', max_log_cnt=5)
        notify.send_log()

    print("*********************************{}*************************************".format('setting'))
    print(args)

    if model_idx_groups is None:
        model_idx_groups_list = get_all_5_fold_models_idxs(model_path)
    else:
        model_idx_groups_list = [model_idx_groups, ]

    print('\ntarget models groups:\n', model_idx_groups_list)

    for model_idx_group in model_idx_groups_list:
        test_5_fold_model_group(model_idx_group, args)


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Model Name or index
    parser.add_argument('--model_idx_groups', default=None, type=str, help='Model Name or index, None for auto')

    # drop_rate, attn_drop_rate, drop_path_rate
    parser.add_argument('--drop_rate', default=0.0, type=float, help='dropout rate , default 0.0')
    parser.add_argument('--attn_drop_rate', default=0.0, type=float, help='dropout rate Aftter Attention, default 0.0')
    parser.add_argument('--drop_path_rate', default=0.0, type=float, help='drop path for stochastic depth, default 0.0')

    # Enviroment parameters
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='use a single GPU with its index, default=0 to use the 0 GPU')

    # Path parameters
    parser.add_argument('--dataroot', default=r'/data/pancreatic-cancer-project/k5_dataset',
                        help='path to dataset')
    parser.add_argument('--model_path', default=r'/home/pancreatic-cancer-project/saved_models',
                        help='path to save model state-dict')
    parser.add_argument('--draw_root', default=r'/home/pancreatic-cancer-project/runs',
                        help='path to draw and save tensorboard output')

    # Help tool parameters  store_False
    parser.add_argument('--paint', action='store_false', help='paint in front desk')  # matplotlib.use('Agg')
    parser.add_argument('--enable_notify', action='store_true', help='enable notify to send email')
    # check tool parameters
    parser.add_argument('--enable_tensorboard', action='store_true', help='enable tensorboard to save status')
    # enable_attention_check = False
    parser.add_argument('--enable_attention_check', action='store_true', help='check and save attention map')
    parser.add_argument('--enable_visualize_check', action='store_true', help='check and save pics')

    # Dataset based parameters
    parser.add_argument('--num_classes', default=2, type=int, help='classification number')
    parser.add_argument('--edge_size', default=384, type=int, help='edge size of input image')  # 224 256 384 1000
    # Test setting parameters
    parser.add_argument('--batch_size', default=1, type=int, help='testing batch_size')
    # check_minibatch for painting pics
    parser.add_argument('--check_minibatch', default=None, type=int, help='check batch_size')

    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
