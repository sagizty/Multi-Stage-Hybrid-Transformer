"""
Testing script  ver： OCT 26th 21：00 official release
"""

from __future__ import print_function, division

from Train import *


def test_model(model, test_dataloader, criterion, class_names, test_dataset_size, model_idx, edge_size,
               check_minibatch=100,
               device=None, draw_path='/home/MSHT/imaging_results', enable_attention_check=None,
               enable_visualize_check=True,
               writer=None):
    """
    Testing iteration

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

    # criterias, initially empty
    running_loss = 0.0
    log_running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in test_dataloader:  # use different dataloder in different phase
        inputs = inputs.to(device)
        # print('inputs[0]',type(inputs[0]))

        labels = labels.to(device)

        # zero the parameter gradients only need in training
        # optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # log criterias: update
        log_running_loss += loss.item()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        # Compute recision and recall for each class.
        for cls_idx in range(len(class_names)):
            # NOTICE remember to put tensor back to cpu
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

        # attach the records to the tensorboard backend
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

            # how many image u want to check, should SMALLER THAN the batchsize

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
                visualize_check(model, test_dataloader, class_names, check_index, num_images=9, device=device,
                                pic_name='Visual_' + str(epoch_idx) + '_I_' + str(index + 1),
                                skip_batch=check_minibatch, draw_path=draw_path, writer=writer)

            model_time = time.time()
            log_running_loss = 0.0

        index += 1
    # json log: update
    json_log['test'][phase] = log_dict

    # log criterias: print
    epoch_loss = running_loss / test_dataset_size
    epoch_acc = running_corrects.double() / test_dataset_size * 100
    print('\nEpoch:  {} \nLoss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    for cls_idx in range(len(class_names)):
        # calculating the confusion matrix
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

    print('\n')

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # attach the records to the tensorboard backend
    if writer is not None:
        writer.close()

    # save json_log  indent=2 for better view
    json.dump(json_log, open(os.path.join(draw_path, test_model_idx + '_log.json'), 'w'), ensure_ascii=False, indent=2)

    return model


def main(args):
    if args.paint:
        # use Agg kernal, not painting in the front-desk
        import matplotlib
        matplotlib.use('Agg')

    gpu_idx = args.gpu_idx  # GPU idx start with0, -1 to use multipel GPU

    enable_notify = args.enable_notify  # False
    enable_tensorboard = args.enable_tensorboard  # False

    enable_attention_check = args.enable_attention_check  # False
    enable_visualize_check = args.enable_visualize_check  # False

    model_idx = args.model_idx  # the model we are going to use. by the format of Model_size_other_info
    # structural parameter
    drop_rate = args.drop_rate
    attn_drop_rate = args.attn_drop_rate
    drop_path_rate = args.drop_path_rate
    use_cls_token = False if args.cls_token_off else True
    use_pos_embedding = False if args.pos_embedding_off else True
    use_att_module = None if args.att_module == 'None' else args.att_module

    # PATH info
    draw_root = args.draw_root
    model_path = args.model_path
    dataroot = args.dataroot

    test_model_idx = 'PC_' + model_idx + '_test'
    draw_path = os.path.join(draw_root, test_model_idx)
    save_model_path = os.path.join(model_path, 'PC_' + model_idx + '.pth')
    # choose the test dataset
    test_dataroot = os.path.join(dataroot, 'test')

    # dataset info
    num_classes = args.num_classes
    class_names = ['negative', 'positive'][0:num_classes]
    edge_size = args.edge_size  # 1000 224 384

    # validating setting
    batch_size = args.batch_size  # 10
    criterion = nn.CrossEntropyLoss()

    # skip minibatch
    check_minibatch = args.check_minibatch if args.check_minibatch is not None else 80 // batch_size

    if enable_notify:
        import notifyemail as notify

        notify.Reboost(mail_host='smtp.163.com', mail_user='tum9598@163.com', mail_pass='EXVGQACCXPUIUQAE',
                       default_reciving_list=['904639643@qq.com', 'pytorch1225@163.com'],
                       log_root_path='log', max_log_cnt=5)

        if enable_tensorboard:
            notify.add_text('testing model_idx: ' + str(model_idx) + '.  update to the tensorboard')
        else:
            notify.add_text('testing model_idx: ' + str(model_idx) + '.  not update to the tensorboard')
        notify.add_text('edge_size =' + str(edge_size))
        notify.add_text('batch_size =' + str(batch_size))
        notify.send_log()

    # get model
    pretrained_backbone = False  # model is trained already, pretrained backbone weight is useless here
    model = get_model(num_classes, edge_size, model_idx, drop_rate, attn_drop_rate, drop_path_rate,
                      pretrained_backbone, use_cls_token, use_pos_embedding, use_att_module)

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

    print("*********************************{}*************************************".format('setting'))
    print(args)

    # Data Augmentation is not used in validating or testing
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

    # test setting is the same as the validate dataset's setting
    test_datasets = torchvision.datasets.ImageFolder(test_dataroot, data_transforms['val'])
    test_dataset_size = len(test_datasets)
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False, num_workers=1)

    test_model(model, test_dataloader, criterion, class_names, test_dataset_size,
               model_idx=model_idx, edge_size=edge_size,
               check_minibatch=check_minibatch, device=device, draw_path=draw_path,
               enable_attention_check=enable_attention_check,
               enable_visualize_check=enable_visualize_check, writer=writer)


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

    # Enviroment parameters
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='use a single GPU with its index, -1 to use multiple GPU')

    # Path parameters
    parser.add_argument('--dataroot', default=r'/data/pancreatic-cancer-project/k5_dataset',
                        help='path to dataset')
    parser.add_argument('--model_path', default=r'/home/pancreatic-cancer-project/saved_models',
                        help='path to save model state-dict')
    parser.add_argument('--draw_root', default=r'/home/pancreatic-cancer-project/runs',
                        help='path to draw and save tensorboard output')

    # Help tool parameters
    parser.add_argument('--paint', action='store_false', help='paint in front desk')  # matplotlib.use('Agg')
    parser.add_argument('--enable_notify', action='store_true', help='enable notify to send email')
    # check tool parameters
    parser.add_argument('--enable_tensorboard', action='store_true', help='enable tensorboard to save status')

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
