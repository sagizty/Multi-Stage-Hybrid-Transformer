"""
Organize log and output excel   script  ver： Jun 27th 18:00
enable_notify
"""


import argparse
import json
import os

try:  # 适配不同系统
    from utils.metrics import *
except:
    from metrics import *


def find_all_files(root, suffix=None):
    '''
    返回特定后缀的所有文件路径列表
    '''
    res = []
    for root, _, files in os.walk(root):
        for f in files:
            if suffix is not None and not f.endswith(suffix):
                continue
            res.append(os.path.join(root, f))
    return res


def read_a_json_log(json_path, record_dir):
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    with open(json_path) as f:
        load_dict = json.load(f)
        # print(load_dict)
        epoch_num = len(load_dict)
        try:
            cls_list = [cls for cls in load_dict[str(1)]['train']]
            test_status = False
        except:
            cls_list = [cls for cls in load_dict['test']['test']]
            test_status = True
        else:
            pass
        cls_num = len(cls_list)

        indicator_list = ['Precision', 'Recall', 'Sensitivity', 'Specificity', 'NPV', 'F1_score']
        indicator_num = len(indicator_list)

        blank_num = cls_num * indicator_num
        first_blank_num = blank_num // 2

        empty_str1 = ' ,'  # 对齐Acc
        for i in range(0, first_blank_num):
            empty_str1 += ' ,'

        empty_str2 = ''
        for i in range(0, blank_num):
            empty_str2 += ' ,'

        result_csv_name = os.path.split(json_path)[1].split('.')[0] + '.csv'
        result_indicators = [os.path.split(json_path)[1].split('.')[0], ]  # 第一个位置留给model name

    with open(os.path.join(record_dir, result_csv_name), 'w') as f_log:
        if test_status:
            # 写头文件1
            f_log.write('Phase:,' + empty_str1 + ' Test\n')
            head = 'Epoch:, '
            class_head = 'Acc, '  # 目标 'Acc, '+ 类别* indicator_list
            for cls in cls_list:
                for indicator in indicator_list:
                    class_head += cls + '_' + indicator + ', '

            # 写头文件2
            f_log.write(head + class_head + '\n')  # Test
            f_log.close()

        else:
            # 写头文件1
            f_log.write('Phase:,' + empty_str1 + ' Train' + empty_str2 + ' Val\n')

            head = 'Epoch:, '
            class_head = 'Acc, '  # 目标 'Acc, '+ 类别* indicator_list
            for cls in cls_list:
                for indicator in indicator_list:
                    class_head += cls + '_' + indicator + ', '

            # 写头文件2
            f_log.write(head + class_head + class_head + '\n')  # Train val
            f_log.close()

    # 初始化最佳
    best_val_acc = 0.0

    for epoch in range(1, epoch_num + 1):
        if test_status:
            epoch = 'test'
        epoch_indicators = [epoch, ]  # 第一个位置留给epoch

        for phase in ['train', 'val']:
            if test_status:
                phase = 'test'

            sum_tp = 0.0

            phase_indicators = [0.0, ]  # 第一个位置留给ACC

            for cls in cls_list:
                log = load_dict[str(epoch)][phase][cls]
                tp = log['tp']
                tn = log['tn']
                fp = log['fp']
                fn = log['fn']

                sum_tp += tp

                Precision = compute_precision(tp, fp)
                Recall = compute_recall(tp, fn)

                Sensitivity = compute_sensitivity(tp, fn)
                Specificity = compute_specificity(tn, fp)

                NPV = compute_NPV(tn, fn)
                F1_score = compute_f1_score(tp, tn, fp, fn)

                cls_indicators = [Precision, Recall, Sensitivity, Specificity, NPV, F1_score]
                phase_indicators.extend(cls_indicators)

            Acc = 100 * (sum_tp / float(tp + tn + fn + fp))  # 直接取最后一个的tp tn fn fp 算总数就行
            phase_indicators[0] = Acc

            epoch_indicators.extend(phase_indicators)

            if Acc >= best_val_acc and phase == 'val':
                best_val_acc = Acc
                best_epoch_indicators = epoch_indicators

            elif test_status:
                with open(os.path.join(record_dir, result_csv_name), 'a') as f_log:
                    for i in epoch_indicators:
                        f_log.write(str(i) + ', ')
                    f_log.write('\n')
                    f_log.close()
                result_indicators.extend(epoch_indicators)
                return result_indicators  # 结束 返回test的log行
            else:
                pass

        # epoch_indicators
        with open(os.path.join(record_dir, result_csv_name), 'a') as f_log:
            for i in epoch_indicators:
                f_log.write(str(i) + ', ')
            f_log.write('\n')

    with open(os.path.join(record_dir, result_csv_name), 'a') as f_log:
        f_log.write('\n')
        f_log.write('\n')
        # 写头文件1
        f_log.write('Phase:,' + empty_str1 + ' Train' + empty_str2 + ' Val\n')
        # 写头文件2
        f_log.write('Best Epoch:, ' + class_head + class_head + '\n')  # Train val

        try:
            for i in best_epoch_indicators:
                f_log.write(str(i) + ', ')
            f_log.close()
            result_indicators.extend(best_epoch_indicators)
            return result_indicators  # 结束 返回best epoch行
        except:
            print('No best_epoch_indicators')
            return result_indicators  # 结束


def read_all_logs(logs_path, record_dir):
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    res = find_all_files(logs_path, suffix='.json')

    result_csv_name = os.path.split(logs_path)[1] + '.csv'

    with open(os.path.join(record_dir, result_csv_name), 'w') as f_log:
        for json_path in res:
            result_indicators = read_a_json_log(json_path, record_dir)  # best_epoch_indicators of a model json log

            for i in result_indicators:
                f_log.write(str(i) + ', ')
            f_log.write('\n')
        f_log.close()


def main(args):
    ONE_LOG = args.ONE_LOG
    draw_root = args.draw_root
    record_dir = args.record_dir

    enable_notify = args.enable_notify  # False

    if ONE_LOG:
        read_a_json_log(draw_root, record_dir)
    else:
        read_all_logs(draw_root, record_dir)

    if enable_notify:
        import notifyemail as notify

        notify.Reboost(mail_host='smtp.163.com', mail_user='xxxx@163.com', mail_pass='xxxx',
                       default_reciving_list=['xxx@163.com'],  # Fixme change here if u want to use notify
                       log_root_path='log', max_log_cnt=5)

        notify.add_text('  ')

        notify.add_text('PATH: ' + str(draw_root))
        notify.add_text('  ')

        if not ONE_LOG:
            for Experiment_idx in os.listdir(draw_root):
                notify.add_text('Experiment idxs: ' + str(Experiment_idx))
                notify.add_text('  ')

        notify.add_file(record_dir)
        notify.send_log()


def get_args_parser():
    parser = argparse.ArgumentParser(description='Log checker')

    parser.add_argument('--ONE_LOG', action='store_true', help='check only one LOG')

    parser.add_argument('--draw_root', default=r'/home/pancreatic-cancer-project/runs',
                        help='path to draw and save tensorboard output')

    parser.add_argument('--record_dir', default=r'/home/pancreatic-cancer-project/CSV_logs',
                        help='path to save csv log output')

    parser.add_argument('--enable_notify', action='store_true', help='enable notify to send email')

    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
