'''
整理log,输出excel  版本： 10月21日 14：00  # 适配不同系统

'''

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
                return result_indicators
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
            return result_indicators  # 结束
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
            result_indicators = read_a_json_log(json_path, record_dir)

            for i in result_indicators:
                f_log.write(str(i) + ', ')
            f_log.write('\n')
        f_log.close()


# json_path = r'C:\Users\admin\Desktop\runs\PC_Hybrid2_384_000_t4_507_e50_2'
# record_dir = r'C:\Users\admin\Desktop'  # csv save dir
# read_json_log(json_path, record_dir)

# read_all_logs(r'/home/pancreatic-cancer-project/runs', r'/home/pancreatic-cancer-project/runs/logs')

# read_all_logs(r'C:\Users\admin\Desktop\colab\runs', r'C:\Users\admin\Desktop\colab\logs')
