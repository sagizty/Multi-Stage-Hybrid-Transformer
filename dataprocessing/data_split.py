'''
版本 08月14日 22：00  划分数据集的代码

参考：https://zhuanlan.zhihu.com/p/199238910

'''
import os
import random
import shutil
from shutil import copy2
from multiprocessing import Pool, cpu_count


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


def make_and_clear_path(file_pack_path):
    if not os.path.exists(file_pack_path):
        os.makedirs(file_pack_path)
    del_file(file_pack_path)


def a_dataset_split(src_data_folder, class_name, train_scale, val_scale, test_scale, com_num=None):
    current_class_data_path = os.path.join(src_data_folder, class_name)
    current_all_data = os.listdir(current_class_data_path)

    current_data_length = len(current_all_data)
    current_data_index_list = list(range(current_data_length))
    random.shuffle(current_data_index_list)

    train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
    val_folder = os.path.join(os.path.join(target_data_folder, 'val'), class_name)
    test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)

    train_stop_flag = current_data_length * train_scale
    val_stop_flag = current_data_length * (train_scale + val_scale)
    current_idx = 0
    train_num = 0
    val_num = 0
    test_num = 0
    for i in current_data_index_list:
        src_img_path = os.path.join(current_class_data_path, current_all_data[i])
        if current_idx <= train_stop_flag:
            copy2(src_img_path, train_folder)
            # print("{}复制到了{}".format(src_img_path, train_folder))
            train_num = train_num + 1

        elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
            copy2(src_img_path, val_folder)
            # print("{}复制到了{}".format(src_img_path, val_folder))
            val_num = val_num + 1

        else:  # TODO 写法是把余下作为测试集合，这个不严谨
            copy2(src_img_path, test_folder)
            # print("{}复制到了{}".format(src_img_path, test_folder))
            test_num = test_num + 1

        current_idx = current_idx + 1

    print("*********************************{}*************************************".format(class_name)+'\n'+
          "{}类按照{}：{}：{}的比例划分完成，一共{}张图片".format(class_name, train_scale, val_scale, test_scale,
                                                current_data_length) + '\n' + "训练集{}：{}张".format(train_folder,
                                                                                                 train_num)
          + '\n' + "验证集{}：{}张".format(val_folder, val_num) + '\n' + "测试集{}：{}张".format(test_folder, test_num)
          + '\n')

    if com_num is not None:
        print('processed class idx:', com_num)


def data_set_split(src_data_folder, target_data_folder='./dataset', train_scale=0.8, val_scale=0.2, test_scale=0.0,
                   Parallel_processing=True):
    """
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: 源文件夹 E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/src_data
    :param target_data_folder: 目标文件夹 E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/target_data
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例

    :param Parallel_processing: 是否并行处理

    :return:
    """
    make_and_clear_path(target_data_folder)
    print("开始数据集划分")
    class_names = os.listdir(src_data_folder)
    # 在目标目录下创建文件夹
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        # 然后在split_path的目录下创建类别文件夹
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            os.makedirs(class_split_path)

    if Parallel_processing:
        # 建立进程池
        tasks_num = len(class_names)
        process_pool = Pool(min(cpu_count() - 2, tasks_num))  # 并行数, 至少留2个核

        com_num = 0
        print("start processing" + str(tasks_num) + " files by multi-process")
        # 安排任务
        for class_name in class_names:
            # Pool.apply_async(要调用的目标,(传递给目标的参数元祖,))
            # 每次循环将会用空闲出来的子进程去调用目标
            com_num += 1
            args = (src_data_folder, class_name, train_scale, val_scale, test_scale, com_num)
            process_pool.apply_async(a_dataset_split, args)

        process_pool.close()  # 关闭进程池，关闭后po不再接收新的请求
        process_pool.join()  # 等待po中所有子进程执行完成，必须放在close语句之后

    else:
        # 按照比例划分数据集，并进行数据图片的复制
        # 按分类遍历
        for class_name in class_names:
            a_dataset_split(src_data_folder, class_name, train_scale, val_scale, test_scale)


def k_fold_split(src_data_folder, target_data_folder='./kfold', k=5):
    """
    读取源数据文件夹，生成划分好的文件夹，分为trian、val 2个文件夹，内有train和val

    :param src_data_folder: 整理好的imagenet 格式的需要做k折划分的文件夹
    :param target_data_folder: 目标大文件夹，内部会生成k个文件夹，k个文件夹为imagenet格式，内有train和val
    :param k: 划分折数

    :return:
    """
    make_and_clear_path(target_data_folder)
    print("开始数据集划分")
    class_names = os.listdir(src_data_folder)  # 获取类别名字

    # 按照比例对每个类别划分数据集，并进行数据图片的复制分发
    for class_name in class_names:  # 首先进行分类遍历

        current_class_data_path = os.path.join(src_data_folder, class_name)
        current_class_data_names = os.listdir(current_class_data_path)

        current_data_length = len(current_class_data_names)
        random.shuffle(current_class_data_names)

        # 分数据
        split_num = current_data_length // k
        # 每split_num个数据放一个包，如果有k+1个包，则最后一个包放不满它最多只有k-1个数据
        temp_split_pack = [current_class_data_names[i:i + split_num] for i in range(0, current_data_length, split_num)]
        fold_name_pack = [temp_split_pack[i] for i in range(0, k)]  # 取前k个包
        if len(temp_split_pack) > k:  # 最后不能均分的话最后一个会多一个pack，将其中内容依次放到不同pack
            for pack_idx, name in enumerate(temp_split_pack[-1]):  # 多的pack最多只有k-1个数据
                fold_name_pack[pack_idx].append(name)

        print("{}类按照{}折交叉验证划分，一共{}张图片".format(class_name, k, current_data_length))

        for p in range(1, k + 1):  # 对于每一折, 从1开始
            # 文件夹
            train_folder = os.path.join(target_data_folder, 'fold_' + str(p), 'train', class_name)
            val_folder = os.path.join(target_data_folder, 'fold_' + str(p), 'val', class_name)
            os.makedirs(train_folder)
            os.makedirs(val_folder)

            pack_idx = p - 1  # 该折数据作为val，余下作为train

            # 复制分发数据
            train_num = 0
            val_num = 0

            for j in range(k):
                if j == pack_idx:
                    for i in fold_name_pack[j]:
                        src_img_path = os.path.join(current_class_data_path, i)
                        copy2(src_img_path, val_folder)
                        val_num += 1
                        # print("{}复制到了{}".format(src_img_path, val_folder))
                else:
                    for i in fold_name_pack[j]:
                        src_img_path = os.path.join(current_class_data_path, i)
                        copy2(src_img_path, train_folder)
                        train_num += 1
                        # print("{}复制到了{}".format(src_img_path, train_folder))
            print("fold {}:  class:{}  train num: {}".format(p, class_name, train_num))
            print("fold {}:  class:{}  val num: {}".format(p, class_name, val_num))


if __name__ == '__main__':
    # src_data_folder = r'C:\Users\admin\Desktop\jpg_dataset'
    # target_data_folder = r'C:\Users\admin\Desktop\ZTY_dataset'
    # data_set_split(src_data_folder, target_data_folder, train_scale=0.7, val_scale=0.1, test_scale=0.2)

    # 现在用于处理imagenet_1k数据以实现预训练
    src_data_folder = r'/data/imagenet/train'
    target_data_folder = r'/data/imagenet_1k'
    data_set_split(src_data_folder, target_data_folder, train_scale=0.7, val_scale=0.1, test_scale=0.2)
    # k_fold_split(src_data_folder, target_data_folder, k=5)
