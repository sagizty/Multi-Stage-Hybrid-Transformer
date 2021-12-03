'''
Organize the data to ensure that all data is in jpg format  ver： AUG 11th 13:46 official release

'''
import os
import re
import csv
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms


def del_file(filepath):
    """
    Delete all files and folders in one directory
    :param filepath: file path
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


def find_all_files(root, suffix=None):
    '''
    Return a list of file paths ended with specific suffix
    '''
    res = []
    for root, _, files in os.walk(root):
        for f in files:
            if suffix is not None and not f.endswith(suffix):
                continue
            res.append(os.path.join(root, f))
    return res


def read_file(f_dir):
    '''
    Read a file and convert it into numpy format
    '''
    f_image = Image.open(f_dir)
    return f_image


def change_shape(image, corp_x=2400, corp_y=1800, f_x=1390, f_y=1038):
    '''
    Resize the image into x*y
    '''
    if image.size[0] > corp_x or image.size[1] > corp_y:
        # Generate an object of CenterCrop class to crop the image from the center into corp_x*corp_y
        crop_obj = torchvision.transforms.CenterCrop((corp_y, corp_x))
        image = crop_obj(image)
        # print(image.size[0], image.size[1])

    image.thumbnail((f_x, f_y), Image.ANTIALIAS)
    return image


def save_file(f_image, save_dir, suffix='.jpg'):
    '''
    Save and rename the images, generate the renamed table
    '''
    filepath, _ = os.path.split(save_dir)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    f_image.save(save_dir + suffix)


def PC_to_stander(root_from=r'C:\Users\admin\Desktop\dataset\PC',
                  root_positive=r'C:\Users\admin\Desktop\jpg_dataset\P',
                  root_negative=r'C:\Users\admin\Desktop\jpg_dataset\N', corp_x=2400, corp_y=1800, f_x=1390, f_y=1038):
    root_target, _ = os.path.split(root_positive)
    make_and_clear_path(root_target)

    f_dir_list = find_all_files(root=root_from)
    # print(f_dir_list)

    name_dict = {}  # Save the new and old names
    old_size_type = []
    size_type = []  # Record all different image sizes (after reshape)

    for seq in tqdm(range(len(f_dir_list))):
        f_dir = f_dir_list[seq]

        if '非癌' in f_dir or '阴性' in f_dir or '良性' in f_dir:
            root_target = root_negative
        else:
            root_target = root_positive

        f_image = read_file(f_dir)

        size = (f_image.size[0], f_image.size[1])
        if size not in old_size_type:
            old_size_type.append(size)

        f_image = change_shape(f_image, corp_x=corp_x, corp_y=corp_y, f_x=f_x, f_y=f_y)

        size = (f_image.size[0], f_image.size[1])
        if size not in size_type:
            size_type.append(size)

        save_dir = os.path.join(root_target, str(seq + 1))  # Set save directory
        name_dict[save_dir] = f_dir

        save_file(f_image, save_dir)

    print('old size type:', old_size_type)
    print('size type: ', size_type)

    root_target, _ = os.path.split(root_positive)
    pd.DataFrame.from_dict(name_dict, orient='index', columns=['origin path']).to_csv(
        os.path.join(root_target, 'name_dict.csv'))


def trans_csv_folder_to_imagefoder(target_path=r'C:\Users\admin\Desktop\MRAS_SEED_dataset',
                                   original_path=r'C:\Users\admin\Desktop\dataset\MARS_SEED_Dataset\train\train_org_image',
                                   csv_path=r'C:\Users\admin\Desktop\dataset\MARS_SEED_Dataset\train\train_label.csv'):
    """
    Original data format: a folder with image inside + a csv file with header which has the name and category of every image.
    Process original dataset and get data packet in image folder format

    :param target_path: the path of target image folder
    :param original_path: The folder with images
    :param csv_path: A csv file with header and the name and category of each image
    """
    idx = -1
    with open(csv_path, "rt", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        make_and_clear_path(target_path)  # Clear target_path
        for row in tqdm(rows):
            idx += 1
            if idx == 0:  # Skip the first header
                continue
            item_path = os.path.join(original_path, row[0])
            if os.path.exists(os.path.join(target_path, row[1])):
                shutil.copy(item_path, os.path.join(target_path, row[1]))
            else:
                os.makedirs(os.path.join(target_path, row[1]))
                shutil.copy(item_path, os.path.join(target_path, row[1]))

        print('total num:', idx)


if __name__ == '__main__':
    PC_to_stander(root_from=r'E:\dataset\PC',
                  root_positive=r'C:\Users\admin\Desktop\jpg_dataset\Positive',
                  root_negative=r'C:\Users\admin\Desktop\jpg_dataset\Negative', corp_x=2400, corp_y=1800, f_x=1390, f_y=1038)
