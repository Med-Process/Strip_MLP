# --------------------------------------------------------
# Copyright (c) 2023 CVIP of SUST
# Licensed under The MIT License [see LICENSE for details]
# Written by Guiping Cao
# --------------------------------------------------------


import os
import random
import shutil

random.seed(1024)

data_path = "/path/to/101_ObjectCategories"
train_dir = "/path/to/101_OCategories_Split82/train"
val_dir = "/path/to/101_OCategories_Split82/val"

if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
    shutil.rmtree(val_dir)
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    os.makedirs(val_dir)

file_list = os.listdir(data_path)
train_ratio = 0.8


for i in range(len(file_list)):
    full_tp = os.path.join(data_path, file_list[i])
    class_files = os.listdir(full_tp)
    random.shuffle(class_files)
    num = int(len(class_files) * train_ratio)
    train_ = class_files[:num]
    val_ = class_files[num:]

    train_dir_root2 = os.path.join(train_dir, file_list[i])
    val_dir_root2 = os.path.join(val_dir, file_list[i])
    os.mkdir(train_dir_root2)
    os.mkdir(val_dir_root2)

    for j in range(len(class_files)):
        if class_files[j] in train_:
            train_save = os.path.join(train_dir_root2, class_files[j])
            src_path = os.path.join(data_path, file_list[i], class_files[j])
            shutil.copy(src_path, train_save)
        elif class_files[j] in val_:
            val_save = os.path.join(val_dir_root2, class_files[j])
            src_path = os.path.join(data_path, file_list[i], class_files[j])
            shutil.copy(src_path, val_save)
        else:
            print("Something wrong with train & val split!...")
        print("finished images", i)

print("end...")