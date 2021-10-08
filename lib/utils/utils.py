# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from itertools import tee

import os
import logging
import time
from pathlib import Path

import re
import cv2
import math
import random
import numpy as np
from numpy.core.records import array
import rasterio as rio
import rasterio
from tqdm import tqdm
from rasterio import path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR

class FullModel(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, inputs, labels, *args, **kwargs):
    out_list = []
    outputs = self.model(inputs, *args, **kwargs)
    for i, loss_ in enumerate(self.loss):
        loss_ = loss_(outputs[i], labels[i])
        out_list.append((torch.unsqueeze(loss_,0), outputs))
    return out_list

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9, nbb_mult=10):
    # scheduler = ExponentialLR(optimizer, gamma=(1-float(cur_iters)/max_iters)**(power))
    # scheduler.step()
    # lr = scheduler.get_last_lr()
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr


def gen_artificial_data(size, nclasses, nsamples, out_path, listpath):

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(os.path.dirname(listpath)):
        os.makedirs(os.path.dirname(listpath))

    # filelist_ = []
    with open(listpath, 'w+') as f:
        for n in range(nsamples):
                
            array = np.random.randn(*size).astype(np.float32)*65535
            mask = np.random.uniform(1, nclasses+1, size[:-1]).astype(np.uint8)

            # np.save(os.path.join(out_path, f'array_{n}.npy'), array)
            cv2.imwrite(os.path.join(out_path, f'mask_{n}.png'), mask)

            with rio.open(os.path.join(out_path, f'array_{n}.tif'),
                'w',
                driver='GTiff',
                height=size[0],
                width=size[1],
                count=size[-1],
                dtype=np.float32) as dst:
                dst.write(array.transpose(-1, 0, 1))

            write_path = re.findall('\\\\data\\\\(.+)', out_path)[0]
            f.write(os.path.join(write_path, f'array_{n}.tif') + '\t' + os.path.join(write_path, f'mask_{n}.png'+'\n'))
        f.close()

def gen_artificial_dataset(size, nclasses, nsamples, dataset_path, listpath):

    for path_ in ['train', 'test', 'val']:
        gen_artificial_data(size, nclasses, nsamples,
        os.path.join(dataset_path, path_),
        os.path.join(listpath, path_+'.lst'))

def get_filelist(path):

    filelist = []
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    for folder in folders:
        images = list(filter(lambda x: x.endswith('.tif'), os.listdir(os.path.join(path, folder))))
        masks = list(filter(lambda x: x.endswith('.png'), os.listdir(os.path.join(path, folder))))
        for image, mask in zip(images, masks):
            filelist.append([os.path.join(path, folder, image), os.path.join(path, folder, mask)])

    return filelist

def split_dataset(filelist, train=0.6, test=0.2, validation=0.2, shuffle=True):
    assert train == 1 - (test + validation)

    if shuffle:
        random.shuffle(filelist)
    length = len(filelist)

    train_length = length - (math.floor(length*validation) + math.floor(length*test))
    validation_length = math.floor(length*validation) 

    train_set = filelist[ : train_length]
    validation_set = filelist[train_length : train_length+validation_length]
    test_set = filelist[train_length+validation_length : ]
    
    assert len(train_set) + len(validation_set) + len(test_set) == length

    return train_set, validation_set, test_set

def write_dataset(path, train, validation, test):

    for set_, name in zip([train, validation, test], ['train', 'val', 'test']):
        with open(os.path.join(path, name+'.lst'), 'w+') as f:
            for image, mask in tqdm(set_, desc=f"Writing {name} set"):
                f.write(f'{image}\t{mask}\n')
            f.close()

def compute_train_stats(train, ignore=[0, 1], class_map=[2,3,4,5,6,7], return_stats=['mean', 'std', 'weights']):
    assert len(return_stats) > 0
    # def assert_(x):
    #     x = rasterio.open(x[0]).read().transpose((1, 2, 0))
    #     if x.shape != (512, 512, 12):
    #         print(x.shape)
    #     x = np.max(x, axis=(0, 1))
    #     if (x == np.nan).any() or (x == 0).any():
    #         print(x)

    # for x in tqdm(train, desc="Asserting"):
    #     assert_(x)

    def mean_(x):
        x = x/np.max(x, axis=(0, 1))
        x = np.mean(x, axis=(0, 1))
        return x
    
    def std_(x):
        x = x/np.max(x, axis=(0, 1))
        x = np.std(x, axis=(0, 1))
        return x

    means = []
    stds = []
    pixels = {class_: 0 for class_ in class_map}
    for sample in tqdm(train, desc='Computing stats'):

        if 'mean' or 'std' in return_stats:
            if sample[0].endswith('.tif'):
                x = rasterio.open(sample[0]).read().transpose((1, 2, 0))
            else:
                x = np.load(sample[0])
            means.append(mean_(x))
            stds.append(std_(x))

        if 'weights' in return_stats:
            y = cv2.imread(sample[1])
            if ignore:
                for l in ignore:
                    y = y[y != l]
            classes, counts = np.unique(y, return_counts=True)
            for class_, count in zip(classes, counts):
                pixels[class_] += count

    return_ = []
    if 'mean' in return_stats:
        mean = np.mean(np.array(means), axis=0)
        return_.append(mean)

    if 'std' in return_stats:
        std = np.mean(np.array(stds), axis=0)
        return_.append(std)

    if 'weights' in return_stats:
        counts = [count for count in pixels.values()]
        weights = 1 / np.log1p(counts)
        weights = len(class_map) * weights / np.sum(weights)
        for class_, weight in zip(pixels.keys(), weights): 
            pixels[class_] = weight
        return_.append(pixels)

    return return_

if __name__ == "__main__":
    # outpath = '.\data\customdataset'
    # gen_artificial_dataset((512, 512, 12), 3, 15, outpath, listpath)
    listpath = '.\data\list\customdataset'

    path = os.path.normpath("D:\Cloudless\data\SENTINEL_2_reference_cloud_masks_Baetens_Hagolle\Reference_dataset\dataset")
    
    filelist = get_filelist(path)
    train_set, validation_set, test_set = split_dataset(filelist, train=0.8, test=0.2, validation=False)
    write_dataset(listpath, train_set, validation_set, test_set)

    mean, std, pixels = compute_train_stats(train_set) #return_stats=['weights']
    print('mean:\n', mean)
    print('std:\n', std)
    print('pixels:\n', pixels)
    # print(train_set[0], validation_set[0], test_set[0])
    # print(len(filelist), filelist[:2])