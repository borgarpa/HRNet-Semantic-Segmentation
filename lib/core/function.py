# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F


from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate

import utils.distributed as dist


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, schedulers, model, writer_dict):
    # Model is instantiated by utils.FullModel to include loss calculation
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, labels_haze, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
        labels_haze = labels_haze.long().cuda()

        (losses_all, _), (losses_haze, _) = model(images, [labels, labels_haze])
        loss = losses_all.mean() + losses_haze.mean()

        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        ### TODO: Update lr scheduler to accomodate Reduce On Plateu
        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average())
            logging.info(msg)

    # for scheduler in schedulers:
    #   scheduler.step()

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict):
    model.eval()
    
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    confusion_matrix_haze = np.zeros(
        (config.DATASET.NUM_CLASSES_HAZE, config.DATASET.NUM_CLASSES_HAZE, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, labels_haze, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            labels_haze = labels_haze.long().cuda()

            (losses_all, pred_all), (losses_haze, pred_haze) = model(image, [label, labels_haze])
            # All classes
            if not isinstance(pred_all, (list, tuple)):
                pred_all = [pred_all]

            for i, x in enumerate(pred_all):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )
            # Binary haze class
            if not isinstance(pred_haze, (list, tuple)):
                pred_haze = [pred_haze]
            for i, x in enumerate(pred_haze):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix_haze[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES_HAZE,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)

            loss = losses_all.mean() + losses_haze.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

        confusion_matrix_haze = torch.from_numpy(confusion_matrix_haze).cuda()
        reduced_confusion_matrix_haze = reduce_tensor(confusion_matrix_haze)
        confusion_matrix_haze = reduced_confusion_matrix_haze.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))

        pos_haze = confusion_matrix_haze[..., i].sum(1)
        res_haze = confusion_matrix_haze[..., i].sum(0)
        tp_haze = np.diag(confusion_matrix_haze[..., i])
        IoU_array_haze = (tp / np.maximum(1.0, pos_haze + res_haze - tp_haze))
        mean_IoU = (IoU_array.mean() + IoU_array_haze.mean())/2
        if dist.get_rank() <= 0:
            logging.info('{} {} {} {} {} {}'.format(i, IoU_array, IoU_array_haze, IoU_array.mean(), IoU_array_haze.mean(), mean_IoU))

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array, IoU_array_haze


def testval(config, test_dataset, testloader, model,
            sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    confusion_matrix_haze = np.zeros(
        (config.DATASET.NUM_CLASSES_HAZE, config.DATASET.NUM_CLASSES_HAZE))

    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, label_haze, _, name, *border_padding = batch
            size = label.size()
            size_haze = label_haze.size()
            pred_all, pred_haze = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if len(border_padding) > 0:
                border_padding = border_padding[0]
                pred_all = pred_all[:, :, 0:pred_all.size(2) - border_padding[0], 0:pred_all.size(3) - border_padding[1]]
                pred_haze = pred_haze[:, :, 0:pred_haze.size(2) - border_padding[0], 0:pred_haze.size(3) - border_padding[1]]

            if pred_all.size()[-2] != size[-2] or pred_all.size()[-1] != size[-1]:
                pred_all = F.interpolate(
                    pred_all, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
            if pred_haze.size()[-2] != size[-2] or pred_haze.size()[-1] != size[-1]:
                pred_haze = F.interpolate(
                    pred_haze, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            confusion_matrix += get_confusion_matrix(
                label,
                pred_all,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
            confusion_matrix_haze += get_confusion_matrix(
                label_haze,
                pred_haze,
                size_haze,
                config.DATASET.NUM_CLASSES_HAZE,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred([pred_all, pred_haze], sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name, folder = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, folder, name)
