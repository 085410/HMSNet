import logging
import os
import time

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

import torch
from torch.nn import functional as F

from train_eval_tools.utils import AverageMeter
from train_eval_tools.utils import get_confusion_matrix
from train_eval_tools.utils import adjust_learning_rate

camvid_class_id_to_name = {
    0: 'sky', 1: 'building', 2: 'pole', 3: 'road', 4: 'pavement',
    5: 'tree', 6: 'signsymbol', 7: 'fence', 8: 'car', 9: 'pedestrian',
    10: 'bicyclist'
}

cityscapes_class_id_to_name = {
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'traffic light',
    7: 'traffic sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle'
}

# 该函数执行一次训练循环，负责处理 前向传播、损失计算、反向传播 和 参数更新
"""
    config：配置文件。
    epoch：当前 epoch
    num_epoch：总的 epoch 数
    epoch_iters：每个 epoch 的迭代次数。
    base_lr：基础学习率
    num_iters：总的迭代次数，用于调整学习率。
    trainloader：训练集的数据加载器。
    optimizer：用于优化模型参数的优化器。
    model：待训练的分割模型。
    writer_dict：字典，包含用于记录训练过程的 TensorBoard。
"""


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    avg_detail_loss1 = AverageMeter()
    avg_context_loss2 = AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        # 从训练数据中获取输入图像、真实标签和边界标签
        images, labels, _, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()

        losses, _, acc, loss_list = model(images, labels)
        loss = losses.mean()
        acc = acc.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        avg_detail_loss1.update(loss_list[0].mean().item())
        avg_context_loss2.update(loss_list[1].mean().item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter + cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}, loss_1: {:.6f}, loss_2: {:.6f}'.format(
                epoch, num_epoch, i_iter, epoch_iters,
                batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                ave_acc.average(), avg_detail_loss1.average(), avg_context_loss2.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1


# 在验证集上对模型进行评估，计算模型在验证集上的 损失 和 IoU，并记录到日志中
def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))

    # 根据数据集选择类别名称映射
    if config.DATASET.DATASET == "cityscapes":
        class_id_to_name = cityscapes_class_id_to_name
    elif config.DATASET.DATASET == "camvid":
        class_id_to_name = camvid_class_id_to_name
    else:
        raise ValueError("=============Unsupported dataset!============")

    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, _, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()

            losses, pred, _, _ = model(image, label)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]

            for i, x in enumerate(pred):
                x = F.interpolate(
                    # size 代表的是输入 label 的维度，通常在图像分割中，label 是一个四维张量，形状为 [batch_size, C, H, W]
                    # size[-2:] 用到了 Python 的切片语法，size[-2:] 表示从 size 这个张量中，获取最后两个维度，也就是 height 和 width
                    # 将 x 的尺寸调整为目标 label 的尺寸（即 H 和 W）
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

            if idx % 10 == 0:
                print(f"Processed {idx} batches")

            # 更新损失
            loss = losses.mean()
            ave_loss.update(loss.item())

    # 计算每个输出的 IoU 和分类准确率
    table_data = []
    for i in range(nums):
        # 计算 TP + FN
        pos = confusion_matrix[..., i].sum(1)
        # 计算 TP + FP
        res = confusion_matrix[..., i].sum(0)
        # TP
        tp = np.diag(confusion_matrix[..., i])
        # Iou
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()

    # 将每个类别的 IoU 和准确率添加到表格数据中
    for class_id, iou in enumerate(IoU_array):
        class_name = class_id_to_name.get(class_id, f"Class {class_id}")  # 获取类别名称
        acc = tp[class_id] / np.maximum(1.0, pos[class_id])
        table_data.append([class_name, f"{iou:.4f}", f"{acc:.4f}"])

    # 打印成表格
    print("-------------------------------------")
    header = ["Class", "IoU", "Accuracy"]
    logging.info("\n" + tabulate(table_data, headers=header, tablefmt="simple"))
    print("-------------------------------------")
    mean_acc = np.mean([tp[class_id] / np.maximum(1.0, pos[class_id]) for class_id in range(len(IoU_array))])
    logging.info(f"Overall Mean IoU: {mean_IoU:.4f}, Mean Accuracy: {mean_acc:.4f}")
    print("\n")

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1

    return ave_loss.average(), mean_IoU, IoU_array


# 在测试集上评估模型性能，并可选保存模型的预测结果
def testval(config, test_dataset, testloader, model, sv_dir='./', sv_pred=True):
    model.eval()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    # 根据数据集选择类别名称映射
    if config.DATASET.DATASET == "cityscapes":
        class_id_to_name = cityscapes_class_id_to_name
    elif config.DATASET.DATASET == "camvid":
        class_id_to_name = camvid_class_id_to_name
    else:
        raise ValueError("=============Unsupported dataset!============")

    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, _, name = batch
            size = label.size()
            # 确保输入数据image被转移到 GPU
            image = image.cuda()
            pred = test_dataset.single_scale_inference(config, model, image)

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info(f' processing: {index} images')
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    # 计算每个输出的 IoU 和分类准确率
    table_data = []
    for class_id, iou in enumerate(IoU_array):
        class_name = class_id_to_name.get(class_id, f"Class {class_id}")  # 获取类别名称
        acc = tp[class_id] / np.maximum(1.0, pos[class_id])
        table_data.append([class_name, f"{iou:.4f}", f"{acc:.4f}"])

    # 打印成表格
    header = ["Class", "IoU", "Accuracy"]
    logging.info("\n" + tabulate(table_data, headers=header, tablefmt="grid"))
    print("\n")
    # 打印总体平均 IoU 和分类准确率
    logging.info(f"Overall Mean IoU: {mean_IoU:.4f}, Pixel Acc: {pixel_acc:.4f}, Mean Accuracy: {mean_acc:.4f}")

    return mean_IoU, IoU_array, pixel_acc, mean_acc
