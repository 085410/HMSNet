import argparse
import os
import pprint

import logging
import timeit

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import models

from settings import config
from settings import update_config


from train_eval_tools.loss_utils.criterion  import CrossEntropy
from train_eval_tools.function import train, validate
from train_eval_tools.utils import create_logger, FullModel

import settings.datasets_process

def parse_args():
    # python train_eval_tools/train.py --cfg settings/data_cfg/camvid/pidnet_small_camvid.yaml GPUS [0] TRAIN.BATCH_SIZE_PER_GPU 6
    # 通过 argparse 库来解析命令行参数
    parser = argparse.ArgumentParser(description='Train segmentation network')

    # 参数：配置文件
    parser.add_argument('--cfg', # 解析yaml格式文件
                        help='experiment configure file name',
                        default="settings/data_cfg/cityscapes/hilbertnet_cityscapes_trainval.yaml",
                        type=str)
    # 参数：随机种子
    parser.add_argument('--seed',
                        type=int,
                        default=304)

    # 参数：修改配置
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    # 为什么设置随机种子：在机器学习任务中，很多操作（比如权重初始化、数据打乱、随机数据增强等）都涉及到随机性。
    # 由于这些操作中存在随机性，每次运行相同的代码，模型的训练结果可能会有差异。
    # 通过设置随机种子，我们可以固定这些随机操作的初始状态，从而使得模型在每次运行时都得到相同的结果，确保实验的可重复性。
    if args.seed > 0:
        import random
        print('正在使用的随机种子---------------Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # 创建日志记录器
    # logger：记录训练时的超参数，进展，损失函数，精度等信息
    # final_output_dir：指定模型的checkpoint和日志文件的保持路径
    # tb_log_dir：为TensorBoard生成的日志指定目录，可视化训练过程
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    # 设置TensorBoard 记录器
    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    if torch.cuda.device_count() != len(gpus):
        print("The gpu numbers do not match!")
        return 0

    # =========================这里可能需要修改模型文件名===================================================
    model = models.vm_hilbertnet.get_tarin_model(config, imgnet_pretrained=False)

    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    print(f"GPU数目：{len(gpus)}")
    print(f"BatchSize 设置为：{batch_size}")

    # prepare data ====================================================================
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('settings.datasets_process.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TRAIN_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=config.TRAIN.MULTI_SCALE,
        flip=config.TRAIN.FLIP,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TRAIN.BASE_SIZE,
        crop_size=crop_size,
        scale_factor=config.TRAIN.SCALE_FACTOR)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True)

    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('settings.datasets_process.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)

    # 根据配置文件决定使用 OHEM（在线难例挖掘）交叉熵损失还是标准交叉熵损失。======================================
    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                         thres=config.LOSS.OHEMTHRES,
                                         min_kept=config.LOSS.OHEMKEEP,
                                         weight=train_dataset.class_weights)
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     weight=train_dataset.class_weights)

    model = FullModel(model, sem_criterion)
    # model = nn.DataParallel(model, device_ids=gpus).cuda()
    model = model.cuda()

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV,
                                    )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))

    best_mIoU = 0
    last_epoch = 0
    flag_rm = config.TRAIN.RESUME

    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']

            model.load_state_dict(
                {k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    # ===================================== 训练部分 =====================================
    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    real_end = end_epoch
    print(f'==============end_epoch值: {real_end}' )

    # 定义验证间隔，确保总共验证 10 次
    validation_interval = real_end // 10  # 每隔 real_end/10 轮进行一次验证

    for epoch in range(last_epoch, real_end):

        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)
        # 开始训练
        train(config, epoch, config.TRAIN.END_EPOCH,
              epoch_iters, config.TRAIN.LR, num_iters,
              trainloader, optimizer, model, writer_dict)

        # 每过 real_end / 10 轮，或者最后 10 轮时进行验证
        if flag_rm == 1 or (epoch % validation_interval == 0 and epoch < real_end - 10) or (epoch >= real_end - 10):
            print(f"===============================开始验证，当前epoch：{epoch}======================================")
            valid_loss, mean_IoU, IoU_array = validate(config, testloader, model, writer_dict)

            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                # 保存最佳权重文件
                torch.save(model.state_dict(),
                           os.path.join(final_output_dir, 'best.pt'))

            # 设置打印信息
            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                valid_loss, mean_IoU, best_mIoU)

            logging.info(msg)
            logging.info(IoU_array)

            # 保存权重
            logger.info(f'=> 保存 checkpoint 权重到 {final_output_dir}/checkpoint_{epoch}.pth.tar')
            torch.save({
                'epoch': epoch + 1,
                'best_mIoU': best_mIoU,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, f'checkpoint_{epoch}.pth.tar'))

        # 如果满足 flag_rm == 1，进行验证后重置 flag_rm
        if flag_rm == 1:
            flag_rm = 0

    # 保存最终权重
    torch.save(model.state_dict(),
               os.path.join(final_output_dir, 'final_state.pt'))

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % np.int32((end - start) / 3600))
    logger.info('Done')


if __name__ == '__main__':
    main()
