#!/usr/bin/env python

# wujian@2018
"""
训练入口脚本。

职责：
1. 从 `conf.py` 读取模型、数据和训练器配置；
2. 构造 Conv-TasNet、数据加载器和训练器；
3. 启动训练，并把关键配置落盘到 checkpoint 目录。

主要输入：
- 命令行参数：GPU 列表、epoch 数、batch size、checkpoint 路径等
- `conf.py`：数据路径、模型超参数、优化器配置

主要输出：
- `best.pt.tar` / `last.pt.tar`
- `mdl.json` / `trainer.json` / `data.json`
- 训练日志
"""

import os
import pprint
import argparse
import random

from libs.trainer import SiSnrTrainer
from libs.dataset import make_dataloader
from libs.utils import dump_json, get_logger

from conv_tas_net import ConvTasNet
from conf import trainer_conf, nnet_conf, train_data, dev_data, chunk_size

def run(args):
    """
    组装训练所需对象并启动训练。

    输入：
    - args.gpus: 形如 "0,1" 的 GPU 字符串
    - args.checkpoint: checkpoint 输出目录
    - args.batch_size / args.num_workers / args.epochs: 训练超参数

    输出：
    - 无显式返回值
    - 训练结果会写入 `args.checkpoint`
    """
    gpuids = tuple(map(int, args.gpus.split(",")))

    nnet = ConvTasNet(**nnet_conf)
    trainer = SiSnrTrainer(nnet,
                           gpuid=gpuids,
                           checkpoint=args.checkpoint,
                           resume=args.resume,
                           **trainer_conf)

    data_conf = {
        "train": train_data,
        "dev": dev_data,
        "chunk_size": chunk_size
    }
    for conf, fname in zip([nnet_conf, trainer_conf, data_conf],
                           ["mdl.json", "trainer.json", "data.json"]):
        dump_json(conf, args.checkpoint, fname)

    train_loader = make_dataloader(train=True,
                                   data_kwargs=train_data,
                                   batch_size=args.batch_size,
                                   chunk_size=chunk_size,
                                   num_workers=args.num_workers)
    dev_loader = make_dataloader(train=False,
                                 data_kwargs=dev_data,
                                 batch_size=args.batch_size,
                                 chunk_size=chunk_size,
                                 num_workers=args.num_workers)

    trainer.run(train_loader, dev_loader, num_epochs=args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to start ConvTasNet training, configured from conf.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpus",
                        type=str,
                        default="0,1",
                        help="Training on which GPUs "
                        "(one or more, egs: 0, \"0,1\")")
    parser.add_argument("--epochs",
                        type=int,
                        default=50,
                        help="Number of training epochs")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Directory to dump models")
    parser.add_argument("--resume",
                        type=str,
                        default="",
                        help="Exist model to resume training from")
    parser.add_argument("--batch-size",
                        type=int,
                        default=16,
                        help="Number of utterances in each batch")
    parser.add_argument("--num-workers",
                        type=int,
                        default=4,
                        help="Number of workers used in data loader")
    args = parser.parse_args()
    logger = get_logger(__name__,
                        file=os.path.join(args.checkpoint, "train.log"),
                        console=True)
    logger.info("Arguments in command:\n{}".format(pprint.pformat(vars(args))))
    logger.info("TensorBoard command:\n  tensorboard --logdir {}".format(
        os.path.join(args.checkpoint, "tensorboard")))

    run(args)
