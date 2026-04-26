"""
项目默认配置文件。

职责：
1. 统一管理采样率、切块长度、说话人数等全局设置；
2. 给训练脚本提供模型超参数、数据路径和训练器参数；
3. 作为实验默认配置的单一入口。

主要输入：
- 本文件本身不接收运行时输入，而是被 `train.py` 直接导入

主要输出：
- `nnet_conf`: Conv-TasNet 构造参数
- `train_data` / `dev_data`: 数据集 scp 路径和采样率
- `trainer_conf`: 优化器和调度器配置
"""

from pathlib import Path

fs = 8000
chunk_len = 4  # (s)
chunk_size = chunk_len * fs
num_spks = 2

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = WORKSPACE_ROOT / "OpenSLR" / "data_for_convtasnet" / "aishell_musan_8k_seg4_shared_snr0_10"


def _subset_dir(name):
    return str((DATA_ROOT / name).resolve()) + "/"

# 模型结构相关配置，会透传给 `ConvTasNet(**nnet_conf)`。
nnet_conf = {
    "L": 20,
    "N": 256,
    "X": 8,
    "R": 4,
    "B": 256,
    "H": 512,
    "P": 3,
    "norm": "BN",
    "num_spks": num_spks,
    "non_linear": "relu"
}

# 训练/验证数据位置，默认使用 Kaldi 风格的 wav scp。
train_dir = _subset_dir("train")
dev_dir = _subset_dir("dev")

train_data = {
    "mix_scp":
    train_dir + "mix.scp",
    "ref_scp":
    [train_dir + "spk{:d}.scp".format(n) for n in range(1, 1 + num_spks)],
    "sample_rate":
    fs,
}

dev_data = {
    "mix_scp": dev_dir + "mix.scp",
    "ref_scp":
    [dev_dir + "spk{:d}.scp".format(n) for n in range(1, 1 + num_spks)],
    "sample_rate": fs,
}

# 优化器与学习率调度配置。
adam_kwargs = {
    "lr": 1e-3,
    "weight_decay": 1e-5,
}

trainer_conf = {
    "optimizer": "adam",
    "optimizer_kwargs": adam_kwargs,
    "min_lr": 1e-8,
    "patience": 2,
    "factor": 0.5,
    "logging_period": 200  # batch number
}
