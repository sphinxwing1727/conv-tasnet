#!/usr/bin/env python

# wujian@2018
"""
推理入口脚本。

职责：
1. 从 checkpoint 目录恢复训练好的 Conv-TasNet；
2. 读取 `mix.scp` 中列出的混合语音；
3. 对每条混合语音做分离，并将结果写成 wav 文件。

主要输入：
- 命令行参数：checkpoint 目录、输入 scp、采样率、输出目录、GPU

主要输出：
- `dump-dir/spk1/*.wav`, `dump-dir/spk2/*.wav`, ...
"""

import os
import argparse

import torch as th
import numpy as np

from conv_tas_net import ConvTasNet

from libs.utils import load_json, get_logger
from libs.audio import WaveReader, write_wav

logger = get_logger(__name__)


class NnetComputer(object):
    """
    推理时使用的模型封装。

    输入：
    - cpt_dir: checkpoint 目录，内部需要有 `mdl.json` 和 `best.pt.tar`
    - gpuid: 推理设备，`-1` 表示 CPU

    输出：
    - `compute()` 返回 `List[np.ndarray]`，每个元素是一位说话人的分离波形
    """

    def __init__(self, cpt_dir, gpuid):
        self.device = th.device(
            "cuda:{}".format(gpuid)) if gpuid >= 0 else th.device("cpu")
        nnet = self._load_nnet(cpt_dir)
        self.nnet = nnet.to(self.device) if gpuid >= 0 else nnet
        # set eval model
        self.nnet.eval()

    def _load_nnet(self, cpt_dir):
        """
        根据保存的模型配置和参数恢复网络。

        输入：
        - cpt_dir: checkpoint 目录

        输出：
        - ConvTasNet 实例，权重已加载，但尚未移动到设备
        """
        nnet_conf = load_json(cpt_dir, "mdl.json")
        nnet = ConvTasNet(**nnet_conf)
        cpt_fname = os.path.join(cpt_dir, "best.pt.tar")
        cpt = th.load(cpt_fname, map_location="cpu")
        nnet.load_state_dict(cpt["model_state_dict"])
        logger.info("Load checkpoint from {}, epoch {:d}".format(
            cpt_fname, cpt["epoch"]))
        return nnet

    def compute(self, samps):
        """
        对单条混合语音做前向分离。

        输入：
        - samps: `np.ndarray[S]`，单通道混合语音

        输出：
        - `List[np.ndarray[S']]`，每个元素对应一路分离结果
        """
        with th.no_grad():
            raw = th.tensor(samps, dtype=th.float32, device=self.device)
            sps = self.nnet(raw)
            sp_samps = [np.squeeze(s.detach().cpu().numpy()) for s in sps]
            return sp_samps


def run(args):
    """
    读取 `mix.scp` 并批量导出分离结果。

    输入：
    - args.input: Kaldi 风格的 wav scp
    - args.checkpoint: checkpoint 目录
    - args.dump_dir: 结果输出目录

    输出：
    - 无显式返回值
    - 分离结果 wav 会落到 `args.dump_dir`
    """
    mix_input = WaveReader(args.input, sample_rate=args.fs)
    computer = NnetComputer(args.checkpoint, args.gpu)
    spk_scps = []
    for key, mix_samps in mix_input:
        logger.info("Compute on utterance {}...".format(key))
        spks = computer.compute(mix_samps)
        if not spk_scps:
            spk_scps = [[] for _ in range(len(spks))]
        norm = np.linalg.norm(mix_samps, np.inf)
        for idx, samps in enumerate(spks):
            samps = samps[:mix_samps.size]
            # norm
            samps = samps * norm / np.max(np.abs(samps))
            wav_path = os.path.abspath(
                os.path.join(args.dump_dir, "spk{}/{}.wav".format(
                    idx + 1, key)))
            write_wav(wav_path, samps, fs=args.fs)
            spk_scps[idx].append("{} {}".format(key, wav_path))
    for idx, lines in enumerate(spk_scps):
        scp_path = os.path.join(args.dump_dir, "spk{}.scp".format(idx + 1))
        with open(scp_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")
        logger.info("Write script {}".format(scp_path))
    logger.info("Compute over {:d} utterances".format(len(mix_input)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to do speech separation in time domain using ConvTasNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint", type=str, help="Directory of checkpoint")
    parser.add_argument(
        "--input", type=str, required=True, help="Script for input waveform")
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="GPU device to offload model to, -1 means running on CPU")
    parser.add_argument(
        "--fs", type=int, default=8000, help="Sample rate for mixture input")
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="sps_tas",
        help="Directory to dump separated results out")
    args = parser.parse_args()
    run(args)
