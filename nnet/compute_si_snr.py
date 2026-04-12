#!/usr/bin/env python

# wujian@2018
"""
评估入口脚本，用于统计分离结果的 SI-SNR / SI-SDR 表现。

职责：
1. 读取分离结果和参考语音对应的 scp；
2. 支持单说话人与多说话人两种评估模式；
3. 多说话人场景下自动做 permutation matching；
4. 按整体或性别分组输出平均指标。
"""

import argparse

from tqdm import tqdm

from collections import defaultdict
from libs.metric import si_snr, permute_si_snr
from libs.audio import WaveReader, SpeakersReader, Reader


class Report(object):
    """
    简单的指标累加器。

    输入：
    - `add(key, val)` 中的 key 是 utterance id，val 是该条语音的 SI-SNR

    输出：
    - `report()` 将打印按性别或整体聚合后的平均分数
    """

    def __init__(self, spk2gender=None):
        self.s2g = Reader(spk2gender) if spk2gender else None
        self.snr = defaultdict(float)
        self.cnt = defaultdict(int)

    def add(self, key, val):
        gender = "NG"
        if self.s2g:
            gender = self.s2g[key]
        # defaultdict 会自动初始化不存在的 key，所以这里不需要判断 gender 是否在 self.snr 中
        self.snr[gender] += val
        self.cnt[gender] += 1

    def report(self):
        print("SI-SDR(dB) Report: ")
        for gender in self.snr:
            tot_snrs = self.snr[gender]
            num_utts = self.cnt[gender]
            print("{}: {:d}/{:.3f}".format(gender, num_utts,
                                           tot_snrs / num_utts))


def run(args):
    """
    根据命令行参数完成评估。

    输入：
    - args.sep_scp: 分离结果 scp，支持单个或逗号分隔的多个 scp
    - args.ref_scp: 参考语音 scp，格式与 sep_scp 对齐
    - args.spk2gender: 可选，说话人性别映射

    输出：
    - 无显式返回值
    - 指标通过标准输出打印
    """
    single_speaker = len(args.sep_scp.split(",")) == 1
    reporter = Report(args.spk2gender)

    if single_speaker:
        sep_reader = WaveReader(args.sep_scp)
        ref_reader = WaveReader(args.ref_scp)
        for key, sep in tqdm(sep_reader):
            ref = ref_reader[key]
            if sep.size != ref.size:
                end = min(sep.size, ref.size)
                sep = sep[:end]
                ref = ref[:end]
            snr = si_snr(sep, ref)
            reporter.add(key, snr)
    else:
        sep_reader = SpeakersReader(args.sep_scp)
        ref_reader = SpeakersReader(args.ref_scp)
        for key, sep_list in tqdm(sep_reader):
            ref_list = ref_reader[key]
            if sep_list[0].size != ref_list[0].size:
                end = min(sep_list[0].size, ref_list[0].size)
                sep_list = [s[:end] for s in sep_list]
                ref_list = [s[:end] for s in ref_list]
            snr = permute_si_snr(sep_list, ref_list)
            reporter.add(key, snr)
    reporter.report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to compute SI-SDR, as metric of the separation quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "sep_scp",
        type=str,
        help="Separated speech scripts, waiting for measure"
        "(support multi-speaker, egs: spk1.scp,spk2.scp)")
    parser.add_argument(
        "ref_scp",
        type=str,
        help="Reference speech scripts, as ground truth for"
        " SI-SDR computation")
    parser.add_argument(
        "--spk2gender",
        type=str,
        default="",
        help="If assigned, report results per gender")
    args = parser.parse_args()
    run(args)
