# wujian@2018
"""
音频读写与 scp 解析工具。

职责：
1. 读写 wav 文件；
2. 解析 Kaldi 风格的 `*.scp`；
3. 提供通用 Reader 和专用于波形读取的 WaveReader。

这些工具是数据加载、推理导出和评估脚本的基础。
"""

import os
import numpy as np
import scipy.io.wavfile as wf

MAX_INT16 = np.iinfo(np.int16).max


def write_wav(fname, samps, fs=16000, normalize=True):
    """
    将波形写成 int16 wav 文件。

    输入：
    - fname: 输出路径
    - samps: `np.ndarray[S]` 或多通道波形
    - fs: 采样率
    - normalize: 是否按 int16 范围缩放

    输出：
    - 无显式返回值，文件写入磁盘
    """
    if normalize:
        samps = samps * MAX_INT16
    # scipy.io.wavfile.write could write single/multi-channel files
    # for multi-channel, accept ndarray [Nsamples, Nchannels]
    if samps.ndim != 1 and samps.shape[0] < samps.shape[1]:
        samps = np.transpose(samps)
        samps = np.squeeze(samps)
    # same as MATLAB and kaldi
    samps_int16 = samps.astype(np.int16)
    fdir = os.path.dirname(fname)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    # NOTE: librosa 0.6.0 seems could not write non-float narray
    #       so use scipy.io.wavfile instead
    wf.write(fname, fs, samps_int16)


def read_wav(fname, normalize=True, return_rate=False):
    """
    读取 wav 文件。

    输入：
    - fname: wav 路径
    - normalize: 是否归一化到 `[-1, 1]`
    - return_rate: 是否同时返回采样率

    输出：
    - `np.ndarray[S]` 或 `np.ndarray[C, S]`
    - 当 `return_rate=True` 时返回 `(sample_rate, samps)`
    """
    # samps_int16: N x C or N
    #   N: number of samples
    #   C: number of channels
    samp_rate, samps_int16 = wf.read(fname)
    # N x C => C x N
    samps = samps_int16.astype(np.float32)
    # tranpose because I used to put channel axis first
    if samps.ndim != 1:
        samps = np.transpose(samps)
    # normalize like MATLAB and librosa
    if normalize:
        samps = samps / MAX_INT16
    if return_rate:
        return samp_rate, samps
    return samps


def parse_scripts(scp_path, value_processor=lambda x: x, num_tokens=2):
    """
    解析 Kaldi 风格的脚本文件。

    输入：
    - scp_path: 脚本路径
    - value_processor: 对 value 做后处理的函数
    - num_tokens: 每行期望的 token 数

    输出：
    - `dict[key, value]`
    """
    scp_dict = dict()
    line = 0
    with open(scp_path, "r") as f:
        for raw_line in f:
            scp_tokens = raw_line.strip().split()
            line += 1
            if num_tokens >= 2 and len(scp_tokens) != num_tokens or len(
                    scp_tokens) < 2:
                raise RuntimeError(
                    "For {}, format error in line[{:d}]: {}".format(
                        scp_path, line, raw_line))
            if num_tokens == 2:
                key, value = scp_tokens
            else:
                key, value = scp_tokens[0], scp_tokens[1:]
            if key in scp_dict:
                raise ValueError("Duplicated key \'{0}\' exists in {1}".format(
                    key, scp_path))
            scp_dict[key] = value_processor(value)
    return scp_dict


class Reader(object):
    """
    通用脚本读取器。

    输入：
    - scp_path: Kaldi 风格脚本文件

    输出：
    - 支持按 key、按索引和按顺序读取
    - 默认 `_load()` 直接返回脚本中的 value
    """

    def __init__(self, scp_path, value_processor=lambda x: x):
        self.index_dict = parse_scripts(
            scp_path, value_processor=value_processor, num_tokens=2)
        self.index_keys = list(self.index_dict.keys())

    def _load(self, key):
        # return path
        return self.index_dict[key]

    # number of utterance
    def __len__(self):
        return len(self.index_dict)

    # avoid key error
    def __contains__(self, key):
        return key in self.index_dict

    # sequential index
    def __iter__(self):
        for key in self.index_keys:
            yield key, self._load(key)

    # random index, support str/int as index
    def __getitem__(self, index):
        if type(index) not in [int, str]:
            raise IndexError("Unsupported index type: {}".format(type(index)))
        if type(index) == int:
            # from int index to key
            num_utts = len(self.index_keys)
            if index >= num_utts or index < 0:
                raise KeyError(
                    "Interger index out of range, {:d} vs {:d}".format(
                        index, num_utts))
            index = self.index_keys[index]
        if index not in self.index_dict:
            raise KeyError("Missing utterance {}!".format(index))
        return self._load(index)


class WaveReader(Reader):
    """
    基于 wav.scp 的波形读取器。

    输入：
    - wav_scp: `key path/to/audio.wav` 格式的 scp
    - sample_rate: 可选，若给定则强制校验采样率
    - normalize: 是否归一化波形幅值

    输出：
    - `__getitem__` / `__iter__` 返回对应 key 的波形数组
    """

    def __init__(self, wav_scp, sample_rate=None, normalize=True):
        super(WaveReader, self).__init__(wav_scp)
        self.samp_rate = sample_rate
        self.normalize = normalize

    def _load(self, key):
        """
        按 key 读取一条波形，并做采样率校验。

        输出：
        - 单通道时为 `np.ndarray[S]`
        - 多通道时为 `np.ndarray[C, S]`
        """
        # return C x N or N
        samp_rate, samps = read_wav(
            self.index_dict[key], normalize=self.normalize, return_rate=True)
        # if given samp_rate, check it
        if self.samp_rate is not None and samp_rate != self.samp_rate:
            raise RuntimeError("SampleRate mismatch: {:d} vs {:d}".format(
                samp_rate, self.samp_rate))
        return samps
