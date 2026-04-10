# wujian@2018
"""
评估指标工具。

职责：
1. 计算单条语音之间的 SI-SNR；
2. 在多说话人场景下枚举排列，返回最佳匹配下的平均 SI-SNR。

该文件主要被 `compute_si_snr.py` 调用。
"""

import numpy as np

from itertools import permutations


def si_snr(x, s, remove_dc=True):
    """
    计算单条分离语音与参考语音之间的 SI-SNR。

    输入：
    - x: `np.ndarray[S]`，分离或增强后的语音
    - s: `np.ndarray[S]`，参考语音
    - remove_dc: 是否先做去均值

    输出：
    - float，当前这一对语音的 SI-SNR
    """

    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    # zero mean, seems do not hurt results
    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        t = np.inner(x_zm, s_zm) * s_zm / vec_l2norm(s_zm)**2
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / vec_l2norm(s)**2
        n = x - t
    return 20 * np.log10(vec_l2norm(t) / vec_l2norm(n))


def permute_si_snr(xlist, slist):
    """
    多说话人版本的 SI-SNR。

    输入：
    - xlist: `List[np.ndarray]`，模型输出的多路分离语音
    - slist: `List[np.ndarray]`，参考语音列表

    输出：
    - float，所有说话人排列中最佳匹配下的平均 SI-SNR
    """

    def si_snr_avg(xlist, slist):
        return sum([si_snr(x, s) for x, s in zip(xlist, slist)]) / len(xlist)

    N = len(xlist)
    if N != len(slist):
        raise RuntimeError(
            "size do not match between xlist and slist: {:d} vs {:d}".format(
                N, len(slist)))
    si_snrs = []
    for order in permutations(range(N)):
        si_snrs.append(si_snr_avg(xlist, [slist[n] for n in order]))
    return max(si_snrs)
