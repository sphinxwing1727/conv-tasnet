#!/usr/bin/env python

"""
语音分离结果的波形对比绘图脚本。

数据流：
1. 命令行传入 sep_scp、ref_scp、可选 mix_scp、采样率 fs、输出目录等参数。
2. run() 根据 sep_scp/ref_scp 是否包含逗号，选择单路 WaveReader 或多路
   SpeakersReader 读取波形。
3. run() 遍历每个 utterance key，按 --keys 和 --max-utts 过滤样本，并取出
   separated、reference、可选 mixture 波形。
4. save_waveform_compare() 将同一个样本内的所有波形裁剪到相同长度，生成统一
   时间轴。
5. 如果是多说话人输出，save_waveform_compare() 调用 best_permutation()，用
   SI-SNR 穷举匹配 separated stream 和 reference stream，避免说话人顺序错位。
6. matplotlib 将 mixture、reference、estimated 波形画到子图中，并保存为
   dump_dir/key.png。
"""

import argparse
import os

from itertools import permutations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from libs.audio import WaveReader, SpeakersReader
from libs.metric import si_snr
from libs.utils import get_logger


logger = get_logger(__name__)

def parse_key_filter(keys):
    """
    解析命令行传入的 utterance key 过滤条件。

    作用：
        将 --keys 的逗号分隔字符串转换成 set，方便 run() 中快速判断是否保留样本。
    输入：
        keys: 字符串，例如 "utt1,utt2"；空字符串表示不过滤。
    输出：
        None 或 set。None 表示保留所有 key；set 表示只保留集合内的 key。
    调用时机：
        run() 开始时调用一次，用于准备后续遍历样本时的过滤条件。
    """
    # 空字符串表示不过滤；非空字符串则只保留用户指定的 utterance。
    if not keys:
        return None
    return {key.strip() for key in keys.split(",") if key.strip()}


def parse_point_range(point_range):
    """
    Parse an optional inclusive sample index range from the CLI.
    """
    if not point_range:
        return None
    bounds = [part.strip() for part in point_range.split(",")]
    if len(bounds) != 2:
        raise argparse.ArgumentTypeError(
            "point range must be formatted as start,end")
    try:
        start, end = [int(part) for part in bounds]
    except ValueError as ex:
        raise argparse.ArgumentTypeError(
            "point range bounds must be integers") from ex
    if start < 0 or end < start:
        raise argparse.ArgumentTypeError(
            "point range must satisfy 0 <= start <= end")
    return start, end


def best_permutation(sep_list, ref_list):
    """
    为多说话人波形寻找最佳参考顺序。

    作用：
        语音分离模型输出的第 1 路、第 2 路不一定对应固定说话人，因此这里穷举
        reference 的排列顺序，选择平均 SI-SNR 最高的一种。为了保持 estimated
        stream 的绘图顺序不变，本函数重排 ref_list，而不是 sep_list。
    输入：
        sep_list: separated/estimated 波形列表，每个元素通常是 numpy 数组。
        ref_list: reference 波形列表，长度应与 sep_list 一致。
    输出：
        matched_refs: 按最佳排列重排后的 reference 波形列表。
        best_pair_scores: 每一路 separated 与匹配 reference 的 SI-SNR 分数列表。
    调用时机：
        save_waveform_compare() 发现 len(sep_list) > 1 时调用，用于多说话人匹配。
    """
    best_order = tuple(range(len(sep_list)))
    best_score = None
    best_pair_scores = None
    # 分离结果没有固定说话人身份，因此尝试所有 reference 顺序，选择平均 SI-SNR
    # 最高的排列用于后续绘图。
    for order in permutations(range(len(ref_list))):
        pair_scores = [
            si_snr(sep_list[idx], ref_list[ref_idx])
            for idx, ref_idx in enumerate(order)
        ]
        avg_score = sum(pair_scores) / len(pair_scores)
        if best_score is None or avg_score > best_score:
            best_score = avg_score
            best_order = order
            best_pair_scores = pair_scores
    matched_refs = [ref_list[idx] for idx in best_order]
    return matched_refs, best_pair_scores


def align_estimated_signs(sep_list, ref_list):
    """
    Align the global sign of each estimated waveform to its matched reference.
    """
    aligned = []
    sign_flips = []
    for sep, ref in zip(sep_list, ref_list):
        if np.inner(sep, ref) < 0:
            aligned.append(-sep)
            sign_flips.append(True)
        else:
            aligned.append(sep)
            sign_flips.append(False)
    return aligned, sign_flips


def format_metric_value(value, unit=""):
    """
    Format an optional scalar metric for plot annotations.
    """
    if value is None:
        return "NA"
    return "{:.3f}{}".format(float(value), unit)


def build_stream_annotation_text(annotation):
    """
    Convert a per-stream annotation dictionary to compact plot text.
    """
    if not annotation:
        return ""

    header = []
    if annotation.get("label"):
        header.append(str(annotation["label"]))
    if "estimated_index" in annotation or "reference_index" in annotation:
        header.append("est{}->ref{}".format(annotation.get("estimated_index", "?"),
                                            annotation.get("reference_index", "?")))

    lines = []
    if header:
        lines.append(" | ".join(header))
    lines.append("Cos(raw/aln): {} / {}".format(
        format_metric_value(annotation.get("cosine")),
        format_metric_value(annotation.get("cosine_aligned"))))
    lines.append("WaveSNR(raw/aln): {} / {}".format(
        format_metric_value(annotation.get("waveform_snr_db"), " dB"),
        format_metric_value(annotation.get("waveform_snr_db_aligned"), " dB")))
    lines.append("Neg(all/low/high): {} / {} / {}".format(
        format_metric_value(annotation.get("frame_negative_ratio")),
        format_metric_value(annotation.get("low_energy_frame_negative_ratio")),
        format_metric_value(annotation.get("high_energy_frame_negative_ratio"))))
    lines.append("Low/High cos gap: {} / {} / {}".format(
        format_metric_value(annotation.get("low_energy_frame_cosine_mean")),
        format_metric_value(annotation.get("high_energy_frame_cosine_mean")),
        format_metric_value(annotation.get("high_low_cosine_gap"))))
    if annotation.get("si_snr_improvement_db") is not None:
        lines.append("Mix SI-SNR imp: {}".format(
            format_metric_value(annotation.get("si_snr_improvement_db"), " dB")))
    if annotation.get("waveform_snr_improvement_db") is not None:
        lines.append("Mix WaveSNR imp: {}".format(
            format_metric_value(annotation.get("waveform_snr_improvement_db"),
                                " dB")))
    return "\n".join(lines)


def save_waveform_compare(key,
                          sep_list,
                          ref_list,
                          dump_dir,
                          fs,
                          mix=None,
                          dpi=150,
                          point_range=None,
                          align_sign=False,
                          stream_annotations=None,
                          pre_matched=False,
                          pair_scores=None):
    """
    保存单个 utterance 的波形对比图。

    作用：
        对齐 separated、reference、可选 mixture 的长度；多说话人时先做 SI-SNR
        最优匹配；然后把每一路 reference 与 estimated 波形叠加画到子图中，并
        保存 PNG。
    输入：
        key: 当前 utterance 的标识，用作图标题和输出文件名。
        sep_list: separated/estimated 波形列表。
        ref_list: reference 波形列表。
        dump_dir: 图片输出目录。
        fs: 采样率，用于把样本点编号转换成秒。
        mix: 可选 mixture 波形；为 None 时不画 mixture 子图。
        dpi: 保存图片时使用的分辨率。
        point_range: 可选 (start, end) 闭区间，仅显示该采样点范围。
        align_sign: 若为 True，则按匹配参考逐路翻转 estimated 的整体符号。
        stream_annotations: 可选，每一路 estimated 对应的文字注释字典列表。
        pre_matched: 若为 True，则认为 sep_list 与 ref_list 已按最佳配对顺序对齐。
        pair_scores: 可选，预先计算好的逐路 SI-SNR 分数。
    输出：
        out_path: 保存后的 PNG 文件路径。
    调用时机：
        run() 遍历到一个需要保存的 utterance 后调用一次。
    """
    plot_streams = [sep_list, ref_list]
    if mix is not None:
        plot_streams.append([mix])
    # 将所有波形裁剪到最短长度，保证叠加绘图时共享完全一致的时间轴。
    min_len = min(signal.size for group in plot_streams for signal in group)
    sep_list = [signal[:min_len] for signal in sep_list]
    ref_list = [signal[:min_len] for signal in ref_list]
    mix = mix[:min_len] if mix is not None else None

    # 多说话人输出需要先把每路 estimated stream 匹配到最接近的 reference，
    # 再画到同一个子图里比较。
    if pre_matched:
        if pair_scores is None:
            pair_scores = [si_snr(sep, ref) for sep, ref in zip(sep_list, ref_list)]
    else:
        if len(sep_list) > 1:
            ref_list, pair_scores = best_permutation(sep_list, ref_list)
        else:
            pair_scores = [si_snr(sep_list[0], ref_list[0])]

    sign_flips = [False for _ in sep_list]
    if align_sign:
        sep_list, sign_flips = align_estimated_signs(sep_list, ref_list)

    if point_range is None:
        plot_start, plot_end = 0, min_len
    else:
        plot_start = min(point_range[0], min_len - 1)
        plot_end = min(point_range[1] + 1, min_len)
        if plot_start >= plot_end:
            raise ValueError("Requested point range is empty for key {}".format(
                key))
        sep_list = [signal[plot_start:plot_end] for signal in sep_list]
        ref_list = [signal[plot_start:plot_end] for signal in ref_list]
        mix = mix[plot_start:plot_end] if mix is not None else None

    if stream_annotations and len(stream_annotations) != len(sep_list):
        raise ValueError("stream_annotations size mismatch: {} vs {}".format(
            len(stream_annotations), len(sep_list)))

    rows = len(sep_list) + (1 if mix is not None else 0)
    row_height = 4.2 if stream_annotations else 2.8
    fig, axes = plt.subplots(rows,
                             1,
                             figsize=(14, row_height * rows),
                             sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    cursor = 0
    # 将样本点编号转换成秒，作为 x 轴。
    time_axis = np.arange(plot_start, plot_end) / float(fs)

    if mix is not None:
        axes[cursor].plot(time_axis, mix, color="0.4", linewidth=0.8)
        axes[cursor].set_title("Mixture")
        axes[cursor].set_ylabel("Amp")
        axes[cursor].grid(alpha=0.25)
        cursor += 1

    for idx, (sep, ref, score) in enumerate(zip(sep_list, ref_list,
                                                pair_scores),
                                            start=1):
        axes[cursor].plot(time_axis,
                          ref,
                          color="black",
                          linewidth=0.9,
                          label="Reference")
        axes[cursor].plot(time_axis,
                          sep,
                          color="tab:orange",
                          linewidth=0.8,
                          alpha=0.85,
                          label="Estimated")
        annotation = (stream_annotations[idx - 1]
                      if stream_annotations else None)
        title = "Stream {} | SI-SNR {:.3f} dB".format(idx, score)
        if annotation and annotation.get("label"):
            title += " | {}".format(annotation["label"])
        if align_sign and sign_flips[idx - 1]:
            title += " | Sign Aligned"
        axes[cursor].set_title(title)
        axes[cursor].set_ylabel("Amp")
        axes[cursor].grid(alpha=0.25)
        axes[cursor].legend(loc="upper right")
        annotation_text = build_stream_annotation_text(annotation)
        if annotation_text:
            axes[cursor].text(0.01,
                              0.98,
                              annotation_text,
                              transform=axes[cursor].transAxes,
                              va="top",
                              ha="left",
                              fontsize=8,
                              family="monospace",
                              bbox={
                                  "boxstyle": "round,pad=0.2",
                                  "facecolor": "white",
                                  "alpha": 0.82,
                                  "edgecolor": "0.7"
                              })
        cursor += 1

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(key)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # 延迟创建输出目录，并为每个 utterance key 保存一张 PNG。
    os.makedirs(dump_dir, exist_ok=True)
    out_path = os.path.join(dump_dir, "{}.png".format(key))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def run(args):
    """
    脚本主流程入口。

    作用：
        根据命令行参数构造 separated/reference/mixture reader，遍历样本，过滤
        需要绘图的 key，并调用 save_waveform_compare() 保存波形对比图。
    输入：
        args: argparse.Namespace，包含 sep_scp、ref_scp、mix_scp、fs、dump_dir、
        max_utts、keys、dpi 等字段。
    输出：
        无显式返回值；副作用是写出 PNG 图片并打印日志。
    调用时机：
        文件作为脚本直接运行时，在 __main__ 中解析完命令行参数后调用。
    """
    selected_keys = parse_key_filter(args.keys)
    single_speaker = len(args.sep_scp.split(",")) == 1

    # 单说话人 scp 每个 key 返回一个波形；多说话人 scp 使用逗号分隔，
    # 每个 key 返回多路波形列表。
    if single_speaker:
        sep_reader = WaveReader(args.sep_scp)
        ref_reader = WaveReader(args.ref_scp)
    else:
        sep_reader = SpeakersReader(args.sep_scp)
        ref_reader = SpeakersReader(args.ref_scp)

    mix_reader = WaveReader(args.mix_scp,
                            sample_rate=args.fs) if args.mix_scp else None

    num_saved = 0
    for key, sep in tqdm(sep_reader):
        # 先应用可选 utterance 过滤条件，再加载 reference/mixture，减少不必要读取。
        if selected_keys is not None and key not in selected_keys:
            continue
        if args.max_utts > 0 and num_saved >= args.max_utts:
            break

        ref = ref_reader[key]
        mix = mix_reader[key] if mix_reader else None
        sep_list = sep if isinstance(sep, list) else [sep]
        ref_list = ref if isinstance(ref, list) else [ref]
        out_path = save_waveform_compare(key,
                                         sep_list,
                                         ref_list,
                                         args.dump_dir,
                                         args.fs,
                                         mix=mix,
                                         dpi=args.dpi,
                                         point_range=args.point_range,
                                         align_sign=args.align_sign)
        logger.info("Save waveform figure to {}".format(out_path))
        num_saved += 1

    logger.info("Saved {:d} waveform comparison figures".format(num_saved))


# 脚本直接运行时，从这里解析命令行参数，然后进入 run() 主流程。
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to save waveform comparison plots for separated results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "sep_scp",
        type=str,
        help="Separated result scripts (support multi-stream, egs: spk1.scp,spk2.scp)"
    )
    parser.add_argument(
        "ref_scp",
        type=str,
        help="Reference waveform scripts aligned with sep_scp")
    parser.add_argument("--mix-scp",
                        type=str,
                        default="",
                        help="Optional mixture waveform script")
    parser.add_argument("--fs",
                        type=int,
                        default=8000,
                        help="Sample rate used for the time axis")
    parser.add_argument("--dump-dir",
                        type=str,
                        default="waveform_compare",
                        help="Directory to dump waveform figures")
    parser.add_argument("--max-utts",
                        type=int,
                        default=20,
                        help="Maximum number of utterances to plot, <=0 means all"
                        )
    parser.add_argument("--keys",
                        type=str,
                        default="",
                        help="Optional comma-separated utterance keys to plot")
    parser.add_argument("--point-range",
                        type=parse_point_range,
                        default=None,
                        help="Optional inclusive sample index range: start,end")
    parser.add_argument("--align-sign",
                        action="store_true",
                        help="Flip estimated waveforms by matched-reference sign"
                        " before plotting")
    parser.add_argument("--dpi",
                        type=int,
                        default=150,
                        help="Figure dpi for saved waveform images")
    args = parser.parse_args()
    run(args)
