#!/usr/bin/env python

"""
Waveform comparison entry script.

Reads separated-result scp files and reference scp files, optionally reads the
mixture scp, then saves waveform comparison figures for selected utterances.
For multi-output separation, the script performs permutation matching before
plotting so the estimated signals align with the best-matched references.
"""

import argparse
import os

from itertools import permutations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from libs.audio import WaveReader
from libs.metric import si_snr
from libs.utils import get_logger


logger = get_logger(__name__)


class SpeakersReader(object):
    """
    Multi-stream waveform reader backed by comma-separated scp paths.
    """

    def __init__(self, scps):
        split_scps = scps.split(",")
        if len(split_scps) == 1:
            raise RuntimeError(
                "Construct SpeakersReader need more than one script, got {}".
                format(scps))
        self.readers = [WaveReader(scp) for scp in split_scps]

    def __len__(self):
        return len(self.readers[0])

    def __getitem__(self, key):
        return [reader[key] for reader in self.readers]

    def __iter__(self):
        first_reader = self.readers[0]
        for key in first_reader.index_keys:
            yield key, self[key]


def parse_key_filter(keys):
    if not keys:
        return None
    return {key.strip() for key in keys.split(",") if key.strip()}

def best_permutation(sep_list, ref_list):
    """
    Return sep_list reordered to the best SI-SNR permutation.
    """
    best_order = tuple(range(len(sep_list)))
    best_score = None
    best_pair_scores = None
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


def save_waveform_compare(key,
                          sep_list,
                          ref_list,
                          dump_dir,
                          fs,
                          mix=None,
                          dpi=150):
    plot_streams = [sep_list, ref_list]
    if mix is not None:
        plot_streams.append([mix])
    min_len = min(signal.size for group in plot_streams for signal in group)
    sep_list = [signal[:min_len] for signal in sep_list]
    ref_list = [signal[:min_len] for signal in ref_list]
    mix = mix[:min_len] if mix is not None else None

    if len(sep_list) > 1:
        ref_list, pair_scores = best_permutation(sep_list, ref_list)
    else:
        pair_scores = [si_snr(sep_list[0], ref_list[0])]

    rows = len(sep_list) + (1 if mix is not None else 0)
    fig, axes = plt.subplots(rows,
                             1,
                             figsize=(14, 2.8 * rows),
                             sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    cursor = 0
    time_axis = np.arange(min_len) / float(fs)

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
        axes[cursor].set_title("Stream {} | SI-SNR {:.3f} dB".format(
            idx, score))
        axes[cursor].set_ylabel("Amp")
        axes[cursor].grid(alpha=0.25)
        axes[cursor].legend(loc="upper right")
        cursor += 1

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(key)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(dump_dir, exist_ok=True)
    out_path = os.path.join(dump_dir, "{}.png".format(key))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def run(args):
    selected_keys = parse_key_filter(args.keys)
    single_speaker = len(args.sep_scp.split(",")) == 1

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
                                         dpi=args.dpi)
        logger.info("Save waveform figure to {}".format(out_path))
        num_saved += 1

    logger.info("Saved {:d} waveform comparison figures".format(num_saved))


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
    parser.add_argument("--dpi",
                        type=int,
                        default=150,
                        help="Figure dpi for saved waveform images")
    args = parser.parse_args()
    run(args)
