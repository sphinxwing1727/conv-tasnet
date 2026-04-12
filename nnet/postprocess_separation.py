#!/usr/bin/env python

"""
Post-process separated waveforms with polarity-aware diagnostics.

This script combines the old SI-SNR evaluation flow with optional waveform
plots and richer statistics that are useful for speech-noise separation:

1. SI-SNR and classic waveform SNR per matched stream
2. Global cosine similarity and sign-flip diagnostics
3. Frame-level negative-correlation ratios
4. Low-energy vs high-energy frame cosine analysis
5. Optional JSON / CSV / plot outputs
"""

import argparse
import csv
import json
import os

from collections import defaultdict
from itertools import permutations

import numpy as np

from tqdm import tqdm

from libs.audio import Reader, SpeakersReader, WaveReader
from libs.metric import si_snr
from libs.utils import get_logger
from plot_waveform_compare import (align_estimated_signs, parse_key_filter,
                                   parse_point_range, save_waveform_compare)


logger = get_logger(__name__)
EPS = 1e-8
SUMMARY_METRICS = [
    "si_snr_db",
    "waveform_snr_db",
    "waveform_snr_db_aligned",
    "cosine",
    "cosine_aligned",
    "sign_flip",
    "frame_negative_ratio",
    "low_energy_frame_negative_ratio",
    "high_energy_frame_negative_ratio",
    "low_energy_frame_cosine_mean",
    "high_energy_frame_cosine_mean",
    "high_low_cosine_gap",
    "si_snr_improvement_db",
    "waveform_snr_improvement_db",
]


def parse_csv_list(raw):
    if not raw:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values if values else None


def cosine_similarity(estimate, reference):
    denom = np.linalg.norm(estimate) * np.linalg.norm(reference)
    if denom <= EPS:
        return 0.0
    return float(np.inner(estimate, reference) / denom)


def waveform_snr(estimate, reference):
    noise = estimate - reference
    ref_pow = np.sum(reference**2)
    noise_pow = np.sum(noise**2)
    return float(10 * np.log10((ref_pow + EPS) / (noise_pow + EPS)))


def rms(samples):
    return float(np.sqrt(np.mean(samples**2, dtype=np.float64) + EPS))


def summarize_values(values):
    filtered = [
        float(val) for val in values if val is not None and not np.isnan(val)
    ]
    if not filtered:
        return None
    arr = np.asarray(filtered, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.1)),
        "p90": float(np.quantile(arr, 0.9)),
    }


def frame_spans(num_samples, frame_len):
    if num_samples <= 0:
        return []
    if num_samples <= frame_len:
        return [(0, num_samples)]
    return [(start, start + frame_len)
            for start in range(0, num_samples - frame_len + 1, frame_len)]


def frame_diagnostics(estimate,
                      reference,
                      frame_len=400,
                      low_energy_quantile=0.3,
                      high_energy_quantile=0.7):
    spans = frame_spans(min(estimate.size, reference.size), frame_len)
    if not spans:
        return {
            "frame_count": 0,
            "frame_negative_ratio": None,
            "low_energy_frame_negative_ratio": None,
            "high_energy_frame_negative_ratio": None,
            "low_energy_frame_cosine_mean": None,
            "high_energy_frame_cosine_mean": None,
            "high_low_cosine_gap": None,
        }

    frame_cosines = []
    frame_rms = []
    for start, end in spans:
        ref_frame = reference[start:end]
        est_frame = estimate[start:end]
        frame_cosines.append(cosine_similarity(est_frame, ref_frame))
        frame_rms.append(rms(ref_frame))

    frame_cosines = np.asarray(frame_cosines, dtype=np.float64)
    frame_rms = np.asarray(frame_rms, dtype=np.float64)

    low_thr = np.quantile(frame_rms, low_energy_quantile)
    high_thr = np.quantile(frame_rms, high_energy_quantile)
    low_mask = frame_rms <= low_thr
    high_mask = frame_rms >= high_thr

    def masked_mean(mask):
        if not np.any(mask):
            return None
        return float(np.mean(frame_cosines[mask]))

    def masked_negative_ratio(mask):
        if not np.any(mask):
            return None
        return float(np.mean(frame_cosines[mask] < 0))

    low_cos_mean = masked_mean(low_mask)
    high_cos_mean = masked_mean(high_mask)
    gap = None
    if low_cos_mean is not None and high_cos_mean is not None:
        gap = high_cos_mean - low_cos_mean

    return {
        "frame_count": int(frame_cosines.size),
        "frame_negative_ratio": float(np.mean(frame_cosines < 0)),
        "low_energy_frame_negative_ratio": masked_negative_ratio(low_mask),
        "high_energy_frame_negative_ratio": masked_negative_ratio(high_mask),
        "low_energy_frame_cosine_mean": low_cos_mean,
        "high_energy_frame_cosine_mean": high_cos_mean,
        "high_low_cosine_gap": gap,
    }


def match_estimates(sep_list, ref_list):
    if len(sep_list) != len(ref_list):
        raise RuntimeError("Number of estimates and references does not match")
    if len(sep_list) == 1:
        return {
            "matched_refs": ref_list,
            "pair_scores": [float(si_snr(sep_list[0], ref_list[0]))],
            "order": (0, )
        }

    best_order = tuple(range(len(ref_list)))
    best_score = None
    best_pair_scores = None
    for order in permutations(range(len(ref_list))):
        pair_scores = [
            float(si_snr(sep_list[idx], ref_list[ref_idx]))
            for idx, ref_idx in enumerate(order)
        ]
        avg_score = sum(pair_scores) / len(pair_scores)
        if best_score is None or avg_score > best_score:
            best_score = avg_score
            best_order = order
            best_pair_scores = pair_scores
    return {
        "matched_refs": [ref_list[idx] for idx in best_order],
        "pair_scores": best_pair_scores,
        "order": best_order
    }


def analyze_pair(raw_estimate,
                 aligned_estimate,
                 reference,
                 mix=None,
                 frame_len=400,
                 low_energy_quantile=0.3,
                 high_energy_quantile=0.7):
    cosine = cosine_similarity(raw_estimate, reference)
    cosine_aligned = cosine_similarity(aligned_estimate, reference)
    metrics = {
        "si_snr_db": float(si_snr(raw_estimate, reference)),
        "waveform_snr_db": waveform_snr(raw_estimate, reference),
        "waveform_snr_db_aligned": waveform_snr(aligned_estimate, reference),
        "cosine": cosine,
        "cosine_aligned": cosine_aligned,
        "sign_flip": bool(cosine < 0),
    }
    metrics.update(
        frame_diagnostics(raw_estimate,
                          reference,
                          frame_len=frame_len,
                          low_energy_quantile=low_energy_quantile,
                          high_energy_quantile=high_energy_quantile))
    if mix is not None:
        baseline_si_snr = float(si_snr(mix, reference))
        baseline_snr = waveform_snr(mix, reference)
        metrics["mix_si_snr_db"] = baseline_si_snr
        metrics["si_snr_improvement_db"] = metrics["si_snr_db"] - baseline_si_snr
        metrics["mix_waveform_snr_db"] = baseline_snr
        metrics["waveform_snr_improvement_db"] = (
            metrics["waveform_snr_db_aligned"] - baseline_snr)
    else:
        metrics["mix_si_snr_db"] = None
        metrics["si_snr_improvement_db"] = None
        metrics["mix_waveform_snr_db"] = None
        metrics["waveform_snr_improvement_db"] = None
    return metrics


def build_label_names(num_refs, ref_labels):
    if ref_labels is None:
        return ["ref{:d}".format(idx + 1) for idx in range(num_refs)]
    if len(ref_labels) != num_refs:
        raise ValueError("Number of --ref-labels must match reference streams")
    return ref_labels


def flatten_records(utterance_records):
    rows = []
    for record in utterance_records:
        for stream in record["streams"]:
            row = {
                "key": record["key"],
                "group": record["group"],
                "num_samples": record["num_samples"],
                "plot_path": record["plot_path"],
            }
            row.update(stream)
            rows.append(row)
    return rows


def summarize_rows(rows):
    summary = {
        "count": len(rows),
        "metrics": {}
    }
    for metric in SUMMARY_METRICS:
        summary["metrics"][metric] = summarize_values(
            [row.get(metric) for row in rows])
    return summary


def build_group_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    output = {}
    for group, group_rows in groups.items():
        by_label = defaultdict(list)
        for row in group_rows:
            by_label[row["label"]].append(row)
        output[group] = {
            "overall": summarize_rows(group_rows),
            "by_label": {
                label: summarize_rows(label_rows)
                for label, label_rows in sorted(by_label.items())
            }
        }
    return output


def print_summary(summary):
    print("Processed utterances: {:d}".format(summary["num_utterances"]))
    print("Processed matched streams: {:d}".format(summary["num_stream_rows"]))
    if summary["num_plots"] > 0:
        print("Saved plots: {:d}".format(summary["num_plots"]))
    print("")

    def print_metric_block(title, metric_block):
        print(title)
        metrics = metric_block["metrics"]
        display_metrics = [
            ("si_snr_db", "SI-SNR (dB)"),
            ("si_snr_improvement_db", "SI-SNR improvement (dB)"),
            ("waveform_snr_db_aligned", "Waveform SNR aligned (dB)"),
            ("waveform_snr_improvement_db", "Waveform SNR improvement (dB)"),
            ("cosine", "Cosine"),
            ("cosine_aligned", "Cosine aligned"),
            ("sign_flip", "Sign flip ratio"),
            ("frame_negative_ratio", "Negative frame ratio"),
            ("low_energy_frame_negative_ratio", "Low-energy negative ratio"),
            ("high_energy_frame_negative_ratio", "High-energy negative ratio"),
            ("low_energy_frame_cosine_mean", "Low-energy cosine"),
            ("high_energy_frame_cosine_mean", "High-energy cosine"),
            ("high_low_cosine_gap", "High-low cosine gap"),
        ]
        for metric_name, metric_label in display_metrics:
            metric_summary = metrics.get(metric_name)
            if not metric_summary:
                continue
            print("  {}: mean={:.4f}, median={:.4f}, p10={:.4f}, p90={:.4f}".
                  format(metric_label, metric_summary["mean"],
                         metric_summary["median"], metric_summary["p10"],
                         metric_summary["p90"]))
        print("")

    print_metric_block("Overall", summary["overall"])
    for label, label_summary in summary["by_label"].items():
        print_metric_block("Label: {}".format(label), label_summary)


def dump_json(path, obj):
    if not path:
        return
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True, sort_keys=False)


def dump_csv(path, rows):
    if not path:
        return
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args):
    if args.frame_len <= 0:
        raise ValueError("--frame-len must be > 0")
    if not 0 <= args.low_energy_quantile <= args.high_energy_quantile <= 1:
        raise ValueError("Need 0 <= low-energy-quantile <= high-energy-quantile <= 1")

    selected_keys = parse_key_filter(args.keys)
    ref_labels = parse_csv_list(args.ref_labels)
    group_reader = Reader(args.group_scp) if args.group_scp else None
    single_stream = len(args.sep_scp.split(",")) == 1

    if single_stream:
        sep_reader = WaveReader(args.sep_scp)
        ref_reader = WaveReader(args.ref_scp)
    else:
        sep_reader = SpeakersReader(args.sep_scp)
        ref_reader = SpeakersReader(args.ref_scp)

    mix_reader = WaveReader(args.mix_scp,
                            sample_rate=args.fs) if args.mix_scp else None

    processed_utts = 0
    saved_plots = 0
    utterance_records = []
    label_names = None

    for key, sep in tqdm(sep_reader):
        if selected_keys is not None and key not in selected_keys:
            continue
        if args.max_utts > 0 and processed_utts >= args.max_utts:
            break

        ref = ref_reader[key]
        mix = mix_reader[key] if mix_reader else None
        sep_list = sep if isinstance(sep, list) else [sep]
        ref_list = ref if isinstance(ref, list) else [ref]

        streams = sep_list + ref_list + ([mix] if mix is not None else [])
        min_len = min(signal.size for signal in streams)
        sep_list = [signal[:min_len].astype(np.float64) for signal in sep_list]
        ref_list = [signal[:min_len].astype(np.float64) for signal in ref_list]
        mix = mix[:min_len].astype(np.float64) if mix is not None else None

        if label_names is None:
            label_names = build_label_names(len(ref_list), ref_labels)

        match = match_estimates(sep_list, ref_list)
        matched_refs = match["matched_refs"]
        aligned_seps, sign_flips = align_estimated_signs(sep_list, matched_refs)

        stream_records = []
        for est_idx, ref_idx in enumerate(match["order"]):
            metrics = analyze_pair(
                sep_list[est_idx],
                aligned_seps[est_idx],
                matched_refs[est_idx],
                mix=mix,
                frame_len=args.frame_len,
                low_energy_quantile=args.low_energy_quantile,
                high_energy_quantile=args.high_energy_quantile)
            stream_records.append({
                "estimated_index": est_idx + 1,
                "reference_index": ref_idx + 1,
                "label": label_names[ref_idx],
                "pair_si_snr_db": match["pair_scores"][est_idx],
                "sign_flip": sign_flips[est_idx],
                **metrics,
            })

        plot_path = ""
        if args.plot_dir and (args.max_plots <= 0 or saved_plots < args.max_plots):
            plot_path = save_waveform_compare(
                key,
                sep_list,
                matched_refs,
                args.plot_dir,
                args.fs,
                mix=mix,
                dpi=args.plot_dpi,
                point_range=args.point_range,
                align_sign=args.align_sign,
                stream_annotations=stream_records,
                pre_matched=True,
                pair_scores=match["pair_scores"])
            logger.info("Save post-process waveform figure to %s", plot_path)
            saved_plots += 1

        utterance_records.append({
            "key":
            key,
            "group":
            group_reader[key] if group_reader else "ALL",
            "num_samples":
            int(min_len),
            "plot_path":
            plot_path,
            "streams":
            stream_records,
        })
        processed_utts += 1

    rows = flatten_records(utterance_records)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    summary = {
        "num_utterances": processed_utts,
        "num_stream_rows": len(rows),
        "num_plots": saved_plots,
        "overall": summarize_rows(rows),
        "by_label": {
            label: summarize_rows(label_rows)
            for label, label_rows in sorted(by_label.items())
        },
        "by_group": build_group_summary(rows),
    }

    summary_payload = {
        "config": vars(args),
        "summary": summary,
    }
    print_summary(summary)
    dump_json(args.summary_json, summary_payload)
    dump_json(args.details_json, {
        "config": vars(args),
        "utterances": utterance_records,
    })
    dump_csv(args.details_csv, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Post-process separated outputs with SI-SNR, polarity diagnostics, and optional plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "sep_scp",
        type=str,
        help="Separated result scripts (support multi-stream, egs: spk1.scp,spk2.scp)"
    )
    parser.add_argument("ref_scp",
                        type=str,
                        help="Reference waveform scripts aligned with sep_scp")
    parser.add_argument("--mix-scp",
                        type=str,
                        default="",
                        help="Optional mixture waveform script")
    parser.add_argument("--fs",
                        type=int,
                        default=8000,
                        help="Sample rate used by the optional mix reader and plots")
    parser.add_argument("--keys",
                        type=str,
                        default="",
                        help="Optional comma-separated utterance keys to process")
    parser.add_argument("--max-utts",
                        type=int,
                        default=0,
                        help="Maximum number of utterances to process, <=0 means all")
    parser.add_argument("--ref-labels",
                        type=str,
                        default="",
                        help="Optional comma-separated labels for reference streams")
    parser.add_argument("--group-scp",
                        type=str,
                        default="",
                        help="Optional key-to-group mapping for grouped summaries")
    parser.add_argument("--spk2gender",
                        dest="group_scp",
                        type=str,
                        help="Backward-compatible alias of --group-scp")
    parser.add_argument("--frame-len",
                        type=int,
                        default=400,
                        help="Frame length in samples for frame-level diagnostics")
    parser.add_argument("--low-energy-quantile",
                        type=float,
                        default=0.3,
                        help="Quantile used to define low-energy frames")
    parser.add_argument("--high-energy-quantile",
                        type=float,
                        default=0.7,
                        help="Quantile used to define high-energy frames")
    parser.add_argument("--plot-dir",
                        type=str,
                        default="",
                        help="Optional directory for waveform comparison plots")
    parser.add_argument("--max-plots",
                        type=int,
                        default=20,
                        help="Maximum number of plots to save when --plot-dir is used")
    parser.add_argument("--point-range",
                        type=parse_point_range,
                        default=None,
                        help="Optional inclusive sample index range for plots: start,end")
    parser.add_argument("--align-sign",
                        action="store_true",
                        help="Flip estimated waveforms by matched-reference sign before plotting")
    parser.add_argument("--plot-dpi",
                        type=int,
                        default=150,
                        help="Figure dpi for saved plots")
    parser.add_argument("--summary-json",
                        type=str,
                        default="",
                        help="Optional JSON path for aggregated summary output")
    parser.add_argument("--details-json",
                        type=str,
                        default="",
                        help="Optional JSON path for per-utterance detailed output")
    parser.add_argument("--details-csv",
                        type=str,
                        default="",
                        help="Optional CSV path for flattened per-stream detailed output")
    args = parser.parse_args()
    run(args)
