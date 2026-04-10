#!/usr/bin/env python

"""
离线生成 AISHELL + MUSAN 的语音降噪训练数据。

输出格式与当前项目的数据读取逻辑兼容：
  out_root/
    train/
      mix.scp
      spk1.scp
      spk2.scp
      speech.scp
      noise.scp
      metadata.jsonl
      mix/*.wav
      speech/*.wav
      noise/*.wav
    dev/
      ...
    test/
      ...

其中：
  - `mix.scp` 对应混合信号；
  - `spk1.scp` / `speech.scp` 对应干净语音；
  - `spk2.scp` / `noise.scp` 对应按目标 SNR 缩放后的纯噪声。

当前项目的训练代码只要求 `mix = ref1 + ref2`，并不关心 ref1/ref2
分别是两个说话人还是“语音 + 噪声”，因此可以直接复用。

AISHELL 语音池默认沿用数据集现成的全局划分：
  - 某些说话人目录下只有 `train`
  - 某些说话人目录下只有 `dev`
  - 某些说话人目录下只有 `test`
脚本会跨所有说话人把同名子目录聚合成全局 train/dev/test 语音池。
"""

import argparse
import json
import math
import random

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from tqdm import tqdm


EPS = 1e-8
SPLITS = ("train", "dev", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare offline AISHELL+MUSAN speech-noise mixtures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--aishell-root",
                        type=str,
                        required=True,
                        help="AISHELL wav root, e.g. .../data_aishell/wav")
    parser.add_argument("--musan-noise-root",
                        type=str,
                        required=True,
                        help="MUSAN noise root, e.g. .../musan/noise")
    parser.add_argument("--out-root",
                        type=str,
                        required=True,
                        help="Output directory for generated dataset")
    parser.add_argument("--sample-rate",
                        type=int,
                        default=8000,
                        help="Target sample rate for generated wavs")
    parser.add_argument("--train-examples",
                        type=int,
                        default=20000,
                        help="Number of train mixtures to generate")
    parser.add_argument("--dev-examples",
                        type=int,
                        default=2000,
                        help="Number of dev mixtures to generate")
    parser.add_argument("--test-examples",
                        type=int,
                        default=2000,
                        help="Number of test mixtures to generate")
    parser.add_argument("--min-snr",
                        type=float,
                        default=0.0,
                        help="Minimum SNR in dB")
    parser.add_argument("--max-snr",
                        type=float,
                        default=15.0,
                        help="Maximum SNR in dB")
    parser.add_argument("--segment-seconds",
                        type=float,
                        default=0.0,
                        help="If > 0, randomly crop each speech sample to a fixed duration")
    parser.add_argument("--pad-shorter",
                        action="store_true",
                        help="When --segment-seconds is set, zero-pad short speech utterances instead of skipping them")
    parser.add_argument("--min-speech-seconds",
                        type=float,
                        default=1.0,
                        help="Skip speech utterances shorter than this duration before optional cropping")
    parser.add_argument("--peak-target",
                        type=float,
                        default=0.9,
                        help="If abs peak exceeds this value, scale speech/noise/mix together")
    parser.add_argument("--noise-split-mode",
                        type=str,
                        default="shared",
                        choices=("shared", "file"),
                        help="How to build noise pools for train/dev/test")
    parser.add_argument("--noise-dev-ratio",
                        type=float,
                        default=0.1,
                        help="Fraction of MUSAN noise files reserved for dev when --noise-split-mode=file")
    parser.add_argument("--noise-test-ratio",
                        type=float,
                        default=0.1,
                        help="Fraction of MUSAN noise files reserved for test when --noise-split-mode=file")
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="Random seed for reproducible generation")
    parser.add_argument("--max-attempts-per-example",
                        type=int,
                        default=50,
                        help="Retry limit when a sampled utterance/noise pair is invalid")
    return parser.parse_args()


def load_wav_mono(path):
    sample_rate, samples = wavfile.read(str(path))
    samples = np.asarray(samples)
    if np.issubdtype(samples.dtype, np.integer):
        scale = float(max(abs(np.iinfo(samples.dtype).min),
                          np.iinfo(samples.dtype).max))
        samples = samples.astype(np.float32) / scale
    else:
        samples = samples.astype(np.float32)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    return sample_rate, samples


def resample_audio(samples, src_rate, dst_rate):
    if src_rate == dst_rate:
        return samples.astype(np.float32, copy=False)
    gcd = math.gcd(src_rate, dst_rate)
    up = dst_rate // gcd
    down = src_rate // gcd
    return resample_poly(samples, up, down).astype(np.float32)


def save_wav(path, sample_rate, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.clip(samples, -1.0, 1.0)
    wavfile.write(str(path), sample_rate,
                  (samples * np.iinfo(np.int16).max).astype(np.int16))


def gather_aishell_speech(root):
    speech = {split: [] for split in SPLITS}
    for speaker_dir in sorted(root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        for split in SPLITS:
            split_dir = speaker_dir / split
            if split_dir.exists():
                speech[split].extend(sorted(split_dir.rglob("*.wav")))
    return speech


def gather_noise_files(root):
    return sorted(root.rglob("*.wav"))


def split_noise_files(noise_files, dev_ratio, test_ratio, seed):
    if dev_ratio < 0 or test_ratio < 0 or dev_ratio + test_ratio >= 1:
        raise ValueError("Need 0 <= noise-dev-ratio + noise-test-ratio < 1")
    shuffled = list(noise_files)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    dev_count = int(total * dev_ratio)
    test_count = int(total * test_ratio)
    train_count = total - dev_count - test_count

    return {
        "train": shuffled[:train_count],
        "dev": shuffled[train_count:train_count + dev_count],
        "test": shuffled[train_count + dev_count:]
    }


def build_noise_pools(noise_files, split_mode, dev_ratio, test_ratio, seed):
    if split_mode == "shared":
        return {split: list(noise_files) for split in SPLITS}
    if split_mode == "file":
        return split_noise_files(noise_files, dev_ratio, test_ratio, seed)
    raise ValueError("Unsupported noise split mode: {}".format(split_mode))


def crop_or_pad(samples, target_len, rng, pad_shorter):
    if target_len <= 0:
        return samples
    cur_len = samples.shape[0]
    if cur_len == target_len:
        return samples
    if cur_len > target_len:
        start = rng.randint(0, cur_len - target_len)
        return samples[start:start + target_len]
    if not pad_shorter:
        return None
    return np.pad(samples, (0, target_len - cur_len), mode="constant")


def sample_noise_segment(samples, target_len, rng):
    cur_len = samples.shape[0]
    if cur_len <= 0:
        return None
    if cur_len >= target_len:
        start = rng.randint(0, cur_len - target_len)
        return samples[start:start + target_len]

    offset = rng.randint(0, cur_len - 1)
    tiled = np.tile(samples, int(math.ceil((target_len + offset) / cur_len)))
    return tiled[offset:offset + target_len]


def rms(samples):
    return float(np.sqrt(np.mean(np.square(samples), dtype=np.float64) + EPS))


def mix_speech_and_noise(speech, noise, snr_db, peak_target):
    speech_rms = rms(speech)
    noise_rms = rms(noise)
    if speech_rms <= EPS or noise_rms <= EPS:
        return None, None, None

    noise_scale = speech_rms / ((10.0**(snr_db / 20.0)) * noise_rms + EPS)
    noise_ref = noise * noise_scale
    mix = speech + noise_ref

    if peak_target > 0:
        peak = max(float(np.max(np.abs(speech))), float(np.max(np.abs(noise_ref))),
                   float(np.max(np.abs(mix))), EPS)
        if peak > peak_target:
            gain = peak_target / peak
            speech = speech * gain
            noise_ref = noise_ref * gain
            mix = mix * gain

    return mix.astype(np.float32), speech.astype(np.float32), noise_ref.astype(
        np.float32)


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def dump_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True, sort_keys=False)


def generate_split(split,
                   num_examples,
                   speech_files,
                   noise_files,
                   out_root,
                   sample_rate,
                   min_snr,
                   max_snr,
                   segment_seconds,
                   pad_shorter,
                   min_speech_seconds,
                   peak_target,
                   seed,
                   max_attempts):
    if num_examples <= 0:
        return {"generated": 0, "skipped": 0}
    if not speech_files:
        raise RuntimeError("No speech files found for split {}".format(split))
    if not noise_files:
        raise RuntimeError("No noise files found for split {}".format(split))

    rng = random.Random(seed)
    subset_dir = out_root / split
    target_len = int(round(segment_seconds * sample_rate)) if segment_seconds > 0 else 0
    min_speech_len = int(round(min_speech_seconds * sample_rate))

    mix_lines = []
    spk1_lines = []
    spk2_lines = []
    speech_lines = []
    noise_lines = []
    skipped = 0

    metadata_path = subset_dir / "metadata.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w") as metadata_f:
        for idx in tqdm(range(num_examples), desc="Generate {}".format(split)):
            created = False
            for _ in range(max_attempts):
                speech_src = Path(rng.choice(speech_files))
                noise_src = Path(rng.choice(noise_files))

                speech_rate, speech = load_wav_mono(speech_src)
                noise_rate, noise = load_wav_mono(noise_src)
                speech = resample_audio(speech, speech_rate, sample_rate)
                noise = resample_audio(noise, noise_rate, sample_rate)

                if speech.shape[0] < min_speech_len:
                    continue
                speech = crop_or_pad(speech, target_len, rng, pad_shorter)
                if speech is None or speech.shape[0] == 0:
                    continue

                noise = sample_noise_segment(noise, speech.shape[0], rng)
                if noise is None:
                    continue

                snr_db = rng.uniform(min_snr, max_snr)
                mix, speech_ref, noise_ref = mix_speech_and_noise(
                    speech, noise, snr_db, peak_target)
                if mix is None:
                    continue

                key = "{}_{:07d}".format(split, idx)
                mix_path = (subset_dir / "mix" / "{}.wav".format(key)).resolve()
                speech_path = (subset_dir / "speech" /
                               "{}.wav".format(key)).resolve()
                noise_path = (subset_dir / "noise" /
                              "{}.wav".format(key)).resolve()

                save_wav(mix_path, sample_rate, mix)
                save_wav(speech_path, sample_rate, speech_ref)
                save_wav(noise_path, sample_rate, noise_ref)

                mix_lines.append("{} {}".format(key, mix_path))
                spk1_lines.append("{} {}".format(key, speech_path))
                spk2_lines.append("{} {}".format(key, noise_path))
                speech_lines.append("{} {}".format(key, speech_path))
                noise_lines.append("{} {}".format(key, noise_path))

                metadata = {
                    "key": key,
                    "split": split,
                    "sample_rate": sample_rate,
                    "num_samples": int(mix.shape[0]),
                    "duration_seconds": round(float(mix.shape[0]) / sample_rate,
                                              4),
                    "snr_db": round(float(snr_db), 4),
                    "speech_source": str(speech_src.resolve()),
                    "noise_source": str(noise_src.resolve()),
                    "label_roles": {
                        "spk1": "speech",
                        "spk2": "noise"
                    }
                }
                metadata_f.write(json.dumps(metadata, ensure_ascii=True))
                metadata_f.write("\n")
                created = True
                break

            if not created:
                skipped += 1

    write_lines(subset_dir / "mix.scp", mix_lines)
    write_lines(subset_dir / "spk1.scp", spk1_lines)
    write_lines(subset_dir / "spk2.scp", spk2_lines)
    write_lines(subset_dir / "speech.scp", speech_lines)
    write_lines(subset_dir / "noise.scp", noise_lines)

    return {"generated": len(mix_lines), "skipped": skipped}


def main():
    args = parse_args()
    aishell_root = Path(args.aishell_root).expanduser().resolve()
    musan_noise_root = Path(args.musan_noise_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    speech_files = gather_aishell_speech(aishell_root)
    noise_files = gather_noise_files(musan_noise_root)
    noise_split = build_noise_pools(noise_files, args.noise_split_mode,
                                    args.noise_dev_ratio,
                                    args.noise_test_ratio, args.seed)

    config = {
        "args": vars(args),
        "speech_split_mode": "official_global_pool",
        "speech_pool_sizes": {
            split: len(speech_files[split])
            for split in SPLITS
        },
        "noise_split_mode": args.noise_split_mode,
        "noise_pool_sizes": {
            split: len(noise_split[split])
            for split in SPLITS
        },
        "label_roles": {
            "spk1": "speech",
            "spk2": "noise"
        }
    }
    dump_json(out_root / "dataset_config.json", config)
    dump_json(out_root / "noise_split.json", {
        split: [str(path.resolve()) for path in noise_split[split]]
        for split in SPLITS
    })

    summary = {}
    split_to_count = {
        "train": args.train_examples,
        "dev": args.dev_examples,
        "test": args.test_examples
    }
    for offset, split in enumerate(SPLITS):
        summary[split] = generate_split(
            split=split,
            num_examples=split_to_count[split],
            speech_files=speech_files[split],
            noise_files=noise_split[split],
            out_root=out_root,
            sample_rate=args.sample_rate,
            min_snr=args.min_snr,
            max_snr=args.max_snr,
            segment_seconds=args.segment_seconds,
            pad_shorter=args.pad_shorter,
            min_speech_seconds=args.min_speech_seconds,
            peak_target=args.peak_target,
            seed=args.seed + offset,
            max_attempts=args.max_attempts_per_example)

    dump_json(out_root / "generation_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
