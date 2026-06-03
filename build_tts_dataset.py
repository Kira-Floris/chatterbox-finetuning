"""
build_tts_dataset.py

Downloads the Afrivoice-Kinyarwanda-ASR dataset directly from HuggingFace,
converts audio (stored as raw WebM bytes) to mono WAV, and writes a
TTS-ready dataset in LJSpeech format:

    MyTTSDataset/
        metadata.csv          ← filename|raw_text|normalized_text  (no header)
        wavs/
            recording_00000001.wav
            recording_00000002.wav
            ...

Usage:
    python build_tts_dataset.py

Optional flags:
    --output_dir      Root output directory       (default: MyTTSDataset)
    --domain          Comma-separated domains, e.g. 'agriculture,health'
                      Available: agriculture, education, financial, government,
                                 health, scripted_education
    --split           Comma-separated split types, e.g. 'train,validation'
                      Available: train, validation, test
    --sample_rate     Target sample rate: 16000, 22050, or 44100  (default: 22050)
    --num_workers     Parallel audio-write workers                 (default: 60)
    --resume          Skip WAV files that already exist            (default: True)
    --cache_dir       HuggingFace cache directory
    --streaming       Stream dataset without full download         (default: False)
    --transcriptions  Path to transcriptions.csv for raw_text replacement
    --backup          Save a .bak copy of metadata before overwriting (default: True)

Requirements:
    pip install datasets pydub soundfile kinya-tn tqdm numpy
    apt install ffmpeg   # required by pydub to decode WebM
"""

import argparse
import csv
import io
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from pydub import AudioSegment
from datasets import load_dataset, get_dataset_split_names
from kinya_tn import text_normalization
from tqdm import tqdm


DATASET_ID         = "Kira-Floris/Afrivoice-Kinyarwanda-ASR"
VALID_DOMAINS      = {"agriculture", "education", "financial", "government",
                      "health", "scripted_education"}
VALID_SPLITS       = {"train", "validation", "test"}
VALID_SAMPLE_RATES = {16000, 22050, 44100}

# WebM/MKV magic bytes — first 4 bytes of any WebM file
WEBM_MAGIC = b'\x1aE\xdf\xa3'


# ── Split name helpers ────────────────────────────────────────────────────────

def parse_hf_split_name(hf_split: str) -> tuple[str, str]:
    for split_type in ("_train", "_validation", "_test"):
        if hf_split.endswith(split_type):
            return hf_split[: -len(split_type)], split_type.lstrip("_")
    return hf_split, "unknown"


def get_target_splits(
    all_hf_splits: list[str],
    domain_filter: set[str] | None,
    split_filter:  set[str] | None,
) -> list[str]:
    result = []
    for hf_split in all_hf_splits:
        domain, split_type = parse_hf_split_name(hf_split)
        if domain_filter and domain     not in domain_filter:
            continue
        if split_filter  and split_type not in split_filter:
            continue
        result.append(hf_split)
    return sorted(result)


# ── Audio decoding ────────────────────────────────────────────────────────────

def bytes_to_numpy(raw: bytes) -> tuple[np.ndarray, int] | tuple[None, None]:
    """
    Decode raw audio bytes → (float32 numpy array, sample_rate).

    Tries soundfile first (fast, handles WAV/FLAC/OGG).
    Falls back to pydub+ffmpeg for WebM/MP3/any container soundfile can't read.
    Returns (None, None) on failure.
    """
    # ── soundfile fast path ───────────────────────────────────────────────────
    try:
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        return arr, sr
    except Exception:
        pass

    # ── pydub / ffmpeg fallback (handles WebM, MP3, M4A, …) ─────────────────
    try:
        seg = AudioSegment.from_file(io.BytesIO(raw))
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples /= float(1 << (8 * seg.sample_width - 1))
        if seg.channels > 1:
            samples = samples.reshape(-1, seg.channels).mean(axis=1)
        return samples, seg.frame_rate
    except Exception:
        pass

    return None, None


def decode_audio(audio_value) -> tuple[np.ndarray, int] | tuple[None, None]:
    """
    Normalise any form the HF `audio` column can take:

      1. Decoded dict  : {"array": ndarray, "sampling_rate": int}
      2. Bytes dict    : {"bytes": b"...", "path": "..."}   ← WebM in this dataset
      3. Plain path    : "/cache/file.wav"
      4. Raw bytes     : b"\\x1aE\\xdf\\xa3..."
    """
    try:
        if isinstance(audio_value, dict) and "array" in audio_value:
            arr = np.array(audio_value["array"], dtype=np.float32)
            sr  = int(audio_value["sampling_rate"])
            return arr, sr

        if isinstance(audio_value, dict):
            raw   = audio_value.get("bytes")
            path  = audio_value.get("path")

            if raw and len(raw) > 4:
                return bytes_to_numpy(raw)

            if path and Path(path).exists():
                arr, sr = sf.read(path, dtype="float32", always_2d=False)
                return arr, sr

            for v in audio_value.values():
                if isinstance(v, (bytes, bytearray)) and len(v) > 44:
                    return bytes_to_numpy(bytes(v))

        if isinstance(audio_value, str) and Path(audio_value).exists():
            arr, sr = sf.read(audio_value, dtype="float32", always_2d=False)
            return arr, sr

        if isinstance(audio_value, (bytes, bytearray)):
            return bytes_to_numpy(bytes(audio_value))

    except Exception:
        pass

    return None, None


# ── Resampling ────────────────────────────────────────────────────────────────

def resample_numpy(array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return array
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(orig_sr, target_sr)
        return resample_poly(array, target_sr // g, orig_sr // g).astype(np.float32)
    except ImportError:
        duration  = len(array) / orig_sr
        n_samples = int(duration * target_sr)
        old_t     = np.linspace(0, duration, len(array))
        new_t     = np.linspace(0, duration, n_samples)
        return np.interp(new_t, old_t, array).astype(np.float32)


# ── WAV writer (runs in thread pool) ─────────────────────────────────────────

def write_wav_worker(args: tuple) -> str | None:
    """Returns dst_path on success, None on failure."""
    local_i, audio_arr, orig_sr, dst_path, target_sr, resume = args

    if resume and Path(dst_path).exists():
        return dst_path

    try:
        if audio_arr.ndim == 2:
            audio_arr = audio_arr.mean(axis=1)
        audio_arr = audio_arr.astype(np.float32)

        if orig_sr != target_sr:
            audio_arr = resample_numpy(audio_arr, orig_sr, target_sr)

        sf.write(dst_path, audio_arr, target_sr, subtype="PCM_16")
        return dst_path
    except Exception:
        return None


# ── Text helpers ──────────────────────────────────────────────────────────────

def normalize_for_tts(text: str) -> str:
    """Strip all punctuation (including - _ ,) then normalize via kinya_tn."""
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"_", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text_normalization.normalize(text)


# ── Raw-text replacement ──────────────────────────────────────────────────────

def replace_raw_text_from_transcriptions(
    metadata_path: Path,
    transcriptions_path: Path,
    backup: bool = True,
    output_path: Path | None = None,
) -> None:
    """
    Replaces raw_text (and re-normalises normalized_text) in metadata_path
    using punctuated text from transcriptions_path, matched on:

        metadata.raw_text  ==  transcriptions.text   (case-insensitive, stripped)

    Unmatched rows are always dropped.

    Args:
        metadata_path:        Path to the pipe-delimited metadata.csv to update.
        transcriptions_path:  Path to transcriptions.csv with 'text' and 'raw_text' cols.
        backup:               Write a .bak copy before overwriting.
        output_path:          Where to write output; defaults to metadata_path (in-place).
    """
    output_path = output_path or metadata_path

    meta = pd.read_csv(
        metadata_path,
        sep="|", header=None,
        names=["filename", "raw_text", "normalized_text"],
        dtype=str, keep_default_na=False,
    )
    trans = pd.read_csv(transcriptions_path, dtype=str, keep_default_na=False)

    if "raw_text" not in trans.columns or "text" not in trans.columns:
        raise ValueError(
            f"transcriptions.csv must have 'raw_text' and 'text' columns. "
            f"Found: {list(trans.columns)}"
        )

    # ── Build lookup: stripped-lowercase text → punctuated raw_text ──────────
    lookup: dict[str, str] = {
        row["text"].strip().lower(): row["raw_text"].strip()
        for _, row in trans.drop_duplicates(subset=["text"]).iterrows()
    }
    print(f"\n🔑 replace_raw_text: {len(lookup):,} lookup entries")

    # ── Match — unmatched rows are dropped ────────────────────────────────────
    matched = unmatched_count = 0
    rows_out = []

    for _, row in meta.iterrows():
        key         = row["raw_text"].strip().lower()
        replacement = lookup.get(key)

        if replacement is None:
            unmatched_count += 1
            continue

        matched += 1
        rows_out.append([row["filename"], replacement, normalize_for_tts(replacement)])

    match_pct = matched / max(len(meta), 1) * 100
    print(f"  ✅ Matched   : {matched:,}  ({match_pct:.1f}%)")
    print(f"  🗑  Dropped   : {unmatched_count:,} unmatched rows")

    # ── Backup + write ────────────────────────────────────────────────────────
    if backup and output_path == metadata_path and metadata_path.exists():
        backup_path = metadata_path.with_suffix(".csv.bak")
        shutil.copy2(metadata_path, backup_path)
        print(f"💾 Backup saved : {backup_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, delimiter="|", quoting=csv.QUOTE_MINIMAL).writerows(rows_out)

    print(f"💾 Saved to     : {output_path.resolve()}  ({len(rows_out):,} rows)\n")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_ljspeech_dataset(
    output_dir:           Path,
    hf_splits:            list[str],
    sample_rate:          int,
    num_workers:          int,
    resume:               bool,
    cache_dir:            str | None,
    streaming:            bool,
    transcriptions_path:  Path | None = None,
    backup:               bool = True,
) -> None:
    wavs_dir      = output_dir / "wavs"
    metadata_path = output_dir / "metadata.csv"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  Dataset      : {DATASET_ID}")
    print(f"  HF splits    : {hf_splits}")
    print(f"  Sample rate  : {sample_rate} Hz")
    print(f"  Output dir   : {output_dir}")
    print(f"{'─'*60}\n")

    global_idx   = 0
    failed_total = 0
    written      = 0

    with open(metadata_path, "w", encoding="utf-8", newline="") as meta_fh:
        writer = csv.writer(meta_fh, delimiter="|", quoting=csv.QUOTE_MINIMAL)

        for hf_split in hf_splits:
            domain, split_type = parse_hf_split_name(hf_split)
            print(f"⬇  Loading: {hf_split}  (domain={domain}, type={split_type})")

            ds = load_dataset(
                DATASET_ID,
                split             = hf_split,
                cache_dir         = cache_dir,
                streaming         = streaming,
                trust_remote_code = True,
            )

            all_rows = list(ds)

            # ── Inspect first row ─────────────────────────────────────────
            if all_rows:
                sample_audio = all_rows[0].get("audio")
                print(f"   Audio column type : {type(sample_audio).__name__}")
                if isinstance(sample_audio, dict):
                    keys = list(sample_audio.keys())
                    print(f"   Audio dict keys   : {keys}")
                    if "bytes" in sample_audio:
                        raw_preview = sample_audio["bytes"][:4] if sample_audio["bytes"] else b""
                        is_webm = raw_preview == WEBM_MAGIC
                        print(f"   Bytes magic       : {raw_preview.hex()}  "
                              f"({'WebM ✅' if is_webm else 'unknown format'})")
                arr, sr = decode_audio(sample_audio)
                if arr is None:
                    print(f"   ⚠  WARNING: could not decode sample row.")
                else:
                    print(f"   ✅ Sample decoded  : shape={arr.shape}, sr={sr} Hz")

            # ── Build job list ─────────────────────────────────────────────
            jobs  = []
            stems = []
            texts = []

            for row in tqdm(all_rows, desc=f"  prep {hf_split}"):
                transcription = (
                    row.get("transcription") or row.get("text") or ""
                ).strip()
                if not transcription:
                    continue

                arr, sr = decode_audio(row.get("audio"))
                if arr is None:
                    failed_total += 1
                    continue

                global_idx += 1
                stem     = f"recording_{global_idx:08d}"
                dst_path = str(wavs_dir / f"{stem}.wav")

                stems.append(stem)
                texts.append(transcription)
                jobs.append((global_idx, arr, sr, dst_path, sample_rate, resume))

            print(f"   {len(jobs):,} jobs queued — writing WAVs …")

            # ── Parallel WAV write ─────────────────────────────────────────
            split_failed: set[int] = set()

            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                future_to_local = {
                    pool.submit(write_wav_worker, job): local_i
                    for local_i, job in enumerate(jobs)
                }
                for future in tqdm(
                    as_completed(future_to_local),
                    total=len(future_to_local),
                    desc=f"  {hf_split}",
                ):
                    local_i = future_to_local[future]
                    if future.result() is None:
                        split_failed.add(local_i)
                        failed_total += 1

            if split_failed:
                print(f"   ⚠  {len(split_failed):,} WAV writes failed.")

            # ── Write metadata rows ────────────────────────────────────────
            for local_i, (stem, raw_text) in enumerate(zip(stems, texts)):
                if local_i in split_failed:
                    continue
                normalized = normalize_for_tts(raw_text)
                row_parts = [stem, raw_text, normalized]
                if sum(p.count("|") for p in row_parts) > 2:
                    failed_total += 1
                    continue
                writer.writerow(row_parts)
                written += 1

            print(f"   ✅ {len(jobs) - len(split_failed):,} entries done for {hf_split}\n")

    # ── Optional: replace raw_text from transcriptions ────────────────────────
    if transcriptions_path is not None and transcriptions_path.exists():
        print("🔄 Replacing raw_text from transcriptions.csv …")
        replace_raw_text_from_transcriptions(
            metadata_path       = metadata_path,
            transcriptions_path = transcriptions_path,
            backup              = backup,
        )
    elif transcriptions_path is not None:
        print(f"⚠  transcriptions file not found, skipping replacement: {transcriptions_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    final_rows = written  # may be reduced by replace step; re-count if needed
    print(f"{'─'*60}")
    print(f"  Total entries  : {final_rows:,}")
    if failed_total:
        print(f"  ⚠  Failures    : {failed_total:,}")
    est_min   = final_rows * 4 / 60
    meets_min = est_min >= 30
    print(f"  Est. duration  : ~{est_min:.1f} min "
          f"({'✅ meets' if meets_min else '⚠  below'} 30-min minimum)")
    print(f"\n  {output_dir}/")
    print(f"  ├── metadata.csv   ({final_rows:,} rows)")
    print(f"  └── wavs/          ({written:,} WAV files @ {sample_rate} Hz)")
    print(f"{'─'*60}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build an LJSpeech TTS dataset from HuggingFace Afrivoice-Kinyarwanda-ASR."
    )
    parser.add_argument("--output_dir",      type=Path, default="MyTTSDataset")
    # parser.add_argument("--domain",          default="scripted_education,agriculture,health,financial,government,education",
    #                     help=f"Comma-separated domains. "
    #                          f"Choices: {', '.join(sorted(VALID_DOMAINS))}")
    parser.add_argument("--domain",          default="scripted_education,health,financial,government",
                        help=f"Comma-separated domains. "
                             f"Choices: {', '.join(sorted(VALID_DOMAINS))}")
    parser.add_argument("--split",           default="train,test,validation",
                        help="Comma-separated split types: train, validation, test.")
    parser.add_argument("--sample_rate",     default=22050, type=int,
                        choices=sorted(VALID_SAMPLE_RATES))
    parser.add_argument("--num_workers",     default=1, type=int)
    parser.add_argument("--resume",          action="store_true", default=True)
    parser.add_argument("--cache_dir",       default=None)
    parser.add_argument("--streaming",       action="store_true", default=True)
    parser.add_argument("--transcriptions",  type=Path, default=None,
                        help="Path to transcriptions.csv for raw_text replacement.")
    parser.add_argument("--backup",          action="store_true", default=True,
                        help="Save a .bak copy of metadata before overwriting.")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()

    domain_filter: set[str] | None = None
    if args.domain:
        domain_filter = {d.strip() for d in args.domain.split(",")}
        invalid = domain_filter - VALID_DOMAINS
        if invalid:
            print(f"ERROR: Unknown domain(s): {invalid}", file=sys.stderr)
            sys.exit(1)

    split_filter: set[str] | None = None
    if args.split:
        split_filter = {s.strip() for s in args.split.split(",")}
        invalid = split_filter - VALID_SPLITS
        if invalid:
            print(f"ERROR: Unknown split type(s): {invalid}", file=sys.stderr)
            sys.exit(1)

    print(f"\n📋 Fetching split list from: {DATASET_ID} …")
    all_hf_splits = get_dataset_split_names(DATASET_ID)
    print(f"   Found {len(all_hf_splits)} splits.")

    target_splits = get_target_splits(all_hf_splits, domain_filter, split_filter)
    if not target_splits:
        print("ERROR: No splits match the filters.", file=sys.stderr)
        sys.exit(1)
    print(f"   Selected: {target_splits}")

    build_ljspeech_dataset(
        output_dir          = output_dir,
        hf_splits           = target_splits,
        sample_rate         = args.sample_rate,
        num_workers         = args.num_workers,
        resume              = args.resume,
        cache_dir           = args.cache_dir,
        streaming           = args.streaming,
        transcriptions_path = args.transcriptions,
        backup              = args.backup,
    )

    print(f"\n🎉 Done! Dataset ready at: {output_dir}\n")


if __name__ == "__main__":
    main()
