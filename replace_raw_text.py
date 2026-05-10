"""
replace_raw_text.py

Replaces the raw_text column in MyTTSDataset/metadata.csv with the
punctuated raw_text from transcriptions.csv, matched on:

    metadata.raw_text  ==  transcriptions.text

Output columns (pipe-delimited, no header):
    filename | raw_text (now with punctuation) | normalized_text

Usage:
    python replace_raw_text.py

Optional flags:
    --metadata        Path to metadata CSV         (default: MyTTSDataset/metadata.csv)
    --transcriptions  Path to transcriptions CSV   (default: transcriptions.csv)
    --output          Output CSV path              (default: overwrites --metadata)
    --backup          Save a .bak copy before overwriting (default: True)
    --unmatched       keep | drop | empty          (default: keep)
"""

import argparse
import csv
import shutil
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata",       default="MyTTSDataset/metadata.csv")
    parser.add_argument("--transcriptions", default="transcriptions.csv")
    parser.add_argument("--output",         default=None)
    parser.add_argument("--backup",         action="store_true", default=True)
    parser.add_argument("--unmatched",      default="keep",
                        choices=["keep", "drop", "empty"])
    args = parser.parse_args()

    metadata_path       = Path(args.metadata)
    transcriptions_path = Path(args.transcriptions)
    output_path         = Path(args.output) if args.output else metadata_path

    # ── Load files ────────────────────────────────────────────────────────────
    print(f"\n📄 Loading metadata       : {metadata_path}")
    meta = pd.read_csv(
        metadata_path,
        sep="|",
        header=None,
        names=["filename", "raw_text", "normalized_text"],
        dtype=str,
        keep_default_na=False,
    )
    print(f"   {len(meta):,} rows")

    print(f"📄 Loading transcriptions : {transcriptions_path}")
    trans = pd.read_csv(transcriptions_path, dtype=str, keep_default_na=False)
    print(f"   {len(trans):,} rows")

    # ── Validate columns ──────────────────────────────────────────────────────
    if "raw_text" not in trans.columns or "text" not in trans.columns:
        raise ValueError(
            f"transcriptions.csv must have 'raw_text' and 'text' columns. "
            f"Found: {list(trans.columns)}"
        )

    # ── Build lookup: normalized_text → punctuated raw_text ──────────────────
    # Normalise both sides: strip whitespace + lowercase for matching only
    trans_dedup = trans.drop_duplicates(subset=["text"])
    lookup: dict[str, str] = {
        row["text"].strip().lower(): row["raw_text"].strip()
        for _, row in trans_dedup.iterrows()
    }
    print(f"\n🔑 Lookup table: {len(lookup):,} unique entries")

    # ── Match and replace ─────────────────────────────────────────────────────
    matched   = 0
    unmatched = 0
    rows_out  = []

    for _, row in meta.iterrows():
        key         = row["raw_text"].strip().lower()
        replacement = lookup.get(key)

        if replacement is not None:
            new_raw = replacement
            matched += 1
        else:
            unmatched += 1
            if args.unmatched == "keep":
                new_raw = row["raw_text"]
            elif args.unmatched == "empty":
                new_raw = ""
            else:  # drop
                continue

        rows_out.append([row["filename"], new_raw, row["normalized_text"]])

    # ── Stats ─────────────────────────────────────────────────────────────────
    total     = matched + (unmatched if args.unmatched != "drop" else 0)
    match_pct = matched / max(len(meta), 1) * 100
    print(f"\n  ✅ Matched   : {matched:,}  ({match_pct:.1f}%)")
    if unmatched:
        print(f"  ⚠  Unmatched : {unmatched:,}  (action: {args.unmatched})")

    # ── Backup ────────────────────────────────────────────────────────────────
    if args.backup and output_path == metadata_path and metadata_path.exists():
        backup_path = metadata_path.with_suffix(".csv.bak")
        shutil.copy2(metadata_path, backup_path)
        print(f"\n💾 Backup saved : {backup_path}")

    # ── Write output using csv.writer (avoids pandas escape issues) ───────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        writer.writerows(rows_out)

    print(f"💾 Saved to     : {output_path.resolve()}")
    print(f"   Final rows   : {len(rows_out):,}")

    # ── Preview ───────────────────────────────────────────────────────────────
    print(f"\n  Preview (first 3 rows):")
    for filename, raw_text, norm_text in rows_out[:3]:
        print(f"    {filename} | {raw_text[:70]} | {norm_text[:40]}")
    print()


if __name__ == "__main__":
    main()