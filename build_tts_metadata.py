"""
download_transcriptions.py

Downloads ONLY the JSON manifest files (no audio, no images, no tar archives)
from DigitalUmuganda/Afrivoice_Kinyarwanda using huggingface_hub, then parses
and combines them all into a single pandas DataFrame saved as CSV.

File structure on the HF repo:
  {domain}_{split}_tarred/
    sharded_manifests_with_image/
      manifest_0.json   ← JSONL, one record per line, text only
      manifest_1.json
      ...

Each JSON line has:
  raw_text, text, duration, LUFS, gender, age_group, location,
  image_category, image_sub_category, audio_filepath, image_filepath,
  shard_id, image_shard_id, creator_id

This script downloads ONLY the manifest_*.json files (text metadata),
NOT the audio/image tar shards.

Requirements:
    pip install huggingface_hub pandas tqdm

Usage:
    python download_transcriptions.py

    # Must be logged in — dataset is gated:
    huggingface-cli login

Optional flags:
    --output        Output CSV path              (default: transcriptions.csv)
    --domain        Comma-separated domains      e.g. 'health,agriculture'
                    Choices: agriculture, health, finance,
                             government, education, scripted_education
    --split         Comma-separated split types  e.g. 'train,val'
                    Choices: train, val, test
                    Note: use 'val' not 'validation' (matches repo folder names)
    --min_duration  Keep clips >= N seconds      (default: none)
    --max_duration  Keep clips <= N seconds      (default: none)
    --min_lufs      Keep clips with LUFS >= N    (default: none)
    --cache_dir     Local directory to cache downloaded JSON files
                    (default: ~/.cache/afrivoice_manifests)
    --no_cache      Re-download even if file already exists in cache
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


DATASET_ID    = "DigitalUmuganda/Afrivoice_Kinyarwanda"
REPO_TYPE     = "dataset"

VALID_DOMAINS = {
    "agriculture",
    "health",
    "finance",
    "government",
    "education",
    "scripted_education",
}

# Note: the repo uses 'val', not 'validation'
VALID_SPLITS = {"train", "val", "test"}

# scripted_education only has a train split
SCRIPTED_EDUCATION_SPLITS = {"train"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def folder_name(domain: str, split: str) -> str:
    """e.g. ('health', 'train') → 'health_train_tarred'"""
    return f"{domain}_{split}_tarred"


def manifest_prefix(domain: str, split: str) -> str:
    """Path prefix for manifest files inside the repo."""
    return f"{folder_name(domain, split)}/sharded_manifests_with_image/"


def list_manifest_files(api: HfApi, domain: str, split: str) -> list[str]:
    """
    List all manifest_*.json paths inside a domain/split folder.
    Returns repo-relative file paths like:
      'health_train_tarred/sharded_manifests_with_image/manifest_0.json'
    """
    prefix = manifest_prefix(domain, split)
    try:
        files = api.list_repo_files(
            repo_id   = DATASET_ID,
            repo_type = REPO_TYPE,
        )
        return sorted(
            f for f in files
            if f.startswith(prefix) and f.endswith(".json")
        )
    except Exception as e:
        print(f"   ⚠  Could not list files for {domain}/{split}: {e}")
        return []


def download_manifest(
    repo_path:  str,
    cache_dir:  Path,
    use_cache:  bool,
) -> Path | None:
    """
    Download a single manifest JSON file from HF hub.
    Returns local path on success, None on failure.
    """
    # Mirror the repo path as a local cache path
    local_path = cache_dir / repo_path
    if use_cache and local_path.exists():
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(
            repo_id   = DATASET_ID,
            repo_type = REPO_TYPE,
            filename  = repo_path,
            local_dir = str(cache_dir),
        )
        return Path(downloaded)
    except Exception as e:
        print(f"   ⚠  Failed to download {repo_path}: {e}")
        return None


def parse_manifest_file(path: Path, domain: str, split: str) -> list[dict]:
    """
    Parse a JSONL manifest file. Each line is one JSON record.
    Injects 'domain' and 'split_type' columns.
    Returns a list of dicts (one per valid line).
    """
    records = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record["domain"]     = domain
                    record["split_type"] = split
                    records.append(record)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"   ⚠  Could not parse {path}: {e}")
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download JSON manifests (text only, no audio) from "
                    "DigitalUmuganda/Afrivoice_Kinyarwanda via huggingface_hub."
    )
    parser.add_argument("--output",       default="transcriptions.csv",
                        help="Output CSV path (default: transcriptions.csv).")
    parser.add_argument("--domain",       default=None,
                        help=f"Comma-separated domains. "
                             f"Choices: {', '.join(sorted(VALID_DOMAINS))}")
    parser.add_argument("--split",        default=None,
                        help="Comma-separated splits: train, val, test. "
                             "Note: use 'val', not 'validation'.")
    parser.add_argument("--min_duration", default=None, type=float,
                        help="Keep only clips >= N seconds.")
    parser.add_argument("--max_duration", default=None, type=float,
                        help="Keep only clips <= N seconds.")
    parser.add_argument("--min_lufs",     default=None, type=float,
                        help="Keep only clips with LUFS >= N (e.g. -35).")
    parser.add_argument("--cache_dir",    default=None,
                        help="Local cache directory for downloaded JSON files. "
                             "Default: ~/.cache/afrivoice_manifests")
    parser.add_argument("--no_cache",     action="store_true", default=False,
                        help="Re-download files even if already cached.")
    parser.add_argument("--num_workers",  default=16, type=int,
                        help="Parallel download workers (default: 16).")
    args = parser.parse_args()

    # ── Parse filters ─────────────────────────────────────────────────────────
    domain_filter: set[str] | None = None
    if args.domain:
        domain_filter = {d.strip().lower() for d in args.domain.split(",")}
        invalid = domain_filter - VALID_DOMAINS
        if invalid:
            print(f"ERROR: Unknown domain(s): {invalid}. "
                  f"Valid: {sorted(VALID_DOMAINS)}", file=sys.stderr)
            sys.exit(1)

    split_filter: set[str] | None = None
    if args.split:
        split_filter = {s.strip().lower() for s in args.split.split(",")}
        invalid = split_filter - VALID_SPLITS
        if invalid:
            print(f"ERROR: Unknown split(s): {invalid}. "
                  f"Valid: {sorted(VALID_SPLITS)}", file=sys.stderr)
            sys.exit(1)

    cache_dir = Path(args.cache_dir) if args.cache_dir else \
                Path.home() / ".cache" / "afrivoice_manifests"
    use_cache = not args.no_cache

    # ── Build list of (domain, split) pairs to process ────────────────────────
    domains = sorted(domain_filter or VALID_DOMAINS)
    splits  = sorted(split_filter  or VALID_SPLITS)

    pairs: list[tuple[str, str]] = []
    for domain in domains:
        for split in splits:
            # scripted_education only has train
            if domain == "scripted_education" and split not in SCRIPTED_EDUCATION_SPLITS:
                continue
            pairs.append((domain, split))

    print(f"\n📋 Dataset : {DATASET_ID}")
    print(f"   Pairs   : {pairs}")
    print(f"   Cache   : {cache_dir}")
    print(f"   Login   : make sure you ran 'huggingface-cli login'\n")

    api = HfApi()

    # ── Step 1: collect all manifest repo paths ───────────────────────────────
    print("🔍 Listing manifest files …")
    all_repo_paths: list[tuple[str, str, str]] = []  # (domain, split, repo_path)

    for domain, split in pairs:
        repo_paths = list_manifest_files(api, domain, split)
        if not repo_paths:
            print(f"   ⚠  No manifests found for {domain}/{split} — skipping.")
            continue
        print(f"   {domain}/{split:<6}  →  {len(repo_paths):>3} manifest files")
        for rp in repo_paths:
            all_repo_paths.append((domain, split, rp))

    if not all_repo_paths:
        print("\nERROR: No manifest files found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n   Total manifest files to download: {len(all_repo_paths):,}\n")

    # ── Step 2: download JSON files in parallel ───────────────────────────────
    print(f"⬇  Downloading JSON manifests ({args.num_workers} workers) …")

    downloaded: list[tuple[str, str, Path]] = []  # (domain, split, local_path)
    failed_downloads = 0

    def _dl_worker(args_tuple):
        domain, split, repo_path = args_tuple
        local = download_manifest(repo_path, cache_dir, use_cache)
        return domain, split, local

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(_dl_worker, t): t for t in all_repo_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Downloading"):
            domain, split, local_path = future.result()
            if local_path is None:
                failed_downloads += 1
            else:
                downloaded.append((domain, split, local_path))

    if failed_downloads:
        print(f"   ⚠  {failed_downloads:,} files failed to download.")
    print(f"   ✅ {len(downloaded):,} files ready to parse.\n")

    # ── Step 3: parse all JSON files ──────────────────────────────────────────
    print("📖 Parsing manifest files …")

    all_records: list[dict] = []

    for domain, split, local_path in tqdm(downloaded, desc="  Parsing"):
        records = parse_manifest_file(local_path, domain, split)
        all_records.extend(records)

    if not all_records:
        print("\nERROR: No records parsed.", file=sys.stderr)
        sys.exit(1)

    print(f"   ✅ {len(all_records):,} total records parsed.\n")

    # ── Step 4: build DataFrame ───────────────────────────────────────────────
    print("🔗 Building DataFrame …")
    df = pd.DataFrame(all_records)

    # Drop empty transcriptions (check both text fields)
    before = len(df)
    for col in ("raw_text", "text"):
        if col in df.columns:
            df = df[df[col].notna() & (df[col].str.strip() != "")]
            break
    dropped_empty = before - len(df)

    # ── Apply numeric filters ──────────────────────────────────────────────────
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
        if args.min_duration is not None:
            df = df[df["duration"] >= args.min_duration]
        if args.max_duration is not None:
            df = df[df["duration"] <= args.max_duration]

    if "LUFS" in df.columns and args.min_lufs is not None:
        df["LUFS"] = pd.to_numeric(df["LUFS"], errors="coerce")
        df = df[df["LUFS"] >= args.min_lufs]

    # ── Drop audio/image path columns (not useful without the tar shards) ─────
    drop_cols = {"audio_filepath", "image_filepath", "image_shard_id"}
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── Reorder columns ────────────────────────────────────────────────────────
    priority = ["raw_text", "text", "domain", "split_type", "duration", "LUFS",
                "gender", "age_group", "location", "image_category",
                "image_sub_category", "shard_id", "creator_id"]
    ordered  = [c for c in priority if c in df.columns]
    rest     = [c for c in df.columns if c not in ordered]
    df       = df[ordered + rest].reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_hours = df["duration"].sum() / 3600 if "duration" in df.columns else 0

    print(f"\n{'─'*60}")
    print(f"  Total rows        : {len(df):,}")
    if dropped_empty:
        print(f"  Empty rows dropped: {dropped_empty:,}")
    print(f"  Total audio hours : {total_hours:.1f}h")
    print(f"  Columns           : {list(df.columns)}")
    print(f"  Saved to          : {output_path.resolve()}")

    print(f"\n  Breakdown by domain:")
    for domain, grp in df.groupby("domain"):
        hrs = grp["duration"].sum() / 3600 if "duration" in df.columns else 0
        print(f"    {domain:<25} {len(grp):>8,} rows  {hrs:>7.1f}h")

    print(f"\n  Breakdown by split:")
    for split_type, grp in df.groupby("split_type"):
        hrs = grp["duration"].sum() / 3600 if "duration" in df.columns else 0
        print(f"    {split_type:<10} {len(grp):>8,} rows  {hrs:>7.1f}h")

    if "gender" in df.columns:
        print(f"\n  Gender breakdown:")
        for gender, count in df["gender"].value_counts().items():
            print(f"    {str(gender):<10} {count:>8,}")

    if "LUFS" in df.columns:
        print(f"\n  LUFS  mean={df['LUFS'].mean():.1f}  "
              f"min={df['LUFS'].min():.1f}  "
              f"max={df['LUFS'].max():.1f}")

    print(f"{'─'*60}")
    print(f"\n🎉 Done!\n")
    print("  Preview (first 3 rows):")
    preview = ["raw_text", "text", "domain", "split_type", "duration", "LUFS", "gender"]
    preview = [c for c in preview if c in df.columns]
    print(df[preview].head(3).to_string(index=False))
    print()


if __name__ == "__main__":
    main()