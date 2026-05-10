# fix_metadata.py
import csv
import shutil
from pathlib import Path

input_path  = Path("MyTTSDataset/metadata.csv")
backup_path = Path("MyTTSDataset/metadata.csv.bak2")

shutil.copy2(input_path, backup_path)
print(f"Backup saved: {backup_path}")

clean_rows = []
dropped    = []

with input_path.open("r", encoding="utf-8") as fh:
    for i, line in enumerate(fh, start=1):
        parts = line.rstrip("\n").split("|")
        if len(parts) == 3:
            clean_rows.append(parts)
        else:
            dropped.append((i, len(parts), line.rstrip()[:100]))

print(f"Kept   : {len(clean_rows):,}")
print(f"Dropped: {len(dropped):,}")

if dropped:
    print(f"\nDropped lines (first 10):")
    for lineno, ncols, content in dropped[:10]:
        print(f"  Line {lineno:>6} ({ncols} fields): {content}")

with input_path.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.writer(fh, delimiter="|", quoting=csv.QUOTE_MINIMAL)
    writer.writerows(clean_rows)

print(f"\n✅ Done → {input_path}")