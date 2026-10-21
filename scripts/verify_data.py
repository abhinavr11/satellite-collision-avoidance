#!/usr/bin/env python3
"""Check downloaded/prepared input hashes using only the Python standard library."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]


def verify(data_dir):
    data_dir = Path(data_dir).resolve()
    failures = []
    checked = {}
    for filename in ("manifest.json", "prepared_files_manifest.json"):
        manifest = data_dir/filename
        if not manifest.is_file():
            failures.append(f"Missing {manifest}; run scripts/download_data.py first")
            continue
        for relative, entry in json.loads(manifest.read_text()).items():
            path = data_dir/relative
            if not path.resolve().is_relative_to(data_dir):
                failures.append(f"Unsafe path in {filename}: {relative}")
                continue
            if not path.is_file():
                failures.append(f"Missing {relative}")
                continue
            if relative not in checked:
                h=hashlib.sha256()
                with path.open("rb") as f:
                    for chunk in iter(lambda:f.read(1024*1024), b""):
                        h.update(chunk)
                checked[relative]=(path.stat().st_size, h.hexdigest())
            actual_size, actual_hash=checked[relative]
            if actual_size != entry["bytes"] or actual_hash != entry["sha256"]:
                failures.append(f"Hash or size mismatch: {relative}")
    summary=data_dir/"validation_summary.json"
    if not summary.is_file():
        failures.append("Missing validation_summary.json; data preparation is incomplete")
    else:
        record=json.loads(summary.read_text())
        if record.get("swarm_reference_files") != 102 or record.get("calibration_pairs_index_rows") != 980:
            failures.append("Unexpected Swarm file count or matching-index size")
    if failures:
        for item in failures:
            print("ERROR:", item, file=sys.stderr)
        raise SystemExit(1)
    result={"data_dir":str(data_dir),"verified_files":len(checked),"verified_bytes":sum(v[0] for v in checked.values())}
    print(json.dumps(result, indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=Path(os.environ.get("SGP4_DATA_DIR",ROOT/"data")))
    args=parser.parse_args()
    verify(args.data_dir)


if __name__ == "__main__":
    main()
