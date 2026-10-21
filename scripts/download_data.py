#!/usr/bin/env python3
"""Download and organize Swarm data. No account or Git LFS is required.

Default: inputs for the three numerical experiments. --all: also download the
ESA conjunction release, live orbital snapshots, space weather, and source docs.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOCK = threading.Lock()
IMMUTABLE = {"historical_orbit_predictions", "precise_reference_trajectory", "historical_conjunctions"}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_core(relative):
    return (relative.startswith("reference_orbits/") or
            relative.startswith("orbit_predictions/historical/") or
            relative.startswith("earth_orientation/") or
            relative == "metadata/historical_tle_git_tree.json")


def save_manifest(data_dir, records):
    partial = data_dir / "manifest.json.tmp"
    partial.write_text(json.dumps(records, indent=2, sort_keys=True)+"\n")
    partial.replace(data_dir / "manifest.json")


def fetch(relative, archived, data_dir, records):
    path = data_dir / relative
    if not path.resolve().is_relative_to(data_dir.resolve()):
        raise ValueError(f"Unsafe manifest path: {relative}")
    old = records.get(relative)
    if path.is_file():
        actual = sha256(path)
        if old and actual == old["sha256"]:
            return "cached"
        if actual == archived["sha256"]:
            with LOCK:
                records[relative] = archived.copy()
                save_manifest(data_dir, records)
            return "cached"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name+".part")
    last_error = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(archived["source_url"], headers={"User-Agent": "CMU-SGP4-experiments/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response, partial.open("wb") as out:
                final_url = response.url
                content_type = response.headers.get("Content-Type", "")
                advertised_size = response.headers.get("Content-Length")
                for chunk in iter(lambda: response.read(1024*1024), b""):
                    out.write(chunk)
            if advertised_size and partial.stat().st_size != int(advertised_size):
                raise ValueError("Incomplete HTTP response")
            actual = sha256(partial)
            strict = archived["category"] in IMMUTABLE
            if strict and actual != archived["sha256"]:
                raise ValueError("Historical source hash differs from the archived snapshot; refusing to use changed data")
            if path.suffix.lower() in {".zip", ".csv", ".tle", ".all", ".dat", ".txt"}:
                with partial.open("rb") as f:
                    head = f.read(200).lstrip().lower()
                if head.startswith((b"<!doctype html", b"<html")):
                    raise ValueError("Provider returned HTML instead of data")
            if path.suffix.lower() == ".zip":
                import zipfile
                with zipfile.ZipFile(partial) as z:
                    bad = z.testzip()
                    if bad:
                        raise ValueError(f"ZIP CRC failure: {bad}")
            partial.replace(path)
            entry = {"source_url": archived["source_url"], "resolved_url": final_url,
                     "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                     "category": archived["category"], "bytes": path.stat().st_size,
                     "sha256": actual, "content_type": content_type,
                     "archived_sha256": archived["sha256"],
                     "matches_archived_snapshot": actual == archived["sha256"]}
            with LOCK:
                records[relative] = entry
                save_manifest(data_dir, records)
            status = "downloaded" if actual == archived["sha256"] else "downloaded updated upstream snapshot"
            print(f"{status}: {relative}", flush=True)
            return status
        except Exception as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(attempt+1)
    raise RuntimeError(f"{relative}: {last_error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Also download and prepare the extra conjunction/orbit/space-weather datasets")
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data", help="Output folder (default: repository/data)")
    parser.add_argument("--download-only", action="store_true", help="Fetch source files without extracting or preparing them")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    recorded = json.loads((ROOT/"config/source_manifest.json").read_text())
    selected = {k:v for k,v in recorded.items() if args.all or is_core(k)}
    current = data_dir/"manifest.json"
    records = json.loads(current.read_text()) if current.exists() else {}
    failures = []
    cached = 0
    print(f"Downloading {'all' if args.all else 'core'} data: {len(selected)} source files into {data_dir}", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {pool.submit(fetch, name, item, data_dir, records): name for name,item in selected.items()}
        for task in as_completed(tasks):
            try:
                cached += task.result() == "cached"
            except Exception as exc:
                failures.append({"file":tasks[task], "error":str(exc)})
                print(f"FAILED: {exc}", flush=True)
    (data_dir/"download_failures.json").write_text(json.dumps(failures, indent=2)+"\n")
    print(f"Sources complete: {len(selected)-len(failures)} OK, {cached} cached, {len(failures)} failed.", flush=True)
    if failures:
        raise SystemExit(1)
    if not args.download_only:
        env = dict(os.environ, SGP4_DATA_DIR=str(data_dir))
        cmd = [sys.executable, str(ROOT/"scripts/prepare_data.py")]
        if not args.all:
            cmd.append("--core")
        subprocess.run(cmd, env=env, check=True)
        subprocess.run([sys.executable, str(ROOT/"scripts/verify_data.py"), "--data-dir", str(data_dir)], check=True)
    print(f"Ready: {data_dir}", flush=True)


if __name__ == "__main__":
    main()
