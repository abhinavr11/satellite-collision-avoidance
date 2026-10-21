#!/usr/bin/env python3
"""Verify downloads, unpack provider files, and make small convenience indexes.

This does not propagate orbits, transform coordinate frames, or fit uncertainty.
"""
from pathlib import Path
import os
from datetime import datetime, timedelta
import csv
import gzip
import hashlib
import json
import math
import shutil
import zipfile

ROOT = Path(os.environ.get("SGP4_DATA_DIR", Path(__file__).resolve().parents[1] / "data")).resolve()
SATELLITES = {"A": ("39452", "L47"), "B": ("39451", "L48"), "C": ("39453", "L49")}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract_zip(path, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            target = dest / entry.filename
            if not target.resolve().is_relative_to(dest.resolve()):
                raise ValueError(f"Unsafe archive path: {entry.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def parse_epoch(line):
    values = line[1:].split()
    return datetime(*map(int, values[:5])) + timedelta(seconds=float(values[5]))


def utc_text(time):
    return time.isoformat(timespec="microseconds") + "Z"


def check_swarm():
    index = []
    for satellite, (norad, sp3id) in SATELLITES.items():
        folder = ROOT / f"reference_orbits/swarm_{satellite}"
        archives = sorted(folder.glob("*.ZIP"))
        assert len(archives) >= 33, f"Missing reference days for Swarm-{satellite}"
        for archive in archives:
            extract_zip(archive, folder / "extracted")
        for path in sorted((folder / "extracted").glob("*.sp3")):
            first = last = None
            epoch_count = positions = velocities = 0
            previous = None
            with path.open() as f:
                header = f.readline().rstrip()
                declared_count = int(header[32:39])
                frame = header[46:51].strip()
                assert frame in {"IGS20", "IGb20"}, (path.name, frame)
                for line in f:
                    if line.startswith("%c M"):
                        assert line.split()[3] == "GPS", (path, line)
                    if line.startswith("* "):
                        current = parse_epoch(line)
                        first = first or current
                        if previous:
                            assert (current - previous).total_seconds() == 10
                        previous = last = current
                        epoch_count += 1
                    elif line.startswith("P"):
                        assert line[1:4] == sp3id
                        xyz = [float(line[start:start + 14]) * 1000 for start in (4, 18, 32)]
                        assert 6.5e6 < math.sqrt(sum(v*v for v in xyz)) < 7.5e6
                        positions += 1
                    elif line.startswith("V"):
                        assert line[1:4] == sp3id
                        vel = [float(line[start:start + 14]) * 0.1 for start in (4, 18, 32)]
                        assert 6000 < math.sqrt(sum(v*v for v in vel)) < 9000
                        velocities += 1
            assert epoch_count == declared_count == positions == velocities == 8640
            # The entire selected January/February 2025 interval has GPS-UTC = 18 s.
            index.append({
                "satellite": f"Swarm-{satellite}", "norad_cat_id": norad, "sp3_id": sp3id,
                "reference_file": str(path.relative_to(ROOT)), "frame": frame,
                "source_time_scale": "GPS", "position_unit": "km", "velocity_unit": "dm/s",
                "epoch_count": epoch_count, "step_seconds": 10,
                "first_epoch_utc": utc_text(first - timedelta(seconds=18)),
                "last_epoch_utc": utc_text(last - timedelta(seconds=18)),
            })
    write_csv(ROOT / "reference_orbits/index.csv", index)
    return index


def check_tles(reference_index):
    jobs = []
    summary = {}
    git_tree = json.loads((ROOT / "metadata/historical_tle_git_tree.json").read_text())
    git_files = {row["path"]: row for row in git_tree["tree"]}
    for satellite, (norad, _) in SATELLITES.items():
        path = ROOT / f"orbit_predictions/historical/Swarm-{satellite}.tle"
        data = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert blob == git_files[f"TLE/Swarm-{satellite}.tle"]["sha"]
        lines = path.read_text().splitlines()
        assert len(lines) % 2 == 0
        unique = {}
        for i in range(0, len(lines), 2):
            line1, line2 = lines[i:i+2]
            assert line1.startswith("1 ") and line2.startswith("2 ")
            assert line1[2:7] == line2[2:7] == norad
            for line in (line1, line2):
                assert len(line) == 69
                checksum = sum(int(c) if c.isdigit() else int(c == "-") for c in line[:68]) % 10
                assert checksum == int(line[68])
            yy = int(line1[18:20])
            year = 2000 + yy if yy < 57 else 1900 + yy
            epoch = datetime(year, 1, 1) + timedelta(days=float(line1[20:32]) - 1)
            unique[(line1, line2)] = epoch
        selected = sorted((epoch, pair) for pair, epoch in unique.items()
                          if datetime(2024, 12, 29) <= epoch < datetime(2025, 2, 3))
        out = ROOT / f"orbit_predictions/calibration_window/Swarm-{satellite}_2025-01.tle"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(line + "\n" for _, pair in selected for line in pair))
        refs = [r for r in reference_index if r["satellite"] == f"Swarm-{satellite}"]
        for pair_number, (epoch, pair) in enumerate(selected):
            if not datetime(2025, 1, 1) <= epoch < datetime(2025, 2, 1):
                continue
            for horizon in (6, 12, 24, 48):
                target = epoch + timedelta(hours=horizon)
                match = next((r for r in refs if
                              datetime.fromisoformat(r["first_epoch_utc"].removesuffix("Z")) <= target <=
                              datetime.fromisoformat(r["last_epoch_utc"].removesuffix("Z"))), None)
                if match:
                    jobs.append({"satellite": f"Swarm-{satellite}", "norad_cat_id": norad,
                                 "tle_file": str(out.relative_to(ROOT)), "tle_pair_index_zero_based": pair_number,
                                 "tle_epoch_utc": utc_text(epoch), "horizon_hours": horizon,
                                 "comparison_time_utc": utc_text(target), "reference_file": match["reference_file"]})
        summary[f"Swarm-{satellite}"] = {
            "norad_cat_id": norad, "total_tle_pairs": len(lines)//2, "unique_tle_pairs": len(unique),
            "window_unique_pairs": len(selected), "first_epoch": utc_text(min(unique.values())),
            "last_epoch": utc_text(max(unique.values())), "git_blob_verified": True,
            "all_tle_checksums_valid": True,
        }
    write_csv(ROOT / "calibration_pairs_index.csv", jobs)
    return summary, len(jobs)


def csv_info(path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        count = 0
        events = set()
        for row in reader:
            assert None not in row, f"Unexpected columns in {path}"
            assert all(value is not None for value in row.values()), f"Short CSV record in {path}"
            count += 1
            if "event_id" in row:
                events.add(row["event_id"])
    return {"file": str(path.relative_to(ROOT)), "rows": count, "column_count": len(reader.fieldnames),
            "event_count": len(events) if events else None, "columns": reader.fieldnames}


def check_conjunctions():
    folder = ROOT / "conjunctions/esa"
    extract_zip(ROOT / "conjunctions/esa_collision_avoidance_dataset.zip", folder)
    nested = next(folder.rglob("train_data.zip"))
    extract_zip(nested, nested.parent)
    tables = [csv_info(path) for path in sorted(folder.rglob("*.csv"))]
    # Keep the provider's raw archive compressed; it includes extra CDM fields.
    raw = next(folder.rglob("raw_data_2015-2019.gz"))
    with gzip.open(raw, "rt") as f:
        header = f.readline()
    (ROOT / "documentation/esa_raw_header.txt").write_text(header)
    with (ROOT / "orbit_predictions/current_active_gp.csv").open(newline="") as f:
        gp = {row["NORAD_CAT_ID"]: row for row in csv.DictReader(f)}
    selected = []
    with (ROOT / "conjunctions/socrates_snapshot.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            count += 1
            first = gp.get(row["NORAD_CAT_ID_1"])
            second = gp.get(row["NORAD_CAT_ID_2"])
            if not first or not second:
                continue
            # Exclude docked/formation objects and require prediction epochs before TCA.
            if float(row["TCA_RELATIVE_SPEED"]) < 1.0:
                continue
            tca = datetime.fromisoformat(row["TCA"])
            if max(datetime.fromisoformat(first["EPOCH"]), datetime.fromisoformat(second["EPOCH"])) >= tca:
                continue
            if len(selected) < 100:
                selected.append(row)
    assert selected, "No current conjunctions matched to both objects' orbit elements"
    write_csv(ROOT / "conjunctions/current_cases_with_orbits.csv", selected)
    selected_ids = {row[f"NORAD_CAT_ID_{i}"] for row in selected for i in (1, 2)}
    write_csv(ROOT / "orbit_predictions/current_case_objects_gp.csv", [gp[key] for key in sorted(selected_ids)])
    return {"esa_tables": tables, "socrates_rows": count, "current_cases_with_both_orbits": len(selected),
            "current_case_objects": len(selected_ids), "active_gp_objects": len(gp)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", action="store_true", help="Prepare only the Swarm inputs needed by the experiments")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for relpath, entry in manifest.items():
        path = ROOT / relpath
        assert path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"], relpath
    refs = check_swarm()
    tles, jobs = check_tles(refs)
    conjunctions = None if args.core else check_conjunctions()
    summary = {"downloaded_files": len(manifest), "all_download_sha256_verified": True,
               "swarm_reference_files": len(refs), "swarm_reference_epochs": sum(r["epoch_count"] for r in refs),
               "calibration_pairs_index_rows": jobs, "historical_tle_validation": tles,
               "conjunction_validation": conjunctions}
    (ROOT / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    derived = {}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and str(path.relative_to(ROOT)) not in manifest and path.suffix.lower() in {".sp3", ".hdr", ".csv", ".tle", ".gz"}:
            derived[str(path.relative_to(ROOT))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (ROOT / "prepared_files_manifest.json").write_text(json.dumps(derived, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in
                      {"historical_tle_validation", "conjunction_validation"}}, indent=2))
    if conjunctions is not None:
        print(json.dumps(conjunctions, indent=2))


if __name__ == "__main__":
    main()
