# Using the finalized GitHub repository

The repository contains five reproducible experiments, the collision-probability implementation, 31 tests, source-data manifests, the shifted-encounter recipe, saved derived results and figures. Downloaded SP3/TLE source collections, virtual environments, temporary files and caches are excluded.

## Push main

Run from the repository folder. The configured remote is `https://github.com/abhinavr11/satellite-collision-avoidance.git` and the finalized branch is `main`.

```bash
git status --short
python3.12 scripts/check_github_ready.py
git push -u origin main
```

The push updates the remote `main` with the committed local changes. Authentication uses the normal Git credential setup; credentials do not belong in source files. The finalization workflow creates a local commit; no remote upload is performed automatically.

For later edits, review and stage the intended files, run the checks, and commit before pushing:

```bash
git add <files-you-changed>
python3.12 scripts/check_github_ready.py
git commit -m "Describe the change"
git push
```

## Clone and reproduce

```bash
git clone --branch main https://github.com/abhinavr11/satellite-collision-avoidance.git
cd satellite-collision-avoidance
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/run_all.py
```

Use `.venv\Scripts\python.exe` on Windows. The default core downloader supplies every input needed by all five experiments. The shifted Swarm A/C bundle is reconstructed automatically from `config/swarm_ac_shifted_encounter.json`; no original machine paths or temporary search files are required. Add `--all` only for the optional conjunction/weather/source-documentation collection.

A custom existing data directory works with:

```bash
.venv/bin/python scripts/run_all.py --data-dir /path/to/data
```

## Check saved results without downloading data

The source-independent numerical tests and saved-result validators require the installed dependencies but no downloaded raw data:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_results.py
.venv/bin/python scripts/check_collision_results.py
.venv/bin/python scripts/check_encounter_forecast_results.py
```

The current comparison, selected split, confidence decisions and both uncertainty assumptions are in [the encounter report](../experiments/encounter_train_test/report.md). [Methods](METHODS.md) explains the conditional probability model and its limits; [validation](VALIDATION.md) records the checks. Reference trajectories supply a binary collision label rather than a reference Pc.

## What stays local

`.gitignore` excludes downloaded `data/`, `data-*` directories, environments, Python caches, local settings and incomplete downloads. Saved derived experiment outputs are intentionally included. Git LFS is not required. Do not force-add the raw data or environment.

The preflight checker verifies staged blobs, including ignored-path violations and the required implementation/tests. It does not push, create a remote repository, or merge anything. Review the [data-source notices](DATA_SOURCES.md); provider terms still apply to data obtained with the downloader.
