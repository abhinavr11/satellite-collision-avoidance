# Using the finalized GitHub repository

The repository contains five reproducible experiments, the collision-probability implementation, tests, source-data manifests, saved results and figures. The selected shifted Swarm A/C case is versioned in `datasets/swarm_ac_shifted_v1`, together with its source files and native timestamps. The larger source collections, environments and caches remain excluded.

## Professor-facing Colab demonstration

After pushing `main`, [open the notebook in Colab](https://colab.research.google.com/github/abhinavr11/satellite-collision-avoidance/blob/main/notebooks/Shifted_Swarm_Demo.ipynb) and choose **Runtime → Run all**. The notebook clones this repository and loads the bundled data to demonstrate both trajectory correction and collision probability. A CPU runtime is sufficient. The essential equations appear in Markdown boxes; derivations and controls are in the appendix. All results use the same shifted encounter.

The notebook has saved outputs from local execution against a separate Git clone. Hosted Colab execution is left to the reader after publication. It does not require the bulk downloader. Keep the data folder intact: SHA-256 checks and common-time validation run when it is loaded.

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

Use `.venv\Scripts\python.exe` on Windows. The default core downloader supplies the larger historical inputs needed for the complete five-experiment pipeline. The encounter code prefers the checked-in bundle; the recipe remains available for reconstruction. Add `--all` only for the optional conjunction/weather/source-documentation collection.

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
.venv/bin/python scripts/check_bundled_encounter.py
```

The current comparison, selected split, confidence decisions and both uncertainty assumptions are in [the encounter report](../experiments/encounter_train_test/report.md). [Methods](METHODS.md) explains the conditional probability model and its limits; [validation](VALIDATION.md) records the checks. Reference trajectories supply a binary collision label rather than a reference Pc.

## What stays local

`.gitignore` excludes bulk `data/`, `data-*` directories, environments, Python caches, notebook rerun outputs in `demo_outputs/`, local settings and incomplete downloads. The selected input bundle, one demonstration notebook, and saved experiment outputs are intentionally included. Git LFS is not required. Do not force-add bulk downloads or environments.

For notebook authoring, `scripts/build_colab_notebook.py` regenerates the narrative and code (clearing saved outputs). Install `requirements-notebook-qa.txt` in a QA environment and stage the intended sources, then run `python scripts/validate_colab_notebook.py`. This executes the notebook against an isolated clone of the staged tree and saves successful outputs. The numerical runtime and QA libraries must be available to that Python interpreter. The validator does not push or change branches.

The preflight checker verifies staged blobs, including ignored-path violations and the required implementation/tests. It does not push, create a remote repository, or merge anything. Review the [data-source notices](DATA_SOURCES.md); provider terms still apply to data obtained with the downloader.
