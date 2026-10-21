# Using this GitHub version

This is a standalone repository folder for the data-download workflow and the three variance/correction experiments. It contains no notebooks, proposal PDFs, raw numerical datasets, or Python environment. Its saved experiment outputs and figures are included.

## Publish the prepared local repository

The local branch is `codex/github-ready`. Create an empty GitHub repository, without initializing another README, then run from this folder:

```bash
python3.12 scripts/check_github_ready.py
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin HEAD:main
```

Replace the example URL. The final command creates/updates remote `main` from the local prepared branch. Authentication uses your normal GitHub SSH or HTTPS setup. No token should be placed in source code or committed configuration. If `origin` already exists, inspect it with `git remote -v` and use `git remote set-url origin ...` if necessary.

For later edits:

```bash
git add .
python3.12 scripts/check_github_ready.py
git commit -m "Describe the change"
git push origin HEAD:main
```

## What is excluded

`.gitignore` excludes `/data/`, virtual environments, Python bytecode, local settings, credentials files, and incomplete downloads. The preflight checker examines the staged/tracked files and fails if raw data, environments, notebooks, PDF proposals, symlinks, machine-specific paths, or oversized ordinary Git blobs are present.

GitHub's [ordinary-file limit is 100 MiB](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). This version avoids the limit by leaving raw data out of Git. **Git LFS is not required.** Do not force-add `data/`.

## Clone and reproduce

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/run_all.py
```

The default downloader obtains the core Swarm inputs. Add `--all` to also obtain the optional conjunction, orbital-snapshot, weather, and documentation files. Internet is needed for installation and downloading. Once data are present, the experiments use the local files.

## Inspect before pushing

```bash
git status --short
git ls-files data
python3.12 scripts/check_github_ready.py
```

`git ls-files data` should print nothing, even after running the downloader. Review the data-source notices; provider terms still apply when someone downloads data using these scripts. Review the results' limitations before describing them as improvements to SGP4.
