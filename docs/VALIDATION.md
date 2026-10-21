# Packaging validation

Validated on **October 21, 2026**. This records checks of the standalone GitHub version, not new scientific validation on an independent test period.

## Clean setup and actual core download

- Created a fresh CPython 3.12 virtual environment outside the GitHub folder.
- Installed the 17-package runtime closure from `requirements-lock.txt` without using the earlier environment's system packages.
- Ran `scripts/download_data.py` into an empty external data directory.
- Downloaded all 108 required core source files using public URLs and no account.
- Recreated the expected Swarm directories, unpacked products, TLE subsets, reference index, and 980-row comparison index.
- Verified 317 downloaded/prepared core files, approximately 215 MB including extracted data.
- Historical products matched their archived hashes. The live IERS Earth-orientation table had updated upstream; its actual hash was recorded, and the numerical checks below still passed.

## Optional full data mode

- Added the 18 optional source files from the original verified local cache to the external test directory.
- Ran `scripts/download_data.py --all` to exercise cache handling and full preparation without unnecessarily redownloading the 221 MB ESA archive.
- All 126 source files passed the full preparation workflow.
- Recreated the original additional conjunction, space-weather, metadata, documentation, and orbital-snapshot layout.
- This was not a claim that every optional live endpoint was freshly downloaded. The core download above was a real empty-directory network test.

## Experiments

- Ran `scripts/run_all.py` using the fresh environment and external data directory.
- All three experiments completed from their copied, relocated source files.
- Passed the numerical regression checks, residual-input hash checks, reference/Cartesian error checks, and exact preservation of unselected forecasts.
- Reproduced the 540/96/344 split and the same 86 selected confidence-gated corrections: 77 improved, nine worsened.
- Reproduced the directional uncertainty model's approximately 6.0 percentage-point coverage mismatch and the reported position-error metrics.

## Repository contents

- Raw data are excluded by `.gitignore`; no input `data/` directory is included in this GitHub version.
- Source URLs and expected hashes are included as configuration metadata.
- Source files compile and local Markdown links resolve.
- Source code, documentation, configuration, saved numerical outputs, and figures are the intended committed contents.
- The preflight script checks staged contents for accidental raw data, environments, notebooks, proposals, machine-specific paths, symlinks, credential patterns, and oversized Git blobs.

The saved plots and results are still based on the same January 2025 study period. A new independent period is needed to confirm the scientific conclusions.
