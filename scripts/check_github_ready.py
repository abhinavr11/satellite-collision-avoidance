#!/usr/bin/env python3
"""Check staged/tracked contents; never creates a remote or uploads anything."""
from pathlib import Path
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
DEMO_NOTEBOOK = 'notebooks/Shifted_Swarm_Demo.ipynb'


def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT)


def main():
    names=[s.decode() for s in git("ls-files","-z").split(b"\0") if s]
    if not names:
        raise SystemExit("No staged/tracked files. Run git add . before this check.")
    forbidden=[]
    size=0
    for name in names:
        path=ROOT/name
        if name.startswith(("data/","data-",".venv/",".venv-","venv/",".local/")) or "__pycache__" in path.parts or path.suffix in {".pdf",".pyc"} or (path.suffix == '.ipynb' and name != DEMO_NOTEBOOK):
            forbidden.append(name)
        index_mode=git("ls-files","-s","--",name).decode().split()[0]
        if index_mode=="120000":
            raise SystemExit(f"Unexpected symlink: {name}")
        blob_size=int(git("cat-file","-s",":"+name))
        size+=blob_size
        if blob_size>100*1024*1024:
            raise SystemExit(f"File exceeds GitHub's ordinary-file limit: {name}")
        if path.suffix in {".py",".md",".json",".txt",".yml",".yaml", ".ipynb"}:
            text=git("show",":"+name).decode(errors="replace")
            if path.suffix == '.ipynb':
                notebook = json.loads(text)
                if notebook.get('nbformat') != 4 or not notebook.get('cells'):
                    raise SystemExit(f'Invalid notebook structure: {name}')
                fragments = []
                for cell in notebook['cells']:
                    fragments.append(''.join(cell['source']))
                    for output in cell.get('outputs', []):
                        if output.get('output_type') == 'error':
                            raise SystemExit(f'Saved notebook error: {name}')
                        fragments.append(''.join(output.get('text', [])))
                        for key, value in output.get('data', {}).items():
                            if key.startswith('text/'):
                                fragments.append(''.join(value))
                text = '\n'.join(fragments)
            local_text=re.sub(r"(?:https?|ftp)://[^\s)>]+", "", text)
            if re.search(r"/(?:Users|home)/[^/\s]+/",local_text):
                raise SystemExit(f"Machine-specific absolute path: {name}")
            if re.search(r"\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16})\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",text):
                raise SystemExit(f"Possible credential in {name}; inspect before publishing")
    if forbidden:
        raise SystemExit("Unwanted staged files: "+", ".join(forbidden))
    required={"README.md","requirements-lock.txt","config/source_manifest.json",
              "config/swarm_ac_shifted_encounter.json","scripts/download_data.py","scripts/run_all.py",
              "scripts/prepare_shifted_encounter.py","scripts/check_results.py",
              "scripts/check_collision_results.py","scripts/check_encounter_forecast_results.py",
              "collision_probability/model.py","collision_probability/probability.py",
              "collision_probability/data.py","collision_probability/evaluation.py",
              "collision_probability/encounter_forecast.py","collision_probability/confidence.py",
              "collision_probability/window_selection.py",
              "experiments/collision_probability/run_experiment.py",
              "experiments/encounter_train_test/run_experiment.py",
              "experiments/encounter_train_test/report.md",
              "experiments/encounter_train_test/summary.json",
              "tests/test_collision_probability.py","tests/test_encounter_forecast.py",
              "tests/test_confidence_gate.py","tests/test_window_balance.py",
              "docs/GITHUB.md","docs/METHODS.md","docs/VALIDATION.md",
              DEMO_NOTEBOOK, "scripts/build_colab_notebook.py",
              "scripts/check_bundled_encounter.py", "scripts/validate_colab_notebook.py",
              "requirements-colab.txt", "datasets/README.md",
              "datasets/swarm_ac_shifted_v1/encounter.json",
              "datasets/swarm_ac_shifted_v1/reference_trajectories.csv",
              "datasets/swarm_ac_shifted_v1/sgp4_trajectories.csv",
              "datasets/swarm_ac_shifted_v1/aligned_trajectories.csv"}
    if not required.issubset(names):
        raise SystemExit("Required project files are not staged")
    # Verify that the dataset and virtual environments will remain excluded later.
    for name in ["data/example.csv","data/reference_orbits/example.sp3",".venv/example.py"]:
        if subprocess.run(["git","check-ignore","--quiet",name],cwd=ROOT).returncode:
            raise SystemExit(f"Ignore rule missing for {name}")
    print(json.dumps({"tracked_files":len(names),"tracked_bytes":size,
                      "bulk_source_data_tracked":False,"bundled_shifted_dataset":True,
                      "notebooks_tracked":[DEMO_NOTEBOOK],
                      "git_lfs_required":False,"branch":git("branch","--show-current").decode().strip()},indent=2))


if __name__ == "__main__":
    main()
