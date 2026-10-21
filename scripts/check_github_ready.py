#!/usr/bin/env python3
"""Check staged/tracked contents; never creates a remote or uploads anything."""
from pathlib import Path
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


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
        if name.startswith(("data/",".venv/",".local/")) or "__pycache__" in path.parts or path.suffix in {".ipynb",".pdf",".pyc"}:
            forbidden.append(name)
        if path.is_symlink():
            raise SystemExit(f"Unexpected symlink: {name}")
        blob_size=int(git("cat-file","-s",":"+name))
        size+=blob_size
        if blob_size>100*1024*1024:
            raise SystemExit(f"File exceeds GitHub's ordinary-file limit: {name}")
        if path.suffix in {".py",".md",".json",".txt",".yml",".yaml"}:
            text=path.read_text(errors="replace")
            local_text=re.sub(r"(?:https?|ftp)://[^\s)>]+", "", text)
            if re.search(r"/(?:Users|home)/[^/\s]+/",local_text):
                raise SystemExit(f"Machine-specific absolute path: {name}")
            if re.search(r"\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16})\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",text):
                raise SystemExit(f"Possible credential in {name}; inspect before publishing")
    if forbidden:
        raise SystemExit("Unwanted staged files: "+", ".join(forbidden))
    required={"README.md","requirements-lock.txt","config/source_manifest.json","scripts/download_data.py","scripts/run_all.py"}
    if not required.issubset(names):
        raise SystemExit("Required project files are not staged")
    # Verify that the dataset and virtual environments will remain excluded later.
    for name in ["data/example.csv","data/reference_orbits/example.sp3",".venv/example.py"]:
        if subprocess.run(["git","check-ignore","--quiet",name],cwd=ROOT).returncode:
            raise SystemExit(f"Ignore rule missing for {name}")
    print(json.dumps({"tracked_files":len(names),"tracked_bytes":size,
                      "raw_data_tracked":False,"notebooks_tracked":False,
                      "git_lfs_required":False,"branch":git("branch","--show-current").decode().strip()},indent=2))


if __name__ == "__main__":
    main()
