#!/usr/bin/env python3
"""Run all five experiments in order, then validate the saved results."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data")
    args=parser.parse_args()
    data_dir=args.data_dir.expanduser().resolve()
    env=dict(os.environ,SGP4_DATA_DIR=str(data_dir))
    subprocess.run([sys.executable,str(ROOT/"scripts/verify_data.py"),"--data-dir",str(data_dir)],check=True,env=env)
    for name in ["sgp4_uncertainty_pilot","sgp4_position_correction_pilot","sgp4_confidence_gated_correction","collision_probability","encounter_train_test"]:
        print(f"\nRunning {name}",flush=True)
        subprocess.run([sys.executable,str(ROOT/"experiments"/name/"run_experiment.py")],check=True,env=env)
    subprocess.run([sys.executable,str(ROOT/"scripts/check_results.py")],check=True,env=env)
    subprocess.run([sys.executable,str(ROOT/"scripts/check_collision_results.py")],check=True,env=env)
    subprocess.run([sys.executable,str(ROOT/"scripts/check_encounter_forecast_results.py")],check=True,env=env)
    subprocess.run([sys.executable,"-m","unittest","discover","-s",str(ROOT/"tests")],check=True,env=env,cwd=ROOT)


if __name__ == "__main__":
    main()
