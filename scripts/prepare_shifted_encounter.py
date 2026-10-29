"""Prepare the two-satellite bundle from core data, or verify an existing bundle."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from collision_probability.data import ensure_encounter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    print(ensure_encounter(args.data_dir.expanduser().resolve(), args.rebuild))
