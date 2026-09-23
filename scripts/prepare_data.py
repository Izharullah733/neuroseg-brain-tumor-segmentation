import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuroseg.data import prepare

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('data/raw/Task01_BrainTumour'))
    p.add_argument('--out', type=Path, default=Path('data/processed'))
    p.add_argument('--size', type=int, default=128)
    a = p.parse_args()
    prepare(a.root, a.out, a.size)
