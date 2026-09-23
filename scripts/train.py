from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuroseg.training import parser, train

if __name__ == '__main__':
    args = parser().parse_args()
    if min(args.epochs, args.steps_per_epoch, args.batch_size, args.threads) < 1:
        raise SystemExit('Epochs, steps, batch size and threads must be positive')
    train(args)
