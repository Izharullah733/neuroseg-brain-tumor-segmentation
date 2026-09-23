import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from neuroseg.data import load_case, file_hash
from neuroseg.inference import predict, load_model
from neuroseg.metrics import evaluate
from neuroseg.reporting import write_report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data/processed'))
    p.add_argument('--checkpoint', type=Path, default=Path('runs/baseline/best.pt'))
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--out', type=Path, default=Path('runs/evaluation'))
    args = p.parse_args()
    manifest = json.loads((args.data / 'manifest.json').read_text())
    _, checkpoint = load_model(args.checkpoint)
    if checkpoint['manifest_sha256'] != file_hash(args.data / 'manifest.json'):
        raise ValueError('Evaluation manifest does not match training provenance')
    results = []
    for case in manifest['splits'][args.split]:
        info = manifest['cases'][case]
        out = args.out / case
        image, _, mask, report = predict(info['image'], args.checkpoint, out)
        _, _, truth = load_case(info['image'], info['label'])
        report['case_id'], report['split'] = case, args.split
        report['metrics'] = evaluate(mask, truth, image.header.get_zooms()[:3])
        write_report(report, out)
        results.append(report)
        print(case, {k: round(v['dice'], 4) for k, v in report['metrics']['regions'].items()}, flush=True)
    rng = np.random.default_rng(42)
    summary = {'split': args.split, 'patients': len(results), 'checkpoint_sha256': file_hash(args.checkpoint),
        'interpretation': 'Pilot only; small-cohort intervals are unstable.', 'region_dice': {}}
    for region in ('WT', 'TC', 'ET'):
        values = np.array([r['metrics']['regions'][region]['dice'] for r in results])
        means = rng.choice(values, size=(2000, len(values)), replace=True).mean(1)
        summary['region_dice'][region] = {'mean': float(values.mean()),
            'patient_bootstrap_95ci': [float(v) for v in np.percentile(means, [2.5, 97.5])]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
