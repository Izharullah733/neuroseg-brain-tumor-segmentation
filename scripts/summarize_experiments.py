"""Build a traceable pilot report from saved artifacts, without inventing scores."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroseg.inference import load_model


if __name__ == '__main__':
    root = Path('runs')
    lines = ['# Local pilot experiment results', '',
        'This is a development pilot on 24 MSD BraTS-derived cases: 16 train, 4 validation,',
        '4 held-out test. **Only validation is reported; the test split remains unused.**', '',
        '## Original-grid validation Dice', '',
        '| Experiment | WT | TC | ET | Best epoch | Completed epochs | Logged training + validation seconds |',
        '|---|---:|---:|---:|---:|---:|---:|']
    details = []
    for history in sorted(root.glob('*/history.jsonl')):
        run = history.parent
        summary_path = root / f'{run.name}_validation' / 'summary.json'
        if not summary_path.exists():
            continue
        rows = [json.loads(line) for line in history.read_text().splitlines() if line]
        summary = json.loads(summary_path.read_text())
        _, checkpoint = load_model(run / 'best.pt')
        scores = summary['region_dice']
        lines.append(f"| {run.name} | {scores['WT']['mean']:.4f} | {scores['TC']['mean']:.4f} | {scores['ET']['mean']:.4f} | {checkpoint['epoch']+1} | {rows[-1]['epoch']} | {sum(r['elapsed_seconds'] for r in rows):.1f} |")
        details += [f'### {run.name}', '', f"- Optimizer steps completed: {rows[-1]['steps']}; selected checkpoint steps: {checkpoint['steps']}.",
            f"- Input channels: {checkpoint['model_config']['in_channels']}; width: {checkpoint['model_config']['width']}.",
            f"- Class weighting: {checkpoint['config'].get('class_balance', 'none')}.",
            f"- Checkpoint SHA256: `{summary['checkpoint_sha256']}`.",
            f'- Detailed metrics and per-case PDF reports: `runs/{run.name}_validation/`.', '']
        for region in ('WT', 'TC', 'ET'):
            lower, upper = scores[region]['patient_bootstrap_95ci']
            details.append(f'- {region} patient-bootstrap 95% interval: [{lower:.4f}, {upper:.4f}].')
        details += ['']
    lines += ['', 'WT = whole tumor; TC = tumor core; ET = enhancing tumor.', '',
        'A zero region score is a model failure, not a missing or omitted result.',
        'Checkpoint selection used resized-grid validation Dice. The table above uses',
        'predictions reconstructed on the native MRI grid, with background masking.', '',
        '## Interpretation and limits', '',
        '- Four validation cases cannot establish clinical accuracy or population generalization.',
        '- These are exploratory single-seed runs; some settings were changed after observing validation failures.',
        '- Training runs may have different stopping points; this is not a fixed-runtime causal comparison.',
        '- Logged durations include validation and sometimes concurrent verification; they are not isolated hardware benchmarks.',
        '- Class weights use only training labels. Held-out test labels did not guide these adjustments.',
        '- HD95, IoU, sensitivity, empty-region status and voxel accuracy are in the per-case JSON reports.',
        '- Confidence intervals from four patients are unstable; zero-width intervals do not prove certainty.',
        '- Prospective use, external validation, uncertainty calibration and publication novelty remain unestablished.', '',
        '## Artifact details', ''] + details
    Path('docs/EXPERIMENT_RESULTS.md').write_text('\n'.join(lines) + '\n')
    print('Wrote docs/EXPERIMENT_RESULTS.md')
