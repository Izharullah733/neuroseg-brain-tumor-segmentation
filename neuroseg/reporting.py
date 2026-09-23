"""Local reports and SQLite history; no network calls."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def write_report(result, output_dir, database='runs/history.sqlite'):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {**result, 'run_id': uuid.uuid4().hex, 'created_utc': datetime.now(timezone.utc).isoformat()}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    with (out / 'volumes.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['region', 'volume_ml'])
        writer.writerows(result['volumes_ml'].items())
    pdf = canvas.Canvas(str(out / 'report.pdf'), pagesize=A4)
    pdf.setTitle('NeuroSeg Research Report')
    pdf.setFont('Helvetica-Bold', 20)
    pdf.drawString(45, 790, 'NeuroSeg | Segmentation Report')
    pdf.setFont('Helvetica', 10)
    lines = ['RESEARCH PILOT - NOT CLINICALLY VALIDATED',
        f"Created: {result['created_utc']}", f"Run: {result['run_id']}",
        f"Training steps: {result['training_steps']}",
        f"Processing: {result['elapsed_seconds']:.2f} seconds",
        'Predicted volumes (regions overlap; do not sum):']
    lines += [f'  {region}: {value:.3f} mL' for region, value in result['volumes_ml'].items()]
    lines += ['WT: whole tumor; TC: tumor core; ET: enhancing tumor.',
        'Volumes use the original MRI affine determinant.', 'Checkpoint SHA256:', result['checkpoint_sha256'],
        'Input SHA256:', result['input_sha256']]
    if 'metrics' in result:
        lines += ['Reference-mask evaluation (original grid):']
        for region, metrics in result['metrics']['regions'].items():
            distance = metrics.get('hd95_mm')
            distance_text = 'unavailable' if distance is None else f'{distance:.2f} mm'
            lines += [f"  {region}: Dice {metrics['dice']:.4f}; IoU {metrics['iou']:.4f}; HD95 {distance_text}"]
    lines += ['HD95 is unavailable when exactly one mask is empty.',
        'Prepared, aligned, skull-stripped inputs are required.',
        'These outputs do not establish a diagnosis or treatment recommendation.']
    for i, line in enumerate(lines):
        pdf.drawString(45, 755 - 19 * i, line)
    pdf.save()
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created_utc TEXT, report_path TEXT, checkpoint_sha256 TEXT)')
        conn.execute('INSERT INTO runs VALUES (?, ?, ?, ?)', (result['run_id'], result['created_utc'], str((out / 'report.json').resolve()), result['checkpoint_sha256']))
    return result
