"""Run with: .venv/Scripts/python -m streamlit run app.py"""
from pathlib import Path
import json
import sqlite3
import tempfile
import uuid
import altair as alt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
import pandas as pd
import streamlit as st

from neuroseg.data import MODALITIES
from neuroseg.inference import predict
from neuroseg.reporting import write_report
from neuroseg.dashboard import markup, page_header, section, case_card, legend, show_preview, read_history, score_card

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='NeuroSeg | MRI Research Studio', page_icon='◈', layout='wide')
markup('<style>' + (ROOT / 'assets/dashboard.css').read_text() + '</style>')

with st.sidebar:
    markup('<div class="brand"><div class="brand-symbol">N</div><div><div class="brand-name">NeuroSeg</div>'
           '<div class="brand-sub">MRI RESEARCH STUDIO</div></div></div><div class="nav-label">WORKSPACE</div>')
    page = st.radio('View', ['MRI analysis', 'Experiments', 'Run history'], label_visibility='collapsed')
    markup('<div class="sidebar-note"><span class="status-dot"></span><strong>Private by design</strong><br>'
           'MRI processing and reports stay on this computer.<br><br>'
           '<strong>Research edition</strong><br>Experimental segmentation for prepared, aligned MRI scans. '
           'Not clinically validated.</div>')

titles = {'MRI analysis': ('MRI analysis', 'From multi-modal scans to a clearer view of tumor regions.'),
          'Experiments': ('Experiment studio', 'Compare recorded runs, inspect learning curves, and understand the evidence.'),
          'Run history': ('Analysis library', 'Your previous analyses and downloadable reports, saved locally.')}
page_header(*titles[page])

if page == 'Experiments':
    histories = sorted((ROOT / 'runs').glob('*/history.jsonl')) if (ROOT / 'runs').exists() else []
    if not histories:
        st.info('No completed training epochs yet. Training results will appear here.')
        st.stop()
    history = st.selectbox('Experiment', histories, format_func=lambda p:p.parent.name.replace('_', ' ').title())
    rows = read_history(history)
    if not rows:
        st.info('Waiting for the first completed epoch.')
        st.stop()
    frame = pd.DataFrame(rows).set_index('epoch')
    summary_path = ROOT / f'runs/{history.parent.name}_validation/summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        section('01', 'Original-resolution validation', 'RECORDED RESULTS')
        for col, (key, label) in zip(st.columns(3), [('WT', 'Whole tumor'), ('TC', 'Tumor core'), ('ET', 'Enhancing tumor')]):
            with col:
                score_card(label, summary['region_dice'][key]['mean'], summary['patients'])
        markup('<div class="research-note">Small-cohort validation results, not clinical accuracy. '
               'The held-out test split is separate. Inspect per-case failures alongside these averages.</div>')
    section('02', 'Learning curves', 'TRAINING LOGS')
    loss_col, dice_col = st.columns(2, gap='medium')
    with loss_col, st.container(border=True, key='loss_card'):
        st.markdown('**Training loss**')
        st.caption('Cross-entropy + foreground Dice · lower is better')
        st.line_chart(frame[['loss']].rename(columns={'loss':'Training loss'}), color='#087f8c', height=260)
    with dice_col, st.container(border=True, key='dice_card'):
        st.markdown('**Validation Dice**')
        st.caption('Resized-grid mean · used for checkpoint selection')
        chart = alt.Chart(frame.reset_index()).mark_line(color='#6475c7').encode(
            x=alt.X('epoch:Q', title='Epoch'),
            y=alt.Y('validation_resized_macro_dice:Q', title='Validation Dice', scale=alt.Scale(domain=[0, 1])),
            tooltip=[alt.Tooltip('epoch:Q', title='Epoch'), alt.Tooltip('validation_resized_macro_dice:Q', title='Dice', format='.3f')]
        ).properties(height=260)
        st.altair_chart(chart, width='stretch')
    for col, label, value in zip(st.columns(3), ['Completed epochs', 'Optimizer steps', 'Logged training + validation'],
                                [str(rows[-1]['epoch']), f"{rows[-1]['steps']:,}", f"{sum(r['elapsed_seconds'] for r in rows)/60:.1f} min"]):
        col.metric(label, value)
    with st.expander('Inspect epoch-by-epoch records'):
        st.dataframe(frame.drop(columns=['validation_patients']), width='stretch')
    st.caption('Each epoch contains sampled optimizer steps. Timings include validation and may include other system activity.')
    st.stop()

if page == 'Run history':
    database = ROOT / 'runs/history.sqlite'
    if database.exists():
        with sqlite3.connect(database) as conn:
            frame = pd.read_sql_query('SELECT id, created_utc, report_path FROM runs ORDER BY created_utc DESC', conn)
        section('01', 'Saved analyses', f'{len(frame)} RECORDS')
        if frame.empty:
            st.info('Run an MRI analysis to create your first report.')
            st.stop()
        display = frame.copy()
        display['Analysis'] = display['id'].str[:10].str.upper()
        display['Created (UTC)'] = pd.to_datetime(display['created_utc'], utc=True).dt.strftime('%d %b %Y · %H:%M')
        display['Status'] = ['Report saved' if Path(p).exists() else 'Report unavailable' for p in display['report_path']]
        st.dataframe(display[['Analysis', 'Created (UTC)', 'Status']], width='stretch', hide_index=True)
        with st.container(border=True, key='history_card'):
            selected = st.selectbox('Open a saved analysis', list(frame.index), format_func=lambda i:f"{display.loc[i, 'Analysis']} · {display.loc[i, 'Created (UTC)']}")
            saved_path = Path(frame.loc[selected, 'report_path'])
            if saved_path.exists():
                saved = json.loads(saved_path.read_text())
                case_card(saved.get('case_id', 'MRI segmentation report'), f"Processed in {saved.get('elapsed_seconds', 0):.1f}s · Research pilot")
                for col, name, label, mime in zip(st.columns(3), ['report.pdf', 'volumes.csv', 'segmentation.nii.gz'], ['PDF report', 'Volume CSV', 'NIfTI mask'], ['application/pdf', 'text/csv', 'application/gzip']):
                    path = saved_path.parent / name
                    if path.exists():
                        col.download_button(label, path.read_bytes(), file_name=name, mime=mime, width='stretch')
            else:
                st.info('This report has been moved or removed from local storage.')
    else:
        st.info('Completed analyses will appear here.')
    st.stop()

checkpoints = sorted((ROOT / 'runs').glob('*/best.pt')) if (ROOT / 'runs').exists() else []
manifest_path = ROOT / 'data/processed/manifest.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
case_count = len(manifest['cases']) if manifest else 0
markup('<div class="hero"><div class="hero-copy"><div class="hero-kicker">MULTI-MODAL BRAIN MRI</div>'
       '<h2>Explore the scan. Understand the regions.</h2>'
       '<p>Inspect MRI slices, compare segmentation overlays, and bring your findings together in a downloadable report.</p></div>'
       f'<div class="hero-facts"><div class="hero-fact"><strong>{case_count}</strong><span>Local cases</span></div>'
       f'<div class="hero-fact"><strong>{len(checkpoints)}</strong><span>Trained models</span></div>'
       '<div class="hero-fact"><strong>4</strong><span>MRI modalities</span></div></div></div>')
markup('<div class="research-note"><strong>Research pilot.</strong> Predictions are experimental and are not clinically validated. '
       'Use the reference comparison to inspect model limitations.</div>')
if not checkpoints:
    st.info('A trained checkpoint is required. The interface will enable analysis after pilot training finishes.')
    st.stop()

left, right = st.columns([1, 2.25], gap='medium')
with left, st.container(border=True, key='setup_card'):
    section('01', 'Prepare an analysis')
    checkpoint = st.selectbox('Trained model', checkpoints, format_func=lambda p:p.parent.name.replace('_', ' ').title())
    source = st.radio('MRI source', ['Local validation example', 'Upload MRI'])
    input_path, uploads, reference_path = None, None, None
    if source == 'Local validation example':
        if manifest:
            case = st.selectbox('Validation case', manifest['splits']['validation'])
            input_path = manifest['cases'][case]['image']
            reference_path = manifest['cases'][case]['label']
            input_key = f'{checkpoint}:{case}'
            case_size = Path(input_path).stat().st_size / 1024**2
            case_card(case, f'{case_size:.1f} MB · 4 MRI sequences · Reference available')
        else:
            st.info('Prepare the dataset to enable local examples.')
            input_key = 'unavailable'
    else:
        mode = st.radio('File format', ['Single 4D NIfTI', 'Four aligned 3D NIfTIs'])
        st.caption('NIfTI volumes (.nii / .nii.gz). JPG/PNG pictures are not supported by this model.')
        markup('<div class="mini-note"><strong>256 MB is the maximum per file, not a required size.</strong> '
               'Our local examples are approximately 7–12 MB.</div>')
        if mode == 'Single 4D NIfTI':
            st.caption('Required channel order: FLAIR, T1w, T1 contrast-enhanced, T2w.')
            uploaded = st.file_uploader('4D volume', type=['nii', 'gz'])
            uploads = [uploaded] if uploaded else None
        else:
            uploads = []
            upload_columns = st.columns(2)
            for i, name in enumerate(MODALITIES):
                with upload_columns[i % 2]:
                    uploads.append(st.file_uploader(name, type=['nii', 'gz'], key=name))
            if not all(uploads):
                uploads = None
        input_key = f'{checkpoint}:{mode}:' + ':'.join(f'{u.file_id}' for u in (uploads or []))
    if st.session_state.get('input_key') != input_key:
        st.session_state.pop('analysis', None)
        st.session_state['input_key'] = input_key
        for axis in range(3):
            st.session_state.pop(f'slice_{axis}', None)
    go = st.button('Run segmentation', type='primary', disabled=not (input_path or uploads), width='stretch')
    st.caption('Processing runs locally. Your scans are not uploaded to an external service.')
    with st.expander('What files can I use?'):
        st.markdown('**One 4D file:** all four sequences in a single NIfTI.\n\n'
                    '**Four 3D files:** FLAIR, T1, contrast-enhanced T1 and T2 from the same patient, on the same grid.\n\n'
                    'Inputs must already be aligned, skull-stripped and use millimeter spatial units.')

if go:
    try:
        with st.spinner('Validating MRI and predicting each slice…'):
            output = ROOT / 'runs/predictions' / uuid.uuid4().hex
            with tempfile.TemporaryDirectory(prefix='neuroseg-', dir=ROOT / 'runs') as temp:
                if uploads:
                    paths = []
                    for i, uploaded in enumerate(uploads):
                        suffix = '.nii.gz' if uploaded.name.lower().endswith('.nii.gz') else '.nii'
                        path = Path(temp) / f'modality_{i}{suffix}'
                        path.write_bytes(uploaded.getbuffer())
                        paths.append(path)
                    if len(paths) == 4:
                        images = [nib.load(p) for p in paths]
                        reference = images[0]
                        for image in images:
                            if len(image.shape) != 3 or image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-4):
                                raise ValueError('Four modalities must be 3D and share the same shape and affine.')
                            if image.header.get_xyzt_units()[0] != 'mm':
                                raise ValueError('All modalities must declare millimeter spatial units.')
                        combined = np.stack([image.get_fdata(dtype=np.float32) for image in images], axis=-1)
                        input_path = Path(temp) / 'combined.nii.gz'
                        nib.save(nib.Nifti1Image(combined, reference.affine, reference.header), input_path)
                    else:
                        input_path = paths[0]
                image, volume, mask, result = predict(input_path, checkpoint, output)
                result = write_report(result, output, ROOT / 'runs/history.sqlite')
                orientation = nib.orientations.io_orientation(image.affine)
                volume = nib.orientations.apply_orientation(volume, orientation)
                mask = nib.orientations.apply_orientation(mask, orientation)
                reference = None
                if reference_path:
                    reference = nib.orientations.apply_orientation(nib.load(reference_path).get_fdata().astype(np.uint8), orientation)
                st.session_state['analysis'] = {'volume': volume, 'mask': mask, 'reference': reference, 'report': result, 'output': str(output)}
    except Exception as error:
        st.error(f'Analysis could not complete: {error}')

with right, st.container(border=True, key='viewer_card'):
    section('02', 'Imaging workspace', 'MRI VIEWER')
    analysis = st.session_state.get('analysis')
    if analysis is None:
        markup('<div class="viewer-bar"><strong>Scan preview</strong><span class="viewer-badge neutral">AWAITING ANALYSIS</span></div>')
        if input_path:
            try:
                show_preview(input_path)
            except (OSError, ValueError) as error:
                st.info(f'Preview unavailable: {error}')
            st.caption('Ready when you are. Select Run segmentation to generate tumor overlays and volume measurements.')
        else:
            markup('<div class="empty-viewer"><div class="scan-frame">+</div><strong>Your imaging workspace</strong>'
                   '<p>Choose a local example or add your prepared MRI volumes. Segmentation results will appear here after analysis.</p></div>')
        legend()
    else:
        volume, mask, report = analysis['volume'], analysis['mask'], analysis['report']
        markup(f'<div class="viewer-bar"><strong>Segmentation result</strong><span class="viewer-badge">'
               f'COMPLETED · {report["elapsed_seconds"]:.1f}s</span></div>')
        control_columns = st.columns(2)
        modality = control_columns[0].selectbox('Displayed modality', range(4), format_func=lambda i:MODALITIES[i])
        plane = control_columns[1].selectbox('View plane', ['Axial', 'Coronal', 'Sagittal'])
        axis = {'Axial': 2, 'Coronal': 1, 'Sagittal': 0}[plane]
        slice_col, opacity_col = st.columns([2, 1])
        z = slice_col.slider('Slice index', 0, volume.shape[axis] - 1, volume.shape[axis] // 2, key=f'slice_{axis}')
        opacity = opacity_col.slider('Overlay opacity', 0.0, 1.0, .45)
        show_reference = analysis.get('reference') is not None and st.checkbox('Compare with validation reference', value=False)
        panels = 3 if show_reference else 2
        fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 5), facecolor='#081823')
        scan = np.take(volume[..., modality], z, axis=axis).T
        positive = scan[scan != 0]
        low, high = np.percentile(positive, [1, 99]) if positive.size else (0, 1)
        colors = ListedColormap(['#000000', '#4ed7b1', '#ffbd69', '#ee6b97'])
        for ax, title in zip(axes, ['Original MRI', 'Predicted regions', 'Reference regions']):
            ax.set_facecolor('#081823')
            ax.imshow(scan, cmap='gray', origin='lower', vmin=low, vmax=high)
            ax.set_title(title, color='#c9dce8', fontsize=11, pad=14)
            ax.axis('off')
        overlay = np.ma.masked_equal(np.take(mask, z, axis=axis).T, 0)
        axes[1].imshow(overlay, cmap=colors, vmin=0, vmax=3, alpha=opacity, origin='lower', interpolation='nearest')
        if show_reference:
            reference_slice = np.ma.masked_equal(np.take(analysis['reference'], z, axis=axis).T, 0)
            axes[2].imshow(reference_slice, cmap=colors, vmin=0, vmax=3, alpha=opacity, origin='lower', interpolation='nearest')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        legend()
        st.caption('Display follows closest RAS orientation. Exports preserve the original MRI grid; oblique scans are not resliced.')
        st.divider()
        section('03', 'Region measurements', 'PREDICTED VOLUME')
        cols = st.columns(3)
        names = {'WT': 'Whole tumor', 'TC': 'Tumor core', 'ET': 'Enhancing tumor'}
        for col, (region, volume_ml) in zip(cols, report['volumes_ml'].items()):
            col.metric(names[region], f'{volume_ml:.2f} mL')
        st.caption('These regions overlap; their volumes must not be added together. Zero predicted volume does not establish absence.')
        st.markdown('**Export this analysis**')
        out = Path(analysis['output'])
        for col, filename, label, mime in zip(st.columns(3), ['report.pdf', 'volumes.csv', 'segmentation.nii.gz'], ['PDF report', 'Volume CSV', 'NIfTI mask'], ['application/pdf', 'text/csv', 'application/gzip']):
            col.download_button(label, (out / filename).read_bytes(), file_name=filename, mime=mime, width='stretch')
        with st.expander('Analysis details and provenance'):
            st.caption(f"Model: {Path(report['checkpoint']).parent.name} · Training steps: {report['training_steps']:,}")
            st.code(report['checkpoint_sha256'], language=None)
            st.caption('Checkpoint SHA256 · Identifies the exact model used for this analysis.')

markup('<div class="footnote"><span>NEUROSEG · RESEARCH EDITION</span>'
       '<span>Local inference · Original-grid export · Reproducible reports</span></div>')
