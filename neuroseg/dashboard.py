"""Presentation helpers. All charts and statistics come from local artifacts."""
from html import escape
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st


def markup(value):
    st.markdown(value, unsafe_allow_html=True)


def page_header(title, subtitle):
    markup(f'<div class="page-kicker">NeuroSeg / Research workspace</div>'
           f'<div class="page-heading"><div><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>'
           '<div class="local-badge"><span class="status-dot"></span>Local processing</div></div>')


def section(number, title, tag=''):
    markup(f'<div class="section-title"><span class="step-number">{escape(number)}</span>'
           f'<strong>{escape(title)}</strong><span>{escape(tag)}</span></div>')


def case_card(title, detail):
    markup(f'<div class="case-strip"><strong>{escape(title)}</strong><span>{escape(detail)}</span></div>')


def legend():
    markup('<div class="legend"><span><i style="background:#4ed7b1"></i>Edema</span>'
           '<span><i style="background:#ffbd69"></i>Non-enhancing tumor</span>'
           '<span><i style="background:#ee6b97"></i>Enhancing tumor</span></div>')


@st.cache_data(max_entries=4, show_spinner=False)
def preview_slice(path, modified):
    """Only a real, unannotated MRI preview; no model predictions before a run."""
    image = nib.load(path)
    z = image.shape[2] // 2
    return np.asarray(image.dataobj[:, :, z, 0], dtype=np.float32).T, z


def show_preview(path):
    scan, z = preview_slice(str(path), Path(path).stat().st_mtime_ns)
    values = scan[scan != 0]
    low, high = np.percentile(values, [1, 99]) if values.size else (0, 1)
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor='#081823')
    ax.set_facecolor('#081823')
    ax.imshow(scan, cmap='gray', origin='lower', vmin=low, vmax=high)
    ax.axis('off')
    fig.subplots_adjust(left=.02, right=.98, top=.98, bottom=.02)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f'Original FLAIR · input-axis slice {z} · Preview only, no segmentation applied.')


def read_history(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A concurrently running trainer can be midway through its last write.
            continue
    return rows


def score_card(label, score, patients):
    score = float(score)
    markup(f'<div class="score-card"><div class="score-label">{escape(label)}</div>'
           f'<strong>{score:.3f}</strong><div class="score-track"><div class="score-fill" '
           f'style="width:{max(0, min(100, score * 100)):.2f}%"></div></div>'
           f'<small>Mean Dice · {int(patients)} validation cases</small></div>')
