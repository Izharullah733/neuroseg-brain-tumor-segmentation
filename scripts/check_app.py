"""Real-checkpoint integration check. Generates one local demo report."""
from pathlib import Path

from streamlit.testing.v1 import AppTest


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / 'app.py')).run(timeout=30)
    assert not app.exception, [e.value for e in app.exception]
    assert app.button and not app.button[0].disabled, 'Prepare data and train a checkpoint first'
    app.button[0].click().run(timeout=90)
    assert not app.exception and not app.error, [e.value for e in app.exception] + [e.value for e in app.error]
    assert len(app.metric) == 3
    for plane in ('Axial', 'Coronal', 'Sagittal'):
        plane_widget = next(widget for widget in app.selectbox if widget.label == 'View plane')
        plane_widget.set_value(plane).run(timeout=30)
        assert not app.exception
    app.checkbox[0].set_value(True).run(timeout=30)
    assert not app.exception
    for page in ('Experiments', 'Run history', 'MRI analysis'):
        app.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not app.exception
    print('PASS: real MRI inference, three view planes, reference overlay, metrics and all pages.')
