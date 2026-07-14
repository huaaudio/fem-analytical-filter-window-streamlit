from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from secondment.fem_analytical_filter_window_app import UPLOAD_AUDIO_LABEL


APP_FILE = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def test_general_audience_defaults_and_primary_cta() -> None:
    app = AppTest.from_file(str(APP_FILE), default_timeout=60).run()

    assert not app.exception

    source = app.selectbox(key="audio_source")
    assert source.value.startswith("Aircraft cabin")
    assert source.options[-1] == UPLOAD_AUDIO_LABEL

    resonance = app.select_slider(key="resonance_hz")
    assert resonance.value == 420.0

    cta = app.button(key="run_comparison")
    assert cta.label == "Create listening comparison"
    assert cta.disabled is False

    visible_copy = " ".join(element.value for element in app.markdown)
    assert "Hear how a metamaterial panel changes sound" in visible_copy
    assert "Create your listening comparison" in visible_copy


def test_malformed_uploaded_wav_shows_friendly_error_without_app_exception() -> None:
    app = AppTest.from_file(str(APP_FILE), default_timeout=60).run()
    app.selectbox(key="audio_source").set_value(UPLOAD_AUDIO_LABEL).run()

    uploader = app.file_uploader(key="audio_upload")
    uploader.upload("broken.wav", b"not a valid WAV file", "audio/wav").run()
    assert app.button(key="run_comparison").disabled is False

    app.button(key="run_comparison").click().run()

    assert not app.exception
    assert len(app.error) == 1
    assert app.error[0].value == (
        "We could not create this comparison. If you uploaded a file, try a standard "
        "PCM or floating-point WAV and make sure it contains audible samples."
    )
