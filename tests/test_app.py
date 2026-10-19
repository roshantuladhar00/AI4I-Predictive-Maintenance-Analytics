import io

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest


def submit_observation(app):
    return next(button for button in app.button if button.label == "Predict observation").click().run()


def test_single_observation_form(project_root):
    app = AppTest.from_file(str(project_root / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Maintenance prediction dashboard"
    app = submit_observation(app)
    assert not app.exception
    assert app.success[0].value == "No failure alert"
    app.number_input(key="air_temperature").set_value(302.7)
    app.number_input(key="process_temperature").set_value(310.5)
    app.number_input(key="rotational_speed").set_value(1263.0)
    app.number_input(key="torque").set_value(67.8)
    app.number_input(key="tool_wear").set_value(197.0)
    app = submit_observation(app)
    assert not app.exception
    assert any("Failure alert" in item.value for item in app.warning)
    assert any("0.976679" == item.value for item in app.metric)


def test_single_observation_invalid_value(project_root):
    app = AppTest.from_file(str(project_root / "app.py"), default_timeout=30).run()
    app.number_input(key="rotational_speed").set_value(0.0)
    app = submit_observation(app)
    assert not app.exception
    assert any("Rotational speed [rpm]: values must be greater than zero" in item.value for item in app.error)
    assert not app.success


class UploadedCSV(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.size = len(payload)
        self.name = "sample_input.csv"


def test_batch_results_and_download(project_root, monkeypatch):
    # Streamlit 1.45 AppTest cannot set the uploader directly.
    # I substitute its uploaded-file return value and exercise the real parser and buttons.
    uploaded = UploadedCSV((project_root / "sample_input.csv").read_bytes())
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    app = AppTest.from_file(str(project_root / "app.py"), default_timeout=30).run()
    assert not app.exception
    app.button(key="predict_csv").click().run()
    assert not app.exception
    result = app.session_state["batch_prediction"]
    assert isinstance(result, pd.DataFrame) and len(result) == 4
    assert result["failure_alert"].sum() >= 1
    assert any(item.proto.label == "Download all predictions" for item in app.get("download_button"))


def test_invalid_csv_batch_does_not_return_partial_predictions(project_root, monkeypatch):
    changed = pd.read_csv(project_root / "sample_input.csv").astype(object)
    changed.loc[1, "Torque [Nm]"] = "invalid"
    uploaded = UploadedCSV(changed.to_csv(index=False).encode("utf-8"))
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    app = AppTest.from_file(str(project_root / "app.py"), default_timeout=30).run()
    app.button(key="predict_csv").click().run()
    assert not app.exception
    assert any("Torque [Nm]" in item.value and "data rows 2" in item.value for item in app.error)
    assert "batch_prediction" not in app.session_state
