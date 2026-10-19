import json
import math

import pytest
from streamlit.testing.v1 import AppTest

from src.presentation import ALERT_COLOR, NO_ALERT_COLOR, GaugeResult, make_alert_gauge

THRESHOLD = 0.06212594932220602


@pytest.mark.parametrize("score", [0.0, 0.0007658689539379, 0.2, 0.9766790999122416, 1.0])
def test_gauge_preserves_the_original_score_and_saved_decision(score):
    result = GaugeResult(score, THRESHOLD)
    figure = make_alert_gauge(score, THRESHOLD)
    indicator = figure.data[0]
    assert result.is_alert == (score >= THRESHOLD)
    assert indicator.value == pytest.approx(score * 100)
    assert indicator.gauge.threshold.value == pytest.approx(THRESHOLD * 100)
    assert list(indicator.gauge.axis.range) == [0, 100]
    assert indicator.gauge.bar.color == (ALERT_COLOR if result.is_alert else NO_ALERT_COLOR)
    assert list(indicator.gauge.steps[0].range) == [0, THRESHOLD * 100]
    assert list(indicator.gauge.steps[1].range) == [THRESHOLD * 100, 100]
    assert "%" not in indicator.number.suffix


def test_alert_boundary_uses_unrounded_original_scores():
    below = math.nextafter(THRESHOLD, 0)
    above = math.nextafter(THRESHOLD, 1)
    assert not GaugeResult(below, THRESHOLD).is_alert
    assert GaugeResult(THRESHOLD, THRESHOLD).is_alert
    assert GaugeResult(above, THRESHOLD).is_alert


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_scores_are_rejected(score):
    with pytest.raises(ValueError, match="failure score"):
        make_alert_gauge(score, THRESHOLD)


def test_gauge_and_presentation_examples_in_the_app(project_root):
    app = AppTest.from_file(str(project_root / "app.py"), default_timeout=30).run()
    assert not app.exception
    app.button(key="example_no_alert").click().run()
    assert not app.exception
    assert app.success[0].value == "No failure alert"
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert spec["data"][0]["value"] == pytest.approx(0.07658689539379, abs=1e-8)
    assert any("I compare the model's alert score of 0.08" in item.value for item in app.markdown)
    app.button(key="example_alert").click().run()
    assert not app.exception
    assert any(item.value == "Failure alert" for item in app.warning)
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert spec["data"][0]["value"] == pytest.approx(97.66790999122416, abs=1e-8)
    assert app.number_input(key="rotational_speed").value == 1263.0
    assert any("I compare the model's alert score of 97.67" in item.value for item in app.markdown)
