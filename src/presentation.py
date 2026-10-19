"""Presentation helpers for the existing frozen failure-alert decision."""

from dataclasses import dataclass
import math

import plotly.graph_objects as go

NO_ALERT_COLOR = "#138A64"
ALERT_COLOR = "#C54446"
NO_ALERT_BAND = "#D8F0E5"
ALERT_BAND = "#F8DDDE"


@dataclass(frozen=True)
class GaugeResult:
    score: float
    threshold: float

    def __post_init__(self):
        if not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("The failure score must be finite and between 0 and 1.")
        if not math.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("The gauge needs a saved threshold strictly between 0 and 1.")

    @property
    def score_points(self) -> float:
        return self.score * 100

    @property
    def threshold_points(self) -> float:
        return self.threshold * 100

    @property
    def is_alert(self) -> bool:
        # The decision uses original scores, including the exact >= boundary.
        return self.score >= self.threshold

    @property
    def number_format(self) -> str:
        # Show more precision when the rounded score and line would look identical.
        return ".4f" if round(self.score_points, 2) == round(self.threshold_points, 2) else ".2f"

    @property
    def score_text(self) -> str:
        return format(self.score_points, self.number_format)

    @property
    def threshold_text(self) -> str:
        return format(self.threshold_points, self.number_format)

    @property
    def status(self) -> str:
        return "Failure alert" if self.is_alert else "No failure alert"

    @property
    def color(self) -> str:
        return ALERT_COLOR if self.is_alert else NO_ALERT_COLOR

    @property
    def summary(self) -> str:
        comparison = "at or above" if self.is_alert else "below"
        return (
            f"The alert score is {self.score_text} out of 100. "
            f"It is {comparison} the alert line of {self.threshold_text}."
        )

    @property
    def presentation_line(self) -> str:
        outcome = (
            "so the model raises a failure alert for this observation."
            if self.is_alert else
            "so the model raises no failure alert for this observation."
        )
        comparison = "at or above" if self.is_alert else "below"
        return (
            f"I compare the model's alert score of {self.score_text} with the alert line of "
            f"{self.threshold_text}. This observation is {comparison} the line, {outcome}"
        )


def make_alert_gauge(score: float, threshold: float) -> go.Figure:
    """Display score * 100 with the same exact decision threshold and two zones."""
    result = GaugeResult(float(score), float(threshold))
    figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result.score_points,
        number={
            "valueformat": result.number_format,
            "suffix": " / 100",
            "font": {"size": 45, "color": result.color},
        },
        title={
            "text": (
                "Failure alert score<br>"
                f"<span style='font-size:13px;color:#617588'>"
                f"Alert line: {result.threshold_text} / 100</span>"
            ),
            "font": {"size": 20, "color": "#152C3D"},
        },
        domain={"x": [0.08, 0.92], "y": [0.19, 0.90]},
        gauge={
            "shape": "angular",
            "axis": {
                "range": [0, 100],
                "tickmode": "array",
                "tickvals": [0, 25, 50, 75, 100],
                "ticktext": ["0", "25", "50", "75", "100"],
                "tickwidth": 1,
                "tickcolor": "#78909C",
                "tickfont": {"size": 12, "color": "#526778"},
            },
            "bar": {"color": result.color, "thickness": 0.33},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, result.threshold_points], "color": NO_ALERT_BAND},
                {"range": [result.threshold_points, 100], "color": ALERT_BAND},
            ],
            "threshold": {
                "line": {"color": "#152C3D", "width": 4},
                "thickness": 1,
                "value": result.threshold_points,
            },
        },
    ))
    figure.add_annotation(
        x=0.12, y=0.04, xref="paper", yref="paper", showarrow=False,
        text=f"<span style='color:{NO_ALERT_COLOR}'>●</span> No-alert zone",
        font={"size": 13, "color": "#526778"}, xanchor="left",
    )
    figure.add_annotation(
        x=0.88, y=0.04, xref="paper", yref="paper", showarrow=False,
        text=f"<span style='color:{ALERT_COLOR}'>●</span> Alert zone",
        font={"size": 13, "color": "#526778"}, xanchor="right",
    )
    figure.update_layout(
        height=355,
        margin={"l": 15, "r": 15, "t": 60, "b": 5},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, sans-serif"},
    )
    return figure
