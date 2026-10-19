"""Load the frozen model and predict from original AI4I input columns."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
import sys
import threading
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_BATCH_ROWS = 20_000
MAX_CSV_BYTES = 10 * 1024 * 1024
RAW_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type",
]
NUMERIC_COLUMNS = RAW_COLUMNS[:-1]
ENGINEERED_COLUMNS = ["Temp_diff", "Torque_RPM_ratio"]
OUTPUT_COLUMNS = [
    "failure_score",
    "failure_alert",
    "prediction_label",
    "decision_threshold",
    "outside_training_range",
    "outside_training_features",
]
ASSET_NAMES = [
    "predictor.joblib",
    "input_schema.json",
    "decision_policy.json",
    "model_metadata.json",
]


class ArtifactError(RuntimeError):
    """A required saved-model artifact is missing or inconsistent."""


class InputValidationError(ValueError):
    """A prediction input does not satisfy the documented input contract."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"Cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{path.name} must contain a JSON object.")
    return value


def asset_signature(project_root: str | Path) -> tuple[tuple[str, int, int], ...]:
    """Return file identities for invalidating the app's resource cache."""
    root = Path(project_root)
    paths = [root / "models/final_evaluation" / name for name in ASSET_NAMES]
    paths.append(root / "config/feature_reference.json")
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise ArtifactError(
            "Missing project files: " + ", ".join(missing)
            + ". Place the app files in the project root and run notebook 09 first."
        )
    return tuple(
        (str(path.relative_to(root)), path.stat().st_mtime_ns, path.stat().st_size)
        for path in paths
    )


def parse_csv(payload: bytes) -> pd.DataFrame:
    """Parse a UTF-8 CSV and reject ambiguous or oversized input."""
    if len(payload) > MAX_CSV_BYTES:
        raise InputValidationError("The CSV must be 10 MB or smaller.")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputValidationError("Save the CSV with UTF-8 encoding before uploading it.") from exc
    if not text.strip():
        raise InputValidationError("The CSV is empty.")
    try:
        reader = csv.reader(io.StringIO(text))
        header = next(row for row in reader if row)
        header = [name.strip() for name in header]
    except (csv.Error, StopIteration) as exc:
        raise InputValidationError("The CSV needs a header row with column names.") from exc
    if any(not name for name in header):
        raise InputValidationError("Every CSV column needs a name.")
    if len(set(header)) != len(header):
        raise InputValidationError("CSV column names must be unique.")
    try:
        frame = pd.read_csv(io.StringIO(text), nrows=MAX_BATCH_ROWS + 1, on_bad_lines="error")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise InputValidationError(f"Cannot read this CSV: {exc}") from exc
    frame.columns = [str(name).strip() for name in frame.columns]
    if len(frame) > MAX_BATCH_ROWS:
        raise InputValidationError(f"A batch can contain at most {MAX_BATCH_ROWS:,} observations.")
    return frame


def _row_examples(mask: pd.Series | np.ndarray) -> str:
    positions = np.flatnonzero(np.asarray(mask)) + 1
    shown = ", ".join(str(value) for value in positions[:5])
    return shown + (", ..." if len(positions) > 5 else "")


def validate_raw_input(frame: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    """Validate and normalize the six original inputs without changing units."""
    if not isinstance(frame, pd.DataFrame):
        raise InputValidationError("Prediction input must be a pandas DataFrame.")
    if frame.empty:
        raise InputValidationError("Provide at least one observation.")
    if len(frame) > MAX_BATCH_ROWS:
        raise InputValidationError(f"A batch can contain at most {MAX_BATCH_ROWS:,} observations.")
    if not frame.columns.is_unique:
        raise InputValidationError("Input column names must be unique.")
    missing = [name for name in RAW_COLUMNS if name not in frame.columns]
    if missing:
        raise InputValidationError("Missing required columns: " + ", ".join(missing))
    collisions = [name for name in OUTPUT_COLUMNS if name in frame.columns]
    if collisions:
        raise InputValidationError(
            "Remove existing prediction columns before scoring again: " + ", ".join(collisions)
        )

    raw = frame.loc[:, RAW_COLUMNS].copy()
    errors = []
    for name in NUMERIC_COLUMNS:
        booleans = raw[name].map(lambda value: isinstance(value, (bool, np.bool_)))
        numeric = pd.to_numeric(raw[name], errors="coerce")
        if pd.api.types.is_complex_dtype(numeric.dtype):
            errors.append(f"{name}: complex numbers are not valid sensor measurements.")
            raw[name] = np.nan
            continue
        finite = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
        invalid = ~finite | booleans.to_numpy()
        if invalid.any():
            errors.append(f"{name}: enter finite numeric values (data rows {_row_examples(invalid)}).")
        elif name in ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]"]:
            invalid = numeric.le(0)
            if invalid.any():
                errors.append(f"{name}: values must be greater than zero (data rows {_row_examples(invalid)}).")
        else:
            invalid = numeric.lt(0)
            if invalid.any():
                errors.append(f"{name}: values cannot be negative (data rows {_row_examples(invalid)}).")
        raw[name] = numeric.astype(float)

    raw["Type"] = raw["Type"].astype("string").str.strip().str.upper()
    invalid_type = ~raw["Type"].isin(schema["valid_Type_values"])
    if invalid_type.any():
        errors.append(
            "Type: use " + ", ".join(schema["valid_Type_values"])
            + f" (data rows {_row_examples(invalid_type)})."
        )
    if errors:
        raise InputValidationError("\n".join(errors))
    raw["Type"] = raw["Type"].astype(str)
    return raw


def prepare_features(raw: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    """Recreate notebook 09's inputs from already validated raw measurements."""
    features = raw.loc[:, RAW_COLUMNS].copy()
    if schema["feature_set"] == "engineered":
        features["Temp_diff"] = features["Process temperature [K]"] - features["Air temperature [K]"]
        features["Torque_RPM_ratio"] = features["Torque [Nm]"] / (features["Rotational speed [rpm]"] + 1)
    return features.loc[:, schema["model_input_columns"]]


@dataclass
class PredictionBundle:
    estimator: Any
    schema: dict[str, Any]
    policy: dict[str, Any]
    metadata: dict[str, Any]
    reference: dict[str, Any]
    _prediction_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def threshold(self) -> float:
        return float(self.policy["threshold"])

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict every valid row, preserve extra columns, and add result columns."""
        raw = validate_raw_input(frame, self.schema)
        features = prepare_features(raw, self.schema)
        classes = np.asarray(self.estimator.classes_)
        positive_position = int(np.flatnonzero(classes == self.policy["positive_class"])[0])
        chunks = []
        with self._prediction_lock, threadpool_limits(limits=1):
            for start in range(0, len(features), 2048):
                chunks.append(
                    self.estimator.predict_proba(features.iloc[start:start + 2048])[:, positive_position]
                )
        scores = np.concatenate(chunks)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise RuntimeError("The saved model returned an invalid failure score.")
        alerts = scores >= self.threshold
        outside_features = [[] for _ in range(len(raw))]
        for name, bounds in self.reference["numeric_training_ranges"].items():
            outside = raw[name].lt(bounds["minimum"]) | raw[name].gt(bounds["maximum"])
            for position in np.flatnonzero(outside.to_numpy()):
                outside_features[position].append(name)

        output = frame.copy()
        for name in RAW_COLUMNS:
            output[name] = raw[name].to_numpy()
        output["failure_score"] = scores
        output["failure_alert"] = alerts
        output["prediction_label"] = np.where(alerts, "Failure alert", "No alert at selected threshold")
        output["decision_threshold"] = self.threshold
        output["outside_training_range"] = [bool(names) for names in outside_features]
        output["outside_training_features"] = [" | ".join(names) for names in outside_features]
        return output


def load_bundle(project_root: str | Path | None = None) -> PredictionBundle:
    """Load my trusted final artifact and verify its schema, policy, and identity."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    asset_signature(root)
    model_dir = root / "models/final_evaluation"
    schema = read_json(model_dir / "input_schema.json")
    policy = read_json(model_dir / "decision_policy.json")
    metadata = read_json(model_dir / "model_metadata.json")
    reference = read_json(root / "config/feature_reference.json")
    try:
        expected_model_columns = RAW_COLUMNS + ENGINEERED_COLUMNS if schema["feature_set"] == "engineered" else RAW_COLUMNS
        if schema["feature_set"] not in {"raw", "engineered"}:
            raise ArtifactError("This app supports the raw or engineered AI4I input schema.")
        if schema["raw_input_columns"] != RAW_COLUMNS or schema["model_input_columns"] != expected_model_columns:
            raise ArtifactError("The input schema differs from the six-input AI4I prediction code.")
        if set(schema["valid_Type_values"]) != {"L", "M", "H"}:
            raise ArtifactError("The saved Type categories differ from L, M, H.")
        if policy["comparison"] != ">=" or policy["positive_class"] != 1:
            raise ArtifactError("The saved policy differs from the evaluated failure decision rule.")
        threshold = float(policy["threshold"])
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ArtifactError("The saved threshold must be a finite score between 0 and 1.")
        for name in ["threshold", "candidate", "score_representation", "source_model_sha256", "protocol_sha256", "validation_FPR_cap"]:
            if metadata[name] != policy[name]:
                raise ArtifactError(f"The final model metadata and decision policy disagree on {name}.")
        if policy["threshold_selection_split"] != "validation":
            raise ArtifactError("The expected policy was selected using validation observations.")
        if not metadata["final_test_scored"] or not metadata["policy_frozen_before_test_scoring"]:
            raise ArtifactError("Complete notebook 09 before running the prediction app.")
        for name in ["data", "split"]:
            if reference["source_checksums"][name] != metadata["input_checksums"][name]:
                raise ArtifactError("The training reference ranges come from a different dataset or split.")
        if (
            reference["model_sha256"] != policy["source_model_sha256"]
            or reference["protocol_sha256"] != policy["protocol_sha256"]
            or reference["threshold_selection"]["threshold"] != threshold
        ):
            raise ArtifactError("The app reference information differs from the frozen model or policy.")
        if set(reference["numeric_training_ranges"]) != set(NUMERIC_COLUMNS):
            raise ArtifactError("The training reference must cover all five sensor inputs.")
        expected_sklearn = metadata["versions"]["scikit_learn"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"A saved artifact is missing or has an invalid field: {exc}") from exc
    if sklearn.__version__ != expected_sklearn:
        raise ArtifactError(
            f"The model was saved with scikit-learn {expected_sklearn}; "
            f"this environment has {sklearn.__version__}. Install the provided requirements."
        )
    model_path = model_dir / "predictor.joblib"
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != policy["source_model_sha256"]:
        raise ArtifactError("The predictor file hash differs from the evaluated model.")
    try:
        estimator = joblib.load(model_path)
    except Exception as exc:
        raise ArtifactError(f"Cannot load the saved model in this environment: {exc}") from exc
    if not hasattr(estimator, "predict_proba") or not np.array_equal(estimator.classes_, [0, 1]):
        raise ArtifactError("The saved model must provide scores for the evaluated classes 0 and 1.")
    if list(estimator.feature_names_in_) != schema["model_input_columns"]:
        raise ArtifactError("The saved model's input columns differ from the schema.")
    return PredictionBundle(estimator, schema, policy, metadata, reference)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict AI4I failure alerts with the frozen model and threshold.")
    parser.add_argument("--input", required=True, type=Path, help="UTF-8 CSV with the six original input columns.")
    parser.add_argument("--output", required=True, type=Path, help="Prediction CSV to create.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Project containing models/final_evaluation and config.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output CSV.")
    arguments = parser.parse_args(argv)
    try:
        if arguments.input.resolve() == arguments.output.resolve():
            raise InputValidationError("Choose an output file different from the input CSV.")
        if arguments.output.exists() and not arguments.overwrite:
            raise InputValidationError("The output already exists. Choose another name or pass --overwrite.")
        if arguments.input.stat().st_size > MAX_CSV_BYTES:
            raise InputValidationError("The CSV must be 10 MB or smaller.")
        frame = parse_csv(arguments.input.read_bytes())
        bundle = load_bundle(arguments.project_root)
        result = bundle.predict_frame(frame)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(arguments.output, index=False)
    except (ArtifactError, InputValidationError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        f"Saved {len(result):,} observations; {int(result['failure_alert'].sum()):,} alerts; "
        f"threshold {bundle.threshold:.15f} -> {arguments.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
