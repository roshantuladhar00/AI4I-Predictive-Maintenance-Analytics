import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.predict import (
    ArtifactError, InputValidationError, OUTPUT_COLUMNS, RAW_COLUMNS,
    load_bundle, parse_csv,
)


def test_predictions_match_frozen_final_evaluation(bundle, project_root):
    report = project_root / "reports/final_evaluation/final_test_predictions.csv"
    if not report.exists():
        pytest.skip("Run notebook 09 to create the prediction comparison report.")
    saved = pd.read_csv(report)
    observed = bundle.predict_frame(saved[RAW_COLUMNS + ["UDI"]])
    np.testing.assert_allclose(observed["failure_score"], saved["failure_score"], atol=1e-10, rtol=0)
    np.testing.assert_array_equal(observed["failure_alert"].astype(int), saved["predicted_failure"])
    np.testing.assert_array_equal(observed["UDI"], saved["UDI"])
    assert observed["failure_alert"].sum() == 149
    assert (observed["decision_threshold"] == bundle.threshold).all()


def test_target_flags_and_ids_do_not_change_scores(bundle, sample):
    base = bundle.predict_frame(sample)
    extra = sample.copy()
    extra["UDI"] = np.arange(len(extra)) + 100_000
    extra["Machine failure"] = [0, 1, 0, 1]
    for name in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        extra[name] = 1
    observed = bundle.predict_frame(extra)
    np.testing.assert_allclose(observed["failure_score"], base["failure_score"], atol=1e-12, rtol=0)
    assert "UDI" in observed and "Machine failure" in observed
    assert extra.columns.tolist() == [name for name in observed if name not in OUTPUT_COLUMNS]


def test_numeric_strings_and_type_normalization(bundle, sample):
    original = bundle.predict_frame(sample)
    changed = sample.astype(str)
    changed["Type"] = " " + changed["Type"].str.lower() + " "
    normalized = bundle.predict_frame(changed)
    np.testing.assert_allclose(normalized["failure_score"], original["failure_score"], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(normalized["Type"], sample["Type"])


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("Type", "X", "Type: use"),
        ("Air temperature [K]", np.nan, "finite numeric"),
        ("Process temperature [K]", np.inf, "finite numeric"),
        ("Torque [Nm]", "broken", "finite numeric"),
        ("Rotational speed [rpm]", 0, "greater than zero"),
        ("Air temperature [K]", 0, "greater than zero"),
        ("Torque [Nm]", -2, "cannot be negative"),
        ("Tool wear [min]", -1, "cannot be negative"),
        ("Torque [Nm]", True, "finite numeric"),
    ],
)
def test_invalid_rows_are_rejected(bundle, sample, column, value, message):
    changed = sample.astype(object)
    changed.loc[1, column] = value
    with pytest.raises(InputValidationError, match=message) as error:
        bundle.predict_frame(changed)
    assert "data rows 2" in str(error.value)


def test_missing_columns_and_empty_batches(bundle, sample):
    with pytest.raises(InputValidationError, match="Missing required columns"):
        bundle.predict_frame(sample.drop(columns="Type"))
    with pytest.raises(InputValidationError, match="at least one"):
        bundle.predict_frame(sample.iloc[:0])


def test_csv_duplicate_headers_and_existing_predictions(bundle, sample):
    with pytest.raises(InputValidationError, match="unique"):
        parse_csv(b"Type,Type\nL,H\n")
    with pytest.raises(InputValidationError, match="empty"):
        parse_csv(b"")
    with pytest.raises(InputValidationError, match="UTF-8"):
        parse_csv(b"\xff\xfe")
    previous = bundle.predict_frame(sample)
    with pytest.raises(InputValidationError, match="Remove existing prediction columns"):
        bundle.predict_frame(previous)


def test_outside_training_range_is_flagged_without_clipping(bundle, sample):
    changed = sample.iloc[:1].copy()
    changed["Torque [Nm]"] = bundle.reference["numeric_training_ranges"]["Torque [Nm]"]["maximum"] + 5
    observed = bundle.predict_frame(changed)
    assert observed["outside_training_range"].iloc[0]
    assert "Torque [Nm]" in observed["outside_training_features"].iloc[0]
    assert observed["Torque [Nm]"].iloc[0] == changed["Torque [Nm]"].iloc[0]


def test_loading_rejects_policy_mismatch_before_model_deserialization(project_root, tmp_path):
    root = tmp_path / "changed_policy"
    shutil.copytree(project_root / "models/final_evaluation", root / "models/final_evaluation")
    shutil.copytree(project_root / "config", root / "config")
    path = root / "models/final_evaluation/decision_policy.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    policy["threshold"] = 0.5
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ArtifactError, match="disagree on threshold"):
        load_bundle(root)


def test_terminal_prediction_and_input_protection(project_root, tmp_path):
    output = tmp_path / "predictions.csv"
    command = [
        sys.executable, "-m", "src.predict",
        "--input", str(project_root / "sample_input.csv"),
        "--output", str(output),
    ]
    completed = subprocess.run(command, cwd=project_root, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert len(pd.read_csv(output)) == 4
    assert "failure_alert" in pd.read_csv(output)
    repeated = subprocess.run(command, cwd=project_root, capture_output=True, text=True)
    assert repeated.returncode == 2
    assert "already exists" in repeated.stderr
    same_file = subprocess.run([
        sys.executable, "-m", "src.predict",
        "--input", str(project_root / "sample_input.csv"),
        "--output", str(project_root / "sample_input.csv"),
        "--overwrite",
    ], cwd=project_root, capture_output=True, text=True)
    assert same_file.returncode == 2
    assert "different from the input" in same_file.stderr
