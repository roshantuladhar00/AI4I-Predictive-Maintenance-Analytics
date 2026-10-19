from pathlib import Path

import pandas as pd
import pytest

from src.predict import ArtifactError, load_bundle

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root():
    return ROOT


@pytest.fixture(scope="session")
def bundle(project_root):
    try:
        return load_bundle(project_root)
    except ArtifactError as exc:
        pytest.fail(f"The app needs the final model artifacts from notebook 09: {exc}")


@pytest.fixture
def sample(project_root):
    return pd.read_csv(project_root / "sample_input.csv")
