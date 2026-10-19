# AI4I Predictive Maintenance Analytics

I built this project to explore machine-failure patterns, compare classification models, explain their predictions, and turn the selected model into a prediction app.

The app accepts a single observation or a CSV. It recreates the same engineered inputs used in my evaluation and applies the saved decision threshold.

## Put the app files in my project

I extract the contents of this package into the repository root: the folder that already contains **models**, **notebook**, **reports**, and **data**. I keep the **src**, **config**, and **.streamlit** subfolders together.

The app needs these existing artifacts from notebook 09:

    models/final_evaluation/predictor.joblib
    models/final_evaluation/input_schema.json
    models/final_evaluation/decision_policy.json
    models/final_evaluation/model_metadata.json

The package supplies:

    app.py
    src/__init__.py
    src/predict.py
    config/feature_reference.json
    sample_input.csv
    requirements.txt
    requirements-dev.txt
    pytest.ini
    tests/
    .streamlit/config.toml
    .gitignore
    README.md

I keep the evaluated model and its policy together. The app verifies their identity and compatibility before loading them. It loads my local model file; uploaded files contain observations in CSV format.

## Run in VS Code on Windows

I open the repository folder in VS Code, then open its terminal.

For a fresh environment, I run:

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -m streamlit run app.py

These commands use the environment's Python directly, so PowerShell activation is optional. If I already have the matching dependencies in my selected Python environment, I run:

    python -m streamlit run app.py

I open the local URL printed in the terminal. I stop the app with Ctrl+C.

The files were tested with Python 3.13.5, Streamlit 1.45.1, and scikit-learn 1.6.1. The scikit-learn version must match the version used to save my model. I use the provided requirements to reproduce the tested package versions.

## Prediction inputs

| CSV column | Unit or category | Accepted values |
|---|---|---|
| Air temperature [K] | Kelvin | Finite number greater than zero |
| Process temperature [K] | Kelvin | Finite number greater than zero |
| Rotational speed [rpm] | rpm | Finite number greater than zero |
| Torque [Nm] | Newton-metres | Finite number, zero or greater |
| Tool wear [min] | Minutes | Finite number, zero or greater |
| Type | Product category | L, M, or H |

I use the exact column names shown above. The CSV parser trims column-name spaces. Product-type values are stripped of surrounding spaces and converted to uppercase. Numeric strings are accepted; units are never converted automatically.

Each CSV can contain up to 20,000 observations and must be UTF-8 encoded and no larger than 10 MB. Empty inputs, duplicate headers, missing columns, non-finite measurements, and invalid categories are rejected with an explanation. A batch with invalid rows is rejected as a whole.

Additional columns, including IDs and known outcomes, are preserved for row matching. They are excluded from the model inputs. Before scoring a previously exported prediction file again, I remove its generated prediction columns.

I download **sample_input.csv** from the app to see a working input format. It contains four example observations from the validation partition and no outcome labels.

## Results

The app adds:

| Column | Meaning |
|---|---|
| failure_score | Uncalibrated model output between 0 and 1 |
| failure_alert | True when the score meets or exceeds the saved threshold |
| prediction_label | Failure alert, or no alert at the selected threshold |
| decision_threshold | Full numeric threshold used for the decision |
| outside_training_range | Whether any sensor measurement is outside its observed training range |
| outside_training_features | Which measurements are outside those ranges |

I read a score as an uncalibrated ensemble output. A score of 0.2 does not establish a calibrated 20% failure probability. A result below the threshold means no alert from this policy; failures can still be missed.

The reference ranges use the original 6,400 training observations. An input outside a range is flagged and still scored. It is not clipped to the training boundary.

Uploaded CSVs and prediction results are held in the running session. The app writes no uploaded files automatically; I choose when to download results.

## Why I use this threshold

I prioritize detecting failures subject to a 5% validation false-positive-rate cap. I searched all distinct validation score thresholds, selected the largest feasible number of detected failures, and broke ties using fewer false alarms followed by a higher threshold.

The selected threshold is **0.06212594932220602**. On validation it detects **50 of 54 failures**, misses **4**, and produces **63 false alarms** among **1,546 normal observations**. Detecting one additional failure requires at least **102 false alarms** in that ranking, which exceeds my cap.

I froze the policy before final-test scoring and kept the original training-only model. The app reads the full threshold from the saved JSON rather than rounding the displayed value or choosing a new cutoff.

## Final evaluation

| Measure | Final test result |
|---|---|
| Observations | 2,000 |
| Actual failures | 68 |
| Detected failures | 61 |
| Missed failures | 7 |
| False alarms | 88 |
| Failure recall | 89.71% |
| False-positive rate among normal observations | 4.55% |
| Precision among alerts | 40.94% |
| False alarms among alerts | 59.06% |
| Average precision | 0.8703 |
| ROC AUC | 0.9730 |

The test FPR's 95% Wilson interval is **3.71% to 5.58%**. The observed FPR is below 5%, and this does not guarantee a future rate below that value. The false-alarm share of alerts uses a different denominator from FPR.

My data are synthetic, the split is random, and the early exploratory notebooks included the final-test observations. I treat this as a project benchmark. Independent machine data, inspection capacity, and operating costs remain necessary for operational acceptance.

## Use the prediction code from the terminal

From the repository root:

    python -m src.predict --input sample_input.csv --output predictions.csv

I can pass a different project path with **--project-root**. To deliberately replace an existing result file, I add **--overwrite**. The command refuses to overwrite its input file.

From Python:

    import pandas as pd
    from src.predict import load_bundle

    predictor = load_bundle()
    observations = pd.read_csv("sample_input.csv")
    results = predictor.predict_frame(observations)
    print(results)

## Verify the app

I install the development requirements and run:

    python -m pip install -r requirements-dev.txt
    python -m pytest

The checks compare predictions with notebook 09, verify the feature preparation and input errors, confirm that outcome columns do not influence scores, and exercise the Streamlit form and CSV results. Prediction comparison tests use the existing report files from notebooks 07 and 09.

## Project workflow

1. Data understanding
2. Exploratory analysis
3. Data cleaning and the SMOTE experiment
4. Hypothesis testing
5. Failure statistical analysis
6. Baseline model training
7. Ensemble and MLP comparison
8. Model explainability
9. Threshold selection and final evaluation
10. Reusable inference and the local prediction app

## References

- [AI4I 2020 dataset](https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset)
- [Streamlit app setup](https://docs.streamlit.io/get-started/tutorials/create-an-app)
- [Streamlit forms](https://docs.streamlit.io/develop/api-reference/execution-flow/st.form)
- [Streamlit app testing](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
- [Threshold tuning](https://scikit-learn.org/1.6/auto_examples/model_selection/plot_cost_sensitive_learning.html)
