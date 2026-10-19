# AI4I Predictive Maintenance Analytics

An end-to-end data analytics and machine learning project that explores equipment failure risk using the AI4I 2020 predictive maintenance dataset.

The project connects sensor data analysis, statistical testing, failure classification, and model explainability in an interactive Streamlit dashboard. It examines how operating conditions relate to machine failure and makes the trade-off between detecting failures and raising false alarms visible.

## Project objective

Identify patterns associated with machine failure, compare predictive models, and explain their predictions to support maintenance analysis. The workflow prioritizes failure detection while measuring the false alarms that accompany each decision threshold.

## Key features

- **Data understanding and preparation:** inspect sensor measurements, validate data quality, and explore failure patterns and class imbalance.
- **Statistical analysis:** examine relationships between operating conditions and failure outcomes through hypothesis testing and descriptive statistics.
- **Model comparison:** evaluate classical classifiers, neural networks, and ensemble methods, including class weighting and a SMOTE experiment.
- **Explainable predictions:** use SHAP to examine global feature importance and the drivers of individual model predictions.
- **Interactive maintenance dashboard:** inspect failure scores on a gauge, simulate sensor changes and maintenance cycles, and explore what-if scenarios.
- **RUL and survival exploration:** demonstrate synthetic remaining-useful-life regression and survival analysis using a tool-wear duration proxy.

## Dashboard

| Page | Purpose |
|---|---|
| Simulation Dashboard | Monitor simulated machine conditions, failure scores, SHAP drivers, risk trends, and maintenance states. Pause and resume using a saved fleet snapshot. |
| WHAT-IF Analysis | Change operating measurements and inspect how the selected machine's prediction responds. |
| Survival Analysis | Explore Kaplan–Meier cohort curves and Cox individual survival estimates under the tool-wear proxy assumption. |

## Analysis workflow

1. Data understanding and exploratory analysis.
2. Data cleaning and class-imbalance experiments.
3. Hypothesis testing and failure statistics.
4. Baseline, ensemble, and deep-learning model comparison.
5. SHAP explainability.
6. Synthetic RUL regression and exploratory survival analysis.
7. Interactive dashboard and scenario analysis.

The notebooks are in `notebook/`. See [START_HERE.md](START_HERE.md) for their execution order and VS Code setup.

## Technology

Python, pandas, NumPy, SciPy, statsmodels, scikit-learn, XGBoost, PyTorch TabNet, TensorFlow/Keras, SHAP, lifelines, Matplotlib, Seaborn, Plotly, and Streamlit.

## Run locally

Use **Python 3.13**. From the project folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

A fresh Git clone does not contain the dashboard model binaries. Training generates the classifier, RUL regressor, evaluation metrics, and decision threshold. If these artifacts are already present, the training step is optional. Restart Streamlit after retraining.

If `py -3.13` is unavailable, use the full path to an installed Python 3.13 interpreter; see [START_HERE.md](START_HERE.md).

## Model results and interpretation

The saved classifier combines random forest, gradient boosting, and MLP predictions using soft-voting weights of **2:2:1**. The random forest member uses isotonic calibration.

At the saved decision threshold of **0.025743180189271207**, the threshold-selection split produces:

| Measure | Result |
|---|---:|
| Failure recall | 95.59% — 65 of 68 failures detected |
| Failure precision | 28.51% |
| False-positive rate | 8.44% — 163 false alarms among 1,932 non-failures |
| Average precision | 0.8807 |
| ROC-AUC | 0.9743 |

These observations were also used to select the threshold, so the results are not an untouched final-test estimate. The decision rule targets at least 95% failure recall; it does not impose a 5% false-positive cap. Metrics should be checked against `models/eval.json` after retraining.

The RUL target is constructed rather than measured, and tool wear serves as a duration proxy in the survival analysis. Interpret those outputs as exploratory demonstrations. [PROJECT_NOTES.md](PROJECT_NOTES.md) explains the model settings, gauge policy, and analytical limitations.
