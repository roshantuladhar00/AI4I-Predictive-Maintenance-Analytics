# AI4I Predictive Maintenance Analytics

An end-to-end data analytics and machine learning project that explores equipment failure risk using the AI4I 2020 predictive maintenance dataset.

The project connects sensor data exploration, statistical analysis, failure classification, and model explainability in an interactive Streamlit dashboard. It examines how operating conditions relate to machine failure and makes the trade-off between detecting failures and raising false alarms visible.

## Project objective

Identify patterns associated with machine failure, compare predictive models, and explain their predictions to support maintenance analysis. The workflow prioritizes failure detection while measuring the false alarms associated with each decision threshold.

## Key features

- **Data exploration and preparation:** inspect sensor measurements, check data quality, and examine failure patterns and class imbalance.
- **Statistical analysis:** investigate relationships between operating conditions and failure outcomes through hypothesis testing and descriptive statistics.
- **Model comparison:** evaluate classical classifiers, neural networks, and ensemble methods, with class weighting and a SMOTE experiment.
- **Explainable predictions:** use SHAP to explore global feature importance and individual prediction drivers.
- **Interactive dashboard:** display failure scores on a gauge, simulate changing machine conditions and maintenance cycles, and explore what-if scenarios.
- **RUL and survival exploration:** demonstrate synthetic remaining-useful-life regression and survival analysis using a tool-wear duration proxy.

## Dataset

The AI4I 2020 dataset contains **10,000 observations** with a binary machine-failure target and five sensor measurements:

| Feature | Unit |
|---|---|
| Air temperature | Kelvin |
| Process temperature | Kelvin |
| Rotational speed | RPM |
| Torque | Newton-metres |
| Tool wear | Minutes |

Product type is an additional categorical input. Temperature difference and the torque-to-RPM ratio are engineered from the sensor measurements. Identifiers and outcome-specific failure flags are excluded from prediction features.

## Dashboard

| Page | Features |
|---|---|
| **Simulation Dashboard** | Machine selection, failure-risk gauge, SHAP drivers, sensor monitoring, risk trends, maintenance states, and Start/Pause/Reset controls. |
| **WHAT-IF Analysis** | Edit operating measurements and inspect how the selected machine's prediction responds. |
| **Survival Analysis** | Explore Kaplan–Meier cohort curves and Cox individual survival estimates under the tool-wear proxy assumption. |

Pausing saves the simulation snapshot; resuming continues from the saved state. Reset starts again from the cleaned observations.

## Analysis workflow

Run each notebook from top to bottom in the following order:

| Notebook | Focus |
|---|---|
| [01_data_understanding.ipynb](notebook/01_data_understanding.ipynb) | Dataset structure, variables, and failure distribution. |
| [02_eda.ipynb](notebook/02_eda.ipynb) | Sensor distributions, relationships, and failure patterns. |
| [03_data_cleaning.ipynb](notebook/03_data_cleaning.ipynb) | Data validation and cleaned dataset preparation. |
| [03b_smote_experiment.ipynb](notebook/03b_smote_experiment.ipynb) | Compare baseline, SMOTE, and class-weighting approaches. |
| [04_hypothesis_testing.ipynb](notebook/04_hypothesis_testing.ipynb) | Statistical investigation of sensor–failure relationships. |
| [05_failure_statistical_analysis.ipynb](notebook/05_failure_statistical_analysis.ipynb) | Failure-mode statistics and operating-condition comparisons. |
| [06_model_training_baseline.ipynb](notebook/06_model_training_baseline.ipynb) | Baseline classifier training and evaluation. |
| [07_ensemble_deep_learning_comparison.ipynb](notebook/07_ensemble_deep_learning_comparison.ipynb) | Classical models, neural networks, voting, and stacking. |
| [08_model_explainability_SHAP.ipynb](notebook/08_model_explainability_SHAP.ipynb) | Global and local prediction explanations. |
| [09_rul_ttf_regression.ipynb](notebook/09_rul_ttf_regression.ipynb) | Constructed RUL targets and regression experiments. |
| [10_survival_analysis1.ipynb](notebook/10_survival_analysis1.ipynb) | Survival curves, hazard modelling, and model comparison. |

Notebook 08 requires the saved deployment classifier. Model comparisons and SHAP calculations can take several minutes.

## Technology

**Python · pandas · NumPy · SciPy · statsmodels · scikit-learn · XGBoost · PyTorch TabNet · TensorFlow/Keras · SHAP · lifelines · Matplotlib · Seaborn · Plotly · Streamlit**

## Local setup

Use **Python 3.13**. For a new checkout, run in PowerShell:

```powershell
git clone https://github.com/roshantuladhar00/AI4I-Predictive-Maintenance-Analytics.git
cd AI4I-Predictive-Maintenance-Analytics
```

Open the project folder in VS Code. From that folder, create the Python environment and install the dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the repository is already cloned, start with the environment-creation command from its existing folder. If `py -3.13` is unavailable, use the full path to an installed Python 3.13 interpreter to create `.venv`.

In VS Code, install the **Python** and **Jupyter** extensions. Select `.venv\Scripts\python.exe` as the Python interpreter and as the notebook kernel. The full dependencies include PyTorch and TensorFlow, so installation can take time.

### Train the deployment models

A fresh Git clone excludes the model binaries. Generate the classifier, RUL regressor, evaluation metrics, and decision threshold with:

```powershell
.\.venv\Scripts\python.exe -m src.train
```

This creates `models/best_model.pkl`, `models/rul_regressor.joblib`, `models/eval.json`, and `models/threshold.json`. Training is optional when the complete, compatible artifacts are already present. Running notebook 09 does not overwrite these deployment artifacts.

### Start the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Open the URL printed in the terminal. Restart Streamlit after retraining so it reloads the saved models and threshold.

## Classification model and results

The deployment classifier combines **random forest, gradient boosting, and MLP** predictions using soft-voting weights of **2:2:1**. The random forest uses 400 trees, balanced class weights, and isotonic calibration. The MLP has hidden layers of 64 and 32 units.

The threshold-selection procedure evaluates the precision–recall curve and chooses the highest-precision threshold that reaches at least **95% failure recall**.

For the saved artifacts, the selected threshold is **0.025743180189271207**:

| Measure | Result |
|---|---:|
| Failures detected | 65 of 68 |
| Missed failures | 3 |
| Failure recall | 95.59% |
| Failure precision | 28.51% |
| False alarms | 163 among 1,932 non-failures |
| False-positive rate | 8.44% |
| Average precision | 0.8807 |
| ROC-AUC | 0.9743 |

These observations were also used to select the threshold, so the results are **threshold-selection results rather than an untouched final-test estimate**. The rule does not impose a 5% false-positive cap. Check `models/eval.json` and `models/threshold.json` after retraining because the results and threshold can change.

### Reading the gauge

The gauge shows the model score on a 0–100 scale, with a marker for the saved classification threshold. Its visual bands are green below 35%, yellow from 35% to below 70%, and red at 70% or higher. The simulation uses 70% for high-risk alerts and 90% for maintenance.

These visual and simulation rules are separate from the classification threshold. A score can cross the threshold while remaining in the green band. Only the random forest member is explicitly calibrated; the ensemble score is not a validated probability of failure within a specified future time window.

## Interpreting the analysis

SHAP explains model behavior relative to a background sample. It does not establish causal sensor effects or statistical significance. Model comparisons that use held-out data for early stopping or weight selection are exploratory; an independent evaluation protocol is needed for an unbiased final performance claim.

The RUL target is constructed from assumed wear limits and sensor stress, rather than measured remaining lifetime. Survival analysis treats tool wear as duration, although the dataset does not provide longitudinal machine lifetimes or verified censoring times. Interpret both as exploratory demonstrations. Statistical fitting should use observed failure labels rather than labels generated during simulation.
