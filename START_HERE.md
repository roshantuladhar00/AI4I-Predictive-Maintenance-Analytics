# AI4I Predictive Maintenance Analytics

This folder contains the complete notebook workflow and the three-page dashboard:

1. **Simulation Dashboard** — select machines, Start/Pause/Reset, failure-risk gauge, SHAP drivers, risk history, maintenance states, and snapshot resume.
2. **WHAT-IF Analysis** — edit sensor values in the table, press **Done**, and inspect the updated gauge and KPIs.
3. **Survival Analysis** — Kaplan–Meier cohorts and Cox analysis.

## Open in VS Code

Extract the ZIP to a **separate folder**. Open that folder using **File → Open Folder** in VS Code. You should see `app`, `data`, `models`, `notebook`, and `src` directly in Explorer.

Keep your current project until you have reviewed this version. These files form one complete project; mixing its model files or source modules with the earlier app will cause inconsistencies.

## Set up Python

In VS Code's PowerShell terminal, opened in this folder:

```powershell
& "C:\Users\rosha\anaconda3\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

These commands use the Python 3.13 installation verified on your computer. The full notebook dependencies include PyTorch and TensorFlow, so installation can take time. Do not use plain `py` for these pinned dependencies because your default launcher selects Python 3.14.

In VS Code, install the **Python** and **Jupyter** extensions. Use **Python: Select Interpreter** to select `.venv\Scripts\python.exe`. Select the same environment in each notebook's kernel picker.

## Run the dashboard

The classifier, RUL regressor, and decision threshold are already included.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The original saved simulation snapshot is included. Click **Reset simulation** for a fresh run from the cleaned dataset. Select a machine and click **Start**; click **Pause** to save the fleet state.

## Work through the notebooks

Run each notebook from top to bottom, in this order:

1. `01_data_understanding.ipynb`
2. `02_eda.ipynb`
3. `03_data_cleaning.ipynb`
4. `03b_smote_experiment.ipynb`
5. `04_hypothesis_testing.ipynb`
6. `05_failure_statistical_analysis.ipynb`
7. `06_model_training_baseline.ipynb`
8. `07_ensemble_deep_learning_comparison.ipynb`
9. `08_model_explainability_SHAP.ipynb`
10. `09_rul_ttf_regression.ipynb`
11. `10_survival_analysis1.ipynb`

Notebook 07 compares the original classical, deep-learning, voting, and stacking methods. Notebook 08 explains the saved deployment model. Notebooks 09 and 10 cover the original RUL and survival demonstrations. Notebook training can take several minutes.

To regenerate both dashboard models and their saved threshold:

```powershell
.\.venv\Scripts\python.exe -m src.train
```

Restart Streamlit after retraining so it reloads the saved models. See `PROJECT_NOTES.md` for the threshold explanation and the limits of the RUL and survival demonstrations.

The ZIP contains model binaries for local use. The original `.gitignore` excludes them from Git because they are large; a fresh Git clone will need `python -m src.train` before launching the dashboard.
