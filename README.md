# AI4I Predictive Maintenance Analytics

A complete AI4I 2020 project covering data understanding, EDA, cleaning, SMOTE, hypothesis testing, failure statistics, classification, ensemble and deep-learning comparison, SHAP explainability, RUL regression, and survival analysis.

The Streamlit dashboard includes **Simulation Dashboard**, **WHAT-IF Analysis**, and **Survival Analysis**, with the original failure-risk gauge, live sensor states, SHAP drivers, maintenance simulation, and saved snapshot resume.

Start with [START_HERE.md](START_HERE.md) for VS Code setup and the notebook order. The trained classifier, RUL regressor, and threshold are included.

From the project folder, after installing the requirements:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

To regenerate the deployment artifacts:

```powershell
.\.venv\Scripts\python.exe -m src.train
```

[PROJECT_NOTES.md](PROJECT_NOTES.md) explains the model, threshold, regenerated metrics, reference source, compatibility changes, and scientific limits of the proxy RUL/survival demonstrations.
