# Gauge dashboard update

I extract these files into the same repository root that contains my existing **models** and **notebook** folders, replacing the earlier app files and keeping **src**, **config**, and **tests** as subfolders.

The app includes:

- A semicircular failure alert gauge on a 0–100 score scale.
- The saved alert line at approximately 6.21.
- Green and red decision zones that follow the saved threshold.
- A plain-language explanation of the displayed result.
- One-click examples for demonstrating each outcome.
- Technical scores and submitted measurements inside an expander.
- The existing CSV prediction and download workflow.

I stop the running app with Ctrl+C, then restart it from the repository root:

    & "$env:USERPROFILE\anaconda3\python.exe" -m streamlit run app.py

If Plotly is missing from my selected Python 3.13 environment, I install the included requirements using that same interpreter:

    & "$env:USERPROFILE\anaconda3\python.exe" -m pip install -r requirements.txt

The score display multiplies the original score by 100. The model, original threshold, input preparation, and downloaded numeric predictions use the existing evaluation procedure.
