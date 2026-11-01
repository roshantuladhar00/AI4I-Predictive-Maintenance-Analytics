"""Free rule-based observation chatbot. No API or network calls.
Copy to src/chatbot.py. From the project root run:
    .venv/Scripts/python.exe -m streamlit run src/chatbot.py
Uses the existing simulation-dashboard classifier, not final_evaluation.
Only load model artifacts from your own trusted project.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SENSORS = ["Air temperature [K]", "Process temperature [K]",
           "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
RAW = SENSORS + ["Type"]
QUESTIONS = ["Explain this prediction", "Which features increased the score?",
             "Which features decreased the score?", "What does SHAP mean?",
             "Show the sensor readings", "Was the prediction correct?",
             "When will it fail?"]
LIMIT = ("This explains model behavior, not physical causes or a future failure time.")


def answer(question, context):
    """Use fixed rules and calculated facts. No language model is involved."""
    q = " ".join(re.sub(r"[^a-z0-9 ]", " ", question.lower()).split())
    score, threshold = context["score"], context["threshold"]
    drivers = context.get("drivers", [])

    def contributions(sign):
        if not drivers:
            return "Calculate the SHAP explanation first to inspect contributions."
        selected = [d for d in drivers if d["value"] * sign > 1e-10]
        if not selected:
            return "No nonzero contributions in this direction were found."
        direction = "raised" if sign > 0 else "lowered"
        lines = [f"- {d['feature']}: {direction} the score by {abs(d['value']):.4f} score units."
                 for d in selected]
        return "\n".join(lines) + "\n\nContributions are relative to the SHAP reference. " + LIMIT

    if any(w in q for w in ["when", "how long", "remaining", "rul", "ttf", "soon"]):
        return "This classifier cannot determine a failure time or measured remaining lifetime. " + LIMIT
    if any(w in q for w in ["repair", "fix", "replace", "maintenance", "should i", "recommend"]):
        return "No verified diagnosis or service instructions are available for this observation. " + LIMIT
    if any(w in q for w in ["what if", "change", "reduce", "increase torque"]):
        return "What-if recalculation is outside this version. Use your existing WHAT-IF dashboard. A changed prediction does not establish a physical effect."
    if any(w in q for w in ["correct", "actual", "true label", "wrong", "mistake"]):
        actual = context.get("actual")
        if actual is None:
            return "No valid observed failure label is available."
        predicted = int(score >= threshold)
        outcome = {(1, 1): "true positive", (0, 0): "true negative",
                   (1, 0): "false positive", (0, 1): "false negative"}[(predicted, actual)]
        return (f"Predicted class: {predicted}; supplied observed label: {actual}. "
                f"This is a **{outcome}** at the saved threshold. "
                "This is descriptive: the selected row may have been used in training.")
    if any(w in q for w in ["sensor", "reading", "measurement"]):
        return "\n".join(f"- {k}: {context['row'][k]}" for k in RAW)
    if any(w in q for w in ["decreased", "lowered", "lower", "negative", "protect"]):
        return contributions(-1)
    if any(w in q for w in ["increased", "raised", "positive", "driver", "contribut", "feature"]):
        return contributions(1)
    if "shap" in q:
        return ("SHAP assigns parts of a prediction's difference from a reference score to inputs. "
                "Positive values raise the score; negative values lower it. The reference uses "
                "32 sampled training rows; engineered features are recomputed for each combination. "
                "Independent masking can create unrealistic combinations of correlated inputs. " + LIMIT)
    if any(w in q for w in ["explain", "why", "prediction", "score", "threshold", "flag", "risk", "summary"]):
        label = "Failure alert" if score >= threshold else "No alert at selected threshold"
        text = (f"**{label}.** Score: **{score:.6f}**; saved threshold: **{threshold:.6f}**. "
                "The rule is score >= threshold. The score is not a validated probability "
                "of failure within a future time window.")
        if drivers:
            strongest = max(drivers, key=lambda d: abs(d["value"]))
            text += (f"\n\nLargest absolute SHAP contribution: **{strongest['feature']}** "
                     f"({strongest['value']:+.4f} score units). Reference score: {context['base']:.6f}.")
        else:
            text += "\n\nCalculate SHAP to inspect individual contributions."
        return text + "\n\n" + LIMIT
    return "I could not match this question to a supported topic. Choose a suggested question. This rule-based assistant has limited wording support."


@st.cache_resource
def load_model(path, signature):
    import joblib
    return joblib.load(path)


def feature_frame(raw):
    result = raw.loc[:, RAW].copy()
    result["Temp_diff"] = result[SENSORS[1]] - result[SENSORS[0]]
    result["Torque_RPM_ratio"] = result[SENSORS[3]] / (result[SENSORS[2]] + 1)
    return result


def score_rows(model, raw):
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Classifier has no failure class 1.")
    values = np.asarray(model.predict_proba(feature_frame(raw)))[:, classes.index(1)]
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Model returned invalid scores.")
    return values


def explain(model, frame, row):
    import shap
    from sklearn.model_selection import train_test_split
    # Reproduce src/train.py's dashboard split for the background.
    train, _ = train_test_split(frame, test_size=0.2, random_state=42,
                               stratify=frame["Machine failure"])
    background = train.sample(n=min(32, len(train)), random_state=42)
    codes = {"L": 0, "M": 1, "H": 2}
    types = np.array(["L", "M", "H"])

    def encode(rows):
        data = rows.loc[:, RAW].copy()
        data["Type"] = data["Type"].map(codes)
        return data.to_numpy(dtype=float)

    def predict(values):
        data = pd.DataFrame(values, columns=RAW)
        indices = data["Type"].to_numpy(dtype=float)
        if not np.isin(indices, [0, 1, 2]).all():
            raise ValueError("Unexpected Type mask value.")
        data["Type"] = types[indices.astype(int)]
        return score_rows(model, data)

    explainer = shap.Explainer(predict, shap.maskers.Independent(encode(background)),
                               algorithm="exact", feature_names=RAW)
    result = explainer(encode(pd.DataFrame([row])), max_evals=64)
    values = np.asarray(result.values).reshape(-1)
    base = float(np.asarray(result.base_values).reshape(-1)[0])
    score = float(score_rows(model, pd.DataFrame([row]))[0])
    if not np.isclose(base + values.sum(), score, atol=1e-6):
        raise ValueError("SHAP contributions do not reconstruct this prediction.")
    drivers = sorted([{"feature": f, "value": float(v)} for f, v in zip(RAW, values)],
                     key=lambda d: abs(d["value"]), reverse=True)
    return {"drivers": drivers, "base": base}


def main():
    st.set_page_config(page_title="Observation Chatbot", layout="wide")
    st.title("Selected Observation Chatbot")
    st.caption("Free rule-based assistant | No API | No data sent to a language model")
    st.info("Standalone page using the simulation dashboard's best_model.pkl and threshold.json. It does not automatically appear inside the existing dashboard and does not use the separate final-evaluation model.")
    try:
        model_path = ROOT / "models/best_model.pkl"
        policy_path = ROOT / "models/threshold.json"
        data_path = ROOT / "data/cleaned/ai4i2020_cleaned.csv"
        for path in [model_path, policy_path, data_path]:
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}. Put chatbot.py in the project's src folder and ensure your dashboard artifacts exist.")
        threshold = float(json.loads(policy_path.read_text())["final_threshold"])
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Saved threshold must be between 0 and 1.")
        frame = pd.read_csv(data_path)
        if frame.empty or not set(RAW + ["Machine failure"]).issubset(frame.columns):
            raise ValueError("CSV needs the six original inputs and Machine failure.")
        for column in SENSORS:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[SENSORS].to_numpy(dtype=float)).all():
            raise ValueError("Sensor values must be finite.")
        if (frame[SENSORS[:3]] <= 0).any().any() or (frame[SENSORS[3:]] < 0).any().any():
            raise ValueError("Invalid sensor ranges in cleaned data.")
        if not frame["Type"].isin(["L", "M", "H"]).all():
            raise ValueError("Type must contain L, M, or H.")
        if not frame["Machine failure"].isin([0, 1]).all():
            raise ValueError("Machine failure must contain 0 or 1.")
        signature = hashlib.sha256(model_path.read_bytes()).hexdigest()
        model = load_model(str(model_path), signature)
        position = st.selectbox("Select observation", range(len(frame)),
            format_func=lambda i: f"Row {i + 1} | UDI {frame.iloc[i].get('UDI', 'unavailable')} | Product ID {frame.iloc[i].get('Product ID', 'unavailable')}")
        row = frame.iloc[position]
        identity = hashlib.sha256((signature + str(threshold) + row.to_json()).encode()).hexdigest()
        if st.session_state.get("observation_chat_identity") != identity:
            st.session_state.observation_chat_identity = identity
            st.session_state.observation_chat_messages = []
            st.session_state.observation_chat_shap = {}
        score = float(score_rows(model, pd.DataFrame([row]))[0])
        context = {"row": row.to_dict(), "score": score, "threshold": threshold,
                   "actual": int(row["Machine failure"]), **st.session_state.observation_chat_shap}
        left, right = st.columns(2)
        with left:
            st.subheader("Observation and prediction")
            st.dataframe(row[RAW].astype(str).rename("Value"), use_container_width=True)
            st.metric("Model score", f"{score:.6f}")
            st.write(f"Saved decision threshold: {threshold:.6f}")
            st.write("**Failure alert**" if score >= threshold else "**No alert at selected threshold**")
            if st.button("Calculate SHAP explanation"):
                try:
                    with st.spinner("Calculating SHAP contributions..."):
                        st.session_state.observation_chat_shap = explain(model, frame, row)
                    st.rerun()
                except Exception as exc:
                    st.error(f"SHAP calculation failed: {exc}")
            if context.get("drivers"):
                st.dataframe(pd.DataFrame(context["drivers"]), hide_index=True)
                st.caption(f"Reference score: {context['base']:.6f}. Positive contributions raise the score; negative contributions lower it.")
            st.caption(LIMIT)
        with right:
            st.subheader("Ask about this observation")
            st.caption("Fixed-rule answers. Conversation resets when the observation or model changes.")
            suggestion = st.selectbox("Suggested question", QUESTIONS)
            submitted = st.button("Ask suggested question")
            for message in st.session_state.observation_chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            question = st.chat_input("Ask about this observation")
            if submitted:
                question = suggestion
            if question:
                st.session_state.observation_chat_messages.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer(question, context)}])
                st.rerun()
            if st.button("Clear conversation"):
                st.session_state.observation_chat_messages = []
                st.rerun()
    except Exception as exc:
        st.error(str(exc))
        st.caption("This module does not train models or edit your project files.")


if __name__ == "__main__":
    main()


