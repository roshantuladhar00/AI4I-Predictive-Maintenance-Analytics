"""Local Streamlit interface for the frozen AI4I prediction procedure."""

from pathlib import Path
import hashlib

import pandas as pd
import streamlit as st

from src.predict import (
    ArtifactError, InputValidationError, MAX_BATCH_ROWS, MAX_CSV_BYTES,
    RAW_COLUMNS, asset_signature, load_bundle, parse_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="AI4I Maintenance Analytics", page_icon="⚙️", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1380px; padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {background: #fff; border: 1px solid #dce5eb;
        border-radius: 12px; padding: 16px;}
    [data-testid="stForm"] {background: #fff; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the saved model…")
def cached_bundle(project_root: str, signature: tuple):
    return load_bundle(project_root)


try:
    signature = asset_signature(PROJECT_ROOT)
    bundle = cached_bundle(str(PROJECT_ROOT), signature)
except ArtifactError as exc:
    st.title("AI4I Maintenance Analytics")
    st.error(str(exc))
    st.info("Place app.py, src, config, and sample_input.csv in your project root, alongside models and notebook.")
    st.stop()

if st.session_state.get("asset_signature") != signature:
    st.session_state.pop("single_prediction", None)
    st.session_state.pop("batch_prediction", None)
    st.session_state["asset_signature"] = signature

test_metrics = bundle.metadata["final_test_metrics"]
selection = bundle.reference["threshold_selection"]

with st.sidebar:
    st.markdown("### AI4I / Maintenance")
    st.caption("Predictive maintenance analytics")
    st.divider()
    st.markdown("**Saved decision policy**")
    st.metric("Alert threshold", f"{bundle.threshold:.6f}")
    st.caption("The full saved threshold is used for every prediction.")
    st.markdown("**Selection objective**")
    st.write(
        f"Maximize validation failure recall with a "
        f"{bundle.policy['validation_FPR_cap']:.0%} false-positive-rate cap."
    )
    st.divider()
    st.caption("Weighted voting · Uncalibrated model score")
    st.caption("Project prototype using the synthetic AI4I dataset.")

st.caption("AI4I 2020 · FROZEN MODEL")
st.title("Maintenance prediction workspace")
st.write("Check one observation or score a CSV using the evaluated model and saved alert policy.")

single_tab, batch_tab, results_tab = st.tabs(["Single observation", "CSV predictions", "Model & results"])

with single_tab:
    st.subheader("Enter machine measurements")
    st.caption("Use kelvin, rpm, newton-metres, and minutes as shown. The initial values are an example observation.")
    with st.form("single_observation"):
        left, right = st.columns(2)
        with left:
            air = st.number_input("Air temperature (K)", min_value=0.0, value=297.4, step=0.1, format="%.2f", key="air_temperature")
            rpm = st.number_input("Rotational speed (rpm)", min_value=0.0, value=1389.0, step=1.0, format="%.1f", key="rotational_speed")
            wear = st.number_input("Tool wear (min)", min_value=0.0, value=54.0, step=1.0, format="%.1f", key="tool_wear")
        with right:
            process = st.number_input("Process temperature (K)", min_value=0.0, value=308.5, step=0.1, format="%.2f", key="process_temperature")
            torque = st.number_input("Torque (Nm)", min_value=0.0, value=39.2, step=0.1, format="%.2f", key="torque")
            product_type = st.selectbox("Product type", bundle.schema["valid_Type_values"], key="product_type")
        single_submitted = st.form_submit_button("Predict observation", type="primary")

    if single_submitted:
        inputs = pd.DataFrame([{
            "Air temperature [K]": air,
            "Process temperature [K]": process,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
            "Type": product_type,
        }])
        try:
            st.session_state["single_prediction"] = bundle.predict_frame(inputs)
        except InputValidationError as exc:
            st.session_state.pop("single_prediction", None)
            st.error(str(exc))

    single_result = st.session_state.get("single_prediction")
    if single_result is not None:
        row = single_result.iloc[0]
        st.subheader("Latest submitted observation")
        if bool(row["failure_alert"]):
            st.warning("Failure alert — this observation crosses the selected alert threshold.")
        else:
            st.success("No alert at the selected threshold.")
        score_column, threshold_column = st.columns(2)
        score_column.metric("Model score · uncalibrated", f"{row['failure_score']:.6f}")
        threshold_column.metric("Decision threshold", f"{bundle.threshold:.6f}")
        st.caption("An alert is raised when the model score is greater than or equal to the saved threshold.")
        if bool(row["outside_training_range"]):
            st.warning(
                "Some inputs fall outside the observed training ranges: "
                + str(row["outside_training_features"])
                + ". Predictions in these ranges need additional validation."
            )
        st.dataframe(single_result[RAW_COLUMNS], hide_index=True, use_container_width=True)
        st.download_button(
            "Download this prediction",
            data=single_result.to_csv(index=False).encode("utf-8"),
            file_name="ai4i_single_prediction.csv", mime="text/csv",
            key="download_single",
        )

with batch_tab:
    st.subheader("Predict from a CSV")
    st.write("Upload the six original input columns. Additional columns, such as observation IDs, are preserved in the result.")
    with st.expander("Required columns and input rules"):
        st.dataframe(
            pd.DataFrame({
                "Column": RAW_COLUMNS,
                "Rule": [
                    "Kelvin; greater than zero",
                    "Kelvin; greater than zero",
                    "rpm; greater than zero",
                    "Nm; zero or greater",
                    "Minutes; zero or greater",
                    "L, M, or H; surrounding spaces and lowercase are normalized",
                ],
            }), hide_index=True, use_container_width=True,
        )
        st.caption(
            f"UTF-8 CSV, at most {MAX_BATCH_ROWS:,} observations and 10 MB. "
            "A batch with invalid inputs is rejected with data-row numbers."
        )
    sample_path = PROJECT_ROOT / "sample_input.csv"
    if sample_path.is_file():
        st.download_button(
            "Download example input CSV", sample_path.read_bytes(),
            file_name="sample_input.csv", mime="text/csv", key="download_example",
        )
    uploaded = st.file_uploader("Choose an input CSV", type=["csv"], key="batch_upload")
    payload = None
    parsed = None
    upload_digest = None
    if uploaded is not None:
        if uploaded.size > MAX_CSV_BYTES:
            st.error("The CSV must be 10 MB or smaller.")
        else:
            payload = uploaded.getvalue()
            upload_digest = hashlib.sha256(payload).hexdigest()
            try:
                parsed = parse_csv(payload)
                st.caption(f"Input preview · {len(parsed):,} observations")
                st.dataframe(parsed.head(5), hide_index=True, use_container_width=True)
            except InputValidationError as exc:
                st.error(str(exc))
    if st.session_state.get("batch_upload_digest") != upload_digest:
        st.session_state.pop("batch_prediction", None)
        st.session_state["batch_upload_digest"] = upload_digest
    batch_submitted = st.button(
        "Predict CSV", type="primary", disabled=parsed is None, key="predict_csv"
    )
    if batch_submitted and parsed is not None:
        try:
            with st.spinner("Scoring observations…"):
                st.session_state["batch_prediction"] = bundle.predict_frame(parsed)
        except InputValidationError as exc:
            st.session_state.pop("batch_prediction", None)
            st.error(str(exc))
    batch_result = st.session_state.get("batch_prediction")
    if batch_result is not None:
        st.subheader("Batch results")
        total_column, alert_column, range_column = st.columns(3)
        total_column.metric("Observations", f"{len(batch_result):,}")
        alert_column.metric("Failure alerts", f"{int(batch_result['failure_alert'].sum()):,}")
        range_column.metric("Outside training ranges", f"{int(batch_result['outside_training_range'].sum()):,}")
        if batch_result["outside_training_range"].any():
            st.warning(
                "Some observations contain values outside the observed training ranges. "
                "The download identifies the affected inputs."
            )
        st.dataframe(batch_result, hide_index=True, use_container_width=True)
        st.download_button(
            "Download all predictions", batch_result.to_csv(index=False).encode("utf-8"),
            file_name="ai4i_batch_predictions.csv", mime="text/csv", key="download_batch",
        )

with results_tab:
    st.subheader("The evaluated model")
    st.write(
        "The app uses the saved weighted voting ensemble and the fixed threshold from notebook 09. "
        "It recreates the same engineered features before scoring each observation."
    )
    recall_column, fpr_column, precision_column = st.columns(3)
    recall_column.metric("Final-test failure recall", f"{test_metrics['recall']:.1%}")
    fpr_column.metric("Final-test false-positive rate", f"{test_metrics['false_positive_rate']:.2%}")
    precision_column.metric("Final-test alert precision", f"{test_metrics['precision']:.1%}")
    st.caption(
        f"Final test: {test_metrics['true_positives']} of {test_metrics['failures']} failures detected; "
        f"{test_metrics['false_negatives']} missed; {test_metrics['false_positives']} false alarms."
    )
    st.markdown("#### Why this threshold?")
    st.write(
        f"I searched the distinct validation score thresholds and allowed at most "
        f"{selection['maximum_false_positives']} false alarms among "
        f"{selection['normal_rows']:,} normal observations. "
        f"The selected operating point detected {selection['true_positives']} of "
        f"{selection['failures']} failures with {selection['false_positives']} false alarms."
    )
    st.write(
        "I maximized the number of detected failures within the 5% validation FPR cap, "
        "then preferred fewer false alarms when recall was tied. "
        "The policy was frozen before final-test scoring."
    )
    st.info(
        "False-positive rate measures false alarms among normal observations. "
        f"On the final test, {test_metrics['false_discovery_rate']:.1%} of alerts were false alarms. "
        "Inspection workload still needs operational review."
    )
    st.markdown("#### Observed training ranges")
    reference_table = pd.DataFrame(bundle.reference["numeric_training_ranges"]).T
    reference_table.index.name = "Input"
    st.dataframe(
        reference_table.rename(columns={"minimum": "Minimum", "maximum": "Maximum"}),
        use_container_width=True,
    )
    st.caption(
        "Ranges use the original training partition. Inputs outside a range are flagged, "
        "and the model still returns a score."
    )
    st.markdown("#### Interpretation")
    st.write(
        "Scores are uncalibrated. This synthetic-data project uses a random split, and the final-test "
        "observations were included in early exploratory analysis. The app demonstrates the evaluated "
        "prediction procedure; operational validation on independent machine data remains pending."
    )
