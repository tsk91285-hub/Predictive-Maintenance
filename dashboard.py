import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("rf_model.pkl")

st.title("Industrial Predictive Maintenance System")

air_temp = st.number_input(
    "Air Temperature (K)",
    value=300.0
)

process_temp = st.number_input(
    "Process Temperature (K)",
    value=310.0
)

rpm = st.number_input(
    "Rotational Speed (RPM)",
    value=1500
)

torque = st.number_input(
    "Torque (Nm)",
    value=40.0
)

tool_wear = st.number_input(
    "Tool Wear (min)",
    value=100
)

if st.button("Analyze Machine"):

    temp_diff = process_temp - air_temp

    power = rpm * torque

    data = np.array([
        [
            air_temp,
            process_temp,
            rpm,
            torque,
            tool_wear,
            temp_diff,
            power
        ]
    ])

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    health_score = round(
        (1 - probability) * 100,
        2
    )

    st.subheader("Results")

    st.write(
        f"Machine Health Score: {health_score}%"
    )

    st.write(
        f"Failure Probability: {round(probability*100,2)}%"
    )

    estimated_rul = max(
        0,
        250 - tool_wear
    )

    st.write(
        f"Estimated Remaining Useful Life: {estimated_rul} hours"
    )

    if probability > 0.8:
        st.error(
            "CRITICAL: Immediate Maintenance Required"
        )

    elif probability > 0.5:
        st.warning(
            "Schedule Maintenance Soon"
        )

    else:
        st.success(
            "Machine Operating Normally"
        )

    feature_names = [
        "Air Temp",
        "Process Temp",
        "RPM",
        "Torque",
        "Tool Wear",
        "Temp Diff",
        "Power"
    ]

    importance = model.feature_importances_

    fig, ax = plt.subplots()

    ax.barh(
        feature_names,
        importance
    )

    ax.set_xlabel("Importance")

    st.pyplot(fig)