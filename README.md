# Industrial Machine Failure Prediction and Predictive Maintenance System

## Project Overview

This project is a Machine Learning-based Predictive Maintenance System developed to predict industrial machine failures before they occur. The system analyzes machine operating parameters and provides maintenance recommendations to reduce downtime and improve operational efficiency.

## Problem Statement

Unexpected machine failures can cause production delays, increased maintenance costs, and reduced productivity. Traditional maintenance approaches often fail to predict failures in advance. This project uses Machine Learning to identify potential failures and support proactive maintenance planning.

## Features

* Machine Failure Prediction
* Machine Health Monitoring
* Predictive Maintenance Recommendations
* Remaining Useful Life (RUL) Estimation
* Interactive Dashboard using Streamlit
* Random Forest Machine Learning Model

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Matplotlib

## Dataset

AI4I Predictive Maintenance Dataset (AI4I 2020)

Dataset Features:

* Air Temperature
* Process Temperature
* Rotational Speed (RPM)
* Torque (Nm)
* Tool Wear (min)

## Project Structure

```text
PredictiveMaintenance/
│
├── data/
│   └── ai4i2020.csv
│
├── train_model.py
├── dashboard.py
├── rf_model.pkl
├── requirements.txt
└── README.md
```

## Installation

### Clone Repository

```bash
git clone https://github.com/tsk91285-hub/Predictive-Maintenance.git
cd Predictive-Maintenance
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python train_model.py
```

## Run Dashboard

```bash
python -m streamlit run dashboard.py
```

## Expected Outcome

The system predicts machine failure probability, evaluates machine health, and provides maintenance recommendations that help industries reduce downtime and improve reliability.

## Learning Outcomes

* Machine Learning Model Development
* Data Preprocessing and Feature Engineering
* Predictive Analytics
* Dashboard Development using Streamlit
* Industrial Predictive Maintenance Concepts

## Future Scope

* Real-Time IoT Sensor Integration
* Cloud Deployment
* Advanced RUL Prediction
* Mobile Application Support
* Deep Learning-based Failure Prediction

## Author

Tara Singh

Machine Learning Internship Project


