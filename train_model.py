import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("data/ai4i2020.csv")

# Feature Engineering
df["Temp_Diff"] = (
    df["Process temperature [K]"]
    - df["Air temperature [K]"]
)

df["Power"] = (
    df["Rotational speed [rpm]"]
    * df["Torque [Nm]"]
)

# Features
X = df[
[
'Air temperature [K]',
'Process temperature [K]',
'Rotational speed [rpm]',
'Torque [Nm]',
'Tool wear [min]',
'Temp_Diff',
'Power'
]]

# Target
y = df['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print(classification_report(y_test, pred))

joblib.dump(model, "rf_model.pkl")

print("Model Saved Successfully")