import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ======================
# Load Dataset
# ======================

X_train = pd.read_csv(
    'machine_predictive_maintenance_preprocessing/X_train.csv'
)
X_test = pd.read_csv(
    'machine_predictive_maintenance_preprocessing/X_test.csv'
)
y_train = pd.read_csv(
    'machine_predictive_maintenance_preprocessing/y_train.csv'
)
y_test = pd.read_csv(
    'machine_predictive_maintenance_preprocessing/y_test.csv'
)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

# ======================
# Enable MLflow Autolog
# ======================

mlflow.autolog()

# ======================
# Training Model
# ======================

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train.values.ravel())

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    print(f"Accuracy : {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall   : {recall}")
    print(f"F1 Score : {f1}")
