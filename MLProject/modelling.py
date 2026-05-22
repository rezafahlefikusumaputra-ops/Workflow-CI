import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

# Ubah y menjadi array 1 dimensi
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Start MLflow
with mlflow.start_run():

    # Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Training
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Akurasi: {accuracy}")

    # Logging MLflow
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

print("Training selesai")
