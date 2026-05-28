FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir pandas numpy scikit-learn mlflow matplotlib seaborn

CMD ["python", "modelling.py"]
