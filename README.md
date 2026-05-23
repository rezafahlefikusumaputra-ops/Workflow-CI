# Workflow CI Machine Learning

Repository ini dibuat untuk memenuhi submission Membangun Sistem Machine Learning Dicoding.

## Struktur Repository

```bash
Workflow-CI/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── MLProject/
│   ├── modelling.py
│   ├── conda.yaml
│   ├── MLproject
│   ├── requirements.txt
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
```

## Workflow CI

Workflow CI menggunakan GitHub Actions untuk menjalankan training machine learning secara otomatis saat terjadi push ke branch `main`.

## Tools

- Python
- MLflow
- Scikit-learn
- GitHub Actions
