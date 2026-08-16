# ❤️ Heart Risk Prediction Model

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A Machine Learning project that predicts the likelihood of **Heart Disease** using patient clinical data. The project follows a complete ML pipeline, including data ingestion, preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, and model serialization.

---

# 🩺 Overview

Heart disease remains one of the leading causes of mortality worldwide. Early detection through predictive analytics can assist healthcare professionals in making informed decisions.

This project leverages **FLAML (Fast Lightweight AutoML)** to automatically identify the best-performing machine learning model and optimize its hyperparameters. After evaluating multiple algorithms, FLAML selected **XGBoost Classifier** as the optimal model for heart disease prediction.

---

# 📊 Dataset

### Features

| Column | Description |
|---------|-------------|
| age | Patient Age |
| gender | Patient Gender |
| impluse | Heart Pulse Rate |
| pressurehight | Systolic Blood Pressure |
| pressurelow | Diastolic Blood Pressure |
| glucose | Blood Glucose Level |
| kcm | CK-MB (Cardiac Marker) |
| troponin | Troponin Level |
| class | Target Variable |

### Target Variable

| Value | Meaning |
|-------|---------|
| 0 | No Heart Disease |
| 1 | Heart Disease |

---

# ✨ Features

- Data Ingestion Pipeline
- Data Cleaning
- Missing Value Handling
- Duplicate Removal
- Feature Scaling
- Categorical Encoding
- SMOTE for Class Balancing
- Automated Model Selection using FLAML AutoML
- Automated Hyperparameter Optimization
- Best Model Selection
- Model Serialization using Pickle
- Modular Project Structure

---

# 📁 Project Structure

```text
Heart-Risk-Prediction-Model/
│
├── data/
│   └── Heart Attack.csv
│
├── models/ 
|   ├── model.log
|   └── model.pkl
│
├── reports/
│
├── research/
│
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   └── model_build.py
│
├── .gitignore
├── LICENSE
├── main.py
├── pyproject.toml
├── README.md
└── uv.lock
```

---

# ⚙️ Machine Learning Pipeline

```
Dataset
     │
     ▼
Data Ingestion
     │
     ▼
Data Cleaning
     │
     ▼
Missing Value Handling
     │
     ▼
Encoding
     │
     ▼
Feature Scaling
     │
     ▼
SMOTE
     │
     ▼
Train-Test Split
     │
     ▼
FLAML AutoML
     │
     ▼
XGBoost Classifier (Selected Automatically)
     │
     ▼
Model Evaluation
     │
     ▼
Save Model (.pkl)
```

---

# 🤖 Model Used

The project utilizes **FLAML AutoML**, a lightweight and efficient automated machine learning library developed by Microsoft.

FLAML automatically:

- Evaluates multiple machine learning algorithms
- Optimizes hyperparameters
- Selects the best-performing model based on the evaluation metric

### Best Model Selected

**XGBoost Classifier**

### Best Hyperparameters

```python
learning_rate = 0.3
max_depth = 6
min_child_weight = 1
n_estimators = 10
```
---

# 📈 Model Performance

| Metric | Score |
|---------|-------|
| Selected Model | XGBoost Classifier |
| Training Accuracy | **99.74%** |
| Testing Accuracy | **97.73%** |

Using FLAML AutoML, the project automatically identified **XGBoost Classifier** as the best-performing algorithm, achieving excellent predictive performance with strong generalization on unseen data.

---
# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- FLAML (AutoML)
- XGBoost
- Imbalanced-Learn (SMOTE)
- Pickle

---


# ⚡ Why FLAML?

Instead of manually comparing multiple machine learning algorithms, this project uses **FLAML AutoML** to automate model selection and hyperparameter optimization.

### Algorithms Evaluated

- XGBoost
- Random Forest
- Extra Trees
- LightGBM (if installed)
- Logistic Regression
- Other supported estimators

FLAML automatically selected **XGBoost Classifier** as the best-performing model based on the evaluation metric.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/deveshdubey18/Heart-Risk-Prediction-Model.git
```

Move into the project directory

```bash
cd Heart-Risk-Prediction-Model
```

Install dependencies

```bash
pip install -r requirements.txt
```

or (if using `uv`)

```bash
uv sync
```

---

# ▶️ Usage

Run the project

```bash
python main.py
```

The trained model will be saved inside the **models/** directory as

```
model.pkl
```

---

# 📌 Future Improvements

- Deploy using Streamlit
- Build REST API using FastAPI
- Docker Containerization
- Model Explainability using SHAP
- Cross Validation
- ROC Curve & AUC Visualization
- Feature Importance Visualization

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

### Devesh Dubey

GitHub: https://github.com/deveshdubey18

If you found this project useful, consider giving it a ⭐ on GitHub!
