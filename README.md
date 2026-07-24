# 📊 Customer Churn Prediction

A Machine Learning web application that predicts whether a telecom customer is likely to churn based on their demographic information, subscription details, and billing history.

The project implements an end-to-end ML pipeline using **Scikit-learn Pipelines**, **Random Forest**, and **Gradio** for deployment.

---

## 🚀 Features

- Predicts customer churn in real time
- Interactive Gradio web interface
- Automated data preprocessing using Scikit-learn Pipelines
- Hyperparameter tuning with GridSearchCV
- Cross-validation for model evaluation
- Model serialization using Pickle/Joblib

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Classifier
- Gradio
- Joblib / Pickle

---

---

## ⚙️ Machine Learning Pipeline

```
Dataset
    │
    ▼
Data Cleaning
    │
    ▼
Feature Engineering
    │
    ▼
ColumnTransformer
    │
    ├── Numerical Pipeline
    │      ├── Missing Value Imputation
    │      └── Standard Scaling
    │
    └── Categorical Pipeline
           ├── Missing Value Imputation
           └── One-Hot Encoding
    │
    ▼
Random Forest Classifier
    │
    ▼
GridSearchCV
    │
    ▼
Best Model
    │
    ▼
Model Serialization (.pkl)
    │
    ▼
Gradio Web App
```

---

## 📈 Model Training

The model was trained using:

- Random Forest Classifier
- Train-Test Split
- Scikit-learn Pipeline
- ColumnTransformer
- StandardScaler
- OneHotEncoder
- SimpleImputer

### Model Evaluation

- Accuracy Score
- 5-Fold Cross Validation
- GridSearchCV Hyperparameter Tuning

---

## 🌐 Web Application

The Gradio interface allows users to:

- Enter customer information
- Predict whether the customer will churn
- Receive predictions instantly

---

## 📸 Example Prediction

### Input

- Gender: Female
- Contract: Month-to-month
- Internet Service: Fiber Optic
- Monthly Charges: 85.4
- Tenure: 8 Months

### Output

```
Churn
```

---

## 🔧 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git

cd customer-churn-prediction
```

Create a virtual environment

```bash
python -m venv venv
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Train the model

```bash
python model.py
```

### Launch the web app

```bash
python app.py
```

---

## 📚 Future Improvements

- XGBoost implementation
- LightGBM comparison
- Explain predictions using SHAP
- Docker deployment
- FastAPI REST API
- Streamlit dashboard
- Cloud deployment (Render/AWS)

---

---

## 👨‍💻 Author

**M Sarwar**

- Python Developer
- Machine Learning Enthusiast
- Generative AI Learner

⭐ If you found this project useful, consider giving it a star!
