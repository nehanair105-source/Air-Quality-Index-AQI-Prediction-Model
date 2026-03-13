# Air Quality Index (AQI) Prediction Model

A machine learning web application that predicts **AQI category** for Indian cities using pollutant and weather data — built with Random Forest and deployed via Streamlit.

##  Project Overview

| Item | Details |
|------|---------|
| **Problem Type** | Multi-class Classification |
| **Dataset Size** | 842,160 records |
| **Best Model** | Random Forest |
| **Accuracy** | 85.3% |
| **F1 Score** | 0.85 (weighted) |

---

##  ML Pipeline

-  Exploratory Data Analysis (EDA)
-  Outlier treatment & data cleaning
-  Feature selection using VIF (Variance Inflation Factor)
-  OneHot Encoding + Standard Scaling
-  Model training — Logistic Regression, XGBoost, Random Forest
-  Hyperparameter tuning using GridSearchCV
-  Model evaluation — Accuracy, F1 Score, Confusion Matrix

---

##  Model Comparison

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Random Forest | **85.3%** | **0.85** |
| XGBoost | 73.5% | 0.74 |
| Logistic Regression | 70.9% | 0.71 |

---

## 🌫️ AQI Categories Predicted

| Category | Description |
|----------|-------------|
|  Good | Air quality is satisfactory |
|  Moderate | Acceptable air quality |
|  Unhealthy for Sensitive Groups | Risk for sensitive people |
|  Unhealthy | Everyone may experience effects |
|  Very Unhealthy | Health alert for all |

---

##  Features Used

**Pollutants:** PM2.5, NO₂, CO, SO₂, O₃, AOD

**Weather:** Dew Point, Cloud Cover, Heavy Rain

**Time:** Season, Time of Day, Weekend

**Location:** State (29 Indian states)

**Events:** Festival Period, Crop Burning Season

---

##  Project Structure
```
my_project/
├── app.py              # Streamlit web application
├── Aqi_project.ipynb   # ML model notebook
├── rf_model.pkl        # Trained Random Forest model
├── scaler.pkl          # StandardScaler
├── encoder.pkl         # OneHotEncoder
└── aqi_encoding.json   # Label mapping
```

---

##  Installation & Run
```bash
# Install dependencies
pip install streamlit scikit-learn xgboost joblib pandas numpy

# Run the app
streamlit run app.py
```

