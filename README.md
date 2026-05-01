# AQI India Predictor

## Project Overview

The AQI India Predictor is an end-to-end Machine Learning application designed to classify air quality levels across Indian regions using environmental, meteorological, and contextual features.

This project integrates a production-style prediction pipeline with an interactive Streamlit interface, ensuring both technical robustness and usability.

The system predicts AQI categories aligned with standard air quality classifications:

* Good
* Moderate
* Unhealthy for Sensitive Groups
* Unhealthy
* Very Unhealthy

---

## Problem Statement

Air pollution is a major public health concern in India. Traditional AQI monitoring systems rely on real-time sensor networks, which may not always be accessible.

This project aims to:

* Provide a data-driven AQI classification system
* Use historical pollutant and weather data to predict air quality
* Enable quick decision-making through an intuitive interface

---

## Machine Learning Pipeline

### 1. Data Processing

The dataset includes pollutant levels, weather conditions, and contextual features.

#### Numerical Features:

* PM2.5, CO, NO₂, SO₂, O₃, AOD
* Dew Point
* Cloud Cover

#### Categorical Features:

* Season
* Time of Day
* State

#### Binary Features:

* Weekend
* Heavy Rain
* Festival Period
* Crop Burning Season

---

### 2. Feature Engineering

* Categorical variables encoded using OneHotEncoder
* Numerical features scaled using StandardScaler
* Feature alignment handled to match training schema

---

### 3. Model Selection

* Algorithm: Random Forest Classifier
* Reason for selection:

  * Handles non-linear relationships effectively
  * Robust to noise and outliers
  * Performs well without extensive feature tuning

---

### 4. Model Performance

* Accuracy: ~85.3%
* Evaluated using train-test split
* Outputs class probabilities for interpretability

---

## System Architecture

```
User Input (Streamlit UI)
        ↓
Data Preprocessing
  - Encoding
  - Scaling
        ↓
Feature Alignment
        ↓
Random Forest Model
        ↓
Prediction + Probabilities
        ↓
UI Visualization (Confidence + Insights)
```

---

## Application (Streamlit)

The frontend is built using Streamlit with custom styling for improved user experience.

### Key Components:

* Pollutant input section
* Weather condition controls
* Time and contextual selectors
* Prediction result display
* Confidence score visualization
* Probability distribution across classes

---

## Project Structure

```
AQI_Project/
│
├── app.py                # Streamlit frontend and inference pipeline
├── rf_model.pkl          # Trained Random Forest model
├── scaler.pkl            # StandardScaler (fitted on training data)
├── encoder.pkl           # OneHotEncoder for categorical variables
├── aqi_encoding.json     # Label mapping (index to AQI category)
├── requirements.txt      # Python dependencies
└── README.md             # Documentation
```

---

## Prediction Workflow

1. User provides input via the interface
2. Inputs are converted into structured DataFrame
3. Categorical features are encoded
4. Numerical features are scaled
5. Features are aligned with training columns
6. Model generates predictions and probabilities
7. Results are displayed with supporting visualizations

```bash
pip install -r requirements.txt
```

### Step 3: Run Application

```bash
streamlit run app.py
```


