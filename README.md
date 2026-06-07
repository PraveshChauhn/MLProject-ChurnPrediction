# Telco Customer Churn Prediction

A machine learning project that analyzes telecom customer data to predict churn, with an interactive web application for real-time predictions.

---

## Project Structure

```
telco-churn-prediction/
│
├── first_telc.csv                          # Reference dataset for feature alignment
├── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Source dataset (Kaggle)
├── tel_churn.csv                           # Processed dataset for model training
├── model.sav                               # Trained Random Forest model (pickle)
│
├── Telco_Churn_Analysis.ipynb              # EDA & data exploration notebook
├── Churn_Analysis_-_Model_Building.ipynb   # Model training & evaluation notebook
│
├── app.py                                  # Flask web application (backend + routes)
└── templates/
    └── home.html                           # Bootstrap 4 form UI with Jinja2 templating
```

---

## Project Overview

Customer churn is when a customer stops using a company's services. Predicting churn helps telecom companies proactively retain customers. This project builds an end-to-end pipeline:

1. **Exploratory Data Analysis** — Understand patterns and factors driving churn
2. **Model Building** — Train and evaluate classification models
3. **Web Application** — Serve predictions through a Flask API

---

## Dataset

- **Source:** [Telco Customer Churn — IBM Sample Dataset (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Records:** ~7,043 customers
- **Target Variable:** `Churn` (Yes / No)
- **Class Distribution:** ~73% No Churn, ~27% Churn (imbalanced)

### Features

| Category | Features |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account Info | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

---

## Exploratory Data Analysis (`Telco_Churn_Analysis.ipynb`)

Key insights discovered:

- **High churn** is associated with:
  - Month-to-month contracts
  - No online security or tech support
  - First year of subscription
  - Fiber optic internet service
  - Higher monthly charges

- **Low churn** is associated with:
  - Long-term contracts (1 or 2 year)
  - No internet service subscription
  - Customers with 5+ years of tenure

- **Minimal impact** on churn: `gender`, `PhoneService`, number of multiple lines

- `TotalCharges` was converted from object to numeric; 11 missing rows (~0.15%) were dropped
- `tenure` was binned into 6-month groups: `1-12`, `13-24`, `25-36`, `37-48`, `49-60`, `61-72`

---

## Model Building (`Churn_Analysis_-_Model_Building.ipynb`)

### Handling Class Imbalance
Used **SMOTEENN** (combination of SMOTE oversampling + Edited Nearest Neighbours undersampling) to balance the dataset before training.

### Models Evaluated

| Model | Technique | Accuracy |
|---|---|---|
| Decision Tree | Without SMOTEENN | Low |
| Decision Tree | With SMOTEENN | ~92% |
| Random Forest | Without SMOTEENN | Moderate |
| Random Forest | With SMOTEENN | **Best** |
| Random Forest | With SMOTEENN + PCA | No improvement |

### Final Model
**Random Forest Classifier with SMOTEENN**
```python
RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    random_state=100,
    max_depth=6,
    min_samples_leaf=8
)
```
The trained model is saved as `model.sav` using `pickle`.

---

## 🌐 Web Application (`app.py`)

A **Flask** web app that accepts customer details via a form and returns a churn prediction with confidence score.

### How It Works

1. User fills in 19 customer feature fields in the UI
2. The input is appended to the reference dataset (`first_telc.csv`) for proper one-hot encoding alignment
3. The trained model (`model.sav`) predicts churn probability
4. Output is displayed: **"Likely to churn"** or **"Likely to continue"** with a confidence percentage

### Input Fields (19 Features)

`SeniorCitizen`, `MonthlyCharges`, `TotalCharges`, `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `tenure`

---

## 🖥️ User Interface (`templates/home.html`)

The frontend is a **Bootstrap 4** single-page form built with Jinja2 templating.

### UI Layout

```
┌─────────────────────────────────────────┐
│         Churn Prediction Form           │
│                                         │
│  SeniorCitizen     [ textarea ]         │
│  MonthlyCharges    [ textarea ]         │
│  TotalCharges      [ textarea ]         │
│  gender            [ textarea ]         │
│  Partner           [ textarea ]         │
│  ...  (19 input fields total)           │
│                                         │
│              [ SUBMIT ]                 │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │  This customer is likely to...  │    │
│  │  Confidence: XX.XX%             │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

### Features
- **19 input fields** — one textarea per customer feature, pre-filled with previous values after form submission (via Jinja2 `{{query1}}` ... `{{query19}}`)
- **Prediction output** — two read-only textareas display the churn result (`{{output1}}`) and the model confidence score (`{{output2}}`) below the form
- **Bootstrap 4** for responsive layout via CDN (no local CSS needed)
- **jQuery + Popper.js** included via CDN for Bootstrap JS components

### Expected Input Values

| Field | Type | Example Values |
|---|---|---|
| `SeniorCitizen` | Binary int | `0` or `1` |
| `MonthlyCharges` | Float | `29.85` |
| `TotalCharges` | Float | `1889.50` |
| `gender` | String | `Male`, `Female` |
| `Partner` | String | `Yes`, `No` |
| `Dependents` | String | `Yes`, `No` |
| `PhoneService` | String | `Yes`, `No` |
| `MultipleLines` | String | `Yes`, `No`, `No phone service` |
| `InternetService` | String | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity` | String | `Yes`, `No`, `No internet service` |
| `OnlineBackup` | String | `Yes`, `No`, `No internet service` |
| `DeviceProtection` | String | `Yes`, `No`, `No internet service` |
| `TechSupport` | String | `Yes`, `No`, `No internet service` |
| `StreamingTV` | String | `Yes`, `No`, `No internet service` |
| `StreamingMovies` | String | `Yes`, `No`, `No internet service` |
| `Contract` | String | `Month-to-month`, `One year`, `Two year` |
| `PaperlessBilling` | String | `Yes`, `No` |
| `PaymentMethod` | String | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `tenure` | Integer | `1` – `72` (months) |

---

## Getting Started

### Prerequisites

```bash
pip install flask pandas numpy scikit-learn imbalanced-learn seaborn matplotlib
```

### Running the Application

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/telco-churn-prediction.git
   cd telco-churn-prediction
   ```

2. **Update file paths in `app.py`**

   Replace the hardcoded Windows paths with your local paths:
   ```python
   # Line 6 — update CSV path
   df_1 = pd.read_csv("first_telc.csv")

   # Line 25 — update model path
   model = pickle.load(open("model.sav", "rb"))
   ```

3. **Ensure required files are present**
   - `first_telc.csv` in the project root
   - `model.sav` in the project root
   - `templates/home.html` in the `templates/` folder

4. **Run the Flask app**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| Web Framework | Flask |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Imbalanced Data | imbalanced-learn (SMOTEENN) |
| Visualization | Seaborn, Matplotlib |
| Model Serialization | Pickle |

---

## Results

The final **Random Forest + SMOTEENN** model achieved:
- **~92% accuracy** on the resampled test set
- Strong **recall and precision for minority class** (churned customers)
- Outperformed Decision Tree and PCA-reduced variants

---

## Notes

- The `first_telc.csv` file is used at inference time to ensure consistent one-hot encoding columns between training and prediction
- `SeniorCitizen` is a binary numeric feature (0 or 1), not a string
- The app runs on `host='0.0.0.0'` and `port=5000` by default

---

## 📄 License

This project is for educational purposes. Dataset credit: IBM / Kaggle.
