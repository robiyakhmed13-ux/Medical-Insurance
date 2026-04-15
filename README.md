# 🏥 Medical Insurance Cost Prediction

A machine learning project that predicts **individual medical insurance costs** using **XGBoost** and **Linear Regression**, based on demographic and health-related features.

---

## 📌 Project Overview

Medical insurance pricing depends on many personal factors. This project trains and compares two regression models — XGBoost and Linear Regression — to predict annual insurance charges for individuals, demonstrating how machine learning can assist in actuarial cost estimation.

| Item | Detail |
|------|--------|
| **Algorithms** | XGBoost Regressor, Linear Regression |
| **Task** | Regression |
| **Dataset** | [Medical Cost Personal Dataset – Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance) |
| **Target** | `charges` — Annual medical insurance cost (USD) |

---

## 📂 Project Structure

```
medical_insurance_prediction/
│
├── medical_insurance_prediction.ipynb   # Jupyter Notebook (full walkthrough)
├── medical_insurance_prediction.py      # Clean Python script
├── requirements.txt                     # Dependencies
├── insurance.csv                        # Dataset (download from Kaggle)
├── eda_plots.png                        # EDA visualizations
├── charges_by_smoker.png                # Insurance charges by smoker status
├── actual_vs_predicted.png              # Actual vs Predicted comparison
├── feature_importances.png              # XGBoost feature importances
└── README.md
```

---

## 📊 Dataset Features

| Feature | Description |
|---------|-------------|
| `age` | Age of the primary beneficiary |
| `sex` | Gender (male=0, female=1) |
| `bmi` | Body Mass Index (healthy range: 18.5–24.9) |
| `children` | Number of children covered by insurance |
| `smoker` | Smoking status (yes=0, no=1) |
| `region` | Residential area (southeast=0, southwest=1, northeast=2, northwest=3) |
| `charges` | ✅ **Target** — Individual medical insurance cost (USD/year) |

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/medical-insurance-prediction.git
cd medical-insurance-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `insurance.csv` from [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance) and place it in the project root.

### 4. Run
```bash
python medical_insurance_prediction.py
```

---

## 🔄 Pipeline

```
Raw CSV Data
    │
    ▼
EDA — Age, BMI, smoker, charges distributions + smoker vs charges
    │
    ▼
Categorical Encoding (sex, smoker, region)
    │
    ▼
Train / Test Split (80% / 20%)
    │
    ▼
XGBoost Regressor Training
Linear Regression Training
    │
    ▼
R² + MAE Comparison + Actual vs Predicted Plots
    │
    ▼
Single-individual Insurance Cost Prediction
```

---

## 📈 Results

| Model | Train R² | Test R² |
|-------|----------|---------|
| XGBoost Regressor | ~0.99 | ~0.90 |
| Linear Regression | ~0.75 | ~0.74 |

> XGBoost significantly outperforms Linear Regression, capturing non-linear relationships like the heavy cost impact of smoking.

---

## 🔑 Key Findings

- **Smokers** pay dramatically higher insurance costs — the single most impactful feature
- **Age** and **BMI** have strong positive correlations with charges
- **Linear Regression** struggles with the non-linear pricing structure; XGBoost handles it well

---

## 🔮 Sample Prediction

```python
# (age, sex, bmi, children, smoker, region)
# sex: male=0, female=1 | smoker: yes=0, no=1 | region: southeast=0...
sample = (62, 1, 26.29, 0, 1, 0)
predict_insurance_cost(xgb_model, sample)
# Output: 🏥 Predicted Insurance Cost: $X,XXX.XX USD/year
```

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas / numpy** — data processing
- **scikit-learn** — Linear Regression, train/test split, metrics
- **xgboost** — gradient boosting regression
- **seaborn / matplotlib** — visualization

---

## 🚀 Future Improvements

- [ ] Hyperparameter tuning for XGBoost (`GridSearchCV`)
- [ ] Add interaction features (e.g., `smoker × bmi`)
- [ ] Try Ridge/Lasso as regularised linear alternatives
- [ ] Deploy as a Streamlit cost estimator web app
- [ ] SHAP values for explainable AI on predictions

---

## 📄 License

MIT License

---

## 🙋 Author

**[Your Name]**  
[GitHub](https://github.com/your-username) | [LinkedIn](https://linkedin.com/in/your-profile)
