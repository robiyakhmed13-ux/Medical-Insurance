# =============================================================================
# Medical Insurance Cost Prediction using XGBoost & Linear Regression
# Author: [Your Name]
# Dataset: https://www.kaggle.com/datasets/mirichoi0218/insurance
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor


# =============================================================================
# 1. Data Loading
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load the medical insurance dataset."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    return df


# =============================================================================
# 2. EDA
# =============================================================================

def plot_eda(df: pd.DataFrame) -> None:
    """Visualise distributions of key features."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Medical Insurance Dataset – EDA", fontsize=16)

    sns.histplot(df['age'],      kde=True,  ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title("Age Distribution")

    sns.countplot(x='sex',      data=df,   ax=axes[0, 1], palette='Set2')
    axes[0, 1].set_title("Sex Distribution")

    sns.histplot(df['bmi'],      kde=True,  ax=axes[0, 2], color='salmon')
    axes[0, 2].set_title("BMI Distribution")
    axes[0, 2].axvline(18.5, color='green', linestyle='--', label='Min normal')
    axes[0, 2].axvline(24.9, color='red',   linestyle='--', label='Max normal')
    axes[0, 2].legend(fontsize=8)

    sns.countplot(x='children', data=df,   ax=axes[1, 0], palette='Blues')
    axes[1, 0].set_title("Number of Children")

    sns.countplot(x='smoker',   data=df,   ax=axes[1, 1], palette='Set1')
    axes[1, 1].set_title("Smoker Distribution")

    sns.histplot(df['charges'],  kde=True,  ax=axes[1, 2], color='goldenrod')
    axes[1, 2].set_title("Insurance Charges (Target)")

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=150)
    plt.show()
    print("EDA plots saved as 'eda_plots.png'")

    # Charges by smoker status
    plt.figure(figsize=(7, 5))
    sns.boxplot(x='smoker', y='charges', data=df, palette='Set1')
    plt.title("Insurance Charges by Smoker Status")
    plt.tight_layout()
    plt.savefig("charges_by_smoker.png", dpi=150)
    plt.show()
    print("Smoker chart saved as 'charges_by_smoker.png'")


# =============================================================================
# 3. Data Preprocessing
# =============================================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features."""
    df.replace({'sex':    {'male': 0, 'female': 1}}, inplace=True)
    df.replace({'smoker': {'yes':  0, 'no':     1}}, inplace=True)
    df['region'] = df['region'].replace({
        'southeast': 0, 'southwest': 1,
        'northeast': 2, 'northwest': 3
    }).astype(int)
    print("Encoding complete.")
    return df


# =============================================================================
# 4. Feature / Target Split
# =============================================================================

def split_features_target(df: pd.DataFrame):
    X = df.drop(columns='charges', axis=1)
    Y = df['charges']
    print(f"Features: {X.shape} | Target: {Y.shape}")
    return X, Y


# =============================================================================
# 5. Train / Test Split
# =============================================================================

def split_data(X, Y, test_size=0.2, random_state=2):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, Y_train, Y_test


# =============================================================================
# 6. Model Training
# =============================================================================

def train_models(X_train, Y_train):
    """Train both XGBoost and Linear Regression."""
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=2)
    xgb_model.fit(X_train, Y_train)

    lin_model = LinearRegression()
    lin_model.fit(X_train, Y_train)

    print("XGBoost and Linear Regression training complete.")
    return xgb_model, lin_model


# =============================================================================
# 7. Model Evaluation
# =============================================================================

def evaluate_models(xgb_model, lin_model, X_train, Y_train, X_test, Y_test) -> None:
    """Compare XGBoost vs Linear Regression: R² and MAE."""
    models = {"XGBoost Regressor": xgb_model, "Linear Regression": lin_model}

    for name, model in models.items():
        train_preds = model.predict(X_train)
        test_preds  = model.predict(X_test)
        print(f"\n── {name} ──")
        print(f"  Train R²  : {r2_score(Y_train, train_preds):.4f}")
        print(f"  Test  R²  : {r2_score(Y_test,  test_preds):.4f}")
        print(f"  Test  MAE : {mean_absolute_error(Y_test, test_preds):.2f}")

    # Actual vs Predicted scatter (both models)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Actual vs Predicted Insurance Charges", fontsize=14)

    for ax, (name, model) in zip(axes, models.items()):
        test_preds = model.predict(X_test)
        ax.scatter(Y_test, test_preds, alpha=0.5, color='steelblue')
        mn = min(Y_test.min(), test_preds.min())
        mx = max(Y_test.max(), test_preds.max())
        ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect fit')
        ax.set_xlabel("Actual Charges ($)")
        ax.set_ylabel("Predicted Charges ($)")
        ax.set_title(name)
        ax.legend()

    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    plt.show()
    print("Plot saved as 'actual_vs_predicted.png'")

    # XGBoost feature importances
    importances = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
    importances.sort_values(ascending=True).plot(
        kind='barh', figsize=(7, 5), color='steelblue',
        title='Feature Importances (XGBoost)'
    )
    plt.tight_layout()
    plt.savefig("feature_importances.png", dpi=150)
    plt.show()
    print("Feature importances saved as 'feature_importances.png'")


# =============================================================================
# 8. Predictive System
# =============================================================================

def predict_insurance_cost(model, input_data: tuple) -> None:
    """
    Predict medical insurance cost for a single individual.

    Parameters
    ----------
    input_data : tuple of 6 values:
        (age, sex, bmi, children, smoker, region)
        sex    : male=0, female=1
        smoker : yes=0, no=1
        region : southeast=0, southwest=1, northeast=2, northwest=3
    """
    arr = np.asarray(input_data).reshape(1, -1)
    cost = model.predict(arr)[0]
    print(f"\n🏥 Predicted Insurance Cost: ${cost:,.2f} USD/year")


# =============================================================================
# Main Pipeline
# =============================================================================

if __name__ == "__main__":
    DATA_PATH = "insurance.csv"   # update path if needed

    df = load_data(DATA_PATH)
    print("\nFirst 5 rows:\n", df.head())
    print("\nStatistical summary:\n", df.describe())

    plot_eda(df)
    df = preprocess_data(df)

    X, Y = split_features_target(df)
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    xgb_model, lin_model = train_models(X_train, Y_train)
    evaluate_models(xgb_model, lin_model, X_train, Y_train, X_test, Y_test)

    # Sample prediction: age=62, female, bmi=26.29, no children, non-smoker, southeast
    sample = (62, 1, 26.29, 0, 1, 0)
    predict_insurance_cost(xgb_model, sample)
