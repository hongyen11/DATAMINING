# =====================================================
# BIS3216 Data Mining Final Assessment
# PM2.5 Prediction (Regression)
# Models: Random Forest (Ensemble) + Neural Network (MLP)
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------
df = pd.read_csv("C:/Users/notco/Downloads/BeijingPM2.5.csv")

# Target and features
target = "pm2.5"
features = ["TEMP", "PRES", "DEWP", "Iws", "Is", "Ir"]

X = df[features]
y = df[target]

# -----------------------------------------------------
# 2. Train-Test Split (80/20)
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------
# 3. RANDOM FOREST REGRESSION (ENSEMBLE)
# -----------------------------------------------------
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

rf.fit(X_train, y_train)

rf_train_pred = rf.predict(X_train)
rf_test_pred  = rf.predict(X_test)

# RF Metrics
rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
rf_test_rmse  = np.sqrt(mean_squared_error(y_test, rf_test_pred))
rf_train_mae  = mean_absolute_error(y_train, rf_train_pred)
rf_test_mae   = mean_absolute_error(y_test, rf_test_pred)
rf_train_r2   = r2_score(y_train, rf_train_pred)
rf_test_r2    = r2_score(y_test, rf_test_pred)

# -----------------------------------------------------
# 4. NEURAL NETWORK REGRESSION (MLP)
# -----------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

mlp.fit(X_train_scaled, y_train)

mlp_train_pred = mlp.predict(X_train_scaled)
mlp_test_pred  = mlp.predict(X_test_scaled)

# MLP Metrics
mlp_train_rmse = np.sqrt(mean_squared_error(y_train, mlp_train_pred))
mlp_test_rmse  = np.sqrt(mean_squared_error(y_test, mlp_test_pred))
mlp_train_mae  = mean_absolute_error(y_train, mlp_train_pred)
mlp_test_mae   = mean_absolute_error(y_test, mlp_test_pred)
mlp_train_r2   = r2_score(y_train, mlp_train_pred)
mlp_test_r2    = r2_score(y_test, mlp_test_pred)

# -----------------------------------------------------
# 5. MODEL COMPARISON TABLE (TEST SET)
# -----------------------------------------------------
results = pd.DataFrame({
    "Model": ["Random Forest", "Neural Network (MLP)"],
    "RMSE": [rf_test_rmse, mlp_test_rmse],
    "MAE":  [rf_test_mae, mlp_test_mae],
    "R²":   [rf_test_r2, mlp_test_r2]
})

print("\n===== TEST SET MODEL PERFORMANCE =====")
print(results.round(3))

# -----------------------------------------------------
# 6. TRAIN vs TEST PERFORMANCE (OVERFITTING CHECK)
# -----------------------------------------------------
train_test_results = pd.DataFrame({
    "Model": ["Random Forest", "Neural Network"],
    "Train RMSE": [rf_train_rmse, mlp_train_rmse],
    "Test RMSE":  [rf_test_rmse, mlp_test_rmse],
    "Train R²":   [rf_train_r2, mlp_train_r2],
    "Test R²":    [rf_test_r2, mlp_test_r2]
})

print("\n===== TRAIN vs TEST PERFORMANCE =====")
print(train_test_results.round(3))

# -----------------------------------------------------
# 7. RANDOM FOREST FEATURE IMPORTANCE
# -----------------------------------------------------
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
sns.barplot(
    x=importances[indices],
    y=np.array(features)[indices],
    palette="magma"
)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 8. ACTUAL vs PREDICTED PLOT
# -----------------------------------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, rf_test_pred, alpha=0.5, label="Random Forest")
plt.scatter(y_test, mlp_test_pred, alpha=0.5, label="Neural Network")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "k--", lw=2)

plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Actual vs Predicted PM2.5")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 9. RESIDUAL DISTRIBUTION
# -----------------------------------------------------
rf_residuals  = y_test - rf_test_pred
mlp_residuals = y_test - mlp_test_pred

plt.figure(figsize=(8,6))
sns.histplot(rf_residuals, bins=40, kde=True, label="Random Forest")
sns.histplot(mlp_residuals, bins=40, kde=True, label="Neural Network")

plt.xlabel("Residual (Actual − Predicted)")
plt.title("Residual Distribution")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 10. RESIDUALS vs PREDICTED
# -----------------------------------------------------
plt.figure(figsize=(8,6))
plt.scatter(rf_test_pred, rf_residuals, alpha=0.5, label="Random Forest")
plt.scatter(mlp_test_pred, mlp_residuals, alpha=0.5, label="Neural Network")

plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Predicted PM2.5")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted Values")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 11. K-Fold Cross-Validation (Robustness Check)
# -----------------------------------------------------
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rf_cv_rmse = -cross_val_score(
    rf, X, y,
    scoring="neg_root_mean_squared_error",
    cv=kf
)

mlp_cv_rmse = -cross_val_score(
    mlp, scaler.fit_transform(X),
    y,
    scoring="neg_root_mean_squared_error",
    cv=kf
)

print("\n===== 5-FOLD CROSS-VALIDATION RMSE =====")
print(f"Random Forest CV RMSE: Mean={rf_cv_rmse.mean():.2f}, Std={rf_cv_rmse.std():.2f}")
print(f"Neural Network CV RMSE: Mean={mlp_cv_rmse.mean():.2f}, Std={mlp_cv_rmse.std():.2f}")
