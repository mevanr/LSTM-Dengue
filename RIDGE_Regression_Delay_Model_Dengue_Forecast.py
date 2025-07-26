import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression

# Set random seed for reproducibility
SEED = 7
np.random.seed(SEED)

# =============================================
# DATA LOADING AND PREPROCESSING (Same as before)
# =============================================

# Load and prepare data
data_path = r"Rain_Data_Case_Dependent_Imputation.txt"
df = pd.read_csv(data_path, sep="\t", header=0)
df.columns = ["Year", "Month", "Cases", "RainFl", "RainDy", "Temp", "Rhumid", "PI", "CI"]
df = df[["Year", "Month", "Cases", "RainFl", "RainDy", "Temp", "Rhumid", "PI", "CI"]]
df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(Day=1))
df.set_index("Date", inplace=True)
df.drop(columns=["Year", "Month"], inplace=True)
df.fillna(df.median(), inplace=True)

# Create lagged features
lags = 3
variables_to_lag = ["RainFl", "RainDy", "Temp", "Rhumid", "PI", "CI"]
for var in variables_to_lag:
    for lag in range(1, lags + 1):
        df[f"{var}_lag_{lag}"] = df[var].shift(lag)
df.dropna(inplace=True)

# =============================================
# SCALE AND PREPARE DATA (Modified for Ridge)
# =============================================

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
X = scaled_data[:, 1:]
y = scaled_data[:, 0]

# Create sequences (but flatten them for Ridge)
seq_length = 12

def create_sequences_for_ridge(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        # Flatten the sequence into one row per time step
        X_seq.append(X[i:i+seq_length].flatten())
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences_for_ridge(X, y, seq_length)

# Split into train/test (keeping temporal order)
train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# =============================================
# RIDGE REGRESSION MODEL TRAINING
# =============================================

# Find optimal alpha using TimeSeriesCrossValidation
alphas = np.logspace(-4, 4, 50)
best_alpha = None
best_score = -np.inf

tscv = TimeSeriesSplit(n_splits=5)

for alpha in alphas:
    ridge = Ridge(alpha=alpha, random_state=SEED)
    scores = []
    for train_index, val_index in tscv.split(X_train):
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]
        
        ridge.fit(X_tr, y_tr)
        score = ridge.score(X_val, y_val)
        scores.append(score)
    
    mean_score = np.mean(scores)
    if mean_score > best_score:
        best_score = mean_score
        best_alpha = alpha

print(f"Best alpha: {best_alpha:.4f}")

# Train final model with best alpha
ridge = Ridge(alpha=best_alpha, random_state=SEED)
ridge.fit(X_train, y_train)

# =============================================
# MODEL EVALUATION (Same structure as LSTM)
# =============================================

# Predictions
y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)

# Inverse transform
def inverse_transform(y_pred, y_true, scaler, X_shape):
    dummy = np.zeros((len(y_pred), X_shape[1] + 1))
    dummy[:, 0] = y_pred.flatten()
    y_pred_actual = scaler.inverse_transform(dummy)[:, 0]
    
    dummy[:, 0] = y_true.flatten()
    y_true_actual = scaler.inverse_transform(dummy)[:, 0]
    return y_pred_actual, y_true_actual

y_train_pred_act, y_train_act = inverse_transform(y_train_pred, y_train, scaler, X.shape)
y_test_pred_act, y_test_act = inverse_transform(y_test_pred, y_test, scaler, X.shape)

# Calculate confidence intervals
train_ci = 1.96 * np.std(y_train_act - y_train_pred_act)
test_ci = 1.96 * np.std(y_test_act - y_test_pred_act)

# =============================================
# PLOTTING RESULTS (Same as before)
# =============================================

import matplotlib.dates as mdates

all_dates = df.index[seq_length:]  # Skip first 'seq_length' points
train_dates = all_dates[:train_size]
test_dates = all_dates[train_size:]

# Plot training set
plt.figure(figsize=(14, 6))
plt.plot(train_dates, y_train_act, label="Actual", color="blue")
plt.plot(train_dates, y_train_pred_act, label="Predicted", color="red", linestyle="--")
plt.fill_between(train_dates,
                 y_train_pred_act - train_ci,
                 y_train_pred_act + train_ci,
                 color="gray", alpha=0.3, label="95% Confidence Interval")
plt.title("Ridge Model: Training Set Performance", pad=20, fontsize=16)

ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Dengue Cases", fontsize=14)
plt.legend(loc='upper right', fontsize=12, frameon=True)
plt.tight_layout()
plt.show()

# Plot test set
plt.figure(figsize=(14, 6))
plt.plot(test_dates, y_test_act, label="Actual", color="blue")
plt.plot(test_dates, y_test_pred_act, label="Predicted", color="red", linestyle="--")
plt.fill_between(test_dates,
                 y_test_pred_act - test_ci,
                 y_test_pred_act + test_ci,
                 color="gray", alpha=0.3, label="95% Confidence Interval")
plt.title("Ridge Model: Test Set Performance", pad=20, fontsize=16)

ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Dengue Cases", fontsize=14)
plt.legend(loc='upper right', fontsize=12, frameon=True)
plt.tight_layout()
plt.show()

# =============================================
# MODEL PERFORMANCE STATISTICS
# =============================================

def calculate_metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

train_metrics = calculate_metrics(y_train_act, y_train_pred_act)
test_metrics = calculate_metrics(y_test_act, y_test_pred_act)

metrics_df = pd.DataFrame({
    "Training Set": train_metrics,
    "Test Set": test_metrics
}).T

print("\nRidge Model Performance Metrics:")
print(metrics_df.round(3))

# =============================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================

# Get feature names (accounting for lagged features)
# feature_names = []
# for i in range(seq_length):
#     for var in df.columns[1:]:  # Skip 'Cases'
#         feature_names.append(f"{var}_t-{seq_length-i}")

# # Plot coefficients
# plt.figure(figsize=(12, 8))
# sorted_idx = np.argsort(np.abs(ridge.coef_))[::-1]
# sorted_coef = ridge.coef_[sorted_idx]
# sorted_names = [feature_names[i] for i in sorted_idx]

# plt.barh(range(len(sorted_coef[:20])), sorted_coef[:20][::-1], align='center')
# plt.yticks(range(len(sorted_coef[:20])), sorted_names[:20][::-1])
# plt.xlabel('Coefficient Value')
# plt.title('Top 20 Ridge Regression Coefficients (by absolute value)')
# plt.tight_layout()
# plt.show()

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def calculate_ridge_aic_bic(model, X, y):
    # Get number of parameters (including intercept)
    n_params = X.shape[1] + 1
    
    # Get predictions and calculate MSE
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    n = len(y)
    
    # Calculate AIC/BIC
    aic = n * np.log(mse) + 2 * n_params
    bic = n * np.log(mse) + np.log(n) * n_params
    
    return aic, bic

# Example usage:
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)  # Flatten time steps

ridge_aic, ridge_bic = calculate_ridge_aic_bic(ridge_model, 
                                              X_test.reshape(X_test.shape[0], -1), 
                                              y_test)
print(f"Ridge AIC: {ridge_aic:.2f}, BIC: {ridge_bic:.2f}")