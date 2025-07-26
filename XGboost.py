import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.feature_selection import mutual_info_regression

# Set seed
SEED = 71
np.random.seed(SEED)

# Load data
data_path = r"Rain_Data_Case_Dependent_Imputation.txt"
df = pd.read_csv(data_path, sep="\t")
df.columns = ["Year", "Month", "Cases", "RainFl", "RainDy", "Temp", "Rhumid", "PI", "CI"]
df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(Day=1))
df.set_index("Date", inplace=True)
df.drop(columns=["Year", "Month"], inplace=True)
df.fillna(df.median(), inplace=True)

# Lag features
lags = [1, 2, 3, 6, 12]
for var in ["Cases", "RainFl", "RainDy", "Temp", "Rhumid", "PI", "CI"]:
    for lag in lags:
        df[f"{var}_lag_{lag}"] = df[var].shift(lag)

# Rolling features
for var in ["Cases", "RainFl", "RainDy", "Temp"]:
    for window in [3, 6, 12]:
        df[f"{var}_rolling_mean_{window}"] = df[var].rolling(window=window).mean()
        df[f"{var}_rolling_std_{window}"] = df[var].rolling(window=window).std()

# Seasonal, diff, EWM
df["month"] = df.index.month
df["quarter"] = df.index.quarter
df["Cases_diff_1"] = df["Cases"].diff(1)
df["Cases_diff_12"] = df["Cases"].diff(12)
df["Cases_ewm_3"] = df["Cases"].ewm(span=3).mean()
df["Cases_ewm_12"] = df["Cases"].ewm(span=12).mean()

df.dropna(inplace=True)

# Feature selection
X = df.drop("Cases", axis=1)
y = df["Cases"]
mi_scores = pd.Series(mutual_info_regression(X, y, random_state=SEED), index=X.columns)
top_features = mi_scores.nlargest(30).index.tolist()
df_filtered = df[["Cases"] + top_features]

# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_filtered)
X = scaled_data[:, 1:]
y = scaled_data[:, 0]

# Create sequences
seq_length = 12
X_seq, y_seq = [], []
for i in range(len(X) - seq_length):
    seq_features = X[i:i + seq_length].flatten()
    X_seq.append(seq_features)
    y_seq.append(y[i + seq_length])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# Train-test split
train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# Custom loss
def asymmetric_loss(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual < 0, -2 * 10.0 * residual, -2 * residual)
    hess = np.where(residual < 0, 2 * 10.0, 2.0)
    return grad, hess

# XGBoost model
xgb_model = xgb.XGBRegressor(
    objective=asymmetric_loss,
    n_estimators=2000,
    learning_rate=0.005,
    max_depth=8,
    subsample=0.7,
    colsample_bytree=0.7,
    colsample_bylevel=0.7,
    gamma=0.5,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    early_stopping_rounds=100,
    tree_method='hist',
    enable_categorical=False
)

# TimeSeries CV
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X_train):
    xgb_model.fit(X_train[train_idx], y_train[train_idx],
                  eval_set=[(X_train[val_idx], y_train[val_idx])],
                  verbose=0)

xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=1)

# Predictions
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

# Inverse transform
def inverse_transform(y_pred, y_true, scaler, df_columns):
    dummy = np.zeros((len(y_pred), len(df_columns)))
    dummy[:, 0] = y_pred
    y_pred_actual = scaler.inverse_transform(dummy)[:, 0]
    dummy[:, 0] = y_true
    y_true_actual = scaler.inverse_transform(dummy)[:, 0]
    return y_pred_actual, y_true_actual

y_train_pred_act, y_train_act = inverse_transform(y_train_pred, y_train, scaler, df_filtered.columns)
y_test_pred_act, y_test_act = inverse_transform(y_test_pred, y_test, scaler, df_filtered.columns)

# Metrics
def calculate_metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100,
        "PeakError": np.mean(np.abs(y_true[y_true > np.percentile(y_true, 75)] -
                                     y_pred[y_true > np.percentile(y_true, 75)]))
    }

metrics_df = pd.DataFrame({
    "Training Set": calculate_metrics(y_train_act, y_train_pred_act),
    "Test Set": calculate_metrics(y_test_act, y_test_pred_act)
}).T

print("\nModel Performance Metrics:")
print(metrics_df.round(3))

# Residual std from training set
train_residuals = y_train_act - y_train_pred_act
residual_std = np.std(train_residuals)
ci = 1.96 * residual_std  # 95% CI

# Dates
all_dates = df_filtered.index[seq_length:]
train_dates = all_dates[:train_size]
test_dates = all_dates[train_size:]

# Plot Training with CI
plt.figure(figsize=(14, 6))
plt.plot(train_dates, y_train_act, label="Actual", color="blue", linewidth=2)
plt.plot(train_dates, y_train_pred_act, label="Predicted", color="red", linestyle="--", linewidth=1.5)
plt.fill_between(train_dates,
                 y_train_pred_act - ci,
                 y_train_pred_act + ci,
                 color='gray', alpha=0.3, label="95% Confidence Interval")
plt.title("Training Set: Actual vs Predicted with 95% Confidence Interval", fontsize=14)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Dengue Cases", fontsize=14)
plt.legend(loc='upper right', fontsize=12, frameon=True)
#plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Plot Test with CI
plt.figure(figsize=(14, 6))
plt.plot(test_dates, y_test_act, label="Actual", color="blue", linewidth=2)
plt.plot(test_dates, y_test_pred_act, label="Predicted", color="red", linestyle="--", linewidth=1.5)
plt.fill_between(test_dates,
                 y_test_pred_act - ci,
                 y_test_pred_act + ci,
                 color='gray', alpha=0.3, label="95% Confidence Interval")
plt.title("Test Set: Actual vs Predicted with 95% Confidence Interval", fontsize=14)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Dengue Cases", fontsize=14)
plt.legend(loc='upper right', fontsize=12, frameon=True)#plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

