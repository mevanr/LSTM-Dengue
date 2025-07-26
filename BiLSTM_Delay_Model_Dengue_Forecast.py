import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Bidirectional, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from statsmodels.stats.outliers_influence import variance_inflation_factor
import tensorflow as tf
from tensorflow.keras.utils import set_random_seed
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set all random seeds for reproducibility
SEED = 7  # 10 #3 # 10 #42  # 10  7 best
np.random.seed(SEED)
tf.random.set_seed(SEED)
set_random_seed(SEED)  # For Keras operations

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
# EXPLORATORY DATA ANALYSIS
# =============================================

# 1. Triangular Correlation Matrix
plt.figure(figsize=(18, 15))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, fmt=".2f", 
            annot_kws={"size": 10}, cbar_kws={"shrink": .8})
plt.title("Correlation Matrix of Variables", pad=20, fontsize=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# Calculate p-values for correlations
p_values = pd.DataFrame(index=df.columns, columns=df.columns)
for col1 in df.columns:
    for col2 in df.columns:
        if col1 != col2:
            _, p_val = stats.pearsonr(df[col1], df[col2])
            p_values.loc[col1, col2] = p_val

plt.figure(figsize=(16, 12))
sns.heatmap(p_values.astype(float), mask=mask, annot=True, cmap='viridis', 
            fmt=".3f", cbar_kws={'label': 'p-value'})
plt.title("Statistical Significance of Correlations (p-values)", pad=20, fontsize=20)
plt.tight_layout()
plt.xticks(rotation=45, ha='right',  fontsize=14)
plt.yticks(rotation=0, fontsize=14)
plt.show()


# 2. Triangular Covariance Matrix
plt.figure(figsize=(18, 15))
cov_matrix = df.cov()
mask = np.triu(np.ones_like(cov_matrix, dtype=bool))
sns.heatmap(cov_matrix, mask=mask, annot=True, cmap='BrBG', center=0, fmt=".1f",
           annot_kws={"size": 10}, cbar_kws={"shrink": .8})
plt.title("Covariance Matrix of Variables", pad=20, fontsize=20)
plt.xticks(rotation=45, ha='right',  fontsize=14)
plt.yticks(rotation=0, fontsize=14)
plt.tight_layout()
plt.show()


# Pairplots for key variables
# Enhanced Pairplot with Log-Log Regression Lines and larger fonts
g = sns.PairGrid(df[['Cases', 'RainFl', 'Temp', 'PI', 'CI']])

# Set the font size for all axes
plt.rcParams.update({
    'xtick.labelsize': 12,  # X-axis tick label size
    'ytick.labelsize': 12   # Y-axis tick label size
})

g.map_upper(sns.scatterplot, alpha=0.6)
g.map_lower(sns.kdeplot, cmap='Blues_d')
g.map_diag(sns.histplot, kde=True)

def log_log_reg(x, y, **kws):
    x_log = np.log(x + 1e-5)
    y_log = np.log(y + 1e-5)
    slope, intercept, r_value, p_value, _ = stats.linregress(x_log, y_log)
    plt.plot(x, np.exp(intercept + slope*np.log(x + 1e-5)), 
             color='red', linewidth=2)
    ax = plt.gca()
    ax.annotate(f"Slope: {slope:.2f}\nR²: {r_value**2:.2f}", 
                xy=(0.7, 0.9), xycoords='axes fraction',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8),
                fontsize=10)  # Font size for annotation

g.map_upper(log_log_reg)

# Adjust axis label sizes
for ax in g.axes.flatten():
    if ax:  # Only if the axis exists
        ax.xaxis.label.set_size(14)  # X-axis label size
        ax.yaxis.label.set_size(14)  # Y-axis label size

plt.suptitle("Pairwise Relationships with Log-Log Regression Lines", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()


# =============================================
# FACTOR IMPORTANCE ANALYSIS
# =============================================

# Calculate VIF scores
vif_data = pd.DataFrame()
vif_data["Variable"] = df.columns
vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]

# Calculate mutual information scores
from sklearn.feature_selection import mutual_info_regression
mi_scores = mutual_info_regression(df.drop('Cases', axis=1), df['Cases'])
mi_scores = pd.Series(mi_scores, index=df.drop('Cases', axis=1).columns)

# Plot importance metrics
plt.figure(figsize=(14, 8))
plt.subplot(1, 2, 1)
sns.barplot(x=vif_data["VIF"], y=vif_data["Variable"], palette='viridis')
plt.title('Variance Inflation Factors (VIF)', pad=15, fontsize=16)
plt.xlabel('VIF Score', fontsize=16)
plt.ylabel('')

plt.subplot(1, 2, 2)
mi_scores.sort_values().plot(kind='barh', color='teal')
plt.title('Mutual Information Scores', pad=15, fontsize=16)
plt.xlabel('MI Score', fontsize=16)

plt.suptitle('Factor Importance Analysis', y=1.02, fontsize=20)
plt.tight_layout()
plt.show()


# =============================================
# MODEL TRAINING
# =============================================

# Scale and prepare data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
X = scaled_data[:, 1:]
y = scaled_data[:, 0]

def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

seq_length = 12
X_seq, y_seq = create_sequences(X, y, seq_length)
train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# Build LSTM model
model = Sequential([
    Input(shape=(seq_length, X_train.shape[2])),
    LayerNormalization(),
    Bidirectional(LSTM(128, activation="relu", return_sequences=True, kernel_regularizer=l2(0.005))),
    Dropout(0.3),
    Bidirectional(LSTM(64, activation="relu", return_sequences=True, kernel_regularizer=l2(0.005))),
    Dropout(0.3),
    Bidirectional(LSTM(32, activation="relu", kernel_regularizer=l2(0.005))),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

# Train model
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)],
    verbose=1
)

# =============================================
# MODEL EVALUATION
# =============================================

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

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

# Plot training set with CI
## First ensure you have this import at the top
import matplotlib.dates as mdates

# Calculate the correct date ranges
all_dates = df.index[seq_length:]  # Skip first 'seq_length' points since we're using lags

# Split dates to match your train/test split
train_size = int(len(X_seq) * 0.8)  # Same split as used for your data
train_dates = all_dates[:train_size]
test_dates = all_dates[train_size:]

# Now plot training set with dates
plt.figure(figsize=(14, 6))
plt.plot(train_dates, y_train_act, label="Actual", color="blue")
plt.plot(train_dates, y_train_pred_act, label="Predicted", color="red", linestyle="--")
plt.fill_between(train_dates,
                 y_train_pred_act - train_ci,
                 y_train_pred_act + train_ci,
                 color="gray", alpha=0.3, label="95% CI")
plt.title("Training Set: Actual vs Predicted with Confidence Interval", pad=20, fontsize=16)

# Format x-axis
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every 3 months
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Dengue Cases", fontsize=14)
plt.legend(loc='upper right', fontsize=12, frameon=True)
plt.tight_layout()
plt.show()

# Plot test set with dates
plt.figure(figsize=(14, 6))
plt.plot(test_dates, y_test_act, label="Actual", color="blue")
plt.plot(test_dates, y_test_pred_act, label="Predicted", color="red", linestyle="--")
plt.fill_between(test_dates,
                 y_test_pred_act - test_ci,
                 y_test_pred_act + test_ci,
                 color="gray", alpha=0.3, label="95% CI")
plt.title("Test Set: Actual vs Predicted with Confidence Interval", pad=20, fontsize=16)

# Format x-axis
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Show every month
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

print("\nModel Performance Metrics:")
print(metrics_df.round(3))

# =============================================
# FORECASTING WITH CONFIDENCE INTERVALS
# =============================================
def forecast_future(model, last_sequence, n_steps, n_features, df, variables_to_lag, lags):
    """Generate future forecasts while maintaining variability in all features"""
    forecasts = []
    current_sequence = last_sequence.copy()
    
    # Create a history DataFrame to store our predictions and feature updates
    history_df = df.copy()
    
    for step in range(n_steps):
        # Get prediction (1x1 array)
        current_pred = model.predict(current_sequence.reshape(1, seq_length, n_features))
        forecasts.append(current_pred[0, 0])
        
        # Create a new row for our "future" data
        new_row = {}
        
        # 1. Update the target variable (Cases)
        new_row['Cases'] = current_pred[0, 0]
        
        # 2. For other features, we need to either:
        #    - Use their typical seasonal patterns (if we don't have future values)
        #    - Or use actual future values if available
        
        # For this example, we'll use the historical median values for future features
        # In a real application, you might want to:
        # - Use actual weather forecasts if available
        # - Use seasonal averages
        # - Use some other estimation method
        
        # Get median values for each month to maintain seasonality
        month = (history_df.index[-1].month + 1) if history_df.index[-1].month < 12 else 1
        monthly_medians = history_df[history_df.index.month == month].median()
        
        # Update the features with median values (maintaining seasonality)
        for var in variables_to_lag:
            new_row[var] = monthly_medians[var]
        
        # Add the new row to our history
        new_date = history_df.index[-1] + pd.DateOffset(months=1)
        new_row_df = pd.DataFrame(new_row, index=[new_date])
        history_df = pd.concat([history_df, new_row_df])
        
        # Update lag features in the history
        for var in variables_to_lag:
            for lag in range(1, lags + 1):
                history_df[f"{var}_lag_{lag}"] = history_df[var].shift(lag)
        
        # Drop the oldest row to maintain same length
        history_df = history_df.iloc[1:]
        
        # Prepare the new sequence from the updated history
        scaled_data = scaler.transform(history_df)
        X = scaled_data[:, 1:]  # All except Cases
        new_sequence = X[-seq_length:]  # Take the last seq_length steps
        
        current_sequence = new_sequence
    
    return np.array(forecasts)

# Get the last sequence from our data
last_sequence = X_seq[-1]
n_features = X_train.shape[2]

# Generate 12-month forecast with variability
forecast_steps = 8
forecast = forecast_future(model, last_sequence, forecast_steps, n_features, df, variables_to_lag, lags)

# Inverse transform the forecast
dummy_array = np.zeros((len(forecast), X.shape[1] + 1))
dummy_array[:, 0] = forecast
forecast_actual = scaler.inverse_transform(dummy_array)[:, 0]

# Create future dates for plotting
last_date = df.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')

# Plot the forecast with historical data

# Filter historical data from 2024
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Historical data from 2024 onwards
df_2024 = df[df.index >= pd.Timestamp("2024-01-01")]
last_hist_date = df.index[-1]
last_hist_value = df['Cases'].iloc[-1]

# Forecast data from 2024 onwards
forecast_mask_2024 = forecast_dates >= pd.Timestamp("2024-01-01")
forecast_dates_2024 = forecast_dates[forecast_mask_2024]
forecast_actual_2024 = forecast_actual[forecast_mask_2024]

# Insert last historical point at the start of forecast
extended_forecast_dates = np.insert(forecast_dates_2024, 0, last_hist_date)
extended_forecast_values = np.insert(forecast_actual_2024, 0, last_hist_value)

# Std deviation for confidence interval
historical_std = df['Cases'].std()
extended_lower = extended_forecast_values - historical_std
extended_upper = extended_forecast_values + historical_std
# =============================================

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(df_2024.index, df_2024['Cases'], label='Historical Data (from 2024)', color='blue', linewidth=5)

# Forecast line with connecting point
plt.plot(extended_forecast_dates, extended_forecast_values, 
         label='Forecast', color='red', marker='o', linewidth=5)

# Confidence band
plt.fill_between(extended_forecast_dates,
                 extended_lower,
                 extended_upper,
                 color='pink', alpha=0.3, label='Expected Range')

plt.title('Dengue Cases Forecast from 2025 with Variability', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Dengue Cases', fontsize=16)
#plt.legend()
plt.legend(loc='upper center', fontsize=16, frameon=True, borderpad=1)
plt.tight_layout()
plt.show()

# Forecast table from 2024 (excluding Dec 2023 historical point)
forecast_df_2024 = pd.DataFrame({
    'Date': forecast_dates_2024,
    'Forecasted Cases': forecast_actual_2024,
    'Lower Bound': forecast_actual_2024 - historical_std,
    'Upper Bound': forecast_actual_2024 + historical_std
})

print("\nForecast from 2024 with Expected Variability:")
print(forecast_df_2024.round(1))


# =============================================
# REGRESSION PLOTS FOR ALL VARIABLES
# =============================================

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
axes = axes.flatten()

for i, var in enumerate(['RainFl', 'RainDy', 'Temp', 'Rhumid', 'PI', 'CI']):
    sns.regplot(x=df[var], y=df['Cases'], ax=axes[i], scatter_kws={'alpha':0.5})
    axes[i].set_title(f"Cases vs {var}")
    axes[i].set_xlabel(var)
    axes[i].set_ylabel("Cases")
    
    # Calculate regression stats
    slope, intercept, r_value, p_value, _ = stats.linregress(df[var], df['Cases'])
    axes[i].annotate(f"R² = {r_value**2:.2f}\np = {p_value:.3f}", 
                    xy=(0.7, 0.9), xycoords='axes fraction')

# Remove empty subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Regression Analysis of Dengue Cases vs Environmental Variables", y=1.02)
plt.tight_layout()
plt.show()



# =============================================
# SCENARIO-BASED FORECASTING WITH PI/CI ADJUSTMENTS
# =============================================

def forecast_with_scenarios(model, last_sequence, n_steps, n_features, df, 
                          variables_to_lag, lags, reduction_scenarios):
    """Generate forecasts for different PI/CI reduction scenarios"""
    scenario_forecasts = {}
    
    for scenario_name, reduction in reduction_scenarios.items():
        print(f"\nProcessing scenario: {scenario_name} (PI/CI reduction: {reduction*100}%)")
        
        forecasts = []
        current_sequence = last_sequence.copy()
        scenario_df = df.copy()
        
        for step in range(n_steps):
            # Get prediction
            current_pred = model.predict(current_sequence.reshape(1, seq_length, n_features))
            forecasts.append(current_pred[0, 0])
            
            # Create new row for our future data
            new_row = {}
            new_row['Cases'] = current_pred[0, 0]
            
            # Get median values for each month (maintaining seasonality)
            month = (scenario_df.index[-1].month + 1) if scenario_df.index[-1].month < 12 else 1
            monthly_medians = scenario_df[scenario_df.index.month == month].median()
            
            # Update features with median values, applying reduction to PI/CI
            for var in variables_to_lag:
                if var in ['PI', 'CI']:
                    # Apply reduction to PI and CI
                    new_row[var] = monthly_medians[var] * (1 - reduction)
                else:
                    new_row[var] = monthly_medians[var]
            
            # Add the new row to our scenario data
            new_date = scenario_df.index[-1] + pd.DateOffset(months=1)
            new_row_df = pd.DataFrame(new_row, index=[new_date])
            scenario_df = pd.concat([scenario_df, new_row_df])
            
            # Update lag features
            for var in variables_to_lag:
                for lag in range(1, lags + 1):
                    scenario_df[f"{var}_lag_{lag}"] = scenario_df[var].shift(lag)
            
            scenario_df = scenario_df.iloc[1:]  # Maintain same length
            
            # Prepare new sequence
            scaled_data = scaler.transform(scenario_df)
            X = scaled_data[:, 1:]  # All except Cases
            new_sequence = X[-seq_length:]  # Take last seq_length steps
            
            current_sequence = new_sequence
        
        # Inverse transform the forecasts
        dummy_array = np.zeros((len(forecasts), X.shape[1] + 1))
        dummy_array[:, 0] = forecasts
        scenario_forecasts[scenario_name] = scaler.inverse_transform(dummy_array)[:, 0]
    
    return scenario_forecasts

# Define reduction scenarios
reduction_scenarios = {
    'Baseline (0% reduction)': 0.0,
    '20% PI/CI reduction': 0.20,
    '40% PI/CI reduction': 0.40,
    '80% PI/CI reduction':0.80
}

# Generate forecasts for all scenarios
scenario_forecasts = forecast_with_scenarios(
    model, last_sequence, forecast_steps, n_features, 
    df, variables_to_lag, lags, reduction_scenarios
)


# =============================================
# SCENARIO FORECAST PLOT 
# =============================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Filter forecast data from 2024 onwards
forecast_2024_mask = forecast_dates >= pd.Timestamp("2024-01-01")
forecast_dates_2024 = forecast_dates[forecast_2024_mask]

# Historical data from 2024
df_2024 = df[df.index >= pd.Timestamp("2024-01-01")]
last_hist_date = df.index[-1]
last_hist_value = df['Cases'].iloc[-1]

# Use a high-contrast color palette
from matplotlib.cm import get_cmap
palette = get_cmap('tab10')  # or try 'Set1' for even more contrast
colors = palette(np.linspace(0, 1, len(scenario_forecasts)))

# Plotting
plt.figure(figsize=(16, 8))
plt.plot(df_2024.index, df_2024['Cases'], label='Historical Data (from 2024)', color='black', linewidth=5)

for (scenario_name, forecast), color in zip(scenario_forecasts.items(), colors):
    # Forecast values from 2024 onwards
    forecast_array = np.array(forecast)
    forecast_2024 = forecast_array[forecast_2024_mask]

    # Prepend the last historical point
    extended_dates = np.insert(forecast_dates_2024, 0, last_hist_date)
    extended_forecast = np.insert(forecast_2024, 0, last_hist_value)

    # Plot the line
    plt.plot(extended_dates, extended_forecast, label=scenario_name, 
             color=color, linestyle='--' if 'reduction' in scenario_name else '-', linewidth=5)
    
#    # Add markers for reduction scenarios only (skip baseline)
#    if 'reduction' in scenario_name:
#        plt.scatter(forecast_dates_2024, forecast_2024, color=color, s=50)

plt.title('Dengue Forecast from 2024 with PI/CI Reduction Scenarios', fontsize=16)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Dengue Cases', fontsize=18)

# Place the legend inside the graph box (in the upper left corner, for example)
plt.legend(loc='upper center', fontsize=16, frameon=True, borderpad=1)

plt.tight_layout()

#plt.figtext(0.5, -0.1,
#           "Note: Forecasts begin one month after the last available data point. Connecting lines added for continuity.\n"
#           "Scenarios with PI and CI reduction are dotted with markers.",
#           ha="center", fontsize=10, wrap=True)

plt.show()

# =============================================
# 1. PERFORMANCE STATISTICS TABLE
# =============================================

# Create a prettier performance table
from tabulate import tabulate

metrics_df = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "R²", "MAPE (%)"],
    "Training Set": [
        f"{train_metrics['RMSE']:.2f}",
        f"{train_metrics['MAE']:.2f}",
        f"{train_metrics['R²']:.3f}",
        f"{train_metrics['MAPE']:.2f}"
    ],
    "Test Set": [
        f"{test_metrics['RMSE']:.2f}",
        f"{test_metrics['MAE']:.2f}",
        f"{test_metrics['R²']:.3f}",
        f"{test_metrics['MAPE']:.2f}"
    ]
})

plt.figure(figsize=(8, 3))
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

table = plt.table(
    cellText=metrics_df.values,
    colLabels=metrics_df.columns,
    cellLoc='center',
    loc='center',
    colColours=['#f7f7f7']*3
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8)
plt.title("Model Performance Statistics", y=1.1, fontsize=14)
plt.tight_layout()
plt.show()




# Create DataFrame for metrics
metrics_df = pd.DataFrame({
    "RMSE": [train_metrics["RMSE"], test_metrics["RMSE"]],
    "MAE": [train_metrics["MAE"], test_metrics["MAE"]],
    "R²": [train_metrics["R²"], test_metrics["R²"]],
    "MAPE": [train_metrics["MAPE"], test_metrics["MAPE"]],
}, index=["Training Set", "Test Set"])

# Format and print
print("LSTM Model Performance Metrics:\n")
print(metrics_df.round(3).to_string())



import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import shapiro, ttest_rel
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# 1. K-fold Cross Validation (K=5)
def kfold_lstm_cv(X_train, y_train, n_splits=5, epochs=100, batch_size=32):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []
    residuals = []
    
    for train_index, val_index in tscv.split(X_train):
        # Split data
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]
        
        # Build LSTM model
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True, 
                               kernel_regularizer=l2(0.005))),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=True,
                               kernel_regularizer=l2(0.005))),
            Dropout(0.3),
            Bidirectional(LSTM(32, kernel_regularizer=l2(0.005))),
            Dropout(0.3),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Train model
        history = model.fit(
            X_tr, y_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
            verbose=0
        )
        
        # Evaluate
        val_loss = model.evaluate(X_val, y_val, verbose=0)
        fold_scores.append(val_loss)
        
        # Store residuals
        y_pred = model.predict(X_val)
        residuals.extend((y_val - y_pred.flatten()).tolist())
    
    print(f"K-Fold CV Results - Mean RMSE: {np.mean(np.sqrt(fold_scores)):.4f} ± {np.std(np.sqrt(fold_scores)):.4f}")
    return np.array(residuals)

# Run K-fold CV
lstm_residuals = kfold_lstm_cv(X_train, y_train)

# 2. Residual Analysis
def analyze_residuals(residuals):
    # Plot residuals
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(residuals)
    plt.title('Residuals Over Time')
    plt.xlabel('Time')
    plt.ylabel('Residual')
    
    plt.subplot(1, 3, 2)
    plt.hist(residuals, bins=30)
    plt.title('Residual Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    
    # ACF of residuals
    plt.subplot(1, 3, 3)
    acf_vals = acf(residuals, nlags=20)
    plt.stem(acf_vals)
    plt.axhspan(-1.96/np.sqrt(len(residuals)), 1.96/np.sqrt(len(residuals)), 
                alpha=0.1, color='blue')
    plt.title('ACF of Residuals')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    
    plt.tight_layout()
    plt.show()
    
    # Ljung-Box test for autocorrelation
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    print("\nLjung-Box Test for Residual Autocorrelation:")
    print(lb_test)
    
    # Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = shapiro(residuals)
    print(f"\nShapiro-Wilk Test for Normality: stat={shapiro_stat:.4f}, p={shapiro_p:.4f}")

analyze_residuals(lstm_residuals)

# 3. Diebold-Mariano Test (simplified implementation)
def diebold_mariano_test(actual, pred1, pred2, horizon=1):
    e1 = actual - pred1
    e2 = actual - pred2
    d = e1**2 - e2**2  # Using squared error
    
    # DM test statistic
    dm_stat = np.mean(d) / np.sqrt(np.var(d)/len(d))
    p_value = 2 * (1 - DescrStatsW(d).tconfint_mean()[1](abs(dm_stat)))
    
    print(f"\nDiebold-Mariano Test Results:")
    print(f"DM Statistic: {dm_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Conclusion: Significant difference in forecasting accuracy")
    else:
        print("Conclusion: No significant difference in forecasting accuracy")

# Example usage (replace with your actual predictions)
# diebold_mariano_test(y_test, lstm_preds, arima_preds)

# 4. VIF Analysis
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = [f"Feature {i}" for i in range(X.shape[1])]
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    return vif_data

# Assuming X_train is your feature matrix (2D array)
print("\nVIF Analysis:")
print(calculate_vif(X_train.reshape(X_train.shape[0], -1)))  # Flatten time steps for VIF



import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense

# Common issues in LSTM AIC/BIC calculation:
def calculate_bilstm_aic_bic(model, X, y):
    # Get number of parameters (ONLY trainable)
    trainable_params = sum([np.prod(w.shape) for w in model.trainable_weights])
    
    # Get predictions
    y_pred = model.predict(X).flatten()
    
    # Calculate MSE (ensure matching shapes)
    mse = mean_squared_error(y, y_pred)
    n = len(y)
    
    # CORRECTED AIC/BIC calculation:
    aic = n * np.log(mse) + 2 * trainable_params
    bic = n * np.log(mse) + np.log(n) * trainable_params
    
    return aic, bic

# Example usage:
bilstm_model = Sequential([
    Bidirectional(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]))),
    Dense(1)
])
bilstm_model.compile(optimizer='adam', loss='mse')
bilstm_model.fit(X_train, y_train, epochs=10, verbose=0)

bilstm_aic, bilstm_bic = calculate_bilstm_aic_bic(bilstm_model, X_test, y_test)
print(f"BiLSTM AIC: {bilstm_aic:.2f}, BIC: {bilstm_bic:.2f}")