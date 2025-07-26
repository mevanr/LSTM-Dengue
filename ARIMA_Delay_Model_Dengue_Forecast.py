import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA

# Set random seed
SEED = 7
np.random.seed(SEED)

# Load and prepare data
data_path = r"Rain_Data_Case_Dependent_Imputation.txt"
df = pd.read_csv(data_path, sep="\t", header=0)
df.columns = ["Year", "Month", "Cases", "RainFl", "RainDy", "Temp", "Rhumid", "PrIndex", "CnIndex"]
df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(Day=1))
df.set_index("Date", inplace=True)
ts = df['Cases']

# Split data
train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]

# Auto ARIMA model selection
auto_model = auto_arima(train, 
                      seasonal=False,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True,
                      random_state=SEED)

print(auto_model.summary())
best_order = auto_model.order

# Fit model on training data
model_train = ARIMA(train, order=best_order)
fitted_model = model_train.fit()

# Get in-sample predictions with confidence intervals
train_pred = fitted_model.get_prediction(start=train.index[0], end=train.index[-1])
train_pred_mean = train_pred.predicted_mean
train_conf_int = train_pred.conf_int()

# Test set predictions (Rolling Forecast with CI)
history = list(train)
test_pred_mean = []
test_conf_int_lower = []
test_conf_int_upper = []

for t in range(len(test)):
    model = ARIMA(history, order=best_order)
    model_fit = model.fit()
    
    # Get forecast with confidence interval
    forecast = model_fit.get_forecast(steps=1)
    test_pred_mean.append(forecast.predicted_mean[0])
    conf_int = forecast.conf_int()[0]  # Get as numpy array
    test_conf_int_lower.append(conf_int[0])
    test_conf_int_upper.append(conf_int[1])
    
    history.append(test[t])

# Convert to pandas Series/DataFrame for plotting
test_pred_mean = pd.Series(test_pred_mean, index=test.index)
test_conf_int = pd.DataFrame({
    'lower': test_conf_int_lower,
    'upper': test_conf_int_upper
}, index=test.index)


# Plot for Training Set Fit with CI
plt.figure(figsize=(14, 6))
plt.plot(train.index, train, label='Actual', color='blue')
plt.plot(train.index, train_pred_mean, label='Fitted', color='red', linestyle='--')
plt.fill_between(train.index, 
                train_conf_int.iloc[:, 0], 
                train_conf_int.iloc[:, 1],
                color='gray', alpha=0.3, label='95% Cofidence Interval')
plt.title(f'ARIMA{best_order} - Training Set Fit', fontsize=14)
plt.legend()
plt.show()

# Plot for Test Set Predictions with CI
plt.figure(figsize=(14, 6))
plt.plot(test.index, test, label='Actual', color='blue')
plt.plot(test.index, test_pred_mean, label='Predicted', color='red', linestyle='--')
plt.fill_between(test.index, 
                test_conf_int['lower'], 
                test_conf_int['upper'],
                color='gray', alpha=0.3, label='95% Cofidence Interval')
plt.title(f'ARIMA{best_order} - Test Set Predictions', fontsize=14)
plt.legend()
plt.show()


# Metrics calculation
def calculate_metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

train_metrics = calculate_metrics(train, train_pred_mean)
test_metrics = calculate_metrics(test, test_pred_mean)

metrics_df = pd.DataFrame({
    "Training Set": train_metrics,
    "Test Set": test_metrics
}).T

print("\nARIMA Model Performance Metrics:")
print(metrics_df.round(3))

# Final forecast
final_model = ARIMA(ts, order=best_order)
final_fit = final_model.fit()

forecast_steps = 12
forecast = final_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

future_dates = pd.date_range(ts.index[-1], periods=forecast_steps+1, freq='M')[1:]

plt.figure(figsize=(14, 7))
plt.plot(ts.index, ts, label='Historical Data', color='blue')
plt.plot(future_dates, forecast_mean, label='Forecast', color='red')
plt.fill_between(future_dates, 
                forecast_conf_int.iloc[:, 0], 
                forecast_conf_int.iloc[:, 1],
                color='pink', alpha=0.3, label='95% CI')

plt.axvline(x=ts.index[-1], color='gray', linestyle='--')
plt.title(f'{forecast_steps}-Month Forecast (ARIMA{best_order})', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def calculate_arima_aic_bic(model_fit):
    # ARIMA model directly provides AIC/BIC
    return model_fit.aic, model_fit.bic

# Example usage:
# Calculate AIC/BIC for the fitted ARIMA model
arima_aic, arima_bic = calculate_arima_aic_bic(fitted_model)  # Use the already fitted model
print(f"\nARIMA Model Information Criteria:")
print(f"AIC: {arima_aic:.2f}")
print(f"BIC: {arima_bic:.2f}")

