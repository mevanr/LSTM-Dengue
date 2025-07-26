# -*- coding: utf-8 -*-
"""
Past PI/CI Imputation Using Rainfall Data Only
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess

import tensorflow as tf
from tensorflow.keras.utils import set_random_seed

# Set all random seeds for reproducibility
SEED = 7  # 10 #3 # 10 #42  # 10  7 best
np.random.seed(SEED)
tf.random.set_seed(SEED)
set_random_seed(SEED)  # For Keras operations

# =============================================
# DATA LOADING AND PREPROCESSING
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


def prepare_data(data_path):
    """Load and prepare the dengue dataset focusing on rainfall"""
    df = pd.read_csv(data_path, sep="\t", header=0)
    df.columns = ["Year", "Month", "Cases", "RainFl", "RainDy", 
                "Temp", "Rhumid", "PI", "CI"]
    
    # Convert to numeric and handle missing values
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create datetime index and sort
    df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(day=1))
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    
    # Add rainfall interaction term
    df['RainInteraction'] = df['RainFl'] * df['RainDy']
    
    return df

def impute_past_values(df, target_col, cutoff_date='2020-01-01', 
                      rain_features=["RainFl", "RainDy", "RainInteraction"]):
    """
    Impute only past missing values before cutoff date using rainfall data
    
    Parameters:
    - df: DataFrame with the data
    - target_col: 'PI' or 'CI' to impute
    - cutoff_date: Only impute values before this date
    - rain_features: List of rainfall-related features to use
    
    Returns:
    - DataFrame with imputed values only for past missing data
    """
    # Make copy to avoid modifying original
    df = df.copy()
    
    # Identify past missing values (before cutoff date)
    past_missing = df[(df[target_col].isna()) & (df.index < cutoff_date)]
    
    if len(past_missing) == 0:
        print(f"No past missing values found for {target_col} before {cutoff_date}")
        return df
    
    # Get known values (can include recent data for training)
    known = df[df[target_col].notna()]
    
    if len(known) < 10:  # Need sufficient training data
        print(f"Insufficient training data ({len(known)} points) for {target_col}")
        return df
    
    # Prepare training data
    X_train = known[rain_features].values
    y_train = known[target_col].values
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=200, random_state=42, oob_score=True)
    model.fit(X_train_scaled, y_train)
    
    # Prepare prediction data (past missing values)
    X_missing = past_missing[rain_features].values
    X_missing_scaled = scaler.transform(X_missing)
    
    # Predict missing values
    y_pred = model.predict(X_missing_scaled)
    
    # Calculate uncertainty using OOB error
    oob_error = y_train - model.oob_prediction_
    uncertainty = 1.5 * np.std(oob_error)
    
    # Fill only the past missing values
    df.loc[past_missing.index, f"{target_col}_imputed"] = y_pred
    df.loc[past_missing.index, f"{target_col}_lower"] = y_pred - uncertainty
    df.loc[past_missing.index, f"{target_col}_upper"] = y_pred + uncertainty
    
    # Keep original values where available
    df[f"{target_col}_final"] = df[target_col].combine_first(df[f"{target_col}_imputed"])
    
    return df

def plot_rainfall_imputation(df, target_col):
    """Plot the rainfall-based imputation results"""
    plt.figure(figsize=(14, 7))
    
    # Plot final values (observed + imputed)
    plt.plot(df.index, df[f"{target_col}_final"], 
            color='blue', label='Final', linewidth=1.5)
    
    # Plot original observed values
    observed = df[df[target_col].notna()]
    plt.scatter(observed.index, observed[target_col],
               color='green', label='Observed', s=40)
    
    # Plot imputed values with uncertainty
    imputed = df[df[f"{target_col}_imputed"].notna()]
    plt.scatter(imputed.index, imputed[f"{target_col}_imputed"],
               color='red', label='Imputed', s=60, marker='x')
    
    # Add uncertainty range
    plt.fill_between(imputed.index, 
                    imputed[f"{target_col}_lower"], 
                    imputed[f"{target_col}_upper"],
                    color='orange', alpha=0.2, label='Uncertainty')
    
    # Add LOWESS smoothed trend
    lowess_fit = lowess(df[f"{target_col}_final"], 
                       pd.to_numeric(df.index), frac=0.1)
    plt.plot(df.index, lowess_fit[:, 1], 
            color='purple', linestyle='--', label='Trend')
    
    plt.title(f"Past {target_col} Imputation Using Rainfall Data")
    plt.ylabel(target_col)
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

def main(data_path, output_path, cutoff_date='2020-01-01'):
    """Main processing function"""
    # Load and prepare data
    df = prepare_data(data_path)
    
    # Impute only past missing values for both indices
    for target_col in ["PI", "CI"]:
        df = impute_past_values(df, target_col, cutoff_date)
        
        # Generate and save plots
        plot = plot_rainfall_imputation(df, target_col)
        plot.savefig(f"{target_col}_rainfall_imputation.png", dpi=300)
        plot.show()
    
    # Save results (only relevant columns)
    output_cols = ["Year", "Month", "RainFl", "RainDy", 
                  "PI", "PI_final", "PI_lower", "PI_upper",
                  "CI", "CI_final", "CI_lower", "CI_upper"]
    
    df[output_cols].to_csv(output_path, index=False)
    print(f"Processing complete. Results saved to {output_path}")
    
    return df

if __name__ == "__main__":
    input_file = "Data_dengue_2015_2024.txt"
    output_file = "Past_PI_CI_Rainfall_Imputation.csv"
    
    # Set cutoff date for what you consider "past" data
    result_df = main(input_file, output_file, cutoff_date='2023-01-01')