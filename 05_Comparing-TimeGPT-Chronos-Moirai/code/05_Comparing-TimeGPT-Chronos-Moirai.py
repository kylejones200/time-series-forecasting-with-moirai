import logging

logger = logging.getLogger(__name__)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'axes.grid': False,
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False
})

BASE_DIR = Path(__file__).resolve().parents[1]

# Load energy indicators data (already filtered to Oklahoma in this extracted CSV)
data_path = BASE_DIR / "data" / "energy_indicators.csv"
df = pd.read_csv(data_path)

year_cols = [col for col in df.columns if col.isdigit()]
year_totals = df[year_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0)

ts = pd.Series(
    data=year_totals.values,
    index=pd.to_datetime(year_totals.index, format="%Y"),
).sort_index()

ts = ts.interpolate(method="linear")

logger.info(f"Time series length: {len(ts)}")
logger.info(f"Date range: {ts.index.min()} to {ts.index.max()}")

# Prepare for forecasting
# Use last 20% for testing
test_size = int(len(ts) * 0.2)
ts_train = ts[:-test_size]
ts_test = ts[-test_size:]

logger.info(f"\nTraining: {len(ts_train)} points ({ts_train.index.min()} to {ts_train.index.max()})")
logger.info(f"Testing: {len(ts_test)} points ({ts_test.index.min()} to {ts_test.index.max()})")

# Install: pip install nixtlats

from nixtlats import TimeGPT
import time

# Initialize TimeGPT (requires API key)
# Get key from: https://nixtla.io/
# timegpt = TimeGPT(token="your_api_key_here")

# For demonstration, we'll show the code structure
# In production, you'd use your actual API key

def forecast_timegpt(ts_data, horizon=24, api_key=None):
    """
    Forecast using TimeGPT API.
    
    Note: Requires API key from nixtla.io
    """
    if api_key is None:
        logger.info("TimeGPT requires API key. Skipping...")
        return None
    
    timegpt = TimeGPT(token=api_key)
    
    # Prepare data in required format
    df_prepared = pd.DataFrame({
        'ds': ts_data.index,
        'y': ts_data.values
    })
    
    start_time = time.time()
    
    # Forecast with prediction intervals
    forecast = timegpt.forecast(
        df=df_prepared,
        h=horizon,
        level=[80, 95]  # 80% and 95% prediction intervals
    )
    
    inference_time = time.time() - start_time
    
    return {
        'forecast': forecast['TimeGPT'].values,
        'lower_80': forecast['TimeGPT-lo-80'].values if 'TimeGPT-lo-80' in forecast.columns else None,
        'upper_80': forecast['TimeGPT-hi-80'].values if 'TimeGPT-hi-80' in forecast.columns else None,
        'time': inference_time
    }

# Example usage (commented out - requires API key)
# timegpt_results = forecast_timegpt(ts_train, horizon=len(ts_test))
# logger.info(f"TimeGPT inference time: {timegpt_results['time']:.3f} seconds")

# Install: pip install chronos-forecasting

from chronos import ChronosPipeline
import torch
import time

def forecast_chronos(ts_data, horizon=24, model_size='tiny'):
    """
    Forecast using Chronos.
    
    Model sizes: 'tiny', 'mini', 'small', 'base', 'large'
    """
    # Load pretrained model
    model_name = f"amazon/chronos-t5-{model_size}"
    logger.info(f"Loading Chronos model: {model_name}")
    
    chronos = ChronosPipeline.from_pretrained(
        model_name,
        device_map="cpu",  # or "cuda" for GPU
        torch_dtype=torch.float32
    )
    
    # Prepare context (last portion of training data)
    context_length = min(512, len(ts_data))
    context = torch.tensor(ts_data.values[-context_length:], dtype=torch.float32)
    
    start_time = time.time()
    
    # Forecast
    forecast = chronos.predict(
        context=context,
        prediction_length=horizon,
        num_samples=100  # For uncertainty estimation
    )
    
    inference_time = time.time() - start_time
    
    # Extract median forecast and intervals
    forecast_median = forecast[0].median(dim=0).values.numpy()
    forecast_lower = forecast[0].quantile(0.1, dim=0).values.numpy()
    forecast_upper = forecast[0].quantile(0.9, dim=0).values.numpy()
    
    return {
        'forecast': forecast_median,
        'lower': forecast_lower,
        'upper': forecast_upper,
        'time': inference_time
    }

# Forecast with Chronos
logger.info("Forecasting with Chronos...")
chronos_results = forecast_chronos(ts_train, horizon=len(ts_test), model_size='tiny')
logger.info(f"Chronos inference time: {chronos_results['time']:.3f} seconds")
logger.info(f"Chronos forecast range: {chronos_results['forecast'].min():.2f} to {chronos_results['forecast'].max():.2f}")

# Install: pip install moirai

from moirai import MoiraiForecaster
import time

def forecast_moirai(ts_data, horizon=24, model_size='base'):
    """
    Forecast using Moirai.
    
    Model sizes: 'small', 'base', 'large'
    """
    # Load pretrained model
    model_name = f"Salesforce/moirai-1.0-R-{model_size}"
    logger.info(f"Loading Moirai model: {model_name}")
    
    moirai = MoiraiForecaster.from_pretrained(model_name)
    
    # Prepare context
    context_length = min(512, len(ts_data))
    context = ts_data.values[-context_length:]
    
    start_time = time.time()
    
    # Forecast
    forecast = moirai.forecast(
        past_data=context,
        horizon=horizon,
        num_samples=100  # For uncertainty
    )
    
    inference_time = time.time() - start_time
    
    # Extract statistics
    forecast_median = np.median(forecast, axis=0)
    forecast_lower = np.percentile(forecast, 10, axis=0)
    forecast_upper = np.percentile(forecast, 90, axis=0)
    
    return {
        'forecast': forecast_median,
        'lower': forecast_lower,
        'upper': forecast_upper,
        'time': inference_time
    }

# Forecast with Moirai
logger.info("Forecasting with Moirai...")
moirai_results = forecast_moirai(ts_train, horizon=len(ts_test), model_size='base')
logger.info(f"Moirai inference time: {moirai_results['time']:.3f} seconds")
logger.info(f"Moirai forecast range: {moirai_results['forecast'].min():.2f} to {moirai_results['forecast'].max():.2f}")

from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(actual, predicted):
    """Calculate forecasting metrics"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# Compare models (TimeGPT skipped if no API key)
results = {}

# Chronos
chronos_metrics = calculate_metrics(ts_test.values, chronos_results['forecast'])
results['Chronos'] = {
    **chronos_metrics,
    'Time': chronos_results['time']
}

# Moirai
moirai_metrics = calculate_metrics(ts_test.values, moirai_results['forecast'])
results['Moirai'] = {
    **moirai_metrics,
    'Time': moirai_results['time']
}

# Create comparison table
comparison_df = pd.DataFrame(results).T
logger.info("=== MODEL COMPARISON ===")
logger.info(comparison_df.round(4))

# Create comparison plot
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Forecasts
ax1 = axes[0]

# Historical data
historical_dates = ts_train.index[-30:]
ax1.plot(historical_dates, ts_train.values[-30:], 
         'k-', linewidth=2, label='Historical', alpha=0.7)

# Actual test values
ax1.plot(ts_test.index, ts_test.values, 
         'o-', linewidth=2, markersize=8, label='Actual', color='black')

# Chronos forecast
ax1.plot(ts_test.index, chronos_results['forecast'], 
         '-', linewidth=2, label='Chronos', color='#1f77b4')
ax1.fill_between(ts_test.index, chronos_results['lower'], chronos_results['upper'],
                 alpha=0.2, color='#1f77b4')

# Moirai forecast
ax1.plot(ts_test.index, moirai_results['forecast'], 
         '-', linewidth=2, label='Moirai', color='#ff7f0e')
ax1.fill_between(ts_test.index, moirai_results['lower'], moirai_results['upper'],
                 alpha=0.2, color='#ff7f0e')

ax1.axvline(ts_test.index[0], color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax1.set_title('Foundation Model Forecasts Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Energy Indicator Value', fontsize=11)
ax1.legend(frameon=True, fancybox=True, shadow=True, loc='best')
# Plot 2: Error comparison
ax2 = axes[1]

models = ['Chronos', 'Moirai']
mae_values = [results[m]['MAE'] for m in models]
rmse_values = [results[m]['RMSE'] for m in models]

x = np.arange(len(models))
width = 0.35

ax2.bar(x - width/2, mae_values, width, label='MAE', color='#1f77b4', alpha=0.8)
ax2.bar(x + width/2, rmse_values, width, label='RMSE', color='#ff7f0e', alpha=0.8)

ax2.set_xlabel('Model', fontsize=11)
ax2.set_ylabel('Error', fontsize=11)
ax2.set_title('Error Metrics Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend()
plt.tight_layout()
plt.savefig('foundation_models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
