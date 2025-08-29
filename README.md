# Nonlinear ARMA (NLARMA) Model for Time Series Forecasting

This repository presents a complete implementation, evaluation, and comparison of **Nonlinear ARMA (NLARMA)** models against traditional **ARMA** models for time series forecasting. The NLARMA approach uses a neural network (MLP) to model complex nonlinear dependencies that classical linear models like ARMA often fail to capture.

## Project Structure

- `narma.py` — Python script implementing the full NLARMA model and comparing it to ARMA using simulated stock price data.
- `simulated_stock_prices.csv` — Simulated dataset with 4 interpretable columns: time, external index, volatility signal, and simulated stock price.
- `presentation/` — LaTeX Beamer presentation slides (PDF & `.tex`) explaining NLARMA, real-world applications, math formulation, and model performance.
- `plots/` — Output graphs comparing predicted vs. actual prices.

## Model Summary

### Traditional ARMA:
- Assumes **linear** relationships.
- Suitable for stationary, low-complexity series.

### Nonlinear ARMA (NARMA):
- Utilizes **MLPRegressor** (Neural Network).
- Captures **nonlinear** dynamics and volatility.
- Provides significantly better RMSE on nonlinear simulated stock data.

## How to Run

Make sure you have Python 3 installed. Then:

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels
python3 narma.py
