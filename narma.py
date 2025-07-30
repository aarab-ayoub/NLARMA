import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# --- 1. Load and Prepare the Data ---

# Load the dataset
try:
    df = pd.read_csv('simulated_stock_prices.csv')
except FileNotFoundError:
    print("Make sure 'simulated_stock_prices.csv' is in the same directory.")
    exit()


# Use only the 'Simulated_Stock_Price' column and drop any initial missing values
price_series = df['Simulated_Stock_Price'].dropna().to_numpy().reshape(-1, 1)


# --- 2. Feature Engineering: Create Lagged Features ---

def create_dataset(dataset, look_back=5):
    """
    Create a dataset suitable for time series forecasting.
    'look_back' is the number of previous time steps to use as input variables (X)
    to predict the next time period (y).
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Define how many previous steps to use for prediction
look_back = 5
X, y = create_dataset(price_series, look_back)


# --- 3. Split and Scale the Data ---

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale the data for better neural network performance
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")


# --- 4. Build and Train the NARMA-like Neural Network ---

# Initialize the MLPRegressor (our nonlinear function)
# - hidden_layer_sizes: Defines the architecture (2 layers with 100 and 50 neurons)
# - max_iter: Number of epochs for training
# - activation: The activation function for the hidden layers
# - solver: The algorithm for weight optimization
# - random_state: For reproducibility
narma_model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    shuffle=False
)

print("\nTraining the NARMA model...")
# Train the model on the scaled training data
narma_model.fit(X_train_scaled, y_train_scaled.ravel())
print("Training complete.")


# --- 5. Make Predictions and Evaluate ---

# Make predictions on the scaled test data
predictions_scaled = narma_model.predict(X_test_scaled)

# Invert the scaling to get the actual predicted prices
predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))

# Calculate the Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"\nRoot Mean Squared Error (RMSE) on Test Data: {rmse:.4f}")


# create ARMA model for comparison
arima_model = ARIMA(y_train, order=(5, 0, 0))
arima_fit = arima_model.fit()
# Make predictions using the ARIMA model
arima_predictions = arima_fit.forecast(steps=len(y_test))
# Calculate RMSE for ARIMA model
arima_rmse = np.sqrt(mean_squared_error(y_test, arima_predictions))
print(f"ARIMA Model RMSE on Test Data: {arima_rmse:.4f}")

# 

# --- 6. Visualize the Results ---

plt.figure(figsize=(15, 7))
plt.title('NARMA Model: Stock Price Prediction vs. Actual')
plt.ylabel('Simulated Stock Price')
plt.xlabel('Time')
plt.grid(True)

# Plot training data
plt.plot(
    np.arange(len(y_train)),
    y_train,
    label='Training Data',
    color='blue'
)
# Plot actual test data
plt.plot(
    np.arange(len(y_train), len(y_train) + len(y_test)),
    y_test,
    label='Actual Test Data',
    color='green'
)
# Plot predicted test data
plt.plot(
    np.arange(len(y_train), len(y_train) + len(y_test)),
    predictions,
    label='Predicted Data',
    color='red',
    linestyle='--'
)

# plt.legend()
# plt.show()

# visualize ARIMA predictions vs NARMA predictions AND actual test data
plt.figure(figsize=(15, 7))
plt.title('ARIMA vs NARMA Predictions')
plt.ylabel('Simulated Stock Price')
plt.xlabel('Time')
plt.grid(True)
# Plot actual test data
plt.plot(
    np.arange(len(y_train), len(y_train) + len(y_test)),
    y_test,
    label='Actual Test Data',
    color='green'
)
# Plot ARIMA predictions
plt.plot(
    np.arange(len(y_train), len(y_train) + len(y_test)),
    arima_predictions,
    label='ARIMA Predictions',
    color='orange',
    linestyle='--'
)
# Plot NARMA predictions
plt.plot(
    np.arange(len(y_train), len(y_train) + len(y_test)),
    predictions,
    label='NARMA Predictions',
    color='red',
    linestyle='--'
)
plt.legend()
plt.show()
