"""
Quick Demo LSTM Time Series Forecaster for Non-Linear Stock Data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import os
import glob

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

def load_sample_data(data_path, max_files=3):
    """Load a sample of stock data files"""
    files = glob.glob(os.path.join(data_path, "*.csv"))[:max_files]
    
    data_list = []
    for file in files:
        try:
            df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df['Symbol'] = os.path.basename(file).replace('.csv', '')
            
            # Basic feature engineering
            df['Returns'] = df['Close'].pct_change()
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Volume_MA'] = df['Volume'].rolling(window=5).mean()
            
            # Remove NaN values
            df.dropna(inplace=True)
            data_list.append(df)
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return pd.concat(data_list, ignore_index=True) if data_list else None

def create_sequences(data, features, target_col, sequence_length=30):
    """Create sequences for LSTM training"""
    scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    X_list, y_list = [], []
    
    for symbol in data['Symbol'].unique():
        symbol_data = data[data['Symbol'] == symbol].copy()
        
        if len(symbol_data) < sequence_length + 1:
            continue
        
        # Scale data
        feature_data = symbol_data[features].values
        target_data = symbol_data[target_col].values.reshape(-1, 1)
        
        # Check for infinite values
        if not np.all(np.isfinite(feature_data)) or not np.all(np.isfinite(target_data)):
            continue
        
        scaled_features = scaler.fit_transform(feature_data)
        scaled_target = target_scaler.fit_transform(target_data)
        
        # Create sequences
        for i in range(sequence_length, len(scaled_features)):
            X_list.append(scaled_features[i-sequence_length:i])
            y_list.append(scaled_target[i])
    
    return np.array(X_list), np.array(y_list), scaler, target_scaler

def build_lstm_model(input_shape, units=50, dropout_rate=0.2):
    """Build a simple LSTM model"""
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units//2),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def main():
    print("=== Quick LSTM Demo ===")
    
    # Load sample data
    print("Loading sample data...")
    train_data = load_sample_data("stocks/train", max_files=3)
    
    if train_data is None:
        print("No data loaded!")
        return
    
    print(f"Loaded data shape: {train_data.shape}")
    print(f"Symbols: {train_data['Symbol'].unique()}")
    
    # Features to use
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'High_Low_Ratio']
    target_col = 'Close'
    
    # Create sequences
    print("Creating sequences...")
    X, y, scaler, target_scaler = create_sequences(train_data, features, target_col, sequence_length=30)
    
    print(f"Sequences shape: {X.shape}, Targets shape: {y.shape}")
    
    if len(X) == 0:
        print("No sequences created!")
        return
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training: {X_train.shape}, Testing: {X_test.shape}")
    
    # Build model
    print("Building LSTM model...")
    model = build_lstm_model((X.shape[1], X.shape[2]))
    print(model.summary())
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,  # Quick training
        batch_size=32,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )
    
    # Evaluate
    print("Evaluating model...")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Inverse transform predictions
    train_pred = target_scaler.inverse_transform(train_pred)
    test_pred = target_scaler.inverse_transform(test_pred)
    y_train_actual = target_scaler.inverse_transform(y_train)
    y_test_actual = target_scaler.inverse_transform(y_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train_actual, train_pred)
    test_mse = mean_squared_error(y_test_actual, test_pred)
    train_r2 = r2_score(y_train_actual, train_pred)
    test_r2 = r2_score(y_test_actual, test_pred)
    
    print(f"\nResults:")
    print(f"Training MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
    print(f"Testing MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()
    
    # Training predictions
    plt.subplot(2, 3, 2)
    sample_size = min(200, len(y_train_actual))
    plt.plot(y_train_actual[:sample_size], label='Actual', alpha=0.7)
    plt.plot(train_pred[:sample_size], label='Predicted', alpha=0.7)
    plt.title('Training Predictions')
    plt.legend()
    
    # Test predictions
    plt.subplot(2, 3, 3)
    sample_size = min(200, len(y_test_actual))
    plt.plot(y_test_actual[:sample_size], label='Actual', alpha=0.7)
    plt.plot(test_pred[:sample_size], label='Predicted', alpha=0.7)
    plt.title('Test Predictions')
    plt.legend()
    
    # Scatter plots
    plt.subplot(2, 3, 4)
    plt.scatter(y_train_actual, train_pred, alpha=0.5)
    plt.plot([y_train_actual.min(), y_train_actual.max()], 
             [y_train_actual.min(), y_train_actual.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Training: Actual vs Predicted')
    
    plt.subplot(2, 3, 5)
    plt.scatter(y_test_actual, test_pred, alpha=0.5)
    plt.plot([y_test_actual.min(), y_test_actual.max()], 
             [y_test_actual.min(), y_test_actual.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Test: Actual vs Predicted')
    
    # Future forecast demo
    plt.subplot(2, 3, 6)
    last_sequence = X_test[-1:].copy()
    future_steps = 10
    forecasts = []
    
    for _ in range(future_steps):
        next_pred = model.predict(last_sequence, verbose=0)
        forecasts.append(next_pred[0, 0])
        
        # Simple update (in practice, you'd need more sophisticated feature updating)
        new_step = last_sequence[0, -1:].copy()
        new_step[0, features.index(target_col)] = next_pred[0, 0]
        last_sequence = np.concatenate([last_sequence[:, 1:], new_step.reshape(1, 1, -1)], axis=1)
    
    # Inverse transform forecasts
    forecasts = target_scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))
    
    plt.plot(range(future_steps), forecasts, 'r-', marker='o', label='Forecast')
    plt.title('Future Forecast (10 steps)')
    plt.xlabel('Future Steps')
    plt.ylabel('Price')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('lstm_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save model
    model.save('lstm_stock_forecaster.h5')
    print(f"\nModel saved as 'lstm_stock_forecaster.h5'")
    print(f"Results plot saved as 'lstm_results.png'")
    
    # Print forecast values
    print(f"\nFuture forecast (next {future_steps} steps):")
    for i, val in enumerate(forecasts.flatten()):
        print(f"Step {i+1}: ${val:.2f}")

if __name__ == "__main__":
    main()
