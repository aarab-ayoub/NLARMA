import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Attention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
import warnings
import os
import glob
from datetime import datetime

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class LSTMTimeSeriesForecaster:
    """
    A comprehensive LSTM model for forecasting non-linear time series data
    """
    
    def __init__(self, sequence_length=60, features=['Close'], target_feature='Close'):
        """
        Initialize the LSTM forecaster
        
        Args:
            sequence_length (int): Number of time steps to look back
            features (list): List of features to use for prediction
            target_feature (str): The target feature to predict
        """
        self.sequence_length = sequence_length
        self.features = features
        self.target_feature = target_feature
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self, train_path, test_path, sample_size=None):
        """
        Load and preprocess the time series data
        
        Args:
            train_path (str): Path to training data directory
            test_path (str): Path to testing data directory
            sample_size (int): Number of files to sample (None for all)
        """
        print("Loading and preprocessing data...")
        
        # Get all CSV files
        train_files = glob.glob(os.path.join(train_path, "*.csv"))
        test_files = glob.glob(os.path.join(test_path, "*.csv"))
        
        if sample_size:
            train_files = train_files[:sample_size]
            test_files = test_files[:min(sample_size//4, len(test_files))]
        
        # Load and combine training data
        train_data_list = []
        for file in train_files:
            try:
                df = pd.read_csv(file)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
                df['Symbol'] = os.path.basename(file).replace('.csv', '')
                train_data_list.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        # Load and combine test data
        test_data_list = []
        for file in test_files:
            try:
                df = pd.read_csv(file)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
                df['Symbol'] = os.path.basename(file).replace('.csv', '')
                test_data_list.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if not train_data_list:
            raise ValueError("No training data loaded!")
        
        self.train_data = pd.concat(train_data_list, ignore_index=True)
        if test_data_list:
            self.test_data = pd.concat(test_data_list, ignore_index=True)
        else:
            self.test_data = None
        
        print(f"Loaded {len(train_files)} training files and {len(test_files)} test files")
        print(f"Training data shape: {self.train_data.shape}")
        if self.test_data is not None:
            print(f"Test data shape: {self.test_data.shape}")
        
        # Feature engineering
        self._feature_engineering()
        
        # Prepare sequences
        self._prepare_sequences()
        
    def _feature_engineering(self):
        """
        Create additional features for better prediction
        """
        print("Engineering features...")
        
        for data in [self.train_data] + ([self.test_data] if self.test_data is not None else []):
            # Technical indicators
            data['Returns'] = data.groupby('Symbol')['Close'].pct_change()
            data['High_Low_Ratio'] = data['High'] / data['Low']
            data['Close_Open_Ratio'] = data['Close'] / data['Open']
            data['Volume_MA'] = data.groupby('Symbol')['Volume'].rolling(window=10).mean().reset_index(0, drop=True)
            data['Price_Volume'] = data['Close'] * data['Volume']
            
            # Moving averages
            data['MA_5'] = data.groupby('Symbol')['Close'].rolling(window=5).mean().reset_index(0, drop=True)
            data['MA_10'] = data.groupby('Symbol')['Close'].rolling(window=10).mean().reset_index(0, drop=True)
            data['MA_20'] = data.groupby('Symbol')['Close'].rolling(window=20).mean().reset_index(0, drop=True)
            
            # Volatility
            data['Volatility'] = data.groupby('Symbol')['Returns'].rolling(window=10).std().reset_index(0, drop=True)
            
            # RSI (Relative Strength Index)
            delta = data.groupby('Symbol')['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            data['BB_Middle'] = data.groupby('Symbol')['Close'].rolling(window=20).mean().reset_index(0, drop=True)
            bb_std = data.groupby('Symbol')['Close'].rolling(window=20).std().reset_index(0, drop=True)
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
            
            # Time-based features
            data['DayOfWeek'] = data['Date'].dt.dayofweek
            data['Month'] = data['Date'].dt.month
            data['Quarter'] = data['Date'].dt.quarter
            
            # Clean data: remove infinite values and NaN
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data.dropna(inplace=True)
            
            # Remove outliers using IQR method for numerical columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['DayOfWeek', 'Month', 'Quarter']:  # Skip categorical-like features
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        # Update feature list if using all features
        if 'all' in self.features:
            numeric_columns = self.train_data.select_dtypes(include=[np.number]).columns
            self.features = [col for col in numeric_columns if col not in ['Date']]
            if self.target_feature not in self.features:
                self.features.append(self.target_feature)
    
    def _prepare_sequences(self):
        """
        Prepare sequences for LSTM training
        """
        print("Preparing sequences...")
        
        # Prepare training sequences
        X_train_list, y_train_list = [], []
        
        # Initialize scalers
        all_feature_data = []
        all_target_data = []
        
        # First pass: collect all data for fitting scalers
        for symbol in self.train_data['Symbol'].unique():
            symbol_data = self.train_data[self.train_data['Symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('Date')
            
            if len(symbol_data) < self.sequence_length + 1:
                continue
            
            # Check if all features exist
            missing_features = [f for f in self.features if f not in symbol_data.columns]
            if missing_features:
                print(f"Warning: Missing features {missing_features} for symbol {symbol}")
                continue
            
            feature_data = symbol_data[self.features].values
            target_data = symbol_data[self.target_feature].values.reshape(-1, 1)
            
            # Check for infinite or NaN values
            if np.any(~np.isfinite(feature_data)) or np.any(~np.isfinite(target_data)):
                print(f"Warning: Non-finite values found for symbol {symbol}, skipping...")
                continue
            
            all_feature_data.append(feature_data)
            all_target_data.append(target_data)
        
        if not all_feature_data:
            raise ValueError("No valid data found for training!")
        
        # Fit scalers on all data
        combined_features = np.vstack(all_feature_data)
        combined_targets = np.vstack(all_target_data)
        
        self.scaler.fit(combined_features)
        self.target_scaler.fit(combined_targets)
        
        # Second pass: create sequences
        for i, symbol in enumerate(self.train_data['Symbol'].unique()):
            symbol_data = self.train_data[self.train_data['Symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('Date')
            
            if len(symbol_data) < self.sequence_length + 1:
                continue
            
            # Check if all features exist
            missing_features = [f for f in self.features if f not in symbol_data.columns]
            if missing_features:
                continue
            
            feature_data = symbol_data[self.features].values
            target_data = symbol_data[self.target_feature].values.reshape(-1, 1)
            
            # Check for infinite or NaN values
            if np.any(~np.isfinite(feature_data)) or np.any(~np.isfinite(target_data)):
                continue
            
            scaled_features = self.scaler.transform(feature_data)
            scaled_target = self.target_scaler.transform(target_data)
            
            # Create sequences
            for j in range(self.sequence_length, len(scaled_features)):
                X_train_list.append(scaled_features[j-self.sequence_length:j])
                y_train_list.append(scaled_target[j])
        
        self.X_train = np.array(X_train_list)
        self.y_train = np.array(y_train_list)
        
        print(f"Training sequences shape: {self.X_train.shape}")
        print(f"Training targets shape: {self.y_train.shape}")
        
        # Prepare test sequences if test data exists
        if self.test_data is not None:
            X_test_list, y_test_list = [], []
            
            for symbol in self.test_data['Symbol'].unique():
                symbol_data = self.test_data[self.test_data['Symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_values('Date')
                
                if len(symbol_data) < self.sequence_length + 1:
                    continue
                
                # Scale features using training scaler
                feature_data = symbol_data[self.features].values
                target_data = symbol_data[self.target_feature].values.reshape(-1, 1)
                
                scaled_features = self.scaler.transform(feature_data)
                scaled_target = self.target_scaler.transform(target_data)
                
                # Create sequences
                for i in range(self.sequence_length, len(scaled_features)):
                    X_test_list.append(scaled_features[i-self.sequence_length:i])
                    y_test_list.append(scaled_target[i])
            
            if X_test_list:
                self.X_test = np.array(X_test_list)
                self.y_test = np.array(y_test_list)
                print(f"Test sequences shape: {self.X_test.shape}")
                print(f"Test targets shape: {self.y_test.shape}")
            else:
                self.X_test = None
                self.y_test = None
    
    def build_model(self, model_type='advanced', lstm_units=[50, 50], dropout_rate=0.2, 
                   learning_rate=0.001, l1_reg=0.01, l2_reg=0.01):
        """
        Build the LSTM model
        
        Args:
            model_type (str): Type of model ('simple', 'bidirectional', 'advanced')
            lstm_units (list): Number of units in each LSTM layer
            dropout_rate (float): Dropout rate
            learning_rate (float): Learning rate for optimizer
            l1_reg (float): L1 regularization
            l2_reg (float): L2 regularization
        """
        print(f"Building {model_type} LSTM model...")
        
        input_shape = (self.sequence_length, len(self.features))
        
        if model_type == 'simple':
            self.model = Sequential([
                LSTM(lstm_units[0], input_shape=input_shape),
                Dropout(dropout_rate),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
        elif model_type == 'bidirectional':
            self.model = Sequential([
                Bidirectional(LSTM(lstm_units[0], return_sequences=True), input_shape=input_shape),
                Dropout(dropout_rate),
                Bidirectional(LSTM(lstm_units[1])),
                Dropout(dropout_rate),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
        elif model_type == 'advanced':
            # Advanced model with attention mechanism
            inputs = Input(shape=input_shape)
            
            # First LSTM layer
            lstm1 = LSTM(lstm_units[0], return_sequences=True, 
                        kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(inputs)
            lstm1 = LayerNormalization()(lstm1)
            lstm1 = Dropout(dropout_rate)(lstm1)
            
            # Second LSTM layer
            lstm2 = LSTM(lstm_units[1], return_sequences=True,
                        kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(lstm1)
            lstm2 = LayerNormalization()(lstm2)
            lstm2 = Dropout(dropout_rate)(lstm2)
            
            # Global average pooling
            pooled = tf.keras.layers.GlobalAveragePooling1D()(lstm2)
            
            # Dense layers
            dense1 = Dense(50, activation='relu', 
                          kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(pooled)
            dense1 = Dropout(dropout_rate)(dense1)
            
            dense2 = Dense(25, activation='relu',
                          kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(dense1)
            dense2 = Dropout(dropout_rate)(dense2)
            
            outputs = Dense(1)(dense2)
            
            self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        print(self.model.summary())
    
    def train_model(self, epochs=5, batch_size=32, validation_split=0.2, patience=15):
        """
        Train the LSTM model
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            validation_split (float): Validation split ratio
            patience (int): Early stopping patience
        """
        print("Training the model...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
            ModelCheckpoint('best_lstm_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        # Train the model
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
    
    def evaluate_model(self):
        """
        Evaluate the model performance
        """
        print("Evaluating model...")
        
        # Predictions on training data
        train_predictions = self.model.predict(self.X_train)
        train_predictions = self.target_scaler.inverse_transform(train_predictions)
        train_actual = self.target_scaler.inverse_transform(self.y_train)
        
        # Calculate training metrics
        train_mse = mean_squared_error(train_actual, train_predictions)
        train_mae = mean_absolute_error(train_actual, train_predictions)
        train_r2 = r2_score(train_actual, train_predictions)
        
        print(f"Training Metrics:")
        print(f"MSE: {train_mse:.4f}")
        print(f"MAE: {train_mae:.4f}")
        print(f"R²: {train_r2:.4f}")
        print(f"RMSE: {np.sqrt(train_mse):.4f}")
        
        # Predictions on test data if available
        if hasattr(self, 'X_test') and self.X_test is not None:
            test_predictions = self.model.predict(self.X_test)
            test_predictions = self.target_scaler.inverse_transform(test_predictions)
            test_actual = self.target_scaler.inverse_transform(self.y_test)
            
            # Calculate test metrics
            test_mse = mean_squared_error(test_actual, test_predictions)
            test_mae = mean_absolute_error(test_actual, test_predictions)
            test_r2 = r2_score(test_actual, test_predictions)
            
            print(f"\nTest Metrics:")
            print(f"MSE: {test_mse:.4f}")
            print(f"MAE: {test_mae:.4f}")
            print(f"R²: {test_r2:.4f}")
            print(f"RMSE: {np.sqrt(test_mse):.4f}")
            
            return {
                'train': {'mse': train_mse, 'mae': train_mae, 'r2': train_r2},
                'test': {'mse': test_mse, 'mae': test_mae, 'r2': test_r2}
            }
        
        return {'train': {'mse': train_mse, 'mae': train_mae, 'r2': train_r2}}
    
    def plot_results(self, figsize=(15, 10)):
        """
        Plot training history and predictions
        """
        plt.figure(figsize=figsize)
        
        # Plot training history
        plt.subplot(2, 3, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(2, 3, 2)
        plt.plot(self.history.history['mae'], label='Training MAE')
        plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        # Plot predictions vs actual (sample)
        train_pred = self.model.predict(self.X_train[:1000])
        train_pred = self.target_scaler.inverse_transform(train_pred)
        train_actual = self.target_scaler.inverse_transform(self.y_train[:1000])
        
        plt.subplot(2, 3, 3)
        plt.scatter(train_actual, train_pred, alpha=0.5)
        plt.plot([train_actual.min(), train_actual.max()], 
                [train_actual.min(), train_actual.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Training: Actual vs Predicted')
        
        # Plot time series prediction (sample)
        plt.subplot(2, 3, 4)
        sample_size = min(200, len(train_actual))
        plt.plot(train_actual[:sample_size], label='Actual', alpha=0.7)
        plt.plot(train_pred[:sample_size], label='Predicted', alpha=0.7)
        plt.title('Training: Time Series Comparison')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        
        # If test data available, plot test results
        if hasattr(self, 'X_test') and self.X_test is not None:
            test_pred = self.model.predict(self.X_test[:1000])
            test_pred = self.target_scaler.inverse_transform(test_pred)
            test_actual = self.target_scaler.inverse_transform(self.y_test[:1000])
            
            plt.subplot(2, 3, 5)
            plt.scatter(test_actual, test_pred, alpha=0.5)
            plt.plot([test_actual.min(), test_actual.max()], 
                    [test_actual.min(), test_actual.max()], 'r--', lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Test: Actual vs Predicted')
            
            plt.subplot(2, 3, 6)
            sample_size = min(200, len(test_actual))
            plt.plot(test_actual[:sample_size], label='Actual', alpha=0.7)
            plt.plot(test_pred[:sample_size], label='Predicted', alpha=0.7)
            plt.title('Test: Time Series Comparison')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def forecast_future(self, steps=30, symbol_data=None):
        """
        Forecast future values
        
        Args:
            steps (int): Number of steps to forecast
            symbol_data (DataFrame): Data for a specific symbol to forecast
        
        Returns:
            array: Forecasted values
        """
        if symbol_data is None:
            # Use the last sequence from training data
            last_sequence = self.X_train[-1:].copy()
        else:
            # Use provided symbol data
            if len(symbol_data) < self.sequence_length:
                raise ValueError(f"Need at least {self.sequence_length} data points")
            
            feature_data = symbol_data[self.features].values[-self.sequence_length:]
            last_sequence = self.scaler.transform(feature_data).reshape(1, self.sequence_length, len(self.features))
        
        forecasts = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Predict next value
            next_pred = self.model.predict(current_sequence)
            forecasts.append(next_pred[0])
            
            # Update sequence (simplified - in practice, you'd need to handle all features)
            # For now, we'll just repeat the last feature values and update the target
            new_step = current_sequence[0, -1:].copy()
            new_step[0, self.features.index(self.target_feature)] = next_pred[0, 0]
            
            # Roll the sequence
            current_sequence = np.concatenate([current_sequence[:, 1:], new_step.reshape(1, 1, -1)], axis=1)
        
        # Inverse transform predictions
        forecasts = np.array(forecasts)
        forecasts = self.target_scaler.inverse_transform(forecasts)
        
        return forecasts.flatten()
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


def main():
    """
    Main function to demonstrate the LSTM forecaster
    """
    print("=== LSTM Time Series Forecaster ===")
    
    # Initialize the forecaster
    # Using multiple features for better prediction
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'High_Low_Ratio', 
               'Close_Open_Ratio', 'MA_5', 'MA_10', 'MA_20', 'Volatility', 'RSI']
    
    forecaster = LSTMTimeSeriesForecaster(
        sequence_length=60,
        features=features,
        target_feature='Close'
    )
    
    # Load and preprocess data
    train_path = "stocks/train"
    test_path = "stocks/test"
    
    # For demonstration, use a sample of files (remove sample_size=10 to use all files)
    forecaster.load_and_preprocess_data(train_path, test_path, sample_size=10)
    
    # Build the model
    forecaster.build_model(
        model_type='advanced',  # Options: 'simple', 'bidirectional', 'advanced'
        lstm_units=[64, 32],
        dropout_rate=0.3,
        learning_rate=0.001
    )
    
    # Train the model
    forecaster.train_model(
        epochs=5,  # Reduce for quick testing
        batch_size=32,
        validation_split=0.2,
        patience=10
    )
    
    # Evaluate the model
    metrics = forecaster.evaluate_model()
    
    # Plot results
    forecaster.plot_results()
    
    # Save the model
    forecaster.save_model('lstm_forecaster.h5')
    
    # Example: Forecast future values for the last sequence
    print("\nForecasting next 30 days...")
    future_forecast = forecaster.forecast_future(steps=30)
    print(f"Forecasted values: {future_forecast[:10]}...")  # Show first 10 values
    
    # Plot the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(future_forecast)), future_forecast, 'r-', label='Forecast', linewidth=2)
    plt.title('Future Price Forecast')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\nForecasting completed!")


if __name__ == "__main__":
    main()