import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class TreasuryNeuralNetwork(nn.Module):
    """
    Neural network for predicting Treasury bond returns.
    """
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=7, dropout_rate=0.2):
        super(TreasuryNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class TreasuryAllocationNeuralModel:
    """
    Neural Network model for predicting optimal Treasury bond allocations.
    Uses a multi-layer perceptron to predict returns for all 7 maturities simultaneously.
    """
    
    def __init__(self, hidden_sizes=[128, 64, 32], dropout_rate=0.2, learning_rate=0.001, 
                 batch_size=32, epochs=100, random_state=42):
        """
        Initialize the Neural Network allocation model.
        
        Args:
            hidden_sizes (list): List of hidden layer sizes
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            random_state (int): Random seed for reproducibility
        """
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Initialize model components
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store feature names and maturity names
        self.feature_names = None
        self.maturity_names = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
        
        # Model performance metrics
        self.performance_metrics = {}
        self.training_history = {'train_loss': [], 'val_loss': []}
        
    def prepare_targets(self, returns_df, forecast_horizon=1):
        """
        Prepare target variables for training.
        
        Args:
            returns_df (pd.DataFrame): Historical returns data
            forecast_horizon (int): Number of periods ahead to predict
            
        Returns:
            pd.DataFrame: Target variables (future returns)
        """
        # Shift returns forward to create targets
        targets = returns_df.shift(-forecast_horizon)
        
        # Remove the last few rows where we don't have targets
        targets = targets.dropna()
        
        return targets
    
    def train(self, features_df, returns_df, forecast_horizon=1, test_size=0.2, 
              early_stopping_patience=10):
        """
        Train the Neural Network model.
        
        Args:
            features_df (pd.DataFrame): Feature matrix
            returns_df (pd.DataFrame): Historical returns data
            forecast_horizon (int): Number of periods ahead to predict
            test_size (float): Proportion of data for testing
            early_stopping_patience (int): Number of epochs to wait before early stopping
            
        Returns:
            dict: Training performance metrics
        """
        print(f"Training Neural Network model with {self.epochs} epochs...")
        print(f"Device: {self.device}")
        
        # Prepare targets
        targets = self.prepare_targets(returns_df, forecast_horizon)
        
        # Align features with targets
        common_index = features_df.index.intersection(targets.index)
        features_aligned = features_df.loc[common_index]
        targets_aligned = targets.loc[common_index]
        
        # Store feature names
        self.feature_names = features_aligned.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_aligned, targets_aligned, 
            test_size=test_size, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test.values).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        input_size = X_train.shape[1]
        output_size = y_train.shape[1]
        self.model = TreasuryNeuralNetwork(
            input_size=input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=output_size,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor).item()
            
            # Record history
            self.training_history['train_loss'].append(train_loss / len(train_loader))
            self.training_history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], "
                      f"Train Loss: {train_loss/len(train_loader):.6f}, "
                      f"Val Loss: {val_loss:.6f}")
            
            # Early stopping check
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            y_train_pred = self.model(X_train_tensor).cpu().numpy()
            y_test_pred = self.model(X_test_tensor).cpu().numpy()
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_metrics(
            y_train, y_train_pred, y_test, y_test_pred
        )
        
        # Print performance summary
        self._print_performance_summary()
        
        return self.performance_metrics
    
    def _calculate_metrics(self, y_train, y_train_pred, y_test, y_test_pred):
        """Calculate model performance metrics."""
        metrics = {}
        
        # Overall R² scores
        metrics['train_r2'] = r2_score(y_train, y_train_pred)
        metrics['test_r2'] = r2_score(y_test, y_test_pred)
        
        # MSE scores
        metrics['train_mse'] = mean_squared_error(y_train, y_train_pred)
        metrics['test_mse'] = mean_squared_error(y_test, y_test_pred)
        
        # Per-maturity R² scores
        maturity_r2_train = []
        maturity_r2_test = []
        
        for i, maturity in enumerate(self.maturity_names):
            train_r2 = r2_score(y_train.iloc[:, i], y_train_pred[:, i])
            test_r2 = r2_score(y_test.iloc[:, i], y_test_pred[:, i])
            maturity_r2_train.append(train_r2)
            maturity_r2_test.append(test_r2)
            
        metrics['maturity_r2_train'] = dict(zip(self.maturity_names, maturity_r2_train))
        metrics['maturity_r2_test'] = dict(zip(self.maturity_names, maturity_r2_test))
        
        return metrics
    
    def _print_performance_summary(self):
        """Print model performance summary."""
        print("\n" + "="*50)
        print("NEURAL NETWORK MODEL PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Overall R² Score (Train): {self.performance_metrics['train_r2']:.4f}")
        print(f"Overall R² Score (Test):  {self.performance_metrics['test_r2']:.4f}")
        print(f"Overall MSE (Train):      {self.performance_metrics['train_mse']:.6f}")
        print(f"Overall MSE (Test):       {self.performance_metrics['test_mse']:.6f}")
        
        print("\nPer-Maturity R² Scores (Test):")
        for maturity, r2 in self.performance_metrics['maturity_r2_test'].items():
            print(f"  {maturity}: {r2:.4f}")
        print("="*50)
    
    def predict_returns(self, features):
        """
        Predict returns for all maturities.
        
        Args:
            features (pd.DataFrame or dict): Feature values
            
        Returns:
            np.array: Predicted returns for each maturity
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        # Ensure features are in the correct order
        if self.feature_names:
            features = features[self.feature_names]
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(features_tensor).cpu().numpy()
        
        return predictions[0]  # Return first (and only) prediction
    
    def predict_allocation(self, features, method='curve_aware', risk_free_rate=0.02):
        """
        Predict optimal allocation weights based on predicted returns and curve shape.
        
        Args:
            features (pd.DataFrame or dict): Feature values
            method (str): Allocation method ('softmax', 'inverse_vol', 'rank', 'curve_aware')
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculation
            
        Returns:
            dict: Allocation weights for each maturity
        """
        predicted_returns = self.predict_returns(features)
        
        if method == 'curve_aware':
            return self._curve_aware_allocation(features, predicted_returns)
        elif method == 'softmax':
            return self._softmax_allocation(predicted_returns)
        elif method == 'inverse_vol':
            return self._inverse_volatility_allocation(predicted_returns)
        elif method == 'rank':
            return self._rank_based_allocation(predicted_returns)
        else:
            raise ValueError(f"Unknown allocation method: {method}")
    
    def _curve_aware_allocation(self, features, predicted_returns):
        """
        Curve-aware allocation that considers yield curve shape and predicted returns.
        """
        # Extract yield curve features if available
        curve_features = {}
        for col in features.columns:
            if 'yield_' in col or 'slope_' in col:
                curve_features[col] = features[col].iloc[0] if hasattr(features, 'iloc') else features[col]
        
        # Base allocation on predicted returns
        base_weights = self._softmax_allocation(predicted_returns, temperature=0.5)
        
        # Adjust based on curve shape
        if 'slope_2y10y' in curve_features:
            slope = curve_features['slope_2y10y']
            
            # Steep curve: favor longer maturities
            if slope > 0.5:  # Steep curve
                adjustment = np.array([0.05, 0.05, 0.05, 0.05, 0.1, 0.15, 0.15])
            # Flat/inverted curve: favor shorter maturities
            elif slope < -0.2:  # Inverted curve
                adjustment = np.array([0.15, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05])
            else:  # Normal curve
                adjustment = np.array([0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.05])
            
            # Apply adjustment
            adjusted_weights = base_weights + adjustment
            adjusted_weights = np.maximum(adjusted_weights, 0.01)  # Minimum 1% allocation
            adjusted_weights = adjusted_weights / adjusted_weights.sum()  # Renormalize
            
            return dict(zip(self.maturity_names, adjusted_weights))
        
        return dict(zip(self.maturity_names, base_weights))
    
    def _softmax_allocation(self, returns, temperature=1.0):
        """
        Softmax-based allocation based on predicted returns.
        """
        # Apply softmax to predicted returns
        exp_returns = np.exp(returns / temperature)
        weights = exp_returns / exp_returns.sum()
        
        return dict(zip(self.maturity_names, weights))
    
    def _inverse_volatility_allocation(self, returns, volatility_window=30):
        """
        Inverse volatility allocation (simplified version).
        """
        # For simplicity, use equal weights since we don't have volatility data
        # In a real implementation, you would use historical volatility
        weights = np.ones(len(returns)) / len(returns)
        
        return dict(zip(self.maturity_names, weights))
    
    def _rank_based_allocation(self, returns):
        """
        Rank-based allocation where higher-ranked maturities get more weight.
        """
        # Rank returns (higher return = higher rank)
        ranks = np.argsort(np.argsort(returns))[::-1] + 1
        
        # Convert ranks to weights (higher rank = higher weight)
        weights = ranks / ranks.sum()
        
        return dict(zip(self.maturity_names, weights))
    
    def save_model(self, filepath):
        """
        Save the trained model and scaler.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        
        # Save PyTorch model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'hidden_sizes': self.hidden_sizes,
                'dropout_rate': self.dropout_rate,
                'input_size': self.model.network[0].in_features,
                'output_size': self.model.network[-1].out_features
            },
            'feature_names': self.feature_names,
            'maturity_names': self.maturity_names,
            'performance_metrics': self.performance_metrics,
            'training_history': self.training_history
        }, f"{filepath}_model.pth")
        
        # Save scaler
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        
        print(f"Model saved to {filepath}_model.pth and {filepath}_scaler.pkl")
    
    def load_model(self, filepath):
        """
        Load a trained model and scaler.
        
        Args:
            filepath (str): Path to load the model from
        """
        # Load PyTorch model
        checkpoint = torch.load(f"{filepath}_model.pth", map_location=self.device)
        
        # Reconstruct model
        model_config = checkpoint['model_config']
        self.model = TreasuryNeuralNetwork(
            input_size=model_config['input_size'],
            hidden_sizes=model_config['hidden_sizes'],
            output_size=model_config['output_size'],
            dropout_rate=model_config['dropout_rate']
        ).to(self.device)
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load other attributes
        self.feature_names = checkpoint['feature_names']
        self.maturity_names = checkpoint['maturity_names']
        self.performance_metrics = checkpoint['performance_metrics']
        self.training_history = checkpoint['training_history']
        
        # Load scaler
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        
        print(f"Model loaded from {filepath}_model.pth and {filepath}_scaler.pkl")
    
    def create_features(self, yield_df, economic_df=None):
        """
        Create features for the neural network model.
        This method is identical to the Random Forest version for compatibility.
        
        Args:
            yield_df (pd.DataFrame): Yield curve data
            economic_df (pd.DataFrame, optional): Economic indicators data
            
        Returns:
            pd.DataFrame: Feature matrix
        """
        features = pd.DataFrame(index=yield_df.index)
        
        # Yield curve features
        maturities = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
        
        # Individual yields
        for maturity in maturities:
            if maturity in yield_df.columns:
                features[f'yield_{maturity.lower()}'] = yield_df[maturity]
        
        # Yield curve slopes
        if '2Y' in yield_df.columns and '10Y' in yield_df.columns:
            features['slope_2y10y'] = yield_df['10Y'] - yield_df['2Y']
        
        if '3M' in yield_df.columns and '10Y' in yield_df.columns:
            features['slope_3m10y'] = yield_df['10Y'] - yield_df['3M']
        
        if '2Y' in yield_df.columns and '5Y' in yield_df.columns:
            features['slope_2y5y'] = yield_df['5Y'] - yield_df['2Y']
        
        # Lagged slopes (for trend information)
        if 'slope_2y10y' in features.columns:
            features['slope_2y10y_lag1'] = features['slope_2y10y'].shift(1)
            features['slope_2y10y_lag3'] = features['slope_2y10y'].shift(3)
            features['slope_2y10y_lag6'] = features['slope_2y10y'].shift(6)
        
        # Short rate features
        if '3M' in yield_df.columns:
            features['short_rate'] = yield_df['3M']
            features['short_rate_lag1'] = yield_df['3M'].shift(1)
            features['short_rate_change'] = yield_df['3M'].diff()
        
        # Time features
        features['month'] = yield_df.index.month
        features['quarter'] = yield_df.index.quarter
        features['year'] = yield_df.index.year
        
        # Cyclical encoding for time features
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['quarter_sin'] = np.sin(2 * np.pi * features['quarter'] / 4)
        features['quarter_cos'] = np.cos(2 * np.pi * features['quarter'] / 4)
        
        # Economic indicators (if available)
        if economic_df is not None:
            for col in economic_df.columns:
                features[f'economic_{col}'] = economic_df[col]
        
        # Remove any infinite or NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.dropna()
        
        return features 