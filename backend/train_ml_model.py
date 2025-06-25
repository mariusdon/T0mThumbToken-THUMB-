import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load training data
npz = np.load('ml_training_data.npz')
X = npz['features']
y = npz['targets']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MSE: {mse:.6f}, R2: {r2:.4f}")

# Save model and scaler
joblib.dump(model, 'ml_allocation_model.pkl')
joblib.dump(scaler, 'ml_allocation_scaler.pkl')
print("Saved model to ml_allocation_model.pkl and scaler to ml_allocation_scaler.pkl") 