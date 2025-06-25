import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    ml_allocation_model = joblib.load(os.path.join(BASE_DIR, 'ml_allocation_model.pkl'))
    ml_allocation_scaler = joblib.load(os.path.join(BASE_DIR, 'ml_allocation_scaler.pkl'))
except Exception as e:
    print(f"Warning: Could not load ML allocation model: {e}")
    ml_allocation_model = None
    ml_allocation_scaler = None 