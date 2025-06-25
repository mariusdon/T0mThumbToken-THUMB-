import subprocess
import os
import time
import uvicorn

def main():
    # First, train the model
    print("Training ML model...")
    try:
        subprocess.run(["python", "train_model.py"], check=True)
        print("Model training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error training model: {e}")
        return

    # Wait a moment for files to be written
    time.sleep(1)

    # Check if model files exist
    if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
        print("Error: Model files not found after training")
        return

    # Start the FastAPI server
    print("\nStarting FastAPI server...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main() 