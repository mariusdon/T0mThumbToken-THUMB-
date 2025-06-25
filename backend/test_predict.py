import requests
import json

# Test the predict endpoint with a POST request
base_url = "http://localhost:8000"

# Sample features for testing
sample_features = [5.4, 5.35, 5.1, 4.75, 4.3, 4.2, 4.35, 320.0, 4.0, 100.0]

print("Testing /predict endpoint with POST request...")
print("=" * 50)

try:
    response = requests.post(
        f"{base_url}/predict",
        json={"features": sample_features},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ /predict: Success")
        print(f"   Response: {data}")
        if 'allocation' in data:
            print(f"   Allocation length: {len(data['allocation'])}")
            print(f"   Sample allocation: {data['allocation'][:3]}")
    else:
        print(f"❌ /predict: HTTP {response.status_code}")
        print(f"   Error: {response.text}")
        
except Exception as e:
    print(f"❌ /predict: Exception - {e}")

print("\n" + "=" * 50)
print("Predict endpoint testing complete!") 