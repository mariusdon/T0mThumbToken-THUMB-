import requests
import json

# Test the backend endpoints
base_url = "http://localhost:8000"

def test_endpoint(endpoint):
    try:
        response = requests.get(f"{base_url}{endpoint}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {endpoint}: Success")
            print(f"   Response keys: {list(data.keys())}")
            if 'features' in data:
                print(f"   Features length: {len(data['features'])}")
                print(f"   Regime: {data.get('regime', 'N/A')}")
            elif 'yields' in data:
                print(f"   Yields length: {len(data['yields'])}")
                print(f"   Sample yields: {data['yields'][:3]}")
            return True
        else:
            print(f"❌ {endpoint}: HTTP {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ {endpoint}: Exception - {e}")
        return False

print("Testing backend endpoints...")
print("=" * 50)

# Test latest features endpoint
print("\n1. Testing /latest-features endpoint:")
test_endpoint("/latest-features")

# Test yield curve endpoint
print("\n2. Testing /yield-curve endpoint:")
test_endpoint("/yield-curve")

# Test predict endpoint
print("\n3. Testing /predict endpoint:")
test_endpoint("/predict")

print("\n" + "=" * 50)
print("Endpoint testing complete!") 