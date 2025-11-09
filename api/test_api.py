import requests
import json

def test_api():
    """Test the Flask API endpoints"""
    base_url = "http://localhost:5000/api"
    
    print("🧪 Testing Parkinson's Detection API")
    print("=" * 50)
    
    # Test health check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test prediction with mock features
    print("\n2. Testing Prediction with Mock Features...")
    mock_features = {
        'age': 65.0,
        'sex': 0,
        'test_time': 5.0,
        'Jitter(%)': 0.005,
        'Jitter(Abs)': 0.00003,
        'Jitter:RAP': 0.003,
        'Jitter:PPQ5': 0.0035,
        'Jitter:DDP': 0.009,
        'Shimmer': 0.025,
        'Shimmer(dB)': 0.25,
        'Shimmer:APQ3': 0.012,
        'Shimmer:APQ5': 0.015,
        'Shimmer:APQ11': 0.020,
        'Shimmer:DDA': 0.036,
        'NHR': 0.015,
        'HNR': 22.0,
        'RPDE': 0.5,
        'DFA': 0.65,
        'PPE': 0.2,
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={'features': mock_features},
            headers={'Content-Type': 'application/json'}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")

if __name__ == "__main__":
    test_api()