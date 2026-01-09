"""
Quick test script to verify MediTrack backend is working
Run this after starting the server to test the API
"""
import requests
import sys
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✓ Health check passed")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Could not connect to server: {e}")
        print("  Make sure the backend is running: python -m app.main")
        return False

def test_root():
    """Test root endpoint"""
    print("\nTesting root endpoint...")
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            print("✓ Root endpoint working")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_stats():
    """Test stats endpoint"""
    print("\nTesting stats endpoint...")
    try:
        response = requests.get(f"{API_URL}/api/wounds/stats/overview}")
        if response.status_code == 200:
            print("✓ Stats endpoint working")
            stats = response.json()
            print(f"  Total scans: {stats.get('total_scans', 0)}")
            print(f"  Total patients: {stats.get('total_patients', 0)}")
            return True
        else:
            print(f"✗ Stats endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("MediTrack Backend Test Suite")
    print("=" * 50)
    
    tests = [
        test_health_check,
        test_root,
        test_stats
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 50)
    
    if passed == len(tests):
        print("\n✅ All tests passed! Backend is working correctly.")
        print("\nNext steps:")
        print("1. Test image analysis: Upload a wound image via /docs")
        print("2. Or use the Streamlit frontend (coming next)")
    else:
        print("\n⚠️  Some tests failed. Check the server logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
