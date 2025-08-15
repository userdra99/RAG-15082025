#!/usr/bin/env python3
"""
Simple BGE-M3 Integration Test

Quick test to verify BGE-M3 is working correctly
"""

import requests
import time
import json

def test_bge_m3_simple():
    """Simple test for BGE-M3 service"""
    
    print("🧪 Testing BGE-M3 Integration")
    print("=" * 40)
    
    # Check service health
    print("1. Checking service health...")
    try:
        response = requests.get("http://localhost:8002/v1/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Service is running. Models: {[m.get('id') for m in models.get('data', [])]}")
        else:
            print(f"❌ Service unhealthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Service unreachable: {e}")
        return False
    
    # Test embedding dimensions
    print("\n2. Testing embedding dimensions...")
    try:
        payload = {
            "input": ["Test embedding for BGE-M3"],
            "model": "BAAI/bge-m3"
        }
        
        response = requests.post(
            "http://localhost:8002/v1/embeddings",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            embeddings = data.get('data', [])
            if embeddings:
                dims = len(embeddings[0].get('embedding', []))
                if dims == 1024:
                    print(f"✅ Correct dimensions: {dims}")
                    return True
                else:
                    print(f"❌ Wrong dimensions: {dims}, expected 1024")
                    return False
            else:
                print("❌ No embeddings returned")
                return False
        else:
            print(f"❌ Embedding request failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False

if __name__ == "__main__":
    # Wait for service to be ready
    print("Waiting for BGE-M3 service to be ready...")
    for i in range(30):  # Wait up to 5 minutes
        try:
            response = requests.get("http://localhost:8002/v1/models", timeout=5)
            if response.status_code == 200:
                print("\n🚀 Service is ready!")
                break
        except:
            pass
        print(f"Waiting... {i+1}/30")
        time.sleep(10)
    else:
        print("❌ Service did not become ready in time")
        exit(1)
    
    # Run the test
    if test_bge_m3_simple():
        print("\n🎉 BGE-M3 test passed!")
        exit(0)
    else:
        print("\n❌ BGE-M3 test failed!")
        exit(1)