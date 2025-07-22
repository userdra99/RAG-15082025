#!/usr/bin/env python3
"""
Test script to verify jinaai/jina-embeddings-v4-vllm-retrieval integration
"""

import os
import requests
import json
import sys
from typing import List

def test_jina_embeddings():
    """Test the jina embeddings model"""
    
    # Test configuration
    api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "sk-12345")
    model_name = "jina-embeddings-v4"
    
    # Test texts
    texts = [
        "Hello world",
        "This is a test sentence for embeddings.",
        "Jina embeddings v4 is a powerful multimodal embedding model."
    ]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        # Test 1: Check if the service is running
        print("🔍 Testing if vLLM service is running...")
        response = requests.get(f"{api_base}/models", headers=headers, timeout=10)
        
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Service is running")
            print(f"Available models: {[m['id'] for m in models.get('data', [])]}")
        else:
            print(f"❌ Service returned status {response.status_code}")
            return False
            
        # Test 2: Test embedding generation
        print("\n🔍 Testing embedding generation...")
        payload = {
            "model": model_name,
            "input": texts,
            "encoding_format": "float",
            "vector_type": "dense"
        }
        
        response = requests.post(f"{api_base}/embeddings", 
                               headers=headers, 
                               json=payload, 
                               timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            embeddings = data.get('data', [])
            
            if len(embeddings) != len(texts):
                print(f"❌ Expected {len(texts)} embeddings, got {len(embeddings)}")
                return False
                
            # Check embedding dimensions
            embedding_dim = len(embeddings[0]['embedding'])
            print(f"✅ Successfully generated {len(embeddings)} embeddings")
            print(f"✅ Embedding dimension: {embedding_dim}")
            
            # Expected dimension for jina-embeddings-v4 is 2048
            if embedding_dim == 2048:
                print("✅ Correct embedding dimension (2048)")
            else:
                print(f"⚠️  Unexpected embedding dimension: {embedding_dim} (expected 2048)")
            
            # Test 3: Test similarity
            print("\n🔍 Testing embedding similarity...")
            
            # Calculate cosine similarity between first two embeddings
            import numpy as np
            
            vec1 = np.array(embeddings[0]['embedding'])
            vec2 = np.array(embeddings[1]['embedding'])
            
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            print(f"✅ Cosine similarity between text 1 and 2: {similarity:.4f}")
            
            # Test 4: Test late-interaction embeddings
            print("\n🔍 Testing late-interaction embeddings...")
            
            payload_late = {
                "model": model_name,
                "input": ["Test late interaction"],
                "encoding_format": "float",
                "vector_type": "late"
            }
            
            response_late = requests.post(f"{api_base}/embeddings", 
                                        headers=headers, 
                                        json=payload_late, 
                                        timeout=30)
            
            if response_late.status_code == 200:
                data_late = response_late.json()
                late_embedding = data_late.get('data', [])[0]['embedding']
                print(f"✅ Late-interaction embedding shape: {np.array(late_embedding).shape}")
            else:
                print(f"⚠️  Late-interaction not supported: {response_late.status_code}")
            
            return True
            
        else:
            print(f"❌ Embedding request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_integration():
    """Test integration with LlamaIndex"""
    print("\n🔍 Testing LlamaIndex integration...")
    
    try:
        # Import LlamaIndex components
        from llama_index.core import Settings
        from llama_index.embeddings.openai import OpenAIEmbedding
        
        # Configure embedding model
        Settings.embed_model = OpenAIEmbedding(
            model="jina-embeddings-v4",
            api_base="http://localhost:8000/v1",
            api_key="sk-12345",
            dimensions=2048
        )
        
        # Test embedding a sample text
        embedding = Settings.embed_model.get_text_embedding("Integration test")
        
        if len(embedding) == 2048:
            print("✅ LlamaIndex integration successful")
            return True
        else:
            print(f"❌ Unexpected embedding dimension: {len(embedding)}")
            return False
            
    except ImportError as e:
        print(f"⚠️  LlamaIndex not available: {e}")
        return True  # Skip integration test if LlamaIndex not installed
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing jinaai/jina-embeddings-v4-vllm-retrieval integration")
    print("=" * 60)
    
    success = True
    
    # Test basic functionality
    success &= test_jina_embeddings()
    
    # Test integration
    success &= test_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed! jina-embeddings-v4 is ready to use.")
    else:
        print("❌ Some tests failed. Check the logs above.")
    
    sys.exit(0 if success else 1)