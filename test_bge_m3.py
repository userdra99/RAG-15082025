#!/usr/bin/env python3
"""
BGE-M3 Integration Test Suite

This script comprehensively tests the BGE-M3 integration including:
- Model loading verification
- Embedding dimension validation  
- Vector database operations
- End-to-end RAG functionality
"""

import requests
import json
import time
import logging
import sys
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BGE_M3_Tester:
    def __init__(self):
        self.llm_endpoint = "http://localhost:8001"
        self.embedding_endpoint = "http://localhost:8002" 
        self.app_endpoint = "http://localhost:5000"
        self.qdrant_endpoint = "http://localhost:6333"
        
        self.test_results = []
    
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        result = {
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name}: {details}")
    
    def test_service_health(self) -> bool:
        """Test basic service health"""
        services = [
            ("LLM Service", f"{self.llm_endpoint}/v1/models"),
            ("Embedding Service", f"{self.embedding_endpoint}/v1/models"),
            ("Main App", f"{self.app_endpoint}/health"),
            ("Qdrant", f"{self.qdrant_endpoint}/collections")
        ]
        
        all_healthy = True
        for service_name, url in services:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    self.log_test_result(f"{service_name} Health", True, f"Status: {response.status_code}")
                else:
                    self.log_test_result(f"{service_name} Health", False, f"Status: {response.status_code}")
                    all_healthy = False
            except Exception as e:
                self.log_test_result(f"{service_name} Health", False, f"Error: {str(e)}")
                all_healthy = False
        
        return all_healthy
    
    def test_bge_m3_model_loaded(self) -> bool:
        """Test that BGE-M3 model is properly loaded"""
        try:
            response = requests.get(f"{self.embedding_endpoint}/v1/models", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                model_names = [model.get('id', '') for model in models_data.get('data', [])]
                
                bge_m3_loaded = any('bge-m3' in model.lower() for model in model_names)
                
                if bge_m3_loaded:
                    self.log_test_result("BGE-M3 Model Loading", True, f"Models: {model_names}")
                    return True
                else:
                    self.log_test_result("BGE-M3 Model Loading", False, f"BGE-M3 not found in: {model_names}")
                    return False
            else:
                self.log_test_result("BGE-M3 Model Loading", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test_result("BGE-M3 Model Loading", False, f"Error: {str(e)}")
            return False
    
    def test_embedding_dimensions(self) -> bool:
        """Test that embeddings have correct 1024 dimensions for BGE-M3"""
        try:
            test_text = "This is a test sentence for embedding dimension validation."
            
            payload = {
                "input": [test_text],
                "model": "BAAI/bge-m3"
            }
            
            response = requests.post(
                f"{self.embedding_endpoint}/v1/embeddings",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                embeddings = data.get('data', [])
                
                if embeddings and len(embeddings) > 0:
                    embedding_vector = embeddings[0].get('embedding', [])
                    dimension = len(embedding_vector)
                    
                    if dimension == 1024:
                        self.log_test_result("Embedding Dimensions", True, f"Correct 1024 dimensions")
                        return True
                    else:
                        self.log_test_result("Embedding Dimensions", False, f"Got {dimension} dimensions, expected 1024")
                        return False
                else:
                    self.log_test_result("Embedding Dimensions", False, "No embeddings returned")
                    return False
            else:
                self.log_test_result("Embedding Dimensions", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Embedding Dimensions", False, f"Error: {str(e)}")
            return False
    
    def test_qdrant_collection(self) -> bool:
        """Test Qdrant collection configuration"""
        try:
            response = requests.get(f"{self.qdrant_endpoint}/collections/documents", timeout=10)
            
            if response.status_code == 200:
                collection_info = response.json()
                vector_size = collection_info.get('result', {}).get('config', {}).get('params', {}).get('vectors', {}).get('size', 0)
                
                if vector_size == 1024:
                    points_count = collection_info.get('result', {}).get('points_count', 0)
                    self.log_test_result("Qdrant Collection", True, f"1024-dim collection with {points_count} points")
                    return True
                else:
                    self.log_test_result("Qdrant Collection", False, f"Collection has {vector_size} dimensions, expected 1024")
                    return False
            else:
                self.log_test_result("Qdrant Collection", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test_result("Qdrant Collection", False, f"Error: {str(e)}")
            return False
    
    def test_app_initialization(self) -> bool:
        """Test main application initialization with BGE-M3"""
        try:
            # Test initialization endpoint
            response = requests.post(f"{self.app_endpoint}/initialize", timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    self.log_test_result("App Initialization", True, result.get('message', ''))
                    return True
                else:
                    self.log_test_result("App Initialization", False, result.get('error', ''))
                    return False
            else:
                self.log_test_result("App Initialization", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test_result("App Initialization", False, f"Error: {str(e)}")
            return False
    
    def test_end_to_end_query(self) -> bool:
        """Test end-to-end query functionality (if documents are available)"""
        try:
            test_query = {
                "query": "What is the purpose of this document collection?"
            }
            
            response = requests.post(
                f"{self.app_endpoint}/query",
                json=test_query,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    answer = result.get('answer', '')
                    sources = result.get('sources', [])
                    self.log_test_result("End-to-End Query", True, f"Got answer with {len(sources)} sources")
                    return True
                else:
                    self.log_test_result("End-to-End Query", False, result.get('error', ''))
                    return False
            else:
                # Query might fail if no documents are indexed yet
                self.log_test_result("End-to-End Query", False, f"HTTP {response.status_code} (may be expected if no documents)")
                return False
                
        except Exception as e:
            self.log_test_result("End-to-End Query", False, f"Error: {str(e)}")
            return False
    
    def test_performance_benchmark(self) -> bool:
        """Basic performance benchmark for BGE-M3"""
        try:
            test_texts = [
                "This is a performance test for BGE-M3 embedding generation.",
                "We are measuring the speed and efficiency of the new embedding model.",
                "BGE-M3 should provide high-quality embeddings with good performance."
            ]
            
            start_time = time.time()
            
            payload = {
                "input": test_texts,
                "model": "BAAI/bge-m3"
            }
            
            response = requests.post(
                f"{self.embedding_endpoint}/v1/embeddings",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                embeddings = data.get('data', [])
                
                if len(embeddings) == len(test_texts):
                    tokens_per_second = len(test_texts) / duration
                    self.log_test_result("Performance Benchmark", True, f"{tokens_per_second:.2f} texts/sec, {duration:.2f}s total")
                    return True
                else:
                    self.log_test_result("Performance Benchmark", False, f"Got {len(embeddings)} embeddings, expected {len(test_texts)}")
                    return False
            else:
                self.log_test_result("Performance Benchmark", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test_result("Performance Benchmark", False, f"Error: {str(e)}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        logger.info("🧪 Starting BGE-M3 Integration Test Suite")
        logger.info("=" * 50)
        
        # Test order matters - basic services first
        tests = [
            ("Service Health Check", self.test_service_health),
            ("BGE-M3 Model Loading", self.test_bge_m3_model_loaded),
            ("Embedding Dimensions", self.test_embedding_dimensions),
            ("Qdrant Collection Config", self.test_qdrant_collection),
            ("App Initialization", self.test_app_initialization),
            ("Performance Benchmark", self.test_performance_benchmark),
            ("End-to-End Query", self.test_end_to_end_query)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\n🔄 Running: {test_name}")
            try:
                test_func()
            except Exception as e:
                self.log_test_result(test_name, False, f"Unexpected error: {str(e)}")
            
            time.sleep(1)  # Brief pause between tests
        
        # Generate summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            "results": self.test_results
        }
        
        logger.info("\n" + "=" * 50)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests} ✅")
        logger.info(f"Failed: {failed_tests} ❌")
        logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        
        if failed_tests == 0:
            logger.info("\n🎉 All tests passed! BGE-M3 integration is working correctly.")
        else:
            logger.info(f"\n⚠️  {failed_tests} test(s) failed. Check the details above.")
        
        return summary

def main():
    """Main function"""
    tester = BGE_M3_Tester()
    summary = tester.run_all_tests()
    
    # Save detailed results
    with open('/tmp/bge_m3_test_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Exit with appropriate code
    sys.exit(0 if summary['failed'] == 0 else 1)

if __name__ == "__main__":
    main()