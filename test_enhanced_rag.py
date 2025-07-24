#!/usr/bin/env python3
"""
Test script to validate the enhanced RAG system with docling contextual chunking and reranker.
"""

import os
import sys
import logging
from pathlib import Path
import time

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_docling_chunking():
    """Test docling contextual chunking functionality"""
    try:
        from app.main import DoclingPDFReader, DoclingDocxReader, DoclingExcelReader
        
        # Test with a sample PDF if available
        data_dir = Path("data")
        if data_dir.exists():
            pdf_files = list(data_dir.glob("*.pdf"))
            if pdf_files:
                reader = DoclingPDFReader()
                docs = reader.load_data(str(pdf_files[0]))
                logger.info(f"✅ Docling PDF chunking: Created {len(docs)} contextual chunks")
                
                # Verify chunk metadata
                if docs:
                    chunk = docs[0]
                    assert "chunk_id" in chunk.metadata
                    assert "chunk_type" in chunk.metadata
                    assert chunk.metadata["chunk_type"] == "contextual"
                    logger.info("✅ Chunk metadata validation passed")
                
                return True
        
        logger.info("ℹ️  No PDF files found for testing, skipping PDF chunking test")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error in docling chunking: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing docling chunking: {e}")
        return False

def test_reranker():
    """Test reranker functionality"""
    try:
        from llama_index.core.postprocessor import SentenceTransformerRerank
        
        # Test reranker initialization
        reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=3
        )
        
        logger.info("✅ Reranker initialization successful")
        
        # Test reranker model info
        model_name = reranker.model
        logger.info(f"✅ Reranker model: {model_name}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error in reranker: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing reranker: {e}")
        return False

def test_system_integration():
    """Test overall system integration"""
    try:
        from app.main import initialize_system, load_documents
        
        # Test system initialization
        client, storage_context = initialize_system()
        
        if client is None or storage_context is None:
            logger.error("❌ System initialization failed")
            return False
        
        logger.info("✅ System initialization successful")
        
        # Test document loading
        documents = load_documents("data")
        logger.info(f"✅ Loaded {len(documents)} documents from data directory")
        
        # Test that documents have contextual chunking metadata
        if documents:
            doc = documents[0]
            assert "chunk_type" in doc.metadata
            assert doc.metadata["chunk_type"] == "contextual"
            logger.info("✅ Contextual chunking metadata validation passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in system integration test: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Enhanced RAG System")
    print("=" * 50)
    
    tests = [
        ("Docling Contextual Chunking", test_docling_chunking),
        ("Reranker", test_reranker),
        ("System Integration", test_system_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            start_time = time.time()
            result = test_func()
            elapsed = time.time() - start_time
            
            if result:
                print(f"✅ {test_name}: PASSED ({elapsed:.2f}s)")
                results.append(True)
            else:
                print(f"❌ {test_name}: FAILED ({elapsed:.2f}s)")
                results.append(False)
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! Enhanced RAG system is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    exit(main())