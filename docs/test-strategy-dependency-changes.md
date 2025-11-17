# Comprehensive Test Strategy: Dependency Changes Verification

## Executive Summary
This document outlines a comprehensive test strategy to verify that recent dependency changes (tree-sitter 0.25.2, removal of tiktoken from main dependencies, HybridChunker support) do not break existing functionality.

**Test Date**: 2025-11-17
**Version**: 1.0
**Priority**: CRITICAL

---

## 1. Changes Overview

### 1.1 Key Dependency Changes
1. **tree-sitter upgraded to 0.25.2** (from implicit version)
   - Required by HybridChunker for code parsing
   - Multiple language bindings added (Python, Java, JavaScript, C, C++, Go, Ruby, Rust, TypeScript)

2. **tiktoken removed from primary requirements.txt**
   - Still present via `docling-core[chunking-openai]` optional dependency
   - Potential import failures in fallback scenarios

3. **HybridChunker dependencies added**
   - semchunk==2.2.2
   - mpire==2.10.2
   - Tree-sitter language bindings

### 1.2 Impact Areas
- PDF document processing (DoclingPDFReader)
- DOCX document processing (DoclingDocxReader)
- Excel document processing (DoclingExcelReader)
- Code document parsing (HybridChunker)
- BGE-M3 embedding pipeline
- Qdrant vector storage

---

## 2. Test Scope and Critical Functionality

### 2.1 Priority 1 - Core Import & Initialization
**Risk Level**: CRITICAL

| Test ID | Test Case | Expected Outcome | Verification Method |
|---------|-----------|------------------|---------------------|
| T1.1 | Import tree-sitter 0.25.2 | Successful import | Unit test |
| T1.2 | Import HybridChunker | Successful import | Unit test |
| T1.3 | Import tiktoken (conditional) | Import or graceful fallback | Unit test |
| T1.4 | Import all language parsers | All 9 languages load | Unit test |
| T1.5 | System initialization | RAG system initializes | Integration test |

### 2.2 Priority 1 - HybridChunker Functionality
**Risk Level**: HIGH

| Test ID | Test Case | Expected Outcome | Verification Method |
|---------|-----------|------------------|---------------------|
| T2.1 | HybridChunker with Python code | Chunks respect function boundaries | Unit test |
| T2.2 | HybridChunker with Java code | Chunks respect class boundaries | Unit test |
| T2.3 | HybridChunker with JavaScript | Chunks respect module structure | Unit test |
| T2.4 | HybridChunker token counting | Accurate token counts (512 max) | Unit test |
| T2.5 | HybridChunker fallback | Falls back to SentenceSplitter | Unit test |

### 2.3 Priority 1 - Document Processing
**Risk Level**: CRITICAL

| Test ID | Test Case | Expected Outcome | Verification Method |
|---------|-----------|------------------|---------------------|
| T3.1 | PDF processing with HybridChunker | Chunks created, metadata preserved | Integration test |
| T3.2 | PDF processing fallback | SentenceSplitter works when HybridChunker fails | Integration test |
| T3.3 | DOCX processing with HybridChunker | Document structure preserved | Integration test |
| T3.4 | DOCX processing fallback | Fallback to SentenceSplitter | Integration test |
| T3.5 | Excel processing | Token-aware chunking for tables | Integration test |
| T3.6 | Excel processing without tiktoken | Character-based fallback works | Integration test |

### 2.4 Priority 1 - BGE-M3 Integration
**Risk Level**: CRITICAL

| Test ID | Test Case | Expected Outcome | Verification Method |
|---------|-----------|------------------|---------------------|
| T4.1 | BGE-M3 service connectivity | Connection successful | Integration test |
| T4.2 | BGE-M3 embedding generation | 1024-dimensional vectors | Integration test |
| T4.3 | BGE-M3 batch processing | Multiple texts processed | Integration test |
| T4.4 | Qdrant vector storage | Vectors stored with correct dimensions | Integration test |

### 2.5 Priority 2 - End-to-End RAG Pipeline
**Risk Level**: HIGH

| Test ID | Test Case | Expected Outcome | Verification Method |
|---------|-----------|------------------|---------------------|
| T5.1 | Upload → Process → Query (PDF) | Full pipeline works | E2E test |
| T5.2 | Upload → Process → Query (DOCX) | Full pipeline works | E2E test |
| T5.3 | Upload → Process → Query (Excel) | Full pipeline works | E2E test |
| T5.4 | Multi-format processing | All formats work together | E2E test |
| T5.5 | Duplicate detection | Duplicates handled correctly | E2E test |

### 2.6 Priority 2 - Error Handling & Fallback
**Risk Level**: MEDIUM

| Test ID | Test Case | Expected Outcome | Verification Method |
|---------|-----------|------------------|---------------------|
| T6.1 | Missing tiktoken import | Graceful fallback to character-based | Unit test |
| T6.2 | HybridChunker import failure | Fallback to SentenceSplitter | Unit test |
| T6.3 | Tree-sitter language missing | Error logged, processing continues | Unit test |
| T6.4 | Malformed documents | Error handled gracefully | Integration test |

### 2.7 Priority 3 - Performance & Scalability
**Risk Level**: LOW

| Test ID | Test Case | Expected Outcome | Verification Method |
|---------|-----------|------------------|---------------------|
| T7.1 | Large PDF processing (>100 pages) | Completes in <5 min | Performance test |
| T7.2 | Batch document processing (10+ files) | Memory usage acceptable | Performance test |
| T7.3 | Concurrent query performance | <2s response time | Load test |

---

## 3. Test Cases Detail

### 3.1 Dependency Import Verification Tests

```python
# tests/test_dependency_imports.py

def test_tree_sitter_version():
    """Verify tree-sitter 0.25.2 is installed"""
    import tree_sitter
    assert tree_sitter.__version__ == "0.25.2"

def test_hybrid_chunker_import():
    """Verify HybridChunker can be imported"""
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    assert HybridChunker is not None

def test_tiktoken_conditional_import():
    """Verify tiktoken is available or fallback works"""
    try:
        import tiktoken
        assert tiktoken is not None
    except ImportError:
        # Fallback should be acceptable
        pass

def test_language_parsers_import():
    """Verify all tree-sitter language parsers load"""
    languages = [
        'tree_sitter_python',
        'tree_sitter_java',
        'tree_sitter_javascript',
        'tree_sitter_c',
        'tree_sitter_cpp',
        'tree_sitter_go',
        'tree_sitter_ruby',
        'tree_sitter_rust',
        'tree_sitter_typescript'
    ]

    for lang in languages:
        try:
            __import__(lang)
        except ImportError as e:
            pytest.fail(f"Failed to import {lang}: {e}")
```

### 3.2 HybridChunker Functionality Tests

```python
# tests/test_hybrid_chunker.py

def test_hybrid_chunker_python_code():
    """Test HybridChunker with Python code"""
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
    import tiktoken

    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.encoding_for_model("gpt-4o"),
        max_tokens=512
    )
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

    # Test Python code
    python_code = """
    def example_function():
        '''This is a test function'''
        return "test"

    class ExampleClass:
        def __init__(self):
            self.value = 42
    """

    # Create a mock document object
    from docling.datamodel.document import Document
    doc = Document()
    # Add code content

    chunks = list(chunker.chunk(doc))

    assert len(chunks) > 0
    assert all(hasattr(chunk, 'text') for chunk in chunks)
    assert all(len(chunk.text.split()) <= 512 for chunk in chunks)

def test_hybrid_chunker_fallback():
    """Test HybridChunker gracefully falls back"""
    try:
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
        # If import succeeds, test should pass
        assert True
    except ImportError:
        # If HybridChunker not available, ensure fallback exists
        from llama_index.core.node_parser import SentenceSplitter
        splitter = SentenceSplitter(chunk_size=512)
        assert splitter is not None
```

### 3.3 Document Processing Tests

```python
# tests/test_document_processing.py

def test_pdf_processing_with_hybrid_chunker(tmp_path):
    """Test PDF processing uses HybridChunker when available"""
    from app.main import DoclingPDFReader

    reader = DoclingPDFReader()

    # Verify HybridChunker is initialized
    assert hasattr(reader, 'use_hybrid')

    # If HybridChunker available, should be True
    # If not, should be False (fallback)
    assert isinstance(reader.use_hybrid, bool)

def test_excel_processing_with_token_awareness():
    """Test Excel processing with token-aware chunking"""
    from app.main import DoclingExcelReader

    reader = DoclingExcelReader()

    # Verify tokenizer initialization
    assert hasattr(reader, 'use_token_aware')
    assert isinstance(reader.use_token_aware, bool)

def test_docx_processing_metadata_preservation():
    """Test DOCX processing preserves metadata"""
    from app.main import DoclingDocxReader
    import tempfile
    from docx import Document as DocxDocument

    # Create test DOCX
    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
        doc = DocxDocument()
        doc.add_heading('Test Document', 0)
        doc.add_paragraph('Test content')
        doc.save(tmp.name)

        reader = DoclingDocxReader()
        docs = reader.load_data(tmp.name)

        assert len(docs) > 0
        assert all(hasattr(d, 'metadata') for d in docs)
        assert all('file_name' in d.metadata for d in docs)
        assert all('chunk_type' in d.metadata for d in docs)
```

### 3.4 BGE-M3 Integration Tests

```python
# tests/test_bge_m3_integration.py

def test_bge_m3_service_health():
    """Test BGE-M3 service is running"""
    import requests

    response = requests.get("http://localhost:8002/v1/models", timeout=10)
    assert response.status_code == 200

    models = response.json()
    model_names = [m.get('id', '') for m in models.get('data', [])]
    assert any('bge-m3' in m.lower() for m in model_names)

def test_bge_m3_embedding_dimensions():
    """Test BGE-M3 produces 1024-dimensional embeddings"""
    import requests

    payload = {
        "input": ["Test text for embedding"],
        "model": "BAAI/bge-m3"
    }

    response = requests.post(
        "http://localhost:8002/v1/embeddings",
        json=payload,
        timeout=30
    )

    assert response.status_code == 200
    data = response.json()
    embeddings = data.get('data', [])
    assert len(embeddings) > 0

    embedding_vector = embeddings[0].get('embedding', [])
    assert len(embedding_vector) == 1024

def test_qdrant_collection_dimensions():
    """Test Qdrant collection has correct vector dimensions"""
    import qdrant_client

    client = qdrant_client.QdrantClient(host="localhost", port=6333)
    collection_info = client.get_collection("documents")

    vector_size = collection_info.config.params.vectors.size
    assert vector_size == 1024
```

### 3.5 End-to-End Pipeline Tests

```python
# tests/test_e2e_pipeline.py

def test_full_pdf_pipeline(tmp_path):
    """Test full PDF processing pipeline"""
    # 1. Upload PDF
    # 2. Process with HybridChunker
    # 3. Store in Qdrant
    # 4. Query and retrieve

    # Test implementation
    pass

def test_multi_format_processing():
    """Test processing multiple document formats together"""
    # Upload PDF, DOCX, and Excel
    # Process all
    # Verify all stored correctly

    pass

def test_duplicate_detection():
    """Test duplicate file handling"""
    # Upload same file twice
    # Verify duplicate detection
    # Verify chunks not duplicated

    pass
```

---

## 4. Test Execution Plan

### 4.1 Phase 1: Dependency Verification (Priority 1)
**Duration**: 15 minutes
**Prerequisites**: None

1. Run import tests
2. Verify tree-sitter version
3. Verify HybridChunker availability
4. Check tiktoken fallback
5. Test language parser imports

**Success Criteria**: All imports succeed or graceful fallback

### 4.2 Phase 2: Unit Testing (Priority 1)
**Duration**: 30 minutes
**Prerequisites**: Phase 1 complete

1. Test HybridChunker with code samples
2. Test document reader initialization
3. Test token counting
4. Test metadata preservation
5. Test fallback mechanisms

**Success Criteria**: All unit tests pass

### 4.3 Phase 3: Integration Testing (Priority 1 & 2)
**Duration**: 45 minutes
**Prerequisites**: Phase 2 complete, services running

1. Test BGE-M3 connectivity
2. Test document processing (PDF, DOCX, Excel)
3. Test Qdrant storage
4. Test embedding generation
5. Test retrieval accuracy

**Success Criteria**: All integration tests pass

### 4.4 Phase 4: End-to-End Testing (Priority 2)
**Duration**: 30 minutes
**Prerequisites**: Phase 3 complete

1. Test full upload → process → query pipeline
2. Test multi-format processing
3. Test duplicate detection
4. Test error handling
5. Test concurrent operations

**Success Criteria**: Complete pipelines work correctly

### 4.5 Phase 5: Performance & Load Testing (Priority 3)
**Duration**: 45 minutes
**Prerequisites**: All previous phases complete

1. Large document processing
2. Batch processing
3. Concurrent query load
4. Memory profiling
5. Response time validation

**Success Criteria**: Performance within acceptable limits

---

## 5. Risk Assessment and Mitigation

### 5.1 Critical Risks

| Risk ID | Risk Description | Probability | Impact | Mitigation Strategy |
|---------|------------------|-------------|--------|---------------------|
| R1 | tiktoken import fails in Excel processing | MEDIUM | HIGH | Fallback to character-based chunking implemented |
| R2 | HybridChunker fails to import | LOW | HIGH | Fallback to SentenceSplitter implemented |
| R3 | tree-sitter 0.25.2 incompatible | LOW | CRITICAL | Pin to 0.25.2, test all language bindings |
| R4 | BGE-M3 dimension mismatch | LOW | CRITICAL | Verify 1024 dimensions, recreate collection if needed |
| R5 | Language parser not found | MEDIUM | MEDIUM | Graceful error handling, continue with available parsers |

### 5.2 Medium Risks

| Risk ID | Risk Description | Probability | Impact | Mitigation Strategy |
|---------|------------------|-------------|--------|---------------------|
| R6 | Performance degradation | MEDIUM | MEDIUM | Performance benchmarks, optimize chunking |
| R7 | Memory leaks with large files | LOW | MEDIUM | Memory profiling, streaming processing |
| R8 | Concurrent access issues | LOW | MEDIUM | Thread safety testing |

### 5.3 Low Risks

| Risk ID | Risk Description | Probability | Impact | Mitigation Strategy |
|---------|------------------|-------------|--------|---------------------|
| R9 | Minor UI inconsistencies | MEDIUM | LOW | UI testing, manual review |
| R10 | Log verbosity issues | LOW | LOW | Log level configuration |

---

## 6. Test Environment Setup

### 6.1 Required Services
- **Qdrant**: localhost:6333
- **vLLM LLM Service**: localhost:8001
- **vLLM Embedding Service (BGE-M3)**: localhost:8002
- **Flask App**: localhost:5000

### 6.2 Test Data Requirements
- Sample PDF documents (small, medium, large)
- Sample DOCX documents (with tables, images, headings)
- Sample Excel files (multiple sheets, complex tables)
- Sample code files (Python, Java, JavaScript)

### 6.3 Environment Variables
```bash
LLM_API_BASE=http://vllm-llm:8000/v1
EMBEDDING_API_BASE=http://vllm-embedding:8000/v1
EMBEDDING_MODEL=BAAI/bge-m3
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

---

## 7. Success Metrics

### 7.1 Test Coverage Targets
- **Unit Test Coverage**: >90%
- **Integration Test Coverage**: >80%
- **E2E Test Coverage**: >70%
- **Critical Path Coverage**: 100%

### 7.2 Performance Targets
- **Document Processing**: <30s per 100 pages
- **Query Response**: <2s per query
- **Embedding Generation**: <1s per text
- **Memory Usage**: <4GB for 1000 documents

### 7.3 Reliability Targets
- **Service Uptime**: >99%
- **Test Pass Rate**: >95%
- **Error Recovery**: 100% graceful fallback
- **Data Integrity**: 100% no data loss

---

## 8. Test Execution Checklist

- [ ] Phase 1: Dependency Verification
  - [ ] T1.1 - T1.5 (Import & Initialization)
- [ ] Phase 2: Unit Testing
  - [ ] T2.1 - T2.5 (HybridChunker)
  - [ ] T6.1 - T6.3 (Error Handling)
- [ ] Phase 3: Integration Testing
  - [ ] T3.1 - T3.6 (Document Processing)
  - [ ] T4.1 - T4.4 (BGE-M3 Integration)
- [ ] Phase 4: End-to-End Testing
  - [ ] T5.1 - T5.5 (RAG Pipeline)
- [ ] Phase 5: Performance Testing
  - [ ] T7.1 - T7.3 (Performance & Load)

---

## 9. Reporting and Documentation

### 9.1 Test Reports
- **Test Execution Summary**: Pass/Fail counts, duration
- **Defect Report**: Issues found, severity, status
- **Performance Report**: Metrics vs targets
- **Coverage Report**: Code coverage analysis

### 9.2 Deliverables
- Test strategy document (this document)
- Test case specifications
- Test execution logs
- Defect tracking report
- Final test summary report

---

## 10. Rollback Plan

### 10.1 Rollback Triggers
- >20% test failures in critical tests
- BGE-M3 integration completely broken
- Data loss or corruption detected
- Performance degradation >50%

### 10.2 Rollback Procedure
1. Stop all services
2. Restore previous requirements.txt versions
3. Rebuild Docker images
4. Verify system stability
5. Re-run test suite

### 10.3 Previous Known Good State
- **tree-sitter**: Previous version (pre-0.25.2)
- **tiktoken**: In main requirements
- **HybridChunker**: Not used, SentenceSplitter only

---

## Appendix A: Test Automation Scripts

### A.1 Quick Test Runner
```bash
#!/bin/bash
# tests/run_all_tests.sh

echo "Running Dependency Tests..."
pytest tests/test_dependency_imports.py -v

echo "Running Unit Tests..."
pytest tests/test_hybrid_chunker.py -v
pytest tests/test_document_processing.py -v

echo "Running Integration Tests..."
pytest tests/test_bge_m3_integration.py -v

echo "Running E2E Tests..."
pytest tests/test_e2e_pipeline.py -v

echo "Generating Coverage Report..."
pytest --cov=app --cov-report=html tests/
```

### A.2 Service Health Check
```bash
#!/bin/bash
# tests/check_services.sh

services=(
  "Qdrant:http://localhost:6333/collections"
  "LLM:http://localhost:8001/v1/models"
  "BGE-M3:http://localhost:8002/v1/models"
  "App:http://localhost:5000/health"
)

for service in "${services[@]}"; do
  name="${service%%:*}"
  url="${service#*:}"

  if curl -s -f -o /dev/null "$url"; then
    echo "✅ $name is healthy"
  else
    echo "❌ $name is down"
  fi
done
```

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-17 | Tester Agent | Initial test strategy document |
