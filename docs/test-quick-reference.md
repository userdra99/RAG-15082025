# Test Strategy Quick Reference

## 📋 Overview
**Total Test Cases**: 42
**Test Phases**: 5
**Estimated Duration**: 2 hours 45 minutes
**Priority**: CRITICAL

---

## 🎯 Critical Test Areas

### 1️⃣ Dependency Verification (15 min)
- ✅ tree-sitter 0.25.2 import
- ✅ HybridChunker availability
- ✅ tiktoken fallback
- ✅ Language parsers (Python, Java, JS, etc.)
- ✅ System initialization

### 2️⃣ HybridChunker Functionality (30 min)
- ✅ Python code parsing
- ✅ Java code parsing
- ✅ JavaScript code parsing
- ✅ Token counting (512 max)
- ✅ Fallback to SentenceSplitter

### 3️⃣ Document Processing (45 min)
- ✅ PDF with HybridChunker
- ✅ PDF fallback
- ✅ DOCX with HybridChunker
- ✅ DOCX fallback
- ✅ Excel token-aware chunking
- ✅ Excel character-based fallback

### 4️⃣ BGE-M3 Integration (45 min)
- ✅ Service connectivity
- ✅ 1024-dimensional embeddings
- ✅ Batch processing
- ✅ Qdrant storage (1024 dims)

### 5️⃣ End-to-End Pipeline (30 min)
- ✅ PDF upload → process → query
- ✅ DOCX upload → process → query
- ✅ Excel upload → process → query
- ✅ Multi-format processing
- ✅ Duplicate detection

---

## 🚨 Critical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| tiktoken import fails | MEDIUM | HIGH | Character-based fallback |
| HybridChunker fails | LOW | HIGH | SentenceSplitter fallback |
| tree-sitter incompatible | LOW | CRITICAL | Pinned to 0.25.2 |
| BGE-M3 dimension mismatch | LOW | CRITICAL | Verify 1024 dims |

---

## 🔧 Quick Start Commands

### Service Health Check
```bash
bash tests/check_services.sh
```

### Run All Tests
```bash
bash tests/run_all_tests.sh
```

### Run Specific Phase
```bash
# Phase 1: Dependencies
pytest tests/test_dependency_imports.py -v

# Phase 2: Unit Tests
pytest tests/test_hybrid_chunker.py -v

# Phase 3: Integration
pytest tests/test_bge_m3_integration.py -v

# Phase 4: E2E
pytest tests/test_e2e_pipeline.py -v
```

---

## 📊 Success Criteria

- ✅ Unit Test Coverage: >90%
- ✅ Integration Coverage: >80%
- ✅ E2E Coverage: >70%
- ✅ Test Pass Rate: >95%
- ✅ Document Processing: <30s per 100 pages
- ✅ Query Response: <2s

---

## 🔄 Rollback Triggers

- ⚠️ >20% critical test failures
- ⚠️ BGE-M3 integration broken
- ⚠️ Data loss/corruption
- ⚠️ Performance degradation >50%

---

## 📦 Test Files Created

1. **Strategy Document**: `/home/dra/rag-zero7-new/RAG-15082025/docs/test-strategy-dependency-changes.md`
2. **Execution Summary**: `/home/dra/rag-zero7-new/RAG-15082025/docs/test-execution-summary.json`
3. **Quick Reference**: `/home/dra/rag-zero7-new/RAG-15082025/docs/test-quick-reference.md`

---

## 🎯 Test Execution Order

1. **Start Services** (Qdrant, vLLM LLM, vLLM BGE-M3, Flask)
2. **Health Check** (verify all services running)
3. **Phase 1** (Dependencies - 15 min)
4. **Phase 2** (Unit Tests - 30 min)
5. **Phase 3** (Integration - 45 min)
6. **Phase 4** (E2E - 30 min)
7. **Phase 5** (Performance - 45 min)
8. **Generate Reports**

---

## 📝 Key Files to Monitor

**Dependencies:**
- `/home/dra/rag-zero7-new/RAG-15082025/app/requirements.txt`
- `/home/dra/rag-zero7-new/RAG-15082025/app/requirements.bge-m3.txt`

**Application:**
- `/home/dra/rag-zero7-new/RAG-15082025/app/main.py`

**Existing Tests:**
- `/home/dra/rag-zero7-new/RAG-15082025/test_bge_m3.py`
- `/home/dra/rag-zero7-new/RAG-15082025/test_simple_bge_m3.py`

---

## 🆘 Troubleshooting

### Import Errors
```python
# Check tree-sitter version
import tree_sitter
print(tree_sitter.__version__)  # Should be 0.25.2

# Check HybridChunker
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

# Check tiktoken (optional)
try:
    import tiktoken
except ImportError:
    print("tiktoken not available - using fallback")
```

### Service Issues
```bash
# Check service status
curl http://localhost:6333/collections  # Qdrant
curl http://localhost:8001/v1/models   # LLM
curl http://localhost:8002/v1/models   # BGE-M3
curl http://localhost:5000/health      # App
```

---

## ✅ Next Actions

1. ⏳ Execute Phase 1: Dependency Verification
2. ⏳ Set up test environment
3. ⏳ Prepare test data
4. ⏳ Run automated test suite
5. ⏳ Generate test reports
6. ⏳ Address critical failures
7. ⏳ Validate performance

---

**Document Version**: 1.0
**Created**: 2025-11-17
**Author**: Tester Agent (Swarm)
