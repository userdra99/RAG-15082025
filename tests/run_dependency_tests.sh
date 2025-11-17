#!/bin/bash
# Dependency Verification Test Runner
# Runs Phase 1 and Phase 2 critical tests

set -e  # Exit on error

echo "=================================================="
echo "DEPENDENCY VERIFICATION TEST SUITE"
echo "=================================================="
echo "Testing: tree-sitter 0.25.2, tiktoken, HybridChunker"
echo "Date: $(date)"
echo "Python: $(python3 --version)"
echo "=================================================="
echo ""

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  WARNING: Not running in a virtual environment"
    echo "Looking for virtual environment..."

    if [ -d "venv" ]; then
        echo "Found venv/, activating..."
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        echo "Found .venv/, activating..."
        source .venv/bin/activate
    elif [ -d "env" ]; then
        echo "Found env/, activating..."
        source env/bin/activate
    else
        echo "❌ No virtual environment found. Please activate your venv first."
        exit 1
    fi
fi

echo "✅ Virtual environment active: $VIRTUAL_ENV"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Run the dependency verification tests
echo "🧪 Running Dependency Verification Tests..."
echo ""

python3 tests/test_dependency_verification.py

TEST_EXIT_CODE=$?

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ ✅ ✅ ALL TESTS PASSED ✅ ✅ ✅"
    echo ""
    echo "Dependency changes are VERIFIED and SAFE for deployment:"
    echo "  ✅ tree-sitter 0.25.2 working correctly"
    echo "  ✅ All 9 language parsers loaded"
    echo "  ✅ tiktoken available via docling-core"
    echo "  ✅ HybridChunker initialized successfully"
    echo "  ✅ Fallback mechanisms verified"
    echo ""
    echo "Ready to commit changes!"
    exit 0
else
    echo "❌ ❌ ❌ TESTS FAILED ❌ ❌ ❌"
    echo ""
    echo "Please review the errors above before committing."
    exit 1
fi
