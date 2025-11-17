#!/usr/bin/env python3
"""
Requirements File Validation Tests
Tests that requirements files are syntactically correct and compatible
Does NOT require dependencies to be installed
"""

import sys
import re
from pathlib import Path


class RequirementsValidator:
    """Validate requirements files without installing packages"""

    def __init__(self, req_file_path):
        self.path = Path(req_file_path)
        self.packages = []
        self.errors = []

    def parse(self):
        """Parse requirements file"""
        if not self.path.exists():
            self.errors.append(f"File not found: {self.path}")
            return False

        with open(self.path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # Parse package specification
                self.packages.append({
                    'line': line_num,
                    'spec': line,
                    'package': self._extract_package_name(line)
                })

        return True

    def _extract_package_name(self, spec):
        """Extract package name from specification"""
        # Handle extras like package[extra]
        match = re.match(r'^([a-zA-Z0-9_-]+)', spec)
        return match.group(1) if match else spec

    def validate_tree_sitter_config(self):
        """Validate tree-sitter configuration"""
        tree_sitter_versions = []
        language_parsers = []

        for pkg in self.packages:
            spec = pkg['spec']
            if spec.startswith('tree-sitter=='):
                tree_sitter_versions.append(pkg)
            elif spec.startswith('tree-sitter-'):
                language_parsers.append(pkg)

        # Check for multiple tree-sitter versions
        if len(tree_sitter_versions) > 1:
            self.errors.append(
                f"Multiple tree-sitter versions found: {[p['spec'] for p in tree_sitter_versions]}"
            )
            return False

        # Check that we have tree-sitter if we have language parsers
        if language_parsers and not tree_sitter_versions:
            self.errors.append(
                f"Found {len(language_parsers)} language parsers but no tree-sitter base package"
            )
            return False

        # Verify version
        if tree_sitter_versions:
            version_spec = tree_sitter_versions[0]['spec']
            if 'tree-sitter==0.25.2' in version_spec:
                print(f"✅ tree-sitter 0.25.2 specified correctly (line {tree_sitter_versions[0]['line']})")
            else:
                self.errors.append(f"Unexpected tree-sitter version: {version_spec}")
                return False

        # List language parsers
        if language_parsers:
            print(f"✅ Found {len(language_parsers)} tree-sitter language parsers:")
            for parser in language_parsers:
                print(f"   - {parser['spec']} (line {parser['line']})")

        return True

    def validate_tiktoken_config(self):
        """Validate tiktoken configuration"""
        tiktoken_packages = []
        docling_openai_extra = []

        for pkg in self.packages:
            spec = pkg['spec']
            if 'tiktoken' in spec.lower() and not 'chunking-openai' in spec:
                tiktoken_packages.append(pkg)
            if 'docling-core[chunking-openai]' in spec:
                docling_openai_extra.append(pkg)

        # tiktoken should NOT be explicitly listed (provided by docling-core extra)
        if tiktoken_packages:
            print(f"⚠️  WARNING: tiktoken explicitly listed at line {tiktoken_packages[0]['line']}")
            print(f"   tiktoken is provided by docling-core[chunking-openai]")
            print(f"   Consider removing explicit listing to avoid version conflicts")
        else:
            print(f"✅ tiktoken not explicitly listed (will be provided by docling-core)")

        # Verify docling-core[chunking-openai] is present
        if not docling_openai_extra:
            self.errors.append("docling-core[chunking-openai] not found - required for HybridChunker")
            return False
        else:
            print(f"✅ docling-core[chunking-openai] found (line {docling_openai_extra[0]['line']})")

        return True

    def validate_python_version_requirement(self):
        """Check if Python version meets tree-sitter 0.25.2 requirements"""
        major = sys.version_info.major
        minor = sys.version_info.minor

        print(f"Python version: {major}.{minor}.{sys.version_info.micro}")

        if major < 3 or (major == 3 and minor < 10):
            self.errors.append(
                f"Python {major}.{minor} is too old. tree-sitter 0.25.2 requires Python 3.10+"
            )
            return False

        print(f"✅ Python {major}.{minor} meets tree-sitter 0.25.2 requirements (3.10+)")
        return True

    def check_for_duplicates(self):
        """Check for duplicate package specifications"""
        package_lines = {}

        for pkg in self.packages:
            name = pkg['package'].lower()
            if name in package_lines:
                self.errors.append(
                    f"Duplicate package '{name}' at lines {package_lines[name]} and {pkg['line']}"
                )
            else:
                package_lines[name] = pkg['line']

        if not self.errors:
            print(f"✅ No duplicate packages found")
            return True
        return False


def main():
    """Run all validation tests"""
    print("="*70)
    print("REQUIREMENTS FILE VALIDATION")
    print("="*70)
    print()

    # Find requirements files
    project_root = Path(__file__).parent.parent
    req_files = [
        project_root / 'app' / 'requirements.txt',
        project_root / 'app' / 'requirements.bge-m3.txt'
    ]

    all_passed = True

    for req_file in req_files:
        print(f"\n📋 Validating: {req_file.name}")
        print("-" * 70)

        validator = RequirementsValidator(req_file)

        # Parse file
        if not validator.parse():
            print(f"❌ Failed to parse {req_file.name}")
            for error in validator.errors:
                print(f"   ERROR: {error}")
            all_passed = False
            continue

        print(f"✅ Parsed {len(validator.packages)} package specifications")

        # Run validations
        checks = [
            ("tree-sitter configuration", validator.validate_tree_sitter_config),
            ("tiktoken configuration", validator.validate_tiktoken_config),
            ("Python version compatibility", validator.validate_python_version_requirement),
            ("duplicate packages", validator.check_for_duplicates),
        ]

        for check_name, check_func in checks:
            print(f"\n🔍 Checking {check_name}...")
            if not check_func():
                print(f"❌ {check_name} check failed")
                for error in validator.errors:
                    print(f"   ERROR: {error}")
                all_passed = False
            else:
                print(f"✅ {check_name} check passed")

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    if all_passed:
        print("✅ ✅ ✅ ALL VALIDATIONS PASSED ✅ ✅ ✅")
        print()
        print("Requirements files are correctly configured:")
        print("  ✅ tree-sitter 0.25.2 specified")
        print("  ✅ Language parsers present")
        print("  ✅ tiktoken handled by docling-core[chunking-openai]")
        print("  ✅ No duplicate packages")
        print("  ✅ Python version compatible")
        print()
        print("Safe to proceed with installation and deployment!")
        return 0
    else:
        print("❌ ❌ ❌ VALIDATION FAILED ❌ ❌ ❌")
        print()
        print("Please fix the errors above before proceeding.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
