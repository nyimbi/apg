#!/usr/bin/env python3
"""
Test runner for crawler tests.

This script provides a convenient way to run different types of tests
with various configurations and options.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    
    if capture_output:
        return subprocess.run(cmd, capture_output=True, text=True)
    else:
        return subprocess.run(cmd)


def get_test_directories() -> List[Path]:
    """Get all test directories."""
    test_root = Path(__file__).parent
    return [
        test_root / "unit",
        test_root / "integration", 
        test_root / "functional",
        test_root / "performance",
        test_root / "security",
        test_root / "usability"
    ]


def run_unit_tests(verbose: bool = False, coverage: bool = True) -> int:
    """Run unit tests."""
    cmd = ["python", "-m", "pytest", "unit/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=packages.crawlers", "--cov-report=term-missing"])
    
    cmd.extend(["-m", "unit"])
    
    result = run_command(cmd)
    return result.returncode


def run_integration_tests(verbose: bool = False) -> int:
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "integration/"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend(["-m", "integration"])
    
    result = run_command(cmd)
    return result.returncode


def run_functional_tests(verbose: bool = False) -> int:
    """Run functional tests."""
    cmd = ["python", "-m", "pytest", "functional/"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend(["-m", "functional"])
    
    result = run_command(cmd)
    return result.returncode


def run_performance_tests(verbose: bool = False) -> int:
    """Run performance tests."""
    cmd = ["python", "-m", "pytest", "performance/"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend(["-m", "performance", "--durations=0"])
    
    result = run_command(cmd)
    return result.returncode


def run_security_tests(verbose: bool = False) -> int:
    """Run security tests."""
    cmd = ["python", "-m", "pytest", "security/"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend(["-m", "security"])
    
    result = run_command(cmd)
    return result.returncode


def run_usability_tests(verbose: bool = False) -> int:
    """Run usability tests."""
    cmd = ["python", "-m", "pytest", "usability/"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend(["-m", "usability"])
    
    result = run_command(cmd)
    return result.returncode


def run_all_tests(verbose: bool = False, coverage: bool = True) -> int:
    """Run all tests."""
    cmd = ["python", "-m", "pytest", "."]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend([
            "--cov=packages.crawlers",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ])
    
    result = run_command(cmd)
    return result.returncode


def run_quick_tests(verbose: bool = False) -> int:
    """Run quick tests (unit + integration)."""
    cmd = ["python", "-m", "pytest", "unit/", "integration/"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend(["-m", "unit or integration"])
    
    result = run_command(cmd)
    return result.returncode


def run_slow_tests(verbose: bool = False) -> int:
    """Run slow tests (performance + functional)."""
    cmd = ["python", "-m", "pytest", "performance/", "functional/"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend(["-m", "performance or functional"])
    
    result = run_command(cmd)
    return result.returncode


def run_tests_by_marker(marker: str, verbose: bool = False) -> int:
    """Run tests by marker."""
    cmd = ["python", "-m", "pytest", "-m", marker]
    
    if verbose:
        cmd.append("-v")
    
    result = run_command(cmd)
    return result.returncode


def run_specific_test(test_path: str, verbose: bool = False) -> int:
    """Run a specific test file or test function."""
    cmd = ["python", "-m", "pytest", test_path]
    
    if verbose:
        cmd.append("-v")
    
    result = run_command(cmd)
    return result.returncode


def generate_coverage_report() -> int:
    """Generate coverage report."""
    cmd = ["python", "-m", "pytest", "--cov=packages.crawlers", "--cov-report=html:htmlcov"]
    
    result = run_command(cmd)
    if result.returncode == 0:
        print("Coverage report generated in htmlcov/")
        print("Open htmlcov/index.html in your browser to view the report")
    
    return result.returncode


def check_test_environment() -> bool:
    """Check if test environment is properly set up."""
    print("Checking test environment...")
    
    # Check if pytest is installed
    try:
        import pytest
        print(f"✓ pytest {pytest.__version__} is installed")
    except ImportError:
        print("✗ pytest is not installed. Run: pip install pytest")
        return False
    
    # Check if pytest-asyncio is installed
    try:
        import pytest_asyncio
        print(f"✓ pytest-asyncio {pytest_asyncio.__version__} is installed")
    except ImportError:
        print("✗ pytest-asyncio is not installed. Run: pip install pytest-asyncio")
        return False
    
    # Check if pytest-cov is installed
    try:
        import pytest_cov
        print(f"✓ pytest-cov {pytest_cov.__version__} is installed")
    except ImportError:
        print("✗ pytest-cov is not installed. Run: pip install pytest-cov")
        return False
    
    # Check if test directories exist
    for test_dir in get_test_directories():
        if test_dir.exists():
            print(f"✓ {test_dir.name} directory exists")
        else:
            print(f"✗ {test_dir.name} directory missing")
            return False
    
    print("✓ Test environment is properly set up")
    return True


def list_available_tests() -> None:
    """List all available tests."""
    print("Available test categories:")
    print("  unit        - Unit tests")
    print("  integration - Integration tests")
    print("  functional  - Functional tests")
    print("  performance - Performance tests")
    print("  security    - Security tests")
    print("  usability   - Usability tests")
    print("  all         - All tests")
    print("  quick       - Quick tests (unit + integration)")
    print("  slow        - Slow tests (performance + functional)")
    print()
    
    print("Available markers:")
    markers = [
        "unit", "integration", "functional", "performance", "security", "usability",
        "slow", "external", "network", "database", "cache", "stealth", "bypass",
        "parser", "crawler", "mock", "real", "regression"
    ]
    
    for marker in markers:
        print(f"  {marker}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for crawler tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py unit                    # Run unit tests
  python run_tests.py all --coverage          # Run all tests with coverage
  python run_tests.py performance --verbose   # Run performance tests with verbose output
  python run_tests.py --marker slow           # Run tests marked as 'slow'
  python run_tests.py --test unit/test_base_crawler.py  # Run specific test file
  python run_tests.py --check                 # Check test environment
  python run_tests.py --list                  # List available tests
        """
    )
    
    parser.add_argument(
        "test_type",
        nargs="?",
        choices=["unit", "integration", "functional", "performance", "security", "usability", "all", "quick", "slow"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Run tests in verbose mode"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        default=True,
        help="Generate coverage report (default: True)"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage report"
    )
    
    parser.add_argument(
        "--marker", "-m",
        help="Run tests with specific marker"
    )
    
    parser.add_argument(
        "--test", "-t",
        help="Run specific test file or function"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check test environment setup"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tests and markers"
    )
    
    parser.add_argument(
        "--coverage-report",
        action="store_true",
        help="Generate coverage report only"
    )
    
    args = parser.parse_args()
    
    # Handle coverage flag
    if args.no_coverage:
        args.coverage = False
    
    # Change to test directory
    os.chdir(Path(__file__).parent)
    
    # Handle special commands
    if args.check:
        if check_test_environment():
            return 0
        else:
            return 1
    
    if args.list:
        list_available_tests()
        return 0
    
    if args.coverage_report:
        return generate_coverage_report()
    
    # Run specific tests
    if args.test:
        return run_specific_test(args.test, args.verbose)
    
    if args.marker:
        return run_tests_by_marker(args.marker, args.verbose)
    
    # Run test categories
    test_runners = {
        "unit": run_unit_tests,
        "integration": run_integration_tests,
        "functional": run_functional_tests,
        "performance": run_performance_tests,
        "security": run_security_tests,
        "usability": run_usability_tests,
        "all": run_all_tests,
        "quick": run_quick_tests,
        "slow": run_slow_tests
    }
    
    runner = test_runners.get(args.test_type)
    if runner:
        if args.test_type in ["unit", "all"]:
            return runner(args.verbose, args.coverage)
        else:
            return runner(args.verbose)
    else:
        print(f"Unknown test type: {args.test_type}")
        return 1


if __name__ == "__main__":
    sys.exit(main())