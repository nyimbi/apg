#!/usr/bin/env python3
"""
APG Event Streaming Bus - Test Runner

Comprehensive test runner for Event Streaming Bus capability with different
test categories and reporting options.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

def run_command(cmd: List[str], description: str) -> int:
	"""Run a command and return the exit code."""
	print(f"\n{'='*60}")
	print(f"Running: {description}")
	print(f"Command: {' '.join(cmd)}")
	print(f"{'='*60}")
	
	try:
		result = subprocess.run(cmd, check=False)
		return result.returncode
	except Exception as e:
		print(f"Error running command: {e}")
		return 1

def run_unit_tests(verbose: bool = False, coverage: bool = True) -> int:
	"""Run unit tests."""
	cmd = ["python", "-m", "pytest", "tests/unit/", "-m", "unit"]
	
	if verbose:
		cmd.append("-v")
	
	if coverage:
		cmd.extend(["--cov=.", "--cov-report=term-missing"])
	
	return run_command(cmd, "Unit Tests")

def run_integration_tests(verbose: bool = False) -> int:
	"""Run integration tests."""
	cmd = ["python", "-m", "pytest", "tests/integration/", "-m", "integration"]
	
	if verbose:
		cmd.append("-v")
	
	# Integration tests typically don't need coverage
	cmd.append("--no-cov")
	
	return run_command(cmd, "Integration Tests")

def run_performance_tests(verbose: bool = False) -> int:
	"""Run performance tests."""
	cmd = ["python", "-m", "pytest", "tests/performance/", "-m", "performance"]
	
	if verbose:
		cmd.append("-v")
	
	# Performance tests don't need coverage
	cmd.extend(["--no-cov", "--durations=0"])
	
	return run_command(cmd, "Performance Tests")

def run_all_tests(verbose: bool = False, coverage: bool = True) -> int:
	"""Run all test categories."""
	total_failures = 0
	
	# Run unit tests
	print("\n" + "="*80)
	print("RUNNING ALL TESTS - UNIT, INTEGRATION, AND PERFORMANCE")
	print("="*80)
	
	unit_result = run_unit_tests(verbose, coverage)
	total_failures += unit_result
	
	integration_result = run_integration_tests(verbose)
	total_failures += integration_result
	
	performance_result = run_performance_tests(verbose)
	total_failures += performance_result
	
	# Summary
	print("\n" + "="*80)
	print("TEST RESULTS SUMMARY")
	print("="*80)
	print(f"Unit Tests: {'PASSED' if unit_result == 0 else 'FAILED'}")
	print(f"Integration Tests: {'PASSED' if integration_result == 0 else 'FAILED'}")
	print(f"Performance Tests: {'PASSED' if performance_result == 0 else 'FAILED'}")
	print(f"\nOverall: {'PASSED' if total_failures == 0 else 'FAILED'}")
	
	return 1 if total_failures > 0 else 0

def run_quick_tests(verbose: bool = False) -> int:
	"""Run quick tests (unit tests only, no slow tests)."""
	cmd = [
		"python", "-m", "pytest", 
		"tests/unit/", 
		"-m", "unit and not slow",
		"--tb=short"
	]
	
	if verbose:
		cmd.append("-v")
	
	return run_command(cmd, "Quick Tests (Unit Only, No Slow)")

def run_specific_test(test_path: str, verbose: bool = False) -> int:
	"""Run a specific test file or test function."""
	cmd = ["python", "-m", "pytest", test_path]
	
	if verbose:
		cmd.append("-v")
	
	return run_command(cmd, f"Specific Test: {test_path}")

def check_test_dependencies() -> bool:
	"""Check if required test dependencies are available."""
	required_packages = [
		"pytest", 
		"pytest-asyncio", 
		"pytest-cov", 
		"pytest-mock"
	]
	
	missing_packages = []
	
	for package in required_packages:
		try:
			__import__(package.replace("-", "_"))
		except ImportError:
			missing_packages.append(package)
	
	if missing_packages:
		print("Missing required test packages:")
		for package in missing_packages:
			print(f"  - {package}")
		print("\nInstall with: pip install " + " ".join(missing_packages))
		return False
	
	return True

def generate_test_report() -> int:
	"""Generate comprehensive test report with coverage."""
	cmd = [
		"python", "-m", "pytest",
		"tests/",
		"--cov=.",
		"--cov-report=html:coverage_html",
		"--cov-report=xml:coverage.xml",
		"--cov-report=term-missing",
		"--junit-xml=test_results.xml",
		"-v"
	]
	
	result = run_command(cmd, "Comprehensive Test Report Generation")
	
	if result == 0:
		print("\nTest reports generated:")
		print("  - HTML Coverage: coverage_html/index.html")
		print("  - XML Coverage: coverage.xml")
		print("  - JUnit XML: test_results.xml")
	
	return result

def lint_code() -> int:
	"""Run code linting and formatting checks."""
	print("\n" + "="*60)
	print("Running Code Quality Checks")
	print("="*60)
	
	# Check if ruff is available
	try:
		# Run ruff for linting
		ruff_result = run_command(["ruff", "check", "."], "Ruff Linting")
		
		# Run ruff for formatting
		format_result = run_command(["ruff", "format", "--check", "."], "Ruff Format Check")
		
		return max(ruff_result, format_result)
		
	except FileNotFoundError:
		print("Ruff not found. Install with: pip install ruff")
		return 1

def main():
	"""Main test runner entry point."""
	parser = argparse.ArgumentParser(
		description="APG Event Streaming Bus Test Runner",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  python run_tests.py                    # Run quick tests
  python run_tests.py --all              # Run all tests
  python run_tests.py --unit             # Run unit tests only
  python run_tests.py --integration      # Run integration tests only
  python run_tests.py --performance      # Run performance tests only
  python run_tests.py --specific tests/unit/test_models.py::TestESEvent::test_event_creation_with_defaults
  python run_tests.py --report           # Generate comprehensive report
  python run_tests.py --lint             # Run code quality checks
		"""
	)
	
	# Test categories
	parser.add_argument(
		"--all", action="store_true",
		help="Run all test categories (unit, integration, performance)"
	)
	parser.add_argument(
		"--unit", action="store_true",
		help="Run unit tests only"
	)
	parser.add_argument(
		"--integration", action="store_true",
		help="Run integration tests only"
	)
	parser.add_argument(
		"--performance", action="store_true",
		help="Run performance tests only"
	)
	parser.add_argument(
		"--quick", action="store_true",
		help="Run quick tests (unit tests, no slow tests)"
	)
	
	# Specific test execution
	parser.add_argument(
		"--specific", type=str,
		help="Run specific test file or test function"
	)
	
	# Reporting
	parser.add_argument(
		"--report", action="store_true",
		help="Generate comprehensive test report with coverage"
	)
	
	# Code quality
	parser.add_argument(
		"--lint", action="store_true",
		help="Run code linting and formatting checks"
	)
	
	# Options
	parser.add_argument(
		"-v", "--verbose", action="store_true",
		help="Verbose output"
	)
	parser.add_argument(
		"--no-cov", action="store_true",
		help="Disable coverage reporting"
	)
	
	args = parser.parse_args()
	
	# Check dependencies
	if not check_test_dependencies():
		return 1
	
	# Change to script directory
	script_dir = Path(__file__).parent
	sys.path.insert(0, str(script_dir))
	
	coverage = not args.no_cov
	
	# Execute based on arguments
	if args.lint:
		return lint_code()
	elif args.report:
		return generate_test_report()
	elif args.all:
		return run_all_tests(args.verbose, coverage)
	elif args.unit:
		return run_unit_tests(args.verbose, coverage)
	elif args.integration:
		return run_integration_tests(args.verbose)
	elif args.performance:
		return run_performance_tests(args.verbose)
	elif args.quick:
		return run_quick_tests(args.verbose)
	elif args.specific:
		return run_specific_test(args.specific, args.verbose)
	else:
		# Default: run quick tests
		return run_quick_tests(args.verbose)

if __name__ == "__main__":
	sys.exit(main())