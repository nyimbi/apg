#!/usr/bin/env python3
"""
APG Budgeting & Forecasting - Test Runner

Comprehensive test runner for the APG Budgeting & Forecasting capability
with support for different test categories and environments.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
import asyncio
import time


def setup_test_environment():
	"""Setup test environment variables and paths."""
	# Add the parent directory to Python path
	current_dir = Path(__file__).parent
	capability_dir = current_dir.parent
	sys.path.insert(0, str(capability_dir))
	
	# Set test environment variables
	os.environ["TESTING"] = "true"
	os.environ["APG_ENVIRONMENT"] = "test"
	os.environ["LOG_LEVEL"] = "WARNING"
	
	# Test database configuration
	if not os.getenv("TEST_DATABASE_URL"):
		os.environ["TEST_DATABASE_URL"] = "postgresql://test_user:test_pass@localhost:5432/test_apg_budgeting_forecasting"


def run_pytest_command(args: List[str]) -> int:
	"""Run pytest with the given arguments."""
	cmd = ["python", "-m", "pytest"] + args
	print(f"Running: {' '.join(cmd)}")
	print("-" * 60)
	
	try:
		result = subprocess.run(cmd, check=False)
		return result.returncode
	except KeyboardInterrupt:
		print("\nTest execution interrupted by user")
		return 130
	except Exception as e:
		print(f"Error running tests: {e}")
		return 1


def run_unit_tests(verbose: bool = False, coverage: bool = True) -> int:
	"""Run unit tests only."""
	args = ["-m", "unit"]
	
	if verbose:
		args.append("-v")
	
	if coverage:
		args.extend(["--cov=budgeting_forecasting", "--cov-report=term-missing"])
	
	args.append("tests/")
	return run_pytest_command(args)


def run_integration_tests(verbose: bool = False) -> int:
	"""Run integration tests only."""
	args = ["-m", "integration"]
	
	if verbose:
		args.append("-v")
	
	args.append("tests/")
	return run_pytest_command(args)


def run_performance_tests(verbose: bool = False) -> int:
	"""Run performance tests only."""
	args = ["-m", "performance"]
	
	if verbose:
		args.append("-v")
	
	args.append("tests/")
	return run_pytest_command(args)


def run_smoke_tests(verbose: bool = False) -> int:
	"""Run smoke tests for basic functionality."""
	args = ["-m", "smoke"]
	
	if verbose:
		args.append("-v")
	
	args.append("tests/")
	return run_pytest_command(args)


def run_all_tests(
	verbose: bool = False, 
	coverage: bool = True, 
	parallel: bool = False,
	fail_fast: bool = False
) -> int:
	"""Run all tests."""
	args = []
	
	if verbose:
		args.append("-v")
	
	if coverage:
		args.extend([
			"--cov=budgeting_forecasting",
			"--cov-report=html:tests/coverage_html",
			"--cov-report=term-missing",
			"--cov-report=xml:tests/coverage.xml"
		])
	
	if parallel:
		args.extend(["-n", "auto"])  # Requires pytest-xdist
	
	if fail_fast:
		args.append("-x")
	
	args.extend(["--junit-xml=tests/junit.xml", "tests/"])
	return run_pytest_command(args)


def run_specific_test(test_path: str, verbose: bool = False) -> int:
	"""Run a specific test file or test function."""
	args = []
	
	if verbose:
		args.append("-v")
	
	args.append(test_path)
	return run_pytest_command(args)


def check_test_dependencies() -> bool:
	"""Check if all test dependencies are available."""
	required_packages = [
		"pytest",
		"pytest-asyncio", 
		"pytest-cov",
		"pytest-httpserver",
		"responses"
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


async def check_test_database() -> bool:
	"""Check if test database is accessible."""
	try:
		import asyncpg
		db_url = os.getenv("TEST_DATABASE_URL")
		if not db_url:
			print("TEST_DATABASE_URL not set")
			return False
		
		conn = await asyncpg.connect(db_url)
		await conn.close()
		print("✓ Test database connection successful")
		return True
		
	except ImportError:
		print("Warning: asyncpg not available, skipping database check")
		return True
	except Exception as e:
		print(f"✗ Test database connection failed: {e}")
		print("Note: Some tests may fail without database connectivity")
		return False


def generate_test_report() -> None:
	"""Generate a comprehensive test report."""
	print("\n" + "=" * 60)
	print("APG Budgeting & Forecasting - Test Report")
	print("=" * 60)
	
	# Check for test artifacts
	coverage_html = Path("tests/coverage_html/index.html")
	junit_xml = Path("tests/junit.xml")
	coverage_xml = Path("tests/coverage.xml")
	
	if coverage_html.exists():
		print(f"📊 Coverage Report: {coverage_html.absolute()}")
	
	if junit_xml.exists():
		print(f"📋 JUnit Report: {junit_xml.absolute()}")
	
	if coverage_xml.exists():
		print(f"📈 Coverage XML: {coverage_xml.absolute()}")
	
	print("=" * 60)


def main():
	"""Main test runner function."""
	parser = argparse.ArgumentParser(
		description="APG Budgeting & Forecasting Test Runner",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  python run_tests.py --all                 # Run all tests
  python run_tests.py --unit                # Run unit tests only
  python run_tests.py --integration         # Run integration tests only
  python run_tests.py --smoke               # Run smoke tests only
  python run_tests.py --performance         # Run performance tests only
  python run_tests.py --specific test_integration.py::TestCoreBudgetManagement::test_complete_budget_lifecycle
  python run_tests.py --all --parallel      # Run all tests in parallel
  python run_tests.py --unit --no-coverage  # Run unit tests without coverage
		"""
	)
	
	# Test category options
	test_group = parser.add_mutually_exclusive_group(required=True)
	test_group.add_argument("--all", action="store_true", help="Run all tests")
	test_group.add_argument("--unit", action="store_true", help="Run unit tests only")
	test_group.add_argument("--integration", action="store_true", help="Run integration tests only")
	test_group.add_argument("--performance", action="store_true", help="Run performance tests only")
	test_group.add_argument("--smoke", action="store_true", help="Run smoke tests only")
	test_group.add_argument("--specific", metavar="TEST_PATH", help="Run specific test file or function")
	
	# Test execution options
	parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
	parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
	parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
	parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
	parser.add_argument("--check-deps", action="store_true", help="Check test dependencies only")
	parser.add_argument("--check-db", action="store_true", help="Check database connectivity only")
	
	args = parser.parse_args()
	
	# Setup test environment
	setup_test_environment()
	
	print("APG Budgeting & Forecasting - Test Runner")
	print("=" * 50)
	
	# Check dependencies if requested
	if args.check_deps:
		if check_test_dependencies():
			print("✓ All test dependencies are available")
			return 0
		else:
			return 1
	
	# Check database if requested
	if args.check_db:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		success = loop.run_until_complete(check_test_database())
		loop.close()
		return 0 if success else 1
	
	# Check dependencies before running tests
	if not check_test_dependencies():
		return 1
	
	# Check database connectivity
	print("Checking test environment...")
	loop = asyncio.new_event_loop()
	asyncio.set_event_loop(loop)
	loop.run_until_complete(check_test_database())
	loop.close()
	
	# Record start time
	start_time = time.time()
	
	# Run the appropriate tests
	exit_code = 0
	
	try:
		if args.all:
			exit_code = run_all_tests(
				verbose=args.verbose,
				coverage=not args.no_coverage,
				parallel=args.parallel,
				fail_fast=args.fail_fast
			)
		elif args.unit:
			exit_code = run_unit_tests(
				verbose=args.verbose,
				coverage=not args.no_coverage
			)
		elif args.integration:
			exit_code = run_integration_tests(verbose=args.verbose)
		elif args.performance:
			exit_code = run_performance_tests(verbose=args.verbose)
		elif args.smoke:
			exit_code = run_smoke_tests(verbose=args.verbose)
		elif args.specific:
			exit_code = run_specific_test(args.specific, verbose=args.verbose)
		
		# Record end time and show duration
		end_time = time.time()
		duration = end_time - start_time
		
		print(f"\nTest execution completed in {duration:.2f} seconds")
		
		# Generate test report
		if not args.specific and not args.check_deps and not args.check_db:
			generate_test_report()
		
		# Exit with appropriate code
		if exit_code == 0:
			print("✓ All tests passed!")
		else:
			print(f"✗ Tests failed with exit code {exit_code}")
		
		return exit_code
		
	except KeyboardInterrupt:
		print("\nTest execution interrupted by user")
		return 130
	except Exception as e:
		print(f"Unexpected error: {e}")
		return 1


if __name__ == "__main__":
	sys.exit(main())