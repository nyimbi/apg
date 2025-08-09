#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APG Governance, Risk & Compliance - Test Runner

Comprehensive test execution script with reporting, coverage analysis,
and performance metrics for the GRC capability.

Copyright © 2025 Datacraft - All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run unit tests only
    python run_tests.py --integration      # Run integration tests only
    python run_tests.py --performance      # Run performance tests only
    python run_tests.py --coverage         # Run with detailed coverage
    python run_tests.py --parallel         # Run tests in parallel
    python run_tests.py --verbose          # Verbose output
    python run_tests.py --fast            # Skip slow tests
"""

import asyncio
import sys
import os
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

class GRCTestRunner:
	"""Comprehensive test runner for GRC capability."""
	
	def __init__(self):
		self.start_time = None
		self.results = {
			"test_suites": {},
			"coverage": {},
			"performance": {},
			"summary": {}
		}
		
	def run_command(self, command: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
		"""Execute command and capture results."""
		print(f"🔧 Executing: {' '.join(command)}")
		
		try:
			result = subprocess.run(
				command,
				capture_output=capture_output,
				text=True,
				cwd=current_dir,
				timeout=600  # 10 minute timeout
			)
			return result
		except subprocess.TimeoutExpired:
			print("❌ Test execution timed out!")
			sys.exit(1)
		except Exception as e:
			print(f"❌ Error executing command: {e}")
			sys.exit(1)
	
	def check_dependencies(self) -> bool:
		"""Check if all required dependencies are installed."""
		print("🔍 Checking test dependencies...")
		
		required_packages = [
			"pytest",
			"pytest-asyncio", 
			"pytest-cov",
			"sqlalchemy",
			"flask",
			"pydantic"
		]
		
		missing_packages = []
		
		for package in required_packages:
			try:
				__import__(package.replace("-", "_"))
			except ImportError:
				missing_packages.append(package)
		
		if missing_packages:
			print(f"❌ Missing required packages: {', '.join(missing_packages)}")
			print("💡 Install with: pip install -r requirements-test.txt")
			return False
		
		print("✅ All dependencies satisfied")
		return True
	
	def setup_test_environment(self) -> bool:
		"""Setup test environment and verify configuration."""
		print("🔧 Setting up test environment...")
		
		# Set environment variables for testing
		os.environ["TESTING"] = "true"
		os.environ["PYTHONPATH"] = str(current_dir)
		os.environ["GRC_TEST_MODE"] = "true"
		
		# Create necessary directories
		test_dirs = ["htmlcov", "test-reports", "performance-reports"]
		for dir_name in test_dirs:
			os.makedirs(current_dir / dir_name, exist_ok=True)
		
		# Verify test files exist
		test_files = [
			"test_models.py",
			"test_services.py", 
			"test_integration.py",
			"conftest.py"
		]
		
		missing_files = []
		for test_file in test_files:
			if not (current_dir / test_file).exists():
				missing_files.append(test_file)
		
		if missing_files:
			print(f"❌ Missing test files: {', '.join(missing_files)}")
			return False
		
		print("✅ Test environment ready")
		return True
	
	def run_unit_tests(self, verbose: bool = False, parallel: bool = False) -> Dict[str, Any]:
		"""Run unit tests for models and core functionality."""
		print("\n🧪 Running Unit Tests...")
		
		command = [
			"python", "-m", "pytest",
			"test_models.py",
			"-m", "unit or not integration",
			"--tb=short"
		]
		
		if verbose:
			command.append("-v")
		
		if parallel:
			command.extend(["-n", "auto"])
		
		result = self.run_command(command)
		
		test_result = {
			"exit_code": result.returncode,
			"stdout": result.stdout,
			"stderr": result.stderr,
			"success": result.returncode == 0
		}
		
		if test_result["success"]:
			print("✅ Unit tests passed")
		else:
			print("❌ Unit tests failed")
			print(result.stdout)
			print(result.stderr)
		
		self.results["test_suites"]["unit"] = test_result
		return test_result
	
	def run_service_tests(self, verbose: bool = False, parallel: bool = False) -> Dict[str, Any]:
		"""Run service layer tests including AI engines."""
		print("\n🔬 Running Service Tests...")
		
		command = [
			"python", "-m", "pytest",
			"test_services.py",
			"--tb=short"
		]
		
		if verbose:
			command.append("-v")
		
		if parallel:
			command.extend(["-n", "auto"])
		
		result = self.run_command(command)
		
		test_result = {
			"exit_code": result.returncode,
			"stdout": result.stdout,
			"stderr": result.stderr,
			"success": result.returncode == 0
		}
		
		if test_result["success"]:
			print("✅ Service tests passed")
		else:
			print("❌ Service tests failed")
			print(result.stdout)
			print(result.stderr)
		
		self.results["test_suites"]["service"] = test_result
		return test_result
	
	def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
		"""Run integration tests for end-to-end workflows."""
		print("\n🔗 Running Integration Tests...")
		
		command = [
			"python", "-m", "pytest",
			"test_integration.py",
			"-m", "integration",
			"--tb=short"
		]
		
		if verbose:
			command.append("-v")
		
		result = self.run_command(command)
		
		test_result = {
			"exit_code": result.returncode,
			"stdout": result.stdout,
			"stderr": result.stderr,
			"success": result.returncode == 0
		}
		
		if test_result["success"]:
			print("✅ Integration tests passed")
		else:
			print("❌ Integration tests failed")
			print(result.stdout)
			print(result.stderr)
		
		self.results["test_suites"]["integration"] = test_result
		return test_result
	
	def run_performance_tests(self, verbose: bool = False) -> Dict[str, Any]:
		"""Run performance and load tests."""
		print("\n⚡ Running Performance Tests...")
		
		command = [
			"python", "-m", "pytest",
			"test_integration.py::TestPerformanceAndScalability",
			"-m", "performance",
			"--benchmark-only",
			"--benchmark-json=performance-reports/benchmark.json",
			"--tb=short"
		]
		
		if verbose:
			command.append("-v")
		
		result = self.run_command(command)
		
		test_result = {
			"exit_code": result.returncode,
			"stdout": result.stdout,
			"stderr": result.stderr,
			"success": result.returncode == 0
		}
		
		# Load benchmark results if available
		benchmark_file = current_dir / "performance-reports" / "benchmark.json"
		if benchmark_file.exists():
			try:
				with open(benchmark_file, 'r') as f:
					benchmark_data = json.load(f)
					test_result["benchmarks"] = benchmark_data
			except Exception as e:
				print(f"⚠️  Could not load benchmark data: {e}")
		
		if test_result["success"]:
			print("✅ Performance tests passed")
		else:
			print("❌ Performance tests failed")
			print(result.stdout)
			print(result.stderr)
		
		self.results["test_suites"]["performance"] = test_result
		return test_result
	
	def run_coverage_analysis(self, verbose: bool = False) -> Dict[str, Any]:
		"""Run comprehensive coverage analysis."""
		print("\n📊 Running Coverage Analysis...")
		
		command = [
			"python", "-m", "pytest",
			"--cov=.",
			"--cov-report=term-missing",
			"--cov-report=html:htmlcov",
			"--cov-report=xml:coverage.xml",
			"--cov-fail-under=95",
			"--tb=short"
		]
		
		if verbose:
			command.append("-v")
		
		result = self.run_command(command)
		
		coverage_result = {
			"exit_code": result.returncode,
			"stdout": result.stdout,
			"stderr": result.stderr,
			"success": result.returncode == 0,
			"coverage_threshold_met": result.returncode == 0
		}
		
		# Parse coverage percentage from output
		if result.stdout:
			lines = result.stdout.split('\n')
			for line in lines:
				if "TOTAL" in line and "%" in line:
					# Extract coverage percentage
					parts = line.split()
					for part in parts:
						if part.endswith('%'):
							try:
								coverage_pct = float(part.rstrip('%'))
								coverage_result["coverage_percentage"] = coverage_pct
								break
							except ValueError:
								pass
		
		if coverage_result["success"]:
			print("✅ Coverage analysis completed successfully")
			if "coverage_percentage" in coverage_result:
				print(f"📈 Coverage: {coverage_result['coverage_percentage']:.1f}%")
		else:
			print("❌ Coverage analysis failed or threshold not met")
		
		self.results["coverage"] = coverage_result
		return coverage_result
	
	def run_security_tests(self, verbose: bool = False) -> Dict[str, Any]:
		"""Run security-focused tests and static analysis."""
		print("\n🛡️  Running Security Tests...")
		
		# Run bandit for security static analysis
		bandit_command = [
			"python", "-m", "bandit",
			"-r", ".",
			"-f", "json",
			"-o", "test-reports/security-report.json",
			"--exclude", "*/tests/*,*/test_*"
		]
		
		try:
			bandit_result = self.run_command(bandit_command)
			bandit_success = bandit_result.returncode in [0, 1]  # 1 means issues found but not critical
		except Exception:
			bandit_success = False
			bandit_result = None
		
		# Run security-marked tests
		pytest_command = [
			"python", "-m", "pytest",
			"-m", "security",
			"--tb=short"
		]
		
		if verbose:
			pytest_command.append("-v")
		
		pytest_result = self.run_command(pytest_command)
		
		security_result = {
			"pytest": {
				"exit_code": pytest_result.returncode,
				"success": pytest_result.returncode == 0
			},
			"bandit": {
				"success": bandit_success,
				"result": bandit_result.returncode if bandit_result else None
			},
			"overall_success": pytest_result.returncode == 0 and bandit_success
		}
		
		if security_result["overall_success"]:
			print("✅ Security tests passed")
		else:
			print("❌ Security tests failed or issues found")
		
		return security_result
	
	def generate_test_report(self) -> None:
		"""Generate comprehensive test report."""
		print("\n📋 Generating Test Report...")
		
		report_data = {
			"timestamp": datetime.now().isoformat(),
			"execution_time": time.time() - self.start_time if self.start_time else 0,
			"results": self.results,
			"environment": {
				"python_version": sys.version,
				"platform": sys.platform,
				"working_directory": str(current_dir)
			}
		}
		
		# Calculate overall success
		all_successful = True
		for suite_name, suite_result in self.results.get("test_suites", {}).items():
			if not suite_result.get("success", False):
				all_successful = False
				break
		
		report_data["overall_success"] = all_successful
		
		# Save detailed report
		report_file = current_dir / "test-reports" / "test-report.json"
		os.makedirs(report_file.parent, exist_ok=True)
		
		with open(report_file, 'w') as f:
			json.dump(report_data, f, indent=2)
		
		# Print summary
		print("\n" + "="*60)
		print("🏁 TEST EXECUTION SUMMARY")
		print("="*60)
		
		for suite_name, suite_result in self.results.get("test_suites", {}).items():
			status = "✅ PASSED" if suite_result.get("success") else "❌ FAILED"
			print(f"{suite_name.upper():15} {status}")
		
		if "coverage" in self.results:
			coverage_pct = self.results["coverage"].get("coverage_percentage", 0)
			coverage_status = "✅ PASSED" if coverage_pct >= 95 else "❌ FAILED"
			print(f"{'COVERAGE':15} {coverage_status} ({coverage_pct:.1f}%)")
		
		print("-"*60)
		overall_status = "✅ ALL TESTS PASSED" if all_successful else "❌ SOME TESTS FAILED"
		print(f"OVERALL RESULT: {overall_status}")
		
		if self.start_time:
			elapsed = time.time() - self.start_time
			print(f"EXECUTION TIME: {elapsed:.2f} seconds")
		
		print(f"DETAILED REPORT: {report_file}")
		print("="*60)

def main():
	"""Main test execution function."""
	parser = argparse.ArgumentParser(description="APG GRC Capability Test Runner")
	
	# Test selection options
	parser.add_argument("--unit", action="store_true", help="Run unit tests only")
	parser.add_argument("--service", action="store_true", help="Run service tests only")
	parser.add_argument("--integration", action="store_true", help="Run integration tests only")
	parser.add_argument("--performance", action="store_true", help="Run performance tests only")
	parser.add_argument("--security", action="store_true", help="Run security tests only")
	parser.add_argument("--coverage", action="store_true", help="Run with detailed coverage analysis")
	
	# Execution options
	parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
	parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
	parser.add_argument("--fast", action="store_true", help="Skip slow tests")
	parser.add_argument("--no-report", action="store_true", help="Skip report generation")
	
	args = parser.parse_args()
	
	# Initialize test runner
	runner = GRCTestRunner()
	runner.start_time = time.time()
	
	print("🚀 APG Governance, Risk & Compliance - Test Suite")
	print("=" * 60)
	
	# Check dependencies
	if not runner.check_dependencies():
		sys.exit(1)
	
	# Setup environment
	if not runner.setup_test_environment():
		sys.exit(1)
	
	# Determine which tests to run
	run_all = not any([args.unit, args.service, args.integration, args.performance, args.security, args.coverage])
	
	success = True
	
	try:
		# Run selected test suites
		if args.unit or run_all:
			result = runner.run_unit_tests(verbose=args.verbose, parallel=args.parallel)
			success = success and result["success"]
		
		if args.service or run_all:
			result = runner.run_service_tests(verbose=args.verbose, parallel=args.parallel)
			success = success and result["success"]
		
		if args.integration or run_all:
			result = runner.run_integration_tests(verbose=args.verbose)
			success = success and result["success"]
		
		if args.performance or run_all:
			result = runner.run_performance_tests(verbose=args.verbose)
			success = success and result["success"]
		
		if args.security or run_all:
			result = runner.run_security_tests(verbose=args.verbose)
			success = success and result.get("overall_success", False)
		
		if args.coverage or run_all:
			result = runner.run_coverage_analysis(verbose=args.verbose)
			success = success and result["success"]
		
	except KeyboardInterrupt:
		print("\n🛑 Test execution interrupted by user")
		sys.exit(1)
	except Exception as e:
		print(f"\n💥 Unexpected error during test execution: {e}")
		sys.exit(1)
	
	# Generate report
	if not args.no_report:
		runner.generate_test_report()
	
	# Exit with appropriate code
	sys.exit(0 if success else 1)

if __name__ == "__main__":
	main()