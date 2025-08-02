#!/usr/bin/env python3
"""
APG Workflow Orchestration Test Runner

Comprehensive test runner with coverage reporting, performance metrics,
and APG-specific test configurations.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import os
import sys
import asyncio
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

class APGTestRunner:
	"""APG-aware test runner with comprehensive reporting."""
	
	def __init__(self, test_dir: Path = None):
		self.test_dir = test_dir or Path(__file__).parent
		self.root_dir = self.test_dir.parent
		self.coverage_threshold = 95.0
		
	def run_unit_tests(self, verbose: bool = True, parallel: bool = False) -> int:
		"""Run unit tests with coverage."""
		print("🧪 Running Unit Tests...")
		
		cmd = [
			"python", "-m", "pytest",
			str(self.test_dir),
			"-m", "unit or not (integration or performance)",
			"--cov", str(self.root_dir),
			"--cov-report", "html:coverage_html",
			"--cov-report", "xml:coverage.xml", 
			"--cov-report", "term-missing",
			f"--cov-fail-under={self.coverage_threshold}",
			"--cov-branch"
		]
		
		if verbose:
			cmd.append("-v")
		
		if parallel:
			cmd.extend(["-n", "auto"])
		
		return subprocess.run(cmd, cwd=self.test_dir).returncode
	
	def run_integration_tests(self, verbose: bool = True) -> int:
		"""Run integration tests."""
		print("🔗 Running Integration Tests...")
		
		cmd = [
			"python", "-m", "pytest",
			str(self.test_dir),
			"-m", "integration",
			"--timeout", "60"
		]
		
		if verbose:
			cmd.append("-v")
		
		return subprocess.run(cmd, cwd=self.test_dir).returncode
	
	def run_performance_tests(self, verbose: bool = True) -> int:
		"""Run performance tests."""
		print("⚡ Running Performance Tests...")
		
		cmd = [
			"python", "-m", "pytest",
			str(self.test_dir),
			"-m", "performance",
			"--benchmark-only",
			"--benchmark-sort", "mean",
			"--timeout", "120"
		]
		
		if verbose:
			cmd.append("-v")
		
		return subprocess.run(cmd, cwd=self.test_dir).returncode
	
	def run_security_tests(self, verbose: bool = True) -> int:
		"""Run security tests."""
		print("🔒 Running Security Tests...")
		
		# Run bandit security linter
		bandit_cmd = [
			"bandit", "-r", str(self.root_dir),
			"-f", "json", "-o", "security_report.json",
			"-x", str(self.test_dir)
		]
		
		bandit_result = subprocess.run(bandit_cmd, cwd=self.test_dir)
		
		# Run security-specific tests
		cmd = [
			"python", "-m", "pytest",
			str(self.test_dir),
			"-m", "security"
		]
		
		if verbose:
			cmd.append("-v")
		
		pytest_result = subprocess.run(cmd, cwd=self.test_dir).returncode
		
		return max(bandit_result.returncode, pytest_result)
	
	def run_api_tests(self, verbose: bool = True) -> int:
		"""Run API endpoint tests."""
		print("🌐 Running API Tests...")
		
		cmd = [
			"python", "-m", "pytest",
			str(self.test_dir / "test_api.py"),
			"-m", "api or not (unit or integration)"
		]
		
		if verbose:
			cmd.append("-v")
		
		return subprocess.run(cmd, cwd=self.test_dir).returncode
	
	def run_database_tests(self, verbose: bool = True) -> int:
		"""Run database-specific tests."""
		print("🗃️ Running Database Tests...")
		
		cmd = [
			"python", "-m", "pytest",
			str(self.test_dir / "test_database.py"),
			"-m", "database or not (unit or integration)"
		]
		
		if verbose:
			cmd.append("-v")
		
		return subprocess.run(cmd, cwd=self.test_dir).returncode
	
	def run_engine_tests(self, verbose: bool = True) -> int:
		"""Run workflow engine tests."""
		print("⚙️ Running Engine Tests...")
		
		cmd = [
			"python", "-m", "pytest",
			str(self.test_dir / "test_engine.py"),
			"-m", "engine or not (unit or integration)"
		]
		
		if verbose:
			cmd.append("-v")
		
		return subprocess.run(cmd, cwd=self.test_dir).returncode
	
	def run_service_tests(self, verbose: bool = True) -> int:
		"""Run service layer tests."""
		print("🔧 Running Service Tests...")
		
		cmd = [
			"python", "-m", "pytest",
			str(self.test_dir / "test_service.py"),
			"-m", "service or not (unit or integration)"
		]
		
		if verbose:
			cmd.append("-v")
		
		return subprocess.run(cmd, cwd=self.test_dir).returncode
	
	def run_management_tests(self, verbose: bool = True) -> int:
		"""Run management layer tests."""
		print("📊 Running Management Tests...")
		
		cmd = [
			"python", "-m", "pytest",
			str(self.test_dir / "test_management.py"),
			"-m", "management or not (unit or integration)"
		]
		
		if verbose:
			cmd.append("-v")
		
		return subprocess.run(cmd, cwd=self.test_dir).returncode
	
	def run_model_tests(self, verbose: bool = True) -> int:
		"""Run model tests."""
		print("📋 Running Model Tests...")
		
		cmd = [
			"python", "-m", "pytest",
			str(self.test_dir / "test_models.py")
		]
		
		if verbose:
			cmd.append("-v")
		
		return subprocess.run(cmd, cwd=self.test_dir).returncode
	
	def run_all_tests(self, verbose: bool = True, parallel: bool = False) -> Dict[str, int]:
		"""Run all test suites."""
		print("🚀 Running All Test Suites...")
		
		results = {}
		
		# Run each test suite
		test_suites = [
			("Models", self.run_model_tests),
			("Database", self.run_database_tests),
			("Service", self.run_service_tests),
			("Engine", self.run_engine_tests),
			("Management", self.run_management_tests),
			("API", self.run_api_tests),
			("Unit", lambda v: self.run_unit_tests(v, parallel)),
			("Integration", self.run_integration_tests),
			("Security", self.run_security_tests),
			("Performance", self.run_performance_tests)
		]
		
		for name, test_func in test_suites:
			print(f"\n{'='*50}")
			print(f"Running {name} Tests")
			print(f"{'='*50}")
			
			start_time = time.time()
			result = test_func(verbose)
			end_time = time.time()
			
			results[name] = {
				"exit_code": result,
				"duration": end_time - start_time,
				"status": "PASSED" if result == 0 else "FAILED"
			}
			
			print(f"\n{name} Tests: {results[name]['status']} "
				  f"({results[name]['duration']:.2f}s)")
		
		return results
	
	def generate_test_report(self, results: Dict[str, Any]) -> None:
		"""Generate comprehensive test report."""
		print(f"\n{'='*60}")
		print("APG WORKFLOW ORCHESTRATION TEST REPORT")
		print(f"{'='*60}")
		
		total_duration = sum(r["duration"] for r in results.values())
		passed_tests = sum(1 for r in results.values() if r["exit_code"] == 0)
		total_tests = len(results)
		
		print(f"Total Test Suites: {total_tests}")
		print(f"Passed: {passed_tests}")
		print(f"Failed: {total_tests - passed_tests}")
		print(f"Total Duration: {total_duration:.2f}s")
		print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
		
		print(f"\n{'Suite':<15} {'Status':<8} {'Duration':<10}")
		print("-" * 35)
		
		for name, result in results.items():
			status_icon = "✅" if result["exit_code"] == 0 else "❌"
			print(f"{name:<15} {status_icon} {result['status']:<6} {result['duration']:>8.2f}s")
		
		# Coverage report location
		coverage_html = self.test_dir / "coverage_html" / "index.html"
		if coverage_html.exists():
			print(f"\n📊 Coverage Report: {coverage_html}")
		
		# Security report location
		security_report = self.test_dir / "security_report.json"
		if security_report.exists():
			print(f"🔒 Security Report: {security_report}")
		
		print(f"\n{'='*60}")
	
	def check_environment(self) -> bool:
		"""Check test environment setup."""
		print("🔍 Checking Test Environment...")
		
		required_packages = [
			"pytest", "pytest-asyncio", "pytest-cov", 
			"fastapi", "sqlalchemy", "redis", "pydantic"
		]
		
		missing_packages = []
		for package in required_packages:
			try:
				__import__(package.replace("-", "_"))
			except ImportError:
				missing_packages.append(package)
		
		if missing_packages:
			print(f"❌ Missing required packages: {', '.join(missing_packages)}")
			print("   Install with: pip install -r requirements.txt")
			return False
		
		print("✅ Test environment ready")
		return True
	
	def setup_test_data(self) -> None:
		"""Setup test data and fixtures."""
		print("📋 Setting up test data...")
		
		# Create test data directory if not exists
		test_data_dir = self.test_dir / "test_data"
		test_data_dir.mkdir(exist_ok=True)
		
		# Create fixtures directory if not exists
		fixtures_dir = self.test_dir / "fixtures"
		fixtures_dir.mkdir(exist_ok=True)
		
		print("✅ Test data setup complete")
	
	def cleanup_test_artifacts(self) -> None:
		"""Clean up test artifacts."""
		print("🧹 Cleaning up test artifacts...")
		
		artifacts_to_clean = [
			"__pycache__",
			".pytest_cache",
			"*.pyc",
			"coverage.xml",
			"security_report.json"
		]
		
		for pattern in artifacts_to_clean:
			for path in self.test_dir.rglob(pattern):
				if path.is_file():
					path.unlink()
				elif path.is_dir():
					import shutil
					shutil.rmtree(path)
		
		print("✅ Cleanup complete")

def main():
	"""Main test runner entry point."""
	parser = argparse.ArgumentParser(
		description="APG Workflow Orchestration Test Runner"
	)
	
	parser.add_argument(
		"--suite", 
		choices=[
			"all", "unit", "integration", "performance", "security",
			"api", "database", "engine", "service", "management", "models"
		],
		default="all",
		help="Test suite to run"
	)
	
	parser.add_argument(
		"--verbose", "-v",
		action="store_true",
		help="Verbose output"
	)
	
	parser.add_argument(
		"--parallel", "-p",
		action="store_true",
		help="Run tests in parallel"
	)
	
	parser.add_argument(
		"--coverage-threshold",
		type=float,
		default=95.0,
		help="Coverage threshold percentage"
	)
	
	parser.add_argument(
		"--no-cleanup",
		action="store_true",
		help="Skip cleanup of test artifacts"
	)
	
	parser.add_argument(
		"--check-env-only",
		action="store_true",
		help="Only check test environment"
	)
	
	args = parser.parse_args()
	
	# Initialize test runner
	runner = APGTestRunner()
	runner.coverage_threshold = args.coverage_threshold
	
	# Check environment
	if not runner.check_environment():
		sys.exit(1)
	
	if args.check_env_only:
		print("✅ Environment check complete")
		sys.exit(0)
	
	# Setup test data
	runner.setup_test_data()
	
	try:
		# Run tests based on suite selection
		if args.suite == "all":
			results = runner.run_all_tests(args.verbose, args.parallel)
			runner.generate_test_report(results)
			
			# Exit with failure if any test suite failed
			exit_code = max(r["exit_code"] for r in results.values())
			
		elif args.suite == "unit":
			exit_code = runner.run_unit_tests(args.verbose, args.parallel)
		elif args.suite == "integration":
			exit_code = runner.run_integration_tests(args.verbose)
		elif args.suite == "performance":
			exit_code = runner.run_performance_tests(args.verbose) 
		elif args.suite == "security":
			exit_code = runner.run_security_tests(args.verbose)
		elif args.suite == "api":
			exit_code = runner.run_api_tests(args.verbose)
		elif args.suite == "database":
			exit_code = runner.run_database_tests(args.verbose)
		elif args.suite == "engine":
			exit_code = runner.run_engine_tests(args.verbose)
		elif args.suite == "service":
			exit_code = runner.run_service_tests(args.verbose)
		elif args.suite == "management":
			exit_code = runner.run_management_tests(args.verbose)
		elif args.suite == "models":
			exit_code = runner.run_model_tests(args.verbose)
		
		# Final status
		if exit_code == 0:
			print("\n✅ All tests passed!")
		else:
			print(f"\n❌ Tests failed with exit code {exit_code}")
		
	finally:
		# Cleanup unless specified otherwise
		if not args.no_cleanup:
			runner.cleanup_test_artifacts()
	
	sys.exit(exit_code)

if __name__ == "__main__":
	main()