#!/usr/bin/env python3
"""
APG Vendor Management - Test Runner
Comprehensive test execution and reporting script

Author: Nyimbi Odero (nyimbi@gmail.com)
Copyright: © 2025 Datacraft (www.datacraft.co.ke)
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Optional


class VendorManagementTestRunner:
	"""Comprehensive test runner for vendor management capability"""
	
	def __init__(self):
		self.test_dir = Path(__file__).parent
		self.project_root = self.test_dir.parent
		self.coverage_threshold = 85
		self.test_results = {}
	
	def run_unit_tests(self, verbose: bool = False) -> bool:
		"""Run unit tests"""
		
		print("🧪 Running Unit Tests...")
		
		cmd = [
			'python', '-m', 'pytest',
			'-m', 'unit',
			'--tb=short',
			f'--cov={self.project_root}',
			'--cov-report=term-missing',
			'--cov-report=html:htmlcov/unit',
			f'--cov-fail-under={self.coverage_threshold}'
		]
		
		if verbose:
			cmd.append('-v')
		
		result = self._run_command(cmd)
		self.test_results['unit'] = result
		
		return result == 0
	
	def run_integration_tests(self, verbose: bool = False) -> bool:
		"""Run integration tests"""
		
		print("🔗 Running Integration Tests...")
		
		cmd = [
			'python', '-m', 'pytest',
			'-m', 'integration',
			'--tb=short',
			'--timeout=60'
		]
		
		if verbose:
			cmd.append('-v')
		
		result = self._run_command(cmd)
		self.test_results['integration'] = result
		
		return result == 0
	
	def run_api_tests(self, verbose: bool = False) -> bool:
		"""Run API tests"""
		
		print("🌐 Running API Tests...")
		
		cmd = [
			'python', '-m', 'pytest',
			'-m', 'api',
			'--tb=short'
		]
		
		if verbose:
			cmd.append('-v')
		
		result = self._run_command(cmd)
		self.test_results['api'] = result
		
		return result == 0
	
	def run_ai_tests(self, verbose: bool = False) -> bool:
		"""Run AI/ML tests"""
		
		print("🤖 Running AI/ML Tests...")
		
		cmd = [
			'python', '-m', 'pytest',
			'-m', 'ai',
			'--tb=short'
		]
		
		if verbose:
			cmd.append('-v')
		
		result = self._run_command(cmd)
		self.test_results['ai'] = result
		
		return result == 0
	
	def run_security_tests(self, verbose: bool = False) -> bool:
		"""Run security tests"""
		
		print("🔒 Running Security Tests...")
		
		cmd = [
			'python', '-m', 'pytest',
			'-m', 'security',
			'--tb=short'
		]
		
		if verbose:
			cmd.append('-v')
		
		result = self._run_command(cmd)
		self.test_results['security'] = result
		
		return result == 0
	
	def run_performance_tests(self, verbose: bool = False) -> bool:
		"""Run performance tests"""
		
		print("⚡ Running Performance Tests...")
		
		cmd = [
			'python', '-m', 'pytest',
			'-m', 'performance',
			'--tb=short',
			'--benchmark-only',
			'--benchmark-sort=mean'
		]
		
		if verbose:
			cmd.append('-v')
		
		result = self._run_command(cmd)
		self.test_results['performance'] = result
		
		return result == 0
	
	def run_smoke_tests(self, verbose: bool = False) -> bool:
		"""Run smoke tests"""
		
		print("💨 Running Smoke Tests...")
		
		cmd = [
			'python', '-m', 'pytest',
			'-m', 'smoke',
			'--tb=line',
			'--maxfail=5'
		]
		
		if verbose:
			cmd.append('-v')
		
		result = self._run_command(cmd)
		self.test_results['smoke'] = result
		
		return result == 0
	
	def run_all_tests(self, verbose: bool = False, fast: bool = False) -> bool:
		"""Run all test suites"""
		
		print("🚀 Running All Tests...")
		print("=" * 60)
		
		test_suites = [
			('smoke', self.run_smoke_tests),
			('unit', self.run_unit_tests),
			('api', self.run_api_tests),
			('ai', self.run_ai_tests),
		]
		
		if not fast:
			test_suites.extend([
				('integration', self.run_integration_tests),
				('security', self.run_security_tests),
				('performance', self.run_performance_tests)
			])
		
		all_passed = True
		start_time = time.time()
		
		for suite_name, test_func in test_suites:
			print(f"\n📋 Test Suite: {suite_name.upper()}")
			print("-" * 40)
			
			suite_start = time.time()
			passed = test_func(verbose)
			suite_end = time.time()
			
			duration = suite_end - suite_start
			status = "✅ PASSED" if passed else "❌ FAILED"
			print(f"{status} ({duration:.2f}s)")
			
			if not passed:
				all_passed = False
				if suite_name == 'smoke':
					print("💥 Smoke tests failed! Stopping execution.")
					break
		
		end_time = time.time()
		total_duration = end_time - start_time
		
		self._print_summary(all_passed, total_duration)
		
		return all_passed
	
	def run_custom_tests(self, markers: List[str], verbose: bool = False) -> bool:
		"""Run tests with custom markers"""
		
		marker_expr = ' and '.join(markers)
		print(f"🎯 Running Custom Tests: {marker_expr}")
		
		cmd = [
			'python', '-m', 'pytest',
			'-m', marker_expr,
			'--tb=short'
		]
		
		if verbose:
			cmd.append('-v')
		
		result = self._run_command(cmd)
		return result == 0
	
	def generate_coverage_report(self) -> bool:
		"""Generate comprehensive coverage report"""
		
		print("📊 Generating Coverage Report...")
		
		cmd = [
			'python', '-m', 'pytest',
			'--cov=..',
			'--cov-report=html:htmlcov',
			'--cov-report=xml:coverage.xml',
			'--cov-report=term-missing',
			'--tb=no',
			'-q'
		]
		
		result = self._run_command(cmd)
		
		if result == 0:
			print(f"📈 Coverage report generated: file://{self.test_dir}/htmlcov/index.html")
		
		return result == 0
	
	def run_linting(self) -> bool:
		"""Run code linting and formatting checks"""
		
		print("🔍 Running Code Quality Checks...")
		
		checks = [
			(['python', '-m', 'black', '--check', '..'], "Black formatting"),
			(['python', '-m', 'isort', '--check-only', '..'], "Import sorting"),
			(['python', '-m', 'flake8', '..'], "Flake8 linting"),
			(['python', '-m', 'mypy', '..'], "Type checking")
		]
		
		all_passed = True
		
		for cmd, description in checks:
			print(f"  🔧 {description}...")
			result = self._run_command(cmd, capture_output=True)
			
			if result == 0:
				print(f"    ✅ {description} passed")
			else:
				print(f"    ❌ {description} failed")
				all_passed = False
		
		return all_passed
	
	def clean_test_artifacts(self) -> None:
		"""Clean test artifacts and cache files"""
		
		print("🧹 Cleaning test artifacts...")
		
		artifacts = [
			'htmlcov',
			'coverage.xml',
			'.coverage',
			'.pytest_cache',
			'__pycache__',
			'*.pyc',
			'.mypy_cache'
		]
		
		for artifact in artifacts:
			artifact_path = self.test_dir / artifact
			if artifact_path.exists():
				if artifact_path.is_dir():
					subprocess.run(['rm', '-rf', str(artifact_path)], capture_output=True)
				else:
					artifact_path.unlink()
		
		print("✨ Cleanup complete")
	
	def _run_command(self, cmd: List[str], capture_output: bool = False) -> int:
		"""Run shell command and return exit code"""
		
		try:
			if capture_output:
				result = subprocess.run(
					cmd, 
					cwd=self.test_dir, 
					capture_output=True, 
					text=True
				)
			else:
				result = subprocess.run(cmd, cwd=self.test_dir)
			
			return result.returncode
			
		except FileNotFoundError:
			print(f"❌ Command not found: {cmd[0]}")
			return 1
		except Exception as e:
			print(f"❌ Error running command: {e}")
			return 1
	
	def _print_summary(self, all_passed: bool, duration: float) -> None:
		"""Print test execution summary"""
		
		print("\n" + "=" * 60)
		print("📋 TEST EXECUTION SUMMARY")
		print("=" * 60)
		
		for suite_name, result_code in self.test_results.items():
			status = "✅ PASSED" if result_code == 0 else "❌ FAILED"
			print(f"  {suite_name.upper():.<20} {status}")
		
		overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
		print(f"\n🎯 Overall Result: {overall_status}")
		print(f"⏱️  Total Duration: {duration:.2f} seconds")
		
		if all_passed:
			print("\n🎉 Great job! All tests are passing.")
		else:
			print("\n🔧 Please fix failing tests before deployment.")
		
		print("=" * 60)


def main():
	"""Main entry point for test runner"""
	
	parser = argparse.ArgumentParser(
		description="APG Vendor Management Test Runner",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  python run_tests.py --all                 # Run all tests
  python run_tests.py --unit --verbose      # Run unit tests with verbose output
  python run_tests.py --smoke --fast        # Quick smoke test
  python run_tests.py --custom ai,unit      # Run AI and unit tests
  python run_tests.py --coverage            # Generate coverage report
  python run_tests.py --lint                # Run code quality checks
  python run_tests.py --clean               # Clean test artifacts
		"""
	)
	
	# Test suite options
	parser.add_argument('--all', action='store_true', help='Run all test suites')
	parser.add_argument('--unit', action='store_true', help='Run unit tests')
	parser.add_argument('--integration', action='store_true', help='Run integration tests')
	parser.add_argument('--api', action='store_true', help='Run API tests')
	parser.add_argument('--ai', action='store_true', help='Run AI/ML tests')
	parser.add_argument('--security', action='store_true', help='Run security tests')
	parser.add_argument('--performance', action='store_true', help='Run performance tests')
	parser.add_argument('--smoke', action='store_true', help='Run smoke tests')
	
	# Custom options
	parser.add_argument('--custom', type=str, help='Run custom test markers (comma-separated)')
	parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
	parser.add_argument('--lint', action='store_true', help='Run code quality checks')
	parser.add_argument('--clean', action='store_true', help='Clean test artifacts')
	
	# Execution options
	parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
	parser.add_argument('--fast', action='store_true', help='Skip slow tests')
	
	args = parser.parse_args()
	
	# Create test runner
	runner = VendorManagementTestRunner()
	
	# Handle special commands
	if args.clean:
		runner.clean_test_artifacts()
		return 0
	
	if args.lint:
		success = runner.run_linting()
		return 0 if success else 1
	
	if args.coverage:
		success = runner.generate_coverage_report()
		return 0 if success else 1
	
	# Run tests
	success = False
	
	if args.all:
		success = runner.run_all_tests(args.verbose, args.fast)
	elif args.custom:
		markers = [m.strip() for m in args.custom.split(',')]
		success = runner.run_custom_tests(markers, args.verbose)
	else:
		# Run individual test suites
		if args.unit:
			success = runner.run_unit_tests(args.verbose)
		elif args.integration:
			success = runner.run_integration_tests(args.verbose)
		elif args.api:
			success = runner.run_api_tests(args.verbose)
		elif args.ai:
			success = runner.run_ai_tests(args.verbose)
		elif args.security:
			success = runner.run_security_tests(args.verbose)
		elif args.performance:
			success = runner.run_performance_tests(args.verbose)
		elif args.smoke:
			success = runner.run_smoke_tests(args.verbose)
		else:
			# Default to smoke tests
			print("No specific test suite selected. Running smoke tests...")
			success = runner.run_smoke_tests(args.verbose)
	
	return 0 if success else 1


if __name__ == '__main__':
	sys.exit(main())