#!/usr/bin/env python3
"""
APG Connection Management Test Runner
Comprehensive test execution script with reporting and coverage

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --fast             # Skip slow tests
    python tests/run_tests.py --unit             # Unit tests only
    python tests/run_tests.py --integration      # Integration tests only
    python tests/run_tests.py --coverage         # With coverage report
    python tests/run_tests.py --parallel         # Parallel execution
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional


class TestRunner:
	"""Test runner with comprehensive options"""

	def __init__(self):
		self.base_dir = Path(__file__).parent.parent
		self.test_dir = Path(__file__).parent
		self.coverage_dir = self.test_dir / 'coverage'
		self.reports_dir = self.test_dir / 'reports'

		# Ensure directories exist
		self.coverage_dir.mkdir(exist_ok=True)
		self.reports_dir.mkdir(exist_ok=True)

	def run_tests(self, args: argparse.Namespace) -> int:
		"""Run tests based on provided arguments"""

		# Build pytest command
		cmd = ['python', '-m', 'pytest']

		# Test selection
		if args.test_type == 'unit':
			cmd.extend(['-k', 'not integration and not performance'])
		elif args.test_type == 'integration':
			cmd.extend(['-k', 'integration'])
		elif args.test_type == 'performance':
			cmd.extend(['-k', 'performance'])

		# Speed options
		if args.fast:
			cmd.extend(['-x', '--tb=short'])  # Stop on first failure, short traceback
		else:
			cmd.extend(['--tb=long'])  # Detailed tracebacks

		# Parallel execution
		if args.parallel:
			cpu_count = os.cpu_count() or 1
			cmd.extend(['-n', str(min(cpu_count, 4))])  # Max 4 processes

		# Coverage
		if args.coverage:
			cmd.extend([
				'--cov=.',
				'--cov-report=html:' + str(self.coverage_dir),
				'--cov-report=term-missing',
				'--cov-report=xml:' + str(self.reports_dir / 'coverage.xml'),
				'--cov-fail-under=80'  # Minimum 80% coverage
			])

		# Output format
		cmd.extend([
			'--verbose',
			'--color=yes',
			'--html=' + str(self.reports_dir / 'test_report.html'),
			'--self-contained-html',
			'--json-report',
			'--json-report-file=' + str(self.reports_dir / 'test_results.json')
		])

		# Warnings
		if not args.no_warnings:
			cmd.append('--disable-warnings')

		# Test directory
		cmd.append(str(self.test_dir / 'ci'))

		# Add specific test files if provided
		if args.tests:
			cmd.extend(args.tests)

		# Set environment variables
		env = os.environ.copy()
		env.update({
			'PYTHONPATH': str(self.base_dir),
			'APG_TEST_MODE': 'true',
			'APG_LOG_LEVEL': 'WARNING' if not args.debug else 'DEBUG'
		})

		print(f"Running tests with command: {' '.join(cmd)}")
		print(f"Test directory: {self.test_dir / 'ci'}")
		print(f"Base directory: {self.base_dir}")
		print("-" * 60)

		# Execute tests
		try:
			result = subprocess.run(cmd, env=env, cwd=self.base_dir)
			return result.returncode
		except KeyboardInterrupt:
			print("\n❌ Tests interrupted by user")
			return 1
		except Exception as e:
			print(f"\n❌ Test execution failed: {e}")
			return 1

	def install_requirements(self) -> bool:
		"""Install test requirements"""
		requirements_file = self.test_dir / 'requirements.txt'

		if not requirements_file.exists():
			print("❌ Test requirements file not found")
			return False

		print("📦 Installing test requirements...")
		try:
			cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
			result = subprocess.run(cmd, check=True, capture_output=True, text=True)
			print("✅ Test requirements installed successfully")
			return True
		except subprocess.CalledProcessError as e:
			print(f"❌ Failed to install requirements: {e}")
			print(f"STDOUT: {e.stdout}")
			print(f"STDERR: {e.stderr}")
			return False

	def check_environment(self) -> bool:
		"""Check if test environment is properly set up"""
		print("🔍 Checking test environment...")

		# Check Python version
		if sys.version_info < (3, 8):
			print("❌ Python 3.8+ required")
			return False

		print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

		# Check required modules
		required_modules = ['pytest', 'sqlalchemy', 'flask', 'asyncio']
		missing_modules = []

		for module in required_modules:
			try:
				__import__(module)
				print(f"✅ {module} available")
			except ImportError:
				missing_modules.append(module)
				print(f"❌ {module} missing")

		if missing_modules:
			print(f"❌ Missing modules: {', '.join(missing_modules)}")
			print("💡 Run with --install-deps to install missing dependencies")
			return False

		print("✅ Test environment ready")
		return True

	def generate_summary(self, exit_code: int) -> None:
		"""Generate test summary"""
		print("\n" + "=" * 60)
		print("📊 TEST EXECUTION SUMMARY")
		print("=" * 60)

		# Test result
		if exit_code == 0:
			print("✅ All tests passed!")
		else:
			print("❌ Some tests failed or encountered errors")

		# Report locations
		reports = []
		if (self.reports_dir / 'test_report.html').exists():
			reports.append(f"📄 HTML Report: {self.reports_dir / 'test_report.html'}")

		if (self.coverage_dir / 'index.html').exists():
			reports.append(f"📊 Coverage Report: {self.coverage_dir / 'index.html'}")

		if (self.reports_dir / 'test_results.json').exists():
			reports.append(f"📋 JSON Results: {self.reports_dir / 'test_results.json'}")

		if reports:
			print("\n📁 Generated Reports:")
			for report in reports:
				print(f"   {report}")

		print(f"\n🔍 Exit Code: {exit_code}")
		print("=" * 60)


def main():
	"""Main entry point"""
	parser = argparse.ArgumentParser(
		description="APG Connection Management Test Runner",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  %(prog)s                          # Run all tests
  %(prog)s --fast                   # Quick test run
  %(prog)s --coverage               # Run with coverage
  %(prog)s --parallel               # Parallel execution
  %(prog)s --unit                   # Unit tests only
  %(prog)s test_service.py          # Run specific test file
  %(prog)s --install-deps           # Install test dependencies
		"""
	)

	# Test selection
	parser.add_argument(
		'--test-type', '-t',
		choices=['all', 'unit', 'integration', 'performance'],
		default='all',
		help='Type of tests to run (default: all)'
	)

	parser.add_argument(
		'tests',
		nargs='*',
		help='Specific test files or patterns to run'
	)

	# Execution options
	parser.add_argument(
		'--fast', '-f',
		action='store_true',
		help='Fast test execution (stop on first failure)'
	)

	parser.add_argument(
		'--parallel', '-p',
		action='store_true',
		help='Run tests in parallel'
	)

	parser.add_argument(
		'--coverage', '-c',
		action='store_true',
		help='Generate coverage reports'
	)

	# Environment options
	parser.add_argument(
		'--install-deps',
		action='store_true',
		help='Install test dependencies before running'
	)

	parser.add_argument(
		'--check-env',
		action='store_true',
		help='Check test environment and exit'
	)

	# Output options
	parser.add_argument(
		'--debug', '-d',
		action='store_true',
		help='Enable debug output'
	)

	parser.add_argument(
		'--no-warnings',
		action='store_true',
		help='Show pytest warnings'
	)

	# Convenience aliases
	parser.add_argument('--unit', action='store_const', const='unit', dest='test_type',
					   help='Run unit tests only (alias for --test-type unit)')
	parser.add_argument('--integration', action='store_const', const='integration', dest='test_type',
					   help='Run integration tests only (alias for --test-type integration)')
	parser.add_argument('--perf', action='store_const', const='performance', dest='test_type',
					   help='Run performance tests only (alias for --test-type performance)')

	args = parser.parse_args()

	runner = TestRunner()

	# Handle special commands
	if args.check_env:
		success = runner.check_environment()
		return 0 if success else 1

	if args.install_deps:
		success = runner.install_requirements()
		if not success:
			return 1

		# Continue to environment check
		if not runner.check_environment():
			return 1
	else:
		# Just check environment
		if not runner.check_environment():
			print("\n💡 Use --install-deps to install missing dependencies")
			return 1

	# Run tests
	exit_code = runner.run_tests(args)

	# Generate summary
	runner.generate_summary(exit_code)

	return exit_code


if __name__ == '__main__':
	sys.exit(main())