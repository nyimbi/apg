#!/usr/bin/env python3
"""
APG Facial Recognition - Test Runner

Comprehensive test runner with categorized test execution,
reporting, and CI/CD integration capabilities.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FacialTestRunner:
	"""Comprehensive test runner for facial recognition capability"""
	
	def __init__(self, base_dir: Optional[Path] = None):
		self.base_dir = base_dir or Path(__file__).parent
		self.results = {}
		
	def run_unit_tests(self, verbose: bool = True) -> Dict:
		"""Run unit tests for models and core components"""
		logger.info("Running unit tests...")
		
		cmd = [
			sys.executable, "-m", "pytest",
			"test_models.py",
			"-m", "unit or not (integration or performance or security)",
			"--tb=short",
			"-v" if verbose else "-q"
		]
		
		result = self._run_pytest(cmd, "unit_tests")
		return result
	
	def run_integration_tests(self, verbose: bool = True) -> Dict:
		"""Run integration tests for service interactions"""
		logger.info("Running integration tests...")
		
		cmd = [
			sys.executable, "-m", "pytest", 
			"test_services.py",
			"-m", "integration or not (unit or performance or security)",
			"--tb=short",
			"-v" if verbose else "-q"
		]
		
		result = self._run_pytest(cmd, "integration_tests")
		return result
	
	def run_api_tests(self, verbose: bool = True) -> Dict:
		"""Run API endpoint tests"""
		logger.info("Running API tests...")
		
		cmd = [
			sys.executable, "-m", "pytest",
			"test_api.py", 
			"--tb=short",
			"-v" if verbose else "-q"
		]
		
		result = self._run_pytest(cmd, "api_tests")
		return result
	
	def run_performance_tests(self, verbose: bool = True) -> Dict:
		"""Run performance and benchmark tests"""
		logger.info("Running performance tests...")
		
		cmd = [
			sys.executable, "-m", "pytest",
			"test_performance.py",
			"-m", "performance",
			"--tb=short",
			"-v" if verbose else "-q",
			"--durations=0"
		]
		
		result = self._run_pytest(cmd, "performance_tests")
		return result
	
	def run_security_tests(self, verbose: bool = True) -> Dict:
		"""Run security validation tests"""
		logger.info("Running security tests...")
		
		cmd = [
			sys.executable, "-m", "pytest",
			"test_performance.py",
			"-m", "security",
			"--tb=short", 
			"-v" if verbose else "-q"
		]
		
		result = self._run_pytest(cmd, "security_tests")
		return result
	
	def run_all_tests(self, verbose: bool = True, parallel: bool = False) -> Dict:
		"""Run complete test suite"""
		logger.info("Running complete test suite...")
		
		cmd = [
			sys.executable, "-m", "pytest",
			"test_models.py",
			"test_services.py", 
			"test_api.py",
			"test_performance.py",
			"--tb=short",
			"-v" if verbose else "-q",
			"--cov=.",
			"--cov-report=html",
			"--cov-report=term-missing"
		]
		
		if parallel:
			cmd.extend(["-n", "auto"])
		
		result = self._run_pytest(cmd, "all_tests")
		return result
	
	def run_quick_tests(self, verbose: bool = True) -> Dict:
		"""Run quick test suite (excluding slow tests)"""
		logger.info("Running quick test suite...")
		
		cmd = [
			sys.executable, "-m", "pytest",
			"test_models.py",
			"test_services.py",
			"test_api.py",
			"-m", "not (slow or performance)",
			"--tb=short",
			"-v" if verbose else "-q",
			"--maxfail=5"
		]
		
		result = self._run_pytest(cmd, "quick_tests")
		return result
	
	def run_ci_tests(self, verbose: bool = True) -> Dict:
		"""Run CI/CD optimized test suite"""
		logger.info("Running CI/CD test suite...")
		
		cmd = [
			sys.executable, "-m", "pytest",
			"test_models.py",
			"test_services.py", 
			"test_api.py",
			"-m", "not (slow or requires_network or requires_gpu)",
			"--tb=line",
			"-v" if verbose else "-q",
			"--cov=.",
			"--cov-report=xml",
			"--junit-xml=test-results.xml",
			"--maxfail=10"
		]
		
		result = self._run_pytest(cmd, "ci_tests")
		return result
	
	def _run_pytest(self, cmd: List[str], test_type: str) -> Dict:
		"""Run pytest command and capture results"""
		start_time = time.time()
		
		try:
			# Change to test directory
			os.chdir(self.base_dir)
			
			# Run pytest
			result = subprocess.run(
				cmd,
				capture_output=True,
				text=True,
				timeout=600  # 10 minute timeout
			)
			
			end_time = time.time()
			duration = end_time - start_time
			
			# Parse results
			test_result = {
				"test_type": test_type,
				"success": result.returncode == 0,
				"return_code": result.returncode,
				"duration": duration,
				"stdout": result.stdout,
				"stderr": result.stderr,
				"command": " ".join(cmd)
			}
			
			# Store results
			self.results[test_type] = test_result
			
			# Log results
			if test_result["success"]:
				logger.info(f"✅ {test_type} completed successfully in {duration:.2f}s")
			else:
				logger.error(f"❌ {test_type} failed in {duration:.2f}s")
				logger.error(f"Error output: {result.stderr}")
			
			return test_result
			
		except subprocess.TimeoutExpired:
			logger.error(f"⏰ {test_type} timed out after 10 minutes")
			return {
				"test_type": test_type,
				"success": False,
				"error": "timeout",
				"duration": 600
			}
		except Exception as e:
			logger.error(f"💥 Error running {test_type}: {str(e)}")
			return {
				"test_type": test_type,
				"success": False,
				"error": str(e),
				"duration": 0
			}
	
	def generate_report(self, output_file: Optional[str] = None) -> Dict:
		"""Generate comprehensive test report"""
		if not self.results:
			logger.warning("No test results to report")
			return {}
		
		# Calculate summary statistics
		total_tests = len(self.results)
		passed_tests = sum(1 for r in self.results.values() if r["success"])
		failed_tests = total_tests - passed_tests
		total_duration = sum(r.get("duration", 0) for r in self.results.values())
		
		report = {
			"summary": {
				"total_test_suites": total_tests,
				"passed_suites": passed_tests,
				"failed_suites": failed_tests,
				"success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
				"total_duration": total_duration,
				"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
			},
			"results": self.results
		}
		
		# Save report if output file specified
		if output_file:
			with open(output_file, 'w') as f:
				json.dump(report, f, indent=2)
			logger.info(f"Test report saved to {output_file}")
		
		return report
	
	def print_summary(self):
		"""Print test execution summary"""
		if not self.results:
			print("No tests executed")
			return
		
		print("\n" + "="*80)
		print("APG FACIAL RECOGNITION - TEST EXECUTION SUMMARY")
		print("="*80)
		
		total_tests = len(self.results)
		passed_tests = sum(1 for r in self.results.values() if r["success"])
		failed_tests = total_tests - passed_tests
		total_duration = sum(r.get("duration", 0) for r in self.results.values())
		
		print(f"Total Test Suites: {total_tests}")
		print(f"Passed: {passed_tests} ✅")
		print(f"Failed: {failed_tests} {'❌' if failed_tests > 0 else ''}")
		print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
		print(f"Total Duration: {total_duration:.2f}s")
		
		print("\nDetailed Results:")
		print("-" * 50)
		
		for test_type, result in self.results.items():
			status = "✅ PASS" if result["success"] else "❌ FAIL"
			duration = result.get("duration", 0)
			print(f"{test_type:20} | {status} | {duration:6.2f}s")
		
		print("=" * 80)

def main():
	"""Main test runner function"""
	parser = argparse.ArgumentParser(
		description="APG Facial Recognition Test Runner",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  python run_tests.py --all                    # Run all tests
  python run_tests.py --unit --api             # Run unit and API tests
  python run_tests.py --quick                  # Run quick test suite
  python run_tests.py --ci                     # Run CI-optimized tests
  python run_tests.py --performance --verbose  # Run performance tests with verbose output
		"""
	)
	
	# Test selection arguments
	parser.add_argument("--unit", action="store_true", help="Run unit tests")
	parser.add_argument("--integration", action="store_true", help="Run integration tests")
	parser.add_argument("--api", action="store_true", help="Run API tests")
	parser.add_argument("--performance", action="store_true", help="Run performance tests")
	parser.add_argument("--security", action="store_true", help="Run security tests")
	parser.add_argument("--all", action="store_true", help="Run all tests")
	parser.add_argument("--quick", action="store_true", help="Run quick test suite")
	parser.add_argument("--ci", action="store_true", help="Run CI/CD test suite")
	
	# Execution options
	parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
	parser.add_argument("--parallel", "-p", action="store_true", help="Run tests in parallel")
	parser.add_argument("--report", help="Generate JSON report file")
	
	args = parser.parse_args()
	
	# Create test runner
	runner = FacialTestRunner()
	
	# Determine which tests to run
	if args.all:
		runner.run_all_tests(verbose=args.verbose, parallel=args.parallel)
	elif args.quick:
		runner.run_quick_tests(verbose=args.verbose)
	elif args.ci:
		runner.run_ci_tests(verbose=args.verbose)
	else:
		# Run selected test suites
		if args.unit:
			runner.run_unit_tests(verbose=args.verbose)
		if args.integration:
			runner.run_integration_tests(verbose=args.verbose)
		if args.api:
			runner.run_api_tests(verbose=args.verbose)
		if args.performance:
			runner.run_performance_tests(verbose=args.verbose)
		if args.security:
			runner.run_security_tests(verbose=args.verbose)
		
		# If no specific tests selected, run quick suite
		if not any([args.unit, args.integration, args.api, args.performance, args.security]):
			print("No specific tests selected, running quick test suite...")
			runner.run_quick_tests(verbose=args.verbose)
	
	# Generate report if requested
	if args.report:
		runner.generate_report(args.report)
	
	# Print summary
	runner.print_summary()
	
	# Exit with error code if any tests failed
	failed_tests = sum(1 for r in runner.results.values() if not r["success"])
	sys.exit(failed_tests)

if __name__ == "__main__":
	main()