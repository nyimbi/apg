"""
APG Accounts Receivable - Performance Test Runner
Comprehensive performance testing suite runner with reporting

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import json


class PerformanceTestRunner:
	"""Comprehensive performance test runner with reporting."""
	
	def __init__(self):
		self.test_results: List[Dict[str, Any]] = []
		self.start_time = None
		self.end_time = None
		self.performance_dir = Path(__file__).parent
	
	def run_test_suite(self, test_patterns: List[str] = None) -> Dict[str, Any]:
		"""Run the complete performance test suite."""
		print("🚀 Starting APG Accounts Receivable Performance Test Suite")
		print("=" * 70)
		
		self.start_time = time.time()
		
		# Default test patterns if none provided
		if test_patterns is None:
			test_patterns = [
				"test_load_performance.py",
				"test_memory_performance.py", 
				"test_scalability.py"
			]
		
		# Run each test file
		for test_pattern in test_patterns:
			test_file = self.performance_dir / test_pattern
			if test_file.exists():
				print(f"\n📊 Running {test_pattern}")
				print("-" * 50)
				
				result = self._run_pytest(test_file)
				self.test_results.append({
					'test_file': test_pattern,
					'result': result,
					'timestamp': datetime.now().isoformat()
				})
			else:
				print(f"⚠️  Test file not found: {test_pattern}")
		
		self.end_time = time.time()
		
		# Generate comprehensive report
		report = self._generate_report()
		self._save_report(report)
		
		return report
	
	def _run_pytest(self, test_file: Path) -> Dict[str, Any]:
		"""Run pytest on a specific test file."""
		cmd = [
			sys.executable, "-m", "pytest",
			str(test_file),
			"-v", "-s", "-m", "performance",
			"--tb=short",
			"--disable-warnings"
		]
		
		start_time = time.time()
		
		try:
			result = subprocess.run(
				cmd,
				capture_output=True,
				text=True,
				timeout=300  # 5 minute timeout per test file
			)
			
			duration = time.time() - start_time
			
			# Parse pytest output for test counts
			output_lines = result.stdout.split('\n')
			
			# Extract test results from pytest output
			passed_count = 0
			failed_count = 0
			skipped_count = 0
			
			for line in output_lines:
				if " passed" in line:
					# Extract numbers from lines like "5 passed, 2 skipped in 1.23s"
					parts = line.split()
					for i, part in enumerate(parts):
						if part == "passed":
							passed_count = int(parts[i-1])
						elif part == "failed":
							failed_count = int(parts[i-1])
						elif part == "skipped":
							skipped_count = int(parts[i-1])
			
			return {
				'status': 'success' if result.returncode == 0 else 'failed',
				'duration': duration,
				'passed': passed_count,
				'failed': failed_count,
				'skipped': skipped_count,
				'output': result.stdout,
				'errors': result.stderr,
				'return_code': result.returncode
			}
			
		except subprocess.TimeoutExpired:
			return {
				'status': 'timeout',
				'duration': 300,
				'passed': 0,
				'failed': 0,
				'skipped': 0,
				'output': '',
				'errors': 'Test timed out after 5 minutes',
				'return_code': -1
			}
		except Exception as e:
			return {
				'status': 'error',
				'duration': time.time() - start_time,
				'passed': 0,
				'failed': 0,
				'skipped': 0,
				'output': '',
				'errors': str(e),
				'return_code': -1
			}
	
	def _generate_report(self) -> Dict[str, Any]:
		"""Generate comprehensive performance test report."""
		total_duration = self.end_time - self.start_time if self.start_time and self.end_time else 0
		
		# Aggregate results
		total_passed = sum(r['result']['passed'] for r in self.test_results)
		total_failed = sum(r['result']['failed'] for r in self.test_results)
		total_skipped = sum(r['result']['skipped'] for r in self.test_results)
		total_tests = total_passed + total_failed + total_skipped
		
		# Calculate success rates
		success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
		
		# Determine overall status
		overall_status = 'PASSED' if total_failed == 0 else 'FAILED'
		
		# Extract performance insights from test outputs
		performance_insights = self._extract_performance_insights()
		
		report = {
			'summary': {
				'overall_status': overall_status,
				'total_duration': total_duration,
				'total_tests': total_tests,
				'passed': total_passed,
				'failed': total_failed,
				'skipped': total_skipped,
				'success_rate': success_rate,
				'timestamp': datetime.now().isoformat()
			},
			'test_results': self.test_results,
			'performance_insights': performance_insights,
			'recommendations': self._generate_recommendations()
		}
		
		return report
	
	def _extract_performance_insights(self) -> Dict[str, Any]:
		"""Extract performance insights from test outputs."""
		insights = {
			'load_performance': {},
			'memory_performance': {},
			'scalability_performance': {}
		}
		
		for test_result in self.test_results:
			test_file = test_result['test_file']
			output = test_result['result']['output']
			
			if 'load_performance' in test_file:
				insights['load_performance'] = self._parse_load_performance(output)
			elif 'memory_performance' in test_file:
				insights['memory_performance'] = self._parse_memory_performance(output)
			elif 'scalability' in test_file:
				insights['scalability_performance'] = self._parse_scalability_performance(output)
		
		return insights
	
	def _parse_load_performance(self, output: str) -> Dict[str, Any]:
		"""Parse load performance metrics from test output."""
		metrics = {}
		
		lines = output.split('\n')
		for line in lines:
			if 'Success Rate:' in line:
				try:
					rate = float(line.split(':')[1].strip().replace('%', ''))
					metrics['success_rate'] = rate
				except (ValueError, IndexError):
					pass
			elif 'Avg Response Time:' in line:
				try:
					time_ms = float(line.split(':')[1].strip().replace('ms', ''))
					metrics['avg_response_time_ms'] = time_ms
				except (ValueError, IndexError):
					pass
			elif 'Throughput:' in line:
				try:
					rps = float(line.split(':')[1].strip().replace('RPS', '').strip())
					metrics['throughput_rps'] = rps
				except (ValueError, IndexError):
					pass
		
		return metrics
	
	def _parse_memory_performance(self, output: str) -> Dict[str, Any]:
		"""Parse memory performance metrics from test output."""
		metrics = {}
		
		lines = output.split('\n')
		for line in lines:
			if 'Peak Memory Increase:' in line:
				try:
					mem_mb = float(line.split(':')[1].strip().replace('MB', ''))
					metrics['peak_memory_increase_mb'] = mem_mb
				except (ValueError, IndexError):
					pass
			elif 'Memory per Customer:' in line:
				try:
					mem_bytes = float(line.split(':')[1].strip().replace('bytes', ''))
					metrics['memory_per_customer_bytes'] = mem_bytes
				except (ValueError, IndexError):
					pass
			elif 'Cleanup Efficiency:' in line:
				try:
					efficiency = float(line.split(':')[1].strip().replace('%', ''))
					metrics['cleanup_efficiency_percent'] = efficiency
				except (ValueError, IndexError):
					pass
		
		return metrics
	
	def _parse_scalability_performance(self, output: str) -> Dict[str, Any]:
		"""Parse scalability performance metrics from test output."""
		metrics = {}
		
		lines = output.split('\n')
		for line in lines:
			if 'Scalability Efficiency:' in line:
				try:
					efficiency = float(line.split(':')[1].strip().replace('%', ''))
					metrics['scalability_efficiency_percent'] = efficiency
				except (ValueError, IndexError):
					pass
			elif 'Performance Degradation:' in line:
				try:
					degradation = float(line.split(':')[1].strip().replace('x', ''))
					metrics['performance_degradation_factor'] = degradation
				except (ValueError, IndexError):
					pass
		
		return metrics
	
	def _generate_recommendations(self) -> List[str]:
		"""Generate performance optimization recommendations."""
		recommendations = []
		
		# Analyze results and provide recommendations
		for test_result in self.test_results:
			if test_result['result']['status'] == 'failed':
				recommendations.append(f"❌ {test_result['test_file']} failed - investigate performance bottlenecks")
			elif test_result['result']['status'] == 'timeout':
				recommendations.append(f"⏱️ {test_result['test_file']} timed out - consider optimizing long-running operations")
		
		# Check specific performance metrics
		insights = self.test_results
		
		# Generic recommendations based on common patterns
		if len([r for r in self.test_results if r['result']['status'] == 'success']) == len(self.test_results):
			recommendations.append("✅ All performance tests passed - system meets performance requirements")
		
		recommendations.extend([
			"📈 Monitor production performance metrics against these benchmarks",
			"🔧 Consider implementing performance alerting based on these thresholds",
			"📊 Run performance tests regularly as part of CI/CD pipeline",
			"⚡ Profile code in production to identify optimization opportunities"
		])
		
		return recommendations
	
	def _save_report(self, report: Dict[str, Any]):
		"""Save performance test report to file."""
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		report_file = self.performance_dir / f"performance_report_{timestamp}.json"
		
		try:
			with open(report_file, 'w') as f:
				json.dump(report, f, indent=2, default=str)
			print(f"\n📄 Performance report saved to: {report_file}")
		except Exception as e:
			print(f"\n⚠️  Failed to save report: {e}")
	
	def print_summary(self, report: Dict[str, Any]):
		"""Print performance test summary."""
		summary = report['summary']
		
		print(f"\n🏁 Performance Test Suite Complete")
		print("=" * 70)
		print(f"Overall Status: {summary['overall_status']}")
		print(f"Total Duration: {summary['total_duration']:.2f} seconds")
		print(f"Tests: {summary['total_tests']} total, {summary['passed']} passed, {summary['failed']} failed, {summary['skipped']} skipped")
		print(f"Success Rate: {summary['success_rate']:.1f}%")
		
		print(f"\n📊 Performance Insights:")
		for category, metrics in report['performance_insights'].items():
			if metrics:
				print(f"  {category}:")
				for metric, value in metrics.items():
					print(f"    {metric}: {value}")
		
		print(f"\n💡 Recommendations:")
		for rec in report['recommendations']:
			print(f"  {rec}")


def main():
	"""Main entry point for performance test runner."""
	runner = PerformanceTestRunner()
	
	# Check if specific test files were provided as arguments
	test_patterns = sys.argv[1:] if len(sys.argv) > 1 else None
	
	try:
		report = runner.run_test_suite(test_patterns)
		runner.print_summary(report)
		
		# Exit with appropriate code
		sys.exit(0 if report['summary']['overall_status'] == 'PASSED' else 1)
		
	except KeyboardInterrupt:
		print("\n⚠️  Performance tests interrupted by user")
		sys.exit(130)
	except Exception as e:
		print(f"\n❌ Performance test runner failed: {e}")
		sys.exit(1)


if __name__ == "__main__":
	main()