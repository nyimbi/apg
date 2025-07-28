#!/usr/bin/env python3
"""APG Cash Management - Test Runner

Comprehensive test runner with reporting and analytics.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import os
import sys
import subprocess
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Test suite configuration
TEST_SUITES = {
    'unit': {
        'description': 'Unit tests for core components',
        'files': ['test_unit_core.py'],
        'markers': ['unit'],
        'timeout': 300
    },
    'integration': {
        'description': 'Integration tests for component interactions',
        'files': ['test_integration_ai_ml.py'],
        'markers': ['integration'],
        'timeout': 600
    },
    'performance': {
        'description': 'Performance and load tests',
        'files': ['test_performance_load.py'],
        'markers': ['performance'],
        'timeout': 1800
    },
    'e2e': {
        'description': 'End-to-end workflow tests',
        'files': ['test_integration_e2e.py'],
        'markers': ['e2e'],
        'timeout': 900
    },
    'ml': {
        'description': 'Machine learning and AI tests',
        'files': ['test_integration_ai_ml.py'],
        'markers': ['ml'],
        'timeout': 1200
    },
    'risk': {
        'description': 'Risk analytics tests',
        'files': ['test_integration_ai_ml.py', 'test_integration_e2e.py'],
        'markers': ['risk'],
        'timeout': 600
    },
    'all': {
        'description': 'Complete test suite',
        'files': ['test_*.py'],
        'markers': [],
        'timeout': 3600
    }
}

class TestRunner:
    """Advanced test runner with reporting and analytics."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.results = {}
        self.test_dir = Path(__file__).parent
        
    def run_suite(
        self, 
        suite_name: str,
        parallel: bool = False,
        coverage: bool = True,
        verbose: bool = True,
        html_report: bool = True
    ) -> Dict:
        """Run a specific test suite."""
        
        if suite_name not in TEST_SUITES:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        suite_config = TEST_SUITES[suite_name]
        
        print(f"\n{'='*80}")
        print(f"Running APG Cash Management Test Suite: {suite_name.upper()}")
        print(f"Description: {suite_config['description']}")
        print(f"{'='*80}\n")
        
        self.start_time = time.time()
        
        # Build pytest command
        cmd = ['python', '-m', 'pytest']
        
        # Add test files or markers
        if suite_config['markers']:
            for marker in suite_config['markers']:
                cmd.extend(['-m', marker])
        else:
            # Add specific files if no markers
            for file_pattern in suite_config['files']:
                cmd.append(file_pattern)
        
        # Add options
        if verbose:
            cmd.append('-v')
        
        if coverage:
            cmd.extend([
                '--cov=../',
                '--cov-report=term-missing',
                '--cov-report=html:htmlcov',
                '--cov-fail-under=80'
            ])
        
        if html_report:
            cmd.extend([
                '--html=reports/test_report.html',
                '--self-contained-html'
            ])
        
        if parallel and suite_name != 'performance':  # Don't parallelize performance tests
            cmd.extend(['-n', 'auto'])
        
        # Add timeout
        cmd.extend(['--timeout', str(suite_config['timeout'])])
        
        # Set working directory
        cmd.extend(['--rootdir', str(self.test_dir.parent)])
        
        print(f"Executing: {' '.join(cmd)}")
        print(f"Working directory: {self.test_dir}")
        print(f"Timeout: {suite_config['timeout']} seconds\n")
        
        # Create reports directory
        reports_dir = self.test_dir / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        # Run tests
        try:
            result = subprocess.run(
                cmd,
                cwd=self.test_dir,
                capture_output=True,
                text=True,
                timeout=suite_config['timeout']
            )
            
            self.end_time = time.time()
            execution_time = self.end_time - self.start_time
            
            # Parse results
            success = result.returncode == 0
            
            self.results[suite_name] = {
                'success': success,
                'execution_time': execution_time,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': ' '.join(cmd)
            }
            
            # Print results
            self._print_results(suite_name)
            
            return self.results[suite_name]
            
        except subprocess.TimeoutExpired:
            self.end_time = time.time()
            execution_time = self.end_time - self.start_time
            
            print(f"❌ Test suite '{suite_name}' timed out after {execution_time:.1f} seconds")
            
            self.results[suite_name] = {
                'success': False,
                'execution_time': execution_time,
                'return_code': -1,
                'stdout': '',
                'stderr': 'Test suite timed out',
                'command': ' '.join(cmd)
            }
            
            return self.results[suite_name]
        
        except Exception as e:
            print(f"❌ Error running test suite '{suite_name}': {e}")
            
            self.results[suite_name] = {
                'success': False,
                'execution_time': 0,
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
                'command': ' '.join(cmd)
            }
            
            return self.results[suite_name]
    
    def run_multiple_suites(
        self, 
        suite_names: List[str],
        **kwargs
    ) -> Dict:
        """Run multiple test suites."""
        
        print(f"\n{'='*100}")
        print(f"APG CASH MANAGEMENT - COMPREHENSIVE TEST EXECUTION")
        print(f"Running {len(suite_names)} test suites: {', '.join(suite_names)}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*100}\n")
        
        overall_start = time.time()
        suite_results = {}
        
        for suite_name in suite_names:
            try:
                result = self.run_suite(suite_name, **kwargs)
                suite_results[suite_name] = result
                
                # Brief status update
                status = "✅ PASSED" if result['success'] else "❌ FAILED"
                print(f"\n{status} | {suite_name.upper()} | {result['execution_time']:.1f}s\n")
                
            except Exception as e:
                print(f"❌ Critical error in suite '{suite_name}': {e}")
                suite_results[suite_name] = {
                    'success': False,
                    'execution_time': 0,
                    'error': str(e)
                }
        
        overall_time = time.time() - overall_start
        
        # Generate comprehensive report
        self._generate_comprehensive_report(suite_results, overall_time)
        
        return suite_results
    
    def _print_results(self, suite_name: str):
        """Print detailed test results."""
        
        result = self.results[suite_name]
        
        print(f"\n{'='*80}")
        print(f"TEST SUITE RESULTS: {suite_name.upper()}")
        print(f"{'='*80}")
        
        # Overall status
        if result['success']:
            print(f"✅ Status: PASSED")
        else:
            print(f"❌ Status: FAILED (Exit code: {result['return_code']})")
        
        print(f"⏱️  Execution time: {result['execution_time']:.2f} seconds")
        
        # Parse stdout for test statistics
        stdout = result['stdout']
        if 'passed' in stdout or 'failed' in stdout:
            # Extract test summary line
            lines = stdout.split('\n')
            for line in lines:
                if 'passed' in line and ('failed' in line or 'error' in line or 'skipped' in line):
                    print(f"📊 Test summary: {line.strip()}")
                    break
        
        # Show coverage if available
        if '--cov' in result['command'] and 'TOTAL' in stdout:
            lines = stdout.split('\n')
            for line in lines:
                if 'TOTAL' in line and '%' in line:
                    print(f"📈 Coverage: {line.strip()}")
                    break
        
        # Show errors if any
        if not result['success'] and result['stderr']:
            print(f"\n❗ Error output:")
            print(result['stderr'][:500] + '...' if len(result['stderr']) > 500 else result['stderr'])
        
        print(f"{'='*80}\n")
    
    def _generate_comprehensive_report(self, suite_results: Dict, total_time: float):
        """Generate comprehensive test execution report."""
        
        print(f"\n{'='*100}")
        print(f"COMPREHENSIVE TEST EXECUTION REPORT")
        print(f"{'='*100}")
        
        # Summary statistics
        total_suites = len(suite_results)
        passed_suites = sum(1 for r in suite_results.values() if r['success'])
        failed_suites = total_suites - passed_suites
        
        print(f"📊 Execution Summary:")
        print(f"   • Total test suites: {total_suites}")
        print(f"   • Passed: {passed_suites}")
        print(f"   • Failed: {failed_suites}")
        print(f"   • Success rate: {(passed_suites/total_suites)*100:.1f}%")
        print(f"   • Total execution time: {total_time:.1f} seconds")
        
        # Per-suite breakdown
        print(f"\n📋 Suite Breakdown:")
        for suite_name, result in suite_results.items():
            status_icon = "✅" if result['success'] else "❌"
            time_str = f"{result['execution_time']:.1f}s"
            print(f"   {status_icon} {suite_name.upper():<15} {time_str:>8}")
        
        # Performance analysis
        fastest_suite = min(suite_results.items(), key=lambda x: x[1]['execution_time'])
        slowest_suite = max(suite_results.items(), key=lambda x: x[1]['execution_time'])
        
        print(f"\n⚡ Performance Analysis:")
        print(f"   • Fastest suite: {fastest_suite[0]} ({fastest_suite[1]['execution_time']:.1f}s)")
        print(f"   • Slowest suite: {slowest_suite[0]} ({slowest_suite[1]['execution_time']:.1f}s)")
        
        # Failed suites analysis
        if failed_suites > 0:
            print(f"\n🔍 Failed Suites Analysis:")
            for suite_name, result in suite_results.items():
                if not result['success']:
                    print(f"   ❌ {suite_name.upper()}:")
                    if 'error' in result:
                        print(f"      Error: {result['error']}")
                    elif result['return_code'] != 0:
                        print(f"      Exit code: {result['return_code']}")
                        if result['stderr']:
                            error_preview = result['stderr'][:200]
                            print(f"      Error preview: {error_preview}...")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if failed_suites == 0:
            print(f"   ✅ All test suites passed successfully!")
            print(f"   ✅ System is ready for production deployment.")
        else:
            print(f"   ⚠️  {failed_suites} test suite(s) failed - review before deployment")
            print(f"   🔧 Check error logs and fix failing tests")
            if total_time > 1800:  # 30 minutes
                print(f"   ⚡ Consider optimizing slow test suites")
        
        # Generate timestamp
        print(f"\n📅 Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*100}\n")
        
        # Save report to file
        self._save_report_to_file(suite_results, total_time)
    
    def _save_report_to_file(self, suite_results: Dict, total_time: float):
        """Save test report to file."""
        
        reports_dir = self.test_dir / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f'test_execution_report_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write("APG Cash Management - Test Execution Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Execution Time: {total_time:.1f} seconds\n\n")
            
            for suite_name, result in suite_results.items():
                f.write(f"Suite: {suite_name.upper()}\n")
                f.write(f"Status: {'PASSED' if result['success'] else 'FAILED'}\n")
                f.write(f"Time: {result['execution_time']:.1f}s\n")
                if not result['success']:
                    f.write(f"Error: {result.get('stderr', 'Unknown error')}\n")
                f.write("\n")
        
        print(f"📄 Detailed report saved to: {report_file}")

def main():
    """Main test runner entry point."""
    
    parser = argparse.ArgumentParser(
        description='APG Cash Management Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available test suites:
  unit        - Unit tests for core components
  integration - Integration tests for component interactions  
  performance - Performance and load tests
  e2e         - End-to-end workflow tests
  ml          - Machine learning and AI tests
  risk        - Risk analytics tests
  all         - Complete test suite

Examples:
  python run_tests.py unit                    # Run unit tests only
  python run_tests.py unit integration        # Run multiple suites
  python run_tests.py all --no-coverage       # Run all tests without coverage
  python run_tests.py performance --parallel  # Run performance tests in parallel
        """
    )
    
    parser.add_argument(
        'suites',
        nargs='+',
        choices=list(TEST_SUITES.keys()),
        help='Test suite(s) to run'
    )
    
    parser.add_argument(
        '--no-coverage',
        action='store_false',
        dest='coverage',
        help='Disable code coverage reporting'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run tests in parallel (where applicable)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_false',
        dest='verbose',
        help='Run tests in quiet mode'
    )
    
    parser.add_argument(
        '--no-html',
        action='store_false',
        dest='html_report',
        help='Disable HTML test report generation'
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner()
    
    try:
        # Run requested test suites
        if len(args.suites) == 1:
            result = runner.run_suite(
                args.suites[0],
                parallel=args.parallel,
                coverage=args.coverage,
                verbose=args.verbose,
                html_report=args.html_report
            )
            
            # Exit with appropriate code
            sys.exit(0 if result['success'] else 1)
        
        else:
            results = runner.run_multiple_suites(
                args.suites,
                parallel=args.parallel,
                coverage=args.coverage,
                verbose=args.verbose,
                html_report=args.html_report
            )
            
            # Exit with appropriate code
            all_passed = all(r['success'] for r in results.values())
            sys.exit(0 if all_passed else 1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n❌ Critical error in test runner: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()