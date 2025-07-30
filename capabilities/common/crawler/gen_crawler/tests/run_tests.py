#!/usr/bin/env python3
"""
Gen Crawler Test Runner
======================

Comprehensive test runner for the gen_crawler package with detailed
reporting and coverage analysis.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --verbose          # Verbose output
    python run_tests.py --module core      # Run specific module tests

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import argparse
import sys
import unittest
import os
import importlib
from pathlib import Path
import time
from typing import List, Dict, Any
import traceback

# Add package to path
package_dir = Path(__file__).parent.parent
sys.path.insert(0, str(package_dir))

class TestResult:
    """Store test execution results."""
    
    def __init__(self):
        self.total_tests = 0
        self.successful_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0
        self.errors = []
        self.failures = []
        self.execution_time = 0.0
        self.coverage_data = {}

class GenCrawlerTestRunner:
    """Custom test runner for gen_crawler package."""
    
    def __init__(self, verbosity: int = 1):
        """Initialize test runner."""
        self.verbosity = verbosity
        self.test_modules = {
            'core': 'test_core',
            'config': 'test_config', 
            'parsers': 'test_parsers',
            'cli': 'test_cli',
            'integration': 'test_integration'
        }
        
        self.unit_modules = ['core', 'config', 'parsers', 'cli']
        self.integration_modules = ['integration']
    
    def discover_tests(self, module_name: str = None) -> unittest.TestSuite:
        """Discover tests in specified module or all modules."""
        suite = unittest.TestSuite()
        
        if module_name:
            if module_name in self.test_modules:
                suite.addTest(self._load_module_tests(self.test_modules[module_name]))
            else:
                print(f"Unknown module: {module_name}")
                print(f"Available modules: {list(self.test_modules.keys())}")
        else:
            # Load all test modules
            for test_module in self.test_modules.values():
                suite.addTest(self._load_module_tests(test_module))
        
        return suite
    
    def _load_module_tests(self, module_name: str) -> unittest.TestSuite:
        """Load tests from a specific module."""
        try:
            # Try to import the test module
            module = importlib.import_module(f'.{module_name}', package='tests')
            
            # Create test suite from module
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(module)
            
            return suite
            
        except ImportError as e:
            print(f"Warning: Could not import {module_name}: {e}")
            return unittest.TestSuite()
        except Exception as e:
            print(f"Error loading tests from {module_name}: {e}")
            return unittest.TestSuite()
    
    def run_tests(self, test_suite: unittest.TestSuite, 
                  run_coverage: bool = False) -> TestResult:
        """Run test suite and return results."""
        
        result = TestResult()
        start_time = time.time()
        
        # Set up test runner
        if self.verbosity >= 2:
            stream = sys.stdout
        else:
            stream = open(os.devnull, 'w')
        
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=self.verbosity,
            buffer=True
        )
        
        try:
            # Run coverage if requested
            if run_coverage:
                result.coverage_data = self._run_with_coverage(test_suite, runner)
            else:
                test_result = runner.run(test_suite)
                self._process_test_result(test_result, result)
        
        except Exception as e:
            result.errors.append(f"Test runner error: {e}")
            print(f"Error running tests: {e}")
            traceback.print_exc()
        
        finally:
            if stream != sys.stdout:
                stream.close()
        
        result.execution_time = time.time() - start_time
        return result
    
    def _run_with_coverage(self, test_suite: unittest.TestSuite, 
                          runner: unittest.TextTestRunner) -> Dict[str, Any]:
        """Run tests with coverage analysis."""
        try:
            import coverage
            
            # Start coverage
            cov = coverage.Coverage(source=['gen_crawler'])
            cov.start()
            
            # Run tests
            test_result = runner.run(test_suite)
            
            # Stop coverage
            cov.stop()
            cov.save()
            
            # Generate coverage report
            coverage_data = {
                'coverage_available': True,
                'total_coverage': 0.0,
                'missing_lines': {},
                'covered_lines': {}
            }
            
            # Get coverage data
            total_statements = 0
            total_missing = 0
            
            for filename in cov.get_data().measured_files():
                analysis = cov.analysis(filename)
                statements, missing, excluded, branches = analysis[1:]
                
                file_coverage = len(statements - missing) / len(statements) * 100 if statements else 0
                
                coverage_data['missing_lines'][filename] = list(missing)
                coverage_data['covered_lines'][filename] = file_coverage
                
                total_statements += len(statements)
                total_missing += len(missing)
            
            if total_statements > 0:
                coverage_data['total_coverage'] = (total_statements - total_missing) / total_statements * 100
            
            # Process test results
            self._process_test_result(test_result, TestResult())
            
            return coverage_data
            
        except ImportError:
            print("Coverage module not available. Install with: pip install coverage")
            return {'coverage_available': False}
        except Exception as e:
            print(f"Coverage analysis failed: {e}")
            return {'coverage_available': False, 'error': str(e)}
    
    def _process_test_result(self, test_result: unittest.TestResult, 
                           result: TestResult) -> None:
        """Process unittest results into our TestResult format."""
        
        result.total_tests = test_result.testsRun
        result.successful_tests = test_result.testsRun - len(test_result.failures) - len(test_result.errors)
        result.failed_tests = len(test_result.failures)
        result.skipped_tests = len(test_result.skipped) if hasattr(test_result, 'skipped') else 0
        
        # Store error details
        for test, error in test_result.errors:
            result.errors.append(f"{test}: {error}")
        
        for test, failure in test_result.failures:
            result.failures.append(f"{test}: {failure}")
    
    def print_results(self, result: TestResult, module_name: str = None) -> None:
        """Print formatted test results."""
        
        print("\n" + "="*80)
        if module_name:
            print(f"Gen Crawler Test Results - {module_name.upper()} Module")
        else:
            print("Gen Crawler Test Results - All Modules")
        print("="*80)
        
        print(f"Total Tests: {result.total_tests}")
        print(f"Successful: {result.successful_tests}")
        print(f"Failed: {result.failed_tests}")
        print(f"Skipped: {result.skipped_tests}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        
        if result.total_tests > 0:
            success_rate = (result.successful_tests / result.total_tests) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        # Print coverage info if available
        if result.coverage_data.get('coverage_available'):
            total_coverage = result.coverage_data.get('total_coverage', 0)
            print(f"Code Coverage: {total_coverage:.1f}%")
        
        # Print errors and failures
        if result.errors:
            print(f"\n{len(result.errors)} ERRORS:")
            for i, error in enumerate(result.errors, 1):
                print(f"{i}. {error}")
        
        if result.failures:
            print(f"\n{len(result.failures)} FAILURES:")
            for i, failure in enumerate(result.failures, 1):
                print(f"{i}. {failure}")
        
        # Overall status
        print("\n" + "-"*80)
        if result.failed_tests == 0 and len(result.errors) == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED")
        print("-"*80)

def check_dependencies() -> Dict[str, bool]:
    """Check availability of test dependencies."""
    
    dependencies = {
        'unittest': True,  # Built-in
        'pytest': False,
        'coverage': False,
        'mock': True,  # Built-in from Python 3.3+
        'gen_crawler_core': False,
        'gen_crawler_config': False,
        'gen_crawler_parsers': False,
        'gen_crawler_cli': False
    }
    
    # Check external dependencies
    try:
        import pytest
        dependencies['pytest'] = True
    except ImportError:
        pass
    
    try:
        import coverage
        dependencies['coverage'] = True
    except ImportError:
        pass
    
    # Check gen_crawler components
    try:
        from gen_crawler.core import GenCrawler
        dependencies['gen_crawler_core'] = True
    except ImportError:
        pass
    
    try:
        from gen_crawler.config import create_gen_config
        dependencies['gen_crawler_config'] = True
    except ImportError:
        pass
    
    try:
        from gen_crawler.parsers import GenContentParser
        dependencies['gen_crawler_parsers'] = True
    except ImportError:
        pass
    
    try:
        from gen_crawler.cli.main import create_cli_parser
        dependencies['gen_crawler_cli'] = True
    except ImportError:
        pass
    
    return dependencies

def print_dependency_status(dependencies: Dict[str, bool]) -> None:
    """Print dependency availability status."""
    
    print("📋 Test Dependencies Status:")
    print("-" * 40)
    
    for dep, available in dependencies.items():
        status = "✅ Available" if available else "❌ Missing"
        print(f"{dep:<25} {status}")
    
    print()
    
    # Check critical dependencies
    critical_missing = []
    for dep in ['gen_crawler_core', 'gen_crawler_config', 'gen_crawler_parsers']:
        if not dependencies[dep]:
            critical_missing.append(dep)
    
    if critical_missing:
        print("⚠️  Critical components missing:")
        for dep in critical_missing:
            print(f"   - {dep}")
        print("   Some tests may be skipped.")
    else:
        print("✅ All critical components available.")

def run_specific_test_method(test_class: str, test_method: str) -> None:
    """Run a specific test method."""
    
    try:
        # Import the test module
        module_name = None
        for module, file in [
            ('core', 'test_core'),
            ('config', 'test_config'),
            ('parsers', 'test_parsers'),
            ('cli', 'test_cli'),
            ('integration', 'test_integration')
        ]:
            try:
                test_module = importlib.import_module(f'.{file}', package='tests')
                if hasattr(test_module, test_class):
                    module_name = file
                    break
            except ImportError:
                continue
        
        if not module_name:
            print(f"Test class {test_class} not found")
            return
        
        # Create test suite
        suite = unittest.TestSuite()
        suite.addTest(getattr(test_module, test_class)(test_method))
        
        # Run test
        runner = GenCrawlerTestRunner(verbosity=2)
        result = runner.run_tests(suite)
        runner.print_results(result, f"{test_class}.{test_method}")
        
    except Exception as e:
        print(f"Error running specific test: {e}")
        traceback.print_exc()

def main():
    """Main test runner function."""
    
    parser = argparse.ArgumentParser(
        description="Gen Crawler Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                     # Run all tests
  python run_tests.py --unit              # Run only unit tests
  python run_tests.py --integration       # Run only integration tests
  python run_tests.py --module core       # Run core module tests
  python run_tests.py --coverage          # Run with coverage
  python run_tests.py --verbose           # Verbose output
  python run_tests.py --deps              # Check dependencies only
        """
    )
    
    parser.add_argument(
        '--unit',
        action='store_true',
        help='Run only unit tests'
    )
    
    parser.add_argument(
        '--integration',
        action='store_true',
        help='Run only integration tests'
    )
    
    parser.add_argument(
        '--module',
        type=str,
        help='Run tests for specific module (core, config, parsers, cli, integration)'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run with coverage analysis'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose test output'
    )
    
    parser.add_argument(
        '--deps',
        action='store_true',
        help='Check dependencies and exit'
    )
    
    parser.add_argument(
        '--test',
        type=str,
        help='Run specific test (format: TestClass.test_method)'
    )
    
    args = parser.parse_args()
    
    # Set verbosity
    verbosity = 2 if args.verbose else 1
    
    # Check dependencies
    dependencies = check_dependencies()
    
    if args.deps:
        print_dependency_status(dependencies)
        return
    
    print_dependency_status(dependencies)
    print()
    
    # Check for critical missing dependencies
    if not any([
        dependencies['gen_crawler_core'],
        dependencies['gen_crawler_config'],
        dependencies['gen_crawler_parsers']
    ]):
        print("❌ Critical components missing. Cannot run tests.")
        print("Make sure gen_crawler package is properly installed.")
        sys.exit(1)
    
    # Create test runner
    runner = GenCrawlerTestRunner(verbosity=verbosity)
    
    # Handle specific test
    if args.test:
        if '.' in args.test:
            test_class, test_method = args.test.split('.', 1)
            run_specific_test_method(test_class, test_method)
        else:
            print("Test format should be: TestClass.test_method")
        return
    
    # Determine which tests to run
    modules_to_run = []
    
    if args.unit:
        modules_to_run = runner.unit_modules
        print("🧪 Running Unit Tests...")
    elif args.integration:
        modules_to_run = runner.integration_modules  
        print("🔗 Running Integration Tests...")
    elif args.module:
        if args.module in runner.test_modules:
            modules_to_run = [args.module]
            print(f"🎯 Running {args.module.upper()} Module Tests...")
        else:
            print(f"Unknown module: {args.module}")
            print(f"Available modules: {list(runner.test_modules.keys())}")
            sys.exit(1)
    else:
        modules_to_run = list(runner.test_modules.keys())
        print("🚀 Running All Tests...")
    
    print()
    
    # Run tests for each module
    all_results = []
    overall_start_time = time.time()
    
    for module in modules_to_run:
        print(f"📦 Testing {module} module...")
        
        # Discover and run tests
        test_suite = runner.discover_tests(module)
        result = runner.run_tests(test_suite, run_coverage=args.coverage)
        
        # Store results
        result.module_name = module
        all_results.append(result)
        
        # Print module results
        runner.print_results(result, module)
        print()
    
    # Print overall summary
    if len(all_results) > 1:
        print("="*80)
        print("OVERALL SUMMARY")
        print("="*80)
        
        total_tests = sum(r.total_tests for r in all_results)
        total_successful = sum(r.successful_tests for r in all_results)
        total_failed = sum(r.failed_tests for r in all_results)
        total_skipped = sum(r.skipped_tests for r in all_results)
        total_time = time.time() - overall_start_time
        
        print(f"Modules Tested: {len(all_results)}")
        print(f"Total Tests: {total_tests}")
        print(f"Total Successful: {total_successful}")
        print(f"Total Failed: {total_failed}")
        print(f"Total Skipped: {total_skipped}")
        print(f"Total Time: {total_time:.2f}s")
        
        if total_tests > 0:
            overall_success_rate = (total_successful / total_tests) * 100
            print(f"Overall Success Rate: {overall_success_rate:.1f}%")
        
        print("\n" + "-"*80)
        if total_failed == 0:
            print("🎉 ALL TESTS PASSED ACROSS ALL MODULES!")
        else:
            print("❌ SOME TESTS FAILED")
            
            # Show which modules had failures
            failed_modules = [r.module_name for r in all_results if r.failed_tests > 0 or r.errors]
            if failed_modules:
                print(f"Modules with failures: {', '.join(failed_modules)}")
        
        print("-"*80)

if __name__ == '__main__':
    main()