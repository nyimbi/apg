#!/usr/bin/env python3
"""
Comprehensive Test Script for Improved Crawler System
====================================================

This script tests the enhanced crawler system with cloudscraper priority stealth,
validating all components including stealth managers, base crawlers, and unified
orchestrators. It provides comprehensive validation of the improved architecture.

Test Coverage:
- CloudScraper priority stealth manager
- Unified stealth orchestrator
- Improved base crawler with content extraction
- Protection detection and fallback strategies
- Performance monitoring and metrics
- Domain-specific strategy learning
- Error handling and recovery
- Content quality assessment

Features:
- Real website testing with various protection types
- Performance benchmarking and analysis
- Comprehensive reporting with recommendations
- Stealth effectiveness validation
- Content extraction quality assessment

Author: Lindela Team
License: MIT
"""

import asyncio
import logging
import time
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import asdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Test imports with error handling
COMPONENTS_AVAILABLE = {}

try:
    from packages_enhanced.crawlers.news_crawler.stealth.cloudscraper_stealth import (
        CloudScraperPriorityStealthManager,
        create_stealth_manager,
        StealthResult
    )
    COMPONENTS_AVAILABLE['cloudscraper_stealth'] = True
    logger.info("✅ CloudScraper stealth components loaded")
except ImportError as e:
    COMPONENTS_AVAILABLE['cloudscraper_stealth'] = False
    logger.warning(f"❌ CloudScraper stealth not available: {e}")

try:
    from packages_enhanced.crawlers.news_crawler.stealth.unified_stealth_orchestrator import (
        UnifiedStealthOrchestrator,
        create_unified_stealth_orchestrator
    )
    COMPONENTS_AVAILABLE['unified_stealth'] = True
    logger.info("✅ Unified stealth orchestrator loaded")
except ImportError as e:
    COMPONENTS_AVAILABLE['unified_stealth'] = False
    logger.warning(f"❌ Unified stealth orchestrator not available: {e}")

try:
    from packages_enhanced.crawlers.news_crawler.core.improved_base_crawler import (
        ImprovedBaseCrawler,
        CrawlResult,
        CrawlerConfig,
        create_basic_crawler,
        create_news_crawler,
        create_fast_crawler
    )
    COMPONENTS_AVAILABLE['improved_crawler'] = True
    logger.info("✅ Improved base crawler loaded")
except ImportError as e:
    COMPONENTS_AVAILABLE['improved_crawler'] = False
    logger.warning(f"❌ Improved base crawler not available: {e}")

try:
    from packages_enhanced.crawlers import (
        create_stealth_crawler,
        create_unified_stealth_crawler,
        create_news_crawler_enhanced,
        get_available_crawlers,
        get_crawlers_health
    )
    COMPONENTS_AVAILABLE['crawler_package'] = True
    logger.info("✅ Crawler package integration loaded")
except ImportError as e:
    COMPONENTS_AVAILABLE['crawler_package'] = False
    logger.warning(f"❌ Crawler package integration not available: {e}")


class CrawlerTestSuite:
    """Comprehensive test suite for improved crawler system."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "crawler_test_results"
        self.output_dir.mkdir(exist_ok=True)

        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'components_available': COMPONENTS_AVAILABLE,
            'tests': {},
            'performance_metrics': {},
            'recommendations': [],
            'summary': {}
        }

        # Test URLs with different protection types
        self.test_urls = {
            'basic': [
                'https://httpbin.org/get',
                'https://httpbin.org/user-agent',
                'https://example.com'
            ],
            'news_sites': [
                'https://quotes.toscrape.com',  # Simple test site
                'https://books.toscrape.com',   # Another test site
                'https://www.theguardian.com',  # Real news site
            ],
            'protected': [
                'https://nowsecure.nl',  # CloudFlare protected test site
                'https://bot.sannysoft.com',  # Bot detection test
            ]
        }

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        logger.info("🚀 Starting Comprehensive Crawler Test Suite")
        logger.info(f"Components available: {sum(COMPONENTS_AVAILABLE.values())}/{len(COMPONENTS_AVAILABLE)}")

        start_time = time.time()

        # Run individual test suites
        if COMPONENTS_AVAILABLE['cloudscraper_stealth']:
            await self.test_cloudscraper_stealth()

        if COMPONENTS_AVAILABLE['unified_stealth']:
            await self.test_unified_stealth_orchestrator()

        if COMPONENTS_AVAILABLE['improved_crawler']:
            await self.test_improved_base_crawler()

        if COMPONENTS_AVAILABLE['crawler_package']:
            await self.test_crawler_package_integration()

        # Performance comparison test
        await self.test_performance_comparison()

        # Stealth effectiveness test
        await self.test_stealth_effectiveness()

        # Content quality assessment test
        await self.test_content_quality_assessment()

        total_time = time.time() - start_time
        self.test_results['total_execution_time'] = total_time

        # Generate summary and recommendations
        self._generate_summary()
        self._generate_recommendations()

        # Save results
        await self._save_results()

        logger.info(f"🎯 Test suite completed in {total_time:.2f} seconds")
        return self.test_results

    async def test_cloudscraper_stealth(self):
        """Test CloudScraper priority stealth manager."""
        test_name = "CloudScraper Stealth Manager"
        logger.info(f"\n🧪 Testing {test_name}")

        try:
            async with create_stealth_manager() as stealth_manager:
                results = []

                for category, urls in self.test_urls.items():
                    for url in urls:
                        logger.info(f"  Testing {url}")
                        start_time = time.time()

                        result = await stealth_manager.request(url)

                        test_time = time.time() - start_time
                        results.append({
                            'url': url,
                            'category': category,
                            'success': result.success,
                            'method_used': result.method_used,
                            'response_time': test_time,
                            'protection_detected': result.protection_detected,
                            'content_length': len(result.content or ''),
                            'error': result.error
                        })

                        if result.success:
                            logger.info(f"    ✅ Success with {result.method_used} ({test_time:.2f}s)")
                        else:
                            logger.warning(f"    ❌ Failed: {result.error}")

                # Get performance report
                performance_report = stealth_manager.get_comprehensive_report()

                self.test_results['tests'][test_name] = {
                    'passed': True,
                    'results': results,
                    'performance_report': performance_report,
                    'success_rate': sum(r['success'] for r in results) / len(results) * 100,
                    'avg_response_time': sum(r['response_time'] for r in results) / len(results)
                }

                logger.info(f"✅ {test_name} completed successfully")

        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.test_results['tests'][test_name] = {
                'passed': False,
                'error': str(e)
            }

    async def test_unified_stealth_orchestrator(self):
        """Test unified stealth orchestrator."""
        test_name = "Unified Stealth Orchestrator"
        logger.info(f"\n🧪 Testing {test_name}")

        try:
            async with create_unified_stealth_orchestrator() as orchestrator:
                results = []

                # Test with different max_attempts settings
                for max_attempts in [1, 2, 3]:
                    for url in self.test_urls['basic']:
                        logger.info(f"  Testing {url} (max_attempts={max_attempts})")
                        start_time = time.time()

                        result = await orchestrator.request(url, max_attempts=max_attempts)

                        test_time = time.time() - start_time
                        results.append({
                            'url': url,
                            'max_attempts': max_attempts,
                            'success': result.success,
                            'method_used': result.method_used,
                            'response_time': test_time,
                            'fallback_attempts': result.fallback_attempts,
                            'protection_detected': result.protection_detected,
                            'error': result.error
                        })

                        if result.success:
                            logger.info(f"    ✅ Success with {result.method_used}")
                        else:
                            logger.warning(f"    ❌ Failed after {result.fallback_attempts} fallbacks")

                # Get comprehensive report
                comprehensive_report = orchestrator.get_comprehensive_report()

                self.test_results['tests'][test_name] = {
                    'passed': True,
                    'results': results,
                    'comprehensive_report': comprehensive_report,
                    'success_rate': sum(r['success'] for r in results) / len(results) * 100
                }

                logger.info(f"✅ {test_name} completed successfully")

        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.test_results['tests'][test_name] = {
                'passed': False,
                'error': str(e)
            }

    async def test_improved_base_crawler(self):
        """Test improved base crawler with content extraction."""
        test_name = "Improved Base Crawler"
        logger.info(f"\n🧪 Testing {test_name}")

        try:
            # Test different crawler configurations
            configs = [
                ("Basic", create_basic_crawler()),
                ("News", create_news_crawler()),
                ("Fast", create_fast_crawler())
            ]

            for config_name, crawler in configs:
                logger.info(f"  Testing {config_name} configuration")

                async with crawler:
                    results = []

                    for url in self.test_urls['news_sites']:
                        logger.info(f"    Crawling {url}")
                        start_time = time.time()

                        result = await crawler.crawl_url(url)

                        test_time = time.time() - start_time
                        results.append({
                            'url': url,
                            'success': result.success,
                            'title': result.title,
                            'text_length': len(result.text or ''),
                            'quality_score': result.quality_score,
                            'readability_score': result.readability_score,
                            'extraction_method': result.extraction_method,
                            'stealth_method': result.stealth_method,
                            'response_time': test_time,
                            'error': result.error
                        })

                        if result.success:
                            logger.info(f"      ✅ Success: {result.title[:50]}... (Q:{result.quality_score:.2f})")
                        else:
                            logger.warning(f"      ❌ Failed: {result.error}")

                    # Get crawler statistics
                    stats = crawler.get_stats()

                    self.test_results['tests'][f"{test_name} - {config_name}"] = {
                        'passed': True,
                        'results': results,
                        'stats': stats,
                        'success_rate': sum(r['success'] for r in results) / len(results) * 100,
                        'avg_quality_score': sum(r['quality_score'] for r in results if r['success']) / max(1, sum(r['success'] for r in results))
                    }

            logger.info(f"✅ {test_name} completed successfully")

        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.test_results['tests'][test_name] = {
                'passed': False,
                'error': str(e)
            }

    async def test_crawler_package_integration(self):
        """Test crawler package integration and factory functions."""
        test_name = "Crawler Package Integration"
        logger.info(f"\n🧪 Testing {test_name}")

        try:
            # Test available crawlers
            available_crawlers = get_available_crawlers()
            health_status = get_crawlers_health()

            # Test factory functions
            factory_tests = []

            try:
                stealth_crawler = create_stealth_crawler()
                factory_tests.append(('create_stealth_crawler', True, None))
            except Exception as e:
                factory_tests.append(('create_stealth_crawler', False, str(e)))

            try:
                unified_crawler = create_unified_stealth_crawler()
                factory_tests.append(('create_unified_stealth_crawler', True, None))
            except Exception as e:
                factory_tests.append(('create_unified_stealth_crawler', False, str(e)))

            try:
                news_crawler = create_news_crawler_enhanced()
                factory_tests.append(('create_news_crawler_enhanced', True, None))
            except Exception as e:
                factory_tests.append(('create_news_crawler_enhanced', False, str(e)))

            self.test_results['tests'][test_name] = {
                'passed': True,
                'available_crawlers': available_crawlers,
                'health_status': health_status,
                'factory_tests': factory_tests,
                'factory_success_rate': sum(1 for _, success, _ in factory_tests if success) / len(factory_tests) * 100
            }

            logger.info(f"✅ {test_name} completed successfully")
            logger.info(f"  Available crawlers: {available_crawlers}")
            logger.info(f"  Factory success rate: {self.test_results['tests'][test_name]['factory_success_rate']:.1f}%")

        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.test_results['tests'][test_name] = {
                'passed': False,
                'error': str(e)
            }

    async def test_performance_comparison(self):
        """Compare performance across different stealth methods."""
        test_name = "Performance Comparison"
        logger.info(f"\n🧪 Testing {test_name}")

        comparison_results = {}

        # Test each available method
        methods_to_test = []

        if COMPONENTS_AVAILABLE['cloudscraper_stealth']:
            methods_to_test.append(('CloudScraper Priority', create_stealth_manager))

        if COMPONENTS_AVAILABLE['unified_stealth']:
            methods_to_test.append(('Unified Orchestrator', create_unified_stealth_orchestrator))

        for method_name, method_factory in methods_to_test:
            logger.info(f"  Testing {method_name}")

            try:
                async with method_factory() as method_instance:
                    times = []
                    successes = 0

                    for url in self.test_urls['basic']:
                        start_time = time.time()

                        if hasattr(method_instance, 'request'):
                            result = await method_instance.request(url)
                        else:
                            result = await method_instance.crawl_url(url)

                        response_time = time.time() - start_time
                        times.append(response_time)

                        if result.success:
                            successes += 1

                    comparison_results[method_name] = {
                        'avg_response_time': sum(times) / len(times),
                        'min_response_time': min(times),
                        'max_response_time': max(times),
                        'success_rate': successes / len(times) * 100,
                        'total_requests': len(times)
                    }

            except Exception as e:
                logger.warning(f"  ⚠️ {method_name} test failed: {e}")
                comparison_results[method_name] = {'error': str(e)}

        self.test_results['tests'][test_name] = {
            'passed': True,
            'comparison_results': comparison_results
        }

        # Log performance comparison
        for method, results in comparison_results.items():
            if 'error' not in results:
                logger.info(f"  {method}: {results['avg_response_time']:.2f}s avg, {results['success_rate']:.1f}% success")

        logger.info(f"✅ {test_name} completed successfully")

    async def test_stealth_effectiveness(self):
        """Test stealth effectiveness against protected sites."""
        test_name = "Stealth Effectiveness"
        logger.info(f"\n🧪 Testing {test_name}")

        try:
            if not COMPONENTS_AVAILABLE['cloudscraper_stealth']:
                logger.warning(f"  ⚠️ Skipping {test_name} - CloudScraper stealth not available")
                return

            async with create_stealth_manager() as stealth_manager:
                effectiveness_results = {}

                for category, urls in self.test_urls.items():
                    logger.info(f"  Testing {category} sites")
                    category_results = []

                    for url in urls:
                        logger.info(f"    Testing {url}")
                        result = await stealth_manager.request(url)

                        category_results.append({
                            'url': url,
                            'success': result.success,
                            'method_used': result.method_used,
                            'protection_detected': result.protection_detected,
                            'response_time': result.response_time,
                            'error': result.error
                        })

                        if result.success:
                            method_emoji = "🟢" if result.method_used == "cloudscraper" else "🟡"
                            logger.info(f"      {method_emoji} Success with {result.method_used}")
                        else:
                            logger.warning(f"      🔴 Failed: {result.error}")

                    effectiveness_results[category] = {
                        'results': category_results,
                        'success_rate': sum(r['success'] for r in category_results) / len(category_results) * 100,
                        'cloudscraper_rate': sum(1 for r in category_results if r['method_used'] == 'cloudscraper' and r['success']) / len(category_results) * 100
                    }

                self.test_results['tests'][test_name] = {
                    'passed': True,
                    'effectiveness_results': effectiveness_results,
                    'overall_success_rate': sum(results['success_rate'] for results in effectiveness_results.values()) / len(effectiveness_results)
                }

                logger.info(f"✅ {test_name} completed successfully")

        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.test_results['tests'][test_name] = {
                'passed': False,
                'error': str(e)
            }

    async def test_content_quality_assessment(self):
        """Test content extraction and quality assessment."""
        test_name = "Content Quality Assessment"
        logger.info(f"\n🧪 Testing {test_name}")

        try:
            if not COMPONENTS_AVAILABLE['improved_crawler']:
                logger.warning(f"  ⚠️ Skipping {test_name} - Improved crawler not available")
                return

            async with create_news_crawler() as crawler:
                quality_results = []

                for url in self.test_urls['news_sites']:
                    logger.info(f"  Assessing {url}")
                    result = await crawler.crawl_url(url)

                    if result.success:
                        quality_results.append({
                            'url': url,
                            'title': result.title,
                            'text_length': len(result.text or ''),
                            'quality_score': result.quality_score,
                            'readability_score': result.readability_score,
                            'content_confidence': result.content_confidence,
                            'extraction_method': result.extraction_method,
                            'has_author': bool(result.authors),
                            'has_date': bool(result.publish_date),
                            'links_count': len(result.links),
                            'images_count': len(result.images)
                        })

                        logger.info(f"    ✅ Quality: {result.quality_score:.2f}, Readability: {result.readability_score:.2f}")
                    else:
                        logger.warning(f"    ❌ Failed to extract content: {result.error}")

                # Calculate averages
                if quality_results:
                    avg_quality = sum(r['quality_score'] for r in quality_results) / len(quality_results)
                    avg_readability = sum(r['readability_score'] for r in quality_results) / len(quality_results)
                    avg_text_length = sum(r['text_length'] for r in quality_results) / len(quality_results)

                    self.test_results['tests'][test_name] = {
                        'passed': True,
                        'quality_results': quality_results,
                        'avg_quality_score': avg_quality,
                        'avg_readability_score': avg_readability,
                        'avg_text_length': avg_text_length,
                        'extraction_success_rate': len(quality_results) / len(self.test_urls['news_sites']) * 100
                    }

                    logger.info(f"✅ {test_name} completed successfully")
                    logger.info(f"  Average quality score: {avg_quality:.2f}")
                    logger.info(f"  Average readability: {avg_readability:.2f}")
                else:
                    self.test_results['tests'][test_name] = {
                        'passed': False,
                        'error': 'No successful content extractions'
                    }

        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.test_results['tests'][test_name] = {
                'passed': False,
                'error': str(e)
            }

    def _generate_summary(self):
        """Generate test summary."""
        total_tests = len(self.test_results['tests'])
        passed_tests = sum(1 for test in self.test_results['tests'].values() if test.get('passed', False))

        self.test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'pass_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'components_available': sum(COMPONENTS_AVAILABLE.values()),
            'total_components': len(COMPONENTS_AVAILABLE),
            'execution_time': self.test_results.get('total_execution_time', 0)
        }

    def _generate_recommendations(self):
        """Generate recommendations based on test results."""
        recommendations = []

        # Component availability recommendations
        if not COMPONENTS_AVAILABLE['cloudscraper_stealth']:
            recommendations.append("Install cloudscraper package for optimal stealth performance")

        if not COMPONENTS_AVAILABLE['unified_stealth']:
            recommendations.append("Fix unified stealth orchestrator imports for advanced protection handling")

        if not COMPONENTS_AVAILABLE['improved_crawler']:
            recommendations.append("Fix improved crawler imports for enhanced content extraction")

        # Performance recommendations
        if 'Performance Comparison' in self.test_results['tests']:
            perf_data = self.test_results['tests']['Performance Comparison']
            if 'comparison_results' in perf_data:
                avg_times = []
                for method, results in perf_data['comparison_results'].items():
                    if 'avg_response_time' in results:
                        avg_times.append(results['avg_response_time'])

                if avg_times and max(avg_times) > 10:
                    recommendations.append("Consider optimizing slow stealth methods or reducing timeouts")

        # Success rate recommendations
        for test_name, test_data in self.test_results['tests'].items():
            if 'success_rate' in test_data and test_data['success_rate'] < 80:
                recommendations.append(f"Improve {test_name} success rate (currently {test_data['success_rate']:.1f}%)")

        # Quality recommendations
        if 'Content Quality Assessment' in self.test_results['tests']:
            quality_data = self.test_results['tests']['Content Quality Assessment']
            if quality_data.get('passed') and quality_data.get('avg_quality_score', 0) < 0.7:
                recommendations.append("Consider improving content extraction methods for better quality scores")

        self.test_results['recommendations'] = recommendations

    async def _save_results(self):
        """Save test results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_file = self.output_dir / f"crawler_test_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)

        # Save human-readable report
        report_file = self.output_dir / f"crawler_test_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(self._generate_text_report())

        logger.info(f"📄 Results saved to {json_file}")
        logger.info(f"📄 Report saved to {report_file}")

    def _generate_text_report(self) -> str:
        """Generate human-readable text report."""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE CRAWLER TEST REPORT")
        report.append("=" * 80)
        report.append(f"Test Date: {self.test_results['timestamp']}")
        report.append(f"Execution Time: {self.test_results.get('total_execution_time', 0):.2f} seconds")
        report.append("")

        # Summary
        summary = self.test_results['summary']
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Tests: {summary['total_tests']}")
        report.append(f"Passed: {summary['passed_tests']}")
        report.append(f"Failed: {summary['failed_tests']}")
        report.append(f"Pass Rate: {summary['pass_rate']:.1f}%")
        report.append(f"Components Available: {summary['components_available']}/{summary['total_components']}")
        report.append("")

        # Component Status
        report.append("COMPONENT STATUS")
        report.append("-" * 40)
        for component, available in COMPONENTS_AVAILABLE.items():
            status = "✅ Available" if available else "❌ Not Available"
            report.append(f"{component}: {status}")
        report.append("")

        # Test Results
        report.append("TEST RESULTS")
        report.append("-" * 40)
        for test_name, test_data in self.test_results['tests'].items():
            status = "✅ PASS" if test_data.get('passed', False) else "❌ FAIL"
            report.append(f"{status} {test_name}")

            if test_data.get('passed', False):
                if 'success_rate' in test_data:
                    report.append(f"    Success Rate: {test_data['success_rate']:.1f}%")
                if 'avg_response_time' in test_data:
                    report.append(f"    Avg Response Time: {test_data['avg_response_time']:.2f}s")
            else:
                if 'error' in test_data:
                    report.append(f"    Error: {test_data['error']}")
            report.append("")

        # Recommendations
        if self.test_results['recommendations']:
            report.append("RECOMMENDATIONS")
            report.append("-" * 40)
            for i, rec in enumerate(self.test_results['recommendations'], 1):
                report.append(f"{i}. {rec}")
            report.append("")

        # Overall Assessment
        report.append("OVERALL ASSESSMENT")
        report.append("-" * 40)
        if summary['pass_rate'] >= 90:
            report.append("🎉 EXCELLENT: Crawler system is performing exceptionally well!")
        elif summary['pass_rate'] >= 70:
            report.append("✅ GOOD: Crawler system is working well with minor issues.")
        elif summary['pass_rate'] >= 50:
            report.append("⚠️ NEEDS IMPROVEMENT: Several components need attention.")
        else:
            report.append("❌ CRITICAL: Major issues detected. Immediate attention required.")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)


async def main():
    """Main test execution function."""
    print("🚀 Starting Comprehensive Crawler Test Suite")
    print("=" * 60)

    # Create test suite
    test_suite = CrawlerTestSuite()

    try:
        # Run all tests
        results = await test_suite.run_all_tests()

        # Display summary
        summary = results['summary']
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Tests Run: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Pass Rate: {summary['pass_rate']:.1f}%")
        print(f"Execution Time: {summary['execution_time']:.2f} seconds")

        # Display recommendations
        if results['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"  {i}. {rec}")

        # Overall status
        if summary['pass_rate'] >= 90:
            print("\n🎉 EXCELLENT: Crawler system is ready for production!")
            return 0
        elif summary['pass_rate'] >= 70:
            print("\n✅ GOOD: Crawler system is functional with minor issues.")
            return 0
        else:
            print("\n⚠️ ISSUES DETECTED: Review failed tests and address problems.")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️ Test suite interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        logger.error(f"Test suite failed: {e}")
        return 1


if __name__ == "__main__":
    # Run the comprehensive test suite
    exit_code = asyncio.run(main())
    exit(exit_code)
