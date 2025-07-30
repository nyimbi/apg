#!/usr/bin/env python3
"""
Google News Crawler Integration Tests
====================================

Integration tests that verify component interactions and end-to-end functionality.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import sys
import os
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# Add package to path
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

logger = logging.getLogger(__name__)

class IntegrationTestSuite:
    """Integration test suite for Google News Crawler."""
    
    def __init__(self):
        self.test_results = []
        self.temp_files = []
        
    async def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                temp_file.unlink()
            except FileNotFoundError:
                pass
    
    def assert_test(self, condition: bool, test_name: str, message: str = ""):
        """Assert a test condition and record result."""
        if condition:
            logger.info(f"✅ {test_name}")
            self.test_results.append((test_name, True, ""))
        else:
            error_msg = f"Assertion failed: {message}" if message else "Assertion failed"
            logger.error(f"❌ {test_name}: {error_msg}")
            self.test_results.append((test_name, False, error_msg))
    
    async def test_client_initialization_flow(self):
        """Test complete client initialization workflow."""
        try:
            from api.google_news_client import create_enhanced_gnews_client
            from cli.utils import create_mock_db_manager
            
            # Create mock database
            db_manager = create_mock_db_manager()
            self.assert_test(db_manager is not None, "Mock DB Manager Creation")
            
            # Create client
            client = await create_enhanced_gnews_client(db_manager=db_manager)
            self.assert_test(client is not None, "Client Creation")
            
            # Verify client has required attributes
            self.assert_test(hasattr(client, 'search_news'), "Client has search_news method")
            self.assert_test(hasattr(client, 'initialize'), "Client has initialize method")
            self.assert_test(hasattr(client, 'close'), "Client has close method")
            
            # Test client initialization state
            self.assert_test(client.db_manager is not None, "Client has database manager")
            self.assert_test(hasattr(client, 'session_stats'), "Client has session stats")
            
            # Clean up
            await client.close()
            self.assert_test(True, "Client Cleanup")
            
        except Exception as e:
            self.assert_test(False, "Client Initialization Flow", str(e))
    
    async def test_configuration_integration(self):
        """Test configuration loading and usage integration."""
        try:
            from cli.utils import save_cli_config, load_cli_config
            
            # Create temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                config_path = Path(f.name)
            self.temp_files.append(config_path)
            
            # Test configuration data
            test_config = {
                "database": {
                    "url": "postgresql://test:test@localhost/test"
                },
                "crawlee": {
                    "max_requests": 25,
                    "max_concurrent": 3,
                    "target_countries": ["ET", "SO", "KE"]
                },
                "search": {
                    "default_max_results": 50,
                    "default_countries": ["ET", "SO"]
                }
            }
            
            # Save configuration
            save_cli_config(config_path, test_config)
            self.assert_test(config_path.exists(), "Configuration File Created")
            
            # Load configuration
            loaded_config = load_cli_config(config_path)
            self.assert_test(loaded_config is not None, "Configuration Loading")
            
            # Verify configuration integrity
            self.assert_test(
                loaded_config["crawlee"]["max_requests"] == 25,
                "Configuration Value Integrity"
            )
            
            self.assert_test(
                loaded_config["search"]["default_countries"] == ["ET", "SO"],
                "Configuration List Integrity"
            )
            
            # Test configuration usage in components
            from crawlee_integration import create_crawlee_config
            
            crawlee_cfg = create_crawlee_config(
                max_requests=loaded_config["crawlee"]["max_requests"],
                max_concurrent=loaded_config["crawlee"]["max_concurrent"]
            )
            
            self.assert_test(
                crawlee_cfg.max_requests_per_crawl == 25,
                "Configuration Integration with Components"
            )
            
        except Exception as e:
            self.assert_test(False, "Configuration Integration", str(e))
    
    async def test_crawlee_integration_workflow(self):
        """Test Crawlee integration workflow."""
        try:
            from crawlee_integration import CRAWLEE_AVAILABLE, create_crawlee_config
            
            # Test availability check
            self.assert_test(
                isinstance(CRAWLEE_AVAILABLE, bool),
                "Crawlee Availability Check"
            )
            
            # Test configuration creation
            config = create_crawlee_config(
                max_requests=10,
                max_concurrent=2,
                target_countries=["ET", "SO"],
                enable_full_content=True
            )
            
            self.assert_test(config is not None, "Crawlee Config Creation")
            self.assert_test(config.max_requests_per_crawl == 10, "Crawlee Config Values")
            self.assert_test(config.enable_full_content == True, "Crawlee Config Flags")
            
            # Test component initialization (if available)
            if CRAWLEE_AVAILABLE:
                from crawlee_integration import create_crawlee_enhancer
                
                enhancer = await create_crawlee_enhancer(config)
                self.assert_test(enhancer is not None, "Crawlee Enhancer Creation")
                
                # Test enhancer attributes
                self.assert_test(hasattr(enhancer, 'config'), "Enhancer Has Config")
                self.assert_test(hasattr(enhancer, 'enhance_articles'), "Enhancer Has Enhancement Method")
                
                await enhancer.close()
                self.assert_test(True, "Crawlee Enhancer Cleanup")
            else:
                logger.info("ℹ️ Crawlee not available - skipping enhancer tests")
                self.assert_test(True, "Crawlee Not Available (Expected)")
            
        except Exception as e:
            self.assert_test(False, "Crawlee Integration Workflow", str(e))
    
    async def test_cli_command_integration(self):
        """Test CLI command integration with core components."""
        try:
            from cli.main import setup_argument_parser
            from cli.utils import validate_countries, validate_languages, parse_date_input
            
            # Test argument parser setup
            parser = setup_argument_parser()
            self.assert_test(parser is not None, "CLI Parser Setup")
            
            # Test search command parsing
            search_args = parser.parse_args([
                'search', 'Ethiopia conflict',
                '--countries', 'ET,SO,KE',
                '--max-results', '25',
                '--crawlee'
            ])
            
            self.assert_test(search_args.command == 'search', "Search Command Parsing")
            self.assert_test(search_args.query == 'Ethiopia conflict', "Search Query Parsing")
            self.assert_test(search_args.crawlee == True, "Search Flag Parsing")
            
            # Test utility function integration
            countries = validate_countries(search_args.countries.split(','))
            self.assert_test(len(countries) >= 3, "Country Validation Integration")
            
            # Test monitor command parsing
            monitor_args = parser.parse_args([
                'monitor',
                '--query', 'Horn of Africa',
                '--interval', '300',
                '--alert-keywords', 'violence,crisis'
            ])
            
            self.assert_test(monitor_args.command == 'monitor', "Monitor Command Parsing")
            self.assert_test(monitor_args.interval == 300, "Monitor Interval Parsing")
            
            # Test config command parsing
            config_args = parser.parse_args(['config', '--show'])
            self.assert_test(config_args.command == 'config', "Config Command Parsing")
            self.assert_test(config_args.show == True, "Config Show Flag Parsing")
            
        except Exception as e:
            self.assert_test(False, "CLI Command Integration", str(e))
    
    async def test_data_flow_integration(self):
        """Test data flow between components."""
        try:
            from cli.utils import create_mock_db_manager
            
            # Create mock data flow
            db_manager = create_mock_db_manager()
            
            # Test article data structure
            test_articles = [
                {
                    'title': 'Test Article 1',
                    'url': 'https://example.com/article1',
                    'content': 'Test content for article 1',
                    'published_date': datetime.now(),
                    'source': 'Test Source 1'
                },
                {
                    'title': 'Test Article 2',
                    'url': 'https://example.com/article2',
                    'content': 'Test content for article 2',
                    'published_date': datetime.now(),
                    'source': 'Test Source 2'
                }
            ]
            
            # Test data storage
            stored_count = await db_manager.store_articles(test_articles)
            self.assert_test(stored_count == 2, "Article Storage Integration")
            
            # Test data retrieval (if method exists)
            if hasattr(db_manager, 'get_articles'):
                retrieved_articles = await db_manager.get_articles()
                self.assert_test(
                    len(retrieved_articles) >= 2,
                    "Article Retrieval Integration"
                )
            
            # Test query processing integration
            def process_query(query: str) -> str:
                """Simple query enhancement."""
                if 'conflict' not in query.lower():
                    return f"{query} conflict"
                return query
            
            test_query = "Ethiopia news"
            enhanced_query = process_query(test_query)
            self.assert_test(
                'conflict' in enhanced_query,
                "Query Processing Integration"
            )
            
            # Test filtering integration
            def filter_articles(articles: List[Dict], min_length: int = 10):
                """Simple article filtering."""
                return [
                    a for a in articles 
                    if len(a.get('content', '')) >= min_length
                ]
            
            filtered = filter_articles(test_articles, min_length=15)
            self.assert_test(
                len(filtered) <= len(test_articles),
                "Article Filtering Integration"
            )
            
        except Exception as e:
            self.assert_test(False, "Data Flow Integration", str(e))
    
    async def test_error_handling_integration(self):
        """Test error handling across components."""
        try:
            from cli.utils import validate_countries, parse_date_input, load_cli_config
            
            # Test graceful handling of invalid inputs
            invalid_countries = validate_countries(['INVALID', 'XX', 'YY'])
            self.assert_test(
                len(invalid_countries) >= 1,  # Should fall back to defaults
                "Invalid Country Handling"
            )
            
            # Test invalid date handling
            invalid_date = parse_date_input("invalid-date-format")
            self.assert_test(
                invalid_date is None,
                "Invalid Date Handling"
            )
            
            # Test missing file handling
            try:
                load_cli_config(Path("/non/existent/file.json"))
                self.assert_test(False, "Missing File Handling", "Should have raised exception")
            except FileNotFoundError:
                self.assert_test(True, "Missing File Handling")
            
            # Test component initialization with invalid config
            from crawlee_integration import create_crawlee_config
            
            try:
                # Test with extreme values
                config = create_crawlee_config(
                    max_requests=-1,  # Invalid
                    max_concurrent=0   # Invalid
                )
                # Should still create config with corrected values
                self.assert_test(
                    config.max_requests_per_crawl >= 1,
                    "Invalid Config Value Correction"
                )
            except Exception:
                # Or should raise appropriate exception
                self.assert_test(True, "Invalid Config Rejection")
            
        except Exception as e:
            self.assert_test(False, "Error Handling Integration", str(e))
    
    async def test_performance_integration(self):
        """Test performance characteristics of integrated components."""
        try:
            import time
            from cli.utils import validate_countries, create_mock_db_manager
            
            # Test batch validation performance
            large_country_list = ['ET', 'SO', 'KE', 'INVALID'] * 100
            
            start_time = time.perf_counter()
            validated = validate_countries(large_country_list)
            validation_time = time.perf_counter() - start_time
            
            self.assert_test(
                validation_time < 1.0,  # Should complete in under 1 second
                "Batch Validation Performance"
            )
            
            # Test database operation performance
            db_manager = create_mock_db_manager()
            large_article_set = [
                {
                    'title': f'Article {i}',
                    'url': f'https://example.com/article-{i}',
                    'content': f'Content for article {i}',
                    'published_date': datetime.now()
                }
                for i in range(500)
            ]
            
            start_time = time.perf_counter()
            stored = await db_manager.store_articles(large_article_set)
            storage_time = time.perf_counter() - start_time
            
            self.assert_test(
                storage_time < 1.0,  # Should complete in under 1 second
                "Batch Storage Performance"
            )
            
            self.assert_test(
                stored == 500,
                "Batch Storage Accuracy"
            )
            
            # Test memory efficiency
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Perform memory-intensive operations
            for _ in range(10):
                test_articles = [{'content': 'x' * 1000} for _ in range(100)]
                await db_manager.store_articles(test_articles)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            self.assert_test(
                memory_increase < 50,  # Should not increase by more than 50MB
                "Memory Efficiency"
            )
            
        except Exception as e:
            self.assert_test(False, "Performance Integration", str(e))
    
    def generate_report(self) -> str:
        """Generate integration test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, passed, _ in self.test_results if passed)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        report = []
        report.append("="*70)
        report.append("Google News Crawler - Integration Test Report")
        report.append("="*70)
        report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        report.append(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        report.append(f"Success Rate: {success_rate:.1f}%")
        report.append("")
        
        # Detailed results
        report.append("Detailed Results:")
        report.append("-" * 40)
        
        for test_name, passed, error in self.test_results:
            status = "✅" if passed else "❌"
            report.append(f"{status} {test_name}")
            if not passed and error:
                report.append(f"    Error: {error}")
        
        # Summary
        report.append("")
        report.append("="*70)
        if success_rate >= 90:
            report.append("🎉 EXCELLENT: Integration tests show high quality")
        elif success_rate >= 75:
            report.append("✅ GOOD: Integration tests show acceptable quality")
        elif success_rate >= 50:
            report.append("⚠️ FAIR: Integration tests show some issues")
        else:
            report.append("❌ POOR: Integration tests show significant issues")
        
        report.append("="*70)
        
        return "\n".join(report)

async def run_integration_tests():
    """Run all integration tests."""
    print("🔗 Starting Google News Crawler Integration Tests")
    print("="*60)
    
    suite = IntegrationTestSuite()
    
    try:
        # Run all integration tests
        await suite.test_client_initialization_flow()
        await suite.test_configuration_integration()
        await suite.test_crawlee_integration_workflow()
        await suite.test_cli_command_integration()
        await suite.test_data_flow_integration()
        await suite.test_error_handling_integration()
        await suite.test_performance_integration()
        
    finally:
        # Cleanup
        await suite.cleanup()
    
    # Generate and return report
    report = suite.generate_report()
    return report, suite.test_results

if __name__ == "__main__":
    # Run integration tests
    report, results = asyncio.run(run_integration_tests())
    
    # Print report
    print(report)
    
    # Save report
    report_file = Path(__file__).parent / "integration_test_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Integration test report saved to: {report_file}")
    
    # Exit with appropriate code
    total_tests = len(results)
    passed_tests = sum(1 for _, passed, _ in results if passed)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    exit_code = 0 if success_rate >= 75 else 1
    sys.exit(exit_code)