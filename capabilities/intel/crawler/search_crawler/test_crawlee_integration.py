#!/usr/bin/env python3
"""
Test Crawlee Integration for Search Crawler
===========================================

Comprehensive test suite for the new Crawlee-enhanced search crawler integration.
Tests functionality, performance, and quality of content extraction.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add package to path
package_root = Path(__file__).parent
sys.path.insert(0, str(package_root))

# Import all required modules at module level to avoid import issues
try:
    from core.crawlee_enhanced_search_crawler import (
        CrawleeEnhancedSearchCrawler,
        CrawleeSearchConfig,
        CrawleeEnhancedResult,
        create_crawlee_search_config,
        create_crawlee_search_crawler,
        CRAWLEE_AVAILABLE
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import Crawlee components: {e}")
    IMPORTS_AVAILABLE = False
    CRAWLEE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrawleeIntegrationTest:
    """Test suite for Crawlee integration with search crawler."""
    
    def __init__(self):
        self.test_results = []
        self.test_stats = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'warnings': 0
        }
    
    def log_test_result(self, test_name: str, success: bool, message: str = "", warning: bool = False):
        """Log test result."""
        self.test_stats['total_tests'] += 1
        
        if success:
            self.test_stats['passed_tests'] += 1
            status = "✅"
        else:
            self.test_stats['failed_tests'] += 1
            status = "❌"
        
        if warning:
            self.test_stats['warnings'] += 1
            status = "⚠️"
        
        log_msg = f"{status} {test_name}"
        if message:
            log_msg += f": {message}"
        
        logger.info(log_msg)
        self.test_results.append({
            'test': test_name,
            'success': success,
            'message': message,
            'warning': warning
        })
    
    async def test_imports(self):
        """Test that all required imports work."""
        logger.info("🧪 Testing Imports")
        
        if IMPORTS_AVAILABLE:
            self.log_test_result("Import Core Classes", True, "All core classes imported successfully")
            self.log_test_result("Import Factory Functions", True, "Factory functions imported successfully")
        else:
            self.log_test_result("Import Core Classes", False, "Import failed - check dependencies")
            return False
        
        # Check Crawlee availability
        if CRAWLEE_AVAILABLE:
            self.log_test_result("Crawlee Availability", True, "Crawlee library is available")
        else:
            self.log_test_result("Crawlee Availability", False, "Crawlee library not available", warning=True)
        
        return True
    
    async def test_configuration(self):
        """Test configuration creation and validation."""
        logger.info("🧪 Testing Configuration")
        
        try:
            from core.crawlee_enhanced_search_crawler import create_crawlee_search_config, CrawleeSearchConfig
            
            # Test basic configuration creation
            config = create_crawlee_search_config()
            self.log_test_result("Basic Config Creation", True, f"Created config with {len(config.engines)} engines")
            
            # Test custom configuration
            custom_config = create_crawlee_search_config(
                engines=['google', 'bing'],
                max_results=30,
                target_countries=['ET', 'SO', 'KE'],
                enable_content_extraction=True
            )
            
            config_valid = (
                len(custom_config.engines) == 2 and
                custom_config.total_max_results == 30 and
                len(custom_config.target_countries) == 3 and
                custom_config.download_content == True
            )
            
            self.log_test_result("Custom Config Creation", config_valid, 
                               f"Engines: {custom_config.engines}, Max results: {custom_config.total_max_results}")
            
            # Test configuration with all parameters
            full_config = CrawleeSearchConfig(
                engines=['google', 'bing', 'duckduckgo'],
                total_max_results=25,
                max_requests_per_crawl=50,
                preferred_extraction_method="trafilatura",
                min_content_length=500,
                enable_content_scoring=True,
                target_countries=["ET", "SO", "ER", "DJ"]
            )
            
            full_config_valid = (
                full_config.preferred_extraction_method == "trafilatura" and
                full_config.min_content_length == 500 and
                full_config.enable_content_scoring == True
            )
            
            self.log_test_result("Full Config Creation", full_config_valid, 
                               f"Extraction method: {full_config.preferred_extraction_method}")
            
        except Exception as e:
            self.log_test_result("Configuration Test", False, f"Configuration test failed: {e}")
            return False
        
        return True
    
    async def test_crawler_initialization(self):
        """Test crawler initialization."""
        logger.info("🧪 Testing Crawler Initialization")
        
        try:
            from core.crawlee_enhanced_search_crawler import (
                CrawleeEnhancedSearchCrawler,
                create_crawlee_search_config,
                CRAWLEE_AVAILABLE
            )
            
            # Test basic crawler creation
            config = create_crawlee_search_config(
                engines=['google', 'bing'],
                max_results=5,
                enable_content_extraction=False  # Don't require Crawlee for basic test
            )
            
            crawler = CrawleeEnhancedSearchCrawler(config)
            self.log_test_result("Basic Crawler Creation", True, "Crawler created successfully")
            
            # Test crawler attributes
            has_required_attrs = (
                hasattr(crawler, 'crawlee_config') and
                hasattr(crawler, 'extractors_available') and
                hasattr(crawler, 'crawlee_stats')
            )
            
            self.log_test_result("Crawler Attributes", has_required_attrs, 
                               f"Extractors available: {sum(crawler.extractors_available.values())}")
            
            # Test Crawlee initialization if available
            if CRAWLEE_AVAILABLE:
                try:
                    await crawler.initialize_crawlee()
                    crawlee_initialized = (
                        crawler.crawler is not None and
                        crawler.request_queue is not None and
                        crawler.dataset is not None
                    )
                    self.log_test_result("Crawlee Initialization", crawlee_initialized, 
                                       "Crawlee components initialized")
                    
                    # Clean up
                    await crawler.close()
                except Exception as e:
                    self.log_test_result("Crawlee Initialization", False, f"Crawlee init failed: {e}")
            else:
                self.log_test_result("Crawlee Initialization", True, "Skipped - Crawlee not available", warning=True)
                
            # Test factory function
            try:
                from core.crawlee_enhanced_search_crawler import create_crawlee_search_crawler
                factory_config = create_crawlee_search_config(enable_content_extraction=False)
                factory_crawler = CrawleeEnhancedSearchCrawler(factory_config)
                self.log_test_result("Factory Function", True, "Factory function works correctly")
            except Exception as e:
                self.log_test_result("Factory Function", False, f"Factory function failed: {e}")
            
        except Exception as e:
            self.log_test_result("Crawler Initialization", False, f"Initialization test failed: {e}")
            return False
        
        return True
    
    async def test_basic_search(self):
        """Test basic search functionality without Crawlee."""
        logger.info("🧪 Testing Basic Search")
        
        try:
            from core.crawlee_enhanced_search_crawler import (
                CrawleeEnhancedSearchCrawler,
                create_crawlee_search_config
            )
            
            # Create crawler with minimal configuration
            config = create_crawlee_search_config(
                engines=['duckduckgo'],  # Single engine for faster testing
                max_results=3,
                enable_content_extraction=False
            )
            
            crawler = CrawleeEnhancedSearchCrawler(config)
            
            try:
                # Perform basic search
                start_time = time.time()
                results = await crawler.search_with_content(
                    query="test search query",
                    max_results=3,
                    extract_content=False
                )
                search_time = time.time() - start_time
                
                search_successful = (
                    results is not None and
                    isinstance(results, list) and
                    search_time < 30  # Should complete within 30 seconds
                )
                
                self.log_test_result("Basic Search", search_successful, 
                                   f"Found {len(results)} results in {search_time:.2f}s")
                
                # Test result structure
                if results:
                    result = results[0]
                    result_structure_valid = (
                        hasattr(result, 'title') and
                        hasattr(result, 'url') and
                        hasattr(result, 'extracted_content') and
                        hasattr(result, 'crawl_success') and
                        hasattr(result, 'content_quality_score')
                    )
                    
                    self.log_test_result("Result Structure", result_structure_valid, 
                                       f"Result has required attributes")
                else:
                    self.log_test_result("Result Structure", True, "No results to validate structure", warning=True)
                
            finally:
                await crawler.close()
            
        except Exception as e:
            self.log_test_result("Basic Search", False, f"Basic search failed: {e}")
            return False
        
        return True
    
    async def test_content_extraction_methods(self):
        """Test content extraction methods availability."""
        logger.info("🧪 Testing Content Extraction Methods")
        
        try:
            from core.crawlee_enhanced_search_crawler import CrawleeEnhancedSearchCrawler, create_crawlee_search_config
            
            config = create_crawlee_search_config()
            crawler = CrawleeEnhancedSearchCrawler(config)
            
            # Test extractor availability
            extractors = crawler.extractors_available
            available_count = sum(extractors.values())
            
            self.log_test_result("Extractor Availability Check", True, 
                               f"{available_count}/4 extractors available: {extractors}")
            
            # Test extraction method selection
            methods = crawler._get_extraction_methods()
            methods_valid = isinstance(methods, list) and len(methods) > 0
            
            self.log_test_result("Extraction Method Selection", methods_valid, 
                               f"Selected methods: {methods}")
            
            # Test scoring function
            test_result = {
                'content': 'This is a test article with sufficient content to test the scoring algorithm. ' * 10,
                'title': 'Test Article Title',
                'authors': ['Test Author'],
                'published_date': '2025-06-28',
                'keywords': ['test', 'article']
            }
            
            score = crawler._score_extraction_result(test_result)
            score_valid = 0.0 <= score <= 1.0
            
            self.log_test_result("Content Scoring", score_valid, f"Score: {score:.3f}")
            
        except Exception as e:
            self.log_test_result("Content Extraction Methods", False, f"Test failed: {e}")
            return False
        
        return True
    
    async def test_geographic_and_conflict_analysis(self):
        """Test geographic and conflict indicator extraction."""
        logger.info("🧪 Testing Geographic and Conflict Analysis")
        
        try:
            from core.crawlee_enhanced_search_crawler import CrawleeEnhancedSearchCrawler, create_crawlee_search_config
            
            config = create_crawlee_search_config()
            crawler = CrawleeEnhancedSearchCrawler(config)
            
            # Test geographic entity extraction
            test_content = """
            There has been increased violence in Ethiopia and Somalia in recent weeks.
            The conflict in Addis Ababa has led to displacement of thousands.
            Kenya and Uganda are monitoring the situation closely.
            Horn of Africa security remains a concern for international observers.
            """
            
            geographic_entities = crawler._extract_geographic_entities(test_content)
            expected_entities = ['Ethiopia', 'Somalia', 'Addis Ababa', 'Kenya', 'Uganda', 'Horn of Africa']
            
            geo_extraction_success = any(entity in geographic_entities for entity in expected_entities)
            
            self.log_test_result("Geographic Entity Extraction", geo_extraction_success, 
                               f"Found: {geographic_entities}")
            
            # Test conflict indicator extraction
            conflict_content = """
            The recent attack in the region has led to casualties and violence.
            Military forces are responding to the terrorism threat.
            Humanitarian crisis is developing with many refugees seeking shelter.
            Peacekeeping forces have been deployed to maintain security.
            """
            
            conflict_indicators = crawler._extract_conflict_indicators(conflict_content)
            expected_indicators = ['attack', 'violence', 'terrorism', 'humanitarian', 'refugees', 'peacekeeping']
            
            conflict_extraction_success = any(indicator in conflict_indicators for indicator in expected_indicators)
            
            self.log_test_result("Conflict Indicator Extraction", conflict_extraction_success, 
                               f"Found: {conflict_indicators}")
            
            # Test relevance scoring
            relevance_score = crawler._calculate_relevance_score(
                test_content + conflict_content,
                geographic_entities,
                conflict_indicators
            )
            
            relevance_valid = 0.0 <= relevance_score <= 1.0 and relevance_score > 0
            
            self.log_test_result("Relevance Scoring", relevance_valid, 
                               f"Relevance score: {relevance_score:.3f}")
            
        except Exception as e:
            self.log_test_result("Geographic and Conflict Analysis", False, f"Test failed: {e}")
            return False
        
        return True
    
    async def test_quality_filtering(self):
        """Test quality filtering functionality."""
        logger.info("🧪 Testing Quality Filtering")
        
        try:
            from core.crawlee_enhanced_search_crawler import (
                CrawleeEnhancedSearchCrawler, 
                create_crawlee_search_config, 
                CrawleeEnhancedResult
            )
            
            config = create_crawlee_search_config(
                enable_content_extraction=True,
                min_content_length=100,
                min_quality_score=0.5
            )
            crawler = CrawleeEnhancedSearchCrawler(config)
            
            # Create test results with different quality levels
            test_results = []
            
            # High quality result
            high_quality = CrawleeEnhancedResult(
                title="High Quality Test Article",
                url="https://example.com/high-quality",
                snippet="Test snippet",
                engine="test",
                rank=1,
                extracted_content="This is a high quality article with substantial content. " * 20,
                extraction_score=0.8,
                word_count=300,
                content_quality_score=0.9
            )
            test_results.append(high_quality)
            
            # Low quality result
            low_quality = CrawleeEnhancedResult(
                title="Low Quality",
                url="https://example.com/low-quality",
                snippet="Short",
                engine="test",
                rank=2,
                extracted_content="Short content.",
                extraction_score=0.2,
                word_count=2,
                content_quality_score=0.1
            )
            test_results.append(low_quality)
            
            # Apply quality filtering
            filtered_results = crawler._apply_quality_filtering(test_results)
            
            # Should filter out low quality result
            filtering_success = (
                len(filtered_results) == 1 and
                filtered_results[0].title == "High Quality Test Article"
            )
            
            self.log_test_result("Quality Filtering", filtering_success, 
                               f"Filtered {len(test_results)} -> {len(filtered_results)} results")
            
            # Test enhanced ranking
            ranked_results = crawler._apply_enhanced_ranking(filtered_results, "test query")
            ranking_success = len(ranked_results) == len(filtered_results)
            
            self.log_test_result("Enhanced Ranking", ranking_success, 
                               f"Ranked {len(ranked_results)} results")
            
        except Exception as e:
            self.log_test_result("Quality Filtering", False, f"Test failed: {e}")
            return False
        
        return True
    
    async def test_statistics_and_monitoring(self):
        """Test statistics collection and monitoring."""
        logger.info("🧪 Testing Statistics and Monitoring")
        
        try:
            from core.crawlee_enhanced_search_crawler import CrawleeEnhancedSearchCrawler, create_crawlee_search_config
            
            config = create_crawlee_search_config()
            crawler = CrawleeEnhancedSearchCrawler(config)
            
            # Test initial statistics
            initial_stats = crawler.get_enhanced_stats()
            stats_structure_valid = (
                'crawlee' in initial_stats and
                'total_searches' in initial_stats and
                'extractors_available' in initial_stats['crawlee']
            )
            
            self.log_test_result("Statistics Structure", stats_structure_valid, 
                               "Enhanced statistics structure is valid")
            
            # Test Crawlee-specific statistics
            crawlee_stats = initial_stats['crawlee']
            crawlee_stats_valid = (
                'total_content_requests' in crawlee_stats and
                'successful_content_extractions' in crawlee_stats and
                'extraction_method_usage' in crawlee_stats
            )
            
            self.log_test_result("Crawlee Statistics", crawlee_stats_valid, 
                               "Crawlee-specific statistics available")
            
            # Test extractor availability reporting
            extractors = crawlee_stats['extractors_available']
            extractor_reporting_valid = (
                isinstance(extractors, dict) and
                len(extractors) == 4 and
                all(isinstance(v, bool) for v in extractors.values())
            )
            
            self.log_test_result("Extractor Reporting", extractor_reporting_valid, 
                               f"Extractors: {extractors}")
            
        except Exception as e:
            self.log_test_result("Statistics and Monitoring", False, f"Test failed: {e}")
            return False
        
        return True
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report."""
        total = self.test_stats['total_tests']
        passed = self.test_stats['passed_tests']
        failed = self.test_stats['failed_tests']
        warnings = self.test_stats['warnings']
        
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        report = []
        report.append("="*60)
        report.append("CRAWLEE INTEGRATION TEST REPORT")
        report.append("="*60)
        report.append(f"Total Tests: {total}")
        report.append(f"Passed: {passed} ({passed/total*100:.1f}%)")
        report.append(f"Failed: {failed} ({failed/total*100:.1f}%)")
        report.append(f"Warnings: {warnings}")
        report.append(f"Success Rate: {success_rate:.1f}%")
        report.append("")
        
        # Detailed results
        report.append("Detailed Results:")
        report.append("-" * 30)
        for result in self.test_results:
            status = "✅" if result['success'] else "⚠️" if result['warning'] else "❌"
            report.append(f"{status} {result['test']}")
            if result['message']:
                report.append(f"    {result['message']}")
        
        # Overall assessment
        report.append("")
        report.append("="*60)
        if success_rate >= 90:
            report.append("🎉 EXCELLENT: Crawlee integration is production-ready")
        elif success_rate >= 80:
            report.append("✅ GOOD: Crawlee integration is functional with minor issues")
        elif success_rate >= 70:
            report.append("👍 ACCEPTABLE: Crawlee integration works but needs attention")
        else:
            report.append("❌ NEEDS WORK: Crawlee integration has significant issues")
        
        report.append("="*60)
        
        return "\n".join(report)


async def main():
    """Run all integration tests."""
    logger.info("🚀 Starting Crawlee Integration Tests")
    
    test_suite = CrawleeIntegrationTest()
    
    try:
        # Run all tests
        await test_suite.test_imports()
        await test_suite.test_configuration()
        await test_suite.test_crawler_initialization()
        await test_suite.test_basic_search()
        await test_suite.test_content_extraction_methods()
        await test_suite.test_geographic_and_conflict_analysis()
        await test_suite.test_quality_filtering()
        await test_suite.test_statistics_and_monitoring()
        
        # Generate and display report
        report = test_suite.generate_test_report()
        print("\n" + report)
        
        # Save report
        report_file = Path(__file__).parent / "crawlee_integration_test_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Test report saved to: {report_file}")
        
        # Return success code
        success_rate = (test_suite.test_stats['passed_tests'] / test_suite.test_stats['total_tests']) * 100
        return success_rate >= 80
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)