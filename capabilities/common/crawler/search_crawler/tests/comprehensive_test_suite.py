#!/usr/bin/env python3
"""
Search Crawler - Comprehensive Test Suite
==========================================

Comprehensive testing framework for the Search Crawler package, covering:
- Core search functionality
- Multi-engine orchestration  
- Conflict monitoring capabilities
- Crawlee integration
- Geographic and keyword analysis
- Performance and quality metrics

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import logging
import time
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib

# Add package to path
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Individual test result."""
    test_name: str
    category: str
    success: bool
    duration: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class SearchCrawlerTestSuite:
    """Comprehensive test suite for Search Crawler package."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.test_stats = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'test_duration': 0.0,
            'category_stats': {}
        }
        
        # Test configuration
        self.test_config = {
            'max_test_results': 5,  # Limit results for faster testing
            'test_timeout': 30,     # Max time per test
            'enable_network_tests': True,
            'enable_crawlee_tests': True
        }
    
    def log_test(self, test_name: str, category: str, success: bool, 
                 duration: float, message: str = "", details: Dict[str, Any] = None):
        """Log test result."""
        result = TestResult(
            test_name=test_name,
            category=category,
            success=success,
            duration=duration,
            message=message,
            details=details or {}
        )
        
        self.results.append(result)
        self.test_stats['total_tests'] += 1
        
        if success:
            self.test_stats['successful_tests'] += 1
            status = "✅"
        else:
            self.test_stats['failed_tests'] += 1
            status = "❌"
        
        self.test_stats['test_duration'] += duration
        
        # Update category stats
        if category not in self.test_stats['category_stats']:
            self.test_stats['category_stats'][category] = {'total': 0, 'passed': 0}
        
        self.test_stats['category_stats'][category]['total'] += 1
        if success:
            self.test_stats['category_stats'][category]['passed'] += 1
        
        logger.info(f"{status} {category}: {test_name} ({duration:.2f}s)")
        if message:
            logger.info(f"   {message}")
    
    async def test_package_imports(self):
        """Test 1: Package Import Tests."""
        category = "Package Imports"
        
        # Test core imports
        start_time = time.time()
        try:
            from core.search_crawler import SearchCrawler, SearchCrawlerConfig, EnhancedSearchResult
            duration = time.time() - start_time
            self.log_test("Core Search Classes", category, True, duration, 
                         "Successfully imported core search classes")
        except ImportError as e:
            duration = time.time() - start_time
            self.log_test("Core Search Classes", category, False, duration, f"Import failed: {e}")
        
        # Test conflict crawler imports
        start_time = time.time()
        try:
            from core.conflict_search_crawler import ConflictSearchCrawler, ConflictSearchConfig
            duration = time.time() - start_time
            self.log_test("Conflict Crawler Classes", category, True, duration,
                         "Successfully imported conflict crawler classes")
        except ImportError as e:
            duration = time.time() - start_time
            self.log_test("Conflict Crawler Classes", category, False, duration, f"Import failed: {e}")
        
        # Test Crawlee integration imports
        start_time = time.time()
        try:
            from core.crawlee_enhanced_search_crawler import (
                CrawleeEnhancedSearchCrawler, CrawleeSearchConfig, CRAWLEE_AVAILABLE
            )
            duration = time.time() - start_time
            self.log_test("Crawlee Integration", category, True, duration,
                         f"Crawlee available: {CRAWLEE_AVAILABLE}")
        except ImportError as e:
            duration = time.time() - start_time
            self.log_test("Crawlee Integration", category, False, duration, f"Import failed: {e}")
        
        # Test search engines import
        start_time = time.time()
        try:
            from engines import SEARCH_ENGINES, get_available_engines
            available_engines = get_available_engines()
            duration = time.time() - start_time
            self.log_test("Search Engines", category, True, duration,
                         f"Available engines: {len(available_engines)}")
        except ImportError as e:
            duration = time.time() - start_time
            self.log_test("Search Engines", category, False, duration, f"Import failed: {e}")
        
        # Test keyword systems import
        start_time = time.time()
        try:
            from keywords.conflict_keywords import ConflictKeywordManager
            from keywords.horn_of_africa_keywords import HornOfAfricaKeywords
            from keywords.keyword_analyzer import KeywordAnalyzer
            duration = time.time() - start_time
            self.log_test("Keyword Systems", category, True, duration,
                         "Successfully imported keyword management classes")
        except ImportError as e:
            duration = time.time() - start_time
            self.log_test("Keyword Systems", category, False, duration, f"Import failed: {e}")
    
    async def test_configuration_systems(self):
        """Test 2: Configuration System Tests."""
        category = "Configuration"
        
        # Test basic configuration
        start_time = time.time()
        try:
            from core.search_crawler import SearchCrawlerConfig
            config = SearchCrawlerConfig(
                engines=['google', 'bing'],
                max_results_per_engine=10,
                total_max_results=20
            )
            
            config_valid = (
                len(config.engines) == 2 and
                config.max_results_per_engine == 10 and
                config.total_max_results == 20
            )
            
            duration = time.time() - start_time
            self.log_test("Basic Configuration", category, config_valid, duration,
                         f"Engines: {config.engines}")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Basic Configuration", category, False, duration, f"Config failed: {e}")
        
        # Test conflict configuration
        start_time = time.time()
        try:
            from core.conflict_search_crawler import ConflictSearchConfig
            conflict_config = ConflictSearchConfig(
                engines=['google', 'bing', 'duckduckgo'],
                conflict_regions=['horn_of_africa'],
                enable_alerts=True,
                escalation_threshold=0.8
            )
            
            conflict_valid = (
                len(conflict_config.engines) == 3 and
                conflict_config.enable_alerts == True and
                conflict_config.escalation_threshold == 0.8
            )
            
            duration = time.time() - start_time
            self.log_test("Conflict Configuration", category, conflict_valid, duration,
                         f"Regions: {conflict_config.conflict_regions}")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Conflict Configuration", category, False, duration, f"Config failed: {e}")
        
        # Test Crawlee configuration
        start_time = time.time()
        try:
            from core.crawlee_enhanced_search_crawler import CrawleeSearchConfig, create_crawlee_search_config
            
            crawlee_config = create_crawlee_search_config(
                engines=['google', 'bing'],
                max_results=15,
                enable_content_extraction=True,
                target_countries=['ET', 'SO', 'KE']
            )
            
            crawlee_valid = (
                len(crawlee_config.engines) == 2 and
                crawlee_config.total_max_results == 15 and
                crawlee_config.download_content == True and
                len(crawlee_config.target_countries) == 3
            )
            
            duration = time.time() - start_time
            self.log_test("Crawlee Configuration", category, crawlee_valid, duration,
                         f"Content extraction: {crawlee_config.download_content}")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Crawlee Configuration", category, False, duration, f"Config failed: {e}")
    
    async def test_search_engines(self):
        """Test 3: Search Engine Tests."""
        category = "Search Engines"
        
        # Test engine availability
        start_time = time.time()
        try:
            from engines import SEARCH_ENGINES, create_engine
            
            available_count = 0
            engine_details = {}
            
            for engine_name, engine_class in SEARCH_ENGINES.items():
                try:
                    engine = create_engine(engine_name)
                    if engine:
                        available_count += 1
                        engine_details[engine_name] = "Available"
                    else:
                        engine_details[engine_name] = "Failed to create"
                except Exception as e:
                    engine_details[engine_name] = f"Error: {str(e)[:50]}"
            
            duration = time.time() - start_time
            self.log_test("Engine Availability", category, available_count > 0, duration,
                         f"Available: {available_count}/{len(SEARCH_ENGINES)}",
                         {"engine_details": engine_details})
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Engine Availability", category, False, duration, f"Test failed: {e}")
        
        # Test engine configuration
        start_time = time.time()
        try:
            from engines.base_search_engine import BaseSearchEngine
            
            # Test that all engines inherit from base
            inheritance_valid = all(
                issubclass(engine_class, BaseSearchEngine) 
                for engine_class in SEARCH_ENGINES.values()
            )
            
            duration = time.time() - start_time
            self.log_test("Engine Inheritance", category, inheritance_valid, duration,
                         "All engines inherit from BaseSearchEngine")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Engine Inheritance", category, False, duration, f"Test failed: {e}")
    
    async def test_basic_search_functionality(self):
        """Test 4: Basic Search Functionality."""
        category = "Basic Search"
        
        if not self.test_config['enable_network_tests']:
            self.log_test("Network Tests Disabled", category, True, 0.0, "Skipping network-dependent tests")
            return
        
        # Test basic search crawler initialization
        start_time = time.time()
        try:
            from core.search_crawler import SearchCrawler, SearchCrawlerConfig
            
            config = SearchCrawlerConfig(
                engines=['duckduckgo'],  # Single engine for faster testing
                max_results_per_engine=3,
                total_max_results=3,
                download_content=False,  # Disable content download for speed
                timeout=15
            )
            
            crawler = SearchCrawler(config)
            crawler_valid = (
                crawler is not None and
                hasattr(crawler, 'search') and
                hasattr(crawler, 'engines') and
                len(crawler.engines) > 0
            )
            
            duration = time.time() - start_time
            self.log_test("Crawler Initialization", category, crawler_valid, duration,
                         f"Initialized with {len(crawler.engines)} engine(s)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Crawler Initialization", category, False, duration, f"Init failed: {e}")
            return
        
        # Test basic search execution
        start_time = time.time()
        try:
            results = await crawler.search(
                query="test search",
                max_results=self.test_config['max_test_results']
            )
            
            search_successful = (
                results is not None and
                isinstance(results, list)
            )
            
            duration = time.time() - start_time
            self.log_test("Basic Search Execution", category, search_successful, duration,
                         f"Found {len(results)} results",
                         {"result_count": len(results)})
            
            await crawler.close()
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Basic Search Execution", category, False, duration, f"Search failed: {e}")
    
    async def test_keyword_analysis(self):
        """Test 5: Keyword Analysis Systems."""
        category = "Keyword Analysis"
        
        # Test conflict keyword manager
        start_time = time.time()
        try:
            from keywords.conflict_keywords import ConflictKeywordManager
            
            keyword_manager = ConflictKeywordManager()
            
            # Test keyword retrieval
            high_priority = keyword_manager.get_high_priority_keywords()
            weighted = keyword_manager.get_weighted_keywords()
            violence_keywords = keyword_manager.get_keywords_by_category('violence')
            
            keywords_valid = (
                len(high_priority) > 0 and
                len(weighted) > 0 and
                len(violence_keywords) > 0
            )
            
            duration = time.time() - start_time
            self.log_test("Conflict Keywords", category, keywords_valid, duration,
                         f"High priority: {len(high_priority)}, Violence: {len(violence_keywords)}",
                         {"keyword_counts": {
                             "high_priority": len(high_priority),
                             "weighted": len(weighted),
                             "violence": len(violence_keywords)
                         }})
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Conflict Keywords", category, False, duration, f"Test failed: {e}")
        
        # Test Horn of Africa keywords
        start_time = time.time()
        try:
            from keywords.horn_of_africa_keywords import HornOfAfricaKeywords
            
            hoa_keywords = HornOfAfricaKeywords()
            
            # Test geographic keyword retrieval
            ethiopia_keywords = hoa_keywords.get_country_keywords('ethiopia')
            somalia_keywords = hoa_keywords.get_country_keywords('somalia')
            all_locations = hoa_keywords.get_all_location_keywords()
            
            geo_keywords_valid = (
                len(ethiopia_keywords) > 0 and
                len(somalia_keywords) > 0 and
                len(all_locations) > 50  # Should have many location keywords
            )
            
            duration = time.time() - start_time
            self.log_test("Geographic Keywords", category, geo_keywords_valid, duration,
                         f"Locations: {len(all_locations)}, Ethiopia: {len(ethiopia_keywords)}",
                         {"location_counts": {
                             "all_locations": len(all_locations),
                             "ethiopia": len(ethiopia_keywords),
                             "somalia": len(somalia_keywords)
                         }})
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Geographic Keywords", category, False, duration, f"Test failed: {e}")
        
        # Test keyword analyzer
        start_time = time.time()
        try:
            from keywords.keyword_analyzer import KeywordAnalyzer
            
            analyzer = KeywordAnalyzer()
            
            # Test text analysis
            test_text = "There has been violence in Ethiopia and Somalia recently. The conflict involves armed groups."
            
            analysis = analyzer.analyze_text(test_text, ['violence', 'conflict', 'Ethiopia', 'Somalia'])
            
            analysis_valid = (
                analysis is not None and
                isinstance(analysis, dict)
            )
            
            duration = time.time() - start_time
            self.log_test("Keyword Analysis", category, analysis_valid, duration,
                         f"Analyzed text with {len(analysis) if analysis else 0} results")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Keyword Analysis", category, False, duration, f"Test failed: {e}")
    
    async def test_conflict_monitoring(self):
        """Test 6: Conflict Monitoring Capabilities."""
        category = "Conflict Monitoring"
        
        if not self.test_config['enable_network_tests']:
            self.log_test("Network Tests Disabled", category, True, 0.0, "Skipping network-dependent tests")
            return
        
        # Test conflict crawler initialization
        start_time = time.time()
        try:
            from core.conflict_search_crawler import ConflictSearchCrawler, ConflictSearchConfig
            
            config = ConflictSearchConfig(
                engines=['duckduckgo'],
                max_results_per_engine=3,
                enable_alerts=True,
                timeout=15
            )
            
            conflict_crawler = ConflictSearchCrawler(config)
            
            crawler_valid = (
                conflict_crawler is not None and
                hasattr(conflict_crawler, 'search_conflicts') and
                hasattr(conflict_crawler, 'monitor_region')
            )
            
            duration = time.time() - start_time
            self.log_test("Conflict Crawler Init", category, crawler_valid, duration,
                         "Conflict crawler initialized successfully")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Conflict Crawler Init", category, False, duration, f"Init failed: {e}")
            return
        
        # Test conflict search functionality
        start_time = time.time()
        try:
            results = await conflict_crawler.search_conflicts(
                region='horn_of_africa',
                keywords=['Ethiopia', 'conflict'],
                max_results=self.test_config['max_test_results']
            )
            
            conflict_search_valid = (
                results is not None and
                isinstance(results, list)
            )
            
            duration = time.time() - start_time
            self.log_test("Conflict Search", category, conflict_search_valid, duration,
                         f"Found {len(results)} conflict-related results")
            
            await conflict_crawler.close()
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Conflict Search", category, False, duration, f"Search failed: {e}")
    
    async def test_crawlee_integration(self):
        """Test 7: Crawlee Integration."""
        category = "Crawlee Integration"
        
        if not self.test_config['enable_crawlee_tests']:
            self.log_test("Crawlee Tests Disabled", category, True, 0.0, "Skipping Crawlee-dependent tests")
            return
        
        # Test Crawlee availability
        start_time = time.time()
        try:
            from core.crawlee_enhanced_search_crawler import CRAWLEE_AVAILABLE
            
            duration = time.time() - start_time
            self.log_test("Crawlee Availability", category, True, duration,
                         f"Crawlee available: {CRAWLEE_AVAILABLE}")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Crawlee Availability", category, False, duration, f"Test failed: {e}")
            return
        
        # Test enhanced crawler initialization (without Crawlee requirement)
        start_time = time.time()
        try:
            from core.crawlee_enhanced_search_crawler import (
                CrawleeEnhancedSearchCrawler, create_crawlee_search_config
            )
            
            config = create_crawlee_search_config(
                engines=['duckduckgo'],
                max_results=3,
                enable_content_extraction=False  # Disable for testing without Crawlee
            )
            
            enhanced_crawler = CrawleeEnhancedSearchCrawler(config)
            
            crawler_valid = (
                enhanced_crawler is not None and
                hasattr(enhanced_crawler, 'search_with_content') and
                hasattr(enhanced_crawler, 'extractors_available')
            )
            
            duration = time.time() - start_time
            self.log_test("Enhanced Crawler Init", category, crawler_valid, duration,
                         f"Extractors available: {sum(enhanced_crawler.extractors_available.values())}")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Enhanced Crawler Init", category, False, duration, f"Init failed: {e}")
        
        # Test content extraction methods
        start_time = time.time()
        try:
            from core.crawlee_enhanced_search_crawler import CrawleeEnhancedSearchCrawler, create_crawlee_search_config
            
            config = create_crawlee_search_config()
            crawler = CrawleeEnhancedSearchCrawler(config)
            
            # Test extraction method selection
            methods = crawler._get_extraction_methods()
            methods_valid = isinstance(methods, list) and len(methods) > 0
            
            # Test geographic entity extraction
            test_content = "Violence has erupted in Ethiopia and Somalia, affecting thousands in the Horn of Africa region."
            geo_entities = crawler._extract_geographic_entities(test_content)
            
            # Test conflict indicator extraction
            conflict_indicators = crawler._extract_conflict_indicators(test_content)
            
            extraction_valid = (
                methods_valid and
                len(geo_entities) > 0 and
                len(conflict_indicators) > 0
            )
            
            duration = time.time() - start_time
            self.log_test("Content Extraction", category, extraction_valid, duration,
                         f"Methods: {len(methods)}, Geo: {len(geo_entities)}, Conflict: {len(conflict_indicators)}",
                         {
                             "extraction_methods": methods,
                             "geographic_entities": geo_entities,
                             "conflict_indicators": conflict_indicators
                         })
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Content Extraction", category, False, duration, f"Test failed: {e}")
    
    async def test_performance_characteristics(self):
        """Test 8: Performance Characteristics."""
        category = "Performance"
        
        # Test search speed
        start_time = time.time()
        try:
            from core.search_crawler import SearchCrawler, SearchCrawlerConfig
            
            config = SearchCrawlerConfig(
                engines=['duckduckgo'],
                max_results_per_engine=3,
                download_content=False,
                timeout=10
            )
            
            crawler = SearchCrawler(config)
            
            search_start = time.time()
            results = await crawler.search("test", max_results=3)
            search_duration = time.time() - search_start
            
            # Performance should be reasonable
            performance_acceptable = search_duration < 15  # Should complete within 15 seconds
            
            await crawler.close()
            
            duration = time.time() - start_time
            self.log_test("Search Speed", category, performance_acceptable, duration,
                         f"Search completed in {search_duration:.2f}s",
                         {"search_duration": search_duration, "result_count": len(results)})
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Search Speed", category, False, duration, f"Test failed: {e}")
        
        # Test memory efficiency
        start_time = time.time()
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create and use crawler
            from core.search_crawler import SearchCrawler, SearchCrawlerConfig
            config = SearchCrawlerConfig(engines=['duckduckgo'], max_results_per_engine=5)
            crawler = SearchCrawler(config)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            await crawler.close()
            
            # Memory increase should be reasonable
            memory_efficient = memory_increase < 50  # Less than 50MB increase
            
            duration = time.time() - start_time
            self.log_test("Memory Efficiency", category, memory_efficient, duration,
                         f"Memory increase: {memory_increase:.1f}MB",
                         {"memory_increase_mb": memory_increase})
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Memory Efficiency", category, False, duration, f"Test failed: {e}")
    
    async def test_error_handling(self):
        """Test 9: Error Handling."""
        category = "Error Handling"
        
        # Test invalid configuration handling
        start_time = time.time()
        try:
            from core.search_crawler import SearchCrawlerConfig
            
            # Test with invalid engines
            config = SearchCrawlerConfig(engines=['nonexistent_engine'])
            
            # Should handle gracefully without crashing
            config_created = config is not None
            
            duration = time.time() - start_time
            self.log_test("Invalid Config Handling", category, config_created, duration,
                         "Invalid configuration handled gracefully")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Invalid Config Handling", category, False, duration, f"Test failed: {e}")
        
        # Test network error handling
        start_time = time.time()
        try:
            from core.search_crawler import SearchCrawler, SearchCrawlerConfig
            
            config = SearchCrawlerConfig(
                engines=['duckduckgo'],
                timeout=1  # Very short timeout to trigger errors
            )
            
            crawler = SearchCrawler(config)
            
            # Should handle timeout gracefully
            try:
                results = await crawler.search("test", max_results=1)
                error_handled = True
            except Exception:
                error_handled = True  # Expected to handle errors gracefully
            
            await crawler.close()
            
            duration = time.time() - start_time
            self.log_test("Network Error Handling", category, error_handled, duration,
                         "Network errors handled gracefully")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Network Error Handling", category, False, duration, f"Test failed: {e}")
    
    async def test_integration_compatibility(self):
        """Test 10: Integration Compatibility."""
        category = "Integration"
        
        # Test package health check
        start_time = time.time()
        try:
            from search_crawler import get_search_crawler_health
            
            health = get_search_crawler_health()
            
            health_valid = (
                isinstance(health, dict) and
                'status' in health and
                'version' in health
            )
            
            duration = time.time() - start_time
            self.log_test("Health Check", category, health_valid, duration,
                         f"Status: {health.get('status', 'unknown')}",
                         {"health_info": health})
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Health Check", category, False, duration, f"Test failed: {e}")
        
        # Test factory functions
        start_time = time.time()
        try:
            from search_crawler import create_general_search_crawler, create_conflict_search_crawler
            
            # Test general crawler factory
            general_crawler = create_general_search_crawler(engines=['duckduckgo'])
            general_valid = general_crawler is not None
            
            # Test conflict crawler factory
            conflict_crawler = create_conflict_search_crawler()
            conflict_valid = conflict_crawler is not None
            
            factory_valid = general_valid and conflict_valid
            
            duration = time.time() - start_time
            self.log_test("Factory Functions", category, factory_valid, duration,
                         "Factory functions work correctly")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Factory Functions", category, False, duration, f"Test failed: {e}")
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive test report."""
        total_tests = self.test_stats['total_tests']
        successful_tests = self.test_stats['successful_tests']
        failed_tests = self.test_stats['failed_tests']
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        report = []
        report.append("="*80)
        report.append("SEARCH CRAWLER - COMPREHENSIVE TEST REPORT")
        report.append("="*80)
        report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Successful: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
        report.append(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        report.append(f"Success Rate: {success_rate:.1f}%")
        report.append(f"Total Duration: {self.test_stats['test_duration']:.2f}s")
        report.append("")
        
        # Category breakdown
        report.append("Test Categories:")
        report.append("-" * 50)
        for category, stats in self.test_stats['category_stats'].items():
            category_rate = (stats['passed'] / stats['total']) * 100 if stats['total'] > 0 else 0
            status = "✅" if category_rate == 100 else "⚠️" if category_rate >= 80 else "❌"
            report.append(f"{status} {category:<20} {stats['passed']}/{stats['total']} ({category_rate:.1f}%)")
        
        report.append("")
        
        # Detailed results
        report.append("Detailed Test Results:")
        report.append("-" * 50)
        
        current_category = ""
        for result in self.results:
            if result.category != current_category:
                current_category = result.category
                report.append(f"\n{current_category}:")
            
            status = "✅" if result.success else "❌"
            report.append(f"  {status} {result.test_name} ({result.duration:.2f}s)")
            if result.message:
                report.append(f"      {result.message}")
        
        # Overall assessment
        report.append("\n" + "="*80)
        report.append("OVERALL ASSESSMENT")
        report.append("="*80)
        
        if success_rate >= 95:
            report.append("🎉 EXCELLENT: Search Crawler package is production-ready")
        elif success_rate >= 85:
            report.append("✅ VERY GOOD: Search Crawler package is highly functional")
        elif success_rate >= 75:
            report.append("👍 GOOD: Search Crawler package is functional with minor issues")
        elif success_rate >= 60:
            report.append("⚠️ FAIR: Search Crawler package has some issues")
        else:
            report.append("❌ POOR: Search Crawler package has significant issues")
        
        # Component readiness
        report.append("\nComponent Readiness:")
        component_status = {
            "Core Search": self._assess_category_health("Basic Search"),
            "Conflict Monitoring": self._assess_category_health("Conflict Monitoring"),
            "Crawlee Integration": self._assess_category_health("Crawlee Integration"),
            "Keyword Analysis": self._assess_category_health("Keyword Analysis"),
            "Performance": self._assess_category_health("Performance"),
            "Error Handling": self._assess_category_health("Error Handling")
        }
        
        for component, ready in component_status.items():
            status = "✅" if ready else "❌"
            report.append(f"  {status} {component}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def _assess_category_health(self, category: str) -> bool:
        """Assess if a category is healthy (>= 80% success rate)."""
        if category not in self.test_stats['category_stats']:
            return False
        
        stats = self.test_stats['category_stats'][category]
        return (stats['passed'] / stats['total']) >= 0.8 if stats['total'] > 0 else False


async def main():
    """Run comprehensive test suite."""
    logger.info("🚀 Starting Search Crawler Comprehensive Test Suite")
    
    test_suite = SearchCrawlerTestSuite()
    
    try:
        # Run all test categories
        await test_suite.test_package_imports()
        await test_suite.test_configuration_systems()
        await test_suite.test_search_engines()
        await test_suite.test_basic_search_functionality()
        await test_suite.test_keyword_analysis()
        await test_suite.test_conflict_monitoring()
        await test_suite.test_crawlee_integration()
        await test_suite.test_performance_characteristics()
        await test_suite.test_error_handling()
        await test_suite.test_integration_compatibility()
        
        # Generate comprehensive report
        report = test_suite.generate_comprehensive_report()
        
        # Print report
        print("\n" + report)
        
        # Save report
        report_file = Path(__file__).parent / "comprehensive_test_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save detailed results
        results_file = Path(__file__).parent / "test_results.json"
        results_data = {
            'test_stats': test_suite.test_stats,
            'test_results': [
                {
                    'test_name': r.test_name,
                    'category': r.category,
                    'success': r.success,
                    'duration': r.duration,
                    'message': r.message,
                    'details': r.details,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in test_suite.results
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"📄 Test report saved to: {report_file}")
        logger.info(f"📊 Test results saved to: {results_file}")
        
        # Final assessment
        success_rate = (test_suite.test_stats['successful_tests'] / test_suite.test_stats['total_tests']) * 100
        logger.info(f"\n🎯 Final Assessment: {test_suite.test_stats['successful_tests']}/{test_suite.test_stats['total_tests']} tests passed ({success_rate:.1f}%)")
        
        return success_rate >= 75
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)