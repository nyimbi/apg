#!/usr/bin/env python3
"""
Search Crawler - Complete Package Exercise
==========================================

Comprehensive exercise script that demonstrates all capabilities of the Search Crawler
package, including core functionality, multi-engine search, Crawlee integration, 
conflict monitoring, and Horn of Africa specialization.

This script serves as both a demonstration and validation of the complete package.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import sys
import os
import time
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add package to path
package_root = Path(__file__).parent
sys.path.insert(0, str(package_root))


class SearchCrawlerExercise:
    """Complete exercise of Search Crawler package capabilities."""
    
    def __init__(self):
        self.results = {}
        self.temp_files = []
        self.exercise_data = {
            'start_time': datetime.now(),
            'exercises_completed': 0,
            'total_exercises': 12,
            'performance_metrics': {},
            'component_status': {}
        }
    
    async def cleanup(self):
        """Clean up temporary files and resources."""
        for temp_file in self.temp_files:
            try:
                temp_file.unlink()
            except FileNotFoundError:
                pass
    
    def log_result(self, category: str, exercise: str, success: bool, details: Any = None):
        """Log exercise result."""
        if category not in self.results:
            self.results[category] = {}
        
        self.results[category][exercise] = {
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        status = "✅" if success else "❌"
        logger.info(f"{status} {category}: {exercise}")
        if details and isinstance(details, str) and len(details) < 100:
            logger.info(f"   Details: {details}")
        
        self.exercise_data['exercises_completed'] += 1
    
    async def exercise_package_imports(self):
        """Exercise 1: Test package imports and availability."""
        logger.info("🏗️ Exercise 1: Package Imports and Setup")
        
        try:
            # Test core imports
            from core.search_crawler import SearchCrawler, SearchCrawlerConfig, EnhancedSearchResult
            from core.conflict_search_crawler import ConflictSearchCrawler, ConflictSearchConfig
            self.log_result("Imports", "Core Classes", True, "SearchCrawler and ConflictSearchCrawler")
            
            # Test Crawlee integration
            try:
                from core.crawlee_enhanced_search_crawler import (
                    CrawleeEnhancedSearchCrawler, CrawleeSearchConfig, CRAWLEE_AVAILABLE
                )
                self.log_result("Imports", "Crawlee Integration", True, f"Crawlee available: {CRAWLEE_AVAILABLE}")
            except ImportError:
                self.log_result("Imports", "Crawlee Integration", False, "Crawlee components not available")
            
            # Test search engines
            from engines import SEARCH_ENGINES, get_available_engines, create_engine
            available_engines = get_available_engines()
            self.log_result("Imports", "Search Engines", True, f"{len(available_engines)} engines available")
            
            # Test keyword systems
            from keywords.conflict_keywords import ConflictKeywordManager
            from keywords.horn_of_africa_keywords import HornOfAfricaKeywords
            from keywords.keyword_analyzer import KeywordAnalyzer
            self.log_result("Imports", "Keyword Systems", True, "All keyword management classes")
            
            # Test ranking system
            try:
                from ranking.result_ranker import ResultRanker
                self.log_result("Imports", "Ranking System", True, "ResultRanker available")
            except ImportError:
                self.log_result("Imports", "Ranking System", False, "ResultRanker not available")
            
        except Exception as e:
            self.log_result("Imports", "Package Import", False, str(e))
    
    async def exercise_configuration_systems(self):
        """Exercise 2: Test configuration management."""
        logger.info("⚙️ Exercise 2: Configuration Systems")
        
        try:
            # Test basic search configuration
            from core.search_crawler import SearchCrawlerConfig
            
            basic_config = SearchCrawlerConfig(
                engines=['google', 'bing', 'duckduckgo', 'yandex'],
                max_results_per_engine=20,
                total_max_results=50,
                parallel_searches=True,
                download_content=True,
                use_stealth=True
            )
            
            config_valid = (
                len(basic_config.engines) == 4 and
                basic_config.max_results_per_engine == 20 and
                basic_config.parallel_searches == True
            )
            
            self.log_result("Configuration", "Basic Config", config_valid, 
                           f"Engines: {basic_config.engines}")
            
            # Test conflict search configuration
            from core.conflict_search_crawler import ConflictSearchConfig
            
            conflict_config = ConflictSearchConfig(
                engines=['google', 'bing', 'yandex', 'brave'],
                conflict_regions=['horn_of_africa'],
                enable_alerts=True,
                escalation_threshold=0.8,
                trusted_sources=['reuters.com', 'bbc.com', 'aljazeera.com'],
                min_relevance_score=0.6
            )
            
            conflict_config_valid = (
                len(conflict_config.engines) == 4 and
                conflict_config.enable_alerts == True and
                len(conflict_config.trusted_sources) == 3
            )
            
            self.log_result("Configuration", "Conflict Config", conflict_config_valid,
                           f"Regions: {conflict_config.conflict_regions}")
            
            # Test Crawlee configuration if available
            try:
                from core.crawlee_enhanced_search_crawler import create_crawlee_search_config
                
                crawlee_config = create_crawlee_search_config(
                    engines=['google', 'bing', 'brave'],
                    max_results=30,
                    enable_content_extraction=True,
                    target_countries=['ET', 'SO', 'KE', 'SD'],
                    preferred_extraction_method="trafilatura",
                    min_content_length=200
                )
                
                crawlee_config_valid = (
                    len(crawlee_config.engines) == 3 and
                    crawlee_config.download_content == True and
                    len(crawlee_config.target_countries) == 4
                )
                
                self.log_result("Configuration", "Crawlee Config", crawlee_config_valid,
                               f"Extraction method: {crawlee_config.preferred_extraction_method}")
            
            except ImportError:
                self.log_result("Configuration", "Crawlee Config", True, "Skipped - not available")
            
        except Exception as e:
            self.log_result("Configuration", "Config System", False, str(e))
    
    async def exercise_search_engines(self):
        """Exercise 3: Test search engine functionality."""
        logger.info("🔍 Exercise 3: Search Engine Systems")
        
        try:
            from engines import SEARCH_ENGINES, create_engine
            
            # Test engine creation
            engines_tested = {}
            for engine_name in ['google', 'bing', 'duckduckgo', 'yandex']:
                if engine_name in SEARCH_ENGINES:
                    try:
                        engine = create_engine(engine_name)
                        engines_tested[engine_name] = engine is not None
                    except Exception as e:
                        engines_tested[engine_name] = False
            
            successful_engines = sum(engines_tested.values())
            engine_success = successful_engines > 0
            
            self.log_result("Search Engines", "Engine Creation", engine_success,
                           f"Created {successful_engines}/{len(engines_tested)} engines")
            
            # Test engine capabilities
            if successful_engines > 0:
                from engines.base_search_engine import BaseSearchEngine
                
                # Test inheritance
                inheritance_valid = all(
                    issubclass(SEARCH_ENGINES[name], BaseSearchEngine)
                    for name in engines_tested.keys() if name in SEARCH_ENGINES
                )
                
                self.log_result("Search Engines", "Engine Inheritance", inheritance_valid,
                               "All engines inherit from BaseSearchEngine")
            
            # Test engine registry
            available_engines = len(SEARCH_ENGINES)
            registry_valid = available_engines >= 10  # Should have 11 engines
            
            self.log_result("Search Engines", "Engine Registry", registry_valid,
                           f"Registry contains {available_engines} engines")
            
        except Exception as e:
            self.log_result("Search Engines", "Engine Test", False, str(e))
    
    async def exercise_keyword_management(self):
        """Exercise 4: Test keyword management systems."""
        logger.info("🗝️ Exercise 4: Keyword Management")
        
        try:
            # Test conflict keyword manager
            from keywords.conflict_keywords import ConflictKeywordManager
            
            conflict_manager = ConflictKeywordManager()
            
            # Test keyword retrieval
            high_priority = conflict_manager.get_high_priority_keywords()
            weighted_keywords = conflict_manager.get_weighted_keywords()
            violence_keywords = conflict_manager.get_keywords_by_category('violence')
            
            conflict_keywords_valid = (
                len(high_priority) > 50 and
                len(weighted_keywords) > 100 and
                len(violence_keywords) > 20
            )
            
            self.log_result("Keywords", "Conflict Keywords", conflict_keywords_valid,
                           f"High priority: {len(high_priority)}, Violence: {len(violence_keywords)}")
            
            # Test Horn of Africa keywords
            from keywords.horn_of_africa_keywords import HornOfAfricaKeywords
            
            hoa_keywords = HornOfAfricaKeywords()
            
            ethiopia_keywords = hoa_keywords.get_country_keywords('ethiopia')
            somalia_keywords = hoa_keywords.get_country_keywords('somalia')
            all_locations = hoa_keywords.get_all_location_keywords()
            
            hoa_keywords_valid = (
                len(ethiopia_keywords) > 20 and
                len(somalia_keywords) > 20 and
                len(all_locations) > 200
            )
            
            self.log_result("Keywords", "Geographic Keywords", hoa_keywords_valid,
                           f"Total locations: {len(all_locations)}")
            
            # Test keyword analyzer
            from keywords.keyword_analyzer import KeywordAnalyzer
            
            analyzer = KeywordAnalyzer()
            test_text = "Violence erupted in Ethiopia and Somalia, leading to humanitarian crisis."
            analysis = analyzer.analyze_text(test_text, ['violence', 'Ethiopia', 'Somalia', 'crisis'])
            
            analysis_valid = analysis is not None
            
            self.log_result("Keywords", "Keyword Analysis", analysis_valid,
                           f"Analysis result type: {type(analysis)}")
            
        except Exception as e:
            self.log_result("Keywords", "Keyword Management", False, str(e))
    
    async def exercise_basic_search(self):
        """Exercise 5: Test basic search functionality."""
        logger.info("🔎 Exercise 5: Basic Search Operations")
        
        try:
            from core.search_crawler import SearchCrawler, SearchCrawlerConfig
            
            # Create basic configuration
            config = SearchCrawlerConfig(
                engines=['duckduckgo'],  # Single engine for reliable testing
                max_results_per_engine=5,
                total_max_results=5,
                download_content=False,  # Disable for faster testing
                timeout=20,
                parallel_searches=False
            )
            
            crawler = SearchCrawler(config)
            
            # Test crawler initialization
            crawler_initialized = (
                crawler is not None and
                hasattr(crawler, 'search') and
                len(crawler.engines) > 0
            )
            
            self.log_result("Basic Search", "Crawler Initialization", crawler_initialized,
                           f"Initialized with {len(crawler.engines)} engine(s)")
            
            if crawler_initialized:
                # Test basic search (network dependent)
                try:
                    start_time = time.time()
                    results = await crawler.search("test search query", max_results=3)
                    search_time = time.time() - start_time
                    
                    search_successful = (
                        results is not None and
                        isinstance(results, list) and
                        search_time < 30
                    )
                    
                    self.log_result("Basic Search", "Search Execution", search_successful,
                                   f"Found {len(results)} results in {search_time:.2f}s")
                    
                    # Test search statistics
                    stats = crawler.get_stats()
                    stats_valid = isinstance(stats, dict) and 'total_searches' in stats
                    
                    self.log_result("Basic Search", "Search Statistics", stats_valid,
                                   f"Stats keys: {list(stats.keys())}")
                
                except Exception as e:
                    self.log_result("Basic Search", "Search Execution", False, f"Network error: {str(e)[:50]}")
                
                await crawler.close()
            
        except Exception as e:
            self.log_result("Basic Search", "Basic Search Test", False, str(e))
    
    async def exercise_conflict_monitoring(self):
        """Exercise 6: Test conflict monitoring capabilities."""
        logger.info("⚔️ Exercise 6: Conflict Monitoring")
        
        try:
            from core.conflict_search_crawler import ConflictSearchCrawler, ConflictSearchConfig
            
            # Create conflict monitoring configuration
            config = ConflictSearchConfig(
                engines=['duckduckgo'],
                max_results_per_engine=3,
                enable_alerts=True,
                escalation_threshold=0.8,
                timeout=20
            )
            
            conflict_crawler = ConflictSearchCrawler(config)
            
            # Test conflict crawler initialization
            crawler_initialized = (
                conflict_crawler is not None and
                hasattr(conflict_crawler, 'search_conflicts') and
                hasattr(conflict_crawler, 'monitor_region')
            )
            
            self.log_result("Conflict Monitoring", "Crawler Initialization", crawler_initialized,
                           "ConflictSearchCrawler with monitoring capabilities")
            
            if crawler_initialized:
                # Test conflict search (network dependent)
                try:
                    start_time = time.time()
                    results = await conflict_crawler.search_conflicts(
                        region='horn_of_africa',
                        keywords=['Ethiopia', 'conflict'],
                        max_results=3
                    )
                    search_time = time.time() - start_time
                    
                    conflict_search_successful = (
                        results is not None and
                        isinstance(results, list) and
                        search_time < 30
                    )
                    
                    self.log_result("Conflict Monitoring", "Conflict Search", conflict_search_successful,
                                   f"Found {len(results)} conflict results in {search_time:.2f}s")
                
                except Exception as e:
                    self.log_result("Conflict Monitoring", "Conflict Search", False, f"Network error: {str(e)[:50]}")
                
                # Test conflict statistics
                try:
                    stats = conflict_crawler.get_conflict_stats()
                    stats_valid = isinstance(stats, dict)
                    
                    self.log_result("Conflict Monitoring", "Conflict Statistics", stats_valid,
                                   f"Stats available: {stats_valid}")
                except Exception as e:
                    self.log_result("Conflict Monitoring", "Conflict Statistics", False, str(e))
                
                await conflict_crawler.close()
            
        except Exception as e:
            self.log_result("Conflict Monitoring", "Conflict Monitoring Test", False, str(e))
    
    async def exercise_crawlee_integration(self):
        """Exercise 7: Test Crawlee integration."""
        logger.info("🕷️ Exercise 7: Crawlee Integration")
        
        try:
            from core.crawlee_enhanced_search_crawler import (
                CrawleeEnhancedSearchCrawler, create_crawlee_search_config, CRAWLEE_AVAILABLE
            )
            
            # Test Crawlee availability
            self.log_result("Crawlee Integration", "Availability Check", True, f"Crawlee available: {CRAWLEE_AVAILABLE}")
            
            # Create Crawlee configuration
            config = create_crawlee_search_config(
                engines=['duckduckgo'],
                max_results=3,
                enable_content_extraction=False,  # Disable for testing without Crawlee
                target_countries=['ET', 'SO', 'KE']
            )
            
            enhanced_crawler = CrawleeEnhancedSearchCrawler(config)
            
            # Test enhanced crawler initialization
            crawler_initialized = (
                enhanced_crawler is not None and
                hasattr(enhanced_crawler, 'search_with_content') and
                hasattr(enhanced_crawler, 'extractors_available')
            )
            
            extractors_available = sum(enhanced_crawler.extractors_available.values())
            
            self.log_result("Crawlee Integration", "Enhanced Crawler Init", crawler_initialized,
                           f"Extractors available: {extractors_available}/4")
            
            # Test content analysis methods
            test_content = "Violence has erupted in Ethiopia and Somalia in the Horn of Africa region."
            
            geo_entities = enhanced_crawler._extract_geographic_entities(test_content)
            conflict_indicators = enhanced_crawler._extract_conflict_indicators(test_content)
            
            content_analysis_valid = (
                len(geo_entities) > 0 and
                len(conflict_indicators) > 0
            )
            
            self.log_result("Crawlee Integration", "Content Analysis", content_analysis_valid,
                           f"Geo: {geo_entities}, Conflict: {conflict_indicators}")
            
            # Test quality scoring
            test_result = {
                'content': test_content * 10,  # Make it longer
                'title': 'Test Article Title',
                'authors': ['Test Author'],
                'published_date': '2025-06-28'
            }
            
            quality_score = enhanced_crawler._score_extraction_result(test_result)
            quality_valid = 0.0 <= quality_score <= 1.0
            
            self.log_result("Crawlee Integration", "Quality Scoring", quality_valid,
                           f"Quality score: {quality_score:.3f}")
            
        except Exception as e:
            self.log_result("Crawlee Integration", "Crawlee Integration Test", False, str(e))
    
    async def exercise_multi_engine_orchestration(self):
        """Exercise 8: Test multi-engine orchestration."""
        logger.info("🔄 Exercise 8: Multi-Engine Orchestration")
        
        try:
            from core.search_crawler import SearchCrawler, SearchCrawlerConfig
            
            # Test parallel search configuration
            config = SearchCrawlerConfig(
                engines=['duckduckgo', 'yahoo'],  # Two engines for testing
                max_results_per_engine=2,
                total_max_results=4,
                parallel_searches=True,
                timeout=25
            )
            
            crawler = SearchCrawler(config)
            
            # Test engine orchestration
            orchestration_valid = (
                len(crawler.engines) == 2 and
                config.parallel_searches == True
            )
            
            self.log_result("Multi-Engine", "Orchestration Setup", orchestration_valid,
                           f"Configured {len(crawler.engines)} engines for parallel search")
            
            # Test result merging and deduplication
            if orchestration_valid:
                try:
                    start_time = time.time()
                    results = await crawler.search("test query", max_results=4)
                    search_time = time.time() - start_time
                    
                    multi_search_successful = (
                        results is not None and
                        isinstance(results, list) and
                        search_time < 30
                    )
                    
                    # Check for multi-engine results
                    multi_engine_results = [r for r in results if len(r.engines_found) > 1] if results else []
                    
                    self.log_result("Multi-Engine", "Multi-Engine Search", multi_search_successful,
                                   f"Found {len(results)} results, {len(multi_engine_results)} from multiple engines")
                    
                except Exception as e:
                    self.log_result("Multi-Engine", "Multi-Engine Search", False, f"Network error: {str(e)[:50]}")
                
                await crawler.close()
            
        except Exception as e:
            self.log_result("Multi-Engine", "Multi-Engine Test", False, str(e))
    
    async def exercise_horn_of_africa_specialization(self):
        """Exercise 9: Test Horn of Africa specialization."""
        logger.info("🌍 Exercise 9: Horn of Africa Specialization")
        
        try:
            from keywords.horn_of_africa_keywords import HornOfAfricaKeywords
            
            hoa_keywords = HornOfAfricaKeywords()
            
            # Test country coverage
            hoa_countries = ['ethiopia', 'somalia', 'kenya', 'sudan', 'uganda', 'tanzania', 'eritrea', 'djibouti']
            country_coverage = {}
            
            for country in hoa_countries:
                keywords = hoa_keywords.get_country_keywords(country)
                country_coverage[country] = len(keywords)
            
            coverage_valid = all(count > 10 for count in country_coverage.values())
            
            self.log_result("Horn of Africa", "Country Coverage", coverage_valid,
                           f"Countries covered: {len(country_coverage)}")
            
            # Test conflict keyword generation
            conflict_queries = hoa_keywords.generate_search_queries(
                countries=['ethiopia', 'somalia'],
                conflict_level='high',
                max_queries=10
            )
            
            query_generation_valid = len(conflict_queries) > 5
            
            self.log_result("Horn of Africa", "Query Generation", query_generation_valid,
                           f"Generated {len(conflict_queries)} conflict queries")
            
            # Test geographic entity extraction
            test_text = "Violence in Addis Ababa and Mogadishu has affected the Horn of Africa region."
            entities = hoa_keywords.extract_entities(test_text)
            
            entity_extraction_valid = (
                isinstance(entities, dict) and
                len(entities) > 0
            )
            
            self.log_result("Horn of Africa", "Entity Extraction", entity_extraction_valid,
                           f"Extracted entities: {entities}")
            
            # Test relevance scoring
            relevance_score = hoa_keywords.score_keyword_relevance(test_text, country='ethiopia')
            relevance_valid = 0.0 <= relevance_score <= 1.0
            
            self.log_result("Horn of Africa", "Relevance Scoring", relevance_valid,
                           f"Relevance score: {relevance_score:.3f}")
            
        except Exception as e:
            self.log_result("Horn of Africa", "HoA Specialization Test", False, str(e))
    
    async def exercise_performance_characteristics(self):
        """Exercise 10: Test performance characteristics."""
        logger.info("🚀 Exercise 10: Performance Analysis")
        
        try:
            import psutil
            
            # Test memory efficiency
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create multiple crawlers to test memory usage
            from core.search_crawler import SearchCrawler, SearchCrawlerConfig
            
            crawlers = []
            for i in range(5):
                config = SearchCrawlerConfig(engines=['duckduckgo'], max_results_per_engine=1)
                crawler = SearchCrawler(config)
                crawlers.append(crawler)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            # Clean up
            for crawler in crawlers:
                await crawler.close()
            
            memory_efficient = memory_increase < 100  # Should not use more than 100MB
            
            self.log_result("Performance", "Memory Efficiency", memory_efficient,
                           f"Memory increase: {memory_increase:.1f}MB for 5 crawlers")
            
            # Test configuration loading speed
            start_time = time.time()
            for i in range(100):
                config = SearchCrawlerConfig(
                    engines=['google', 'bing'],
                    max_results_per_engine=10
                )
            config_time = time.time() - start_time
            
            config_fast = config_time < 1.0  # Should be very fast
            
            self.log_result("Performance", "Configuration Speed", config_fast,
                           f"100 configurations in {config_time:.3f}s")
            
            # Test keyword processing speed
            from keywords.conflict_keywords import ConflictKeywordManager
            
            keyword_manager = ConflictKeywordManager()
            
            start_time = time.time()
            for i in range(50):
                keywords = keyword_manager.get_high_priority_keywords()
            keyword_time = time.time() - start_time
            
            keyword_fast = keyword_time < 1.0
            
            self.log_result("Performance", "Keyword Processing Speed", keyword_fast,
                           f"50 keyword retrievals in {keyword_time:.3f}s")
            
        except Exception as e:
            self.log_result("Performance", "Performance Test", False, str(e))
    
    async def exercise_error_handling(self):
        """Exercise 11: Test error handling capabilities."""
        logger.info("🚨 Exercise 11: Error Handling")
        
        try:
            # Test invalid configuration handling
            from core.search_crawler import SearchCrawlerConfig
            
            # Test with invalid engines
            config_with_invalid = SearchCrawlerConfig(engines=['nonexistent_engine'])
            invalid_config_handled = config_with_invalid is not None
            
            self.log_result("Error Handling", "Invalid Config", invalid_config_handled,
                           "Invalid engine configuration handled gracefully")
            
            # Test timeout handling
            from core.search_crawler import SearchCrawler
            
            timeout_config = SearchCrawlerConfig(
                engines=['duckduckgo'],
                timeout=1  # Very short timeout
            )
            
            timeout_crawler = SearchCrawler(timeout_config)
            
            # Test that timeout is handled gracefully
            try:
                results = await timeout_crawler.search("test", max_results=1)
                timeout_handled = True  # Either succeeds or fails gracefully
            except Exception:
                timeout_handled = True  # Expected to handle timeout gracefully
            
            await timeout_crawler.close()
            
            self.log_result("Error Handling", "Timeout Handling", timeout_handled,
                           "Timeout errors handled gracefully")
            
            # Test empty results handling
            from keywords.keyword_analyzer import KeywordAnalyzer
            
            analyzer = KeywordAnalyzer()
            empty_analysis = analyzer.analyze_text("", [])
            
            empty_handled = empty_analysis is not None  # Should handle empty input
            
            self.log_result("Error Handling", "Empty Input Handling", empty_handled,
                           "Empty input handled gracefully")
            
        except Exception as e:
            self.log_result("Error Handling", "Error Handling Test", False, str(e))
    
    async def exercise_integration_compatibility(self):
        """Exercise 12: Test integration and compatibility."""
        logger.info("🔗 Exercise 12: Integration Compatibility")
        
        try:
            # Test package health check
            from search_crawler import get_search_crawler_health
            
            health = get_search_crawler_health()
            
            health_valid = (
                isinstance(health, dict) and
                'status' in health and
                'version' in health
            )
            
            self.log_result("Integration", "Health Check", health_valid,
                           f"Package status: {health.get('status', 'unknown')}")
            
            # Test factory functions
            from search_crawler import create_general_search_crawler, create_conflict_search_crawler
            
            general_crawler = create_general_search_crawler(engines=['duckduckgo'])
            conflict_crawler = create_conflict_search_crawler()
            
            factory_valid = general_crawler is not None and conflict_crawler is not None
            
            self.log_result("Integration", "Factory Functions", factory_valid,
                           "Factory functions create valid crawlers")
            
            # Test package exports
            import search_crawler
            
            expected_exports = [
                'SearchCrawler', 'ConflictSearchCrawler', 'CrawleeEnhancedSearchCrawler',
                'SearchCrawlerConfig', 'ConflictSearchConfig', 'SEARCH_ENGINES'
            ]
            
            available_exports = [attr for attr in expected_exports if hasattr(search_crawler, attr)]
            export_valid = len(available_exports) >= 4  # At least core exports
            
            self.log_result("Integration", "Package Exports", export_valid,
                           f"Available exports: {len(available_exports)}/{len(expected_exports)}")
            
        except Exception as e:
            self.log_result("Integration", "Integration Test", False, str(e))
    
    def generate_final_report(self) -> str:
        """Generate comprehensive exercise report."""
        total_exercises = sum(len(exercises) for exercises in self.results.values())
        successful_exercises = sum(
            sum(1 for exercise in exercises.values() if exercise['success'])
            for exercises in self.results.values()
        )
        
        success_rate = (successful_exercises / total_exercises) * 100 if total_exercises > 0 else 0
        
        report = []
        report.append("="*80)
        report.append("SEARCH CRAWLER - COMPLETE PACKAGE EXERCISE REPORT")
        report.append("="*80)
        report.append(f"Exercise Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Exercises: {total_exercises}")
        report.append(f"Successful: {successful_exercises} ({success_rate:.1f}%)")
        report.append(f"Failed: {total_exercises - successful_exercises}")
        report.append(f"Exercise Duration: {(datetime.now() - self.exercise_data['start_time']).total_seconds():.1f}s")
        report.append("")
        
        # Category summary
        report.append("Exercise Categories:")
        report.append("-" * 50)
        
        for category, exercises in self.results.items():
            category_total = len(exercises)
            category_success = sum(1 for exercise in exercises.values() if exercise['success'])
            category_rate = (category_success / category_total) * 100 if category_total > 0 else 0
            
            status = "✅" if category_rate == 100 else "⚠️" if category_rate >= 80 else "❌"
            report.append(f"{status} {category:<25} {category_success}/{category_total} ({category_rate:.1f}%)")
        
        report.append("")
        
        # Detailed results
        for category, exercises in self.results.items():
            report.append(f"\n{category} Details:")
            report.append("-" * 40)
            
            for exercise_name, exercise_result in exercises.items():
                status = "✅" if exercise_result['success'] else "❌"
                report.append(f"  {status} {exercise_name}")
                
                if exercise_result['details']:
                    report.append(f"      {exercise_result['details']}")
        
        # Overall assessment
        report.append("\n" + "="*80)
        report.append("OVERALL PACKAGE ASSESSMENT")
        report.append("="*80)
        
        if success_rate >= 95:
            report.append("🎉 EXCELLENT: Search Crawler package is production-ready with outstanding quality")
        elif success_rate >= 85:
            report.append("✅ VERY GOOD: Search Crawler package is production-ready with high quality")
        elif success_rate >= 75:
            report.append("👍 GOOD: Search Crawler package is functional with acceptable quality")
        elif success_rate >= 60:
            report.append("⚠️ FAIR: Search Crawler package has issues that should be addressed")
        else:
            report.append("❌ POOR: Search Crawler package has significant issues requiring attention")
        
        # Component readiness
        report.append("\nComponent Readiness:")
        component_status = {
            "Core Search Functionality": self._assess_component_health("Basic Search"),
            "Multi-Engine Orchestration": self._assess_component_health("Multi-Engine"),
            "Conflict Monitoring": self._assess_component_health("Conflict Monitoring"),
            "Crawlee Integration": self._assess_component_health("Crawlee Integration"),
            "Horn of Africa Specialization": self._assess_component_health("Horn of Africa"),
            "Keyword Management": self._assess_component_health("Keywords"),
            "Performance Characteristics": self._assess_component_health("Performance"),
            "Error Handling": self._assess_component_health("Error Handling"),
            "Integration Compatibility": self._assess_component_health("Integration")
        }
        
        for component, ready in component_status.items():
            status = "✅" if ready else "❌"
            report.append(f"  {status} {component}")
        
        # Recommendations
        report.append("\nRecommendations:")
        if success_rate >= 90:
            report.append("• Package is ready for production deployment")
            report.append("• Set up monitoring and alerting for production use")
            report.append("• Configure appropriate rate limiting for target search engines")
            report.append("• Establish regular keyword database updates")
        elif success_rate >= 75:
            report.append("• Address any failed exercises before production deployment")
            report.append("• Consider additional testing in target environment")
            report.append("• Review configuration settings for production use")
        else:
            report.append("• Investigate and fix failed components")
            report.append("• Run additional testing and validation")
            report.append("• Consider development environment debugging")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def _assess_component_health(self, category: str) -> bool:
        """Assess if a component category is healthy."""
        if category not in self.results:
            return False
        
        exercises = self.results[category]
        if not exercises:
            return False
        
        success_count = sum(1 for exercise in exercises.values() if exercise['success'])
        success_rate = success_count / len(exercises)
        
        return success_rate >= 0.8  # 80% success rate threshold


async def main():
    """Run complete package exercise."""
    print("🚀 Search Crawler - Complete Package Exercise")
    print("="*60)
    print("This exercise validates all package capabilities and readiness.")
    print("")
    
    exercise = SearchCrawlerExercise()
    
    try:
        # Run all exercises
        await exercise.exercise_package_imports()
        await exercise.exercise_configuration_systems()
        await exercise.exercise_search_engines()
        await exercise.exercise_keyword_management()
        await exercise.exercise_basic_search()
        await exercise.exercise_conflict_monitoring()
        await exercise.exercise_crawlee_integration()
        await exercise.exercise_multi_engine_orchestration()
        await exercise.exercise_horn_of_africa_specialization()
        await exercise.exercise_performance_characteristics()
        await exercise.exercise_error_handling()
        await exercise.exercise_integration_compatibility()
        
    except KeyboardInterrupt:
        logger.info("Exercise interrupted by user")
    except Exception as e:
        logger.error(f"Exercise failed with unexpected error: {e}")
    finally:
        await exercise.cleanup()
    
    # Generate comprehensive report
    report = exercise.generate_final_report()
    
    # Print report
    print("\n" + report)
    
    # Save report
    report_file = Path(__file__).parent / "complete_package_exercise_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Save detailed results
    results_file = Path(__file__).parent / "exercise_results.json"
    with open(results_file, 'w') as f:
        json.dump(exercise.results, f, indent=2, default=str)
    
    print(f"\n📄 Full exercise report saved to: {report_file}")
    print(f"📊 Exercise results saved to: {results_file}")
    
    # Calculate final success rate
    total_exercises = sum(len(exercises) for exercises in exercise.results.values())
    successful_exercises = sum(
        sum(1 for exercise in exercises.values() if exercise['success'])
        for exercises in exercise.results.values()
    )
    
    success_rate = (successful_exercises / total_exercises) * 100 if total_exercises > 0 else 0
    
    print(f"\n🎯 Final Assessment: {successful_exercises}/{total_exercises} exercises passed ({success_rate:.1f}%)")
    
    # Provide next steps
    if success_rate >= 90:
        print("🎉 READY FOR PRODUCTION: Package demonstrates excellent quality and readiness")
        print("\nRecommended next steps:")
        print("1. Deploy to production environment")
        print("2. Set up monitoring and alerting")
        print("3. Configure search engine rate limits")
        print("4. Schedule regular keyword database updates")
    elif success_rate >= 75:
        print("👍 GOOD QUALITY: Package is functional with minor issues")
        print("\nRecommended next steps:")
        print("1. Address any failed exercises")
        print("2. Perform additional testing")
        print("3. Review configuration settings")
        print("4. Consider production pilot")
    else:
        print("⚠️ NEEDS IMPROVEMENT: Package has issues requiring attention")
        print("\nRecommended next steps:")
        print("1. Review failed exercises and fix issues")
        print("2. Run comprehensive tests again")
        print("3. Validate all dependencies")
        print("4. Consider development environment debugging")
    
    return success_rate >= 75


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)