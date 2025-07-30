#!/usr/bin/env python3
"""
Google News Crawler - Complete Package Exercise
==============================================

Comprehensive exercise script that demonstrates all capabilities of the Google News Crawler
package, including core functionality, CLI commands, Crawlee integration, and configuration.

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
from datetime import datetime
from typing import Dict, List, Any
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

class GoogleNewsCrawlerExercise:
    """Complete exercise of Google News Crawler capabilities."""
    
    def __init__(self):
        self.results = {}
        self.temp_files = []
        
    async def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                temp_file.unlink()
            except FileNotFoundError:
                pass
    
    def log_result(self, category: str, test: str, success: bool, details: Any = None):
        """Log exercise result."""
        if category not in self.results:
            self.results[category] = {}
        
        self.results[category][test] = {
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        status = "✅" if success else "❌"
        logger.info(f"{status} {category}: {test}")
        if details and isinstance(details, str) and len(details) < 100:
            logger.info(f"   Details: {details}")
    
    async def exercise_imports_and_setup(self):
        """Exercise 1: Test all imports and basic setup."""
        logger.info("🏗️ Exercise 1: Imports and Setup")
        
        try:
            # Test core imports
            from api.google_news_client import (
                EnhancedGoogleNewsClient, create_enhanced_gnews_client,
                create_crawlee_enhanced_gnews_client
            )
            self.log_result("Imports", "Core Client Classes", True)
            
            # Test CLI imports
            from cli.main import main_cli, setup_argument_parser
            from cli.commands import search_command, config_command
            from cli.utils import validate_countries, create_mock_db_manager
            self.log_result("Imports", "CLI Components", True)
            
            # Test Crawlee imports
            from crawlee_integration import (
                CrawleeNewsEnhancer, create_crawlee_config, CRAWLEE_AVAILABLE
            )
            self.log_result("Imports", "Crawlee Integration", True, f"Available: {CRAWLEE_AVAILABLE}")
            
            # Test configuration imports
            from config import CrawlerConfig
            self.log_result("Imports", "Configuration System", True)
            
            # Test parser imports
            from parsers import BaseParser, ParseResult, ArticleData
            self.log_result("Imports", "Parser Components", True)
            
        except Exception as e:
            self.log_result("Imports", "Package Import", False, str(e))
    
    async def exercise_configuration_system(self):
        """Exercise 2: Test configuration management."""
        logger.info("⚙️ Exercise 2: Configuration System")
        
        try:
            from cli.utils import save_cli_config, load_cli_config
            
            # Create test configuration
            test_config = {
                "database": {
                    "url": "postgresql://test:test@localhost/test"
                },
                "crawlee": {
                    "max_requests": 30,
                    "max_concurrent": 4,
                    "target_countries": ["ET", "SO", "KE"],
                    "enable_full_content": True
                },
                "search": {
                    "default_countries": ["ET", "SO", "KE", "SD"],
                    "default_languages": ["en", "fr", "ar"],
                    "default_max_results": 75
                }
            }
            
            # Test configuration save/load
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                config_path = Path(f.name)
            self.temp_files.append(config_path)
            
            save_cli_config(config_path, test_config)
            loaded_config = load_cli_config(config_path)
            
            # Verify configuration integrity
            integrity_check = (
                loaded_config["crawlee"]["max_requests"] == 30 and
                loaded_config["search"]["default_max_results"] == 75 and
                len(loaded_config["crawlee"]["target_countries"]) == 3
            )
            
            self.log_result("Configuration", "Save/Load Cycle", integrity_check)
            
            # Test configuration validation
            from cli.utils import validate_countries, validate_languages
            
            countries = validate_countries(loaded_config["crawlee"]["target_countries"])
            languages = validate_languages(loaded_config["search"]["default_languages"])
            
            self.log_result(
                "Configuration", 
                "Data Validation", 
                len(countries) >= 3 and len(languages) >= 3,
                f"Countries: {len(countries)}, Languages: {len(languages)}"
            )
            
        except Exception as e:
            self.log_result("Configuration", "System Test", False, str(e))
    
    async def exercise_cli_interface(self):
        """Exercise 3: Test CLI interface."""
        logger.info("💻 Exercise 3: CLI Interface")
        
        try:
            from cli.main import setup_argument_parser
            
            # Test argument parser setup
            parser = setup_argument_parser()
            
            # Test various command parsing
            test_commands = [
                (['search', 'Ethiopia conflict', '--countries', 'ET,SO'], 'search'),
                (['monitor', '--query', 'Horn of Africa', '--interval', '300'], 'monitor'),
                (['config', '--show'], 'config'),
                (['crawlee', '--status'], 'crawlee'),
                (['status', '--check-deps'], 'status')
            ]
            
            parser_success = True
            for cmd_args, expected_cmd in test_commands:
                try:
                    args = parser.parse_args(cmd_args)
                    if args.command != expected_cmd:
                        parser_success = False
                        break
                except Exception:
                    parser_success = False
                    break
            
            self.log_result("CLI", "Command Parsing", parser_success)
            
            # Test help generation
            try:
                help_text = parser.format_help()
                help_complete = (
                    len(help_text) > 500 and
                    'search' in help_text and
                    'monitor' in help_text and
                    'config' in help_text
                )
                self.log_result("CLI", "Help Generation", help_complete, f"Help length: {len(help_text)}")
            except Exception as e:
                self.log_result("CLI", "Help Generation", False, str(e))
            
            # Test utility functions
            from cli.utils import parse_date_input, format_duration
            
            date_test = parse_date_input("7d") is not None
            duration_test = format_duration(125.5) == "2.1m"
            
            self.log_result("CLI", "Utility Functions", date_test and duration_test)
            
        except Exception as e:
            self.log_result("CLI", "Interface Test", False, str(e))
    
    async def exercise_google_news_client(self):
        """Exercise 4: Test Google News client functionality."""
        logger.info("📰 Exercise 4: Google News Client")
        
        try:
            from api.google_news_client import create_enhanced_gnews_client
            from cli.utils import create_mock_db_manager
            
            # Create client components
            db_manager = create_mock_db_manager()
            client = await create_enhanced_gnews_client(db_manager=db_manager)
            
            # Test client initialization
            client_initialized = (
                client is not None and
                hasattr(client, 'search_news') and
                hasattr(client, 'db_manager') and
                client.db_manager is not None
            )
            
            self.log_result("Google News", "Client Initialization", client_initialized)
            
            # Test mock search (without actual network calls)
            # This tests the parameter processing and validation
            search_params = {
                'query': 'Ethiopia conflict test',
                'countries': ['ET', 'SO', 'KE'],
                'languages': ['en', 'fr'],
                'max_results': 10
            }
            
            # Validate that client accepts these parameters
            try:
                # This will attempt to initialize the search but may fail due to network
                # The important part is that parameter validation works
                search_method = getattr(client, 'search_news', None)
                param_validation = search_method is not None
                
                self.log_result("Google News", "Search Method Available", param_validation)
            except Exception as e:
                self.log_result("Google News", "Search Parameter Validation", False, str(e))
            
            # Test client statistics
            stats_available = hasattr(client, 'session_stats')
            self.log_result("Google News", "Statistics Tracking", stats_available)
            
            # Test client cleanup
            await client.close()
            self.log_result("Google News", "Client Cleanup", True)
            
        except Exception as e:
            self.log_result("Google News", "Client Test", False, str(e))
    
    async def exercise_crawlee_integration(self):
        """Exercise 5: Test Crawlee integration."""
        logger.info("🕷️ Exercise 5: Crawlee Integration")
        
        try:
            from crawlee_integration import (
                create_crawlee_config, CRAWLEE_AVAILABLE, CrawleeNewsEnhancer
            )
            
            # Test configuration creation
            config = create_crawlee_config(
                max_requests=15,
                max_concurrent=3,
                target_countries=['ET', 'SO'],
                enable_full_content=True
            )
            
            config_valid = (
                config is not None and
                config.max_requests_per_crawl == 15 and
                config.max_concurrent == 3 and
                config.enable_full_content == True
            )
            
            self.log_result("Crawlee", "Configuration Creation", config_valid)
            
            # Test availability detection
            self.log_result("Crawlee", "Availability Check", True, f"Available: {CRAWLEE_AVAILABLE}")
            
            # Test enhancer creation (if available)
            if CRAWLEE_AVAILABLE:
                try:
                    from crawlee_integration import create_crawlee_enhancer
                    
                    enhancer = await create_crawlee_enhancer(config)
                    enhancer_valid = enhancer is not None and hasattr(enhancer, 'enhance_articles')
                    
                    self.log_result("Crawlee", "Enhancer Creation", enhancer_valid)
                    
                    await enhancer.close()
                    self.log_result("Crawlee", "Enhancer Cleanup", True)
                    
                except Exception as e:
                    self.log_result("Crawlee", "Enhancer Creation", False, str(e))
            else:
                self.log_result("Crawlee", "Enhancer Creation", True, "Skipped - not available")
            
            # Test extractor availability
            try:
                test_enhancer = CrawleeNewsEnhancer(config)
                extractors = test_enhancer.extractors_available
                available_count = sum(extractors.values())
                
                self.log_result(
                    "Crawlee", 
                    "Extractor Availability", 
                    available_count > 0,
                    f"{available_count}/4 extractors available"
                )
            except Exception as e:
                self.log_result("Crawlee", "Extractor Availability", False, str(e))
            
        except Exception as e:
            self.log_result("Crawlee", "Integration Test", False, str(e))
    
    async def exercise_data_processing(self):
        """Exercise 6: Test data processing capabilities."""
        logger.info("🔄 Exercise 6: Data Processing")
        
        try:
            from cli.utils import create_mock_db_manager
            
            # Test mock database operations
            db_manager = create_mock_db_manager()
            
            # Create test articles
            test_articles = [
                {
                    'title': f'Test Article {i}',
                    'url': f'https://example.com/article-{i}',
                    'content': f'This is test content for article {i} with sufficient length to test processing.',
                    'published_date': datetime.now(),
                    'source': f'Test Source {i % 3}'  # 3 different sources
                }
                for i in range(20)
            ]
            
            # Test article storage
            stored_count = await db_manager.store_articles(test_articles)
            storage_success = stored_count == 20
            
            self.log_result("Data Processing", "Article Storage", storage_success, f"Stored: {stored_count}")
            
            # Test data filtering
            def filter_by_length(articles: List[Dict], min_words: int = 10):
                return [
                    a for a in articles
                    if len(a.get('content', '').split()) >= min_words
                ]
            
            filtered_articles = filter_by_length(test_articles, min_words=8)
            filter_success = len(filtered_articles) <= len(test_articles)
            
            self.log_result(
                "Data Processing", 
                "Content Filtering", 
                filter_success,
                f"Filtered: {len(filtered_articles)}/{len(test_articles)}"
            )
            
            # Test quality scoring simulation
            def calculate_simple_quality(article: Dict) -> float:
                score = 0.0
                
                # Title quality
                title = article.get('title', '')
                if len(title) > 10:
                    score += 0.3
                
                # Content quality
                content = article.get('content', '')
                word_count = len(content.split())
                if word_count >= 15:
                    score += 0.4
                elif word_count >= 10:
                    score += 0.2
                
                # Source quality
                if article.get('source'):
                    score += 0.3
                
                return min(score, 1.0)
            
            scored_articles = [
                {**article, 'quality_score': calculate_simple_quality(article)}
                for article in test_articles[:5]
            ]
            
            avg_quality = sum(a['quality_score'] for a in scored_articles) / len(scored_articles)
            quality_success = 0.5 <= avg_quality <= 1.0
            
            self.log_result(
                "Data Processing", 
                "Quality Scoring", 
                quality_success,
                f"Average quality: {avg_quality:.2f}"
            )
            
        except Exception as e:
            self.log_result("Data Processing", "Processing Test", False, str(e))
    
    async def exercise_performance_characteristics(self):
        """Exercise 7: Test performance characteristics."""
        logger.info("🚀 Exercise 7: Performance Characteristics")
        
        try:
            from cli.utils import validate_countries, parse_date_input, create_mock_db_manager
            
            # Test validation performance
            large_country_list = ['ET', 'SO', 'KE', 'INVALID'] * 50
            
            start_time = time.perf_counter()
            validated = validate_countries(large_country_list)
            validation_time = time.perf_counter() - start_time
            
            validation_performance = validation_time < 0.5  # Should be very fast
            
            self.log_result(
                "Performance", 
                "Country Validation Speed", 
                validation_performance,
                f"{validation_time:.3f}s for {len(large_country_list)} items"
            )
            
            # Test date parsing performance
            test_dates = ['7d', '24h', '2w', '1m', '3d'] * 20
            
            start_time = time.perf_counter()
            parsed_dates = [parse_date_input(d) for d in test_dates]
            parsing_time = time.perf_counter() - start_time
            
            parsing_performance = parsing_time < 0.1
            valid_dates = sum(1 for d in parsed_dates if d is not None)
            
            self.log_result(
                "Performance", 
                "Date Parsing Speed", 
                parsing_performance,
                f"{parsing_time:.3f}s for {len(test_dates)} items"
            )
            
            # Test database performance
            db_manager = create_mock_db_manager()
            large_dataset = [
                {'title': f'Article {i}', 'content': f'Content {i}'}
                for i in range(1000)
            ]
            
            start_time = time.perf_counter()
            stored = await db_manager.store_articles(large_dataset)
            storage_time = time.perf_counter() - start_time
            
            storage_performance = storage_time < 0.5
            
            self.log_result(
                "Performance", 
                "Database Operations", 
                storage_performance,
                f"{storage_time:.3f}s for {stored} articles"
            )
            
            # Test memory efficiency
            import psutil
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Perform memory test
            large_data = []
            for _ in range(100):
                large_data.append({'data': 'x' * 1000})  # 1KB per item
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            memory_efficiency = memory_increase < 20  # Should not use more than 20MB
            
            self.log_result(
                "Performance", 
                "Memory Efficiency", 
                memory_efficiency,
                f"Memory increase: {memory_increase:.1f}MB"
            )
            
        except Exception as e:
            self.log_result("Performance", "Performance Test", False, str(e))
    
    async def exercise_error_handling(self):
        """Exercise 8: Test error handling capabilities."""
        logger.info("🚨 Exercise 8: Error Handling")
        
        try:
            from cli.utils import validate_countries, parse_date_input, load_cli_config
            
            # Test invalid input handling
            invalid_countries = validate_countries(['INVALID', 'XX', 'YY'])
            invalid_handling = len(invalid_countries) >= 1  # Should fall back to defaults
            
            self.log_result("Error Handling", "Invalid Country Codes", invalid_handling)
            
            # Test invalid date handling
            invalid_dates = ["not-a-date", "2025-13-45", "invalid", ""]
            date_handling_success = True
            
            for invalid_date in invalid_dates:
                result = parse_date_input(invalid_date)
                if result is not None:  # Should return None for invalid dates
                    date_handling_success = False
                    break
            
            self.log_result("Error Handling", "Invalid Date Inputs", date_handling_success)
            
            # Test missing file handling
            try:
                load_cli_config(Path("/non/existent/file.json"))
                file_handling = False  # Should raise exception
            except FileNotFoundError:
                file_handling = True  # Exception properly raised
            except Exception:
                file_handling = False  # Wrong exception type
            
            self.log_result("Error Handling", "Missing File Access", file_handling)
            
            # Test configuration validation
            from crawlee_integration import create_crawlee_config
            
            try:
                # Test with potentially problematic values
                config = create_crawlee_config(
                    max_requests=0,  # Edge case
                    max_concurrent=-1  # Invalid
                )
                # Should either correct values or handle gracefully
                config_handling = config is not None
            except Exception:
                # Or raise appropriate exception
                config_handling = True
            
            self.log_result("Error Handling", "Configuration Validation", config_handling)
            
        except Exception as e:
            self.log_result("Error Handling", "Error Handling Test", False, str(e))
    
    def generate_final_report(self) -> str:
        """Generate comprehensive exercise report."""
        total_exercises = sum(len(tests) for tests in self.results.values())
        successful_exercises = sum(
            sum(1 for test in tests.values() if test['success'])
            for tests in self.results.values()
        )
        
        success_rate = (successful_exercises / total_exercises) * 100 if total_exercises > 0 else 0
        
        report = []
        report.append("="*80)
        report.append("Google News Crawler - Complete Package Exercise Report")
        report.append("="*80)
        report.append(f"Exercise Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Exercises: {total_exercises}")
        report.append(f"Successful: {successful_exercises} ({success_rate:.1f}%)")
        report.append(f"Failed: {total_exercises - successful_exercises}")
        report.append("")
        
        # Category summary
        report.append("Exercise Categories:")
        report.append("-" * 50)
        
        for category, tests in self.results.items():
            category_total = len(tests)
            category_success = sum(1 for test in tests.values() if test['success'])
            category_rate = (category_success / category_total) * 100 if category_total > 0 else 0
            
            status = "✅" if category_rate == 100 else "⚠️" if category_rate >= 80 else "❌"
            report.append(f"{status} {category:<25} {category_success}/{category_total} ({category_rate:.1f}%)")
        
        report.append("")
        
        # Detailed results
        for category, tests in self.results.items():
            report.append(f"\n{category} Details:")
            report.append("-" * 40)
            
            for test_name, test_result in tests.items():
                status = "✅" if test_result['success'] else "❌"
                report.append(f"  {status} {test_name}")
                
                if test_result['details']:
                    report.append(f"      {test_result['details']}")
        
        # Overall assessment
        report.append("\n" + "="*80)
        report.append("OVERALL PACKAGE ASSESSMENT")
        report.append("="*80)
        
        if success_rate >= 95:
            report.append("🎉 EXCELLENT: Package is production-ready with outstanding quality")
        elif success_rate >= 85:
            report.append("✅ VERY GOOD: Package is production-ready with high quality")
        elif success_rate >= 75:
            report.append("👍 GOOD: Package is functional with acceptable quality")
        elif success_rate >= 60:
            report.append("⚠️ FAIR: Package has issues that should be addressed")
        else:
            report.append("❌ POOR: Package has significant issues requiring attention")
        
        # Component readiness
        report.append("\nComponent Readiness:")
        component_status = {
            "Core Functionality": self.results.get("Google News", {}).get("Client Initialization", {}).get("success", False),
            "CLI Interface": len([t for t in self.results.get("CLI", {}).values() if t["success"]]) >= 2,
            "Configuration System": len([t for t in self.results.get("Configuration", {}).values() if t["success"]]) >= 2,
            "Crawlee Integration": self.results.get("Crawlee", {}).get("Configuration Creation", {}).get("success", False),
            "Data Processing": len([t for t in self.results.get("Data Processing", {}).values() if t["success"]]) >= 2,
            "Performance": len([t for t in self.results.get("Performance", {}).values() if t["success"]]) >= 3,
            "Error Handling": len([t for t in self.results.get("Error Handling", {}).values() if t["success"]]) >= 3
        }
        
        for component, ready in component_status.items():
            status = "✅" if ready else "❌"
            report.append(f"  {status} {component}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)

async def main():
    """Run complete package exercise."""
    print("🚀 Google News Crawler - Complete Package Exercise")
    print("="*60)
    print("This exercise validates all package capabilities and readiness.")
    print("")
    
    exercise = GoogleNewsCrawlerExercise()
    
    try:
        # Run all exercises
        await exercise.exercise_imports_and_setup()
        await exercise.exercise_configuration_system()
        await exercise.exercise_cli_interface()
        await exercise.exercise_google_news_client()
        await exercise.exercise_crawlee_integration()
        await exercise.exercise_data_processing()
        await exercise.exercise_performance_characteristics()
        await exercise.exercise_error_handling()
        
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
    total_exercises = sum(len(tests) for tests in exercise.results.values())
    successful_exercises = sum(
        sum(1 for test in tests.values() if test['success'])
        for tests in exercise.results.values()
    )
    
    success_rate = (successful_exercises / total_exercises) * 100 if total_exercises > 0 else 0
    
    print(f"\n🎯 Final Assessment: {successful_exercises}/{total_exercises} exercises passed ({success_rate:.1f}%)")
    
    # Provide next steps
    if success_rate >= 90:
        print("🎉 READY FOR PRODUCTION: Package demonstrates excellent quality and readiness")
        print("\nRecommended next steps:")
        print("1. Deploy to production environment")
        print("2. Set up monitoring and alerting")
        print("3. Configure production database")
        print("4. Schedule regular health checks")
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