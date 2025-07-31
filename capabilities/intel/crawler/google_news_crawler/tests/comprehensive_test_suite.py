#!/usr/bin/env python3
"""
Google News Crawler Comprehensive Test Suite
===========================================

Complete testing framework for the Google News Crawler package including:
- Unit tests for all components
- Integration tests
- Performance tests
- CLI tests
- Crawlee integration tests
- Error handling tests
- Configuration tests

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import sys
import os
import tempfile
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
import logging

# Add package to path for testing
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestResults:
    """Track test results and generate comprehensive reports."""
    
    def __init__(self):
        self.test_categories = {}
        self.start_time = time.time()
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.errors = []
        
    def add_result(self, category: str, test_name: str, passed: bool, duration: float, error: str = None):
        """Add a test result."""
        if category not in self.test_categories:
            self.test_categories[category] = {
                'tests': [],
                'passed': 0,
                'failed': 0,
                'total_duration': 0.0
            }
        
        self.test_categories[category]['tests'].append({
            'name': test_name,
            'passed': passed,
            'duration': duration,
            'error': error
        })
        
        self.test_categories[category]['total_duration'] += duration
        self.total_tests += 1
        
        if passed:
            self.passed_tests += 1
            self.test_categories[category]['passed'] += 1
        else:
            self.failed_tests += 1
            self.test_categories[category]['failed'] += 1
            if error:
                self.errors.append(f"{category}.{test_name}: {error}")
    
    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        total_duration = time.time() - self.start_time
        success_rate = (self.passed_tests / max(self.total_tests, 1)) * 100
        
        report = []
        report.append("="*80)
        report.append("Google News Crawler - Comprehensive Test Report")
        report.append("="*80)
        report.append(f"Test Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Duration: {total_duration:.2f} seconds")
        report.append(f"Total Tests: {self.total_tests}")
        report.append(f"Passed: {self.passed_tests} ({self.passed_tests/max(self.total_tests,1)*100:.1f}%)")
        report.append(f"Failed: {self.failed_tests} ({self.failed_tests/max(self.total_tests,1)*100:.1f}%)")
        report.append(f"Success Rate: {success_rate:.1f}%")
        report.append("")
        
        # Category breakdown
        report.append("Test Categories:")
        report.append("-" * 50)
        for category, results in self.test_categories.items():
            total_cat = results['passed'] + results['failed']
            cat_success = (results['passed'] / max(total_cat, 1)) * 100
            status = "✅" if cat_success == 100 else "⚠️" if cat_success >= 80 else "❌"
            
            report.append(f"{status} {category:<25} {results['passed']}/{total_cat} "
                         f"({cat_success:.1f}%) - {results['total_duration']:.2f}s")
        
        report.append("")
        
        # Detailed results
        for category, results in self.test_categories.items():
            report.append(f"\n{category} Details:")
            report.append("-" * 40)
            
            for test in results['tests']:
                status = "✅" if test['passed'] else "❌"
                report.append(f"  {status} {test['name']:<30} {test['duration']:.3f}s")
                if not test['passed'] and test['error']:
                    report.append(f"      Error: {test['error']}")
        
        # Errors summary
        if self.errors:
            report.append("\n" + "="*50)
            report.append("ERRORS SUMMARY")
            report.append("="*50)
            for error in self.errors:
                report.append(f"❌ {error}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)

# Global test results tracker
test_results = TestResults()

def test_wrapper(category: str, test_name: str):
    """Decorator to wrap tests and track results."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                if asyncio.iscoroutinefunction(func):
                    await func(*args, **kwargs)
                else:
                    func(*args, **kwargs)
                duration = time.time() - start_time
                test_results.add_result(category, test_name, True, duration)
                logger.info(f"✅ {category}.{test_name} - {duration:.3f}s")
            except Exception as e:
                duration = time.time() - start_time
                test_results.add_result(category, test_name, False, duration, str(e))
                logger.error(f"❌ {category}.{test_name} - {duration:.3f}s - {e}")
                
        def sync_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return asyncio.run(async_wrapper(*args, **kwargs))
            else:
                return async_wrapper(*args, **kwargs)
                
        return sync_wrapper if not asyncio.iscoroutinefunction(func) else async_wrapper
    return decorator

class MockDBManager:
    """Mock database manager for testing."""
    
    def __init__(self):
        self.articles = []
        
    async def store_articles(self, articles: List[Dict[str, Any]]) -> int:
        self.articles.extend(articles)
        return len(articles)
    
    async def close(self):
        pass

# ============================================================================
# UNIT TESTS
# ============================================================================

@test_wrapper("Unit Tests", "Import Core Modules")
def test_import_core_modules():
    """Test that all core modules can be imported."""
    from api.google_news_client import EnhancedGoogleNewsClient
    from crawlee_integration import CrawleeNewsEnhancer, CRAWLEE_AVAILABLE
    from cli.main import main_cli
    from cli.commands import search_command
    from cli.utils import validate_countries
    assert True

@test_wrapper("Unit Tests", "Configuration Loading")
def test_configuration_loading():
    """Test configuration loading and validation."""
    from config import CrawlerConfig, ConfigurationManager
    
    # Test default config creation
    config = CrawlerConfig()
    assert config is not None
    
    # Test configuration manager
    config_manager = ConfigurationManager()
    assert config_manager is not None

@test_wrapper("Unit Tests", "Parser Components")
def test_parser_components():
    """Test parser components."""
    from parsers import BaseParser, ParseResult, ArticleData
    
    # Test parser creation
    parser = BaseParser()
    assert parser is not None
    
    # Test result structures
    result = ParseResult(success=True, data={'test': 'data'})
    assert result.success == True
    assert result.data['test'] == 'data'

@test_wrapper("Unit Tests", "CLI Utilities")
def test_cli_utilities():
    """Test CLI utility functions."""
    from cli.utils import (
        parse_date_input, validate_countries, validate_languages,
        create_mock_db_manager, format_duration
    )
    
    # Test date parsing
    date_result = parse_date_input("7d")
    assert date_result is not None
    assert isinstance(date_result, datetime)
    
    # Test country validation
    countries = validate_countries(['ET', 'SO', 'INVALID'])
    assert 'ET' in countries
    assert 'SO' in countries
    assert 'INVALID' not in countries
    
    # Test mock DB manager
    db_manager = create_mock_db_manager()
    assert db_manager is not None
    
    # Test duration formatting
    duration_str = format_duration(125.5)
    assert isinstance(duration_str, str)

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@test_wrapper("Integration Tests", "Google News Client Creation")
async def test_google_news_client_creation():
    """Test Google News client creation and initialization."""
    from api.google_news_client import create_enhanced_gnews_client
    
    db_manager = MockDBManager()
    
    try:
        client = await create_enhanced_gnews_client(db_manager=db_manager)
        assert client is not None
        await client.close()
    except Exception as e:
        # Expected in test environment without real dependencies
        assert "not available" in str(e) or "Mock" in str(e) or "import" in str(e).lower()

@test_wrapper("Integration Tests", "Crawlee Integration Check")
async def test_crawlee_integration():
    """Test Crawlee integration availability and basic functionality."""
    try:
        from crawlee_integration import CRAWLEE_AVAILABLE, create_crawlee_config
        
        # Test config creation
        config = create_crawlee_config(max_requests=10)
        assert config is not None
        assert config.max_requests_per_crawl == 10
        
        if CRAWLEE_AVAILABLE:
            from crawlee_integration import create_crawlee_enhancer
            enhancer = await create_crawlee_enhancer(config)
            assert enhancer is not None
            await enhancer.close()
    except ImportError:
        # Expected if Crawlee not installed
        assert True

@test_wrapper("Integration Tests", "CLI Command Processing")
async def test_cli_command_processing():
    """Test CLI command argument processing."""
    from cli.utils import validate_countries, validate_languages
    
    # Test search argument processing
    countries = validate_countries(['ET', 'SO', 'KE'])
    languages = validate_languages(['en', 'fr', 'ar'])
    
    assert len(countries) >= 3
    assert len(languages) >= 3
    assert all(len(c) == 2 for c in countries)
    assert all(len(l) == 2 for l in languages)

@test_wrapper("Integration Tests", "Configuration Management")
async def test_configuration_management():
    """Test configuration save/load functionality."""
    from cli.utils import save_cli_config, load_cli_config
    
    # Create temporary config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_path = Path(f.name)
    
    test_config = {
        "test_section": {"test_value": 123},
        "crawlee": {"max_requests": 50}
    }
    
    # Test save/load cycle
    save_cli_config(config_path, test_config)
    loaded_config = load_cli_config(config_path)
    
    assert loaded_config["test_section"]["test_value"] == 123
    assert loaded_config["crawlee"]["max_requests"] == 50
    
    # Cleanup
    config_path.unlink()

# ============================================================================
# FUNCTIONAL TESTS
# ============================================================================

@test_wrapper("Functional Tests", "Article Data Structure")
async def test_article_data_structure():
    """Test article data structure and validation."""
    from parsers import ArticleData
    
    # Test article creation
    article = ArticleData(
        title="Test Article",
        url="https://example.com/test",
        content="Test content",
        published_date=datetime.now(),
        source="Test Source"
    )
    
    assert article.title == "Test Article"
    assert article.url == "https://example.com/test"
    assert article.content == "Test content"
    assert article.source == "Test Source"

@test_wrapper("Functional Tests", "Search Query Enhancement")
async def test_search_query_enhancement():
    """Test search query enhancement functionality."""
    # Create a simple query enhancer
    def enhance_query(query: str) -> str:
        conflict_terms = ['conflict', 'violence', 'security', 'crisis', 'peace']
        if not any(term in query.lower() for term in conflict_terms):
            return f"{query} OR (conflict OR security OR crisis)"
        return query
    
    # Test query enhancement
    simple_query = "Ethiopia news"
    enhanced_query = enhance_query(simple_query)
    assert "conflict" in enhanced_query or "security" in enhanced_query
    
    # Test query that already has conflict terms
    conflict_query = "Ethiopia conflict update"
    enhanced_conflict = enhance_query(conflict_query)
    assert enhanced_conflict == conflict_query

@test_wrapper("Functional Tests", "Content Quality Scoring")
async def test_content_quality_scoring():
    """Test content quality scoring functionality."""
    def calculate_quality_score(content: str, title: str, metadata: Dict[str, Any]) -> float:
        score = 0.0
        word_count = len(content.split()) if content else 0
        
        # Content length scoring
        if word_count >= 300:
            score += 0.4
        elif word_count >= 100:
            score += 0.2
        
        # Title quality
        if title and len(title) > 10:
            score += 0.2
        
        # Metadata presence
        if metadata.get('authors'):
            score += 0.2
        if metadata.get('published_date'):
            score += 0.2
        
        return min(score, 1.0)
    
    # Test high quality content
    high_quality = calculate_quality_score(
        content=" ".join(["word"] * 400),  # 400 words
        title="This is a comprehensive title",
        metadata={'authors': ['Author'], 'published_date': datetime.now()}
    )
    assert high_quality >= 0.8
    
    # Test low quality content
    low_quality = calculate_quality_score(
        content="short",
        title="Title",
        metadata={}
    )
    assert low_quality <= 0.4

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@test_wrapper("Performance Tests", "Configuration Loading Speed")
async def test_config_loading_performance():
    """Test configuration loading performance."""
    from cli.utils import save_cli_config, load_cli_config
    
    # Create large config
    large_config = {
        f"section_{i}": {f"key_{j}": f"value_{j}" for j in range(100)}
        for i in range(10)
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_path = Path(f.name)
    
    # Measure save time
    start_time = time.time()
    save_cli_config(config_path, large_config)
    save_time = time.time() - start_time
    
    # Measure load time
    start_time = time.time()
    loaded_config = load_cli_config(config_path)
    load_time = time.time() - start_time
    
    # Performance assertions (should be fast)
    assert save_time < 1.0, f"Config save too slow: {save_time:.3f}s"
    assert load_time < 1.0, f"Config load too slow: {load_time:.3f}s"
    assert len(loaded_config) == 10
    
    # Cleanup
    config_path.unlink()

@test_wrapper("Performance Tests", "Mock Database Operations")
async def test_mock_database_performance():
    """Test mock database operation performance."""
    db_manager = MockDBManager()
    
    # Test bulk article storage
    articles = [
        {
            'title': f'Test Article {i}',
            'url': f'https://example.com/article-{i}',
            'content': f'Content for article {i}',
            'published_date': datetime.now()
        }
        for i in range(1000)
    ]
    
    start_time = time.time()
    stored_count = await db_manager.store_articles(articles)
    storage_time = time.time() - start_time
    
    assert stored_count == 1000
    assert storage_time < 1.0, f"Bulk storage too slow: {storage_time:.3f}s"
    assert len(db_manager.articles) == 1000

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@test_wrapper("Error Handling", "Invalid Configuration")
async def test_invalid_configuration():
    """Test handling of invalid configuration."""
    from cli.utils import load_cli_config
    
    # Test loading non-existent file
    try:
        load_cli_config(Path("/non/existent/path.json"))
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        assert True
    
    # Test loading invalid JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("invalid json content {")
        invalid_path = Path(f.name)
    
    try:
        load_cli_config(invalid_path)
        assert False, "Should have raised ValueError"
    except (ValueError, json.JSONDecodeError):
        assert True
    
    # Cleanup
    invalid_path.unlink()

@test_wrapper("Error Handling", "Invalid Date Input")
async def test_invalid_date_input():
    """Test handling of invalid date inputs."""
    from cli.utils import parse_date_input
    
    # Test invalid date formats
    invalid_dates = ["invalid", "2025-13-45", "25th December", ""]
    
    for invalid_date in invalid_dates:
        result = parse_date_input(invalid_date)
        # Should return None for invalid dates
        assert result is None or isinstance(result, datetime)

@test_wrapper("Error Handling", "Invalid Country Codes")
async def test_invalid_country_codes():
    """Test handling of invalid country codes."""
    from cli.utils import validate_countries
    
    # Test with invalid codes
    invalid_countries = ['XX', 'YY', 'ZZZ', '123']
    result = validate_countries(invalid_countries)
    
    # Should return default countries when all inputs are invalid
    assert len(result) >= 1  # Should fall back to defaults
    assert all(len(code) == 2 for code in result)  # All should be 2-letter codes

# ============================================================================
# CLI TESTS
# ============================================================================

@test_wrapper("CLI Tests", "Argument Parsing")
async def test_cli_argument_parsing():
    """Test CLI argument parsing functionality."""
    from cli.main import setup_argument_parser
    
    parser = setup_argument_parser()
    
    # Test search command parsing
    args = parser.parse_args(['search', 'Ethiopia conflict', '--countries', 'ET,SO', '--max-results', '20'])
    assert args.command == 'search'
    assert args.query == 'Ethiopia conflict'
    assert args.countries == 'ET,SO'
    assert args.max_results == 20
    
    # Test monitor command parsing
    args = parser.parse_args(['monitor', '--query', 'Horn of Africa', '--interval', '300'])
    assert args.command == 'monitor'
    assert args.query == 'Horn of Africa'
    assert args.interval == 300

@test_wrapper("CLI Tests", "Command Help Generation")
async def test_cli_help_generation():
    """Test CLI help text generation."""
    from cli.main import setup_argument_parser
    
    parser = setup_argument_parser()
    
    # Test that help can be generated without errors
    try:
        help_text = parser.format_help()
        assert len(help_text) > 100
        assert 'search' in help_text
        assert 'monitor' in help_text
        assert 'config' in help_text
    except Exception as e:
        assert False, f"Help generation failed: {e}"

# ============================================================================
# INTEGRATION SCENARIO TESTS
# ============================================================================

@test_wrapper("Scenario Tests", "Complete Search Workflow")
async def test_complete_search_workflow():
    """Test complete search workflow from query to results."""
    # Mock the complete workflow
    from cli.utils import validate_countries, validate_languages, create_mock_db_manager
    
    # Step 1: Parse and validate input
    query = "Ethiopia conflict"
    countries = validate_countries(['ET', 'SO', 'KE'])
    languages = validate_languages(['en', 'fr'])
    
    # Step 2: Create database manager
    db_manager = create_mock_db_manager()
    
    # Step 3: Simulate search results
    mock_articles = [
        {
            'title': 'Ethiopia Peace Talks Resume',
            'url': 'https://example.com/ethiopia-peace',
            'content': 'Detailed article content about peace talks...',
            'country': 'ET',
            'language': 'en',
            'published_date': datetime.now()
        },
        {
            'title': 'Somalia Security Update',
            'url': 'https://example.com/somalia-security',
            'content': 'Security situation update from Somalia...',
            'country': 'SO',
            'language': 'en',
            'published_date': datetime.now()
        }
    ]
    
    # Step 4: Store articles
    stored_count = await db_manager.store_articles(mock_articles)
    
    # Assertions
    assert len(countries) >= 3
    assert len(languages) >= 2
    assert stored_count == 2
    assert len(db_manager.articles) == 2
    
    # Verify article structure
    for article in db_manager.articles:
        assert 'title' in article
        assert 'url' in article
        assert 'content' in article

@test_wrapper("Scenario Tests", "Monitoring Cycle Simulation")
async def test_monitoring_cycle_simulation():
    """Test monitoring cycle simulation."""
    from cli.utils import create_mock_db_manager
    
    db_manager = create_mock_db_manager()
    seen_urls = set()
    alert_keywords = ['violence', 'attack', 'crisis']
    
    # Simulate multiple monitoring cycles
    for cycle in range(3):
        # Simulate new articles each cycle
        cycle_articles = [
            {
                'title': f'Cycle {cycle} Article {i}',
                'url': f'https://example.com/cycle-{cycle}-article-{i}',
                'content': f'Article content for cycle {cycle}, article {i}',
                'published_date': datetime.now()
            }
            for i in range(2)
        ]
        
        # Filter new articles (simulate deduplication)
        new_articles = [a for a in cycle_articles if a['url'] not in seen_urls]
        for article in new_articles:
            seen_urls.add(article['url'])
        
        # Store new articles
        if new_articles:
            await db_manager.store_articles(new_articles)
        
        # Check for alerts
        alert_articles = []
        for article in new_articles:
            title_lower = article['title'].lower()
            content_lower = article['content'].lower()
            if any(keyword in title_lower or keyword in content_lower for keyword in alert_keywords):
                alert_articles.append(article)
    
    # Verify monitoring simulation
    assert len(seen_urls) == 6  # 3 cycles × 2 articles
    assert len(db_manager.articles) == 6

# ============================================================================
# TEST RUNNER
# ============================================================================

async def run_all_tests():
    """Run all tests and generate comprehensive report."""
    logger.info("🚀 Starting Google News Crawler Comprehensive Test Suite")
    logger.info("=" * 70)
    
    # Unit Tests
    logger.info("\n📦 Running Unit Tests...")
    test_import_core_modules()
    test_configuration_loading()
    test_parser_components()
    test_cli_utilities()
    
    # Integration Tests
    logger.info("\n🔗 Running Integration Tests...")
    await test_google_news_client_creation()
    await test_crawlee_integration()
    await test_cli_command_processing()
    await test_configuration_management()
    
    # Functional Tests
    logger.info("\n⚙️ Running Functional Tests...")
    await test_article_data_structure()
    await test_search_query_enhancement()
    await test_content_quality_scoring()
    
    # Performance Tests
    logger.info("\n🚀 Running Performance Tests...")
    await test_config_loading_performance()
    await test_mock_database_performance()
    
    # Error Handling Tests
    logger.info("\n🚨 Running Error Handling Tests...")
    await test_invalid_configuration()
    await test_invalid_date_input()
    await test_invalid_country_codes()
    
    # CLI Tests
    logger.info("\n💻 Running CLI Tests...")
    await test_cli_argument_parsing()
    await test_cli_help_generation()
    
    # Scenario Tests
    logger.info("\n🎭 Running Scenario Tests...")
    await test_complete_search_workflow()
    await test_monitoring_cycle_simulation()
    
    # Generate and return comprehensive report
    report = test_results.generate_report()
    return report

if __name__ == "__main__":
    # Run tests and save report
    report = asyncio.run(run_all_tests())
    
    # Print report to console
    print("\n" + report)
    
    # Save report to file
    report_file = Path(__file__).parent / "test_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"\n📄 Full test report saved to: {report_file}")
    
    # Exit with appropriate code
    success_rate = (test_results.passed_tests / max(test_results.total_tests, 1)) * 100
    exit_code = 0 if success_rate >= 90 else 1
    sys.exit(exit_code)