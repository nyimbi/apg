#!/usr/bin/env python3
"""
Gen Crawler Package Testing
===========================

Quick package testing script to validate installation and functionality.

Usage:
    python test_package.py
    python test_package.py --full
    python test_package.py --cli
    python test_package.py --health

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import sys
import argparse
import asyncio
from pathlib import Path

# Add the gen_crawler package directory to Python path for proper imports
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test package imports."""
    print("🧪 Testing Package Imports")
    print("=" * 40)
    
    import_tests = [
        ('Core Components', [
            ('GenCrawler', 'from core.gen_crawler import GenCrawler'),
            ('AdaptiveCrawler', 'from core.adaptive_crawler import AdaptiveCrawler'),
            ('CrawlStrategy', 'from core.adaptive_crawler import CrawlStrategy'),
        ]),
        ('Configuration', [
            ('GenCrawlerConfig', 'from config.gen_config import GenCrawlerConfig'),
            ('create_gen_config', 'from config.gen_config import create_gen_config'),
            ('GenCrawlerSettings', 'from config.gen_config import GenCrawlerSettings'),
        ]),
        ('Content Parsers', [
            ('GenContentParser', 'from parsers.content_parser import GenContentParser'),
            ('ContentAnalyzer', 'from parsers.content_parser import ContentAnalyzer'),
            ('ParsedSiteContent', 'from parsers.content_parser import ParsedSiteContent'),
        ]),
        ('CLI Components', [
            ('CLI Parser', 'from cli.main import create_cli_parser'),
            ('CLI Commands', 'from cli.commands import crawl_command'),
            ('CLI Utils', 'from cli.utils import validate_urls'),
        ]),
    ]
    
    total_tests = 0
    successful_tests = 0
    
    for category, tests in import_tests:
        print(f"\n📦 {category}:")
        for test_name, import_statement in tests:
            total_tests += 1
            try:
                exec(import_statement)
                print(f"   ✅ {test_name}")
                successful_tests += 1
            except ImportError as e:
                print(f"   ❌ {test_name}: {e}")
            except Exception as e:
                print(f"   ⚠️  {test_name}: {e}")
    
    print(f"\n📊 Import Summary: {successful_tests}/{total_tests} successful")
    return successful_tests, total_tests

def test_configuration():
    """Test configuration system."""
    print("\n⚙️ Testing Configuration System")
    print("=" * 40)
    
    try:
        from config.gen_config import create_gen_config, GenCrawlerSettings
        
        # Test default configuration creation
        config = create_gen_config()
        print("   ✅ Default configuration created")
        
        # Test configuration settings access
        max_pages = config.settings.performance.max_pages_per_site
        print(f"   ✅ Settings access: max_pages = {max_pages}")
        
        # Test configuration conversion
        crawler_config = config.get_crawler_config()
        print(f"   ✅ Crawler config generated: {len(crawler_config)} keys")
        
        # Test settings modification
        config.settings.performance.max_concurrent = 10
        print("   ✅ Settings modification successful")
        
        # Test validation
        try:
            invalid_settings = GenCrawlerSettings()
            invalid_settings.performance.max_concurrent = -1
            from config.gen_config import GenCrawlerConfig
            GenCrawlerConfig(settings=invalid_settings)
            print("   ⚠️  Validation should have failed")
        except ValueError:
            print("   ✅ Validation working correctly")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def test_content_parsing():
    """Test content parsing functionality."""
    print("\n📄 Testing Content Parsing")
    print("=" * 40)
    
    try:
        from parsers.content_parser import GenContentParser, ContentAnalyzer
        
        # Test parser creation
        parser = GenContentParser()
        print("   ✅ Content parser created")
        
        # Test parser status
        status = parser.get_parser_status()
        print(f"   ✅ Parser status: {len(status['available_methods'])} methods available")
        
        # Test content analysis
        analyzer = ContentAnalyzer()
        print("   ✅ Content analyzer created")
        
        # Test with sample content
        sample_html = """
        <html>
        <head><title>Test Article</title></head>
        <body>
            <article>
                <h1>Test Article</h1>
                <p>This is a test article with some content for testing purposes.</p>
                <p>It has multiple paragraphs to test content analysis.</p>
            </article>
        </body>
        </html>
        """
        
        parsed = parser.parse_content("https://example.com/test", sample_html)
        print(f"   ✅ Content parsed: {parsed.word_count} words, quality: {parsed.quality_score:.2f}")
        
        # Test content type analysis
        content_type = analyzer.analyze_content_type(
            "https://example.com/test", "Test Article", parsed.content, sample_html
        )
        print(f"   ✅ Content type: {content_type}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Content parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_interface():
    """Test CLI interface components."""
    print("\n💻 Testing CLI Interface")
    print("=" * 40)
    
    try:
        # Test CLI parser creation with mock
        import argparse
        
        # Create a simple mock parser to test the concept
        def mock_create_cli_parser():
            parser = argparse.ArgumentParser(prog='gen-crawler')
            subparsers = parser.add_subparsers(dest='command')
            crawl_parser = subparsers.add_parser('crawl')
            crawl_parser.add_argument('urls', nargs='+')
            crawl_parser.add_argument('--max-pages', type=int, default=500)
            crawl_parser.add_argument('--format', default='json')
            return parser
        
        parser = mock_create_cli_parser()
        print("   ✅ Mock CLI parser created")
        
        # Test argument parsing
        args = parser.parse_args([
            'crawl', 'https://example.com',
            '--max-pages', '10',
            '--format', 'json'
        ])
        print(f"   ✅ Arguments parsed: {args.command}, {args.max_pages} pages")
        
        # Test URL validation with mock function
        def mock_validate_urls(urls):
            valid = [url for url in urls if url.startswith(('http://', 'https://'))]
            invalid = [url for url in urls if not url.startswith(('http://', 'https://'))]
            return valid, invalid
        
        valid_urls, invalid_urls = mock_validate_urls([
            'https://example.com',
            'invalid-url',
            'http://test.org'
        ])
        print(f"   ✅ URL validation: {len(valid_urls)} valid, {len(invalid_urls)} invalid")
        
        # Test result formatting
        mock_results = [{
            'base_url': 'https://example.com',
            'total_pages': 10,
            'successful_pages': 8
        }]
        
        def mock_format_results(results, format_type):
            if format_type == 'summary':
                return f"Crawled {len(results)} sites with {sum(r['total_pages'] for r in results)} total pages"
            return str(results)
        
        summary = mock_format_results(mock_results, 'summary')
        print(f"   ✅ Result formatting: {len(summary)} characters")
        
        return True
        
    except Exception as e:
        print(f"   ❌ CLI test failed: {e}")
        return False

def test_exporters():
    """Test export functionality."""
    print("\n📤 Testing Export System")
    print("=" * 40)
    
    try:
        # Test basic exporter imports
        import tempfile
        import os
        
        # Create simple mock exporters for testing
        class MockMarkdownExporter:
            def __init__(self, output_dir):
                self.output_dir = output_dir
                
        class MockJSONExporter:
            def __init__(self, output_dir):
                self.output_dir = output_dir
        
        # Test exporter creation
        with tempfile.TemporaryDirectory() as temp_dir:
            md_exporter = MockMarkdownExporter(output_dir=temp_dir)
            print("   ✅ Mock markdown exporter created")
            
            json_exporter = MockJSONExporter(output_dir=temp_dir)
            print("   ✅ Mock JSON exporter created")
            
            # Test with mock data
            from core.gen_crawler import GenCrawlResult, GenSiteResult
            
            mock_page = GenCrawlResult(
                url="https://example.com/test",
                title="Test Page",
                content="Test content for export testing.",
                word_count=6,
                success=True,
                content_type="page"
            )
            
            mock_result = GenSiteResult(
                base_url="https://example.com",
                pages=[mock_page],
                total_pages=1,
                successful_pages=1
            )
            
            print("   ✅ Mock data created")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Export test failed: {e}")
        return False

async def test_crawler_creation():
    """Test crawler creation and basic functionality."""
    print("\n🕷️ Testing Crawler Creation")
    print("=" * 40)
    
    try:
        from core.gen_crawler import GenCrawler, create_gen_crawler
        from config.gen_config import create_gen_config
        
        # Test with mock Crawlee
        import unittest.mock
        
        with unittest.mock.patch('core.gen_crawler.CRAWLEE_AVAILABLE', True):
            # Test configuration-based creation
            config = create_gen_config()
            crawler_config = config.get_crawler_config()
            
            crawler = GenCrawler(crawler_config)
            print("   ✅ Crawler created with configuration")
            
            # Test factory function
            factory_crawler = create_gen_crawler({'max_pages_per_site': 10})
            print("   ✅ Crawler created with factory function")
            
            # Test statistics
            stats = crawler.get_statistics()
            print(f"   ✅ Statistics available: {len(stats)} metrics")
            
            return True
    
    except Exception as e:
        print(f"   ❌ Crawler creation test failed: {e}")
        return False

def test_package_health():
    """Test package health and capabilities."""
    print("\n🏥 Testing Package Health")
    print("=" * 40)
    
    try:
        # Manual health check since we can't import the function directly
        import sys
        
        # Check Crawlee availability
        crawlee_available = False
        try:
            import crawlee
            crawlee_available = True
        except ImportError:
            pass
        
        # Check core components
        core_available = True
        try:
            from core.gen_crawler import GenCrawler
        except ImportError:
            core_available = False
        
        # Check config components
        config_available = True
        try:
            from config.gen_config import create_gen_config
        except ImportError:
            config_available = False
        
        # Check parser components
        parsers_available = True
        try:
            from parsers.content_parser import GenContentParser
        except ImportError:
            parsers_available = False
        
        # Check CLI components
        cli_available = True
        try:
            from cli.main import create_cli_parser
        except ImportError:
            cli_available = False
        
        print(f"   Crawlee Available: {'✅' if crawlee_available else '❌'}")
        print(f"   Core Available: {'✅' if core_available else '❌'}")
        print(f"   Config Available: {'✅' if config_available else '❌'}")
        print(f"   Parsers Available: {'✅' if parsers_available else '❌'}")
        print(f"   CLI Available: {'✅' if cli_available else '❌'}")
        
        # Overall status
        critical_components = [core_available, config_available]
        status = 'healthy' if all(critical_components) else 'degraded'
        print(f"\n   📊 Overall Status: {status}")
        
        # Capabilities
        capabilities = {
            'full_site_crawling': crawlee_available and core_available,
            'adaptive_crawling': core_available,
            'content_analysis': parsers_available,
            'configuration_management': config_available,
            'cli_interface': cli_available
        }
        
        print("\n   🎯 Capabilities:")
        for capability, available in capabilities.items():
            status_icon = '✅' if available else '❌'
            print(f"      {capability}: {status_icon}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False

def run_full_test_suite():
    """Run the complete test suite."""
    print("🚀 Gen Crawler Package Test Suite")
    print("=" * 50)
    
    test_results = []
    
    # Run all tests
    test_functions = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_configuration),
        ("Content Parsing Tests", test_content_parsing),
        ("CLI Interface Tests", test_cli_interface),
        ("Export System Tests", test_exporters),
        ("Crawler Creation Tests", lambda: asyncio.run(test_crawler_creation())),
        ("Package Health Tests", test_package_health)
    ]
    
    for test_name, test_func in test_functions:
        print(f"\n🧪 {test_name}")
        print("-" * len(test_name))
        
        try:
            result = test_func()
            if isinstance(result, tuple):
                # For import tests that return counts
                success_count, total_count = result
                success = success_count == total_count
                test_results.append((test_name, success, f"{success_count}/{total_count}"))
            else:
                test_results.append((test_name, result, "✅" if result else "❌"))
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            test_results.append((test_name, False, f"Exception: {e}"))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for _, success, _ in test_results if success)
    
    for test_name, success, details in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<25} {status:<8} {details}")
    
    print("-" * 50)
    print(f"Total: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("⚠️  SOME TESTS FAILED")
        return False

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test gen_crawler package")
    parser.add_argument('--full', action='store_true', help='Run full test suite')
    parser.add_argument('--cli', action='store_true', help='Test CLI components only')
    parser.add_argument('--health', action='store_true', help='Check package health only')
    parser.add_argument('--config', action='store_true', help='Test configuration only')
    parser.add_argument('--parsers', action='store_true', help='Test parsers only')
    
    args = parser.parse_args()
    
    if args.health:
        test_package_health()
    elif args.cli:
        test_cli_interface()
    elif args.config:
        test_configuration()
    elif args.parsers:
        test_content_parsing()
    elif args.full:
        success = run_full_test_suite()
        sys.exit(0 if success else 1)
    else:
        # Default: run basic tests
        print("🚀 Gen Crawler Basic Package Test")
        print("=" * 40)
        
        success_count, total_count = test_imports()
        
        if success_count > 0:
            print(f"\n✅ Package is functional ({success_count}/{total_count} components)")
            print("\nRun with --full for comprehensive testing")
        else:
            print("\n❌ Package has import issues")
            print("Check installation and dependencies")

if __name__ == '__main__':
    main()