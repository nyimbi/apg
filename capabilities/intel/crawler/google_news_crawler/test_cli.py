#!/usr/bin/env python3
"""
Google News Crawler CLI Test Script
==================================

Simple test script to verify CLI functionality without full installation.

Usage:
    python test_cli.py

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the CLI module to path
cli_path = Path(__file__).parent / 'cli'
sys.path.insert(0, str(cli_path))

async def test_cli_imports():
    """Test that CLI modules can be imported."""
    print("🧪 Testing CLI imports...")
    
    try:
        from cli.main import main_cli
        from cli.commands import search_command, config_command, status_command
        from cli.utils import parse_date_input, validate_countries, create_mock_db_manager
        print("✅ All CLI modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

async def test_utility_functions():
    """Test CLI utility functions."""
    print("\n🔧 Testing utility functions...")
    
    try:
        from cli.utils import (
            parse_date_input, validate_countries, validate_languages,
            create_mock_db_manager, format_duration, extract_domain
        )
        
        # Test date parsing
        date_result = parse_date_input("7d")
        assert date_result is not None, "Date parsing failed"
        print("   ✅ Date parsing works")
        
        # Test country validation
        countries = validate_countries(['ET', 'SO', 'INVALID'])
        assert 'ET' in countries and 'SO' in countries, "Country validation failed"
        print("   ✅ Country validation works")
        
        # Test mock DB manager
        db_manager = create_mock_db_manager()
        articles_stored = await db_manager.store_articles([{'title': 'Test Article'}])
        assert articles_stored == 1, "Mock DB manager failed"
        print("   ✅ Mock DB manager works")
        
        # Test domain extraction
        domain = extract_domain("https://www.bbc.com/news/world")
        assert domain == "www.bbc.com", "Domain extraction failed"
        print("   ✅ Domain extraction works")
        
        print("✅ All utility functions work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Utility test failed: {e}")
        return False

async def test_mock_search():
    """Test mock search functionality."""
    print("\n🔍 Testing mock search...")
    
    try:
        # Simulate search arguments
        class MockArgs:
            query = "Ethiopia conflict"
            countries = "ET,SO,KE"
            languages = "en,fr"
            max_results = 10
            crawlee = False
            export = None
            format = "table"
            source_filter = None
            since = None
            until = None
        
        # Test search command preparation
        from cli.utils import validate_countries, validate_languages, create_mock_db_manager
        
        args = MockArgs()
        countries = [c.strip().upper() for c in args.countries.split(',')]
        languages = [l.strip().lower() for l in args.languages.split(',')]
        
        countries = validate_countries(countries)
        languages = validate_languages(languages)
        
        assert len(countries) >= 2, "Country validation failed"
        assert len(languages) >= 1, "Language validation failed"
        
        # Test mock database
        db_manager = create_mock_db_manager()
        assert db_manager is not None, "Mock DB creation failed"
        
        print("   ✅ Search argument processing works")
        print(f"   📍 Countries: {countries}")
        print(f"   🗣️ Languages: {languages}")
        print("✅ Mock search test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock search test failed: {e}")
        return False

async def test_config_functionality():
    """Test configuration functionality."""
    print("\n⚙️ Testing configuration...")
    
    try:
        from cli.utils import load_cli_config, save_cli_config
        import tempfile
        import json
        
        # Test config save/load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = Path(f.name)
        
        test_config = {
            "test": {"value": 123},
            "crawlee": {"max_requests": 50}
        }
        
        # Save config
        save_cli_config(config_path, test_config)
        assert config_path.exists(), "Config file not created"
        
        # Load config
        loaded_config = load_cli_config(config_path)
        assert loaded_config["test"]["value"] == 123, "Config loading failed"
        
        # Cleanup
        config_path.unlink()
        
        print("   ✅ Configuration save/load works")
        print("✅ Configuration test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def test_crawlee_integration_check():
    """Test Crawlee integration availability check."""
    print("\n🕷️ Testing Crawlee integration check...")
    
    try:
        # Test import check
        try:
            from crawlee_integration import CRAWLEE_AVAILABLE
            crawlee_status = CRAWLEE_AVAILABLE
            print(f"   🕷️ Crawlee available: {crawlee_status}")
        except ImportError:
            print("   ⚠️ Crawlee integration module not found (expected in test)")
            crawlee_status = False
        
        # Test extraction methods availability
        extractors_available = {
            'trafilatura': False,
            'newspaper': False, 
            'readability': False,
            'beautifulsoup': False
        }
        
        try:
            import trafilatura
            extractors_available['trafilatura'] = True
        except ImportError:
            pass
            
        try:
            from newspaper import Article
            extractors_available['newspaper'] = True
        except ImportError:
            pass
            
        try:
            from readability import Document
            extractors_available['readability'] = True
        except ImportError:
            pass
            
        try:
            from bs4 import BeautifulSoup
            extractors_available['beautifulsoup'] = True
        except ImportError:
            pass
        
        available_count = sum(extractors_available.values())
        print(f"   📊 Available extractors: {available_count}/4")
        for method, available in extractors_available.items():
            status = "✅" if available else "❌"
            print(f"      {method}: {status}")
        
        print("✅ Crawlee integration check completed")
        return True
        
    except Exception as e:
        print(f"❌ Crawlee integration check failed: {e}")
        return False

async def main():
    """Run all CLI tests."""
    print("🚀 Google News Crawler CLI Test Suite")
    print("=" * 50)
    
    tests = [
        ("CLI Imports", test_cli_imports()),
        ("Utility Functions", test_utility_functions()),
        ("Mock Search", test_mock_search()),
        ("Configuration", test_config_functionality()),
        ("Crawlee Integration", test_crawlee_integration_check())
    ]
    
    results = []
    for test_name, test_coro in tests:
        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("-" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print("-" * 30)
    print(f"Total: {passed}/{len(results)} tests passed ({passed/len(results)*100:.1f}%)")
    
    if passed == len(results):
        print("\n🎉 All tests passed! CLI is ready for use.")
        print("\nNext steps:")
        print("1. Install CLI: pip install -e .")
        print("2. Test CLI: gnews-crawler --help")
        print("3. Run search: gnews-crawler search 'Ethiopia conflict'")
        return True
    else:
        print(f"\n⚠️ {len(results) - passed} tests failed. Check dependencies and setup.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)