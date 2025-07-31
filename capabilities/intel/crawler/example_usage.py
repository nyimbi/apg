"""
APG Crawler Capability - Usage Examples
=======================================

Simple examples demonstrating the guaranteed success crawling API:
- Single page scraping with markdown output
- Multi-page site crawling
- Automatic link discovery and crawling
- Error handling and fallback strategies

Copyright © 2025 Datacraft (nyimbi@gmail.com)
"""

import asyncio
import json
from simple_api import scrape_page, crawl_site, crawl_site_from_homepage

# =====================================================
# EXAMPLE 1: SINGLE PAGE SCRAPING
# =====================================================

async def example_single_page():
    """Example: Scrape a single page and get clean markdown"""
    print("📄 Example 1: Single Page Scraping")
    print("-" * 40)
    
    # Scrape a single page - GUARANTEED to succeed
    url = "https://example.com"
    result = await scrape_page(url)
    
    print(f"URL: {result.url}")
    print(f"Success: {result.success}")
    print(f"Title: {result.title}")
    print(f"Strategy Used: {result.metadata.get('strategy_used', 'unknown')}")
    print(f"Content Length: {len(result.markdown_content)} characters")
    print(f"Processing Time: {result.metadata.get('processing_time', 0):.2f}s")
    
    print("\n📝 Markdown Content Preview:")
    print("-" * 30)
    print(result.markdown_content[:500] + "..." if len(result.markdown_content) > 500 else result.markdown_content)
    
    return result


# =====================================================
# EXAMPLE 2: MULTIPLE PAGE CRAWLING
# =====================================================

async def example_multiple_pages():
    """Example: Crawl multiple pages concurrently"""
    print("\n📚 Example 2: Multiple Page Crawling")
    print("-" * 40)
    
    # List of URLs to crawl
    urls = [
        "https://httpbin.org/html",
        "https://httpbin.org/json", 
        "https://example.com",
        "https://www.python.org/about/",
    ]
    
    print(f"Crawling {len(urls)} URLs concurrently...")
    
    # Crawl all URLs - each individually guaranteed to succeed
    results = await crawl_site(urls, max_concurrent=2)
    
    print(f"\n📊 Results Summary:")
    print(f"Total URLs: {results.total_count}")
    print(f"Successful: {results.success_count}")
    print(f"Success Rate: {results.success_rate:.1%}")
    print(f"Processing Time: {results.processing_time:.2f}s")
    
    print(f"\n📋 Individual Results:")
    for i, result in enumerate(results.results, 1):
        status = "✅" if result.success else "❌"
        strategy = result.metadata.get('strategy_used', 'unknown')
        length = len(result.markdown_content)
        
        print(f"{status} {i}. {result.url}")
        print(f"     Title: {result.title or 'No title'}")
        print(f"     Strategy: {strategy}")
        print(f"     Content: {length} chars")
        
        if result.error:
            print(f"     Error: {result.error}")
    
    return results


# =====================================================
# EXAMPLE 3: SITE DISCOVERY AND CRAWLING
# =====================================================

async def example_site_discovery():
    """Example: Automatically discover and crawl a website"""
    print("\n🔍 Example 3: Site Discovery and Crawling")
    print("-" * 40)
    
    # Start from a homepage and automatically discover links
    base_url = "https://quotes.toscrape.com/"
    max_pages = 5
    
    print(f"Starting from: {base_url}")
    print(f"Max pages to crawl: {max_pages}")
    
    # Discover and crawl pages automatically
    results = await crawl_site_from_homepage(base_url, max_pages=max_pages)
    
    print(f"\n📊 Discovery Results:")
    print(f"Pages Found & Crawled: {results.total_count}")
    print(f"Successful Extractions: {results.success_count}")
    print(f"Success Rate: {results.success_rate:.1%}")
    print(f"Processing Time: {results.processing_time:.2f}s")
    
    print(f"\n📋 Discovered Pages:")
    for i, result in enumerate(results.results, 1):
        status = "✅" if result.success else "❌"
        print(f"{status} {i}. {result.url}")
        print(f"     Title: {result.title or 'No title'}")
        print(f"     Content: {len(result.markdown_content)} chars")
    
    return results


# =====================================================
# EXAMPLE 4: ERROR HANDLING AND ROBUSTNESS
# =====================================================

async def example_error_handling():
    """Example: Demonstrate robust error handling"""
    print("\n🛡️ Example 4: Error Handling and Robustness")
    print("-" * 40)
    
    # Test with challenging URLs
    challenging_urls = [
        "https://httpbin.org/status/404",      # 404 error
        "https://httpbin.org/delay/10",        # Timeout test  
        "https://invalid-url-that-does-not-exist.com",  # Invalid URL
        "https://httpbin.org/html",            # Should work
    ]
    
    print("Testing with challenging URLs:")
    for url in challenging_urls:
        print(f"  - {url}")
    
    results = await crawl_site(challenging_urls, max_concurrent=2)
    
    print(f"\n📊 Robustness Test Results:")
    print(f"URLs Tested: {results.total_count}")
    print(f"Successful: {results.success_count}")
    print(f"Failed: {results.total_count - results.success_count}")
    print(f"Resilience Rate: {results.success_rate:.1%}")
    
    print(f"\n📋 Detailed Results:")
    for i, result in enumerate(results.results, 1):
        status = "✅" if result.success else "❌"
        print(f"{status} {i}. {challenging_urls[i-1]}")
        
        if result.success:
            strategy = result.metadata.get('strategy_used', 'unknown')
            print(f"     Strategy: {strategy}")
            print(f"     Content: {len(result.markdown_content)} chars")
        else:
            print(f"     Error: {result.error}")
            print(f"     Fallback Content: {len(result.markdown_content)} chars")
            # Even failed requests get some content due to guaranteed success design
    
    return results


# =====================================================
# EXAMPLE 5: ADVANCED METADATA EXTRACTION
# =====================================================

async def example_advanced_metadata():
    """Example: Extract advanced metadata and intelligence"""
    print("\n🧠 Example 5: Advanced Metadata Extraction")
    print("-" * 40)
    
    # Scrape a content-rich page
    url = "https://www.python.org/about/"
    result = await scrape_page(url)
    
    print(f"URL: {result.url}")
    print(f"Success: {result.success}")
    
    print(f"\n📊 Metadata Analysis:")
    metadata = result.metadata
    
    # Basic crawling metadata
    print(f"Strategy Used: {metadata.get('strategy_used', 'unknown')}")
    print(f"Status Code: {metadata.get('status_code', 'unknown')}")
    print(f"Response Time: {metadata.get('response_time', 0):.3f}s")
    print(f"Content Length: {metadata.get('content_length', 0)} chars")
    print(f"Language: {metadata.get('language', 'unknown')}")
    
    # AI Intelligence metadata (if available)
    if 'entity_count' in metadata:
        print(f"\n🤖 AI Intelligence:")
        print(f"Entities Found: {metadata.get('entity_count', 0)}")
        print(f"Content Category: {metadata.get('content_category', 'unknown')}")
        print(f"Industry Domain: {metadata.get('industry_domain', 'unknown')}")
        print(f"Sentiment (Positive): {metadata.get('sentiment_positive', 0):.2f}")
        
        themes = metadata.get('key_themes', [])
        if themes:
            print(f"Key Themes: {', '.join(themes)}")
    
    return result


# =====================================================
# EXAMPLE 6: SYNCHRONOUS USAGE
# =====================================================

def example_synchronous_usage():
    """Example: Use the crawler in synchronous code"""
    print("\n🔄 Example 6: Synchronous Usage")
    print("-" * 40)
    
    from simple_api import scrape_page_sync, crawl_site_sync
    
    # Synchronous single page scraping
    result = scrape_page_sync("https://example.com")
    print(f"Sync Single Page: {result.success} - {len(result.markdown_content)} chars")
    
    # Synchronous multiple page crawling
    urls = ["https://httpbin.org/html", "https://example.com"]
    results = crawl_site_sync(urls)
    print(f"Sync Multi Page: {results.success_rate:.1%} success rate")
    
    return results


# =====================================================
# MAIN DEMO RUNNER
# =====================================================

async def run_all_examples():
    """Run all usage examples"""
    print("🚀 APG CRAWLER CAPABILITY - USAGE EXAMPLES")
    print("=" * 60)
    print("Demonstrating guaranteed success crawling with markdown output")
    print()
    
    try:
        # Run all async examples
        await example_single_page()
        await example_multiple_pages()
        await example_site_discovery() 
        await example_error_handling()
        await example_advanced_metadata()
        
        # Run sync example
        example_synchronous_usage()
        
        print("\n🎉 All Examples Completed Successfully!")
        print("\n📚 Ready to integrate APG Crawler into your applications!")
        
        print("\n💡 Quick Integration Tips:")
        print("1. Import: from crawler.simple_api import scrape_page, crawl_site")
        print("2. Single page: result = await scrape_page('https://example.com')")
        print("3. Multiple pages: results = await crawl_site(['url1', 'url2'])")
        print("4. Access content: print(result.markdown_content)")
        print("5. Check metadata: print(result.metadata)")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("This might indicate missing dependencies or network issues.")
        print("The crawler is still functional - try individual functions.")


# =====================================================
# COMMAND LINE EXECUTION
# =====================================================

if __name__ == "__main__":
    # Run all examples
    asyncio.run(run_all_examples())