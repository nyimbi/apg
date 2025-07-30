"""
News Site Crawling Example for Gen Crawler
==========================================

Advanced example showing how to crawl news sites using gen_crawler
with content filtering, classification, and analysis.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsAnalyzer:
    """Analyzer for news content and site characteristics."""
    
    def __init__(self):
        self.news_indicators = [
            'news', 'article', 'story', 'report', 'breaking',
            'politics', 'world', 'local', 'business', 'sports'
        ]
        
        self.conflict_keywords = [
            'conflict', 'war', 'violence', 'protest', 'crisis',
            'attack', 'military', 'security', 'peace', 'tension'
        ]
    
    def analyze_news_content(self, parsed_content) -> Dict[str, Any]:
        """Analyze parsed content for news characteristics."""
        analysis = {
            'is_news': False,
            'news_score': 0.0,
            'conflict_related': False,
            'conflict_score': 0.0,
            'categories': [],
            'urgency_level': 'low'
        }
        
        if not parsed_content.content:
            return analysis
        
        content_lower = parsed_content.content.lower()
        title_lower = parsed_content.title.lower() if parsed_content.title else ""
        
        # Check for news indicators
        news_matches = sum(1 for indicator in self.news_indicators 
                          if indicator in content_lower or indicator in title_lower)
        analysis['news_score'] = min(news_matches / len(self.news_indicators), 1.0)
        analysis['is_news'] = analysis['news_score'] > 0.2
        
        # Check for conflict content
        conflict_matches = sum(1 for keyword in self.conflict_keywords 
                              if keyword in content_lower or keyword in title_lower)
        analysis['conflict_score'] = min(conflict_matches / len(self.conflict_keywords), 1.0)
        analysis['conflict_related'] = analysis['conflict_score'] > 0.1
        
        # Determine urgency
        urgent_indicators = ['breaking', 'urgent', 'emergency', 'crisis']
        if any(indicator in title_lower for indicator in urgent_indicators):
            analysis['urgency_level'] = 'high'
        elif any(indicator in content_lower for indicator in urgent_indicators):
            analysis['urgency_level'] = 'medium'
        
        # Categorize content
        if 'politics' in content_lower or 'government' in content_lower:
            analysis['categories'].append('politics')
        if 'business' in content_lower or 'economy' in content_lower:
            analysis['categories'].append('business')
        if 'sports' in content_lower:
            analysis['categories'].append('sports')
        if 'technology' in content_lower or 'tech' in content_lower:
            analysis['categories'].append('technology')
        
        return analysis

async def crawl_news_site(site_url: str, max_articles: int = 100) -> Dict[str, Any]:
    """
    Crawl a news site and analyze content.
    
    Args:
        site_url: URL of the news site to crawl
        max_articles: Maximum number of articles to crawl
        
    Returns:
        Dictionary containing crawl results and analysis
    """
    try:
        from gen_crawler.core import GenCrawler
        from gen_crawler.config import create_gen_config
        
        logger.info(f"🗞️ Starting news site crawl: {site_url}")
        
        # Create news-optimized configuration
        config = create_gen_config()
        config.settings.performance.max_pages_per_site = max_articles
        config.settings.performance.max_concurrent = 3
        config.settings.performance.crawl_delay = 2.0
        config.settings.content_filters.include_patterns = [
            'article', 'news', 'story', 'post', 'breaking'
        ]
        config.settings.content_filters.exclude_patterns = [
            'tag', 'category', 'archive', 'login', 'subscribe', 
            'newsletter', 'advertisement'
        ]
        config.settings.content_filters.min_content_length = 200
        
        # Create crawler
        crawler_config = config.get_crawler_config()
        crawler = GenCrawler(crawler_config)
        await crawler.initialize()
        
        # Crawl the site
        result = await crawler.crawl_site(site_url)
        
        # Analyze content
        news_analyzer = NewsAnalyzer()
        analyzed_articles = []
        
        for page in result.pages:
            if page.success and page.content_type == 'article':
                analysis = news_analyzer.analyze_news_content(page)
                
                article_data = {
                    'url': page.url,
                    'title': page.title,
                    'word_count': page.word_count,
                    'content_preview': page.content[:200] + '...' if page.content else '',
                    'crawl_time': page.crawl_time,
                    'analysis': analysis
                }
                analyzed_articles.append(article_data)
        
        # Generate summary statistics
        total_articles = len(analyzed_articles)
        news_articles = sum(1 for a in analyzed_articles if a['analysis']['is_news'])
        conflict_articles = sum(1 for a in analyzed_articles if a['analysis']['conflict_related'])
        high_urgency = sum(1 for a in analyzed_articles if a['analysis']['urgency_level'] == 'high')
        
        summary = {
            'site_url': site_url,
            'crawl_timestamp': datetime.now().isoformat(),
            'total_pages_crawled': result.total_pages,
            'successful_pages': result.successful_pages,
            'articles_found': total_articles,
            'news_articles': news_articles,
            'conflict_related_articles': conflict_articles,
            'high_urgency_articles': high_urgency,
            'crawl_time': result.total_time,
            'success_rate': result.success_rate,
            'articles': analyzed_articles[:10]  # Include top 10 articles
        }
        
        # Cleanup
        await crawler.cleanup()
        
        return summary
        
    except Exception as e:
        logger.error(f"Error crawling news site {site_url}: {e}")
        return {'error': str(e), 'site_url': site_url}

async def crawl_multiple_news_sites(news_sites: List[str]) -> Dict[str, Any]:
    """
    Crawl multiple news sites and provide comparative analysis.
    
    Args:
        news_sites: List of news site URLs to crawl
        
    Returns:
        Comprehensive analysis of all sites
    """
    logger.info(f"📰 Starting multi-site news crawl: {len(news_sites)} sites")
    
    all_results = []
    
    for site_url in news_sites:
        logger.info(f"Processing: {site_url}")
        
        result = await crawl_news_site(site_url, max_articles=50)
        all_results.append(result)
        
        # Small delay between sites
        await asyncio.sleep(2)
    
    # Generate comparative analysis
    total_articles = sum(r.get('articles_found', 0) for r in all_results if 'error' not in r)
    total_news = sum(r.get('news_articles', 0) for r in all_results if 'error' not in r)
    total_conflict = sum(r.get('conflict_related_articles', 0) for r in all_results if 'error' not in r)
    
    successful_sites = [r for r in all_results if 'error' not in r]
    failed_sites = [r for r in all_results if 'error' in r]
    
    comparative_analysis = {
        'summary': {
            'sites_processed': len(news_sites),
            'successful_sites': len(successful_sites),
            'failed_sites': len(failed_sites),
            'total_articles_found': total_articles,
            'total_news_articles': total_news,
            'total_conflict_articles': total_conflict,
            'analysis_timestamp': datetime.now().isoformat()
        },
        'site_results': all_results,
        'top_sites_by_articles': sorted(
            successful_sites, 
            key=lambda x: x.get('articles_found', 0), 
            reverse=True
        )[:5],
        'conflict_monitoring': {
            'total_conflict_articles': total_conflict,
            'sites_with_conflict_content': sum(
                1 for r in successful_sites 
                if r.get('conflict_related_articles', 0) > 0
            ),
            'avg_conflict_articles_per_site': (
                total_conflict / len(successful_sites) 
                if successful_sites else 0
            )
        }
    }
    
    return comparative_analysis

async def monitor_breaking_news(news_sites: List[str], check_interval: int = 300):
    """
    Monitor news sites for breaking news and high-urgency content.
    
    Args:
        news_sites: List of news sites to monitor
        check_interval: Time between checks in seconds
    """
    logger.info(f"🚨 Starting breaking news monitor for {len(news_sites)} sites")
    logger.info(f"Check interval: {check_interval} seconds")
    
    previous_articles = set()
    
    while True:
        try:
            logger.info("🔍 Checking for breaking news...")
            
            current_articles = set()
            breaking_news_found = []
            
            for site_url in news_sites:
                try:
                    result = await crawl_news_site(site_url, max_articles=20)
                    
                    if 'error' not in result:
                        for article in result.get('articles', []):
                            article_id = f"{site_url}:{article['title']}"
                            current_articles.add(article_id)
                            
                            # Check if this is new breaking news
                            if (article_id not in previous_articles and 
                                article['analysis']['urgency_level'] == 'high'):
                                breaking_news_found.append({
                                    'site': site_url,
                                    'title': article['title'],
                                    'url': article['url'],
                                    'urgency': article['analysis']['urgency_level'],
                                    'conflict_related': article['analysis']['conflict_related']
                                })
                
                except Exception as e:
                    logger.warning(f"Error checking {site_url}: {e}")
                    continue
            
            # Report breaking news
            if breaking_news_found:
                logger.warning(f"🚨 BREAKING NEWS DETECTED: {len(breaking_news_found)} articles")
                for news in breaking_news_found:
                    logger.warning(f"   📰 {news['site']}: {news['title'][:100]}...")
                    if news['conflict_related']:
                        logger.warning(f"      ⚠️ CONFLICT-RELATED CONTENT")
            else:
                logger.info("✅ No breaking news detected")
            
            previous_articles = current_articles
            
            # Wait for next check
            logger.info(f"⏰ Waiting {check_interval} seconds until next check...")
            await asyncio.sleep(check_interval)
            
        except KeyboardInterrupt:
            logger.info("🛑 Breaking news monitoring stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in breaking news monitor: {e}")
            await asyncio.sleep(60)  # Wait a minute before retrying

async def main():
    """Run news crawling examples."""
    logger.info("📰 Gen Crawler News Site Examples")
    logger.info("=" * 50)
    
    # Sample news sites (using public examples)
    news_sites = [
        "https://httpbin.org",  # Test site
        # Add real news sites as needed
        # "https://www.example-news.com",
        # "https://www.another-news-site.com"
    ]
    
    examples = [
        "Single Site Crawl",
        "Multiple Sites Analysis", 
        "Breaking News Monitor"
    ]
    
    print("\nAvailable examples:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    
    try:
        choice = input("\nChoose an example (1-3) or 'all': ").strip().lower()
        
        if choice == '1' or choice == 'all':
            logger.info("\n🗞️ Running: Single Site Crawl")
            if news_sites:
                result = await crawl_news_site(news_sites[0])
                print(json.dumps(result, indent=2, default=str))
        
        if choice == '2' or choice == 'all':
            logger.info("\n📰 Running: Multiple Sites Analysis")
            analysis = await crawl_multiple_news_sites(news_sites)
            print(json.dumps(analysis['summary'], indent=2, default=str))
        
        if choice == '3':
            logger.info("\n🚨 Running: Breaking News Monitor")
            logger.info("Press Ctrl+C to stop monitoring")
            await monitor_breaking_news(news_sites, check_interval=60)
    
    except KeyboardInterrupt:
        logger.info("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Example error: {e}")

if __name__ == "__main__":
    asyncio.run(main())