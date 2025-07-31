#!/usr/bin/env python3
"""
Crawlee-Enhanced Search Crawler Demo
====================================

Demonstration of the Crawlee-enhanced search crawler that combines multi-engine
search capabilities with robust content downloading and extraction.

This example shows how to:
1. Perform multi-engine searches
2. Download full content from search result URLs using Crawlee
3. Extract and analyze content with multiple parsing methods
4. Apply quality filtering and enhanced ranking
5. Extract geographic and conflict indicators
6. Generate comprehensive reports

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Crawlee-enhanced search crawler
try:
    from ..core.crawlee_enhanced_search_crawler import (
        CrawleeEnhancedSearchCrawler,
        CrawleeSearchConfig,
        CrawleeEnhancedResult,
        create_crawlee_search_config,
        create_crawlee_search_crawler,
        CRAWLEE_AVAILABLE
    )
except ImportError:
    logger.error("Could not import Crawlee-enhanced search crawler")
    raise


class CrawleeSearchDemo:
    """Demonstrates Crawlee-enhanced search crawler capabilities."""
    
    def __init__(self, output_dir: str = "./crawlee_demo_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    async def run_basic_demo(self):
        """Run basic Crawlee-enhanced search demonstration."""
        logger.info("🚀 Starting Crawlee-Enhanced Search Crawler Demo")
        
        if not CRAWLEE_AVAILABLE:
            logger.error("❌ Crawlee not available - install with: pip install crawlee")
            return
        
        # Create configuration optimized for content extraction
        config = create_crawlee_search_config(
            engines=['google', 'bing', 'duckduckgo', 'brave'],
            max_results=20,
            enable_content_extraction=True,
            target_countries=["ET", "SO", "KE", "SD", "UG"],
            # Content extraction settings
            preferred_extraction_method="auto",
            enable_image_extraction=True,
            enable_link_extraction=True,
            min_content_length=300,
            min_quality_score=0.4,
            # Performance settings
            max_concurrent=3,
            crawl_delay=2.0,
            max_retries=2
        )
        
        # Create and initialize crawler
        crawler = await create_crawlee_search_crawler(config)
        
        try:
            # Perform enhanced search with content extraction
            query = "Ethiopia Somalia conflict news humanitarian crisis"
            logger.info(f"🔍 Searching for: '{query}'")
            
            results = await crawler.search_with_content(
                query=query,
                max_results=15,
                extract_content=True
            )
            
            self.results = results
            
            # Display results summary
            self._display_results_summary(results)
            
            # Generate detailed analysis
            await self._generate_detailed_analysis(results)
            
            # Save results
            await self._save_results(results, "basic_demo")
            
        finally:
            await crawler.close()
    
    async def run_conflict_monitoring_demo(self):
        """Demonstrate conflict monitoring capabilities."""
        logger.info("🔍 Starting Conflict Monitoring Demo")
        
        if not CRAWLEE_AVAILABLE:
            logger.error("❌ Crawlee not available")
            return
        
        # Configure for Horn of Africa conflict monitoring
        config = CrawleeSearchConfig(
            engines=['google', 'bing', 'yandex', 'startpage'],
            total_max_results=30,
            download_content=True,
            
            # Conflict-specific settings
            target_countries=["ET", "SO", "ER", "DJ", "KE", "UG", "TZ", "SD", "SS"],
            target_languages=["en", "fr", "ar"],
            
            # Enhanced content extraction
            preferred_extraction_method="trafilatura",
            enable_content_scoring=True,
            min_quality_score=0.5,
            min_content_length=400,
            
            # Performance optimization
            max_concurrent=4,
            request_handler_timeout=45,
            navigation_timeout=20
        )
        
        crawler = CrawleeEnhancedSearchCrawler(config)
        await crawler.initialize_crawlee()
        
        try:
            # Test multiple conflict-related queries
            queries = [
                "Horn of Africa violence displacement refugees",
                "Ethiopia Tigray conflict humanitarian situation",
                "Somalia al-Shabaab security situation",
                "Sudan conflict Darfur humanitarian crisis"
            ]
            
            all_results = []
            
            for query in queries:
                logger.info(f"🔍 Processing query: '{query}'")
                
                results = await crawler.search_with_content(
                    query=query,
                    max_results=8,
                    extract_content=True
                )
                
                all_results.extend(results)
                
                # Brief pause between queries
                await asyncio.sleep(3)
            
            # Analyze conflict indicators
            conflict_analysis = self._analyze_conflict_indicators(all_results)
            
            # Display conflict monitoring summary
            self._display_conflict_summary(all_results, conflict_analysis)
            
            # Save comprehensive results
            await self._save_results(all_results, "conflict_monitoring", conflict_analysis)
            
        finally:
            await crawler.close()
    
    async def run_performance_comparison_demo(self):
        """Compare performance with and without Crawlee content extraction."""
        logger.info("⚡ Starting Performance Comparison Demo")
        
        if not CRAWLEE_AVAILABLE:
            logger.error("❌ Crawlee not available")
            return
        
        query = "East Africa security conflict news"
        
        # Test without content extraction
        config_basic = create_crawlee_search_config(
            engines=['google', 'bing', 'duckduckgo'],
            max_results=10,
            enable_content_extraction=False
        )
        
        crawler_basic = CrawleeEnhancedSearchCrawler(config_basic)
        
        start_time = datetime.now()
        basic_results = await crawler_basic.search_with_content(
            query=query,
            extract_content=False
        )
        basic_time = (datetime.now() - start_time).total_seconds()
        await crawler_basic.close()
        
        # Test with full content extraction
        config_enhanced = create_crawlee_search_config(
            engines=['google', 'bing', 'duckduckgo'],
            max_results=10,
            enable_content_extraction=True,
            max_concurrent=2
        )
        
        crawler_enhanced = await create_crawlee_search_crawler(config_enhanced)
        
        start_time = datetime.now()
        enhanced_results = await crawler_enhanced.search_with_content(
            query=query,
            extract_content=True
        )
        enhanced_time = (datetime.now() - start_time).total_seconds()
        await crawler_enhanced.close()
        
        # Display comparison
        self._display_performance_comparison(
            basic_results, basic_time,
            enhanced_results, enhanced_time
        )
    
    def _display_results_summary(self, results: List[CrawleeEnhancedResult]):
        """Display summary of search results."""
        logger.info("\n" + "="*60)
        logger.info("CRAWLEE-ENHANCED SEARCH RESULTS SUMMARY")
        logger.info("="*60)
        
        logger.info(f"📊 Total Results: {len(results)}")
        
        # Content extraction statistics
        with_content = [r for r in results if r.extracted_content]
        logger.info(f"📝 Results with Content: {len(with_content)}/{len(results)} ({len(with_content)/len(results)*100:.1f}%)")
        
        # Quality statistics
        avg_quality = sum(r.content_quality_score for r in results) / len(results) if results else 0
        logger.info(f"⭐ Average Quality Score: {avg_quality:.3f}")
        
        # Content statistics
        if with_content:
            avg_word_count = sum(r.word_count for r in with_content) / len(with_content)
            logger.info(f"📖 Average Word Count: {avg_word_count:.0f}")
            
            total_reading_time = sum(r.reading_time_minutes for r in with_content)
            logger.info(f"⏱️ Total Reading Time: {total_reading_time:.1f} minutes")
        
        # Geographic relevance
        with_geo = [r for r in results if r.geographic_entities]
        logger.info(f"🌍 Results with Geographic Entities: {len(with_geo)}")
        
        # Conflict relevance
        with_conflict = [r for r in results if r.conflict_indicators]
        logger.info(f"⚔️ Results with Conflict Indicators: {len(with_conflict)}")
        
        # Top results
        logger.info("\n🏆 TOP 5 RESULTS:")
        for i, result in enumerate(results[:5], 1):
            logger.info(f"{i}. {result.title[:60]}...")
            logger.info(f"   URL: {result.url}")
            logger.info(f"   Engines: {', '.join(result.engines_found)}")
            logger.info(f"   Quality: {result.content_quality_score:.3f}, "
                       f"Relevance: {result.relevance_score:.3f}")
            if result.extracted_content:
                logger.info(f"   Content: {result.word_count} words, "
                           f"Method: {result.extraction_method}")
            logger.info("")
    
    def _analyze_conflict_indicators(self, results: List[CrawleeEnhancedResult]) -> Dict[str, Any]:
        """Analyze conflict indicators across all results."""
        analysis = {
            'total_results': len(results),
            'conflict_related': 0,
            'geographic_entities': {},
            'conflict_indicators': {},
            'high_relevance_results': [],
            'urgent_indicators': [],
            'sources': {},
            'extraction_methods': {}
        }
        
        for result in results:
            # Count conflict-related results
            if result.conflict_indicators:
                analysis['conflict_related'] += 1
            
            # Aggregate geographic entities
            for entity in result.geographic_entities:
                analysis['geographic_entities'][entity] = \
                    analysis['geographic_entities'].get(entity, 0) + 1
            
            # Aggregate conflict indicators
            for indicator in result.conflict_indicators:
                analysis['conflict_indicators'][indicator] = \
                    analysis['conflict_indicators'].get(indicator, 0) + 1
            
            # Identify high relevance results
            if result.relevance_score > 0.7:
                analysis['high_relevance_results'].append({
                    'title': result.title,
                    'url': result.url,
                    'relevance_score': result.relevance_score,
                    'conflict_indicators': result.conflict_indicators,
                    'geographic_entities': result.geographic_entities
                })
            
            # Check for urgent indicators
            urgent_terms = ['breaking', 'urgent', 'emergency', 'crisis', 'attack', 'bombing']
            content_lower = (result.extracted_content or '').lower()
            for term in urgent_terms:
                if term in content_lower:
                    analysis['urgent_indicators'].append({
                        'result': result.title,
                        'term': term,
                        'url': result.url
                    })
            
            # Track sources
            domain = result.url.split('/')[2] if '/' in result.url else result.url
            analysis['sources'][domain] = analysis['sources'].get(domain, 0) + 1
            
            # Track extraction methods
            method = result.extraction_method
            analysis['extraction_methods'][method] = \
                analysis['extraction_methods'].get(method, 0) + 1
        
        # Sort by frequency
        analysis['geographic_entities'] = dict(sorted(
            analysis['geographic_entities'].items(),
            key=lambda x: x[1], reverse=True
        ))
        analysis['conflict_indicators'] = dict(sorted(
            analysis['conflict_indicators'].items(),
            key=lambda x: x[1], reverse=True
        ))
        
        return analysis
    
    def _display_conflict_summary(self, results: List[CrawleeEnhancedResult], analysis: Dict[str, Any]):
        """Display conflict monitoring summary."""
        logger.info("\n" + "="*60)
        logger.info("CONFLICT MONITORING ANALYSIS")
        logger.info("="*60)
        
        logger.info(f"📊 Total Results Analyzed: {analysis['total_results']}")
        logger.info(f"⚔️ Conflict-Related Results: {analysis['conflict_related']} "
                   f"({analysis['conflict_related']/analysis['total_results']*100:.1f}%)")
        
        # Top geographic entities
        logger.info("\n🌍 TOP GEOGRAPHIC ENTITIES:")
        for entity, count in list(analysis['geographic_entities'].items())[:10]:
            logger.info(f"   {entity}: {count} mentions")
        
        # Top conflict indicators
        logger.info("\n⚔️ TOP CONFLICT INDICATORS:")
        for indicator, count in list(analysis['conflict_indicators'].items())[:10]:
            logger.info(f"   {indicator}: {count} mentions")
        
        # High relevance results
        if analysis['high_relevance_results']:
            logger.info(f"\n🔥 HIGH RELEVANCE RESULTS ({len(analysis['high_relevance_results'])}):")
            for result in analysis['high_relevance_results'][:5]:
                logger.info(f"   • {result['title'][:50]}... (Score: {result['relevance_score']:.3f})")
        
        # Urgent indicators
        if analysis['urgent_indicators']:
            logger.info(f"\n🚨 URGENT INDICATORS DETECTED ({len(analysis['urgent_indicators'])}):")
            for indicator in analysis['urgent_indicators'][:5]:
                logger.info(f"   • '{indicator['term']}' in: {indicator['result'][:40]}...")
        
        # Source distribution
        logger.info("\n📰 TOP SOURCES:")
        for source, count in list(analysis['sources'].items())[:5]:
            logger.info(f"   {source}: {count} articles")
        
        # Extraction method performance
        logger.info("\n🔧 EXTRACTION METHODS:")
        for method, count in analysis['extraction_methods'].items():
            logger.info(f"   {method}: {count} results")
    
    def _display_performance_comparison(
        self,
        basic_results: List[CrawleeEnhancedResult],
        basic_time: float,
        enhanced_results: List[CrawleeEnhancedResult],
        enhanced_time: float
    ):
        """Display performance comparison."""
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE COMPARISON")
        logger.info("="*60)
        
        logger.info(f"📊 Basic Search (No Content):")
        logger.info(f"   Results: {len(basic_results)}")
        logger.info(f"   Time: {basic_time:.2f} seconds")
        logger.info(f"   Speed: {len(basic_results)/basic_time:.1f} results/second")
        
        logger.info(f"\n📊 Enhanced Search (With Crawlee Content):")
        logger.info(f"   Results: {len(enhanced_results)}")
        logger.info(f"   Time: {enhanced_time:.2f} seconds")
        logger.info(f"   Speed: {len(enhanced_results)/enhanced_time:.1f} results/second")
        
        with_content = [r for r in enhanced_results if r.extracted_content]
        if with_content:
            avg_content_length = sum(len(r.extracted_content or '') for r in with_content) / len(with_content)
            logger.info(f"   Content Extracted: {len(with_content)}/{len(enhanced_results)} results")
            logger.info(f"   Avg Content Length: {avg_content_length:.0f} characters")
        
        overhead = (enhanced_time - basic_time) / basic_time * 100
        logger.info(f"\n⚡ Performance Overhead: {overhead:.1f}%")
        logger.info(f"💡 Content Extraction Time: ~{enhanced_time - basic_time:.2f} seconds")
    
    async def _generate_detailed_analysis(self, results: List[CrawleeEnhancedResult]):
        """Generate detailed analysis of results."""
        logger.info("\n🔍 Generating Detailed Analysis...")
        
        # Content quality analysis
        quality_distribution = {
            'excellent': [r for r in results if r.content_quality_score > 0.8],
            'good': [r for r in results if 0.6 < r.content_quality_score <= 0.8],
            'fair': [r for r in results if 0.4 < r.content_quality_score <= 0.6],
            'poor': [r for r in results if r.content_quality_score <= 0.4]
        }
        
        logger.info("📊 Content Quality Distribution:")
        for quality, results_list in quality_distribution.items():
            logger.info(f"   {quality.title()}: {len(results_list)} results")
        
        # Geographic coverage analysis
        countries_mentioned = {}
        for result in results:
            for entity in result.geographic_entities:
                if entity in ["Ethiopia", "Somalia", "Kenya", "Sudan", "Uganda", "Tanzania", "Eritrea", "Djibouti"]:
                    countries_mentioned[entity] = countries_mentioned.get(entity, 0) + 1
        
        if countries_mentioned:
            logger.info("\n🌍 Country Coverage:")
            for country, count in sorted(countries_mentioned.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"   {country}: {count} mentions")
    
    async def _save_results(
        self,
        results: List[CrawleeEnhancedResult],
        demo_name: str,
        analysis: Dict[str, Any] = None
    ):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results as JSON
        results_file = self.output_dir / f"{demo_name}_{timestamp}_results.json"
        
        results_data = {
            'demo_name': demo_name,
            'timestamp': timestamp,
            'total_results': len(results),
            'results': [result.to_dict() for result in results]
        }
        
        if analysis:
            results_data['analysis'] = analysis
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"💾 Results saved to: {results_file}")
        
        # Save summary report
        summary_file = self.output_dir / f"{demo_name}_{timestamp}_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Crawlee-Enhanced Search Crawler Demo: {demo_name}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Results: {len(results)}\n\n")
            
            # Content statistics
            with_content = [r for r in results if r.extracted_content]
            f.write(f"Results with Content: {len(with_content)}/{len(results)}\n")
            
            if with_content:
                avg_quality = sum(r.content_quality_score for r in with_content) / len(with_content)
                avg_words = sum(r.word_count for r in with_content) / len(with_content)
                f.write(f"Average Quality Score: {avg_quality:.3f}\n")
                f.write(f"Average Word Count: {avg_words:.0f}\n\n")
            
            # Top results
            f.write("Top Results:\n")
            f.write("-"*20 + "\n")
            for i, result in enumerate(results[:10], 1):
                f.write(f"{i}. {result.title}\n")
                f.write(f"   URL: {result.url}\n")
                f.write(f"   Quality: {result.content_quality_score:.3f}\n")
                f.write(f"   Method: {result.extraction_method}\n\n")
        
        logger.info(f"📄 Summary saved to: {summary_file}")


async def main():
    """Main demonstration function."""
    demo = CrawleeSearchDemo()
    
    try:
        logger.info("🎯 Starting Crawlee-Enhanced Search Crawler Demonstrations")
        
        # Run basic demo
        await demo.run_basic_demo()
        
        # Wait between demos
        await asyncio.sleep(5)
        
        # Run conflict monitoring demo
        await demo.run_conflict_monitoring_demo()
        
        # Wait between demos
        await asyncio.sleep(5)
        
        # Run performance comparison
        await demo.run_performance_comparison_demo()
        
        logger.info("\n✅ All demonstrations completed successfully!")
        logger.info(f"📁 Results saved to: {demo.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())