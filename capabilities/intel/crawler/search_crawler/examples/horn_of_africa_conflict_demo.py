"""
Horn of Africa Conflict Monitoring Demo
=======================================

Comprehensive demonstration of the search crawler for Horn of Africa conflict monitoring.
Uses multiple search engines to find conflict-related news and content.

Author: Lindela Development Team
Version: 1.0.0
License: MIT
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import search crawler components
try:
    from ..core.search_crawler import SearchCrawler, SearchCrawlerConfig
    from ..core.conflict_search_crawler import ConflictSearchCrawler, ConflictSearchConfig
    from ..keywords.horn_of_africa_keywords import HornOfAfricaKeywords
    from ..keywords.conflict_keywords import ConflictKeywordManager
    from ..keywords.keyword_analyzer import KeywordAnalyzer
    from ..engines import get_available_engines
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import search crawler components: {e}")
    COMPONENTS_AVAILABLE = False


class HornOfAfricaConflictMonitor:
    """Comprehensive conflict monitoring system for Horn of Africa."""
    
    def __init__(self, output_dir: str = "./results"):
        """Initialize the conflict monitor."""
        if not COMPONENTS_AVAILABLE:
            raise ImportError("Search crawler components not available")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize keyword systems
        self.horn_keywords = HornOfAfricaKeywords()
        self.conflict_keywords = ConflictKeywordManager()
        self.keyword_analyzer = KeywordAnalyzer()
        
        # Initialize search crawler with multiple engines
        self.crawler_config = ConflictSearchConfig(
            engines=['google', 'bing', 'duckduckgo', 'yandex', 'brave', 'startpage'],
            max_results_per_engine=20,
            total_max_results=100,
            download_content=True,
            parse_content=True,
            min_relevance_score=0.6,
            enable_alerts=True,
            escalation_threshold=0.8
        )
        
        self.conflict_crawler = ConflictSearchCrawler(self.crawler_config)
        
        # Results storage
        self.results = []
        self.alerts = []
        self.statistics = {}
    
    async def monitor_conflicts(self, countries=None, time_range='week', max_results=50):
        """
        Monitor conflicts across Horn of Africa countries.
        
        Args:
            countries: List of countries to monitor (None for all)
            time_range: Time range for results ('day', 'week', 'month')
            max_results: Maximum results to process
        """
        logger.info("Starting Horn of Africa conflict monitoring...")
        
        if countries is None:
            countries = ['somalia', 'ethiopia', 'eritrea', 'djibouti', 'sudan', 'south_sudan']
        
        all_results = []
        
        for country in countries:
            logger.info(f"Monitoring conflicts in {country.title()}...")
            
            try:
                # Generate search queries for this country
                country_keywords = self.horn_keywords.get_country_keywords(country)
                conflict_queries = self.horn_keywords.generate_search_queries(
                    countries=[country],
                    conflict_level='high',
                    max_queries=10
                )
                
                # Search for each query
                for query in conflict_queries[:5]:  # Limit to top 5 queries per country
                    logger.info(f"Searching: {query}")
                    
                    results = await self.conflict_crawler.search_conflicts(
                        region='horn_of_africa',
                        keywords=[query],
                        max_results=10,
                        time_range=time_range
                    )
                    
                    if results:
                        all_results.extend(results)
                        logger.info(f"Found {len(results)} results for query: {query}")
                        
                        # Check for high-priority alerts
                        alerts = [r for r in results if r.requires_alert]
                        if alerts:
                            self.alerts.extend(alerts)
                            logger.warning(f"⚠️  {len(alerts)} HIGH PRIORITY alerts found!")
                    
                    # Small delay between queries
                    await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error monitoring {country}: {e}")
                continue
        
        # Remove duplicates and rank results
        unique_results = self._deduplicate_results(all_results)
        ranked_results = unique_results[:max_results]
        
        self.results = ranked_results
        logger.info(f"Monitoring complete. Found {len(ranked_results)} unique results.")
        
        return ranked_results
    
    def _deduplicate_results(self, results):
        """Remove duplicate results based on URL."""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # Sort by conflict score
        unique_results.sort(key=lambda x: x.conflict_score, reverse=True)
        return unique_results
    
    async def analyze_keyword_effectiveness(self):
        """Analyze which keywords are most effective."""
        if not self.results:
            logger.warning("No results to analyze")
            return
        
        logger.info("Analyzing keyword effectiveness...")
        
        # Collect all text content
        all_text = []
        for result in self.results:
            text = f"{result.title} {result.snippet}"
            if result.content:
                text += f" {result.content[:1000]}"  # First 1000 chars
            all_text.append(text)
        
        combined_text = " ".join(all_text)
        
        # Get high-priority conflict keywords
        conflict_keywords = self.conflict_keywords.get_high_priority_keywords()
        
        # Analyze keyword effectiveness
        analyses = self.keyword_analyzer.analyze_text(combined_text, conflict_keywords)
        
        # Generate report
        report = self.keyword_analyzer.generate_keyword_report(analyses)
        
        # Save report
        report_file = self.output_dir / f"keyword_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Keyword analysis saved to {report_file}")
        
        # Get suggestions for new keywords
        suggestions = self.keyword_analyzer.suggest_new_keywords(analyses)
        if suggestions:
            logger.info(f"Suggested new keywords: {', '.join(suggestions[:5])}")
        
        return analyses
    
    def generate_conflict_report(self):
        """Generate comprehensive conflict monitoring report."""
        if not self.results:
            logger.warning("No results to report")
            return
        
        logger.info("Generating conflict monitoring report...")
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_results': len(self.results),
                'total_alerts': len(self.alerts),
                'monitoring_period': 'last week',
                'engines_used': self.crawler_config.engines,
                'countries_monitored': ['Somalia', 'Ethiopia', 'Eritrea', 'Djibouti', 'Sudan', 'South Sudan']
            },
            'summary': {
                'high_priority_alerts': len([r for r in self.results if r.conflict_score >= 0.8]),
                'medium_priority': len([r for r in self.results if 0.6 <= r.conflict_score < 0.8]),
                'low_priority': len([r for r in self.results if r.conflict_score < 0.6]),
                'average_conflict_score': sum(r.conflict_score for r in self.results) / len(self.results),
                'locations_mentioned': self._get_location_distribution(),
                'top_sources': self._get_top_sources()
            },
            'alerts': [
                {
                    'title': result.title,
                    'url': result.url,
                    'conflict_score': result.conflict_score,
                    'alert_reasons': result.alert_reasons,
                    'locations': [loc['name'] for loc in result.locations_mentioned],
                    'escalation_indicators': result.escalation_indicators,
                    'timestamp': result.timestamp.isoformat(),
                    'source_credibility': result.source_credibility
                }
                for result in self.alerts[:20]  # Top 20 alerts
            ],
            'top_results': [
                {
                    'title': result.title,
                    'url': result.url,
                    'snippet': result.snippet,
                    'conflict_score': result.conflict_score,
                    'engines_found': result.engines_found,
                    'locations': [loc['name'] for loc in result.locations_mentioned],
                    'sentiment': result.sentiment_label,
                    'is_breaking': result.is_breaking,
                    'timestamp': result.timestamp.isoformat()
                }
                for result in self.results[:50]  # Top 50 results
            ],
            'statistics': {
                'engine_performance': self._get_engine_performance(),
                'temporal_distribution': self._get_temporal_distribution(),
                'sentiment_analysis': self._get_sentiment_distribution()
            }
        }
        
        # Save JSON report
        report_file = self.output_dir / f"conflict_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # Save human-readable report
        text_report = self._generate_text_report(report)
        text_file = self.output_dir / f"conflict_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        logger.info(f"Conflict report saved to {report_file} and {text_file}")
        
        return report
    
    def _get_location_distribution(self):
        """Get distribution of mentioned locations."""
        locations = {}
        for result in self.results:
            for location in result.locations_mentioned:
                name = location['name']
                locations[name] = locations.get(name, 0) + 1
        
        return dict(sorted(locations.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _get_top_sources(self):
        """Get top news sources."""
        sources = {}
        for result in self.results:
            try:
                from urllib.parse import urlparse
                domain = urlparse(result.url).netloc
                sources[domain] = sources.get(domain, 0) + 1
            except:
                continue
        
        return dict(sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _get_engine_performance(self):
        """Get search engine performance statistics."""
        engine_stats = {}
        for result in self.results:
            for engine in result.engines_found:
                if engine not in engine_stats:
                    engine_stats[engine] = {'results': 0, 'avg_score': 0, 'scores': []}
                engine_stats[engine]['results'] += 1
                engine_stats[engine]['scores'].append(result.conflict_score)
        
        # Calculate averages
        for engine, stats in engine_stats.items():
            if stats['scores']:
                stats['avg_score'] = sum(stats['scores']) / len(stats['scores'])
            del stats['scores']  # Remove raw scores to save space
        
        return engine_stats
    
    def _get_temporal_distribution(self):
        """Get temporal distribution of results."""
        now = datetime.now()
        buckets = {
            'last_24h': 0,
            'last_week': 0,
            'last_month': 0,
            'older': 0
        }
        
        for result in self.results:
            age = now - result.timestamp
            if age.days == 0:
                buckets['last_24h'] += 1
            elif age.days <= 7:
                buckets['last_week'] += 1
            elif age.days <= 30:
                buckets['last_month'] += 1
            else:
                buckets['older'] += 1
        
        return buckets
    
    def _get_sentiment_distribution(self):
        """Get sentiment distribution of results."""
        sentiments = {}
        for result in self.results:
            label = result.sentiment_label
            sentiments[label] = sentiments.get(label, 0) + 1
        
        return sentiments
    
    def _generate_text_report(self, report_data):
        """Generate human-readable text report."""
        lines = []
        lines.append("HORN OF AFRICA CONFLICT MONITORING REPORT")
        lines.append("=" * 50)
        lines.append(f"Generated: {report_data['metadata']['generated_at']}")
        lines.append(f"Total Results: {report_data['metadata']['total_results']}")
        lines.append(f"Total Alerts: {report_data['metadata']['total_alerts']}")
        lines.append("")
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 20)
        summary = report_data['summary']
        lines.append(f"High Priority Alerts: {summary['high_priority_alerts']}")
        lines.append(f"Medium Priority: {summary['medium_priority']}")
        lines.append(f"Low Priority: {summary['low_priority']}")
        lines.append(f"Average Conflict Score: {summary['average_conflict_score']:.3f}")
        lines.append("")
        
        # Top locations
        lines.append("TOP AFFECTED LOCATIONS")
        lines.append("-" * 25)
        for location, count in list(summary['locations_mentioned'].items())[:5]:
            lines.append(f"  {location}: {count} mentions")
        lines.append("")
        
        # Top sources
        lines.append("TOP NEWS SOURCES")
        lines.append("-" * 17)
        for source, count in list(summary['top_sources'].items())[:5]:
            lines.append(f"  {source}: {count} articles")
        lines.append("")
        
        # Critical alerts
        if report_data['alerts']:
            lines.append("CRITICAL ALERTS")
            lines.append("-" * 15)
            for i, alert in enumerate(report_data['alerts'][:5], 1):
                lines.append(f"{i}. {alert['title']}")
                lines.append(f"   Score: {alert['conflict_score']:.3f}")
                lines.append(f"   Reasons: {', '.join(alert['alert_reasons'])}")
                lines.append(f"   URL: {alert['url']}")
                lines.append("")
        
        return "\n".join(lines)
    
    async def close(self):
        """Clean up resources."""
        if hasattr(self.conflict_crawler, 'close'):
            await self.conflict_crawler.close()


async def main():
    """Main demonstration function."""
    print("🔍 Horn of Africa Conflict Monitoring Demo")
    print("=" * 50)
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Search crawler components not available!")
        return
    
    print(f"Available search engines: {get_available_engines()}")
    print()
    
    # Initialize monitor
    monitor = HornOfAfricaConflictMonitor("./demo_results")
    
    try:
        # Monitor conflicts
        print("📡 Monitoring conflicts across Horn of Africa...")
        results = await monitor.monitor_conflicts(
            countries=['somalia', 'ethiopia', 'sudan'],
            time_range='week',
            max_results=30
        )
        
        if results:
            print(f"✅ Found {len(results)} conflict-related results")
            
            # Show sample results
            print("\n🔥 Top 5 Results:")
            for i, result in enumerate(results[:5], 1):
                print(f"{i}. {result.title}")
                print(f"   Score: {result.conflict_score:.3f}")
                print(f"   Engines: {', '.join(result.engines_found)}")
                if result.locations_mentioned:
                    locations = [loc['name'] for loc in result.locations_mentioned[:2]]
                    print(f"   Locations: {', '.join(locations)}")
                print()
            
            # Analyze keywords
            print("📊 Analyzing keyword effectiveness...")
            await monitor.analyze_keyword_effectiveness()
            
            # Generate report
            print("📄 Generating comprehensive report...")
            report = monitor.generate_conflict_report()
            
            print(f"✅ Report generated with {len(monitor.alerts)} alerts")
            
        else:
            print("⚠️  No results found")
    
    except Exception as e:
        logger.error(f"Error during monitoring: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await monitor.close()
    
    print("\n🎯 Demo completed!")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())