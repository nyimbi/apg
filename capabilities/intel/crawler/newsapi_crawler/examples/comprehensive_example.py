#!/usr/bin/env python3
"""
Comprehensive NewsAPI Crawler Example
====================================

This example demonstrates the full capabilities of the NewsAPI crawler package,
including advanced searching, caching, batch processing, and content analysis.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('newsapi_crawler.log')
    ]
)
logger = logging.getLogger("NewsAPICrawlerExample")

# Import our package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
try:
    from packages_enhanced.crawlers.newsapi_crawler import (
        create_advanced_client,
        create_batch_client,
        ArticleCollection,
        NewsArticle,
        calculate_relevance_score,
        filter_articles_by_keywords,
        extract_locations
    )
    from packages_enhanced.crawlers.newsapi_crawler.parsers import (
        ArticleParser,
        ContentExtractor,
        EventDetector
    )
    from packages_enhanced.crawlers.newsapi_crawler.config import (
        create_config_with_defaults
    )
except ImportError as e:
    logger.error(f"Error importing NewsAPI crawler: {e}")
    print(f"Error: {e}")
    print("Make sure the packages_enhanced directory is in your Python path.")
    sys.exit(1)


async def search_horn_of_africa_conflicts():
    """Search for conflict news in the Horn of Africa."""
    logger.info("Starting comprehensive NewsAPI crawler example")

    # Create client with cache directory
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    client = await create_advanced_client(
        cache_dir=cache_dir,
        cache_ttl=3600  # 1 hour
    )

    try:
        # Set up search parameters
        countries = ["Ethiopia", "Somalia", "Sudan", "South Sudan", "Kenya"]
        keywords = ["conflict", "violence", "war", "peace", "agreement", "protest"]

        # Date range for last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        logger.info(f"Searching for conflict news in Horn of Africa from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Search for each country
        all_articles = []
        for country in countries:
            # Create search query combining country and conflict keywords
            query = f"{country} AND ({' OR '.join(keywords)})"
            logger.info(f"Searching for: {query}")

            try:
                articles = await client.search_with_date_range(
                    query=query,
                    start_date=start_date,
                    end_date=end_date,
                    language="en",
                    sort_by="relevancy"
                )

                logger.info(f"Found {len(articles)} articles for {country}")
                all_articles.extend(articles)

                # Small delay between requests
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error searching for {country}: {e}")

        # Remove duplicates by URL
        unique_urls = set()
        unique_articles = []
        for article in all_articles:
            if article['url'] not in unique_urls:
                unique_urls.add(article['url'])
                unique_articles.append(article)

        logger.info(f"Found {len(unique_articles)} unique articles")

        # Create a properly structured collection
        collection = ArticleCollection(
            articles=[NewsArticle.from_api_response(a) for a in unique_articles],
            total_results=len(unique_articles),
            status="ok"
        )

        # Extract full text and perform content analysis
        await process_articles(collection)

        # Save results
        save_results(collection)

        return collection

    finally:
        # Close the client
        await client.close()


async def process_articles(collection: ArticleCollection):
    """Process articles to extract full text and analyze content."""
    logger.info(f"Processing {len(collection.articles)} articles")

    # Create parsers
    article_parser = ArticleParser()
    content_extractor = ContentExtractor()
    event_detector = EventDetector()

    # Process each article
    for i, article in enumerate(collection.articles):
        logger.info(f"Processing article {i+1}/{len(collection.articles)}: {article.title}")

        # Skip if no URL
        if not article.url:
            continue

        try:
            # Extract full text if not already present
            if not article.full_text and article.url:
                full_content = await content_extractor.extract_from_url(article.url)
                if 'text' in full_content and full_content['text']:
                    article.full_text = full_content['text']

                    # Update metadata if available
                    if 'authors' in full_content and full_content['authors']:
                        article.author = ', '.join(full_content['authors'])

                    if 'keywords' in full_content and full_content['keywords']:
                        article.keywords = full_content['keywords']

            # Skip further processing if no content
            if not article.full_text and not article.content and not article.description:
                continue

            # Extract content for processing
            content_text = article.full_text or article.content or article.description or ''

            # Extract entities and locations
            if hasattr(article_parser, 'extract_entities') and article_parser.use_nlp:
                entities = article_parser.extract_entities(content_text)
                if entities:
                    article.entities = entities

                # Extract locations
                locations = article_parser.extract_locations(content_text)
                if locations:
                    article.locations = locations
            else:
                # Fallback to simple location extraction
                location_names = extract_locations(content_text)
                if location_names:
                    article.locations = [{"text": loc, "type": "LOC"} for loc in location_names]

            # Extract keywords if not already present
            if not article.keywords and hasattr(article_parser, 'extract_keywords'):
                keywords = article_parser.extract_keywords(content_text)
                if keywords:
                    article.keywords = keywords

            # Detect events
            events = event_detector.detect_events(content_text)
            if events:
                article.metadata["events"] = events

            # Small delay to prevent system overload
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error processing article {article.url}: {e}")


def save_results(collection: ArticleCollection):
    """Save processed results to files."""
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full article collection
    collection_file = os.path.join(output_dir, f"articles_{timestamp}.json")
    with open(collection_file, 'w') as f:
        json.dump(collection.to_dict(), f, indent=2)
    logger.info(f"Saved article collection to {collection_file}")

    # Extract events into a separate file
    events = []
    for article in collection.articles:
        if "events" in article.metadata:
            for event in article.metadata["events"]:
                event_data = event.copy()
                event_data["article_title"] = article.title
                event_data["article_url"] = article.url
                event_data["source"] = article.source.name if hasattr(article.source, 'name') else str(article.source)
                event_data["published_at"] = str(article.published_at) if article.published_at else None
                events.append(event_data)

    if events:
        events_file = os.path.join(output_dir, f"events_{timestamp}.json")
        with open(events_file, 'w') as f:
            json.dump(events, f, indent=2)
        logger.info(f"Saved {len(events)} events to {events_file}")

    # Create a summary file
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_articles": len(collection.articles),
        "total_events": len(events),
        "sources": {}
    }

    # Count articles by source
    for article in collection.articles:
        source_name = article.source.name if hasattr(article.source, 'name') else str(article.source)
        if source_name not in summary["sources"]:
            summary["sources"][source_name] = 0
        summary["sources"][source_name] += 1

    summary_file = os.path.join(output_dir, f"summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")


async def batch_search_example():
    """Example of using batch client for multiple queries."""
    logger.info("Starting batch search example")

    # Create batch client
    batch_client = await create_batch_client()

    try:
        # Define multiple queries
        queries = [
            "Ethiopia peace agreement",
            "Somalia security situation",
            "Sudan conflict",
            "South Sudan violence",
            "Kenya election"
        ]

        # Process queries in batch
        results = await batch_client.process_queries(
            queries=queries,
            language="en",
            sort_by="relevancy",
            from_param=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        )

        logger.info("Batch search results:")
        for query, articles in results.items():
            logger.info(f"  {query}: {len(articles)} articles")

        return results

    finally:
        await batch_client.close()


async def main():
    """Main entry point."""
    try:
        # Run the comprehensive example
        collection = await search_horn_of_africa_conflicts()

        # Print summary
        print("\n" + "="*50)
        print(f"Found {len(collection.articles)} articles")

        # Count articles by country
        country_counts = {}
        for article in collection.articles:
            for location in getattr(article, 'locations', []):
                location_text = location.get('text', '') if isinstance(location, dict) else str(location)
                for country in ["Ethiopia", "Somalia", "Sudan", "South Sudan", "Kenya"]:
                    if country in location_text:
                        if country not in country_counts:
                            country_counts[country] = 0
                        country_counts[country] += 1

        print("\nArticles by country:")
        for country, count in country_counts.items():
            print(f"  {country}: {count}")

        # Count events
        event_count = sum(1 for article in collection.articles if "events" in getattr(article, 'metadata', {}))
        print(f"\nDetected {event_count} conflict events")
        print("="*50)

        # Optionally run batch example
        # await batch_search_example()

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSearch interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
