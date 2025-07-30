#!/usr/bin/env python3
"""
Basic Twitter Crawler Usage Examples
====================================

This script demonstrates basic usage of the Lindela Twitter Crawler package.
It shows how to perform simple searches, authenticate, and process results.

Author: Lindela Development Team
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add the package to the path (adjust as needed)
sys.path.append('/Users/nyimbiodero/src/pjs/lindela/src/lindela')

from packages_enhanced.crawlers.twitter_crawler import (
    TwitterCrawler, TwitterConfig, TwitterSearchEngine,
    SearchQuery, TweetFilter, DateRange, quick_search
)


async def example_1_quick_search():
    """Example 1: Quick search without authentication"""
    print("=== Example 1: Quick Search ===")
    
    try:
        # Simple search for conflict-related tweets
        results = await quick_search(
            query="armed conflict OR terrorism",
            max_results=20,
            language="en"
        )
        
        print(f"Found {len(results)} tweets")
        
        # Display first 3 tweets
        for i, tweet in enumerate(results[:3], 1):
            print(f"\n{i}. Tweet ID: {tweet.get('id', 'N/A')}")
            print(f"   User: @{tweet.get('username', 'N/A')}")
            print(f"   Text: {tweet.get('text', '')[:100]}...")
            print(f"   Created: {tweet.get('created_at', 'N/A')}")
            print(f"   Retweets: {tweet.get('retweet_count', 0)}")
    
    except Exception as e:
        print(f"Error in quick search: {e}")


async def example_2_authenticated_crawler():
    """Example 2: Using authenticated crawler (requires credentials)"""
    print("\n=== Example 2: Authenticated Crawler ===")
    
    # Get credentials from environment variables
    username = os.getenv('TWITTER_USERNAME')
    password = os.getenv('TWITTER_PASSWORD')
    email = os.getenv('TWITTER_EMAIL')
    
    if not (username and password):
        print("Skipping authenticated example - no credentials provided")
        print("Set TWITTER_USERNAME and TWITTER_PASSWORD environment variables to test")
        return
    
    try:
        # Create configuration
        config = TwitterConfig(
            username=username,
            password=password,
            email=email,
            rate_limit_requests_per_minute=20,  # Conservative rate limiting
            auto_save_session=True,
            session_file="example_session.pkl"
        )
        
        # Initialize crawler
        crawler = TwitterCrawler(config)
        await crawler.initialize()
        
        print("✅ Authentication successful!")
        
        # Search for tweets
        tweets = await crawler.search_tweets(
            query="Syria conflict",
            count=15,
            result_type="recent"
        )
        
        print(f"Retrieved {len(tweets)} tweets")
        
        # Get user information
        user_info = await crawler.get_user_by_username("UN")
        if user_info:
            print(f"\nUser Info - @{user_info['username']}:")
            print(f"  Display Name: {user_info['display_name']}")
            print(f"  Followers: {user_info['followers_count']:,}")
            print(f"  Verified: {user_info['verified']}")
        
        # Get crawler status
        status = crawler.get_status()
        print(f"\nCrawler Status: {status['status']}")
        print(f"Rate Limit Status: {status['rate_limit_status']['can_make_request']}")
        
        # Clean up
        await crawler.close()
        
    except Exception as e:
        print(f"Error in authenticated example: {e}")


async def example_3_advanced_search():
    """Example 3: Advanced search with filters"""
    print("\n=== Example 3: Advanced Search ===")
    
    try:
        # Create search engine
        search_engine = TwitterSearchEngine()
        
        # Configure date filter (last 3 days)
        date_filter = DateRange(
            start_date=datetime.now() - timedelta(days=3),
            end_date=datetime.now()
        )
        
        # Configure tweet filters
        tweet_filter = TweetFilter(
            exclude_retweets=True,  # Original content only
            min_followers=100,      # Users with at least 100 followers
            languages=["en"],       # English only
            date_range=date_filter,
            min_retweets=5         # Tweets with at least 5 retweets
        )
        
        # Create search query
        query = SearchQuery(
            query="humanitarian crisis OR refugee",
            max_results=30,
            tweet_filter=tweet_filter,
            sort_by="popular",
            deduplicate=True
        )
        
        # Execute search
        result = await search_engine.search(query)
        
        print(f"Found {len(result.tweets)} tweets")
        print(f"Search execution time: {result.execution_time:.2f} seconds")
        
        # Analyze results
        hashtags = result.get_hashtags()
        mentions = result.get_mentions()
        urls = result.get_urls()
        
        print(f"\nTop hashtags: {hashtags[:5]}")
        print(f"Top mentions: {mentions[:5]}")
        print(f"URLs found: {len(urls)}")
        
        # Filter by engagement
        high_engagement = result.filter_by_engagement(min_engagement=20)
        print(f"High engagement tweets: {len(high_engagement.tweets)}")
        
        # Display top tweet
        if result.tweets:
            top_tweet = result.tweets[0]
            print(f"\nTop Tweet:")
            print(f"  Text: {top_tweet.get('text', '')[:150]}...")
            print(f"  Engagement: {top_tweet.get('retweet_count', 0) + top_tweet.get('favorite_count', 0)}")
            print(f"  User: @{top_tweet.get('username', 'N/A')} ({top_tweet.get('user', {}).get('followers_count', 0):,} followers)")
        
    except Exception as e:
        print(f"Error in advanced search: {e}")


async def example_4_user_timeline():
    """Example 4: User timeline crawling (requires authentication)"""
    print("\n=== Example 4: User Timeline ===")
    
    username = os.getenv('TWITTER_USERNAME')
    password = os.getenv('TWITTER_PASSWORD')
    
    if not (username and password):
        print("Skipping user timeline example - no credentials provided")
        return
    
    try:
        # Create crawler
        config = TwitterConfig(username=username, password=password)
        crawler = TwitterCrawler(config)
        await crawler.initialize()
        
        # Get tweets from specific users
        users_to_check = ["UN", "Reuters", "BBCBreaking"]
        
        for user in users_to_check:
            try:
                # Get user info
                user_info = await crawler.get_user_by_username(user)
                if not user_info:
                    print(f"User @{user} not found")
                    continue
                
                # Get recent tweets
                tweets = await crawler.get_user_tweets(
                    username=user,
                    count=10,
                    include_retweets=False
                )
                
                print(f"\n@{user} ({user_info['followers_count']:,} followers):")
                
                for tweet in tweets[:3]:  # Show top 3 tweets
                    print(f"  • {tweet.get('text', '')[:100]}...")
                    print(f"    Engagement: {tweet.get('retweet_count', 0)} RT, {tweet.get('favorite_count', 0)} ❤️")
                
            except Exception as e:
                print(f"Error getting tweets for @{user}: {e}")
        
        await crawler.close()
        
    except Exception as e:
        print(f"Error in user timeline example: {e}")


async def example_5_data_export():
    """Example 5: Data processing and export"""
    print("\n=== Example 5: Data Export ===")
    
    try:
        from packages_enhanced.crawlers.twitter_crawler.data import (
            TwitterDataProcessor, TweetModel, ExportFormat
        )
        
        # Get some sample tweets
        results = await quick_search("conflict monitoring", max_results=10)
        
        # Create data processor
        processor = TwitterDataProcessor()
        
        # Process raw tweets into structured models
        tweet_models = await processor.process_raw_tweets(
            results,
            session_id="example_session",
            store=False  # Don't store in database for this example
        )
        
        print(f"Processed {len(tweet_models)} tweets")
        
        # Export to different formats
        json_file = processor.exporter.export_tweets(
            tweet_models,
            ExportFormat.JSON,
            "example_tweets.json"
        )
        
        csv_file = processor.exporter.export_tweets(
            tweet_models,
            ExportFormat.CSV,
            "example_tweets.csv"
        )
        
        print(f"Exported to:")
        print(f"  JSON: {json_file}")
        print(f"  CSV: {csv_file}")
        
        # Show processing stats
        stats = processor.get_processing_stats()
        print(f"\nProcessing Stats:")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Total errors: {stats['total_errors']}")
        print(f"  Last processed: {stats['last_processed']}")
        
        # Analyze a tweet model
        if tweet_models:
            tweet = tweet_models[0]
            print(f"\nSample Tweet Model:")
            print(f"  ID: {tweet.id}")
            print(f"  Text: {tweet.text[:100]}...")
            print(f"  User: @{tweet.username} ({tweet.user_followers_count:,} followers)")
            print(f"  Engagement Score: {tweet.engagement_score}")
            print(f"  Has Location: {tweet.has_location}")
            print(f"  Has Media: {tweet.has_media}")
            print(f"  Hashtags: {tweet.hashtags}")
        
    except ImportError as e:
        print(f"Data processing not available: {e}")
    except Exception as e:
        print(f"Error in data export example: {e}")


async def main():
    """Run all examples"""
    print("🐦 Lindela Twitter Crawler - Basic Usage Examples")
    print("=" * 50)
    
    # Run examples
    await example_1_quick_search()
    await example_2_authenticated_crawler()
    await example_3_advanced_search()
    await example_4_user_timeline()
    await example_5_data_export()
    
    print("\n✅ All examples completed!")
    print("\nNote: Some examples require Twitter credentials set as environment variables:")
    print("  export TWITTER_USERNAME='your_username'")
    print("  export TWITTER_PASSWORD='your_password'")
    print("  export TWITTER_EMAIL='your_email@example.com'")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())