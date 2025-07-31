#!/usr/bin/env python3
"""
Conflict Monitoring Examples
============================

This script demonstrates conflict monitoring capabilities of the Twitter Crawler.
It shows how to set up real-time monitoring, generate alerts, and analyze conflict data.

Author: Lindela Development Team
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add the package to the path (adjust as needed)
sys.path.append('/Users/nyimbiodero/src/pjs/lindela/src/lindela')

from packages_enhanced.crawlers.twitter_crawler import (
    ConflictMonitor, MonitoringConfig, AlertLevel, EventType,
    TwitterAnalyzer, quick_conflict_analysis, quick_sentiment_analysis
)
from packages_enhanced.crawlers.twitter_crawler.search import ConflictSearchTemplates
from packages_enhanced.crawlers.twitter_crawler.data import TweetModel


async def example_1_conflict_search_templates():
    """Example 1: Using conflict search templates"""
    print("=== Example 1: Conflict Search Templates ===")
    
    try:
        from packages_enhanced.crawlers.twitter_crawler import TwitterSearchEngine
        
        search_engine = TwitterSearchEngine()
        
        # Test different conflict event types
        event_types = [
            ("armed_conflict", "Syria"),
            ("refugee_crisis", "Somalia"),
            ("terrorism", None),  # Global terrorism search
            ("protest", "Hong Kong")
        ]
        
        for event_type, location in event_types:
            print(f"\n--- Searching for {event_type} in {location or 'global'} ---")
            
            try:
                result = await search_engine.search_conflict_events(
                    event_type=event_type,
                    location=location,
                    max_results=10
                )
                
                print(f"Found {len(result.tweets)} tweets")
                
                # Show sample tweets
                for i, tweet in enumerate(result.tweets[:2], 1):
                    print(f"  {i}. {tweet.get('text', '')[:80]}...")
                    print(f"     User: @{tweet.get('username', 'N/A')}")
                    print(f"     Engagement: {tweet.get('retweet_count', 0)} RT")
                
            except Exception as e:
                print(f"Error searching for {event_type}: {e}")
    
    except ImportError:
        print("Search engine not available")
    except Exception as e:
        print(f"Error in conflict search example: {e}")


async def example_2_conflict_analysis():
    """Example 2: Conflict relevance analysis"""
    print("\n=== Example 2: Conflict Analysis ===")
    
    # Sample texts to analyze
    test_texts = [
        "Breaking: Armed clashes reported in Damascus between government forces and rebels",
        "Humanitarian aid convoy attacked while delivering supplies to refugee camp",
        "Terrorist bombing kills 15 people in shopping center",
        "Peaceful protest turned violent as police used tear gas",
        "Beautiful sunset over the mountains today #nature",
        "URGENT: Explosion heard in downtown area, multiple casualties reported",
        "UN calls for immediate ceasefire in ongoing conflict",
        "Families flee their homes as fighting intensifies"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: {text[:60]}...")
        
        # Analyze conflict relevance
        conflict_result = quick_conflict_analysis(text)
        
        print(f"   Conflict-related: {conflict_result.is_conflict_related}")
        print(f"   Category: {conflict_result.category.value}")
        print(f"   Confidence: {conflict_result.confidence:.3f}")
        print(f"   Urgency: {conflict_result.urgency_score:.3f}")
        print(f"   Threat level: {conflict_result.threat_level}")
        
        if conflict_result.keywords_found:
            print(f"   Keywords: {', '.join(conflict_result.keywords_found[:3])}")
        
        if conflict_result.entities['locations']:
            print(f"   Locations: {', '.join(conflict_result.entities['locations'])}")
        
        # Also do sentiment analysis
        sentiment_result = quick_sentiment_analysis(text)
        print(f"   Sentiment: {sentiment_result.category.value} ({sentiment_result.polarity:.3f})")


async def example_3_monitoring_setup():
    """Example 3: Setting up conflict monitoring (simulation)"""
    print("\n=== Example 3: Monitoring Setup ===")
    
    try:
        # Create monitoring configuration
        config = MonitoringConfig(
            keywords=["armed conflict", "terrorism", "bombing", "attack"],
            hashtags=["#Syria", "#Ukraine", "#Yemen"],
            locations=["Syria", "Ukraine", "Yemen", "Somalia"],
            languages=["en", "ar"],
            alert_threshold=5,  # Low threshold for demo
            time_window_minutes=30,
            critical_threshold=15,
            check_interval_seconds=60,  # Check every minute for demo
            alert_cooldown_minutes=10,
            max_alerts_per_hour=20
        )
        
        print("Monitoring Configuration:")
        print(f"  Keywords: {config.keywords}")
        print(f"  Locations: {config.locations}")
        print(f"  Alert threshold: {config.alert_threshold} tweets")
        print(f"  Time window: {config.time_window_minutes} minutes")
        print(f"  Check interval: {config.check_interval_seconds} seconds")
        
        # Create monitor (don't start for this example)
        monitor = ConflictMonitor(config)
        
        # Set up alert handlers
        def handle_critical_alert(alert):
            print(f"\n🚨 CRITICAL ALERT 🚨")
            print(f"Title: {alert.title}")
            print(f"Location: {alert.location}")
            print(f"Tweet count: {alert.tweet_count}")
            print(f"Keywords: {', '.join(alert.keywords_triggered)}")
            print(f"Urgency: {alert.urgency_score:.3f}")
            print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        def handle_high_alert(alert):
            print(f"\n⚠️  HIGH ALERT")
            print(f"Title: {alert.title}")
            print(f"Tweet count: {alert.tweet_count}")
        
        def handle_medium_alert(alert):
            print(f"\n📊 Medium Alert: {alert.title} ({alert.tweet_count} tweets)")
        
        # Register callbacks
        monitor.alert_system.add_alert_callback(AlertLevel.CRITICAL, handle_critical_alert)
        monitor.alert_system.add_alert_callback(AlertLevel.HIGH, handle_high_alert)
        monitor.alert_system.add_alert_callback(AlertLevel.MEDIUM, handle_medium_alert)
        
        print("\n✅ Alert handlers configured")
        print("💡 In a real scenario, you would call:")
        print("   await monitor.start_monitoring()")
        print("   # Monitor runs continuously until stopped")
        print("   await monitor.stop_monitoring()")
        
    except Exception as e:
        print(f"Error in monitoring setup: {e}")


async def example_4_alert_simulation():
    """Example 4: Simulate alert generation"""
    print("\n=== Example 4: Alert Simulation ===")
    
    try:
        from packages_enhanced.crawlers.twitter_crawler.monitoring import AlertSystem
        
        config = MonitoringConfig(alert_threshold=3, critical_threshold=10)
        alert_system = AlertSystem(config)
        
        # Simulate finding conflict tweets
        simulated_tweets = [
            {
                'id': f'tweet_{i}',
                'text': f'Breaking: Armed conflict escalates in region {i}',
                'created_at': datetime.now(),
                'username': f'reporter_{i}',
                'retweet_count': 10 + i * 5,
                'favorite_count': 20 + i * 3,
                'reply_count': 5 + i
            }
            for i in range(1, 8)
        ]
        
        # Test different alert scenarios
        scenarios = [
            (EventType.ARMED_CONFLICT, ["armed conflict", "violence"], 5, "Damascus"),
            (EventType.TERRORISM, ["bombing", "attack"], 12, "Paris"),
            (EventType.PROTESTS, ["protest", "unrest"], 8, "Hong Kong"),
            (EventType.REFUGEE_CRISIS, ["refugees", "displaced"], 6, "Somalia")
        ]
        
        for event_type, keywords, tweet_count, location in scenarios:
            tweets_subset = simulated_tweets[:tweet_count]
            
            alert = alert_system.generate_alert(
                event_type=event_type,
                keywords_triggered=keywords,
                tweet_count=tweet_count,
                tweets=tweets_subset,
                location=location
            )
            
            if alert:
                print(f"\n📢 Generated Alert:")
                print(f"   ID: {alert.id}")
                print(f"   Level: {alert.level.value.upper()}")
                print(f"   Type: {alert.event_type.value}")
                print(f"   Title: {alert.title}")
                print(f"   Location: {alert.location}")
                print(f"   Tweet count: {alert.tweet_count}")
                print(f"   Engagement score: {alert.engagement_score}")
                print(f"   Description: {alert.description[:100]}...")
            else:
                print(f"No alert generated for {event_type.value} (cooldown or limits)")
        
        # Show alert statistics
        active_alerts = alert_system.get_active_alerts()
        critical_alerts = alert_system.get_alerts_by_level(AlertLevel.CRITICAL)
        high_alerts = alert_system.get_alerts_by_level(AlertLevel.HIGH)
        
        print(f"\n📊 Alert Statistics:")
        print(f"   Total alerts: {len(alert_system.alerts)}")
        print(f"   Active alerts: {len(active_alerts)}")
        print(f"   Critical alerts: {len(critical_alerts)}")
        print(f"   High alerts: {len(high_alerts)}")
        
        # Acknowledge an alert
        if alert_system.alerts:
            alert_to_ack = alert_system.alerts[0]
            success = alert_system.acknowledge_alert(alert_to_ack.id, "analyst_1")
            print(f"   Alert acknowledgment: {'✅' if success else '❌'}")
    
    except Exception as e:
        print(f"Error in alert simulation: {e}")


async def example_5_comprehensive_analysis():
    """Example 5: Comprehensive tweet analysis"""
    print("\n=== Example 5: Comprehensive Analysis ===")
    
    try:
        # Create sample tweets for analysis
        sample_tweets_data = [
            {
                'id': '1',
                'text': 'BREAKING: Armed clashes in Damascus as government forces advance',
                'created_at': datetime.now(),
                'user': {'screen_name': 'syria_reporter', 'followers_count': 5000, 'verified': False},
                'retweet_count': 45,
                'favorite_count': 78,
                'reply_count': 12,
                'hashtags': ['Syria', 'Damascus', 'Breaking'],
                'lang': 'en'
            },
            {
                'id': '2', 
                'text': 'Humanitarian aid finally reaches displaced families in refugee camp',
                'created_at': datetime.now(),
                'user': {'screen_name': 'aid_worker', 'followers_count': 1200, 'verified': True},
                'retweet_count': 23,
                'favorite_count': 89,
                'reply_count': 15,
                'hashtags': ['humanitarian', 'refugees', 'aid'],
                'lang': 'en'
            },
            {
                'id': '3',
                'text': 'Peaceful protest in downtown turned chaotic after police intervention',
                'created_at': datetime.now(),
                'user': {'screen_name': 'local_news', 'followers_count': 15000, 'verified': True},
                'retweet_count': 67,
                'favorite_count': 134,
                'reply_count': 28,
                'hashtags': ['protest', 'police', 'downtown'],
                'lang': 'en'
            }
        ]
        
        # Convert to TweetModel objects
        tweet_models = []
        for data in sample_tweets_data:
            tweet_model = TweetModel.from_raw_tweet(data)
            tweet_models.append(tweet_model)
        
        # Create analyzer
        analyzer = TwitterAnalyzer()
        
        # Analyze each tweet
        print("Individual Tweet Analysis:")
        analyses = []
        
        for i, tweet in enumerate(tweet_models, 1):
            print(f"\n--- Tweet {i} ---")
            print(f"Text: {tweet.text}")
            
            analysis = analyzer.analyze_tweet(tweet)
            analyses.append({'tweet_id': tweet.id, 'analysis': analysis})
            
            # Show sentiment
            if 'sentiment' in analysis:
                sentiment = analysis['sentiment']
                print(f"Sentiment: {sentiment.category.value} (polarity: {sentiment.polarity:.3f})")
                if sentiment.emotions:
                    top_emotions = sorted(sentiment.emotions.items(), key=lambda x: x[1], reverse=True)[:2]
                    print(f"Top emotions: {', '.join([f'{e}:{v:.2f}' for e, v in top_emotions])}")
            
            # Show conflict analysis
            if 'conflict' in analysis:
                conflict = analysis['conflict']
                print(f"Conflict: {conflict.is_conflict_related} ({conflict.category.value})")
                print(f"Urgency: {conflict.urgency_score:.3f}, Threat: {conflict.threat_level}")
                if conflict.keywords_found:
                    print(f"Keywords: {', '.join(conflict.keywords_found[:3])}")
        
        # Generate summary
        summary = analyzer.get_analysis_summary(analyses)
        
        print(f"\n📊 Analysis Summary:")
        print(f"Total tweets analyzed: {summary['total_tweets']}")
        
        if 'sentiment_summary' in summary:
            sentiment_summary = summary['sentiment_summary']
            print(f"\nSentiment Analysis:")
            print(f"  Average polarity: {sentiment_summary.get('average_polarity', 0):.3f}")
            print(f"  Positive ratio: {sentiment_summary.get('positive_ratio', 0):.1%}")
            print(f"  Negative ratio: {sentiment_summary.get('negative_ratio', 0):.1%}")
            print(f"  Neutral ratio: {sentiment_summary.get('neutral_ratio', 0):.1%}")
            
            if 'category_distribution' in sentiment_summary:
                print(f"  Category distribution: {sentiment_summary['category_distribution']}")
        
        if 'conflict_summary' in summary:
            conflict_summary = summary['conflict_summary']
            print(f"\nConflict Analysis:")
            print(f"  Conflict-related tweets: {conflict_summary.get('total_conflict_related', 0)}")
            print(f"  Conflict ratio: {conflict_summary.get('conflict_ratio', 0):.1%}")
            print(f"  Average urgency: {conflict_summary.get('average_urgency', 0):.3f}")
            print(f"  High urgency count: {conflict_summary.get('high_urgency_count', 0)}")
            
            if 'category_distribution' in conflict_summary:
                print(f"  Category distribution: {conflict_summary['category_distribution']}")
            
            if 'threat_level_distribution' in conflict_summary:
                print(f"  Threat levels: {conflict_summary['threat_level_distribution']}")
    
    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")


async def example_6_real_time_simulation():
    """Example 6: Real-time monitoring simulation"""
    print("\n=== Example 6: Real-time Monitoring Simulation ===")
    
    print("🚧 This example simulates real-time monitoring behavior")
    print("In a real scenario, this would continuously monitor Twitter")
    
    try:
        # Simulate monitoring cycle
        config = MonitoringConfig(
            keywords=["urgent", "breaking", "crisis"],
            alert_threshold=3,
            check_interval_seconds=5
        )
        
        monitor = ConflictMonitor(config)
        
        # Simulate tweet streams over time
        tweet_streams = [
            # First check - normal activity
            {
                'timestamp': datetime.now(),
                'tweets': [
                    {'text': 'Regular news update about local events', 'urgency': 0.2},
                    {'text': 'Weather forecast for tomorrow', 'urgency': 0.1}
                ]
            },
            # Second check - increasing activity
            {
                'timestamp': datetime.now() + timedelta(minutes=5),
                'tweets': [
                    {'text': 'URGENT: Situation developing downtown', 'urgency': 0.8},
                    {'text': 'Breaking: Emergency services responding', 'urgency': 0.9},
                    {'text': 'Crisis management team activated', 'urgency': 0.7}
                ]
            },
            # Third check - critical activity
            {
                'timestamp': datetime.now() + timedelta(minutes=10),
                'tweets': [
                    {'text': 'CRITICAL: Major incident confirmed', 'urgency': 0.95},
                    {'text': 'Breaking: Multiple casualties reported', 'urgency': 0.9},
                    {'text': 'URGENT: Evacuation orders issued', 'urgency': 0.85},
                    {'text': 'Emergency: All units respond immediately', 'urgency': 0.98},
                    {'text': 'Crisis escalating rapidly', 'urgency': 0.8}
                ]
            }
        ]
        
        print("\nSimulating monitoring checks:")
        
        for i, stream in enumerate(tweet_streams, 1):
            print(f"\n--- Check {i} at {stream['timestamp'].strftime('%H:%M:%S')} ---")
            print(f"Found {len(stream['tweets'])} tweets")
            
            # Analyze urgency
            high_urgency = [t for t in stream['tweets'] if t['urgency'] > 0.7]
            critical_urgency = [t for t in stream['tweets'] if t['urgency'] > 0.9]
            
            if critical_urgency:
                print(f"🚨 CRITICAL: {len(critical_urgency)} critical tweets detected!")
                for tweet in critical_urgency:
                    print(f"   • {tweet['text']} (urgency: {tweet['urgency']:.2f})")
            elif high_urgency:
                print(f"⚠️  HIGH: {len(high_urgency)} high-urgency tweets detected")
                for tweet in high_urgency:
                    print(f"   • {tweet['text']} (urgency: {tweet['urgency']:.2f})")
            else:
                print("✅ Normal activity levels")
            
            # Simulate alert generation
            if len(stream['tweets']) >= config.alert_threshold:
                if len(critical_urgency) >= 2:
                    print("   → Generated CRITICAL alert")
                elif len(high_urgency) >= 2:
                    print("   → Generated HIGH alert")
                elif len(stream['tweets']) >= 5:
                    print("   → Generated MEDIUM alert")
        
        print("\n💡 In production, monitoring would:")
        print("   • Run continuously in the background")
        print("   • Send real alerts via email/SMS/Slack")
        print("   • Store all data in database")
        print("   • Generate periodic reports")
        print("   • Integrate with mapping systems")
    
    except Exception as e:
        print(f"Error in real-time simulation: {e}")


async def main():
    """Run all conflict monitoring examples"""
    print("🛡️  Lindela Twitter Crawler - Conflict Monitoring Examples")
    print("=" * 60)
    
    # Run examples
    await example_1_conflict_search_templates()
    await example_2_conflict_analysis()
    await example_3_monitoring_setup()
    await example_4_alert_simulation()
    await example_5_comprehensive_analysis()
    await example_6_real_time_simulation()
    
    print("\n✅ All conflict monitoring examples completed!")
    print("\n🔒 Security Note: For production use:")
    print("   • Use secure credential storage")
    print("   • Implement proper logging and monitoring")
    print("   • Set up database backups")
    print("   • Configure alert delivery systems")
    print("   • Implement access controls")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())