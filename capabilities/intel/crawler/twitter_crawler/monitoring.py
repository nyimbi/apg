"""
Twitter Monitoring Module
=========================

Real-time Twitter monitoring and alerting system for conflict detection.
Provides continuous monitoring, alert generation, and trend analysis
specifically designed for conflict monitoring and early warning systems.

Author: Lindela Development Team
Version: 2.0.0
License: MIT
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable, Set
from enum import Enum
from datetime import datetime, timedelta
import json
import re
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from .core import TwitterCrawler, TwitterConfig, CrawlerError
    from .search import TwitterSearchEngine, SearchQuery, QueryBuilder, ConflictSearchTemplates
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    logger.warning("Core Twitter modules not available")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class AlertLevel(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MonitoringStatus(Enum):
    """Monitoring system status"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class EventType(Enum):
    """Types of monitored events"""
    ARMED_CONFLICT = "armed_conflict"
    TERRORISM = "terrorism"
    PROTESTS = "protests"
    NATURAL_DISASTER = "natural_disaster"
    REFUGEE_CRISIS = "refugee_crisis"
    HUMANITARIAN_CRISIS = "humanitarian_crisis"
    POLITICAL_UNREST = "political_unrest"
    CYBER_ATTACK = "cyber_attack"
    CUSTOM = "custom"


@dataclass
class MonitoringConfig:
    """Configuration for Twitter monitoring"""
    # Core monitoring settings
    keywords: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    users_to_monitor: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=lambda: ["en"])
    
    # Alert thresholds
    alert_threshold: int = 10  # Number of tweets to trigger alert
    time_window_minutes: int = 60  # Time window for threshold
    critical_threshold: int = 50  # Critical alert threshold
    
    # Monitoring intervals
    check_interval_seconds: int = 300  # 5 minutes
    trend_analysis_interval_minutes: int = 60  # 1 hour
    
    # Alert settings
    alert_cooldown_minutes: int = 30
    max_alerts_per_hour: int = 10
    enable_escalation: bool = True
    
    # Data retention
    max_stored_tweets: int = 10000
    data_retention_hours: int = 72
    
    # Advanced settings
    sentiment_analysis: bool = True
    network_analysis: bool = False
    geographic_clustering: bool = True
    
    # Filters
    min_followers_threshold: int = 100
    verified_users_weight: float = 2.0
    exclude_retweets: bool = False


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    level: AlertLevel
    event_type: EventType
    title: str
    description: str
    timestamp: datetime
    location: Optional[str] = None
    keywords_triggered: List[str] = field(default_factory=list)
    tweet_count: int = 0
    engagement_score: float = 0.0
    related_tweets: List[str] = field(default_factory=list)  # Tweet IDs
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'id': self.id,
            'level': self.level.value,
            'event_type': self.event_type.value,
            'title': self.title,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'location': self.location,
            'keywords_triggered': self.keywords_triggered,
            'tweet_count': self.tweet_count,
            'engagement_score': self.engagement_score,
            'related_tweets': self.related_tweets,
            'metadata': self.metadata,
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None
        }


@dataclass
class TrendData:
    """Trend analysis data"""
    keyword: str
    location: Optional[str]
    tweet_count: int
    engagement_total: int
    sentiment_score: float
    velocity: float  # Tweets per hour
    acceleration: float  # Change in velocity
    time_window: timedelta
    top_tweets: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def trend_strength(self) -> float:
        """Calculate trend strength score"""
        base_score = min(self.tweet_count / 100.0, 1.0)  # Normalize to 0-1
        velocity_factor = min(self.velocity / 50.0, 2.0)  # Up to 2x multiplier
        engagement_factor = min(self.engagement_total / 1000.0, 1.5)  # Up to 1.5x multiplier
        
        return base_score * velocity_factor * engagement_factor


class AlertSystem:
    """Alert generation and management system"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.alert_counts: Dict[str, int] = defaultdict(int)
        self.last_alert_times: Dict[str, datetime] = {}
        self.callbacks: Dict[AlertLevel, List[Callable]] = {
            level: [] for level in AlertLevel
        }
        
    def add_alert_callback(self, level: AlertLevel, callback: Callable):
        """Add callback for specific alert level"""
        self.callbacks[level].append(callback)
    
    def generate_alert(
        self,
        event_type: EventType,
        keywords_triggered: List[str],
        tweet_count: int,
        tweets: List[Dict[str, Any]],
        location: Optional[str] = None
    ) -> Optional[Alert]:
        """Generate alert based on detected patterns"""
        # Check cooldown
        alert_key = f"{event_type.value}_{location or 'global'}"
        last_alert = self.last_alert_times.get(alert_key)
        if last_alert and (datetime.now() - last_alert).seconds < self.config.alert_cooldown_minutes * 60:
            return None
        
        # Check hourly limit
        hour_key = datetime.now().strftime('%Y%m%d%H')
        if self.alert_counts[hour_key] >= self.config.max_alerts_per_hour:
            return None
        
        # Determine alert level
        level = self._determine_alert_level(tweet_count, tweets)
        
        # Calculate engagement score
        engagement_score = sum(
            tweet.get('retweet_count', 0) + 
            tweet.get('favorite_count', 0) + 
            tweet.get('reply_count', 0)
            for tweet in tweets
        )
        
        # Create alert
        alert = Alert(
            id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{event_type.value}",
            level=level,
            event_type=event_type,
            title=self._generate_alert_title(event_type, location, keywords_triggered),
            description=self._generate_alert_description(event_type, tweet_count, location, tweets),
            timestamp=datetime.now(),
            location=location,
            keywords_triggered=keywords_triggered,
            tweet_count=tweet_count,
            engagement_score=engagement_score,
            related_tweets=[tweet.get('id', '') for tweet in tweets[:10]]  # Top 10 tweets
        )
        
        # Store alert
        self.alerts.append(alert)
        self.alert_history.append(alert)
        self.last_alert_times[alert_key] = datetime.now()
        self.alert_counts[hour_key] += 1
        
        # Trigger callbacks
        self._trigger_callbacks(alert)
        
        # Cleanup old alerts
        self._cleanup_old_alerts()
        
        return alert
    
    def _determine_alert_level(self, tweet_count: int, tweets: List[Dict[str, Any]]) -> AlertLevel:
        """Determine alert severity level"""
        if tweet_count >= self.config.critical_threshold:
            return AlertLevel.CRITICAL
        elif tweet_count >= self.config.alert_threshold * 2:
            return AlertLevel.HIGH
        elif tweet_count >= self.config.alert_threshold:
            return AlertLevel.MEDIUM
        else:
            return AlertLevel.LOW
    
    def _generate_alert_title(
        self,
        event_type: EventType,
        location: Optional[str],
        keywords: List[str]
    ) -> str:
        """Generate alert title"""
        location_str = f" in {location}" if location else ""
        keywords_str = ", ".join(keywords[:3])  # First 3 keywords
        
        return f"{event_type.value.replace('_', ' ').title()}{location_str} - {keywords_str}"
    
    def _generate_alert_description(
        self,
        event_type: EventType,
        tweet_count: int,
        location: Optional[str],
        tweets: List[Dict[str, Any]]
    ) -> str:
        """Generate alert description"""
        location_str = f" in {location}" if location else ""
        
        # Get top hashtags and mentions
        hashtags = set()
        mentions = set()
        for tweet in tweets[:20]:  # Analyze top 20 tweets
            hashtags.update(tweet.get('hashtags', []))
            text = tweet.get('text', '')
            mentions.update(re.findall(r'@(\w+)', text))
        
        top_hashtags = list(hashtags)[:5]
        top_mentions = list(mentions)[:5]
        
        description = f"Detected {tweet_count} tweets related to {event_type.value.replace('_', ' ')}{location_str}."
        
        if top_hashtags:
            description += f" Top hashtags: {', '.join(f'#{tag}' for tag in top_hashtags)}."
        
        if top_mentions:
            description += f" Key mentions: {', '.join(f'@{mention}' for mention in top_mentions)}."
        
        return description
    
    def _trigger_callbacks(self, alert: Alert):
        """Trigger registered callbacks for alert"""
        for callback in self.callbacks[alert.level]:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def _cleanup_old_alerts(self):
        """Remove old alerts to prevent memory issues"""
        cutoff_time = datetime.now() - timedelta(hours=self.config.data_retention_hours)
        
        # Keep recent alerts
        self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]
        
        # Limit history size
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unacknowledged) alerts"""
        return [alert for alert in self.alerts if not alert.acknowledged]
    
    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """Get alerts by severity level"""
        return [alert for alert in self.alerts if alert.level == level]


class TrendAnalyzer:
    """Analyzes trends in Twitter data"""
    
    def __init__(self):
        self.trend_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=24))  # 24 hours
        
    def analyze_trends(
        self,
        keywords: List[str],
        tweets_by_keyword: Dict[str, List[Dict[str, Any]]],
        time_window: timedelta = timedelta(hours=1)
    ) -> List[TrendData]:
        """Analyze trends for given keywords"""
        trends = []
        
        for keyword in keywords:
            tweets = tweets_by_keyword.get(keyword, [])
            
            if not tweets:
                continue
            
            # Calculate metrics
            tweet_count = len(tweets)
            engagement_total = sum(
                tweet.get('retweet_count', 0) + 
                tweet.get('favorite_count', 0) + 
                tweet.get('reply_count', 0)
                for tweet in tweets
            )
            
            # Calculate velocity (tweets per hour)
            velocity = tweet_count / (time_window.total_seconds() / 3600)
            
            # Calculate acceleration (change in velocity)
            acceleration = self._calculate_acceleration(keyword, velocity)
            
            # Simple sentiment analysis
            sentiment_score = self._calculate_sentiment(tweets)
            
            # Get top tweets by engagement
            top_tweets = sorted(
                tweets,
                key=lambda x: x.get('retweet_count', 0) + x.get('favorite_count', 0),
                reverse=True
            )[:5]
            
            trend = TrendData(
                keyword=keyword,
                location=None,  # Could be enhanced to analyze by location
                tweet_count=tweet_count,
                engagement_total=engagement_total,
                sentiment_score=sentiment_score,
                velocity=velocity,
                acceleration=acceleration,
                time_window=time_window,
                top_tweets=top_tweets
            )
            
            trends.append(trend)
            
            # Store for acceleration calculation
            self.trend_history[keyword].append(velocity)
        
        return sorted(trends, key=lambda x: x.trend_strength, reverse=True)
    
    def _calculate_acceleration(self, keyword: str, current_velocity: float) -> float:
        """Calculate velocity acceleration"""
        history = self.trend_history[keyword]
        
        if len(history) < 2:
            return 0.0
        
        previous_velocity = history[-1] if history else 0.0
        return current_velocity - previous_velocity
    
    def _calculate_sentiment(self, tweets: List[Dict[str, Any]]) -> float:
        """Simple sentiment analysis based on keywords"""
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'success', 'victory', 'win', 'peace', 'safe', 'help', 'support'
        }
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disaster', 'crisis',
            'attack', 'violence', 'war', 'conflict', 'bomb', 'kill', 'death',
            'terror', 'fear', 'danger', 'threat', 'emergency'
        }
        
        total_score = 0
        total_tweets = 0
        
        for tweet in tweets:
            text = tweet.get('text', '').lower()
            words = set(text.split())
            
            positive_count = len(words.intersection(positive_words))
            negative_count = len(words.intersection(negative_words))
            
            if positive_count + negative_count > 0:
                tweet_score = (positive_count - negative_count) / (positive_count + negative_count)
                total_score += tweet_score
                total_tweets += 1
        
        return total_score / total_tweets if total_tweets > 0 else 0.0


class ConflictMonitor:
    """Specialized monitor for conflict-related events"""
    
    def __init__(self, config: MonitoringConfig, crawler: Optional[TwitterCrawler] = None):
        self.config = config
        self.crawler = crawler or TwitterCrawler(TwitterConfig())
        self.search_engine = TwitterSearchEngine(self.crawler)
        self.alert_system = AlertSystem(config)
        self.trend_analyzer = TrendAnalyzer()
        
        self.status = MonitoringStatus.STOPPED
        self.monitoring_task: Optional[asyncio.Task] = None
        self.tweet_cache: deque = deque(maxlen=config.max_stored_tweets)
        self.last_check: Optional[datetime] = None
        
        # Event type patterns
        self.event_patterns = {
            EventType.ARMED_CONFLICT: [
                'armed conflict', 'military operation', 'fighting', 'battle',
                'combat', 'clashes', 'warfare', 'bombing', 'airstrike'
            ],
            EventType.TERRORISM: [
                'terrorist attack', 'terrorism', 'suicide bomber', 'explosion',
                'terror', 'bombing', 'attack', 'blast'
            ],
            EventType.PROTESTS: [
                'protest', 'demonstration', 'riot', 'unrest', 'uprising',
                'march', 'rally', 'civil disobedience'
            ],
            EventType.REFUGEE_CRISIS: [
                'refugees', 'displaced', 'evacuation', 'displacement',
                'asylum seekers', 'migration', 'humanitarian crisis'
            ]
        }
    
    async def start_monitoring(self):
        """Start the monitoring process"""
        if self.status == MonitoringStatus.RUNNING:
            logger.warning("Monitoring already running")
            return
        
        self.status = MonitoringStatus.STARTING
        
        try:
            # Initialize crawler
            await self.crawler.initialize()
            
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.status = MonitoringStatus.RUNNING
            
            logger.info("Conflict monitoring started")
            
        except Exception as e:
            self.status = MonitoringStatus.ERROR
            logger.error(f"Failed to start monitoring: {e}")
            raise
    
    async def stop_monitoring(self):
        """Stop the monitoring process"""
        self.status = MonitoringStatus.STOPPED
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Conflict monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.status == MonitoringStatus.RUNNING:
            try:
                await self._check_for_events()
                await asyncio.sleep(self.config.check_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _check_for_events(self):
        """Check for conflict events"""
        self.last_check = datetime.now()
        
        # Check each event type
        for event_type, patterns in self.event_patterns.items():
            await self._check_event_type(event_type, patterns)
        
        # Perform trend analysis
        if self.config.trend_analysis_interval_minutes > 0:
            await self._analyze_trends()
    
    async def _check_event_type(self, event_type: EventType, patterns: List[str]):
        """Check for specific event type"""
        try:
            # Build search query
            builder = QueryBuilder()
            for i, pattern in enumerate(patterns):
                operator = FilterOperator.OR if i > 0 else FilterOperator.AND
                builder.add_keyword(pattern, operator)
            
            # Add location filters
            for location in self.config.locations:
                builder.add_location(location)
            
            # Search for tweets
            search_query = SearchQuery(
                query=builder.build(),
                max_results=200,
                search_type=SearchType.RECENT
            )
            
            result = await self.search_engine.search(search_query)
            tweets = result.tweets
            
            # Store tweets in cache
            self.tweet_cache.extend(tweets)
            
            # Check if threshold is met
            if len(tweets) >= self.config.alert_threshold:
                # Group by location if possible
                tweets_by_location = self._group_tweets_by_location(tweets)
                
                for location, location_tweets in tweets_by_location.items():
                    if len(location_tweets) >= self.config.alert_threshold:
                        alert = self.alert_system.generate_alert(
                            event_type=event_type,
                            keywords_triggered=patterns,
                            tweet_count=len(location_tweets),
                            tweets=location_tweets,
                            location=location
                        )
                        
                        if alert:
                            logger.info(f"Generated {alert.level.value} alert: {alert.title}")
            
        except Exception as e:
            logger.error(f"Error checking event type {event_type}: {e}")
    
    def _group_tweets_by_location(self, tweets: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group tweets by detected location"""
        # Simple implementation - could be enhanced with NLP location extraction
        location_groups = defaultdict(list)
        
        for tweet in tweets:
            # Check user location
            user = tweet.get('user', {})
            user_location = user.get('location', '')
            
            # Check for location mentions in tweet text
            text = tweet.get('text', '').lower()
            
            # Match against configured locations
            matched_location = None
            for location in self.config.locations:
                if location.lower() in user_location.lower() or location.lower() in text:
                    matched_location = location
                    break
            
            location_groups[matched_location or 'unknown'].append(tweet)
        
        return dict(location_groups)
    
    async def _analyze_trends(self):
        """Analyze trends in collected data"""
        try:
            # Get recent tweets for analysis
            recent_time = datetime.now() - timedelta(hours=1)
            recent_tweets = [
                tweet for tweet in self.tweet_cache
                if self._parse_tweet_time(tweet.get('created_at', '')) > recent_time
            ]
            
            # Group tweets by keywords
            tweets_by_keyword = defaultdict(list)
            
            for tweet in recent_tweets:
                text = tweet.get('text', '').lower()
                for event_type, patterns in self.event_patterns.items():
                    for pattern in patterns:
                        if pattern.lower() in text:
                            tweets_by_keyword[pattern].append(tweet)
            
            # Analyze trends
            all_patterns = [pattern for patterns in self.event_patterns.values() for pattern in patterns]
            trends = self.trend_analyzer.analyze_trends(all_patterns, tweets_by_keyword)
            
            # Log significant trends
            for trend in trends[:5]:  # Top 5 trends
                if trend.trend_strength > 0.5:  # Significant threshold
                    logger.info(f"Trending: {trend.keyword} - Strength: {trend.trend_strength:.2f}, "
                               f"Velocity: {trend.velocity:.2f} tweets/hour")
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
    
    def _parse_tweet_time(self, time_str: str) -> datetime:
        """Parse tweet timestamp"""
        try:
            # Twitter time format: "Mon Oct 05 20:20:25 +0000 2020"
            return datetime.strptime(time_str, "%a %b %d %H:%M:%S %z %Y")
        except:
            return datetime.now()
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'status': self.status.value,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'cached_tweets': len(self.tweet_cache),
            'active_alerts': len(self.alert_system.get_active_alerts()),
            'total_alerts': len(self.alert_system.alerts),
            'crawler_status': self.crawler.get_status()
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get alerts from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_system.alerts
            if alert.timestamp > cutoff_time
        ]


# Utility functions
async def create_conflict_monitor(
    keywords: List[str],
    locations: Optional[List[str]] = None,
    alert_threshold: int = 10,
    check_interval_minutes: int = 5
) -> ConflictMonitor:
    """Create and configure a conflict monitor"""
    config = MonitoringConfig(
        keywords=keywords,
        locations=locations or [],
        alert_threshold=alert_threshold,
        check_interval_seconds=check_interval_minutes * 60
    )
    
    monitor = ConflictMonitor(config)
    await monitor.start_monitoring()
    return monitor


class TwitterMonitor:
    """General-purpose Twitter monitoring system."""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.is_monitoring = False
        self.alerts = []
        self.last_check = None
        
    async def start_monitoring(self):
        """Start monitoring Twitter for specified criteria."""
        if not CORE_AVAILABLE:
            raise ImportError("Core Twitter modules not available")
        
        self.is_monitoring = True
        self.last_check = datetime.now()
        logger.info("Twitter monitoring started")
        
        # Start monitoring loop
        while self.is_monitoring:
            try:
                await self._check_and_alert()
                await asyncio.sleep(self.config.check_interval_seconds)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def stop_monitoring(self):
        """Stop the monitoring process."""
        self.is_monitoring = False
        logger.info("Twitter monitoring stopped")
    
    async def _check_and_alert(self):
        """Check for new tweets matching criteria and generate alerts."""
        # Placeholder implementation
        logger.debug("Checking for new tweets...")
        # In a real implementation, this would search for tweets and analyze them
    
    def get_recent_alerts(self, limit: int = 10) -> List[Alert]:
        """Get recent alerts."""
        return self.alerts[-limit:]
    
    def is_active(self) -> bool:
        """Check if monitoring is currently active."""
        return self.is_monitoring


class TrendMonitor:
    """Monitor Twitter trends and trending topics."""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.trends = []
        
    async def get_current_trends(self, location_id: int = 1) -> List[str]:
        """Get current trending topics."""
        # Placeholder implementation
        logger.info(f"Fetching trends for location {location_id}")
        return ["#ExampleTrend", "#PlaceholderTopic"]
    
    async def analyze_trend_changes(self) -> Dict[str, Any]:
        """Analyze changes in trending topics."""
        return {
            "new_trends": [],
            "declining_trends": [],
            "stable_trends": []
        }


__all__ = [
    'ConflictMonitor', 'TwitterMonitor', 'TrendMonitor', 'AlertSystem',
    'MonitoringConfig', 'Alert', 'TrendData', 'TrendAnalyzer',
    'AlertLevel', 'MonitoringStatus', 'EventType',
    'create_conflict_monitor'
]