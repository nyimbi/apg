"""
Twitter Data Processing Module
==============================

Data models, processing, and storage functionality for Twitter crawler.
Provides structured data handling, export capabilities, and database integration.

Author: Lindela Development Team
Version: 2.0.0
License: MIT
"""

import json
import csv
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union, Iterator
from enum import Enum
from datetime import datetime
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import sqlalchemy
    from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
    Base = declarative_base()
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Base = None

try:
    import pymongo
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False


class ExportFormat(Enum):
    """Supported export formats"""
    JSON = "json"
    CSV = "csv"
    JSONL = "jsonl"  # JSON Lines
    PARQUET = "parquet"
    EXCEL = "excel"
    XML = "xml"


class StorageBackend(Enum):
    """Supported storage backends"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    FILE = "file"


@dataclass
class TweetModel:
    """Structured tweet data model"""
    # Core tweet data
    id: str
    text: str
    created_at: datetime
    language: Optional[str] = None
    
    # User data
    user_id: Optional[str] = None
    username: Optional[str] = None
    user_display_name: Optional[str] = None
    user_followers_count: int = 0
    user_verified: bool = False
    user_location: Optional[str] = None
    
    # Engagement metrics
    retweet_count: int = 0
    favorite_count: int = 0
    reply_count: int = 0
    quote_count: int = 0
    
    # Content analysis
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    media_urls: List[str] = field(default_factory=list)
    
    # Thread information
    in_reply_to_status_id: Optional[str] = None
    in_reply_to_user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Geolocation
    coordinates: Optional[Dict[str, float]] = None
    place_name: Optional[str] = None
    place_country: Optional[str] = None
    
    # Tweet type
    is_retweet: bool = False
    is_quote_tweet: bool = False
    is_reply: bool = False
    retweeted_status_id: Optional[str] = None
    quoted_status_id: Optional[str] = None
    
    # Analysis results
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    conflict_relevance_score: Optional[float] = None
    event_type: Optional[str] = None
    location_extracted: Optional[str] = None
    
    # Metadata
    source: str = "twitter"
    processed_at: Optional[datetime] = None
    crawl_session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.processed_at is None:
            self.processed_at = datetime.now()
        
        # Extract reply information
        if self.in_reply_to_status_id:
            self.is_reply = True
        
        # Ensure coordinates are properly formatted
        if self.coordinates and isinstance(self.coordinates, dict):
            if 'lat' not in self.coordinates or 'lon' not in self.coordinates:
                self.coordinates = None
    
    @classmethod
    def from_raw_tweet(cls, raw_tweet: Dict[str, Any], session_id: Optional[str] = None) -> 'TweetModel':
        """Create TweetModel from raw Twitter API response"""
        user = raw_tweet.get('user', {})
        
        # Parse coordinates
        coordinates = None
        if raw_tweet.get('coordinates'):
            coords = raw_tweet['coordinates']['coordinates']
            coordinates = {'lon': coords[0], 'lat': coords[1]}
        elif raw_tweet.get('geo'):
            coords = raw_tweet['geo']['coordinates']
            coordinates = {'lat': coords[0], 'lon': coords[1]}
        
        # Parse place information
        place = raw_tweet.get('place', {})
        place_name = place.get('full_name') or place.get('name')
        place_country = place.get('country_code')
        
        # Parse created_at
        created_at = raw_tweet.get('created_at')
        if isinstance(created_at, str):
            try:
                created_at = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
            except ValueError:
                created_at = datetime.now()
        elif not isinstance(created_at, datetime):
            created_at = datetime.now()
        
        return cls(
            id=str(raw_tweet.get('id', '')),
            text=raw_tweet.get('full_text') or raw_tweet.get('text', ''),
            created_at=created_at,
            language=raw_tweet.get('lang'),
            
            # User data
            user_id=str(user.get('id', '')),
            username=user.get('screen_name'),
            user_display_name=user.get('name'),
            user_followers_count=user.get('followers_count', 0),
            user_verified=user.get('verified', False),
            user_location=user.get('location'),
            
            # Engagement
            retweet_count=raw_tweet.get('retweet_count', 0),
            favorite_count=raw_tweet.get('favorite_count', 0),
            reply_count=raw_tweet.get('reply_count', 0),
            quote_count=raw_tweet.get('quote_count', 0),
            
            # Content
            hashtags=raw_tweet.get('hashtags', []),
            mentions=raw_tweet.get('mentions', []),
            urls=raw_tweet.get('urls', []),
            
            # Thread
            in_reply_to_status_id=raw_tweet.get('in_reply_to_status_id_str'),
            in_reply_to_user_id=raw_tweet.get('in_reply_to_user_id_str'),
            conversation_id=raw_tweet.get('conversation_id_str'),
            
            # Geo
            coordinates=coordinates,
            place_name=place_name,
            place_country=place_country,
            
            # Type
            is_retweet=raw_tweet.get('retweeted', False),
            is_quote_tweet=raw_tweet.get('is_quote_status', False),
            retweeted_status_id=raw_tweet.get('retweeted_status', {}).get('id_str'),
            quoted_status_id=raw_tweet.get('quoted_status_id_str'),
            
            # Session
            crawl_session_id=session_id
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        
        # Convert datetime objects to ISO strings
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.processed_at:
            data['processed_at'] = self.processed_at.isoformat()
        
        return data
    
    @property
    def engagement_score(self) -> float:
        """Calculate total engagement score"""
        return (
            self.retweet_count * 3 +
            self.favorite_count * 2 +
            self.reply_count * 1 +
            self.quote_count * 2
        )
    
    @property
    def has_media(self) -> bool:
        """Check if tweet has media attachments"""
        return len(self.media_urls) > 0
    
    @property
    def has_location(self) -> bool:
        """Check if tweet has location information"""
        return (
            self.coordinates is not None or
            self.place_name is not None or
            self.user_location is not None
        )


@dataclass
class UserModel:
    """Structured user data model"""
    id: str
    username: str
    display_name: str
    description: Optional[str] = None
    
    # Metrics
    followers_count: int = 0
    following_count: int = 0
    tweet_count: int = 0
    listed_count: int = 0
    
    # Profile
    verified: bool = False
    protected: bool = False
    location: Optional[str] = None
    url: Optional[str] = None
    profile_image_url: Optional[str] = None
    banner_url: Optional[str] = None
    
    # Dates
    created_at: Optional[datetime] = None
    last_tweet_at: Optional[datetime] = None
    
    # Analysis
    bot_score: Optional[float] = None
    influence_score: Optional[float] = None
    activity_score: Optional[float] = None
    
    # Metadata
    first_crawled_at: Optional[datetime] = None
    last_updated_at: Optional[datetime] = None
    crawl_count: int = 0
    
    @classmethod
    def from_raw_user(cls, raw_user: Dict[str, Any]) -> 'UserModel':
        """Create UserModel from raw Twitter API response"""
        created_at = raw_user.get('created_at')
        if isinstance(created_at, str):
            try:
                created_at = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
            except ValueError:
                created_at = None
        
        return cls(
            id=str(raw_user.get('id', '')),
            username=raw_user.get('screen_name', ''),
            display_name=raw_user.get('name', ''),
            description=raw_user.get('description'),
            
            followers_count=raw_user.get('followers_count', 0),
            following_count=raw_user.get('friends_count', 0),
            tweet_count=raw_user.get('statuses_count', 0),
            listed_count=raw_user.get('listed_count', 0),
            
            verified=raw_user.get('verified', False),
            protected=raw_user.get('protected', False),
            location=raw_user.get('location'),
            url=raw_user.get('url'),
            profile_image_url=raw_user.get('profile_image_url_https'),
            banner_url=raw_user.get('profile_banner_url'),
            
            created_at=created_at,
            first_crawled_at=datetime.now()
        )


# SQLAlchemy models for database storage
if SQLALCHEMY_AVAILABLE:
    class SQLTweet(Base):
        __tablename__ = 'tweets'
        
        id = Column(String, primary_key=True)
        text = Column(Text)
        created_at = Column(DateTime)
        language = Column(String(10))
        
        user_id = Column(String)
        username = Column(String)
        user_display_name = Column(String)
        user_followers_count = Column(Integer, default=0)
        user_verified = Column(Boolean, default=False)
        
        retweet_count = Column(Integer, default=0)
        favorite_count = Column(Integer, default=0)
        reply_count = Column(Integer, default=0)
        
        hashtags_json = Column(Text)  # JSON string
        urls_json = Column(Text)  # JSON string
        
        coordinates_lat = Column(Float)
        coordinates_lon = Column(Float)
        place_name = Column(String)
        place_country = Column(String(10))
        
        is_retweet = Column(Boolean, default=False)
        is_reply = Column(Boolean, default=False)
        
        sentiment_score = Column(Float)
        conflict_relevance_score = Column(Float)
        
        processed_at = Column(DateTime, default=datetime.now)
        crawl_session_id = Column(String)
    
    class SQLUser(Base):
        __tablename__ = 'users'
        
        id = Column(String, primary_key=True)
        username = Column(String, unique=True)
        display_name = Column(String)
        description = Column(Text)
        
        followers_count = Column(Integer, default=0)
        following_count = Column(Integer, default=0)
        tweet_count = Column(Integer, default=0)
        
        verified = Column(Boolean, default=False)
        location = Column(String)
        
        created_at = Column(DateTime)
        first_crawled_at = Column(DateTime, default=datetime.now)
        last_updated_at = Column(DateTime, default=datetime.now)


class DataExporter:
    """Export Twitter data to various formats"""
    
    def __init__(self, output_dir: str = "exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_tweets(
        self,
        tweets: List[TweetModel],
        format: ExportFormat,
        filename: Optional[str] = None
    ) -> str:
        """Export tweets to specified format"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tweets_{timestamp}.{format.value}"
        
        filepath = self.output_dir / filename
        
        if format == ExportFormat.JSON:
            return self._export_json(tweets, filepath)
        elif format == ExportFormat.CSV:
            return self._export_csv(tweets, filepath)
        elif format == ExportFormat.JSONL:
            return self._export_jsonl(tweets, filepath)
        elif format == ExportFormat.EXCEL and PANDAS_AVAILABLE:
            return self._export_excel(tweets, filepath)
        elif format == ExportFormat.PARQUET and PANDAS_AVAILABLE:
            return self._export_parquet(tweets, filepath)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, tweets: List[TweetModel], filepath: Path) -> str:
        """Export to JSON format"""
        data = [tweet.to_dict() for tweet in tweets]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        return str(filepath)
    
    def _export_csv(self, tweets: List[TweetModel], filepath: Path) -> str:
        """Export to CSV format"""
        if not tweets:
            return str(filepath)
        
        fieldnames = list(tweets[0].to_dict().keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for tweet in tweets:
                row = tweet.to_dict()
                # Convert lists to comma-separated strings for CSV
                for key, value in row.items():
                    if isinstance(value, list):
                        row[key] = ', '.join(str(v) for v in value)
                    elif isinstance(value, dict):
                        row[key] = json.dumps(value)
                
                writer.writerow(row)
        
        return str(filepath)
    
    def _export_jsonl(self, tweets: List[TweetModel], filepath: Path) -> str:
        """Export to JSON Lines format"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for tweet in tweets:
                json.dump(tweet.to_dict(), f, ensure_ascii=False, default=str)
                f.write('\n')
        
        return str(filepath)
    
    def _export_excel(self, tweets: List[TweetModel], filepath: Path) -> str:
        """Export to Excel format"""
        data = [tweet.to_dict() for tweet in tweets]
        df = pd.DataFrame(data)
        
        # Convert list columns to strings
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                )
        
        df.to_excel(filepath, index=False)
        return str(filepath)
    
    def _export_parquet(self, tweets: List[TweetModel], filepath: Path) -> str:
        """Export to Parquet format"""
        data = [tweet.to_dict() for tweet in tweets]
        df = pd.DataFrame(data)
        
        # Handle complex types for Parquet
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x)
                )
        
        df.to_parquet(filepath, index=False)
        return str(filepath)


class DataStorage(ABC):
    """Abstract base class for data storage"""
    
    @abstractmethod
    async def store_tweet(self, tweet: TweetModel) -> bool:
        """Store a tweet"""
        pass
    
    @abstractmethod
    async def store_tweets(self, tweets: List[TweetModel]) -> int:
        """Store multiple tweets, return count stored"""
        pass
    
    @abstractmethod
    async def get_tweet(self, tweet_id: str) -> Optional[TweetModel]:
        """Retrieve a tweet by ID"""
        pass
    
    @abstractmethod
    async def search_tweets(
        self,
        query: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[TweetModel]:
        """Search tweets with filters"""
        pass
    
    @abstractmethod
    async def delete_tweet(self, tweet_id: str) -> bool:
        """Delete a tweet"""
        pass


class SQLiteStorage(DataStorage):
    """SQLite storage implementation"""
    
    def __init__(self, db_path: str = "twitter_data.db"):
        self.db_path = db_path
        self.engine = None
        self.SessionLocal = None
        
        if SQLALCHEMY_AVAILABLE:
            self._initialize_db()
    
    def _initialize_db(self):
        """Initialize SQLite database"""
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    async def store_tweet(self, tweet: TweetModel) -> bool:
        """Store a single tweet"""
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy required for database storage")
        
        try:
            session = self.SessionLocal()
            
            # Convert TweetModel to SQLTweet
            sql_tweet = SQLTweet(
                id=tweet.id,
                text=tweet.text,
                created_at=tweet.created_at,
                language=tweet.language,
                user_id=tweet.user_id,
                username=tweet.username,
                user_display_name=tweet.user_display_name,
                user_followers_count=tweet.user_followers_count,
                user_verified=tweet.user_verified,
                retweet_count=tweet.retweet_count,
                favorite_count=tweet.favorite_count,
                reply_count=tweet.reply_count,
                hashtags_json=json.dumps(tweet.hashtags),
                urls_json=json.dumps(tweet.urls),
                coordinates_lat=tweet.coordinates.get('lat') if tweet.coordinates else None,
                coordinates_lon=tweet.coordinates.get('lon') if tweet.coordinates else None,
                place_name=tweet.place_name,
                place_country=tweet.place_country,
                is_retweet=tweet.is_retweet,
                is_reply=tweet.is_reply,
                sentiment_score=tweet.sentiment_score,
                conflict_relevance_score=tweet.conflict_relevance_score,
                processed_at=tweet.processed_at,
                crawl_session_id=tweet.crawl_session_id
            )
            
            session.merge(sql_tweet)  # Use merge to handle duplicates
            session.commit()
            session.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing tweet {tweet.id}: {e}")
            return False
    
    async def store_tweets(self, tweets: List[TweetModel]) -> int:
        """Store multiple tweets"""
        stored_count = 0
        
        for tweet in tweets:
            if await self.store_tweet(tweet):
                stored_count += 1
        
        return stored_count
    
    async def get_tweet(self, tweet_id: str) -> Optional[TweetModel]:
        """Retrieve a tweet by ID"""
        if not SQLALCHEMY_AVAILABLE:
            return None
        
        try:
            session = self.SessionLocal()
            sql_tweet = session.query(SQLTweet).filter(SQLTweet.id == tweet_id).first()
            session.close()
            
            if sql_tweet:
                return self._sql_tweet_to_model(sql_tweet)
            
        except Exception as e:
            logger.error(f"Error retrieving tweet {tweet_id}: {e}")
        
        return None
    
    async def search_tweets(
        self,
        query: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[TweetModel]:
        """Search tweets with filters"""
        if not SQLALCHEMY_AVAILABLE:
            return []
        
        try:
            session = self.SessionLocal()
            sql_query = session.query(SQLTweet)
            
            if query:
                sql_query = sql_query.filter(SQLTweet.text.contains(query))
            
            if start_date:
                sql_query = sql_query.filter(SQLTweet.created_at >= start_date)
            
            if end_date:
                sql_query = sql_query.filter(SQLTweet.created_at <= end_date)
            
            sql_tweets = sql_query.order_by(SQLTweet.created_at.desc()).limit(limit).all()
            session.close()
            
            return [self._sql_tweet_to_model(sql_tweet) for sql_tweet in sql_tweets]
            
        except Exception as e:
            logger.error(f"Error searching tweets: {e}")
            return []
    
    async def delete_tweet(self, tweet_id: str) -> bool:
        """Delete a tweet"""
        if not SQLALCHEMY_AVAILABLE:
            return False
        
        try:
            session = self.SessionLocal()
            result = session.query(SQLTweet).filter(SQLTweet.id == tweet_id).delete()
            session.commit()
            session.close()
            
            return result > 0
            
        except Exception as e:
            logger.error(f"Error deleting tweet {tweet_id}: {e}")
            return False
    
    def _sql_tweet_to_model(self, sql_tweet: 'SQLTweet') -> TweetModel:
        """Convert SQLTweet to TweetModel"""
        coordinates = None
        if sql_tweet.coordinates_lat and sql_tweet.coordinates_lon:
            coordinates = {'lat': sql_tweet.coordinates_lat, 'lon': sql_tweet.coordinates_lon}
        
        hashtags = []
        if sql_tweet.hashtags_json:
            try:
                hashtags = json.loads(sql_tweet.hashtags_json)
            except:
                pass
        
        urls = []
        if sql_tweet.urls_json:
            try:
                urls = json.loads(sql_tweet.urls_json)
            except:
                pass
        
        return TweetModel(
            id=sql_tweet.id,
            text=sql_tweet.text,
            created_at=sql_tweet.created_at,
            language=sql_tweet.language,
            user_id=sql_tweet.user_id,
            username=sql_tweet.username,
            user_display_name=sql_tweet.user_display_name,
            user_followers_count=sql_tweet.user_followers_count,
            user_verified=sql_tweet.user_verified,
            retweet_count=sql_tweet.retweet_count,
            favorite_count=sql_tweet.favorite_count,
            reply_count=sql_tweet.reply_count,
            hashtags=hashtags,
            urls=urls,
            coordinates=coordinates,
            place_name=sql_tweet.place_name,
            place_country=sql_tweet.place_country,
            is_retweet=sql_tweet.is_retweet,
            is_reply=sql_tweet.is_reply,
            sentiment_score=sql_tweet.sentiment_score,
            conflict_relevance_score=sql_tweet.conflict_relevance_score,
            processed_at=sql_tweet.processed_at,
            crawl_session_id=sql_tweet.crawl_session_id
        )


class TwitterDataProcessor:
    """Main data processing coordinator"""
    
    def __init__(self, storage: Optional[DataStorage] = None, export_dir: str = "exports"):
        self.storage = storage or SQLiteStorage()
        self.exporter = DataExporter(export_dir)
        self.processing_stats = {
            'total_processed': 0,
            'total_stored': 0,
            'total_errors': 0,
            'last_processed': None
        }
    
    async def process_raw_tweets(
        self,
        raw_tweets: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        store: bool = True
    ) -> List[TweetModel]:
        """Process raw tweets into structured models"""
        processed_tweets = []
        
        for raw_tweet in raw_tweets:
            try:
                tweet_model = TweetModel.from_raw_tweet(raw_tweet, session_id)
                processed_tweets.append(tweet_model)
                self.processing_stats['total_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing tweet: {e}")
                self.processing_stats['total_errors'] += 1
        
        # Store tweets if requested
        if store and processed_tweets:
            stored_count = await self.storage.store_tweets(processed_tweets)
            self.processing_stats['total_stored'] += stored_count
        
        self.processing_stats['last_processed'] = datetime.now()
        
        return processed_tweets
    
    async def export_recent_tweets(
        self,
        hours: int = 24,
        format: ExportFormat = ExportFormat.JSON,
        filename: Optional[str] = None
    ) -> str:
        """Export recent tweets"""
        start_date = datetime.now() - timedelta(hours=hours)
        tweets = await self.storage.search_tweets(start_date=start_date, limit=10000)
        
        return self.exporter.export_tweets(tweets, format, filename)
    
    async def cleanup_old_data(self, days: int = 30) -> int:
        """Clean up old data (placeholder - implement based on storage backend)"""
        # This would be implemented based on the specific storage backend
        logger.info(f"Cleanup requested for data older than {days} days")
        return 0
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.processing_stats.copy()


# Utility functions
def create_data_processor(
    db_path: str = "twitter_data.db",
    export_dir: str = "exports"
) -> TwitterDataProcessor:
    """Create a configured data processor"""
    storage = SQLiteStorage(db_path)
    return TwitterDataProcessor(storage, export_dir)


async def quick_export_tweets(
    tweets: List[Dict[str, Any]],
    format: ExportFormat = ExportFormat.JSON,
    filename: Optional[str] = None
) -> str:
    """Quick export function for tweet data"""
    # Convert raw tweets to models
    tweet_models = []
    for raw_tweet in tweets:
        try:
            tweet_models.append(TweetModel.from_raw_tweet(raw_tweet))
        except Exception as e:
            logger.error(f"Error converting tweet: {e}")
    
    # Export
    exporter = DataExporter()
    return exporter.export_tweets(tweet_models, format, filename)


__all__ = [
    'TweetModel', 'UserModel', 'TwitterDataProcessor', 'DataExporter', 'DataStorage',
    'SQLiteStorage', 'ExportFormat', 'StorageBackend',
    'create_data_processor', 'quick_export_tweets'
]