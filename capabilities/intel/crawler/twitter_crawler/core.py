"""
Core Twitter Crawler Module
============================

Core Twitter crawling functionality using the twikit library.
Provides authentication, session management, and basic crawling operations.

Author: Lindela Development Team
Version: 2.0.0
License: MIT
"""

import asyncio
import logging
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
from datetime import datetime, timedelta
import pickle
import os

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import twikit
    from twikit import Client
    from twikit.errors import TwitterException, Unauthorized, TooManyRequests
    TWIKIT_AVAILABLE = True
    TwikitException = TwitterException  # Alias for backwards compatibility
except ImportError:
    TWIKIT_AVAILABLE = False
    logger.debug("twikit not available. Install with: pip install twikit")
    Client = None
    TwitterException = Exception
    TwikitException = Exception
    Unauthorized = Exception
    TooManyRequests = Exception

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class CrawlerError(Exception):
    """Base exception for Twitter crawler errors"""
    pass


class AuthenticationError(CrawlerError):
    """Authentication-related errors"""
    pass


class RateLimitError(CrawlerError):
    """Rate limiting errors"""
    pass


class SessionError(CrawlerError):
    """Session management errors"""
    pass


class CrawlerStatus(Enum):
    """Crawler operational status"""
    IDLE = "idle"
    AUTHENTICATING = "authenticating"
    ACTIVE = "active"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class TwitterConfig:
    """Configuration for Twitter crawler"""
    # Authentication
    username: Optional[str] = None
    password: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    
    # Session management
    session_file: str = "twitter_session.pkl"
    auto_save_session: bool = True
    session_timeout: int = 3600  # 1 hour
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 30
    rate_limit_requests_per_hour: int = 1000
    wait_on_rate_limit: bool = True
    rate_limit_buffer: float = 0.1  # 10% buffer
    
    # Retry configuration
    max_retries: int = 3
    backoff_factor: float = 2.0
    retry_on_status: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    
    # Timeouts
    connect_timeout: int = 30
    read_timeout: int = 60
    
    # User agent and headers
    user_agent: Optional[str] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)
    
    # Proxy settings
    proxy: Optional[str] = None
    proxy_auth: Optional[tuple] = None
    
    # Logging
    log_level: str = "INFO"
    log_requests: bool = False
    
    # Advanced settings
    verify_ssl: bool = True
    enable_cookies: bool = True
    max_concurrent_requests: int = 5


@dataclass
class RateLimitInfo:
    """Rate limit tracking information"""
    requests_made: int = 0
    requests_per_minute: int = 0
    requests_per_hour: int = 0
    last_request_time: Optional[datetime] = None
    reset_time_minute: Optional[datetime] = None
    reset_time_hour: Optional[datetime] = None
    is_limited: bool = False


class TwitterSession:
    """Manages Twitter session state and persistence"""
    
    def __init__(self, config: TwitterConfig):
        self.config = config
        self.client: Optional[Client] = None
        self.session_data: Dict[str, Any] = {}
        self.is_authenticated = False
        self.session_file_path = config.session_file
        self.last_activity = datetime.now()
        
    def save_session(self) -> bool:
        """Save session to file"""
        try:
            if self.client and self.config.auto_save_session:
                session_data = {
                    'cookies': self.client.cookies,
                    'headers': getattr(self.client, 'headers', {}),
                    'last_activity': self.last_activity,
                    'is_authenticated': self.is_authenticated
                }
                
                with open(self.session_file_path, 'wb') as f:
                    pickle.dump(session_data, f)
                
                logger.debug(f"Session saved to {self.session_file_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
        
        return False
    
    def load_session(self) -> bool:
        """Load session from file"""
        try:
            if os.path.exists(self.session_file_path):
                with open(self.session_file_path, 'rb') as f:
                    session_data = pickle.load(f)
                
                # Check if session is not expired
                last_activity = session_data.get('last_activity')
                if last_activity and (datetime.now() - last_activity).seconds < self.config.session_timeout:
                    self.session_data = session_data
                    self.is_authenticated = session_data.get('is_authenticated', False)
                    self.last_activity = last_activity
                    
                    logger.debug(f"Session loaded from {self.session_file_path}")
                    return True
                else:
                    logger.debug("Session expired, will need to re-authenticate")
                    self.clear_session()
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
        
        return False
    
    def clear_session(self):
        """Clear session data"""
        self.session_data = {}
        self.is_authenticated = False
        
        if os.path.exists(self.session_file_path):
            try:
                os.remove(self.session_file_path)
                logger.debug("Session file removed")
            except Exception as e:
                logger.error(f"Failed to remove session file: {e}")
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
        if self.config.auto_save_session:
            self.save_session()


class RateLimiter:
    """Handles rate limiting for Twitter API requests"""
    
    def __init__(self, config: TwitterConfig):
        self.config = config
        self.rate_info = RateLimitInfo()
        self.request_times: List[datetime] = []
        
    def can_make_request(self) -> bool:
        """Check if a request can be made without hitting rate limits"""
        now = datetime.now()
        
        # Clean old request times
        self._cleanup_old_requests(now)
        
        # Check minute limit
        minute_requests = len([t for t in self.request_times if (now - t).seconds < 60])
        if minute_requests >= self.config.rate_limit_requests_per_minute:
            return False
        
        # Check hour limit
        hour_requests = len([t for t in self.request_times if (now - t).seconds < 3600])
        if hour_requests >= self.config.rate_limit_requests_per_hour:
            return False
        
        return True
    
    def wait_if_needed(self) -> float:
        """Wait if rate limited, return wait time"""
        if not self.can_make_request():
            wait_time = self._calculate_wait_time()
            if self.config.wait_on_rate_limit and wait_time > 0:
                logger.info(f"Rate limited, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                return wait_time
            else:
                raise RateLimitError("Rate limit exceeded and wait_on_rate_limit is False")
        return 0.0
    
    def record_request(self):
        """Record a request for rate limiting"""
        now = datetime.now()
        self.request_times.append(now)
        self.rate_info.requests_made += 1
        self.rate_info.last_request_time = now
        
        # Clean old requests
        self._cleanup_old_requests(now)
    
    def _cleanup_old_requests(self, now: datetime):
        """Remove request times older than 1 hour"""
        cutoff = now - timedelta(hours=1)
        self.request_times = [t for t in self.request_times if t > cutoff]
    
    def _calculate_wait_time(self) -> float:
        """Calculate how long to wait before next request"""
        now = datetime.now()
        
        # Check minute limit
        minute_requests = [t for t in self.request_times if (now - t).seconds < 60]
        if len(minute_requests) >= self.config.rate_limit_requests_per_minute:
            oldest_in_minute = min(minute_requests)
            wait_time = 60 - (now - oldest_in_minute).seconds + 1
            return wait_time
        
        # Check hour limit
        hour_requests = [t for t in self.request_times if (now - t).seconds < 3600]
        if len(hour_requests) >= self.config.rate_limit_requests_per_hour:
            oldest_in_hour = min(hour_requests)
            wait_time = 3600 - (now - oldest_in_hour).seconds + 1
            return wait_time
        
        return 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        now = datetime.now()
        self._cleanup_old_requests(now)
        
        minute_requests = len([t for t in self.request_times if (now - t).seconds < 60])
        hour_requests = len([t for t in self.request_times if (now - t).seconds < 3600])
        
        return {
            "requests_last_minute": minute_requests,
            "requests_last_hour": hour_requests,
            "minute_limit": self.config.rate_limit_requests_per_minute,
            "hour_limit": self.config.rate_limit_requests_per_hour,
            "can_make_request": self.can_make_request(),
            "estimated_wait_time": self._calculate_wait_time()
        }


class TwitterCrawler:
    """Main Twitter crawler class using twikit"""
    
    def __init__(self, config: TwitterConfig):
        if not TWIKIT_AVAILABLE:
            raise ImportError("twikit is required for TwitterCrawler. Install with: pip install twikit")
        
        self.config = config
        self.client: Optional[Client] = None
        self.session = TwitterSession(config)
        self.rate_limiter = RateLimiter(config)
        self.status = CrawlerStatus.IDLE
        self.error_count = 0
        self.last_error: Optional[Exception] = None
        self.callbacks: Dict[str, List[Callable]] = {}
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, config.log_level.upper()))
        
    async def initialize(self) -> bool:
        """Initialize the Twitter crawler"""
        try:
            self.client = Client()
            
            # If credentials are provided, try to authenticate
            if self.config.username and self.config.password:
                self.status = CrawlerStatus.AUTHENTICATING
                # Try to load existing session
                if self.session.load_session():
                    if await self._validate_session():
                        self.status = CrawlerStatus.ACTIVE
                        self.logger.info("Existing session validated successfully")
                        return True
                
                # Authenticate with credentials
                return await self._authenticate()
            else:
                # No credentials provided - use unauthenticated session
                self.logger.info("No credentials provided, using unauthenticated session for search operations")
                self.status = CrawlerStatus.ACTIVE
                self.session.is_authenticated = False
                return True
                
        except Exception as e:
            self.status = CrawlerStatus.ERROR
            self.last_error = e
            self.logger.error(f"Failed to initialize crawler: {e}")
            return False
    
    async def _authenticate(self) -> bool:
        """Authenticate with Twitter using twikit"""
        try:
            self.logger.info("Authenticating with Twitter...")
            
            # Use twikit's login method
            await self.client.login(
                auth_info_1=self.config.username or self.config.email,
                auth_info_2=self.config.email or self.config.phone,
                password=self.config.password
            )
            
            self.session.is_authenticated = True
            self.session.update_activity()
            self.status = CrawlerStatus.ACTIVE
            
            self.logger.info("Authentication successful")
            self._trigger_callback('authentication_success')
            return True
            
        except Unauthorized as e:
            self.status = CrawlerStatus.ERROR
            self.last_error = e
            self.logger.error(f"Authentication failed: {e}")
            self._trigger_callback('authentication_failed', error=e)
            raise AuthenticationError(f"Authentication failed: {e}")
        
        except Exception as e:
            self.status = CrawlerStatus.ERROR
            self.last_error = e
            self.logger.error(f"Unexpected error during authentication: {e}")
            raise AuthenticationError(f"Authentication error: {e}")
    
    async def _validate_session(self) -> bool:
        """Validate existing session"""
        try:
            if not self.client:
                return False
            
            # Only validate session if we have credentials
            if not (self.config.username and self.config.password):
                return False
            
            # Try a simple API call to validate session
            # This is a placeholder - implement based on twikit's session validation
            await self.client.get_user_by_screen_name('twitter')
            return True
            
        except Exception as e:
            self.logger.debug(f"Session validation failed: {e}")
            return False
    
    async def make_request(self, request_func: Callable, *args, **kwargs) -> Any:
        """Make a request with rate limiting and error handling"""
        attempt = 0
        
        while attempt < self.config.max_retries:
            try:
                # Check rate limits
                wait_time = self.rate_limiter.wait_if_needed()
                if wait_time > 0:
                    self.status = CrawlerStatus.RATE_LIMITED
                    await asyncio.sleep(wait_time)
                    self.status = CrawlerStatus.ACTIVE
                
                # Make the request
                self.rate_limiter.record_request()
                self.session.update_activity()
                
                result = await request_func(*args, **kwargs)
                
                # Reset error count on success
                self.error_count = 0
                return result
                
            except TooManyRequests as e:
                self.logger.warning(f"Rate limited by Twitter: {e}")
                self.status = CrawlerStatus.RATE_LIMITED
                
                # Wait for rate limit reset
                reset_time = getattr(e, 'reset_time', 900)  # Default 15 minutes
                await asyncio.sleep(reset_time)
                self.status = CrawlerStatus.ACTIVE
                
            except TwikitException as e:
                attempt += 1
                self.error_count += 1
                self.last_error = e
                
                # Check if this is a 403 error and we're not authenticated
                if "403" in str(e) and not self.session.is_authenticated:
                    self.logger.info("Search operation requires authentication or is rate limited")
                    raise CrawlerError(f"Search failed: {e}")
                
                if attempt >= self.config.max_retries:
                    self.logger.error(f"Request failed after {self.config.max_retries} attempts: status: {getattr(e, 'status_code', 'unknown')}, message: {getattr(e, 'message', str(e))}")
                    raise CrawlerError(f"Request failed: status: {getattr(e, 'status_code', 'unknown')}, message: {getattr(e, 'message', str(e))}")
                
                # Exponential backoff
                wait_time = (self.config.backoff_factor ** attempt)
                self.logger.warning(f"Request failed (attempt {attempt}), retrying in {wait_time}s: status: {getattr(e, 'status_code', 'unknown')}, message: {getattr(e, 'message', str(e))}")
                await asyncio.sleep(wait_time)
            
            except Exception as e:
                self.error_count += 1
                self.last_error = e
                self.logger.error(f"Unexpected error in request: {e}")
                raise CrawlerError(f"Unexpected error: {e}")
    
    def add_callback(self, event: str, callback: Callable):
        """Add event callback"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    def remove_callback(self, event: str, callback: Callable):
        """Remove event callback"""
        if event in self.callbacks and callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
    
    def _trigger_callback(self, event: str, **kwargs):
        """Trigger event callbacks"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(**kwargs)
                except Exception as e:
                    self.logger.error(f"Error in callback for event {event}: {e}")
    
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information by username"""
        async def _get_user():
            user = await self.client.get_user_by_screen_name(username)
            return self._user_to_dict(user) if user else None
        
        return await self.make_request(_get_user)
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information by user ID"""
        async def _get_user():
            user = await self.client.get_user_by_id(user_id)
            return self._user_to_dict(user) if user else None
        
        return await self.make_request(_get_user)
    
    async def get_user_tweets(
        self,
        username: str,
        count: int = 20,
        include_replies: bool = False,
        include_retweets: bool = True
    ) -> List[Dict[str, Any]]:
        """Get tweets from a user's timeline"""
        async def _get_tweets():
            user = await self.client.get_user_by_screen_name(username)
            if not user:
                return []
            
            tweets = await user.get_tweets(
                tweet_type='Tweets',
                count=count
            )
            
            return [self._tweet_to_dict(tweet) for tweet in tweets]
        
        return await self.make_request(_get_tweets)
    
    async def search_tweets(
        self,
        query: str,
        count: int = 20,
        result_type: str = 'recent'
    ) -> List[Dict[str, Any]]:
        """Search for tweets"""
        async def _search():
            # Map result_type to twikit product parameter
            product_map = {
                'recent': 'Latest',
                'latest': 'Latest',
                'popular': 'Top',
                'top': 'Top',
                'media': 'Media'
            }
            product = product_map.get(result_type.lower(), 'Latest')
            
            tweets = await self.client.search_tweet(query, product=product, count=count)
            return [self._tweet_to_dict(tweet) for tweet in tweets]
        
        return await self.make_request(_search)
    
    def _user_to_dict(self, user) -> Dict[str, Any]:
        """Convert twikit user object to dictionary"""
        return {
            'id': getattr(user, 'id', None),
            'username': getattr(user, 'screen_name', None),
            'display_name': getattr(user, 'name', None),
            'description': getattr(user, 'description', None),
            'followers_count': getattr(user, 'followers_count', 0),
            'following_count': getattr(user, 'friends_count', 0),
            'tweet_count': getattr(user, 'statuses_count', 0),
            'verified': getattr(user, 'verified', False),
            'profile_image_url': getattr(user, 'profile_image_url_https', None),
            'created_at': getattr(user, 'created_at', None),
            'location': getattr(user, 'location', None),
            'url': getattr(user, 'url', None)
        }
    
    def _tweet_to_dict(self, tweet) -> Dict[str, Any]:
        """Convert twikit tweet object to dictionary"""
        return {
            'id': getattr(tweet, 'id', None),
            'id_str': str(getattr(tweet, 'id', '')),
            'text': getattr(tweet, 'full_text', getattr(tweet, 'text', None)),
            'created_at': getattr(tweet, 'created_at', None),
            'user': self._user_to_dict(getattr(tweet, 'user', None)) if hasattr(tweet, 'user') else None,
            'retweet_count': getattr(tweet, 'retweet_count', 0),
            'favorite_count': getattr(tweet, 'favorite_count', 0),
            'reply_count': getattr(tweet, 'reply_count', 0),
            'quote_count': getattr(tweet, 'quote_count', 0),
            'view_count': getattr(tweet, 'view_count', None),
            'lang': getattr(tweet, 'lang', None),
            'hashtags': [tag.get('text', '') for tag in getattr(tweet, 'hashtags', [])],
            'urls': [url.get('expanded_url', '') for url in getattr(tweet, 'urls', [])],
            'mentions': [mention.get('screen_name', '') for mention in getattr(tweet, 'user_mentions', [])],
            'is_retweet': getattr(tweet, 'retweeted', False),
            'in_reply_to_status_id': getattr(tweet, 'in_reply_to_status_id', None),
            'in_reply_to_user_id': getattr(tweet, 'in_reply_to_user_id', None),
            'in_reply_to_screen_name': getattr(tweet, 'in_reply_to_screen_name', None),
            'quoted_status_id': getattr(tweet, 'quoted_status_id', None),
            'retweeted_status': getattr(tweet, 'retweeted_status', {}),
            'quoted_status': getattr(tweet, 'quoted_status', {}),
            'conversation_id': getattr(tweet, 'conversation_id', None),
            'possibly_sensitive': getattr(tweet, 'possibly_sensitive', False),
            'source': getattr(tweet, 'source', None),
            'withheld_copyright': getattr(tweet, 'withheld_copyright', False),
            'withheld_in_countries': getattr(tweet, 'withheld_in_countries', [])
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get crawler status information"""
        return {
            'status': self.status.value,
            'is_authenticated': self.session.is_authenticated,
            'error_count': self.error_count,
            'last_error': str(self.last_error) if self.last_error else None,
            'last_activity': self.session.last_activity.isoformat() if self.session.last_activity else None,
            'rate_limit_status': self.rate_limiter.get_status()
        }
    
    async def close(self):
        """Close the crawler and clean up resources"""
        self.status = CrawlerStatus.STOPPED
        
        if self.session and self.config.auto_save_session:
            self.session.save_session()
        
        if self.client:
            # Close client connections if available
            pass
        
        self.logger.info("Twitter crawler closed")


# Utility functions
def create_twitter_crawler(
    username: Optional[str] = None,
    password: Optional[str] = None,
    email: Optional[str] = None,
    **config_kwargs
) -> TwitterCrawler:
    """Create a configured Twitter crawler"""
    config = TwitterConfig(
        username=username,
        password=password,
        email=email,
        **config_kwargs
    )
    return TwitterCrawler(config)


async def quick_tweet_search(
    query: str,
    count: int = 20,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Quick tweet search function"""
    crawler = create_twitter_crawler(username=username, password=password)
    
    try:
        await crawler.initialize()
        return await crawler.search_tweets(query, count=count)
    finally:
        await crawler.close()


__all__ = [
    'TwitterCrawler', 'TwitterConfig', 'TwitterSession', 'RateLimiter',
    'CrawlerError', 'AuthenticationError', 'RateLimitError', 'SessionError',
    'CrawlerStatus', 'RateLimitInfo',
    'create_twitter_crawler', 'quick_tweet_search'
]