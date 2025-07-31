"""
Comment Parser Module
=====================

Specialized parsers for YouTube comment data extraction and analysis.
Handles comment threads, sentiment analysis, and content moderation.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
import re
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from .base_parser import BaseParser, ParseResult, ParseStatus, ContentType
from ..api.data_models import CommentData

logger = logging.getLogger(__name__)

# Optional dependencies for sentiment analysis
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


class CommentParser(BaseParser):
    """Main comment parser for extracting comment data."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.thread_parser = CommentThreadParser(**kwargs)
        self.analyzer = CommentAnalyzer(**kwargs)
    
    async def parse(self, data: Union[Dict, List, str], content_type: ContentType) -> ParseResult:
        """Parse comment data from various sources."""
        start_time = time.time()
        
        try:
            comments = []
            
            if isinstance(data, list):
                # List of comments
                for comment_data in data:
                    comment = await self._parse_single_comment(comment_data)
                    if comment:
                        comments.append(comment)
            elif isinstance(data, dict):
                if 'items' in data:
                    # YouTube API comment list response
                    for item in data['items']:
                        comment = await self._parse_api_comment(item)
                        if comment:
                            comments.append(comment)
                else:
                    # Single comment
                    comment = await self._parse_single_comment(data)
                    if comment:
                        comments.append(comment)
            elif isinstance(data, str):
                # HTML content with comments
                comments = await self._parse_comments_from_html(data)
            
            # Analyze comments if requested
            if self.config.analyze_sentiment and comments:
                for comment in comments:
                    sentiment_result = await self.analyzer.analyze_sentiment(comment.text)
                    if sentiment_result.is_successful():
                        comment.sentiment = sentiment_result.data.get('sentiment')
                        comment.sentiment_score = sentiment_result.data.get('score', 0.0)
            
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=comments,
                execution_time=execution_time,
                metadata={
                    'parser': 'CommentParser',
                    'total_comments': len(comments),
                    'has_sentiment': self.config.analyze_sentiment
                }
            )
            
        except Exception as e:
            logger.error(f"Comment parsing failed: {e}")
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _parse_single_comment(self, data: Dict[str, Any]) -> Optional[CommentData]:
        """Parse a single comment from dictionary data."""
        try:
            # Handle both API format and custom format
            if 'snippet' in data:
                # YouTube API format
                return await self._parse_api_comment(data)
            else:
                # Custom format
                return CommentData(
                    comment_id=data.get('id', data.get('comment_id', '')),
                    author_name=data.get('author', data.get('author_name', '')),
                    author_channel_id=data.get('author_channel_id'),
                    text=data.get('text', data.get('content', '')),
                    like_count=int(data.get('like_count', 0)),
                    reply_count=int(data.get('reply_count', 0)),
                    published_at=self._parse_datetime(data.get('published_at')),
                    updated_at=self._parse_datetime(data.get('updated_at')),
                    parent_id=data.get('parent_id'),
                    is_reply=bool(data.get('parent_id')),
                    author_channel_url=data.get('author_channel_url')
                )
        except Exception as e:
            logger.error(f"Failed to parse comment: {e}")
            return None
    
    async def _parse_api_comment(self, item: Dict[str, Any]) -> Optional[CommentData]:
        """Parse comment from YouTube API response."""
        try:
            snippet = item.get('snippet', {})
            
            # Handle both comment threads and direct comments
            if 'topLevelComment' in snippet:
                # Comment thread
                comment_snippet = snippet['topLevelComment']['snippet']
                comment_id = snippet['topLevelComment']['id']
            else:
                # Direct comment
                comment_snippet = snippet
                comment_id = item['id']
            
            return CommentData(
                comment_id=comment_id,
                author_name=comment_snippet.get('authorDisplayName', ''),
                author_channel_id=comment_snippet.get('authorChannelId', {}).get('value'),
                text=comment_snippet.get('textDisplay', comment_snippet.get('textOriginal', '')),
                like_count=int(comment_snippet.get('likeCount', 0)),
                reply_count=int(snippet.get('totalReplyCount', 0)),
                published_at=self._parse_datetime(comment_snippet.get('publishedAt')),
                updated_at=self._parse_datetime(comment_snippet.get('updatedAt')),
                parent_id=comment_snippet.get('parentId'),
                is_reply=bool(comment_snippet.get('parentId')),
                author_channel_url=comment_snippet.get('authorChannelUrl')
            )
        except Exception as e:
            logger.error(f"Failed to parse API comment: {e}")
            return None
    
    async def _parse_comments_from_html(self, html: str) -> List[CommentData]:
        """Parse comments from HTML content (web scraping)."""
        comments = []
        # This would implement HTML parsing logic for comment sections
        # For now, return empty list as web scraping is complex
        logger.warning("HTML comment parsing not fully implemented")
        return comments


class CommentThreadParser(BaseParser):
    """Parser for comment threads and reply structures."""
    
    async def parse(self, data: Union[Dict, List], content_type: ContentType) -> ParseResult:
        """Parse comment thread structure."""
        start_time = time.time()
        
        try:
            threads = []
            
            if isinstance(data, list):
                # List of comment threads
                for thread_data in data:
                    thread = await self._parse_thread(thread_data)
                    if thread:
                        threads.append(thread)
            elif isinstance(data, dict):
                if 'items' in data:
                    # YouTube API comment threads response
                    for item in data['items']:
                        thread = await self._parse_api_thread(item)
                        if thread:
                            threads.append(thread)
                else:
                    # Single thread
                    thread = await self._parse_thread(data)
                    if thread:
                        threads.append(thread)
            
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=threads,
                execution_time=execution_time,
                metadata={'parser': 'CommentThreadParser', 'total_threads': len(threads)}
            )
            
        except Exception as e:
            logger.error(f"Comment thread parsing failed: {e}")
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _parse_thread(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a single comment thread."""
        try:
            return {
                'thread_id': data.get('id', ''),
                'top_level_comment': data.get('top_level_comment'),
                'replies': data.get('replies', []),
                'total_reply_count': int(data.get('total_reply_count', 0)),
                'can_reply': data.get('can_reply', True)
            }
        except Exception as e:
            logger.error(f"Failed to parse thread: {e}")
            return None
    
    async def _parse_api_thread(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse comment thread from YouTube API response."""
        try:
            snippet = item.get('snippet', {})
            
            thread = {
                'thread_id': item['id'],
                'video_id': snippet.get('videoId'),
                'top_level_comment': snippet.get('topLevelComment'),
                'total_reply_count': int(snippet.get('totalReplyCount', 0)),
                'can_reply': snippet.get('canReply', True),
                'replies': []
            }
            
            # Parse replies if present
            if 'replies' in item:
                for reply in item['replies'].get('comments', []):
                    reply_data = await self._parse_reply(reply)
                    if reply_data:
                        thread['replies'].append(reply_data)
            
            return thread
        except Exception as e:
            logger.error(f"Failed to parse API thread: {e}")
            return None
    
    async def _parse_reply(self, reply: Dict[str, Any]) -> Optional[CommentData]:
        """Parse a comment reply."""
        try:
            snippet = reply.get('snippet', {})
            
            return CommentData(
                comment_id=reply['id'],
                author_name=snippet.get('authorDisplayName', ''),
                author_channel_id=snippet.get('authorChannelId', {}).get('value'),
                text=snippet.get('textDisplay', ''),
                like_count=int(snippet.get('likeCount', 0)),
                reply_count=0,  # Replies don't have their own replies
                published_at=self._parse_datetime(snippet.get('publishedAt')),
                updated_at=self._parse_datetime(snippet.get('updatedAt')),
                parent_id=snippet.get('parentId'),
                is_reply=True,
                author_channel_url=snippet.get('authorChannelUrl')
            )
        except Exception as e:
            logger.error(f"Failed to parse reply: {e}")
            return None


class CommentAnalyzer(BaseParser):
    """Parser for comment content analysis and sentiment."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sentiment_analyzer = None
        
        # Initialize sentiment analyzer if available
        if HAS_NLTK:
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            except LookupError:
                logger.warning("NLTK VADER lexicon not found. Run nltk.download('vader_lexicon')")
    
    async def analyze_sentiment(self, text: str) -> ParseResult:
        """Analyze sentiment of comment text."""
        start_time = time.time()
        
        try:
            sentiment_data = {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'method': 'none'
            }
            
            # Clean text for analysis
            clean_text = self._clean_text(text)
            
            if not clean_text:
                return ParseResult(
                    status=ParseStatus.SUCCESS,
                    data=sentiment_data,
                    execution_time=time.time() - start_time
                )
            
            # Use NLTK VADER if available
            if self.sentiment_analyzer:
                scores = self.sentiment_analyzer.polarity_scores(clean_text)
                compound_score = scores['compound']
                
                if compound_score >= 0.05:
                    sentiment = 'positive'
                elif compound_score <= -0.05:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                sentiment_data = {
                    'sentiment': sentiment,
                    'score': compound_score,
                    'confidence': abs(compound_score),
                    'method': 'nltk_vader',
                    'details': scores
                }
            
            # Fallback to TextBlob if NLTK not available
            elif HAS_TEXTBLOB:
                blob = TextBlob(clean_text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    sentiment = 'positive'
                elif polarity < -0.1:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                sentiment_data = {
                    'sentiment': sentiment,
                    'score': polarity,
                    'confidence': abs(polarity),
                    'method': 'textblob',
                    'subjectivity': blob.sentiment.subjectivity
                }
            
            # Simple keyword-based fallback
            else:
                sentiment_data = self._keyword_sentiment_analysis(clean_text)
            
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=sentiment_data,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove empty lines
        text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
        
        return text
    
    def _keyword_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Simple keyword-based sentiment analysis fallback."""
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'awesome', 'love', 'like', 'best',
            'fantastic', 'wonderful', 'perfect', 'nice', 'cool', 'thanks', 'thank'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'hate', 'dislike', 'worst', 'horrible',
            'disgusting', 'stupid', 'dumb', 'sucks', 'trash', 'garbage', 'boring'
        ]
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            score = min(positive_count / len(words), 1.0)
        elif negative_count > positive_count:
            sentiment = 'negative'
            score = -min(negative_count / len(words), 1.0)
        else:
            sentiment = 'neutral'
            score = 0.0
        
        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': abs(score),
            'method': 'keyword_based',
            'positive_count': positive_count,
            'negative_count': negative_count
        }
    
    async def detect_spam(self, comment: CommentData) -> ParseResult:
        """Detect if comment is likely spam."""
        start_time = time.time()
        
        try:
            spam_indicators = []
            spam_score = 0.0
            
            # Check for excessive capitalization
            if len(comment.text) > 0:
                caps_ratio = sum(1 for c in comment.text if c.isupper()) / len(comment.text)
                if caps_ratio > 0.7:
                    spam_indicators.append('excessive_caps')
                    spam_score += 0.3
            
            # Check for repetitive characters
            if re.search(r'(.)\1{4,}', comment.text):
                spam_indicators.append('repetitive_chars')
                spam_score += 0.2
            
            # Check for excessive punctuation
            punct_count = len(re.findall(r'[!?]{3,}', comment.text))
            if punct_count > 0:
                spam_indicators.append('excessive_punctuation')
                spam_score += 0.2
            
            # Check for URLs (common in spam)
            url_count = len(re.findall(r'http\S+|www\S+', comment.text))
            if url_count > 0:
                spam_indicators.append('contains_urls')
                spam_score += 0.4
            
            # Check for very short comments with high engagement (suspicious)
            if len(comment.text.split()) < 3 and comment.like_count > 50:
                spam_indicators.append('suspicious_engagement')
                spam_score += 0.3
            
            is_spam = spam_score > 0.5
            
            result_data = {
                'is_spam': is_spam,
                'spam_score': min(spam_score, 1.0),
                'indicators': spam_indicators,
                'confidence': min(spam_score, 1.0)
            }
            
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=result_data,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Spam detection failed: {e}")
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )


# Utility functions
def extract_mentions(text: str) -> List[str]:
    """Extract @mentions from comment text."""
    return re.findall(r'@(\w+)', text)


def extract_hashtags(text: str) -> List[str]:
    """Extract #hashtags from comment text."""
    return re.findall(r'#(\w+)', text)


def calculate_comment_engagement_score(comment: CommentData) -> float:
    """Calculate engagement score for a comment."""
    # Simple scoring: likes + replies with diminishing returns
    likes_score = min(comment.like_count / 100.0, 1.0)  # Cap at 100 likes = 1.0
    replies_score = min(comment.reply_count / 20.0, 1.0)  # Cap at 20 replies = 1.0
    
    return (likes_score * 0.7) + (replies_score * 0.3)


def filter_comments_by_sentiment(comments: List[CommentData], sentiment: str) -> List[CommentData]:
    """Filter comments by sentiment."""
    return [c for c in comments if hasattr(c, 'sentiment') and c.sentiment == sentiment]


def get_comment_statistics(comments: List[CommentData]) -> Dict[str, Any]:
    """Get statistics about a list of comments."""
    if not comments:
        return {}
    
    total_likes = sum(c.like_count for c in comments)
    total_replies = sum(c.reply_count for c in comments)
    avg_length = sum(len(c.text.split()) for c in comments) / len(comments)
    
    # Sentiment distribution (if available)
    sentiment_counts = {}
    if any(hasattr(c, 'sentiment') for c in comments):
        sentiments = [c.sentiment for c in comments if hasattr(c, 'sentiment')]
        sentiment_counts = {
            'positive': sentiments.count('positive'),
            'negative': sentiments.count('negative'),
            'neutral': sentiments.count('neutral')
        }
    
    return {
        'total_comments': len(comments),
        'total_likes': total_likes,
        'total_replies': total_replies,
        'average_likes': total_likes / len(comments),
        'average_replies': total_replies / len(comments),
        'average_word_count': avg_length,
        'sentiment_distribution': sentiment_counts
    }


__all__ = [
    'CommentParser',
    'CommentThreadParser',
    'CommentAnalyzer',
    'extract_mentions',
    'extract_hashtags',
    'calculate_comment_engagement_score',
    'filter_comments_by_sentiment',
    'get_comment_statistics'
]