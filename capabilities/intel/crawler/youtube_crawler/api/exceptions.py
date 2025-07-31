"""
YouTube Crawler Exceptions Module
=================================

Custom exception classes for YouTube content crawling operations.
Provides specific error handling for different failure scenarios.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

from typing import Optional, Dict, Any, List


class YouTubeCrawlerError(Exception):
    """Base exception for YouTube crawler operations."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        retry_after: Optional[int] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.retry_after = retry_after

    def __str__(self) -> str:
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.retry_after:
            base_msg += f" (retry after {self.retry_after}s)"
        return base_msg

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details,
            'retry_after': self.retry_after
        }


class APIQuotaExceededError(YouTubeCrawlerError):
    """Raised when YouTube API quota is exceeded."""

    def __init__(
        self,
        message: str = "YouTube API quota exceeded",
        quota_limit: Optional[int] = None,
        quota_used: Optional[int] = None,
        reset_time: Optional[str] = None,
        **kwargs
    ):
        details = {
            'quota_limit': quota_limit,
            'quota_used': quota_used,
            'reset_time': reset_time
        }
        super().__init__(
            message=message,
            error_code="QUOTA_EXCEEDED",
            details=details,
            **kwargs
        )
        self.quota_limit = quota_limit
        self.quota_used = quota_used
        self.reset_time = reset_time


class RateLimitExceededError(YouTubeCrawlerError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        requests_per_minute: Optional[int] = None,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        details = {
            'requests_per_minute': requests_per_minute,
            'rate_limit_type': 'requests_per_minute'
        }
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details,
            retry_after=retry_after,
            **kwargs
        )
        self.requests_per_minute = requests_per_minute


class VideoNotFoundError(YouTubeCrawlerError):
    """Raised when a video cannot be found or accessed."""

    def __init__(
        self,
        video_id: str,
        message: Optional[str] = None,
        reason: Optional[str] = None,
        **kwargs
    ):
        if message is None:
            message = f"Video not found: {video_id}"

        details = {
            'video_id': video_id,
            'reason': reason
        }
        super().__init__(
            message=message,
            error_code="VIDEO_NOT_FOUND",
            details=details,
            **kwargs
        )
        self.video_id = video_id
        self.reason = reason


class ChannelNotFoundError(YouTubeCrawlerError):
    """Raised when a channel cannot be found or accessed."""

    def __init__(
        self,
        channel_id: str,
        message: Optional[str] = None,
        reason: Optional[str] = None,
        **kwargs
    ):
        if message is None:
            message = f"Channel not found: {channel_id}"

        details = {
            'channel_id': channel_id,
            'reason': reason
        }
        super().__init__(
            message=message,
            error_code="CHANNEL_NOT_FOUND",
            details=details,
            **kwargs
        )
        self.channel_id = channel_id
        self.reason = reason


class PlaylistNotFoundError(YouTubeCrawlerError):
    """Raised when a playlist cannot be found or accessed."""

    def __init__(
        self,
        playlist_id: str,
        message: Optional[str] = None,
        reason: Optional[str] = None,
        **kwargs
    ):
        if message is None:
            message = f"Playlist not found: {playlist_id}"

        details = {
            'playlist_id': playlist_id,
            'reason': reason
        }
        super().__init__(
            message=message,
            error_code="PLAYLIST_NOT_FOUND",
            details=details,
            **kwargs
        )
        self.playlist_id = playlist_id
        self.reason = reason


class AccessRestrictedError(YouTubeCrawlerError):
    """Raised when content access is restricted."""

    def __init__(
        self,
        resource_id: str,
        resource_type: str = "video",
        restriction_type: Optional[str] = None,
        message: Optional[str] = None,
        **kwargs
    ):
        if message is None:
            message = f"Access restricted for {resource_type}: {resource_id}"

        details = {
            'resource_id': resource_id,
            'resource_type': resource_type,
            'restriction_type': restriction_type
        }
        super().__init__(
            message=message,
            error_code="ACCESS_RESTRICTED",
            details=details,
            **kwargs
        )
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.restriction_type = restriction_type


class AuthenticationError(YouTubeCrawlerError):
    """Raised when authentication fails."""

    def __init__(
        self,
        message: str = "Authentication failed",
        auth_type: Optional[str] = None,
        **kwargs
    ):
        details = {'auth_type': auth_type}
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_FAILED",
            details=details,
            **kwargs
        )
        self.auth_type = auth_type


class ConfigurationError(YouTubeCrawlerError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        message: str,
        config_field: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None,
        **kwargs
    ):
        details = {
            'config_field': config_field,
            'expected_type': expected_type,
            'actual_value': str(actual_value) if actual_value is not None else None
        }
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details,
            **kwargs
        )
        self.config_field = config_field
        self.expected_type = expected_type
        self.actual_value = actual_value


class ParsingError(YouTubeCrawlerError):
    """Raised when data parsing fails."""

    def __init__(
        self,
        message: str,
        parser_type: Optional[str] = None,
        source_data: Optional[str] = None,
        parse_stage: Optional[str] = None,
        **kwargs
    ):
        details = {
            'parser_type': parser_type,
            'source_data': source_data[:500] if source_data else None,  # Truncate for safety
            'parse_stage': parse_stage
        }
        super().__init__(
            message=message,
            error_code="PARSING_ERROR",
            details=details,
            **kwargs
        )
        self.parser_type = parser_type
        self.source_data = source_data
        self.parse_stage = parse_stage


class NetworkError(YouTubeCrawlerError):
    """Raised when network operations fail."""

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
        **kwargs
    ):
        details = {
            'url': url,
            'status_code': status_code,
            'response_text': response_text[:500] if response_text else None
        }
        super().__init__(
            message=message,
            error_code="NETWORK_ERROR",
            details=details,
            **kwargs
        )
        self.url = url
        self.status_code = status_code
        self.response_text = response_text


class DatabaseError(YouTubeCrawlerError):
    """Raised when database operations fail."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ):
        details = {
            'operation': operation,
            'table': table,
            'query': query[:200] if query else None  # Truncate for safety
        }
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details=details,
            **kwargs
        )
        self.operation = operation
        self.table = table
        self.query = query


class ValidationError(YouTubeCrawlerError):
    """Raised when data validation fails."""

    def __init__(
        self,
        message: str,
        validation_errors: Optional[List[str]] = None,
        field_name: Optional[str] = None,
        **kwargs
    ):
        details = {
            'validation_errors': validation_errors or [],
            'field_name': field_name
        }
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details,
            **kwargs
        )
        self.validation_errors = validation_errors or []
        self.field_name = field_name


class CacheError(YouTubeCrawlerError):
    """Raised when cache operations fail."""

    def __init__(
        self,
        message: str,
        cache_backend: Optional[str] = None,
        operation: Optional[str] = None,
        key: Optional[str] = None,
        **kwargs
    ):
        details = {
            'cache_backend': cache_backend,
            'operation': operation,
            'key': key
        }
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            details=details,
            **kwargs
        )
        self.cache_backend = cache_backend
        self.operation = operation
        self.key = key


class TranscriptError(YouTubeCrawlerError):
    """Raised when transcript extraction fails."""

    def __init__(
        self,
        message: str,
        video_id: Optional[str] = None,
        language: Optional[str] = None,
        transcript_type: Optional[str] = None,
        **kwargs
    ):
        details = {
            'video_id': video_id,
            'language': language,
            'transcript_type': transcript_type
        }
        super().__init__(
            message=message,
            error_code="TRANSCRIPT_ERROR",
            details=details,
            **kwargs
        )
        self.video_id = video_id
        self.language = language
        self.transcript_type = transcript_type


class CommentError(YouTubeCrawlerError):
    """Raised when comment extraction fails."""

    def __init__(
        self,
        message: str,
        video_id: Optional[str] = None,
        comment_id: Optional[str] = None,
        **kwargs
    ):
        details = {
            'video_id': video_id,
            'comment_id': comment_id
        }
        super().__init__(
            message=message,
            error_code="COMMENT_ERROR",
            details=details,
            **kwargs
        )
        self.video_id = video_id
        self.comment_id = comment_id


class ThumbnailError(YouTubeCrawlerError):
    """Raised when thumbnail processing fails."""

    def __init__(
        self,
        message: str,
        video_id: Optional[str] = None,
        thumbnail_url: Optional[str] = None,
        size: Optional[str] = None,
        **kwargs
    ):
        details = {
            'video_id': video_id,
            'thumbnail_url': thumbnail_url,
            'size': size
        }
        super().__init__(
            message=message,
            error_code="THUMBNAIL_ERROR",
            details=details,
            **kwargs
        )
        self.video_id = video_id
        self.thumbnail_url = thumbnail_url
        self.size = size


# Exception utilities

def handle_api_error(response_data: Dict[str, Any]) -> YouTubeCrawlerError:
    """Convert API error response to appropriate exception."""
    error_info = response_data.get('error', {})
    error_code = error_info.get('code')
    error_message = error_info.get('message', 'Unknown API error')

    # Map specific error codes to exceptions
    if error_code == 403:
        if 'quota' in error_message.lower():
            return APIQuotaExceededError(error_message)
        else:
            return AccessRestrictedError(
                resource_id="unknown",
                message=error_message,
                restriction_type="api_forbidden"
            )
    elif error_code == 404:
        return YouTubeCrawlerError(
            message=error_message,
            error_code="NOT_FOUND"
        )
    elif error_code == 429:
        return RateLimitExceededError(error_message)
    elif error_code == 401:
        return AuthenticationError(error_message)
    else:
        return YouTubeCrawlerError(
            message=error_message,
            error_code=str(error_code) if error_code else "API_ERROR",
            details=error_info
        )


def is_retriable_error(error: Exception) -> bool:
    """Check if an error is retriable."""
    if isinstance(error, YouTubeCrawlerError):
        retriable_codes = {
            "NETWORK_ERROR",
            "RATE_LIMIT_EXCEEDED",
            "DATABASE_ERROR",
            "CACHE_ERROR"
        }
        return error.error_code in retriable_codes

    # Network-related exceptions are generally retriable
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True

    return False


def get_retry_delay(error: Exception, attempt: int) -> int:
    """Get recommended retry delay based on error type."""
    if isinstance(error, RateLimitExceededError) and error.retry_after:
        return error.retry_after
    elif isinstance(error, APIQuotaExceededError):
        return 3600  # 1 hour for quota exceeded
    elif isinstance(error, NetworkError):
        return min(2 ** attempt, 60)  # Exponential backoff, max 60s
    else:
        return min(attempt * 5, 30)  # Linear backoff, max 30s


class ErrorReporter:
    """Utility class for error reporting and metrics."""

    def __init__(self):
        self.error_counts = {}
        self.error_history = []

    def report_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Report an error for tracking."""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        error_record = {
            'timestamp': str(datetime.utcnow()),
            'error_type': error_type,
            'message': str(error),
            'context': context or {}
        }

        if isinstance(error, YouTubeCrawlerError):
            error_record.update(error.to_dict())

        self.error_history.append(error_record)

        # Keep only last 1000 errors
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics."""
        return {
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts.copy(),
            'most_common_error': max(self.error_counts, key=self.error_counts.get) if self.error_counts else None,
            'recent_errors': self.error_history[-10:]  # Last 10 errors
        }

    def clear_history(self):
        """Clear error history."""
        self.error_counts.clear()
        self.error_history.clear()


# Global error reporter instance
error_reporter = ErrorReporter()
