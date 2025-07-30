"""
Base Parser Framework
=====================

Core parser framework for YouTube content extraction.
Provides abstract base classes and common functionality for all parsers.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ParseStatus(Enum):
    """Parse operation status."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class ContentType(Enum):
    """Content type enumeration."""
    VIDEO = "video"
    CHANNEL = "channel"
    PLAYLIST = "playlist"
    COMMENT = "comment"
    TRANSCRIPT = "transcript"
    THUMBNAIL = "thumbnail"
    METADATA = "metadata"


class MediaType(Enum):
    """Media type enumeration."""
    JSON = "application/json"
    HTML = "text/html"
    XML = "application/xml"
    TEXT = "text/plain"
    BINARY = "application/octet-stream"
    IMAGE = "image/*"
    VIDEO = "video/*"
    AUDIO = "audio/*"


@dataclass
class ParseResult:
    """Result of a parsing operation."""
    status: ParseStatus
    data: Optional[Any] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def is_successful(self) -> bool:
        """Check if parsing was successful."""
        return self.status == ParseStatus.SUCCESS

    def is_partial(self) -> bool:
        """Check if parsing was partially successful."""
        return self.status == ParseStatus.PARTIAL

    def has_data(self) -> bool:
        """Check if result contains data."""
        return self.data is not None

    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'status': self.status.value,
            'data': self.data,
            'error_message': self.error_message,
            'error_code': self.error_code,
            'execution_time': self.execution_time,
            'metadata': self.metadata,
            'warnings': self.warnings
        }


@dataclass
class ParserConfig:
    """Configuration for parsers."""
    # General settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0

    # Content extraction settings
    extract_metadata: bool = True
    extract_statistics: bool = True
    extract_content: bool = True
    extract_links: bool = False
    extract_images: bool = False

    # Processing settings
    validate_input: bool = True
    validate_output: bool = True
    normalize_data: bool = True
    clean_text: bool = True
    preserve_formatting: bool = False

    # Performance settings
    max_content_length: int = 1024 * 1024  # 1MB
    max_items_per_batch: int = 100
    enable_compression: bool = False

    # Output settings
    include_raw_data: bool = False
    include_debug_info: bool = False
    output_format: str = "json"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'enable_caching': self.enable_caching,
            'cache_ttl': self.cache_ttl,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'timeout': self.timeout,
            'extract_metadata': self.extract_metadata,
            'extract_statistics': self.extract_statistics,
            'extract_content': self.extract_content,
            'extract_links': self.extract_links,
            'extract_images': self.extract_images,
            'validate_input': self.validate_input,
            'validate_output': self.validate_output,
            'normalize_data': self.normalize_data,
            'clean_text': self.clean_text,
            'preserve_formatting': self.preserve_formatting,
            'max_content_length': self.max_content_length,
            'max_items_per_batch': self.max_items_per_batch,
            'enable_compression': self.enable_compression,
            'include_raw_data': self.include_raw_data,
            'include_debug_info': self.include_debug_info,
            'output_format': self.output_format
        }


class BaseParser(ABC):
    """Abstract base class for all parsers."""

    def __init__(self, config: Optional[ParserConfig] = None):
        self.config = config or ParserConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stats = {
            'total_parsed': 0,
            'successful_parsed': 0,
            'failed_parsed': 0,
            'total_time': 0.0,
            'start_time': time.time()
        }

    @property
    def name(self) -> str:
        """Get parser name."""
        return self.__class__.__name__

    @property
    @abstractmethod
    def supported_content_types(self) -> List[ContentType]:
        """Get list of supported content types."""
        pass

    @property
    @abstractmethod
    def supported_media_types(self) -> List[MediaType]:
        """Get list of supported media types."""
        pass

    @abstractmethod
    async def parse(self, content: Any, content_type: ContentType, **kwargs) -> ParseResult:
        """Parse content and return structured data."""
        pass

    def validate_input(self, content: Any, content_type: ContentType) -> bool:
        """Validate input content."""
        if not self.config.validate_input:
            return True

        try:
            # Check content type support
            if content_type not in self.supported_content_types:
                self.logger.warning(f"Unsupported content type: {content_type}")
                return False

            # Check content size
            if hasattr(content, '__len__'):
                if len(content) > self.config.max_content_length:
                    self.logger.warning(f"Content too large: {len(content)} bytes")
                    return False

            # Check content is not empty
            if not content:
                self.logger.warning("Empty content provided")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            return False

    def validate_output(self, data: Any) -> bool:
        """Validate output data."""
        if not self.config.validate_output:
            return True

        try:
            # Basic validation - data should not be None
            if data is None:
                return False

            # Additional validation can be implemented by subclasses
            return True

        except Exception as e:
            self.logger.error(f"Output validation error: {e}")
            return False

    def normalize_data(self, data: Any) -> Any:
        """Normalize parsed data."""
        if not self.config.normalize_data:
            return data

        try:
            # Basic normalization - can be extended by subclasses
            if isinstance(data, dict):
                return self._normalize_dict(data)
            elif isinstance(data, list):
                return [self.normalize_data(item) for item in data]
            else:
                return data

        except Exception as e:
            self.logger.error(f"Data normalization error: {e}")
            return data

    def _normalize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize dictionary data."""
        normalized = {}
        for key, value in data.items():
            # Normalize keys (lowercase, replace spaces with underscores)
            normalized_key = key.lower().replace(' ', '_')
            normalized[normalized_key] = self.normalize_data(value)
        return normalized

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not self.config.clean_text or not isinstance(text, str):
            return text

        try:
            # Remove excessive whitespace
            text = ' '.join(text.split())

            # Remove special characters if needed
            import re
            text = re.sub(r'[^\w\s\-_.,!?;:]', '', text)

            return text.strip()

        except Exception as e:
            self.logger.error(f"Text cleaning error: {e}")
            return text

    async def parse_with_retry(self, content: Any, content_type: ContentType, **kwargs) -> ParseResult:
        """Parse content with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()

                # Validate input
                if not self.validate_input(content, content_type):
                    return ParseResult(
                        status=ParseStatus.FAILED,
                        error_message="Input validation failed",
                        error_code="INVALID_INPUT"
                    )

                # Parse content
                result = await self.parse(content, content_type, **kwargs)
                result.execution_time = time.time() - start_time

                # Update statistics
                self.stats['total_parsed'] += 1
                self.stats['total_time'] += result.execution_time

                if result.is_successful():
                    self.stats['successful_parsed'] += 1
                else:
                    self.stats['failed_parsed'] += 1

                return result

            except Exception as e:
                last_error = e
                self.logger.warning(f"Parse attempt {attempt + 1} failed: {e}")

                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    break

        # All retries failed
        self.stats['total_parsed'] += 1
        self.stats['failed_parsed'] += 1

        return ParseResult(
            status=ParseStatus.FAILED,
            error_message=f"Parsing failed after {self.config.max_retries + 1} attempts: {last_error}",
            error_code="RETRY_EXHAUSTED"
        )

    async def batch_parse(
        self,
        items: List[tuple],  # (content, content_type, kwargs)
        **kwargs
    ) -> List[ParseResult]:
        """Parse multiple items in batch."""
        results = []

        # Process in batches
        batch_size = self.config.max_items_per_batch
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]

            # Parse batch items concurrently
            tasks = []
            for content, content_type, item_kwargs in batch:
                merged_kwargs = {**kwargs, **item_kwargs}
                task = self.parse_with_retry(content, content_type, **merged_kwargs)
                tasks.append(task)

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    results.append(ParseResult(
                        status=ParseStatus.FAILED,
                        error_message=str(result),
                        error_code="BATCH_ERROR"
                    ))
                else:
                    results.append(result)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get parser statistics."""
        runtime = time.time() - self.stats['start_time']
        total_parsed = self.stats['total_parsed']

        return {
            'parser_name': self.name,
            'total_parsed': total_parsed,
            'successful_parsed': self.stats['successful_parsed'],
            'failed_parsed': self.stats['failed_parsed'],
            'success_rate': (self.stats['successful_parsed'] / max(total_parsed, 1)) * 100,
            'average_parse_time': self.stats['total_time'] / max(total_parsed, 1),
            'total_runtime': runtime,
            'throughput': total_parsed / max(runtime, 1),
            'supported_content_types': [ct.value for ct in self.supported_content_types],
            'supported_media_types': [mt.value for mt in self.supported_media_types]
        }

    def reset_statistics(self):
        """Reset parser statistics."""
        self.stats = {
            'total_parsed': 0,
            'successful_parsed': 0,
            'failed_parsed': 0,
            'total_time': 0.0,
            'start_time': time.time()
        }


class ParserRegistry:
    """Registry for managing parsers."""

    def __init__(self):
        self._parsers: Dict[str, Type[BaseParser]] = {}
        self._instances: Dict[str, BaseParser] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def register_parser(self, name: str, parser_class: Type[BaseParser]):
        """Register a parser class."""
        if not issubclass(parser_class, BaseParser):
            raise ValueError(f"Parser class must inherit from BaseParser: {parser_class}")

        self._parsers[name] = parser_class
        self.logger.info(f"Registered parser: {name}")

    def unregister_parser(self, name: str):
        """Unregister a parser."""
        if name in self._parsers:
            del self._parsers[name]
            if name in self._instances:
                del self._instances[name]
            self.logger.info(f"Unregistered parser: {name}")

    def get_parser(self, name: str, config: Optional[ParserConfig] = None) -> BaseParser:
        """Get parser instance."""
        if name not in self._parsers:
            raise ValueError(f"Parser not found: {name}")

        # Return cached instance if config is None
        if config is None and name in self._instances:
            return self._instances[name]

        # Create new instance
        parser_class = self._parsers[name]
        instance = parser_class(config)

        # Cache instance if no specific config provided
        if config is None:
            self._instances[name] = instance

        return instance

    def list_parsers(self) -> List[str]:
        """List all registered parsers."""
        return list(self._parsers.keys())

    def get_parser_info(self, name: str) -> Dict[str, Any]:
        """Get information about a parser."""
        if name not in self._parsers:
            raise ValueError(f"Parser not found: {name}")

        parser_class = self._parsers[name]

        # Create temporary instance to get info
        temp_instance = parser_class()

        return {
            'name': name,
            'class': parser_class.__name__,
            'module': parser_class.__module__,
            'supported_content_types': [ct.value for ct in temp_instance.supported_content_types],
            'supported_media_types': [mt.value for mt in temp_instance.supported_media_types],
            'doc': parser_class.__doc__
        }

    def get_compatible_parsers(self, content_type: ContentType) -> List[str]:
        """Get parsers compatible with given content type."""
        compatible = []

        for name, parser_class in self._parsers.items():
            temp_instance = parser_class()
            if content_type in temp_instance.supported_content_types:
                compatible.append(name)

        return compatible

    def clear_cache(self):
        """Clear cached parser instances."""
        self._instances.clear()
        self.logger.info("Cleared parser instance cache")


# Global parser registry
parser_registry = ParserRegistry()


# Utility functions for working with parsers

def create_parser_config(**kwargs) -> ParserConfig:
    """Create parser configuration with custom settings."""
    return ParserConfig(**kwargs)


def get_parser_for_content(content_type: ContentType, config: Optional[ParserConfig] = None) -> Optional[BaseParser]:
    """Get the best parser for given content type."""
    compatible_parsers = parser_registry.get_compatible_parsers(content_type)

    if not compatible_parsers:
        return None

    # Return the first compatible parser (could implement more sophisticated selection)
    return parser_registry.get_parser(compatible_parsers[0], config)


async def parse_content(
    content: Any,
    content_type: ContentType,
    parser_name: Optional[str] = None,
    config: Optional[ParserConfig] = None,
    **kwargs
) -> ParseResult:
    """Parse content using specified or auto-selected parser."""
    try:
        if parser_name:
            parser = parser_registry.get_parser(parser_name, config)
        else:
            parser = get_parser_for_content(content_type, config)

        if not parser:
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=f"No suitable parser found for content type: {content_type}",
                error_code="NO_PARSER_FOUND"
            )

        return await parser.parse_with_retry(content, content_type, **kwargs)

    except Exception as e:
        logger.error(f"Error parsing content: {e}")
        return ParseResult(
            status=ParseStatus.FAILED,
            error_message=str(e),
            error_code="PARSER_ERROR"
        )


def get_registry_stats() -> Dict[str, Any]:
    """Get parser registry statistics."""
    return {
        'total_parsers': len(parser_registry.list_parsers()),
        'registered_parsers': parser_registry.list_parsers(),
        'cached_instances': len(parser_registry._instances),
        'content_type_coverage': {
            ct.value: len(parser_registry.get_compatible_parsers(ct))
            for ct in ContentType
        }
    }
