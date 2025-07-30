"""
Metadata Extractor with Utils Integration
==========================================

Metadata extraction from news articles using structured data and utils packages.

Author: Lindela Development Team
Version: 4.0.0
License: MIT
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse

# Import utils packages
try:
    from ....utils.validation import validate_url, validate_email
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    
    def validate_url(url): return True
    def validate_email(email): return True

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    BeautifulSoup = None

logger = logging.getLogger(__name__)


@dataclass
class ArticleMetadata:
    """Extracted article metadata."""
    # Basic metadata
    title: str = ""
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    author: str = ""
    authors: List[str] = field(default_factory=list)
    
    # Publication info
    publication_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    publisher: str = ""
    publication: str = ""
    
    # URLs and links
    canonical_url: str = ""
    image_url: str = ""
    video_urls: List[str] = field(default_factory=list)
    
    # Social media
    twitter_handle: str = ""
    facebook_url: str = ""
    
    # Technical metadata
    language: str = ""
    charset: str = ""
    content_type: str = ""
    
    # Structured data
    structured_data: List[Dict[str, Any]] = field(default_factory=list)
    schema_org: Dict[str, Any] = field(default_factory=dict)
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    
    # Quality indicators
    metadata_completeness: float = 0.0
    extraction_confidence: float = 0.0


class MetadataExtractor:
    """Metadata extractor with utils integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize metadata extractor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Extraction configuration
        self.extract_structured_data = self.config.get('extract_structured_data', True)
        self.extract_social_media = self.config.get('extract_social_media', True)
        self.validate_urls = self.config.get('validate_urls', True)
        
        logger.info("MetadataExtractor initialized")
    
    async def extract(self, html: str, url: str = "") -> ArticleMetadata:
        """
        Extract metadata from HTML.
        
        Args:
            html: HTML content
            url: Source URL for context
            
        Returns:
            ArticleMetadata with extracted information
        """
        if not BS4_AVAILABLE:
            logger.error("BeautifulSoup not available")
            return ArticleMetadata()
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            metadata = ArticleMetadata()
            
            # Extract basic metadata
            self._extract_basic_metadata(soup, metadata)
            
            # Extract meta tags
            self._extract_meta_tags(soup, metadata)
            
            # Extract OpenGraph metadata
            self._extract_opengraph(soup, metadata)
            
            # Extract Twitter Card metadata
            self._extract_twitter_card(soup, metadata)
            
            # Extract structured data
            if self.extract_structured_data:
                self._extract_structured_data(soup, metadata)
            
            # Extract additional metadata
            self._extract_additional_metadata(soup, metadata, url)
            
            # Calculate quality scores
            self._calculate_quality_scores(metadata)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return ArticleMetadata()
    
    def _extract_basic_metadata(self, soup: BeautifulSoup, metadata: ArticleMetadata):
        """Extract basic HTML metadata."""
        try:
            # Title
            title_tag = soup.find('title')
            if title_tag:
                metadata.title = title_tag.get_text(strip=True)
            
            # Language
            html_tag = soup.find('html')
            if html_tag and html_tag.get('lang'):
                metadata.language = html_tag['lang']
            
            # Charset
            charset_meta = soup.find('meta', charset=True)
            if charset_meta:
                metadata.charset = charset_meta['charset']
            
        except Exception as e:
            logger.warning(f"Basic metadata extraction failed: {e}")
    
    def _extract_meta_tags(self, soup: BeautifulSoup, metadata: ArticleMetadata):
        """Extract standard meta tags."""
        try:
            meta_tags = soup.find_all('meta')
            
            for tag in meta_tags:
                name = tag.get('name', '').lower()
                content = tag.get('content', '')
                
                if not content:
                    continue
                
                if name == 'description':
                    metadata.description = content
                elif name == 'keywords':
                    metadata.keywords = [k.strip() for k in content.split(',')]
                elif name == 'author':
                    metadata.author = content
                    if content not in metadata.authors:
                        metadata.authors.append(content)
                elif name in ['date', 'article:published_time', 'datePublished']:
                    metadata.publication_date = self._parse_date(content)
                elif name in ['last-modified', 'article:modified_time', 'dateModified']:
                    metadata.modified_date = self._parse_date(content)
                elif name == 'publisher':
                    metadata.publisher = content
                elif name == 'publication':
                    metadata.publication = content
                
        except Exception as e:
            logger.warning(f"Meta tags extraction failed: {e}")
    
    def _extract_opengraph(self, soup: BeautifulSoup, metadata: ArticleMetadata):
        """Extract OpenGraph metadata."""
        try:
            og_tags = soup.find_all('meta', property=lambda x: x and x.startswith('og:'))
            
            for tag in og_tags:
                property_name = tag['property']
                content = tag.get('content', '')
                
                if not content:
                    continue
                
                if property_name == 'og:title' and not metadata.title:
                    metadata.title = content
                elif property_name == 'og:description' and not metadata.description:
                    metadata.description = content
                elif property_name == 'og:image':
                    metadata.image_url = content
                elif property_name == 'og:url':
                    metadata.canonical_url = content
                elif property_name == 'og:site_name':
                    metadata.publication = content
                elif property_name == 'og:type':
                    if 'article' in content.lower():
                        metadata.content_type = 'article'
                elif property_name == 'og:video':
                    metadata.video_urls.append(content)
                
        except Exception as e:
            logger.warning(f"OpenGraph extraction failed: {e}")
    
    def _extract_twitter_card(self, soup: BeautifulSoup, metadata: ArticleMetadata):
        """Extract Twitter Card metadata."""
        try:
            twitter_tags = soup.find_all('meta', name=lambda x: x and x.startswith('twitter:'))
            
            for tag in twitter_tags:
                name = tag['name']
                content = tag.get('content', '')
                
                if not content:
                    continue
                
                if name == 'twitter:title' and not metadata.title:
                    metadata.title = content
                elif name == 'twitter:description' and not metadata.description:
                    metadata.description = content
                elif name == 'twitter:image':
                    if not metadata.image_url:
                        metadata.image_url = content
                elif name == 'twitter:site':
                    metadata.twitter_handle = content
                elif name == 'twitter:creator':
                    if content not in metadata.authors:
                        metadata.authors.append(content)
                
        except Exception as e:
            logger.warning(f"Twitter Card extraction failed: {e}")
    
    def _extract_structured_data(self, soup: BeautifulSoup, metadata: ArticleMetadata):
        """Extract structured data (JSON-LD, microdata)."""
        try:
            # JSON-LD
            json_ld_scripts = soup.find_all('script', type='application/ld+json')
            
            for script in json_ld_scripts:
                try:
                    data = json.loads(script.string)
                    metadata.structured_data.append(data)
                    
                    # Extract specific fields from structured data
                    if isinstance(data, dict):
                        self._process_structured_data_object(data, metadata)
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                self._process_structured_data_object(item, metadata)
                                
                except json.JSONDecodeError:
                    continue
            
            # Microdata (basic extraction)
            microdata_items = soup.find_all(attrs={'itemtype': True})
            for item in microdata_items:
                itemtype = item.get('itemtype', '')
                if 'Article' in itemtype or 'NewsArticle' in itemtype:
                    self._extract_microdata(item, metadata)
                    
        except Exception as e:
            logger.warning(f"Structured data extraction failed: {e}")
    
    def _process_structured_data_object(self, data: Dict[str, Any], metadata: ArticleMetadata):
        """Process a structured data object."""
        try:
            schema_type = data.get('@type', '').lower()
            
            if 'article' in schema_type or 'newsarticle' in schema_type:
                # Article-specific structured data
                if 'headline' in data and not metadata.title:
                    metadata.title = data['headline']
                
                if 'description' in data and not metadata.description:
                    metadata.description = data['description']
                
                if 'author' in data:
                    authors = data['author']
                    if isinstance(authors, list):
                        for author in authors:
                            if isinstance(author, dict) and 'name' in author:
                                metadata.authors.append(author['name'])
                            elif isinstance(author, str):
                                metadata.authors.append(author)
                    elif isinstance(authors, dict) and 'name' in authors:
                        metadata.authors.append(authors['name'])
                    elif isinstance(authors, str):
                        metadata.authors.append(authors)
                
                if 'datePublished' in data:
                    metadata.publication_date = self._parse_date(data['datePublished'])
                
                if 'dateModified' in data:
                    metadata.modified_date = self._parse_date(data['dateModified'])
                
                if 'publisher' in data:
                    publisher = data['publisher']
                    if isinstance(publisher, dict) and 'name' in publisher:
                        metadata.publisher = publisher['name']
                    elif isinstance(publisher, str):
                        metadata.publisher = publisher
                
                if 'image' in data:
                    image = data['image']
                    if isinstance(image, list) and image:
                        metadata.image_url = image[0] if isinstance(image[0], str) else image[0].get('url', '')
                    elif isinstance(image, dict):
                        metadata.image_url = image.get('url', '')
                    elif isinstance(image, str):
                        metadata.image_url = image
                
                if 'keywords' in data:
                    keywords = data['keywords']
                    if isinstance(keywords, list):
                        metadata.keywords.extend(keywords)
                    elif isinstance(keywords, str):
                        metadata.keywords.extend([k.strip() for k in keywords.split(',')])
            
            # Store complete structured data
            metadata.schema_org.update(data)
            
        except Exception as e:
            logger.warning(f"Structured data processing failed: {e}")
    
    def _extract_microdata(self, item, metadata: ArticleMetadata):
        """Extract microdata from an item."""
        try:
            # Find itemprop elements
            props = item.find_all(attrs={'itemprop': True})
            
            for prop in props:
                prop_name = prop.get('itemprop', '')
                
                if prop_name == 'headline' and not metadata.title:
                    metadata.title = prop.get_text(strip=True)
                elif prop_name == 'description' and not metadata.description:
                    metadata.description = prop.get_text(strip=True)
                elif prop_name == 'author':
                    author_text = prop.get_text(strip=True)
                    if author_text not in metadata.authors:
                        metadata.authors.append(author_text)
                elif prop_name == 'datePublished':
                    date_text = prop.get('datetime') or prop.get_text(strip=True)
                    metadata.publication_date = self._parse_date(date_text)
                
        except Exception as e:
            logger.warning(f"Microdata extraction failed: {e}")
    
    def _extract_additional_metadata(self, soup: BeautifulSoup, metadata: ArticleMetadata, url: str):
        """Extract additional metadata and links."""
        try:
            # Canonical URL
            canonical = soup.find('link', rel='canonical')
            if canonical and canonical.get('href'):
                metadata.canonical_url = canonical['href']
            elif url:
                metadata.canonical_url = url
            
            # Alternative URLs and feeds
            alternates = soup.find_all('link', rel='alternate')
            for alt in alternates:
                href = alt.get('href', '')
                if 'rss' in alt.get('type', '').lower() or 'feed' in href.lower():
                    # Could store feed URLs if needed
                    pass
            
            # Extract from URL structure
            if url:
                parsed_url = urlparse(url)
                if not metadata.publication:
                    # Try to extract publication from domain
                    domain_parts = parsed_url.netloc.split('.')
                    if len(domain_parts) >= 2:
                        metadata.publication = domain_parts[-2].title()
            
            # Look for author information in common locations
            if not metadata.authors:
                author_selectors = [
                    '.author', '.byline', '[rel="author"]', '.article-author',
                    '.post-author', '.writer', '.journalist'
                ]
                
                for selector in author_selectors:
                    author_elem = soup.select_one(selector)
                    if author_elem:
                        author_text = author_elem.get_text(strip=True)
                        if author_text and len(author_text) < 100:  # Reasonable author name length
                            metadata.authors.append(author_text)
                            break
            
        except Exception as e:
            logger.warning(f"Additional metadata extraction failed: {e}")
    
    def _parse_date(self, date_string: str) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_string:
            return None
        
        try:
            # Common date formats
            formats = [
                '%Y-%m-%dT%H:%M:%S%z',  # ISO format with timezone
                '%Y-%m-%dT%H:%M:%S',    # ISO format without timezone
                '%Y-%m-%d %H:%M:%S',    # Standard datetime
                '%Y-%m-%d',             # Date only
                '%d/%m/%Y',             # DD/MM/YYYY
                '%m/%d/%Y',             # MM/DD/YYYY
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_string, fmt)
                except ValueError:
                    continue
            
            # Try parsing with dateutil if available
            try:
                from dateutil.parser import parse
                return parse(date_string)
            except ImportError:
                pass
            
        except Exception as e:
            logger.warning(f"Date parsing failed for '{date_string}': {e}")
        
        return None
    
    def _calculate_quality_scores(self, metadata: ArticleMetadata):
        """Calculate metadata quality scores."""
        try:
            # Completeness score
            total_fields = 15  # Number of important metadata fields
            filled_fields = 0
            
            if metadata.title: filled_fields += 1
            if metadata.description: filled_fields += 1
            if metadata.authors: filled_fields += 1
            if metadata.publication_date: filled_fields += 1
            if metadata.publisher: filled_fields += 1
            if metadata.canonical_url: filled_fields += 1
            if metadata.image_url: filled_fields += 1
            if metadata.keywords: filled_fields += 1
            if metadata.language: filled_fields += 1
            if metadata.content_type: filled_fields += 1
            if metadata.structured_data: filled_fields += 2  # Worth more
            if metadata.schema_org: filled_fields += 2  # Worth more
            if metadata.tags: filled_fields += 1
            
            metadata.metadata_completeness = min(1.0, filled_fields / total_fields)
            
            # Confidence score based on structured data and validation
            confidence_factors = []
            
            # Structured data boosts confidence
            if metadata.structured_data:
                confidence_factors.append(0.4)
            
            # Multiple sources boost confidence
            source_count = len([x for x in [metadata.title, metadata.description, 
                               metadata.authors, metadata.publication_date] if x])
            if source_count >= 3:
                confidence_factors.append(0.3)
            
            # URL validation
            if metadata.canonical_url and VALIDATION_AVAILABLE:
                if validate_url(metadata.canonical_url):
                    confidence_factors.append(0.2)
            
            # Completeness factor
            confidence_factors.append(metadata.metadata_completeness * 0.1)
            
            metadata.extraction_confidence = sum(confidence_factors)
            
        except Exception as e:
            logger.warning(f"Quality score calculation failed: {e}")
    
    async def extract_batch(self, html_list: List[str], urls: List[str] = None) -> List[ArticleMetadata]:
        """Extract metadata from multiple HTML documents."""
        urls = urls or [""] * len(html_list)
        results = []
        
        for i, html in enumerate(html_list):
            url = urls[i] if i < len(urls) else ""
            result = await self.extract(html, url)
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extractor statistics."""
        return {
            'bs4_available': BS4_AVAILABLE,
            'validation_available': VALIDATION_AVAILABLE,
            'extract_structured_data': self.extract_structured_data,
            'extract_social_media': self.extract_social_media,
            'validate_urls': self.validate_urls
        }


# Utility functions
def quick_extract_metadata(html: str, url: str = "", **kwargs) -> ArticleMetadata:
    """Quick metadata extraction function."""
    extractor = MetadataExtractor(kwargs)
    import asyncio
    return asyncio.run(extractor.extract(html, url))