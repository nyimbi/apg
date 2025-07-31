"""
Advanced Content Parser for Generation Crawler
==============================================

Intelligent content parsing and analysis for the gen_crawler package
using multiple extraction methods and AI-powered content analysis.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin
import html

# Content extraction libraries
try:
    from bs4 import BeautifulSoup, Comment
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    BeautifulSoup = None

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    trafilatura = None

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    Article = None

try:
    from readability import Document
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False
    Document = None

logger = logging.getLogger(__name__)

@dataclass
class ParsedSiteContent:
    """Structured representation of parsed content from a site."""
    url: str
    title: str = ""
    content: str = ""
    cleaned_content: str = ""
    summary: str = ""
    authors: List[str] = field(default_factory=list)
    publish_date: Optional[datetime] = None
    language: str = "unknown"
    keywords: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_type: str = "unknown"
    word_count: int = 0
    reading_time_minutes: int = 0
    quality_score: float = 0.0
    extraction_method: str = "unknown"
    parse_timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_article(self) -> bool:
        """Check if content appears to be an article."""
        return (
            self.content_type == "article" or
            (self.word_count > 300 and self.title and len(self.title) > 10)
        )
    
    @property
    def is_high_quality(self) -> bool:
        """Check if content meets high quality criteria."""
        return (
            self.quality_score > 0.7 and
            self.word_count > 200 and
            bool(self.title) and
            len(self.cleaned_content) > 500
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'url': self.url,
            'title': self.title,
            'content': self.content,
            'cleaned_content': self.cleaned_content,
            'summary': self.summary,
            'authors': self.authors,
            'publish_date': self.publish_date.isoformat() if self.publish_date else None,
            'language': self.language,
            'keywords': self.keywords,
            'links': self.links,
            'images': self.images,
            'metadata': self.metadata,
            'content_type': self.content_type,
            'word_count': self.word_count,
            'reading_time_minutes': self.reading_time_minutes,
            'quality_score': self.quality_score,
            'extraction_method': self.extraction_method,
            'parse_timestamp': self.parse_timestamp.isoformat(),
            'is_article': self.is_article,
            'is_high_quality': self.is_high_quality
        }

class ContentAnalyzer:
    """Analyzes and scores content quality and relevance."""
    
    def __init__(self):
        """Initialize the content analyzer."""
        self.article_indicators = [
            'article', 'story', 'news', 'post', 'blog', 'report',
            'analysis', 'opinion', 'editorial', 'feature'
        ]
        
        self.quality_indicators = [
            'byline', 'author', 'date', 'timestamp', 'published',
            'updated', 'category', 'tags', 'share', 'comment'
        ]
        
        self.low_quality_indicators = [
            'advertisement', 'sponsored', 'promo', 'ad-',
            'popup', 'overlay', 'sidebar', 'footer', 'header'
        ]
    
    def analyze_content_type(self, url: str, title: str, content: str, 
                           html_content: str = "") -> str:
        """
        Analyze and classify content type.
        
        Args:
            url: Page URL
            title: Page title
            content: Extracted text content
            html_content: Raw HTML content
            
        Returns:
            Content type classification
        """
        if not content or len(content.strip()) < 50:
            return "insufficient_content"
        
        url_lower = url.lower()
        title_lower = title.lower() if title else ""
        content_lower = content.lower()
        
        # Check for article indicators in URL
        if any(indicator in url_lower for indicator in self.article_indicators):
            return "article"
        
        # Check for article indicators in title
        if any(indicator in title_lower for indicator in self.article_indicators):
            return "article"
        
        # Analyze content structure
        word_count = len(content.split())
        
        # Check for article-like structure
        if word_count > 500:
            # Look for paragraphs and article structure
            paragraph_count = content.count('\n\n') + content.count('. ')
            if paragraph_count > 3:
                return "article"
        
        # Check for specific page types
        if any(pattern in url_lower for pattern in ['category', 'tag', 'archive']):
            return "listing"
        
        if any(pattern in url_lower for pattern in ['about', 'contact', 'privacy']):
            return "page"
        
        if word_count > 200:
            return "content_page"
        elif word_count > 50:
            return "snippet"
        else:
            return "minimal_content"
    
    def calculate_quality_score(self, parsed_content: ParsedSiteContent, 
                              html_content: str = "") -> float:
        """
        Calculate content quality score (0.0 to 1.0).
        
        Args:
            parsed_content: Parsed content object
            html_content: Raw HTML content for additional analysis
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        
        # Title quality (0.2 max)
        if parsed_content.title:
            title_score = min(len(parsed_content.title) / 60, 1.0) * 0.2
            score += title_score
        
        # Content length quality (0.3 max)
        word_count = parsed_content.word_count
        if word_count > 0:
            # Optimal range: 300-2000 words
            if word_count >= 300:
                length_score = min(word_count / 2000, 1.0) * 0.3
            else:
                length_score = (word_count / 300) * 0.3
            score += length_score
        
        # Structure quality (0.2 max)
        if parsed_content.cleaned_content:
            # Check for paragraph structure
            paragraphs = parsed_content.cleaned_content.count('\n\n')
            sentences = parsed_content.cleaned_content.count('. ')
            
            if paragraphs > 0 and sentences > 0:
                structure_score = min((paragraphs + sentences) / 20, 1.0) * 0.2
                score += structure_score
        
        # Metadata quality (0.2 max)
        metadata_score = 0.0
        if parsed_content.authors:
            metadata_score += 0.05
        if parsed_content.publish_date:
            metadata_score += 0.05
        if parsed_content.keywords:
            metadata_score += 0.05
        if len(parsed_content.metadata) > 0:
            metadata_score += 0.05
        
        score += metadata_score
        
        # Language and readability (0.1 max)
        if parsed_content.language != "unknown":
            score += 0.05
        
        # Check for quality indicators in HTML
        if html_content:
            html_lower = html_content.lower()
            quality_count = sum(1 for indicator in self.quality_indicators 
                              if indicator in html_lower)
            low_quality_count = sum(1 for indicator in self.low_quality_indicators 
                                  if indicator in html_lower)
            
            html_quality = (quality_count - low_quality_count) / 10
            score += max(0, min(html_quality, 0.05))
        
        return max(0.0, min(score, 1.0))

class GenContentParser:
    """
    Advanced content parser using multiple extraction methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the content parser.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.analyzer = ContentAnalyzer()
        
        # Available extraction methods
        self.extraction_methods = []
        if TRAFILATURA_AVAILABLE:
            self.extraction_methods.append('trafilatura')
        if NEWSPAPER_AVAILABLE:
            self.extraction_methods.append('newspaper')
        if READABILITY_AVAILABLE:
            self.extraction_methods.append('readability')
        if BEAUTIFULSOUP_AVAILABLE:
            self.extraction_methods.append('beautifulsoup')
        
        logger.info(f"Content parser initialized with methods: {self.extraction_methods}")
    
    def parse_content(self, url: str, html_content: str, 
                     preferred_method: Optional[str] = None) -> ParsedSiteContent:
        """
        Parse content from HTML using the best available method.
        
        Args:
            url: Source URL
            html_content: Raw HTML content
            preferred_method: Preferred extraction method
            
        Returns:
            ParsedSiteContent object
        """
        if not html_content:
            return ParsedSiteContent(url=url, extraction_method="none")
        
        # Try extraction methods in order of preference
        methods_to_try = []
        if preferred_method and preferred_method in self.extraction_methods:
            methods_to_try.append(preferred_method)
        
        # Add other methods
        for method in ['trafilatura', 'newspaper', 'readability', 'beautifulsoup']:
            if method in self.extraction_methods and method != preferred_method:
                methods_to_try.append(method)
        
        best_result = None
        best_score = 0.0
        
        for method in methods_to_try:
            try:
                result = self._extract_with_method(url, html_content, method)
                if result:
                    # Calculate quality score
                    score = self.analyzer.calculate_quality_score(result, html_content)
                    result.quality_score = score
                    
                    if score > best_score:
                        best_result = result
                        best_score = score
                        
            except Exception as e:
                logger.debug(f"Extraction method {method} failed for {url}: {e}")
                continue
        
        if best_result:
            # Final content analysis
            best_result.content_type = self.analyzer.analyze_content_type(
                url, best_result.title, best_result.content, html_content
            )
            best_result.word_count = len(best_result.content.split()) if best_result.content else 0
            best_result.reading_time_minutes = max(1, best_result.word_count // 200)
            
            return best_result
        
        # Fallback to basic parsing
        return self._basic_parse(url, html_content)
    
    def _extract_with_method(self, url: str, html_content: str, method: str) -> Optional[ParsedSiteContent]:
        """Extract content using a specific method."""
        
        if method == 'trafilatura' and TRAFILATURA_AVAILABLE:
            return self._extract_with_trafilatura(url, html_content)
        
        elif method == 'newspaper' and NEWSPAPER_AVAILABLE:
            return self._extract_with_newspaper(url, html_content)
        
        elif method == 'readability' and READABILITY_AVAILABLE:
            return self._extract_with_readability(url, html_content)
        
        elif method == 'beautifulsoup' and BEAUTIFULSOUP_AVAILABLE:
            return self._extract_with_beautifulsoup(url, html_content)
        
        return None
    
    def _extract_with_trafilatura(self, url: str, html_content: str) -> Optional[ParsedSiteContent]:
        """Extract content using Trafilatura."""
        try:
            # Extract main content
            content = trafilatura.extract(html_content, include_comments=False, 
                                        include_tables=True)
            if not content:
                return None
            
            # Extract metadata
            metadata = trafilatura.extract_metadata(html_content)
            
            result = ParsedSiteContent(
                url=url,
                title=metadata.title if metadata and metadata.title else "",
                content=content,
                cleaned_content=content,
                authors=[metadata.author] if metadata and metadata.author else [],
                publish_date=metadata.date if metadata and metadata.date else None,
                language=metadata.language if metadata and metadata.language else "unknown",
                extraction_method="trafilatura"
            )
            
            if metadata:
                result.metadata = {
                    'site_name': metadata.sitename,
                    'description': metadata.description,
                    'categories': metadata.categories,
                    'tags': metadata.tags
                }
                
                if metadata.tags:
                    result.keywords = metadata.tags
            
            return result
            
        except Exception as e:
            logger.debug(f"Trafilatura extraction failed: {e}")
            return None
    
    def _extract_with_newspaper(self, url: str, html_content: str) -> Optional[ParsedSiteContent]:
        """Extract content using Newspaper3k."""
        try:
            article = Article(url)
            article.set_html(html_content)
            article.parse()
            
            result = ParsedSiteContent(
                url=url,
                title=article.title or "",
                content=article.text or "",
                cleaned_content=article.text or "",
                authors=list(article.authors) if article.authors else [],
                publish_date=article.publish_date,
                language="unknown",
                extraction_method="newspaper"
            )
            
            # Extract additional metadata
            if hasattr(article, 'meta_data'):
                result.metadata = dict(article.meta_data)
            
            # Extract keywords if available
            try:
                article.nlp()
                if article.keywords:
                    result.keywords = list(article.keywords)
                if article.summary:
                    result.summary = article.summary
            except:
                pass  # NLP processing is optional
            
            return result
            
        except Exception as e:
            logger.debug(f"Newspaper extraction failed: {e}")
            return None
    
    def _extract_with_readability(self, url: str, html_content: str) -> Optional[ParsedSiteContent]:
        """Extract content using Readability."""
        try:
            doc = Document(html_content)
            title = doc.title()
            content_html = doc.summary()
            
            # Convert HTML to text
            if BEAUTIFULSOUP_AVAILABLE and content_html:
                soup = BeautifulSoup(content_html, 'html.parser')
                content_text = soup.get_text(separator=' ', strip=True)
            else:
                # Basic HTML stripping
                content_text = re.sub(r'<[^>]+>', ' ', content_html)
                content_text = re.sub(r'\s+', ' ', content_text).strip()
            
            if not content_text:
                return None
            
            result = ParsedSiteContent(
                url=url,
                title=title or "",
                content=content_text,
                cleaned_content=content_text,
                extraction_method="readability"
            )
            
            return result
            
        except Exception as e:
            logger.debug(f"Readability extraction failed: {e}")
            return None
    
    def _extract_with_beautifulsoup(self, url: str, html_content: str) -> Optional[ParsedSiteContent]:
        """Extract content using BeautifulSoup."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Remove comments
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                comment.extract()
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else ""
            
            # Try to find main content area
            content_selectors = [
                'article', '[role="main"]', '.content', '.post-content',
                '.entry-content', '.article-content', 'main', '.main'
            ]
            
            content_element = None
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    break
            
            # If no main content found, use body
            if not content_element:
                content_element = soup.find('body')
            
            if not content_element:
                return None
            
            # Extract text content
            content_text = content_element.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            content_text = re.sub(r'\s+', ' ', content_text).strip()
            
            if not content_text:
                return None
            
            result = ParsedSiteContent(
                url=url,
                title=title,
                content=content_text,
                cleaned_content=content_text,
                extraction_method="beautifulsoup"
            )
            
            # Extract meta information
            meta_tags = soup.find_all('meta')
            metadata = {}
            
            for meta in meta_tags:
                name = meta.get('name', '').lower()
                property_name = meta.get('property', '').lower()
                content = meta.get('content', '')
                
                if name == 'author':
                    result.authors = [content]
                elif name == 'description':
                    metadata['description'] = content
                elif name == 'keywords':
                    result.keywords = [k.strip() for k in content.split(',')]
                elif property_name == 'og:title':
                    if not result.title:
                        result.title = content
                elif property_name == 'article:author':
                    result.authors = [content]
                
            result.metadata = metadata
            
            return result
            
        except Exception as e:
            logger.debug(f"BeautifulSoup extraction failed: {e}")
            return None
    
    def _basic_parse(self, url: str, html_content: str) -> ParsedSiteContent:
        """Basic fallback parsing without external libraries."""
        # Extract title using regex
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        title = html.unescape(title_match.group(1).strip()) if title_match else ""
        
        # Basic HTML stripping
        text_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
        text_content = re.sub(r'<style[^>]*>.*?</style>', '', text_content, flags=re.IGNORECASE | re.DOTALL)
        text_content = re.sub(r'<[^>]+>', ' ', text_content)
        text_content = html.unescape(text_content)
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        
        return ParsedSiteContent(
            url=url,
            title=title,
            content=text_content,
            cleaned_content=text_content,
            extraction_method="basic"
        )
    
    def get_parser_status(self) -> Dict[str, Any]:
        """Get status of available parsing methods."""
        return {
            'available_methods': self.extraction_methods,
            'trafilatura_available': TRAFILATURA_AVAILABLE,
            'newspaper_available': NEWSPAPER_AVAILABLE,
            'readability_available': READABILITY_AVAILABLE,
            'beautifulsoup_available': BEAUTIFULSOUP_AVAILABLE,
            'total_methods': len(self.extraction_methods)
        }

def create_content_parser(config: Optional[Dict[str, Any]] = None) -> GenContentParser:
    """Factory function to create a GenContentParser."""
    return GenContentParser(config)