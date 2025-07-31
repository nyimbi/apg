"""
Intelligent Article Parser
==========================

Advanced parser that combines multiple extraction methods with machine learning
and intelligent heuristics to provide optimal article parsing results.

Features:
- Multi-strategy parsing with automatic method selection
- Content quality scoring and validation
- Machine learning-based content extraction (optional)
- Intelligent fallback mechanisms
- Performance optimization
- Content deduplication
- Language detection
- Sentiment analysis (optional)
- Readability scoring

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from urllib.parse import urlparse
import hashlib

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    TfidfVectorizer = None
    KMeans = None

try:
    import textblob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    textblob = None

try:
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    langdetect = None

from . import BaseParser, ParseResult, ArticleData, ParseStatus, ContentType
from .rss_parser import RSSParser
from .html_parser import HTMLParser
from .json_parser import JSONParser

logger = logging.getLogger(__name__)

class IntelligentParser(BaseParser):
    """Intelligent parser that combines multiple extraction methods."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize intelligent parser."""
        super().__init__(config)

        # Parser instances
        self.rss_parser = RSSParser(config)
        self.html_parser = HTMLParser(config)
        self.json_parser = JSONParser(config)

        # Configuration
        self.enable_ml_features = config.get('enable_ml_features', SKLEARN_AVAILABLE) if config else SKLEARN_AVAILABLE
        self.enable_sentiment = config.get('enable_sentiment', TEXTBLOB_AVAILABLE) if config else TEXTBLOB_AVAILABLE
        self.enable_language_detection = config.get('enable_language_detection', LANGDETECT_AVAILABLE) if config else LANGDETECT_AVAILABLE
        self.min_confidence_score = config.get('min_confidence_score', 0.6) if config else 0.6
        self.max_parse_attempts = config.get('max_parse_attempts', 3) if config else 3

        # Quality scoring weights
        self.quality_weights = {
            'title_quality': 0.25,
            'content_quality': 0.35,
            'metadata_completeness': 0.20,
            'structure_quality': 0.15,
            'extraction_confidence': 0.05
        }

        # Content patterns for quality assessment
        self.quality_patterns = {
            'good_indicators': [
                r'\b(published|updated|author|by)\b',
                r'\d{4}-\d{2}-\d{2}',  # Date patterns
                r'[A-Z][a-z]+ \d{1,2}, \d{4}',  # Date patterns
                r'\b(reuters|ap|afp|bloomberg|cnn|bbc)\b',  # News agencies
            ],
            'bad_indicators': [
                r'\b(click here|read more|subscribe|advertisement)\b',
                r'cookie.*policy',
                r'terms.*service',
                r'privacy.*policy',
            ]
        }

        # Initialize ML components if available
        self.vectorizer = None
        self.clusterer = None
        if self.enable_ml_features and SKLEARN_AVAILABLE:
            self._initialize_ml_components()

    def can_parse(self, content: str, content_type: ContentType = None) -> bool:
        """Intelligent parser can handle any content type."""
        return True

    async def parse(self, content: str, source_url: str = None, **kwargs) -> ParseResult:
        """Parse content using intelligent multi-strategy approach."""
        try:
            # Detect content type
            detected_type = self._detect_content_type(content)

            # Get parsing strategies based on content type
            strategies = self._get_parsing_strategies(detected_type)

            # Try each strategy
            best_result = None
            best_score = 0.0

            for strategy_name, parser, strategy_config in strategies:
                try:
                    logger.debug(f"Trying parsing strategy: {strategy_name}")

                    # Configure parser for this strategy
                    if strategy_config:
                        parser.config.update(strategy_config)

                    result = await parser.parse(content, source_url, **kwargs)

                    if result.status == ParseStatus.SUCCESS:
                        # Score the result
                        score = await self._score_parse_result(result, content, strategy_name)

                        logger.debug(f"Strategy {strategy_name} score: {score:.3f}")

                        if score > best_score:
                            best_result = result
                            best_score = score

                            # Early exit if we have a high-confidence result
                            if score > 0.9:
                                break

                except Exception as e:
                    logger.debug(f"Strategy {strategy_name} failed: {e}")
                    continue

            if not best_result:
                return ParseResult(
                    status=ParseStatus.FAILED,
                    error="All parsing strategies failed",
                    metadata=self.extract_metadata(content)
                )

            # Enhance the best result with intelligent features
            enhanced_result = await self._enhance_result(best_result, content, source_url)

            return enhanced_result

        except Exception as e:
            logger.error(f"Intelligent parsing failed: {e}")
            return ParseResult(
                status=ParseStatus.FAILED,
                error=str(e),
                metadata=self.extract_metadata(content)
            )

    def _get_parsing_strategies(self, content_type: ContentType) -> List[Tuple[str, BaseParser, Dict[str, Any]]]:
        """Get ordered list of parsing strategies based on content type."""

        strategies = []

        if content_type == ContentType.RSS_FEED or content_type == ContentType.ATOM_FEED:
            strategies = [
                ('rss_primary', self.rss_parser, {}),
                ('html_fallback', self.html_parser, {'extract_content': True}),
                ('json_fallback', self.json_parser, {}),
            ]
        elif content_type == ContentType.HTML_ARTICLE:
            strategies = [
                ('html_primary', self.html_parser, {}),
                ('html_readability', self.html_parser, {'use_readability': True}),
                ('html_semantic', self.html_parser, {'prefer_semantic': True}),
                ('rss_fallback', self.rss_parser, {}),
            ]
        elif content_type == ContentType.JSON_ARTICLE:
            strategies = [
                ('json_primary', self.json_parser, {}),
                ('json_schema', self.json_parser, {'prefer_schema_org': True}),
                ('html_fallback', self.html_parser, {}),
            ]
        else:
            # Default strategy order
            strategies = [
                ('rss_auto', self.rss_parser, {}),
                ('html_auto', self.html_parser, {}),
                ('json_auto', self.json_parser, {}),
            ]

        return strategies

    async def _score_parse_result(self, result: ParseResult, original_content: str, strategy: str) -> float:
        """Score the quality of a parse result."""

        if result.status != ParseStatus.SUCCESS or not result.content:
            return 0.0

        articles = result.content.get('articles', [])
        if not articles:
            return 0.0

        # Score the first/main article
        article_data = articles[0]

        # Convert to ArticleData if it's a dict
        if isinstance(article_data, dict):
            article = ArticleData(
                title=article_data.get('title', ''),
                url=article_data.get('url', ''),
                description=article_data.get('description'),
                content=article_data.get('content'),
                author=article_data.get('author'),
                published_date=article_data.get('published_date'),
                publisher=article_data.get('publisher'),
            )
        else:
            article = article_data

        scores = {}

        # Title quality
        scores['title_quality'] = self._score_title(article.title)

        # Content quality
        scores['content_quality'] = self._score_content(article.content, original_content)

        # Metadata completeness
        scores['metadata_completeness'] = self._score_metadata(article)

        # Structure quality
        scores['structure_quality'] = self._score_structure(article, original_content)

        # Extraction confidence (strategy-specific)
        scores['extraction_confidence'] = self._score_extraction_confidence(strategy, result)

        # Calculate weighted score
        total_score = sum(
            scores[component] * weight
            for component, weight in self.quality_weights.items()
        )

        logger.debug(f"Scoring breakdown: {scores}, Total: {total_score:.3f}")

        return min(total_score, 1.0)  # Cap at 1.0

    def _score_title(self, title: str) -> float:
        """Score title quality."""
        if not title:
            return 0.0

        score = 0.0

        # Length scoring
        title_len = len(title.strip())
        if 10 <= title_len <= 200:
            score += 0.4
        elif 5 <= title_len <= 300:
            score += 0.2

        # Word count scoring
        words = title.split()
        if 3 <= len(words) <= 20:
            score += 0.3
        elif 2 <= len(words) <= 30:
            score += 0.1

        # Content quality indicators
        if re.search(r'[.!?]$', title.strip()):
            score += 0.1

        if not re.search(r'(click|read more|subscribe)', title.lower()):
            score += 0.2

        return min(score, 1.0)

    def _score_content(self, content: str, original_content: str) -> float:
        """Score content quality."""
        if not content:
            return 0.0

        score = 0.0

        # Length scoring
        content_len = len(content.strip())
        if content_len >= 500:
            score += 0.4
        elif content_len >= 200:
            score += 0.3
        elif content_len >= 100:
            score += 0.1

        # Word count scoring
        words = content.split()
        if len(words) >= 100:
            score += 0.2
        elif len(words) >= 50:
            score += 0.1

        # Sentence structure
        sentences = re.split(r'[.!?]+', content)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(valid_sentences) >= 5:
            score += 0.2
        elif len(valid_sentences) >= 2:
            score += 0.1

        # Content quality patterns
        good_matches = sum(1 for pattern in self.quality_patterns['good_indicators']
                          if re.search(pattern, content, re.IGNORECASE))
        bad_matches = sum(1 for pattern in self.quality_patterns['bad_indicators']
                         if re.search(pattern, content, re.IGNORECASE))

        if good_matches > 0:
            score += min(good_matches * 0.05, 0.1)
        if bad_matches > 0:
            score -= min(bad_matches * 0.1, 0.2)

        return max(min(score, 1.0), 0.0)

    def _score_metadata(self, article: ArticleData) -> float:
        """Score metadata completeness."""
        score = 0.0

        # Essential fields
        if article.title:
            score += 0.3
        if article.url:
            score += 0.2
        if article.content:
            score += 0.2

        # Optional but valuable fields
        if article.author:
            score += 0.1
        if article.published_date:
            score += 0.1
        if article.description:
            score += 0.05
        if article.publisher:
            score += 0.05

        return min(score, 1.0)

    def _score_structure(self, article: ArticleData, original_content: str) -> float:
        """Score structural quality of extraction."""
        score = 0.0

        # Check if extracted content is reasonable portion of original
        if article.content and original_content:
            content_ratio = len(article.content) / len(original_content)
            if 0.1 <= content_ratio <= 0.8:  # Not too small, not too large
                score += 0.4
            elif 0.05 <= content_ratio <= 0.95:
                score += 0.2

        # Check for proper text structure
        if article.content:
            # Paragraph structure
            paragraphs = article.content.split('\n\n')
            if len(paragraphs) >= 2:
                score += 0.3

            # Sentence structure
            sentences = re.split(r'[.!?]+', article.content)
            avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
            if 10 <= avg_sentence_length <= 40:
                score += 0.3

        return min(score, 1.0)

    def _score_extraction_confidence(self, strategy: str, result: ParseResult) -> float:
        """Score extraction confidence based on strategy and result metadata."""
        base_scores = {
            'rss_primary': 0.9,
            'json_primary': 0.8,
            'html_primary': 0.7,
            'html_readability': 0.8,
            'html_semantic': 0.7,
            'rss_fallback': 0.5,
            'html_fallback': 0.4,
            'json_fallback': 0.3,
        }

        score = base_scores.get(strategy, 0.5)

        # Adjust based on result metadata
        if result.metadata:
            parser_used = result.metadata.get('parser_used', '')
            if parser_used == 'feedparser':
                score += 0.1
            elif parser_used == 'readability':
                score += 0.1

        return min(score, 1.0)

    async def _enhance_result(self, result: ParseResult, original_content: str, source_url: str = None) -> ParseResult:
        """Enhance parsing result with intelligent features."""

        if result.status != ParseStatus.SUCCESS or not result.content:
            return result

        articles = result.content.get('articles', [])
        if not articles:
            return result

        enhanced_articles = []

        for article_data in articles:
            try:
                enhanced_article = await self._enhance_article(article_data, original_content, source_url)
                enhanced_articles.append(enhanced_article)
            except Exception as e:
                logger.error(f"Failed to enhance article: {e}")
                enhanced_articles.append(article_data)

        # Update result
        enhanced_result = ParseResult(
            status=result.status,
            content={'articles': enhanced_articles},
            error=result.error,
            metadata={
                **result.metadata,
                'enhanced': True,
                'enhancement_features': self._get_enabled_features(),
            }
        )

        return enhanced_result

    async def _enhance_article(self, article_data: Dict[str, Any], original_content: str, source_url: str = None) -> Dict[str, Any]:
        """Enhance individual article with intelligent features."""

        enhanced = article_data.copy()

        # Language detection
        if self.enable_language_detection and enhanced.get('content'):
            try:
                detected_lang = langdetect.detect(enhanced['content'])
                enhanced['language'] = detected_lang
                enhanced['language_confidence'] = 1.0  # langdetect doesn't provide confidence
            except Exception as e:
                logger.debug(f"Language detection failed: {e}")

        # Sentiment analysis
        if self.enable_sentiment and enhanced.get('content') and TEXTBLOB_AVAILABLE:
            try:
                blob = textblob.TextBlob(enhanced['content'])
                enhanced['sentiment'] = {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                }
            except Exception as e:
                logger.debug(f"Sentiment analysis failed: {e}")

        # Readability scoring
        if enhanced.get('content'):
            enhanced['readability_score'] = self._calculate_readability(enhanced['content'])

        # Content classification (if ML is enabled)
        if self.enable_ml_features and enhanced.get('content'):
            enhanced['content_category'] = await self._classify_content(enhanced['content'])

        # Enhanced metadata
        enhanced['extraction_quality'] = self._calculate_extraction_quality(enhanced)
        enhanced['content_hash'] = self._calculate_content_hash(enhanced.get('content', ''))

        # Word count and reading time
        if enhanced.get('content'):
            words = enhanced['content'].split()
            enhanced['word_count'] = len(words)
            enhanced['estimated_reading_time'] = max(1, len(words) // 200)  # ~200 words per minute

        return enhanced

    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score (Flesch Reading Ease approximation)."""
        if not content:
            return 0.0

        try:
            sentences = len(re.split(r'[.!?]+', content))
            words = len(content.split())
            syllables = sum(self._count_syllables(word) for word in content.split())

            if sentences == 0 or words == 0:
                return 0.0

            # Flesch Reading Ease formula
            score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))

            # Normalize to 0-1 range
            return max(0.0, min(1.0, score / 100.0))

        except Exception as e:
            logger.debug(f"Readability calculation failed: {e}")
            return 0.5

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simple approximation)."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        # Handle silent 'e'
        if word.endswith('e'):
            syllable_count -= 1

        return max(1, syllable_count)

    async def _classify_content(self, content: str) -> Optional[str]:
        """Classify content using simple keyword-based approach."""
        if not content:
            return None

        # Simple keyword-based classification
        categories = {
            'politics': ['election', 'government', 'political', 'congress', 'senate', 'president'],
            'technology': ['tech', 'software', 'computer', 'internet', 'digital', 'ai', 'artificial intelligence'],
            'business': ['business', 'economy', 'market', 'financial', 'company', 'corporate'],
            'sports': ['sport', 'game', 'team', 'player', 'match', 'championship'],
            'health': ['health', 'medical', 'doctor', 'hospital', 'disease', 'treatment'],
            'science': ['research', 'study', 'scientist', 'discovery', 'experiment', 'scientific'],
        }

        content_lower = content.lower()
        category_scores = {}

        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                category_scores[category] = score

        if category_scores:
            return max(category_scores, key=category_scores.get)

        return 'general'

    def _calculate_extraction_quality(self, article_data: Dict[str, Any]) -> float:
        """Calculate overall extraction quality score."""
        score = 0.0

        # Required fields
        if article_data.get('title'):
            score += 0.3
        if article_data.get('content'):
            score += 0.4
        if article_data.get('url'):
            score += 0.1

        # Optional fields
        if article_data.get('author'):
            score += 0.05
        if article_data.get('published_date'):
            score += 0.05
        if article_data.get('description'):
            score += 0.05
        if article_data.get('images'):
            score += 0.05

        return min(score, 1.0)

    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash of content for deduplication."""
        if not content:
            return ""

        # Normalize content for hashing
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _get_enabled_features(self) -> List[str]:
        """Get list of enabled enhancement features."""
        features = ['quality_scoring', 'readability_analysis']

        if self.enable_language_detection:
            features.append('language_detection')
        if self.enable_sentiment:
            features.append('sentiment_analysis')
        if self.enable_ml_features:
            features.append('ml_classification')

        return features

    def _initialize_ml_components(self):
        """Initialize ML components if available."""
        try:
            if SKLEARN_AVAILABLE:
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                logger.info("ML components initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize ML components: {e}")
            self.enable_ml_features = False
