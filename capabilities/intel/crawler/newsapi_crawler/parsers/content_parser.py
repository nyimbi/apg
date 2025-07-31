#!/usr/bin/env python3
"""
NewsAPI Content Parser Module
============================

Advanced parsing and content extraction for news articles from NewsAPI.
This module provides functionality to extract structured information from
news article content, including entities, locations, and events.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Some features will be limited.")

try:
    import newspaper
    from newspaper import Article as NewspaperArticle
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    logger.debug("newspaper3k not available. Using basic text extraction.")

# Constants for conflict-related keywords
CONFLICT_KEYWORDS = {
    "general": [
        "conflict", "violence", "war", "battle", "fighting", "clash", "combat",
        "hostility", "confrontation", "insurgency", "rebellion", "uprising"
    ],
    "weapons": [
        "weapon", "firearm", "gun", "rifle", "artillery", "missile", "bomb",
        "explosive", "grenade", "mortar", "tank", "airstrike", "drone"
    ],
    "casualties": [
        "casualty", "death", "killed", "wounded", "injured", "victim",
        "fatality", "massacre", "genocide", "displaced", "refugee"
    ],
    "actors": [
        "military", "army", "soldier", "rebel", "militant", "fighter",
        "terrorist", "militia", "guerrilla", "force", "troop", "insurgent"
    ],
    "peace": [
        "peace", "ceasefire", "truce", "negotiation", "agreement", "treaty",
        "resolution", "settlement", "reconciliation", "mediation", "dialogue"
    ]
}


class ArticleParser:
    """Parser for extracting and enriching article content."""

    def __init__(self, use_nlp: bool = True, use_newspaper: bool = True):
        """
        Initialize the article parser.

        Args:
            use_nlp: Use NLP for content processing if available
            use_newspaper: Use newspaper3k for full text extraction if available
        """
        self.use_nlp = use_nlp and SPACY_AVAILABLE
        self.use_newspaper = use_newspaper and NEWSPAPER_AVAILABLE
        self.nlp = None

        # Initialize NLP if requested and available
        if self.use_nlp:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model: en_core_web_sm")
            except OSError:
                try:
                    # Try the smaller model as fallback
                    self.nlp = spacy.load("en_core_web_md")
                    logger.info("Loaded spaCy model: en_core_web_md")
                except OSError:
                    logger.warning("No spaCy model available. Install with: python -m spacy download en_core_web_sm")
                    self.use_nlp = False

    async def parse_article(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and enrich an article from NewsAPI.

        Args:
            article_data: Article data from NewsAPI

        Returns:
            Enriched article data
        """
        # Create a copy to avoid modifying the original
        article = article_data.copy()

        # Extract full text if possible
        if self.use_newspaper and article.get('url'):
            try:
                full_text = await self.extract_full_text(article['url'])
                if full_text:
                    article['full_text'] = full_text
            except Exception as e:
                logger.error(f"Error extracting full text: {str(e)}")

        # Extract content for processing
        content_text = article.get('full_text') or article.get('content') or article.get('description') or ''

        # Skip processing if no content
        if not content_text:
            return article

        # Extract entities if NLP is available
        if self.use_nlp and self.nlp:
            try:
                entities = self.extract_entities(content_text)
                if entities:
                    article['entities'] = entities

                # Extract locations
                locations = self.extract_locations(content_text)
                if locations:
                    article['locations'] = locations

                # Extract keywords
                keywords = self.extract_keywords(content_text)
                if keywords:
                    article['keywords'] = keywords

                # Calculate sentiment
                sentiment = self.analyze_sentiment(content_text)
                if sentiment is not None:
                    article['sentiment'] = sentiment
            except Exception as e:
                logger.error(f"Error in NLP processing: {str(e)}")

        # Check for conflict indicators
        conflict_indicators = self.detect_conflict_indicators(content_text)
        if conflict_indicators:
            article['conflict_indicators'] = conflict_indicators

        return article

    async def extract_full_text(self, url: str) -> Optional[str]:
        """
        Extract full text from an article URL using newspaper3k.

        Args:
            url: Article URL

        Returns:
            Full article text or None if extraction failed
        """
        if not NEWSPAPER_AVAILABLE:
            return None

        try:
            article = NewspaperArticle(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            logger.error(f"Error extracting full text from {url}: {str(e)}")
            return None

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.

        Args:
            text: Text to process

        Returns:
            List of entities with type and text
        """
        if not self.use_nlp or not self.nlp:
            return []

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entity = {
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            }
            entities.append(entity)

        return entities

    def extract_locations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract location entities from text.

        Args:
            text: Text to process

        Returns:
            List of location entities
        """
        if not self.use_nlp or not self.nlp:
            return []

        doc = self.nlp(text)
        locations = []

        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC", "FAC"]:
                location = {
                    "text": ent.text,
                    "type": ent.label_
                }
                locations.append(location)

        return locations

    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text.

        Args:
            text: Text to process
            max_keywords: Maximum number of keywords to extract

        Returns:
            List of keywords
        """
        if not self.use_nlp or not self.nlp:
            return []

        doc = self.nlp(text)
        keywords = []

        # Get tokens that are proper nouns, nouns, or adjectives
        # and not stopwords or punctuation
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space:
                continue

            if token.pos_ in ["PROPN", "NOUN", "ADJ"]:
                keywords.append(token.lemma_)

        # Remove duplicates and limit to max_keywords
        unique_keywords = list(set(keywords))
        return unique_keywords[:max_keywords]

    def analyze_sentiment(self, text: str) -> Optional[float]:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score between -1.0 (negative) and 1.0 (positive)
        """
        if not self.use_nlp or not self.nlp:
            return None

        try:
            # Simple rule-based sentiment analysis
            doc = self.nlp(text)

            if hasattr(doc, "sentiment"):
                # Some spaCy models have this
                return doc.sentiment
            else:
                # Calculate based on polarity lexicon
                # This is a simplified approach
                positive_words = {"good", "great", "excellent", "positive", "happy", "peaceful", "agreement"}
                negative_words = {"bad", "terrible", "negative", "sad", "conflict", "war", "crisis", "violence", "death"}

                tokens = [token.text.lower() for token in doc]
                positive_count = sum(1 for token in tokens if token in positive_words)
                negative_count = sum(1 for token in tokens if token in negative_words)

                if positive_count == 0 and negative_count == 0:
                    return 0.0

                return (positive_count - negative_count) / (positive_count + negative_count)
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return None

    def detect_conflict_indicators(self, text: str) -> Dict[str, Any]:
        """
        Detect conflict-related indicators in text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with conflict indicators and scores
        """
        text_lower = text.lower()
        results = {}

        # Check for keywords in each category
        for category, keywords in CONFLICT_KEYWORDS.items():
            matches = []
            for keyword in keywords:
                if keyword in text_lower:
                    matches.append(keyword)

            if matches:
                results[category] = matches

        # Calculate overall conflict score if any indicators found
        if results:
            # Simple scoring: percentage of categories with matches
            category_count = len(CONFLICT_KEYWORDS)
            categories_with_matches = len(results)
            conflict_score = categories_with_matches / category_count
            results["conflict_score"] = conflict_score

        return results


class ContentExtractor:
    """Advanced content extraction from article text."""

    def __init__(self):
        """Initialize the content extractor."""
        self.article_parser = ArticleParser()

    async def extract_from_url(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a URL.

        Args:
            url: Article URL

        Returns:
            Extracted content
        """
        if not NEWSPAPER_AVAILABLE:
            return {"error": "newspaper3k not available"}

        try:
            article = NewspaperArticle(url)
            article.download()
            article.parse()

            if NEWSPAPER_AVAILABLE:
                article.nlp()

            content = {
                "url": url,
                "title": article.title,
                "text": article.text,
                "publish_date": article.publish_date.isoformat() if article.publish_date else None,
                "authors": article.authors,
                "top_image": article.top_image,
                "images": list(article.images),
                "language": article.meta_lang
            }

            if hasattr(article, "keywords") and article.keywords:
                content["keywords"] = article.keywords

            if hasattr(article, "summary") and article.summary:
                content["summary"] = article.summary

            return content
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return {"error": str(e), "url": url}

    def extract_article_metadata(self, article_html: str) -> Dict[str, Any]:
        """
        Extract metadata from article HTML.

        Args:
            article_html: HTML content

        Returns:
            Extracted metadata
        """
        if not NEWSPAPER_AVAILABLE:
            return {}

        try:
            from newspaper import fulltext
            from bs4 import BeautifulSoup

            # Extract full text
            text = fulltext(article_html)

            # Parse HTML
            soup = BeautifulSoup(article_html, 'html.parser')

            # Extract metadata
            metadata = {}

            # Title
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.text.strip()

            # Meta tags
            for meta in soup.find_all('meta'):
                name = meta.get('name', meta.get('property', ''))
                content = meta.get('content', '')

                if name and content:
                    # Clean up name (remove og: prefix etc.)
                    clean_name = name.replace('og:', '').replace('twitter:', '')
                    metadata[clean_name] = content

            # Extract publish date
            publish_date = None
            date_patterns = [
                r'datetime="([^"]+)"',
                r'datePublished":"([^"]+)"',
                r'publish-date="([^"]+)"',
                r'published_time" content="([^"]+)"'
            ]

            for pattern in date_patterns:
                match = re.search(pattern, article_html)
                if match:
                    publish_date = match.group(1)
                    break

            if publish_date:
                metadata['publish_date'] = publish_date

            # Add full text
            metadata['text'] = text

            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return {}


class EventDetector:
    """Detector for conflict events in news articles."""

    def __init__(self):
        """Initialize the event detector."""
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("SpaCy model not available. Event detection will be limited.")

    def detect_events(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect conflict events in text.

        Args:
            text: Text to analyze

        Returns:
            List of detected events
        """
        events = []

        # Skip if no NLP
        if not self.nlp:
            return self._rule_based_event_detection(text)

        try:
            doc = self.nlp(text)

            # Find sentences with conflict indicators
            for sent in doc.sents:
                conflict_score = self._calculate_conflict_score(sent.text)

                if conflict_score > 0.3:  # Threshold for conflict relevance
                    # Extract entities involved
                    entities = []
                    locations = []

                    for ent in sent.ents:
                        if ent.label_ in ["GPE", "LOC", "FAC"]:
                            locations.append({"text": ent.text, "type": ent.label_})
                        elif ent.label_ in ["ORG", "PERSON", "NORP"]:
                            entities.append({"text": ent.text, "type": ent.label_})

                    # Create event
                    event = {
                        "text": sent.text,
                        "conflict_score": conflict_score,
                        "entities": entities,
                        "locations": locations,
                    }

                    # Try to determine event type
                    event_type = self._determine_event_type(sent.text)
                    if event_type:
                        event["event_type"] = event_type

                    events.append(event)

            return events
        except Exception as e:
            logger.error(f"Error in event detection: {str(e)}")
            return self._rule_based_event_detection(text)

    def _calculate_conflict_score(self, text: str) -> float:
        """
        Calculate conflict relevance score for a text.

        Args:
            text: Text to analyze

        Returns:
            Conflict score between 0.0 and 1.0
        """
        text_lower = text.lower()

        # Count conflict keywords
        keyword_count = 0
        total_keywords = 0

        for category, keywords in CONFLICT_KEYWORDS.items():
            for keyword in keywords:
                total_keywords += 1
                if keyword in text_lower:
                    keyword_count += 1

        # Basic score based on keyword matches
        if total_keywords == 0:
            return 0.0

        return min(1.0, keyword_count / (total_keywords * 0.2))  # Scale to make it easier to reach high scores

    def _determine_event_type(self, text: str) -> Optional[str]:
        """
        Determine the type of conflict event.

        Args:
            text: Event text

        Returns:
            Event type or None if undetermined
        """
        text_lower = text.lower()

        # Violence indicators
        if any(word in text_lower for word in ["killed", "death", "fatality", "casualties", "died"]):
            return "VIOLENCE_WITH_CASUALTIES"

        if any(word in text_lower for word in ["attack", "strike", "bomb", "explosion", "shooting"]):
            return "ATTACK"

        if any(word in text_lower for word in ["protest", "demonstration", "rally", "riot"]):
            return "PROTEST"

        if any(word in text_lower for word in ["peace", "ceasefire", "truce", "agreement", "treaty"]):
            return "PEACE_PROCESS"

        if any(word in text_lower for word in ["displaced", "refugee", "fled", "evacuation"]):
            return "DISPLACEMENT"

        # Generic conflict if other indicators present
        if any(word in text_lower for word in ["conflict", "war", "fighting", "clash", "battle"]):
            return "ARMED_CONFLICT"

        return None

    def _rule_based_event_detection(self, text: str) -> List[Dict[str, Any]]:
        """
        Perform rule-based event detection when NLP is not available.

        Args:
            text: Text to analyze

        Returns:
            List of detected events
        """
        events = []

        # Split into sentences (simple approach)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        for sentence in sentences:
            conflict_score = self._calculate_conflict_score(sentence)

            if conflict_score > 0.3:
                event = {
                    "text": sentence,
                    "conflict_score": conflict_score
                }

                # Try to determine event type
                event_type = self._determine_event_type(sentence)
                if event_type:
                    event["event_type"] = event_type

                events.append(event)

        return events
