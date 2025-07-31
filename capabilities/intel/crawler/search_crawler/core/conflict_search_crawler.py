"""
Conflict-Focused Search Crawler
================================

Specialized search crawler for conflict monitoring, intelligence gathering,
and geopolitical analysis with advanced filtering and scoring.

Features:
- Conflict-specific keyword optimization
- Location-based filtering and scoring
- Temporal relevance analysis
- Source credibility assessment
- Entity extraction and relationship mapping
- Sentiment analysis for conflict escalation
- Multi-language support for regional conflicts
- Real-time alert generation

Author: Lindela Development Team
Version: 1.0.0
License: MIT
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# Import base search crawler
from .search_crawler import SearchCrawler, SearchCrawlerConfig, EnhancedSearchResult

# Import keyword systems
from ..keywords.conflict_keywords import ConflictKeywordManager
from ..keywords.horn_of_africa_keywords import HornOfAfricaKeywords
from ..keywords.keyword_analyzer import KeywordAnalyzer

# Import stealth news crawler
try:
    from ...news_crawler.core.enhanced_news_crawler import NewsCrawler, create_stealth_crawler
    from ...news_crawler.bypass.bypass_manager import BypassManager, create_stealth_bypass_config
    STEALTH_CRAWLER_AVAILABLE = True
except ImportError:
    STEALTH_CRAWLER_AVAILABLE = False
    NewsCrawler = None
    BypassManager = None

# Import NLP libraries for analysis
try:
    import spacy
    from textblob import TextBlob
    try:
        nlp = spacy.load("en_core_web_sm")
        NLP_AVAILABLE = True
    except OSError:
        # Model not found, try without spacy
        print("Warning: spaCy model 'en_core_web_sm' not found. NLP features disabled.")
        nlp = None
        NLP_AVAILABLE = False
except ImportError:
    NLP_AVAILABLE = False
    nlp = None
    TextBlob = None

logger = logging.getLogger(__name__)


@dataclass
class ConflictSearchConfig(SearchCrawlerConfig):
    """Configuration for conflict-focused search crawler."""
    # Conflict-specific parameters
    conflict_regions: List[str] = field(default_factory=lambda: ['horn_of_africa', 'middle_east', 'ukraine'])
    monitor_keywords: List[str] = field(default_factory=list)
    
    # Specific target locations with hierarchical search
    target_locations: List[str] = field(default_factory=lambda: ['Aweil', 'Karamoja', 'Mandera', 'Assosa'])
    min_articles_per_location: int = 50
    enable_hierarchical_search: bool = True
    
    # Stealth integration
    enable_stealth_download: bool = True
    stealth_aggressive: bool = False
    
    # Scoring weights
    location_weight: float = 0.3
    temporal_weight: float = 0.25
    keyword_weight: float = 0.25
    source_weight: float = 0.2
    
    # Filtering
    min_relevance_score: float = 0.6
    max_age_days: int = 7
    trusted_sources: List[str] = field(default_factory=lambda: [
        'reuters.com', 'apnews.com', 'bbc.com', 'aljazeera.com',
        'france24.com', 'dw.com', 'africanews.com', 'bloomberg.com'
    ])
    
    # Alert thresholds
    enable_alerts: bool = True
    escalation_threshold: float = 0.8
    critical_keywords: List[str] = field(default_factory=lambda: [
        'casualties', 'killed', 'explosion', 'attack', 'invasion',
        'emergency', 'crisis', 'escalation', 'violence'
    ])
    
    # Analysis options
    extract_entities: bool = True
    analyze_sentiment: bool = True
    detect_locations: bool = True
    track_developments: bool = True


@dataclass
class ConflictSearchResult(EnhancedSearchResult):
    """Enhanced search result with conflict-specific analysis."""
    # Conflict relevance
    conflict_score: float = 0.0
    is_conflict_related: bool = False
    conflict_type: Optional[str] = None
    
    # Location data
    locations_mentioned: List[Dict[str, Any]] = field(default_factory=list)
    primary_location: Optional[Dict[str, Any]] = None
    distance_from_region: Optional[float] = None
    
    # Temporal analysis
    event_date: Optional[datetime] = None
    is_breaking: bool = False
    temporal_relevance: float = 0.0
    
    # Entity extraction
    entities: Dict[str, List[str]] = field(default_factory=dict)
    key_actors: List[str] = field(default_factory=list)
    
    # Sentiment analysis
    sentiment_score: float = 0.0
    sentiment_label: str = 'neutral'
    escalation_indicators: List[str] = field(default_factory=list)
    
    # Source credibility
    source_credibility: float = 0.0
    is_trusted_source: bool = False
    
    # Alert status
    requires_alert: bool = False
    alert_reasons: List[str] = field(default_factory=list)


class ConflictSearchCrawler(SearchCrawler):
    """Specialized search crawler for conflict monitoring and analysis."""
    
    def __init__(self, config: Optional[ConflictSearchConfig] = None):
        """Initialize conflict search crawler."""
        self.conflict_config = config or ConflictSearchConfig()
        super().__init__(self.conflict_config)
        
        # Initialize conflict keyword systems
        self.keyword_systems = {
            'general': ConflictKeywordManager(),
            'horn_of_africa': HornOfAfricaKeywords()
        }
        self.keyword_analyzer = KeywordAnalyzer()
        
        # Initialize stealth news crawler for content downloading
        self.stealth_crawler = None
        if STEALTH_CRAWLER_AVAILABLE and self.conflict_config.enable_stealth_download:
            self.stealth_crawler = create_stealth_crawler(
                aggressive=self.conflict_config.stealth_aggressive
            )
            
        # Location hierarchy mapping for targeted locations
        self.location_hierarchy = self._initialize_location_hierarchy()
        
        # Initialize geocoder for location analysis
        self.geolocator = Nominatim(user_agent="conflict_search_crawler")
        self.location_cache: Dict[str, Any] = {}
        
        # Region centers for distance calculation
        self.region_centers = {
            'horn_of_africa': (8.0, 48.0),  # Approximate center
            'middle_east': (29.0, 42.0),
            'ukraine': (48.5, 31.0),
            'sahel': (15.0, 10.0),
            'myanmar': (22.0, 96.0)
        }
        
        # Conflict type patterns
        self.conflict_patterns = {
            'military': r'(militar|armed|soldier|troop|force|army|navy|air force)',
            'terrorism': r'(terror|extremis|militant|insurgent|rebel|guerrilla)',
            'civil_unrest': r'(protest|demonstrat|riot|unrest|uprising|revolt)',
            'political': r'(coup|election|government|regime|political|power)',
            'ethnic': r'(ethnic|tribal|clan|sectarian|religious|communal)',
            'border': r'(border|territorial|dispute|incursion|cross-border)'
        }
        
        # Development tracking
        self.tracked_events: Dict[str, List[ConflictSearchResult]] = defaultdict(list)
        
        # Alert queue
        self.alert_queue: List[ConflictSearchResult] = []
    
    def _initialize_location_hierarchy(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize hierarchical location mapping for target locations."""
        return {
            'Aweil': {
                'country': ['South Sudan'],
                'state_province': ['Northern Bahr el Ghazal', 'Northern Bahr el Ghazal State'],
                'region_district': ['Aweil North', 'Aweil South', 'Aweil West', 'Aweil East', 'Aweil Center'],
                'related_areas': ['Wau', 'Bentiu', 'Bahr el Ghazal', 'Western Bahr el Ghazal', 'Warrap'],
                'conflict_keywords': ['cattle raiding', 'ethnic violence', 'displacement', 'Dinka', 'Arab tribes']
            },
            'Karamoja': {
                'country': ['Uganda'],
                'state_province': ['Karamoja Region', 'Northeastern Uganda'],
                'region_district': ['Kotido', 'Kaabong', 'Abim', 'Moroto', 'Nakapiripirit', 'Amudat', 'Napak', 'Nabilatuk', 'Karenga'],
                'related_areas': ['Turkana', 'Pokot', 'Teso', 'Acholi', 'Northern Uganda'],
                'conflict_keywords': ['cattle rustling', 'pastoral conflict', 'Karimojong', 'disarmament', 'drought conflict']
            },
            'Mandera': {
                'country': ['Kenya'],
                'state_province': ['North Eastern Province', 'Northeastern Kenya'],
                'region_district': ['Mandera County', 'Mandera North', 'Mandera South', 'Mandera West', 'Mandera East', 'Banisa', 'Lafey'],
                'related_areas': ['Wajir', 'Garissa', 'Somali Region', 'Gedo', 'Lower Juba', 'Ethiopian border', 'Somali border'],
                'conflict_keywords': ['Al-Shabaab', 'cross-border attacks', 'pastoralist conflict', 'clan violence', 'border security']
            },
            'Assosa': {
                'country': ['Ethiopia'],
                'state_province': ['Benishangul-Gumuz', 'Benishangul-Gumuz Region'],
                'region_district': ['Assosa Zone', 'Bambasi', 'Kurmuk', 'Sherkole', 'Homosha', 'Oda Buldigilu'],
                'related_areas': ['Blue Nile', 'Sudan border', 'Metekel', 'Kamashi', 'Grand Ethiopian Renaissance Dam'],
                'conflict_keywords': ['ethnic violence', 'displacement', 'Gumuz', 'Oromo', 'land disputes', 'GERD']
            }
        }
    
    async def search_conflicts(
        self,
        region: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        max_results: int = 50,
        time_range: Optional[str] = 'week',
        **kwargs
    ) -> List[ConflictSearchResult]:
        """
        Search for conflict-related content with specialized analysis.
        
        Args:
            region: Specific region to focus on
            keywords: Additional keywords to include
            max_results: Maximum number of results
            time_range: Time filter (day, week, month)
            **kwargs: Additional search parameters
            
        Returns:
            List of conflict-analyzed search results
        """
        # Build conflict-optimized query
        query = self._build_conflict_query(region, keywords)
        
        # Add time filter
        if time_range:
            kwargs['time_filter'] = time_range
        
        # Perform search with base crawler
        base_results = await self.search(
            query=query,
            max_results=max_results * 2,  # Get extra for filtering
            download_content=True,
            **kwargs
        )
        
        # Convert and analyze results
        conflict_results = []
        for result in base_results:
            conflict_result = await self._analyze_conflict_result(result, region)
            
            # Filter by relevance
            if conflict_result.conflict_score >= self.conflict_config.min_relevance_score:
                conflict_results.append(conflict_result)
        
        # Sort by conflict score
        conflict_results.sort(key=lambda x: x.conflict_score, reverse=True)
        
        # Limit results
        final_results = conflict_results[:max_results]
        
        # Check for alerts
        if self.conflict_config.enable_alerts:
            self._check_alerts(final_results)
        
        # Track developments
        if self.conflict_config.track_developments:
            self._track_developments(final_results)
        
        return final_results
    
    async def search_target_locations(
        self,
        target_location: Optional[str] = None,
        max_results: int = None,
        time_range: str = 'week',
        **kwargs
    ) -> List[ConflictSearchResult]:
        """
        Search for conflicts in target locations using hierarchical approach.
        
        Args:
            target_location: Specific target location (Aweil, Karamoja, Mandera, Assosa)
            max_results: Maximum results per location
            time_range: Time filter
            **kwargs: Additional search parameters
            
        Returns:
            List of conflict search results
        """
        if max_results is None:
            max_results = self.conflict_config.min_articles_per_location
            
        target_locations = [target_location] if target_location else self.conflict_config.target_locations
        all_results = []
        
        for location in target_locations:
            self.logger.info(f"Starting hierarchical search for {location}")
            
            # Step 1: Search specific location
            location_results = await self._search_location_hierarchical(
                location, max_results, time_range, **kwargs
            )
            
            self.logger.info(f"Found {len(location_results)} results for {location}")
            all_results.extend(location_results)
        
        # Sort all results by conflict score
        all_results.sort(key=lambda x: x.conflict_score, reverse=True)
        
        return all_results
    
    async def _search_location_hierarchical(
        self,
        location: str,
        max_results: int,
        time_range: str,
        **kwargs
    ) -> List[ConflictSearchResult]:
        """
        Perform hierarchical search for a specific location.
        Priority: specific -> district/regional -> country level
        """
        if location not in self.location_hierarchy:
            self.logger.warning(f"Location {location} not in hierarchy, falling back to regular search")
            return await self.search_conflicts(region=location, max_results=max_results, time_range=time_range, **kwargs)
        
        hierarchy = self.location_hierarchy[location]
        all_results = []
        results_per_level = max_results // 3  # Distribute across levels
        
        # Level 1: Specific location + conflict keywords
        self.logger.info(f"Level 1: Searching specific location '{location}'")
        specific_results = await self._search_level(
            [location] + hierarchy.get('conflict_keywords', []),
            results_per_level,
            time_range,
            f"specific_{location}",
            **kwargs
        )
        all_results.extend(specific_results)
        
        # Level 2: District/Regional level if not enough results
        if len(specific_results) < results_per_level:
            self.logger.info(f"Level 2: Searching district/regional level for {location}")
            needed = results_per_level - len(specific_results)
            district_results = await self._search_level(
                hierarchy.get('region_district', []) + hierarchy.get('related_areas', []),
                needed + results_per_level,
                time_range,
                f"district_{location}",
                **kwargs
            )
            all_results.extend(district_results)
        
        # Level 3: Country level if still not enough results
        if len(all_results) < (results_per_level * 2):
            self.logger.info(f"Level 3: Searching country level for {location}")
            needed = max_results - len(all_results)
            country_results = await self._search_level(
                hierarchy.get('country', []) + hierarchy.get('state_province', []),
                needed,
                time_range,
                f"country_{location}",
                **kwargs
            )
            all_results.extend(country_results)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results[:max_results]
    
    async def _search_level(
        self,
        location_terms: List[str],
        max_results: int,
        time_range: str,
        level_name: str,
        **kwargs
    ) -> List[ConflictSearchResult]:
        """Search at a specific hierarchical level."""
        level_results = []
        
        # Get conflict keywords for this search
        conflict_keywords = self.keyword_systems['general'].get_high_priority_keywords()[:10]
        
        # Search with each search engine
        available_engines = list(self.engines.keys())
        results_per_engine = max(1, max_results // len(available_engines))
        
        for engine in available_engines:
            for location_term in location_terms[:3]:  # Limit to top 3 location terms
                for conflict_kw in conflict_keywords[:5]:  # Top 5 conflict keywords
                    query = f'"{location_term}" {conflict_kw}'
                    
                    try:
                        # Search with single engine
                        engine_results = await self.search_conflicts(
                            region=None,
                            keywords=[location_term, conflict_kw],
                            max_results=results_per_engine,
                            engines=[engine],
                            time_range=time_range,
                            **kwargs
                        )
                        
                        level_results.extend(engine_results)
                        
                        if len(level_results) >= max_results:
                            break
                            
                    except Exception as e:
                        self.logger.error(f"Search failed for {query} on {engine}: {e}")
                        continue
                
                if len(level_results) >= max_results:
                    break
            
            if len(level_results) >= max_results:
                break
        
        self.logger.info(f"{level_name}: Found {len(level_results)} results")
        return level_results[:max_results]
    
    def _build_conflict_query(self, region: Optional[str], keywords: Optional[List[str]]) -> str:
        """Build optimized search query for conflict monitoring."""
        query_parts = []
        
        # Add region-specific keywords
        if region:
            if region == 'horn_of_africa' and region in self.keyword_systems:
                region_keywords = self.keyword_systems[region].get_conflict_keywords()[:5]
                query_parts.extend(region_keywords)
            else:
                query_parts.append(region)
        
        # Add general conflict keywords
        general_keywords = self.keyword_systems['general'].get_high_priority_keywords()[:3]
        query_parts.extend(general_keywords)
        
        # Add custom keywords
        if keywords:
            query_parts.extend(keywords)
        
        # Add monitor keywords from config
        if self.conflict_config.monitor_keywords:
            query_parts.extend(self.conflict_config.monitor_keywords)
        
        # Build query with OR operators
        query = ' OR '.join([f'"{kw}"' if ' ' in kw else kw for kw in query_parts])
        
        # Add news focus
        query += ' (news OR report OR update OR latest)'
        
        return query
    
    async def _analyze_conflict_result(
        self,
        result: EnhancedSearchResult,
        region: Optional[str]
    ) -> ConflictSearchResult:
        """Analyze search result for conflict relevance and extract information."""
        # Convert to ConflictSearchResult
        conflict_result = ConflictSearchResult(
            title=result.title,
            url=result.url,
            snippet=result.snippet,
            engine=result.engine,
            rank=result.rank,
            timestamp=result.timestamp,
            metadata=result.metadata,
            engines_found=result.engines_found,
            engine_ranks=result.engine_ranks,
            content=result.content,
            parsed_content=result.parsed_content,
            relevance_score=result.relevance_score
        )
        
        # Download full content with stealth if enabled and available
        enhanced_content = result.content
        if self.stealth_crawler and self.conflict_config.enable_stealth_download:
            try:
                self.logger.debug(f"Downloading content with stealth for: {result.url}")
                article = await self.stealth_crawler.crawl_url(result.url)
                if article and article.content:
                    enhanced_content = article.content
                    # Update result with enhanced content
                    conflict_result.content = enhanced_content
                    if hasattr(conflict_result, 'parsed_content'):
                        conflict_result.parsed_content = {
                            'title': article.title,
                            'content': article.content,
                            'authors': article.authors,
                            'publish_date': article.publish_date,
                            'quality_score': article.quality_score
                        }
            except Exception as e:
                self.logger.warning(f"Stealth download failed for {result.url}: {e}")
        
        # Combine all text for analysis
        full_text = f"{result.title} {result.snippet}"
        if enhanced_content:
            full_text += f" {enhanced_content[:2000]}"  # Limit content length
        
        # Analyze conflict relevance
        conflict_analysis = self._analyze_conflict_relevance(full_text, region)
        conflict_result.conflict_score = conflict_analysis['score']
        conflict_result.is_conflict_related = conflict_analysis['is_conflict']
        conflict_result.conflict_type = conflict_analysis['type']
        
        # Extract locations
        if self.conflict_config.detect_locations:
            locations = self._extract_locations(full_text)
            conflict_result.locations_mentioned = locations
            if locations:
                conflict_result.primary_location = locations[0]
                if region:
                    conflict_result.distance_from_region = self._calculate_distance_from_region(
                        locations[0], region
                    )
        
        # Temporal analysis
        temporal_analysis = self._analyze_temporal_relevance(result, full_text)
        conflict_result.event_date = temporal_analysis['event_date']
        conflict_result.is_breaking = temporal_analysis['is_breaking']
        conflict_result.temporal_relevance = temporal_analysis['relevance']
        
        # Entity extraction
        if self.conflict_config.extract_entities and NLP_AVAILABLE:
            entities = self._extract_entities(full_text)
            conflict_result.entities = entities
            conflict_result.key_actors = self._identify_key_actors(entities)
        
        # Sentiment analysis
        if self.conflict_config.analyze_sentiment and NLP_AVAILABLE:
            sentiment = self._analyze_sentiment(full_text)
            conflict_result.sentiment_score = sentiment['score']
            conflict_result.sentiment_label = sentiment['label']
            conflict_result.escalation_indicators = self._detect_escalation_indicators(full_text)
        
        # Source credibility
        source_analysis = self._analyze_source_credibility(result.url)
        conflict_result.source_credibility = source_analysis['credibility']
        conflict_result.is_trusted_source = source_analysis['is_trusted']
        
        # Calculate final conflict score
        conflict_result.conflict_score = self._calculate_final_score(conflict_result)
        
        # Check if alert is needed
        if conflict_result.conflict_score >= self.conflict_config.escalation_threshold:
            conflict_result.requires_alert = True
            conflict_result.alert_reasons.append('High conflict score')
        
        if conflict_result.escalation_indicators:
            conflict_result.requires_alert = True
            conflict_result.alert_reasons.append('Escalation indicators detected')
        
        return conflict_result
    
    def _analyze_conflict_relevance(self, text: str, region: Optional[str]) -> Dict[str, Any]:
        """Analyze text for conflict relevance."""
        text_lower = text.lower()
        
        # Check for conflict keywords
        keyword_matches = 0
        matched_keywords = []
        
        # Check general conflict keywords
        general_keywords = self.keyword_systems['general'].get_high_priority_keywords()
        for keyword in general_keywords:
            if keyword.lower() in text_lower:
                keyword_matches += 1
                matched_keywords.append(keyword)
        
        # Check region-specific keywords
        if region == 'horn_of_africa' and region in self.keyword_systems:
            region_keywords = self.keyword_systems[region].get_conflict_keywords()
            for keyword in region_keywords:
                if keyword.lower() in text_lower:
                    keyword_matches += 2  # Double weight for region-specific
                    matched_keywords.append(keyword)
        
        # Determine conflict type
        conflict_type = None
        max_matches = 0
        
        for c_type, pattern in self.conflict_patterns.items():
            matches = len(re.findall(pattern, text_lower))
            if matches > max_matches:
                max_matches = matches
                conflict_type = c_type
        
        # Calculate score
        score = min(1.0, keyword_matches / 10.0)  # Normalize to 0-1
        
        # Boost score for critical keywords
        for critical in self.conflict_config.critical_keywords:
            if critical in text_lower:
                score = min(1.0, score + 0.1)
        
        return {
            'score': score,
            'is_conflict': score > 0.3,
            'type': conflict_type,
            'matched_keywords': matched_keywords[:10]  # Top 10
        }
    
    def _extract_locations(self, text: str) -> List[Dict[str, Any]]:
        """Extract and geocode locations from text."""
        locations = []
        
        if not NLP_AVAILABLE:
            return locations
        
        try:
            # Use spaCy for location extraction
            doc = nlp(text)
            location_entities = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
            
            # Deduplicate and geocode
            seen = set()
            for loc_name in location_entities:
                if loc_name.lower() not in seen:
                    seen.add(loc_name.lower())
                    
                    # Check cache first
                    if loc_name in self.location_cache:
                        locations.append(self.location_cache[loc_name])
                    else:
                        # Geocode location
                        try:
                            location = self.geolocator.geocode(loc_name)
                            if location:
                                loc_data = {
                                    'name': loc_name,
                                    'latitude': location.latitude,
                                    'longitude': location.longitude,
                                    'address': location.address
                                }
                                self.location_cache[loc_name] = loc_data
                                locations.append(loc_data)
                        except Exception as e:
                            self.logger.debug(f"Geocoding failed for {loc_name}: {e}")
        
        except Exception as e:
            self.logger.error(f"Location extraction failed: {e}")
        
        return locations
    
    def _calculate_distance_from_region(self, location: Dict[str, Any], region: str) -> float:
        """Calculate distance from location to region center."""
        if region not in self.region_centers:
            return 0.0
        
        try:
            loc_coords = (location['latitude'], location['longitude'])
            region_coords = self.region_centers[region]
            distance = geodesic(loc_coords, region_coords).kilometers
            return distance
        except Exception:
            return 0.0
    
    def _analyze_temporal_relevance(self, result: EnhancedSearchResult, text: str) -> Dict[str, Any]:
        """Analyze temporal relevance of the result."""
        # Check for breaking news indicators
        breaking_patterns = [
            r'breaking\s*:?', r'just\s+in\s*:?', r'urgent\s*:?',
            r'developing\s*:?', r'update\s*:?', r'latest\s*:?'
        ]
        
        is_breaking = any(re.search(pattern, text.lower()) for pattern in breaking_patterns)
        
        # Extract date from metadata or text
        event_date = result.timestamp
        
        # Look for date patterns in text
        date_patterns = [
            r'(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)',
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}/\d{1,2}/\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text.lower())
            if match:
                # Simple date extraction (would need proper parsing in production)
                break
        
        # Calculate temporal relevance
        age = datetime.now() - event_date
        if age.days == 0:
            relevance = 1.0
        elif age.days <= 1:
            relevance = 0.9
        elif age.days <= 3:
            relevance = 0.7
        elif age.days <= 7:
            relevance = 0.5
        else:
            relevance = max(0.1, 1.0 - (age.days / 30))
        
        return {
            'event_date': event_date,
            'is_breaking': is_breaking,
            'relevance': relevance,
            'age_days': age.days
        }
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text."""
        entities = defaultdict(list)
        
        if not NLP_AVAILABLE:
            return dict(entities)
        
        try:
            doc = nlp(text[:5000])  # Limit text length
            
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'EVENT']:
                    entities[ent.label_].append(ent.text)
            
            # Deduplicate
            for label in entities:
                entities[label] = list(set(entities[label]))
        
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
        
        return dict(entities)
    
    def _identify_key_actors(self, entities: Dict[str, List[str]]) -> List[str]:
        """Identify key actors from extracted entities."""
        key_actors = []
        
        # Priority: Organizations and prominent persons
        if 'ORG' in entities:
            key_actors.extend(entities['ORG'][:5])
        
        if 'PERSON' in entities:
            # Filter for likely important persons (would need better heuristics)
            persons = [p for p in entities['PERSON'] if len(p.split()) >= 2]
            key_actors.extend(persons[:3])
        
        return key_actors
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of the text."""
        if not TextBlob:
            return {'score': 0.0, 'label': 'neutral'}
        
        try:
            blob = TextBlob(text[:1000])  # Limit text length
            polarity = blob.sentiment.polarity
            
            # Determine label
            if polarity < -0.3:
                label = 'very_negative'
            elif polarity < -0.1:
                label = 'negative'
            elif polarity > 0.3:
                label = 'positive'
            elif polarity > 0.1:
                label = 'slightly_positive'
            else:
                label = 'neutral'
            
            # For conflict monitoring, negative sentiment might indicate escalation
            return {
                'score': polarity,
                'label': label,
                'subjectivity': blob.sentiment.subjectivity
            }
        
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return {'score': 0.0, 'label': 'neutral'}
    
    def _detect_escalation_indicators(self, text: str) -> List[str]:
        """Detect indicators of conflict escalation."""
        indicators = []
        text_lower = text.lower()
        
        escalation_patterns = {
            'casualties': r'(\d+\s+)?(killed|dead|died|casualties|injured|wounded)',
            'violence': r'(violence|violent|attack|assault|strike|bombing|explosion)',
            'military_action': r'(deploy|mobiliz|reinforce|advance|retreat|offensive)',
            'emergency': r'(emergency|crisis|critical|severe|catastroph|disaster)',
            'displacement': r'(flee|fled|refugee|displaced|evacuat|exodus)'
        }
        
        for indicator, pattern in escalation_patterns.items():
            if re.search(pattern, text_lower):
                indicators.append(indicator)
        
        return indicators
    
    def _analyze_source_credibility(self, url: str) -> Dict[str, Any]:
        """Analyze source credibility based on domain."""
        from urllib.parse import urlparse
        
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace('www.', '')
        
        # Check if trusted source
        is_trusted = any(trusted in domain for trusted in self.conflict_config.trusted_sources)
        
        # Calculate credibility score
        if is_trusted:
            credibility = 0.9
        elif any(indicator in domain for indicator in ['news', 'reuters', 'ap', 'afp']):
            credibility = 0.7
        elif any(indicator in domain for indicator in ['blog', 'twitter', 'facebook']):
            credibility = 0.3
        else:
            credibility = 0.5
        
        return {
            'domain': domain,
            'is_trusted': is_trusted,
            'credibility': credibility
        }
    
    def _calculate_final_score(self, result: ConflictSearchResult) -> float:
        """Calculate final conflict relevance score."""
        weights = self.conflict_config
        
        # Component scores
        location_score = 0.0
        if result.primary_location and result.distance_from_region is not None:
            # Closer is better (inverse distance)
            location_score = max(0, 1.0 - (result.distance_from_region / 1000))
        
        # Weighted combination
        final_score = (
            result.conflict_score * weights.keyword_weight +
            result.temporal_relevance * weights.temporal_weight +
            location_score * weights.location_weight +
            result.source_credibility * weights.source_weight
        )
        
        # Boost for multiple engines
        if len(result.engines_found) > 1:
            final_score *= 1.1
        
        # Boost for escalation indicators
        if result.escalation_indicators:
            final_score *= 1.2
        
        return min(1.0, final_score)
    
    def _check_alerts(self, results: List[ConflictSearchResult]):
        """Check results for alert conditions."""
        for result in results:
            if result.requires_alert:
                self.alert_queue.append(result)
                self.logger.warning(
                    f"ALERT: {result.title} - Score: {result.conflict_score:.2f} - "
                    f"Reasons: {', '.join(result.alert_reasons)}"
                )
    
    def _track_developments(self, results: List[ConflictSearchResult]):
        """Track conflict developments over time."""
        for result in results:
            if result.conflict_type:
                # Group by conflict type and location
                if result.primary_location:
                    key = f"{result.conflict_type}_{result.primary_location['name']}"
                else:
                    key = result.conflict_type
                
                self.tracked_events[key].append(result)
                
                # Keep only recent events
                cutoff = datetime.now() - timedelta(days=30)
                self.tracked_events[key] = [
                    r for r in self.tracked_events[key] 
                    if r.timestamp > cutoff
                ]
    
    async def monitor_region(
        self,
        region: str,
        interval: int = 300,  # 5 minutes
        duration: Optional[int] = None
    ):
        """
        Continuously monitor a region for conflicts.
        
        Args:
            region: Region to monitor
            interval: Seconds between searches
            duration: Total monitoring duration in seconds (None for infinite)
        """
        start_time = time.time()
        
        while True:
            try:
                # Search for recent conflicts
                results = await self.search_conflicts(
                    region=region,
                    time_range='day',
                    max_results=20
                )
                
                self.logger.info(
                    f"Region monitor [{region}]: Found {len(results)} results, "
                    f"{len([r for r in results if r.requires_alert])} alerts"
                )
                
                # Process alerts
                if self.alert_queue:
                    await self._process_alerts()
                
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # Wait for next iteration
                await asyncio.sleep(interval)
            
            except Exception as e:
                self.logger.error(f"Error in region monitoring: {e}")
                await asyncio.sleep(interval)
    
    async def _process_alerts(self):
        """Process queued alerts."""
        # In a real implementation, this would send notifications
        alerts = self.alert_queue.copy()
        self.alert_queue.clear()
        
        for alert in alerts:
            self.logger.info(f"Processing alert: {alert.title}")
            # Here you would send email, SMS, webhook, etc.
    
    def get_conflict_stats(self) -> Dict[str, Any]:
        """Get conflict-specific statistics."""
        base_stats = self.get_stats()
        
        conflict_stats = {
            **base_stats,
            'total_alerts': len(self.alert_queue),
            'tracked_conflicts': len(self.tracked_events),
            'recent_escalations': sum(
                1 for events in self.tracked_events.values()
                for event in events
                if event.escalation_indicators
            )
        }
        
        return conflict_stats
    
    async def cleanup(self):
        """Clean up resources including stealth crawler."""
        # Clean up base crawler
        await super().cleanup()
        
        # Clean up stealth crawler
        if self.stealth_crawler:
            await self.stealth_crawler.cleanup()
            
        self.logger.info("Conflict search crawler cleanup completed")


# Convenience functions
def create_conflict_search_crawler(config: Optional[Dict[str, Any]] = None) -> ConflictSearchCrawler:
    """Create a conflict search crawler with configuration."""
    if config:
        crawler_config = ConflictSearchConfig(**config)
    else:
        crawler_config = ConflictSearchConfig()
    
    return ConflictSearchCrawler(crawler_config)


def create_target_location_crawler(
    target_locations: List[str] = None,
    min_articles_per_location: int = 50,
    enable_stealth: bool = True,
    stealth_aggressive: bool = False
) -> ConflictSearchCrawler:
    """
    Create a conflict search crawler optimized for target locations.
    
    Args:
        target_locations: List of target locations (default: Aweil, Karamoja, Mandera, Assosa)
        min_articles_per_location: Minimum articles to find per location
        enable_stealth: Enable stealth downloading
        stealth_aggressive: Use aggressive stealth settings
        
    Returns:
        Configured ConflictSearchCrawler
    """
    if target_locations is None:
        target_locations = ['Aweil', 'Karamoja', 'Mandera', 'Assosa']
    
    config = ConflictSearchConfig(
        target_locations=target_locations,
        min_articles_per_location=min_articles_per_location,
        enable_hierarchical_search=True,
        enable_stealth_download=enable_stealth,
        stealth_aggressive=stealth_aggressive,
        
        # Optimize for conflict monitoring
        engines=['google', 'bing', 'duckduckgo', 'yandex', 'brave', 'startpage', 
                'searx', 'mojeek', 'swisscows', 'baidu'],  # Use all 10 engines
        max_results_per_engine=max(5, min_articles_per_location // 10),
        total_max_results=min_articles_per_location * len(target_locations),
        
        # Enable comprehensive analysis
        extract_entities=True,
        analyze_sentiment=True,
        detect_locations=True,
        track_developments=True,
        enable_alerts=True,
        
        # Optimize scoring for conflict relevance
        min_relevance_score=0.5,
        escalation_threshold=0.7,
        
        # Enable stealth features
        use_stealth=enable_stealth,
        rotate_user_agents=True,
        
        # Longer cache for efficiency
        cache_ttl=7200,  # 2 hours
        
        # Conservative rate limiting for stealth
        min_delay_between_searches=2.0,
        max_concurrent_downloads=3
    )
    
    return ConflictSearchCrawler(config)