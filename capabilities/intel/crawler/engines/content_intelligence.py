"""
APG Crawler Capability - AI-Powered Content Intelligence System
===============================================================

Advanced content intelligence with APG NLP integration:
- Business entity extraction and recognition
- Semantic understanding and context analysis
- Content classification and categorization
- Industry-specific knowledge extraction
- APG ecosystem integration for enhanced processing
- Multi-language support and text normalization
- Content scoring and quality assessment

Copyright © 2025 Datacraft (nyimbi@gmail.com)
"""

import asyncio
import logging
import re
import json
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import hashlib

# NLP and ML libraries
import spacy
from spacy.lang.en import English
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import langdetect
from langdetect.lang_detect_exception import LangDetectException

# APG integrations (would be actual APG modules in production)
from ..views import BusinessEntity, ContentCategory, SemanticTag
from .content_pipeline import ContentExtractionResult

# =====================================================
# CONFIGURATION AND TYPES
# =====================================================

logger = logging.getLogger(__name__)

class EntityType(str, Enum):
	"""Business entity types"""
	ORGANIZATION = "organization"
	PERSON = "person"
	LOCATION = "location"
	PRODUCT = "product"
	EVENT = "event"
	TECHNOLOGY = "technology"
	FINANCIAL = "financial"
	DATE = "date"
	MONEY = "money"
	PERCENTAGE = "percentage"

class ContentCategory(str, Enum):
	"""Content classification categories"""
	NEWS_ARTICLE = "news_article"
	BLOG_POST = "blog_post"
	PRODUCT_DESCRIPTION = "product_description"
	TECHNICAL_DOCUMENTATION = "technical_documentation"
	FINANCIAL_REPORT = "financial_report"
	PRESS_RELEASE = "press_release"
	RESEARCH_PAPER = "research_paper"
	SOCIAL_MEDIA = "social_media"
	FORUM_POST = "forum_post"
	OTHER = "other"

class IndustryDomain(str, Enum):
	"""Industry domain classifications"""
	TECHNOLOGY = "technology"
	FINANCE = "finance"
	HEALTHCARE = "healthcare"
	RETAIL = "retail"
	MANUFACTURING = "manufacturing"
	ENERGY = "energy"
	EDUCATION = "education"
	GOVERNMENT = "government"
	MEDIA = "media"
	GENERAL = "general"

@dataclass
class ExtractedEntity:
	"""Extracted business entity"""
	text: str
	entity_type: EntityType
	confidence: float
	start_pos: int
	end_pos: int
	context: str
	normalized_value: Optional[str] = None
	metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContentClassification:
	"""Content classification result"""
	primary_category: ContentCategory
	confidence: float
	secondary_categories: List[Tuple[ContentCategory, float]]
	industry_domain: IndustryDomain
	domain_confidence: float
	content_type_indicators: List[str]

@dataclass
class SemanticAnalysis:
	"""Semantic analysis result"""
	sentiment: Dict[str, float]  # positive, negative, neutral
	key_themes: List[Tuple[str, float]]  # theme, importance
	semantic_tags: List[str]
	readability_score: float
	complexity_score: float
	language_quality: float

@dataclass
class BusinessIntelligence:
	"""Business intelligence extracted from content"""
	market_signals: List[Dict[str, Any]]
	competitive_mentions: List[Dict[str, Any]]
	trend_indicators: List[Dict[str, Any]]
	financial_metrics: List[Dict[str, Any]]
	technology_mentions: List[Dict[str, Any]]
	people_mentions: List[Dict[str, Any]]

@dataclass
class ContentIntelligenceResult:
	"""Complete content intelligence analysis result"""
	url: str
	extracted_entities: List[ExtractedEntity]
	content_classification: ContentClassification
	semantic_analysis: SemanticAnalysis
	business_intelligence: BusinessIntelligence
	processing_metadata: Dict[str, Any]
	success: bool
	error: Optional[str] = None


# =====================================================
# NLP PROCESSING ENGINE
# =====================================================

class NLPProcessingEngine:
	"""Advanced NLP processing with spaCy and NLTK"""
	
	def __init__(self):
		self.nlp = None
		self.lemmatizer = None
		self.stop_words = None
		self._setup_nltk()
	
	def _setup_nltk(self):
		"""Setup NLTK resources"""
		try:
			nltk.download('punkt', quiet=True)
			nltk.download('stopwords', quiet=True)
			nltk.download('wordnet', quiet=True)
			nltk.download('averaged_perceptron_tagger', quiet=True)
			
			self.lemmatizer = WordNetLemmatizer()
			self.stop_words = set(stopwords.words('english'))
		except Exception as e:
			logger.warning(f"NLTK setup failed: {e}")
	
	def load_nlp_model(self):
		"""Lazy load spaCy NLP model"""
		if not self.nlp:
			try:
				self.nlp = spacy.load("en_core_web_sm")
			except OSError:
				logger.warning("spaCy English model not found, using basic tokenizer")
				self.nlp = English()
				# Add basic pipeline components
				if not self.nlp.has_pipe('ner'):
					# Add basic NER if not present
					pass
	
	async def extract_entities(self, text: str, context: str = "") -> List[ExtractedEntity]:
		"""Extract named entities from text"""
		self.load_nlp_model()
		
		entities = []
		doc = self.nlp(text)
		
		for ent in doc.ents:
			# Map spaCy entity types to our EntityType enum
			entity_type = self._map_spacy_entity_type(ent.label_)
			if entity_type:
				# Calculate confidence based on entity characteristics
				confidence = self._calculate_entity_confidence(ent, doc)
				
				entity = ExtractedEntity(
					text=ent.text,
					entity_type=entity_type,
					confidence=confidence,
					start_pos=ent.start_char,
					end_pos=ent.end_char,
					context=context,
					normalized_value=self._normalize_entity_value(ent.text, entity_type),
					metadata={
						'spacy_label': ent.label_,
						'lemma': ent.lemma_,
						'pos_tag': ent.pos_
					}
				)
				entities.append(entity)
		
		return entities
	
	def _map_spacy_entity_type(self, spacy_label: str) -> Optional[EntityType]:
		"""Map spaCy entity labels to our EntityType enum"""
		mapping = {
			'ORG': EntityType.ORGANIZATION,
			'PERSON': EntityType.PERSON,
			'GPE': EntityType.LOCATION,  # Geopolitical entities
			'LOC': EntityType.LOCATION,
			'PRODUCT': EntityType.PRODUCT,
			'EVENT': EntityType.EVENT,
			'DATE': EntityType.DATE,
			'MONEY': EntityType.MONEY,
			'PERCENT': EntityType.PERCENTAGE,
			'WORK_OF_ART': EntityType.PRODUCT,
			'LAW': EntityType.OTHER,
			'LANGUAGE': EntityType.OTHER,
			'NORP': EntityType.ORGANIZATION,  # Nationalities, organizations
		}
		return mapping.get(spacy_label)
	
	def _calculate_entity_confidence(self, entity, doc) -> float:
		"""Calculate confidence score for extracted entity"""
		base_confidence = 0.7
		
		# Boost confidence for proper nouns
		if entity.pos_ in ['PROPN']:
			base_confidence += 0.1
		
		# Boost confidence for capitalized entities
		if entity.text[0].isupper():
			base_confidence += 0.05
		
		# Boost confidence for longer entities
		if len(entity.text) > 10:
			base_confidence += 0.05
		
		# Reduce confidence for very short entities
		if len(entity.text) < 3:
			base_confidence -= 0.1
		
		return min(base_confidence, 1.0)
	
	def _normalize_entity_value(self, text: str, entity_type: EntityType) -> str:
		"""Normalize entity values for consistency"""
		if entity_type == EntityType.ORGANIZATION:
			# Remove common suffixes like Inc., Corp., etc.
			normalized = re.sub(r'\b(Inc\.|Corp\.|LLC|Ltd\.|Company)\b', '', text).strip()
			return normalized
		elif entity_type == EntityType.PERSON:
			# Standardize name format
			return ' '.join(word.capitalize() for word in text.split())
		elif entity_type == EntityType.MONEY:
			# Extract numeric value
			numeric = re.findall(r'[\d,]+\.?\d*', text)
			return numeric[0] if numeric else text
		elif entity_type == EntityType.PERCENTAGE:
			# Extract percentage value
			numeric = re.findall(r'\d+\.?\d*', text)
			return f"{numeric[0]}%" if numeric else text
		else:
			return text.strip()
	
	async def analyze_sentiment(self, text: str) -> Dict[str, float]:
		"""Analyze sentiment using TextBlob"""
		try:
			blob = TextBlob(text)
			polarity = blob.sentiment.polarity  # -1 to 1
			subjectivity = blob.sentiment.subjectivity  # 0 to 1
			
			# Convert polarity to positive/negative/neutral scores
			if polarity > 0.1:
				positive = polarity
				negative = 0.0
				neutral = 1.0 - polarity
			elif polarity < -0.1:
				positive = 0.0
				negative = abs(polarity)
				neutral = 1.0 - abs(polarity)
			else:
				positive = 0.0
				negative = 0.0
				neutral = 1.0
			
			return {
				'positive': positive,
				'negative': negative,
				'neutral': neutral,
				'subjectivity': subjectivity
			}
		except Exception as e:
			logger.warning(f"Sentiment analysis failed: {e}")
			return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'subjectivity': 0.5}
	
	async def extract_key_themes(self, text: str, max_themes: int = 10) -> List[Tuple[str, float]]:
		"""Extract key themes using keyword extraction"""
		self.load_nlp_model()
		
		doc = self.nlp(text)
		
		# Extract noun phrases and keywords
		themes = {}
		
		# Process noun chunks
		for chunk in doc.noun_chunks:
			if len(chunk.text) > 2 and chunk.root.pos_ in ['NOUN', 'PROPN']:
				clean_text = chunk.text.lower().strip()
				if clean_text not in self.stop_words:
					themes[clean_text] = themes.get(clean_text, 0) + 1
		
		# Process individual tokens
		for token in doc:
			if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
				not token.is_stop and 
				not token.is_punct and 
				len(token.text) > 2):
				lemma = token.lemma_.lower()
				themes[lemma] = themes.get(lemma, 0) + 1
		
		# Calculate theme importance scores
		total_count = sum(themes.values())
		theme_scores = [
			(theme, count / total_count) 
			for theme, count in themes.items()
		]
		
		# Sort by importance and return top themes
		theme_scores.sort(key=lambda x: x[1], reverse=True)
		return theme_scores[:max_themes]
	
	async def calculate_readability(self, text: str) -> float:
		"""Calculate readability score (Flesch Reading Ease)"""
		try:
			sentences = sent_tokenize(text)
			words = word_tokenize(text)
			
			if not sentences or not words:
				return 0.0
			
			# Calculate averages
			avg_sentence_length = len(words) / len(sentences)
			
			# Count syllables (simplified)
			def count_syllables(word):
				word = word.lower()
				syllables = len(re.findall(r'[aeiouy]+', word))
				return max(1, syllables)
			
			total_syllables = sum(count_syllables(word) for word in words)
			avg_syllables_per_word = total_syllables / len(words)
			
			# Flesch Reading Ease formula
			score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
			
			# Normalize to 0-1 scale
			return max(0.0, min(1.0, score / 100.0))
			
		except Exception as e:
			logger.warning(f"Readability calculation failed: {e}")
			return 0.5


# =====================================================
# CONTENT CLASSIFICATION ENGINE
# =====================================================

class ContentClassificationEngine:
	"""Content classification and categorization"""
	
	def __init__(self):
		self.category_keywords = self._build_category_keywords()
		self.industry_keywords = self._build_industry_keywords()
	
	def _build_category_keywords(self) -> Dict[ContentCategory, List[str]]:
		"""Build keyword mappings for content categories"""
		return {
			ContentCategory.NEWS_ARTICLE: [
				'breaking', 'reported', 'sources', 'according to', 'journalist',
				'newspaper', 'correspondent', 'wire service', 'press'
			],
			ContentCategory.BLOG_POST: [
				'posted by', 'author:', 'blog', 'opinion', 'personal', 'thoughts',
				'my take', 'i think', 'in my opinion', 'comments'
			],
			ContentCategory.PRODUCT_DESCRIPTION: [
				'features', 'specifications', 'buy now', 'price', 'product',
				'available', 'order', 'purchase', 'technical specs'
			],
			ContentCategory.TECHNICAL_DOCUMENTATION: [
				'documentation', 'api', 'reference', 'guide', 'tutorial',
				'installation', 'configuration', 'usage', 'parameters'
			],
			ContentCategory.FINANCIAL_REPORT: [
				'revenue', 'earnings', 'fiscal', 'quarterly', 'annual report',
				'financial results', 'investor', 'sec filing', 'balance sheet'
			],
			ContentCategory.PRESS_RELEASE: [
				'press release', 'announces', 'company news', 'for immediate release',
				'media contact', 'pr newswire', 'business wire'
			],
			ContentCategory.RESEARCH_PAPER: [
				'abstract', 'methodology', 'conclusions', 'references', 'study',
				'research', 'analysis', 'findings', 'peer reviewed'
			],
			ContentCategory.SOCIAL_MEDIA: [
				'tweet', 'facebook', 'instagram', 'linkedin', 'social',
				'share', 'like', 'follow', 'hashtag', 'mention'
			],
			ContentCategory.FORUM_POST: [
				'forum', 'thread', 'reply', 'post', 'discussion', 'community',
				'member', 'moderator', 'topic', 'board'
			]
		}
	
	def _build_industry_keywords(self) -> Dict[IndustryDomain, List[str]]:
		"""Build keyword mappings for industry domains"""
		return {
			IndustryDomain.TECHNOLOGY: [
				'software', 'hardware', 'ai', 'machine learning', 'cloud',
				'saas', 'startup', 'tech', 'digital', 'platform', 'api'
			],
			IndustryDomain.FINANCE: [
				'bank', 'financial', 'investment', 'trading', 'stock',
				'market', 'fund', 'capital', 'finance', 'money', 'credit'
			],
			IndustryDomain.HEALTHCARE: [
				'medical', 'health', 'hospital', 'doctor', 'patient',
				'treatment', 'drug', 'pharmaceutical', 'clinical', 'healthcare'
			],
			IndustryDomain.RETAIL: [
				'retail', 'store', 'shopping', 'consumer', 'brand',
				'merchandise', 'sales', 'customer', 'e-commerce', 'marketplace'
			],
			IndustryDomain.MANUFACTURING: [
				'manufacturing', 'factory', 'production', 'industrial',
				'supply chain', 'logistics', 'automation', 'machinery'
			],
			IndustryDomain.ENERGY: [
				'energy', 'oil', 'gas', 'renewable', 'solar', 'wind',
				'power', 'electricity', 'utility', 'nuclear', 'coal'
			],
			IndustryDomain.EDUCATION: [
				'education', 'school', 'university', 'student', 'teacher',
				'learning', 'academic', 'curriculum', 'degree', 'campus'
			],
			IndustryDomain.GOVERNMENT: [
				'government', 'policy', 'regulation', 'public', 'federal',
				'state', 'municipal', 'agency', 'administration', 'congress'
			],
			IndustryDomain.MEDIA: [
				'media', 'news', 'television', 'radio', 'broadcast',
				'journalism', 'content', 'entertainment', 'streaming'
			]
		}
	
	async def classify_content(self, text: str, metadata: Dict[str, Any] = None) -> ContentClassification:
		"""Classify content category and industry domain"""
		text_lower = text.lower()
		metadata = metadata or {}
		
		# Calculate category scores
		category_scores = {}
		for category, keywords in self.category_keywords.items():
			score = sum(1 for keyword in keywords if keyword in text_lower)
			if score > 0:
				category_scores[category] = score / len(keywords)
		
		# Calculate industry scores
		industry_scores = {}
		for industry, keywords in self.industry_keywords.items():
			score = sum(1 for keyword in keywords if keyword in text_lower)
			if score > 0:
				industry_scores[industry] = score / len(keywords)
		
		# Determine primary category
		if category_scores:
			primary_category = max(category_scores.items(), key=lambda x: x[1])
			primary_cat, primary_conf = primary_category
		else:
			primary_cat, primary_conf = ContentCategory.OTHER, 0.1
		
		# Get secondary categories
		secondary_categories = [
			(cat, score) for cat, score in category_scores.items() 
			if cat != primary_cat and score > 0.1
		]
		secondary_categories = sorted(secondary_categories, key=lambda x: x[1], reverse=True)[:3]
		
		# Determine industry domain
		if industry_scores:
			industry_item = max(industry_scores.items(), key=lambda x: x[1])
			industry_domain, domain_confidence = industry_item
		else:
			industry_domain, domain_confidence = IndustryDomain.GENERAL, 0.1
		
		# Extract content type indicators
		indicators = []
		url = metadata.get('url', '')
		if 'blog' in url:
			indicators.append('blog_url')
		if 'news' in url:
			indicators.append('news_url')
		if 'press' in text_lower:
			indicators.append('press_content')
		if 'technical' in text_lower or 'documentation' in text_lower:
			indicators.append('technical_content')
		
		return ContentClassification(
			primary_category=primary_cat,
			confidence=primary_conf,
			secondary_categories=secondary_categories,
			industry_domain=industry_domain,
			domain_confidence=domain_confidence,
			content_type_indicators=indicators
		)


# =====================================================
# BUSINESS INTELLIGENCE ENGINE
# =====================================================

class BusinessIntelligenceEngine:
	"""Extract business intelligence from content"""
	
	def __init__(self):
		self.financial_patterns = self._build_financial_patterns()
		self.market_signal_patterns = self._build_market_patterns()
		self.tech_patterns = self._build_tech_patterns()
	
	def _build_financial_patterns(self) -> List[Tuple[str, str]]:
		"""Build regex patterns for financial information"""
		return [
			(r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?', 'monetary_amount'),
			(r'revenue.*?\$[\d,]+', 'revenue_mention'),
			(r'profit.*?\$[\d,]+', 'profit_mention'),
			(r'(?:raised|funding|investment).*?\$[\d,]+', 'funding_mention'),
			(r'valuation.*?\$[\d,]+', 'valuation_mention'),
			(r'market cap.*?\$[\d,]+', 'market_cap_mention'),
			(r'stock price.*?\$[\d,]+', 'stock_price_mention'),
		]
	
	def _build_market_patterns(self) -> List[Tuple[str, str]]:
		"""Build patterns for market signals"""
		return [
			(r'(?:growth|increase|up|rising).*?(\d+(?:\.\d+)?%)', 'positive_growth'),
			(r'(?:decline|decrease|down|falling).*?(\d+(?:\.\d+)?%)', 'negative_growth'),
			(r'market share.*?(\d+(?:\.\d+)?%)', 'market_share'),
			(r'(?:merger|acquisition|acquired|bought)', 'ma_activity'),
			(r'(?:partnership|collaboration|alliance)', 'partnership'),
			(r'(?:ipo|public offering|going public)', 'ipo_activity'),
			(r'(?:bankruptcy|bankrupt|chapter 11)', 'financial_distress'),
		]
	
	def _build_tech_patterns(self) -> List[Tuple[str, str]]:
		"""Build patterns for technology mentions"""
		return [
			(r'(?:artificial intelligence|ai|machine learning|ml)', 'ai_tech'),
			(r'(?:cloud computing|cloud|saas|paas|iaas)', 'cloud_tech'),
			(r'(?:blockchain|cryptocurrency|bitcoin|ethereum)', 'blockchain_tech'),
			(r'(?:5g|internet of things|iot|edge computing)', 'emerging_tech'),
			(r'(?:cybersecurity|security|encryption|privacy)', 'security_tech'),
			(r'(?:mobile|app|smartphone|tablet)', 'mobile_tech'),
		]
	
	async def extract_business_intelligence(self, text: str, entities: List[ExtractedEntity]) -> BusinessIntelligence:
		"""Extract comprehensive business intelligence"""
		
		# Extract financial metrics
		financial_metrics = self._extract_financial_metrics(text)
		
		# Extract market signals
		market_signals = self._extract_market_signals(text)
		
		# Extract competitive mentions
		competitive_mentions = self._extract_competitive_mentions(text, entities)
		
		# Extract trend indicators
		trend_indicators = self._extract_trend_indicators(text)
		
		# Extract technology mentions
		technology_mentions = self._extract_technology_mentions(text)
		
		# Extract people mentions
		people_mentions = self._extract_people_mentions(entities)
		
		return BusinessIntelligence(
			market_signals=market_signals,
			competitive_mentions=competitive_mentions,
			trend_indicators=trend_indicators,
			financial_metrics=financial_metrics,
			technology_mentions=technology_mentions,
			people_mentions=people_mentions
		)
	
	def _extract_financial_metrics(self, text: str) -> List[Dict[str, Any]]:
		"""Extract financial metrics from text"""
		metrics = []
		
		for pattern, metric_type in self.financial_patterns:
			matches = re.finditer(pattern, text, re.IGNORECASE)
			for match in matches:
				metrics.append({
					'type': metric_type,
					'value': match.group(0),
					'position': match.span(),
					'context': text[max(0, match.start()-50):match.end()+50]
				})
		
		return metrics
	
	def _extract_market_signals(self, text: str) -> List[Dict[str, Any]]:
		"""Extract market signals and trends"""
		signals = []
		
		for pattern, signal_type in self.market_signal_patterns:
			matches = re.finditer(pattern, text, re.IGNORECASE)
			for match in matches:
				signals.append({
					'type': signal_type,
					'text': match.group(0),
					'position': match.span(),
					'context': text[max(0, match.start()-50):match.end()+50]
				})
		
		return signals
	
	def _extract_competitive_mentions(self, text: str, entities: List[ExtractedEntity]) -> List[Dict[str, Any]]:
		"""Extract competitive intelligence"""
		competitive_mentions = []
		
		# Find organization entities that might be competitors
		organizations = [e for e in entities if e.entity_type == EntityType.ORGANIZATION]
		
		for org in organizations:
			# Look for competitive context around the organization mention
			start_pos = max(0, org.start_pos - 100)
			end_pos = min(len(text), org.end_pos + 100)
			context = text[start_pos:end_pos]
			
			competitive_indicators = [
				'competitor', 'rival', 'competing', 'market leader',
				'alternative', 'versus', 'compared to', 'against'
			]
			
			if any(indicator in context.lower() for indicator in competitive_indicators):
				competitive_mentions.append({
					'organization': org.text,
					'confidence': org.confidence,
					'context': context,
					'competitive_indicators': [
						ind for ind in competitive_indicators 
						if ind in context.lower()
					]
				})
		
		return competitive_mentions
	
	def _extract_trend_indicators(self, text: str) -> List[Dict[str, Any]]:
		"""Extract trend and future direction indicators"""
		trends = []
		
		trend_patterns = [
			(r'(?:trending|trend|popular|growing|emerging)', 'positive_trend'),
			(r'(?:declining|decreasing|obsolete|legacy)', 'negative_trend'),
			(r'(?:future|upcoming|next year|roadmap)', 'future_indicator'),
			(r'(?:adoption|implementation|deployment)', 'adoption_indicator'),
		]
		
		for pattern, trend_type in trend_patterns:
			matches = re.finditer(pattern, text, re.IGNORECASE)
			for match in matches:
				trends.append({
					'type': trend_type,
					'text': match.group(0),
					'position': match.span(),
					'context': text[max(0, match.start()-50):match.end()+50]
				})
		
		return trends
	
	def _extract_technology_mentions(self, text: str) -> List[Dict[str, Any]]:
		"""Extract technology and innovation mentions"""
		tech_mentions = []
		
		for pattern, tech_type in self.tech_patterns:
			matches = re.finditer(pattern, text, re.IGNORECASE)
			for match in matches:
				tech_mentions.append({
					'type': tech_type,
					'technology': match.group(0),
					'position': match.span(),
					'context': text[max(0, match.start()-50):match.end()+50]
				})
		
		return tech_mentions
	
	def _extract_people_mentions(self, entities: List[ExtractedEntity]) -> List[Dict[str, Any]]:
		"""Extract people mentions with roles and context"""
		people_mentions = []
		
		people_entities = [e for e in entities if e.entity_type == EntityType.PERSON]
		
		for person in people_entities:
			# Analyze context for role indicators
			context = person.context.lower()
			
			role_indicators = {
				'ceo': ['ceo', 'chief executive', 'chief executive officer'],
				'cto': ['cto', 'chief technology', 'chief technical officer'],
				'founder': ['founder', 'co-founder', 'founded'],
				'investor': ['investor', 'investment', 'invested'],
				'analyst': ['analyst', 'research', 'analysis'],
				'executive': ['executive', 'president', 'vice president', 'vp'],
			}
			
			detected_roles = []
			for role, indicators in role_indicators.items():
				if any(indicator in context for indicator in indicators):
					detected_roles.append(role)
			
			people_mentions.append({
				'name': person.text,
				'confidence': person.confidence,
				'roles': detected_roles,
				'context': person.context
			})
		
		return people_mentions


# =====================================================
# MAIN CONTENT INTELLIGENCE ENGINE
# =====================================================

class ContentIntelligenceEngine:
	"""Main AI-powered content intelligence system"""
	
	def __init__(self):
		self.nlp_engine = NLPProcessingEngine()
		self.classification_engine = ContentClassificationEngine()
		self.business_intelligence_engine = BusinessIntelligenceEngine()
	
	async def analyze_content(self, extraction_result: ContentExtractionResult, 
							 business_context: Dict[str, Any] = None) -> ContentIntelligenceResult:
		"""Perform comprehensive content intelligence analysis"""
		
		business_context = business_context or {}
		
		try:
			# Get the best available content
			content_text = (
				extraction_result.cleaned_content or 
				extraction_result.main_content or 
				extraction_result.raw_content
			)
			
			if not content_text or len(content_text) < 50:
				return ContentIntelligenceResult(
					url=extraction_result.url,
					extracted_entities=[],
					content_classification=ContentClassification(
						primary_category=ContentCategory.OTHER,
						confidence=0.0,
						secondary_categories=[],
						industry_domain=IndustryDomain.GENERAL,
						domain_confidence=0.0,
						content_type_indicators=[]
					),
					semantic_analysis=SemanticAnalysis(
						sentiment={'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'subjectivity': 0.5},
						key_themes=[],
						semantic_tags=[],
						readability_score=0.0,
						complexity_score=0.0,
						language_quality=0.0
					),
					business_intelligence=BusinessIntelligence(
						market_signals=[],
						competitive_mentions=[],
						trend_indicators=[],
						financial_metrics=[],
						technology_mentions=[],
						people_mentions=[]
					),
					processing_metadata={'error': 'Insufficient content for analysis'},
					success=False,
					error="Content too short for meaningful analysis"
				)
			
			# Extract entities
			logger.info(f"Extracting entities from content: {extraction_result.url}")
			extracted_entities = await self.nlp_engine.extract_entities(
				content_text, 
				context=business_context.get('domain', '')
			)
			
			# Classify content
			logger.info(f"Classifying content: {extraction_result.url}")
			content_classification = await self.classification_engine.classify_content(
				content_text, 
				metadata={'url': extraction_result.url}
			)
			
			# Semantic analysis
			logger.info(f"Analyzing semantics: {extraction_result.url}")
			sentiment = await self.nlp_engine.analyze_sentiment(content_text)
			key_themes = await self.nlp_engine.extract_key_themes(content_text)
			readability = await self.nlp_engine.calculate_readability(content_text)
			
			# Generate semantic tags from themes and entities
			semantic_tags = [theme[0] for theme in key_themes[:10]]
			semantic_tags.extend([e.text.lower() for e in extracted_entities[:5]])
			semantic_tags = list(set(semantic_tags))  # Remove duplicates
			
			# Calculate complexity and quality scores
			complexity_score = self._calculate_complexity_score(content_text, extracted_entities)
			language_quality = self._calculate_language_quality(content_text, extraction_result)
			
			semantic_analysis = SemanticAnalysis(
				sentiment=sentiment,
				key_themes=key_themes,
				semantic_tags=semantic_tags,
				readability_score=readability,
				complexity_score=complexity_score,
				language_quality=language_quality
			)
			
			# Business intelligence extraction
			logger.info(f"Extracting business intelligence: {extraction_result.url}")
			business_intelligence = await self.business_intelligence_engine.extract_business_intelligence(
				content_text, extracted_entities
			)
			
			# Processing metadata
			processing_metadata = {
				'content_length': len(content_text),
				'entity_count': len(extracted_entities),
				'theme_count': len(key_themes),
				'processing_time': datetime.utcnow().isoformat(),
				'nlp_model': 'spacy_en_core_web_sm',
				'business_context': business_context
			}
			
			return ContentIntelligenceResult(
				url=extraction_result.url,
				extracted_entities=extracted_entities,
				content_classification=content_classification,
				semantic_analysis=semantic_analysis,
				business_intelligence=business_intelligence,
				processing_metadata=processing_metadata,
				success=True
			)
			
		except Exception as e:
			logger.error(f"Content intelligence analysis failed for {extraction_result.url}: {str(e)}")
			return ContentIntelligenceResult(
				url=extraction_result.url,
				extracted_entities=[],
				content_classification=ContentClassification(
					primary_category=ContentCategory.OTHER,
					confidence=0.0,
					secondary_categories=[],
					industry_domain=IndustryDomain.GENERAL,
					domain_confidence=0.0,
					content_type_indicators=[]
				),
				semantic_analysis=SemanticAnalysis(
					sentiment={'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'subjectivity': 0.5},
					key_themes=[],
					semantic_tags=[],
					readability_score=0.0,
					complexity_score=0.0,
					language_quality=0.0
				),
				business_intelligence=BusinessIntelligence(
					market_signals=[],
					competitive_mentions=[],
					trend_indicators=[],
					financial_metrics=[],
					technology_mentions=[],
					people_mentions=[]
				),
				processing_metadata={'error': str(e)},
				success=False,
				error=str(e)
			)
	
	def _calculate_complexity_score(self, text: str, entities: List[ExtractedEntity]) -> float:
		"""Calculate content complexity based on various factors"""
		try:
			# Basic metrics
			word_count = len(text.split())
			sentence_count = len([s for s in text.split('.') if s.strip()])
			
			if sentence_count == 0:
				return 0.0
			
			avg_sentence_length = word_count / sentence_count
			
			# Vocabulary complexity (unique words ratio)
			words = text.lower().split()
			unique_words = set(words)
			vocabulary_complexity = len(unique_words) / len(words) if words else 0
			
			# Entity density
			entity_density = len(entities) / word_count if word_count > 0 else 0
			
			# Technical terms (approximation)
			technical_patterns = [
				r'\b[A-Z]{2,}\b',  # Acronyms
				r'\b\w+(?:tion|sion|ment|ness|ity)\b',  # Complex suffixes
				r'\b(?:however|therefore|consequently|furthermore)\b'  # Complex connectors
			]
			
			technical_count = sum(
				len(re.findall(pattern, text, re.IGNORECASE)) 
				for pattern in technical_patterns
			)
			technical_density = technical_count / word_count if word_count > 0 else 0
			
			# Combine factors
			complexity_score = (
				min(avg_sentence_length / 20, 1.0) * 0.3 +  # Sentence length factor
				vocabulary_complexity * 0.3 +  # Vocabulary factor
				min(entity_density * 10, 1.0) * 0.2 +  # Entity factor
				min(technical_density * 5, 1.0) * 0.2  # Technical factor
			)
			
			return min(complexity_score, 1.0)
			
		except Exception:
			return 0.5
	
	def _calculate_language_quality(self, text: str, extraction_result: ContentExtractionResult) -> float:
		"""Calculate language quality score"""
		try:
			quality_score = 0.0
			
			# Spelling and grammar approximation (basic checks)
			# Count potential spelling errors (words with unusual patterns)
			words = text.split()
			potential_errors = len([
				w for w in words 
				if len(w) > 3 and 
				not re.match(r'^[a-zA-Z]+$', w) and 
				not re.match(r'^\d+$', w)
			])
			
			spelling_quality = max(0, 1.0 - (potential_errors / len(words) * 2)) if words else 0
			quality_score += spelling_quality * 0.4
			
			# Structure quality (presence of proper formatting)
			has_paragraphs = '\n\n' in text or len(text.split('\n')) > 3
			has_sentences = text.count('.') > 2
			has_capitalization = sum(1 for c in text if c.isupper()) > 0
			
			structure_quality = (
				(0.4 if has_paragraphs else 0) +
				(0.4 if has_sentences else 0) +
				(0.2 if has_capitalization else 0)
			)
			quality_score += structure_quality * 0.3
			
			# Content completeness (based on extraction success)
			if extraction_result.title:
				quality_score += 0.1
			if extraction_result.author:
				quality_score += 0.1
			if extraction_result.publish_date:
				quality_score += 0.1
			
			return min(quality_score, 1.0)
			
		except Exception:
			return 0.5
	
	async def batch_analyze(self, extraction_results: List[ContentExtractionResult],
						   business_context: Dict[str, Any] = None) -> List[ContentIntelligenceResult]:
		"""Analyze multiple content extraction results in parallel"""
		
		tasks = [
			self.analyze_content(result, business_context)
			for result in extraction_results
		]
		
		return await asyncio.gather(*tasks, return_exceptions=False)


# =====================================================
# EXPORTS
# =====================================================

__all__ = [
	'ContentIntelligenceEngine',
	'ContentIntelligenceResult',
	'ExtractedEntity',
	'ContentClassification',
	'SemanticAnalysis',
	'BusinessIntelligence',
	'EntityType',
	'ContentCategory',
	'IndustryDomain'
]