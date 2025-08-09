#!/usr/bin/env python3
"""
APG Cache Management (CACH) - Content Analysis Engine
Content-aware optimization with semantic understanding

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import logging
import json
import hashlib
import zlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import re
import mimetypes

from .models import CacheEntry, CompressionAlgorithm


class ContentType(str, Enum):
	"""Detected content types for optimization"""
	TEXT = "text"
	JSON = "json"
	XML = "xml"
	HTML = "html"
	CSS = "css"
	JAVASCRIPT = "javascript"
	IMAGE = "image"
	VIDEO = "video"
	AUDIO = "audio"
	BINARY = "binary"
	STRUCTURED_DATA = "structured_data"
	API_RESPONSE = "api_response"
	USER_DATA = "user_data"
	CONFIGURATION = "configuration"


class CompressionStrategy(str, Enum):
	"""Compression strategies based on content"""
	AGGRESSIVE = "aggressive"  # Maximum compression
	BALANCED = "balanced"     # Balance compression vs speed
	FAST = "fast"            # Prioritize speed
	NONE = "none"            # No compression
	ADAPTIVE = "adaptive"     # AI-determined strategy


@dataclass
class ContentProfile:
	"""Comprehensive content analysis profile"""
	content_type: ContentType
	mime_type: str
	size_bytes: int
	
	# Content characteristics
	text_ratio: float = 0.0  # Proportion of text content
	entropy: float = 0.0     # Shannon entropy
	compressibility_score: float = 0.0
	structure_complexity: float = 0.0
	
	# Semantic analysis
	semantic_tags: List[str] = field(default_factory=list)
	entities_detected: List[str] = field(default_factory=list)
	language_detected: Optional[str] = None
	domain_category: Optional[str] = None
	
	# Optimization hints
	recommended_compression: CompressionAlgorithm = CompressionAlgorithm.LZ4
	recommended_strategy: CompressionStrategy = CompressionStrategy.BALANCED
	serialization_hints: Dict[str, Any] = field(default_factory=dict)
	
	# Relationships
	related_content_patterns: List[str] = field(default_factory=list)
	version_lineage: Optional[str] = None
	
	# Analysis metadata
	analyzed_at: datetime = field(default_factory=datetime.utcnow)
	analysis_confidence: float = 0.0
	analysis_duration_ms: float = 0.0


@dataclass
class OptimizationRecommendation:
	"""Content-specific optimization recommendation"""
	key: str
	content_profile: ContentProfile
	
	# Storage optimizations
	compression_algorithm: CompressionAlgorithm
	expected_compression_ratio: float
	serialization_format: str
	
	# Access optimizations
	prefetch_priority: float
	cache_tier_preference: str
	ttl_recommendation: Optional[int]
	
	# Performance predictions
	size_reduction_percent: float
	access_speed_improvement: float
	storage_cost_reduction: float
	
	confidence_score: float
	reasoning: str


class ContentAnalysisEngine:
	"""
	Revolutionary content-aware optimization engine
	Revolutionary Differentiator #5: Content-Aware Optimization
	"""
	
	def __init__(self, config: Dict[str, Any] = None):
		self.config = config or {}
		self.logger = logging.getLogger('cach.content_analysis')
		
		# Analysis state
		self.content_profiles: Dict[str, ContentProfile] = {}
		self.content_patterns: Dict[ContentType, List[str]] = defaultdict(list)
		self.optimization_history: List[OptimizationRecommendation] = []
		
		# Content type detectors
		self.mime_detector = mimetypes.MimeTypes()
		self.text_patterns = self._build_text_patterns()
		self.structured_patterns = self._build_structured_patterns()
		
		# Compression analyzers
		self.compression_analyzers = {
			CompressionAlgorithm.GZIP: self._analyze_gzip_suitability,
			CompressionAlgorithm.LZ4: self._analyze_lz4_suitability,
			CompressionAlgorithm.ZSTD: self._analyze_zstd_suitability,
			CompressionAlgorithm.BROTLI: self._analyze_brotli_suitability
		}
		
		# Semantic analyzers
		self.entity_extractors = {
			ContentType.JSON: self._extract_json_entities,
			ContentType.TEXT: self._extract_text_entities,
			ContentType.HTML: self._extract_html_entities,
			ContentType.API_RESPONSE: self._extract_api_entities
		}
		
		# Configuration
		self.min_analysis_size = self.config.get('min_analysis_size', 100)  # bytes
		self.max_analysis_size = self.config.get('max_analysis_size', 10485760)  # 10MB
		self.entropy_sample_size = 1024  # Sample size for entropy calculation
		
	async def analyze_content(self, key: str, data: bytes) -> ContentProfile:
		"""
		Comprehensive content analysis for optimization
		Semantic data understanding and intelligent optimization
		"""
		
		analysis_start = datetime.utcnow()
		
		# Basic content detection
		content_type, mime_type = await self._detect_content_type(key, data)
		
		# Create initial profile
		profile = ContentProfile(
			content_type=content_type,
			mime_type=mime_type,
			size_bytes=len(data)
		)
		
		# Skip analysis for very small or very large content
		if len(data) < self.min_analysis_size:
			profile.analysis_confidence = 0.3
			return profile
		
		if len(data) > self.max_analysis_size:
			# Sample large content
			sample_data = self._sample_large_content(data)
			data = sample_data
		
		# Perform detailed analysis
		await self._analyze_content_characteristics(data, profile)
		await self._analyze_semantic_content(key, data, profile)
		await self._analyze_compression_suitability(data, profile)
		await self._analyze_content_relationships(key, profile)
		
		# Calculate final confidence and duration
		profile.analysis_confidence = self._calculate_analysis_confidence(profile)
		profile.analysis_duration_ms = (datetime.utcnow() - analysis_start).total_seconds() * 1000
		
		# Store profile for future reference
		self.content_profiles[key] = profile
		
		self.logger.debug(f"Content analysis for {key}: {content_type.value} "
						 f"({profile.analysis_confidence:.2f} confidence)")
		
		return profile
	
	async def generate_optimization_recommendations(self, key: str, entry: CacheEntry,
													profile: Optional[ContentProfile] = None) -> OptimizationRecommendation:
		"""
		Generate content-aware optimization recommendations
		Custom serialization and compression optimization
		"""
		
		if not profile:
			profile = self.content_profiles.get(key)
			if not profile:
				# Analyze content on-demand
				profile = await self.analyze_content(key, entry.value)
		
		# Generate optimization recommendation
		recommendation = OptimizationRecommendation(
			key=key,
			content_profile=profile,
			compression_algorithm=profile.recommended_compression,
			expected_compression_ratio=await self._predict_compression_ratio(profile),
			serialization_format=self._recommend_serialization_format(profile),
			prefetch_priority=self._calculate_prefetch_priority(profile),
			cache_tier_preference=self._recommend_cache_tier(profile),
			ttl_recommendation=self._recommend_ttl(profile),
			size_reduction_percent=0.0,  # Will be calculated
			access_speed_improvement=0.0,  # Will be calculated
			storage_cost_reduction=0.0,  # Will be calculated
			confidence_score=profile.analysis_confidence,
			reasoning=""
		)
		
		# Calculate performance predictions
		await self._calculate_optimization_impact(recommendation, entry)
		
		# Generate reasoning
		recommendation.reasoning = self._generate_optimization_reasoning(recommendation)
		
		# Store recommendation
		self.optimization_history.append(recommendation)
		self._cleanup_optimization_history()
		
		return recommendation
	
	async def optimize_content_serialization(self, data: Any, content_type: ContentType) -> Tuple[bytes, str]:
		"""
		Optimize content serialization based on content type
		Intelligent object serialization strategies
		"""
		
		if content_type == ContentType.JSON:
			return await self._optimize_json_serialization(data)
		elif content_type == ContentType.XML:
			return await self._optimize_xml_serialization(data)
		elif content_type == ContentType.STRUCTURED_DATA:
			return await self._optimize_structured_serialization(data)
		elif content_type == ContentType.TEXT:
			return await self._optimize_text_serialization(data)
		else:
			# Default serialization
			if isinstance(data, bytes):
				return data, "bytes"
			elif isinstance(data, str):
				return data.encode('utf-8'), "utf-8"
			else:
				return json.dumps(data, separators=(',', ':')).encode('utf-8'), "json-compact"
	
	async def analyze_content_relationships(self, entries: Dict[str, CacheEntry]) -> Dict[str, List[str]]:
		"""
		Analyze relationships between cached content items
		Content relationship modeling for intelligent prefetching
		"""
		
		relationships = defaultdict(list)
		
		# Analyze content similarity
		for key1, entry1 in entries.items():
			profile1 = self.content_profiles.get(key1)
			if not profile1:
				continue
			
			for key2, entry2 in entries.items():
				if key1 >= key2:  # Avoid duplicate comparisons
					continue
				
				profile2 = self.content_profiles.get(key2)
				if not profile2:
					continue
				
				similarity = self._calculate_content_similarity(profile1, profile2)
				if similarity > 0.7:  # High similarity threshold
					relationships[key1].append(key2)
					relationships[key2].append(key1)
		
		# Analyze structural patterns
		structural_relationships = await self._analyze_structural_patterns(entries)
		for key, related_keys in structural_relationships.items():
			relationships[key].extend(related_keys)
		
		# Analyze temporal access patterns
		temporal_relationships = await self._analyze_temporal_access_patterns(entries)
		for key, related_keys in temporal_relationships.items():
			relationships[key].extend(related_keys)
		
		# Remove duplicates
		for key in relationships:
			relationships[key] = list(set(relationships[key]))
		
		return dict(relationships)
	
	async def get_content_insights(self) -> Dict[str, Any]:
		"""Get comprehensive content analysis insights"""
		
		# Content type distribution
		content_types = Counter(profile.content_type for profile in self.content_profiles.values())
		
		# Compression effectiveness by type
		compression_stats = defaultdict(list)
		for recommendation in self.optimization_history:
			content_type = recommendation.content_profile.content_type
			compression_stats[content_type.value].append(recommendation.expected_compression_ratio)
		
		compression_avg = {
			content_type: sum(ratios) / len(ratios) if ratios else 0.0
			for content_type, ratios in compression_stats.items()
		}
		
		# Optimization impact
		total_size_reduction = sum(rec.size_reduction_percent for rec in self.optimization_history)
		avg_confidence = sum(profile.analysis_confidence for profile in self.content_profiles.values()) / max(len(self.content_profiles), 1)
		
		return {
			'total_profiles': len(self.content_profiles),
			'content_type_distribution': dict(content_types),
			'average_analysis_confidence': avg_confidence,
			'compression_effectiveness': compression_avg,
			'total_optimization_recommendations': len(self.optimization_history),
			'estimated_size_reduction_percent': total_size_reduction / max(len(self.optimization_history), 1),
			'analysis_coverage': {
				'text_content': content_types.get(ContentType.TEXT, 0),
				'structured_data': content_types.get(ContentType.JSON, 0) + content_types.get(ContentType.XML, 0),
				'binary_content': content_types.get(ContentType.BINARY, 0),
				'media_content': content_types.get(ContentType.IMAGE, 0) + content_types.get(ContentType.VIDEO, 0)
			}
		}
	
	# Private implementation methods
	
	async def _detect_content_type(self, key: str, data: bytes) -> Tuple[ContentType, str]:
		"""Detect content type from key and data"""
		
		# Try MIME type detection from key
		mime_type, _ = self.mime_detector.guess_type(key)
		
		if not mime_type:
			# Detect from content
			mime_type = await self._detect_mime_from_content(data)
		
		# Map MIME type to ContentType
		content_type = self._map_mime_to_content_type(mime_type)
		
		# Additional heuristic detection
		if content_type == ContentType.BINARY:
			content_type = await self._heuristic_content_detection(key, data)
		
		return content_type, mime_type or "application/octet-stream"
	
	async def _detect_mime_from_content(self, data: bytes) -> Optional[str]:
		"""Detect MIME type from content analysis"""
		
		if not data:
			return None
		
		# Sample first 1KB for analysis
		sample = data[:1024]
		
		try:
			# Try to decode as text
			text = sample.decode('utf-8', errors='ignore')
			
			if text.strip().startswith('{') and '}' in text:
				return "application/json"
			elif text.strip().startswith('<') and '>' in text:
				if '<!DOCTYPE html' in text or '<html' in text:
					return "text/html"
				else:
					return "application/xml"
			elif 'function' in text or 'var ' in text or 'const ' in text:
				return "application/javascript"
			elif any(css_indicator in text for css_indicator in ['{', '}', ':', ';', 'color', 'font']):
				return "text/css"
			else:
				return "text/plain"
		
		except UnicodeDecodeError:
			pass
		
		# Binary content detection
		if sample.startswith(b'\xFF\xD8\xFF'):
			return "image/jpeg"
		elif sample.startswith(b'\x89PNG\r\n\x1a\n'):
			return "image/png"
		elif sample.startswith(b'GIF87a') or sample.startswith(b'GIF89a'):
			return "image/gif"
		elif sample.startswith(b'%PDF'):
			return "application/pdf"
		
		return "application/octet-stream"
	
	def _map_mime_to_content_type(self, mime_type: Optional[str]) -> ContentType:
		"""Map MIME type to ContentType enum"""
		
		if not mime_type:
			return ContentType.BINARY
		
		mime_type = mime_type.lower()
		
		if mime_type.startswith('text/'):
			if mime_type == 'text/html':
				return ContentType.HTML
			elif mime_type == 'text/css':
				return ContentType.CSS
			else:
				return ContentType.TEXT
		
		elif mime_type == 'application/json':
			return ContentType.JSON
		elif mime_type in ['application/xml', 'text/xml']:
			return ContentType.XML
		elif mime_type == 'application/javascript':
			return ContentType.JAVASCRIPT
		elif mime_type.startswith('image/'):
			return ContentType.IMAGE
		elif mime_type.startswith('video/'):
			return ContentType.VIDEO
		elif mime_type.startswith('audio/'):
			return ContentType.AUDIO
		else:
			return ContentType.BINARY
	
	async def _heuristic_content_detection(self, key: str, data: bytes) -> ContentType:
		"""Additional heuristic content type detection"""
		
		# Key-based detection
		key_lower = key.lower()
		
		if 'api' in key_lower or 'response' in key_lower:
			return ContentType.API_RESPONSE
		elif 'user' in key_lower or 'profile' in key_lower:
			return ContentType.USER_DATA
		elif 'config' in key_lower or 'settings' in key_lower:
			return ContentType.CONFIGURATION
		
		# Content pattern analysis
		if len(data) > 0:
			try:
				# Try JSON parsing
				json.loads(data.decode('utf-8'))
				return ContentType.STRUCTURED_DATA
			except (json.JSONDecodeError, UnicodeDecodeError):
				pass
		
		return ContentType.BINARY
	
	async def _analyze_content_characteristics(self, data: bytes, profile: ContentProfile) -> None:
		"""Analyze basic content characteristics"""
		
		# Calculate text ratio
		try:
			text = data.decode('utf-8', errors='ignore')
			printable_chars = sum(1 for c in text if c.isprintable())
			profile.text_ratio = printable_chars / max(len(text), 1)
		except:
			profile.text_ratio = 0.0
		
		# Calculate entropy
		profile.entropy = self._calculate_shannon_entropy(data)
		
		# Calculate compressibility score
		profile.compressibility_score = await self._estimate_compressibility(data)
		
		# Analyze structure complexity
		if profile.content_type in [ContentType.JSON, ContentType.XML, ContentType.HTML]:
			profile.structure_complexity = await self._analyze_structure_complexity(data)
	
	async def _analyze_semantic_content(self, key: str, data: bytes, profile: ContentProfile) -> None:
		"""Analyze semantic content for optimization hints"""
		
		# Extract entities based on content type
		if profile.content_type in self.entity_extractors:
			extractor = self.entity_extractors[profile.content_type]
			profile.entities_detected = await extractor(data)
		
		# Detect language for text content
		if profile.content_type == ContentType.TEXT and profile.text_ratio > 0.8:
			profile.language_detected = await self._detect_language(data)
		
		# Categorize domain
		profile.domain_category = self._categorize_domain(key)
		
		# Generate semantic tags
		profile.semantic_tags = await self._generate_semantic_tags(key, data, profile)
	
	async def _analyze_compression_suitability(self, data: bytes, profile: ContentProfile) -> None:
		"""Analyze compression suitability and recommend algorithm"""
		
		compression_scores = {}
		
		for algorithm, analyzer in self.compression_analyzers.items():
			score = await analyzer(data, profile)
			compression_scores[algorithm] = score
		
		# Select best compression algorithm
		best_algorithm = max(compression_scores.items(), key=lambda x: x[1])
		profile.recommended_compression = best_algorithm[0]
		
		# Determine compression strategy
		if profile.entropy > 7.5:  # High entropy, hard to compress
			profile.recommended_strategy = CompressionStrategy.FAST
		elif profile.compressibility_score > 0.8:  # Highly compressible
			profile.recommended_strategy = CompressionStrategy.AGGRESSIVE
		else:
			profile.recommended_strategy = CompressionStrategy.BALANCED
	
	async def _analyze_content_relationships(self, key: str, profile: ContentProfile) -> None:
		"""Analyze content relationships for prefetching hints"""
		
		# Pattern-based relationships
		key_pattern = self._extract_key_pattern(key)
		if key_pattern:
			profile.related_content_patterns.append(key_pattern)
		
		# Version lineage detection
		version_pattern = self._detect_version_pattern(key)
		if version_pattern:
			profile.version_lineage = version_pattern
	
	def _calculate_shannon_entropy(self, data: bytes) -> float:
		"""Calculate Shannon entropy of data"""
		
		if not data:
			return 0.0
		
		# Sample data if too large
		sample = data[:self.entropy_sample_size] if len(data) > self.entropy_sample_size else data
		
		# Calculate byte frequency
		byte_counts = Counter(sample)
		total_bytes = len(sample)
		
		# Calculate entropy
		entropy = 0.0
		for count in byte_counts.values():
			probability = count / total_bytes
			if probability > 0:
				entropy -= probability * math.log2(probability)
		
		return entropy
	
	async def _estimate_compressibility(self, data: bytes) -> float:
		"""Estimate how well data will compress"""
		
		if len(data) < 100:
			return 0.5  # Uncertain for small data
		
		# Quick compression test with zlib
		try:
			compressed = zlib.compress(data[:1024])  # Sample first 1KB
			ratio = len(compressed) / min(len(data), 1024)
			compressibility = 1.0 - ratio
			return max(0.0, min(compressibility, 1.0))
		except:
			return 0.5
	
	async def _analyze_structure_complexity(self, data: bytes) -> float:
		"""Analyze structural complexity of structured content"""
		
		try:
			if data.startswith(b'{') or data.startswith(b'['):
				# JSON complexity
				obj = json.loads(data.decode('utf-8'))
				return self._calculate_json_complexity(obj)
			elif data.startswith(b'<'):
				# XML/HTML complexity
				return self._calculate_markup_complexity(data.decode('utf-8'))
		except:
			pass
		
		return 0.5
	
	def _calculate_json_complexity(self, obj: Any, depth: int = 0) -> float:
		"""Calculate JSON structure complexity"""
		
		if depth > 10:  # Prevent infinite recursion
			return 1.0
		
		if isinstance(obj, dict):
			complexity = len(obj) * 0.1
			for value in obj.values():
				complexity += self._calculate_json_complexity(value, depth + 1) * 0.1
		elif isinstance(obj, list):
			complexity = len(obj) * 0.05
			for item in obj[:10]:  # Sample first 10 items
				complexity += self._calculate_json_complexity(item, depth + 1) * 0.05
		else:
			complexity = 0.1
		
		return min(complexity, 1.0)
	
	def _calculate_markup_complexity(self, text: str) -> float:
		"""Calculate markup (HTML/XML) complexity"""
		
		# Count tags
		tag_count = len(re.findall(r'<[^>]+>', text))
		
		# Count nesting levels
		max_depth = 0
		current_depth = 0
		for char in text:
			if char == '<':
				if text[text.index(char):].startswith('</'):
					current_depth -= 1
				else:
					current_depth += 1
					max_depth = max(max_depth, current_depth)
		
		complexity = (tag_count * 0.01) + (max_depth * 0.1)
		return min(complexity, 1.0)
	
	# Compression analysis methods
	
	async def _analyze_gzip_suitability(self, data: bytes, profile: ContentProfile) -> float:
		"""Analyze suitability for GZIP compression"""
		
		score = 0.5  # Base score
		
		# Text content compresses well with GZIP
		if profile.text_ratio > 0.8:
			score += 0.3
		
		# Lower entropy indicates better compression
		if profile.entropy < 6.0:
			score += 0.2
		
		# Structured data compresses well
		if profile.content_type in [ContentType.JSON, ContentType.XML, ContentType.HTML]:
			score += 0.2
		
		return min(score, 1.0)
	
	async def _analyze_lz4_suitability(self, data: bytes, profile: ContentProfile) -> float:
		"""Analyze suitability for LZ4 compression"""
		
		score = 0.7  # LZ4 is generally good
		
		# Fast compression for frequently accessed content
		if profile.content_type == ContentType.API_RESPONSE:
			score += 0.2
		
		# Good for structured data with repetition
		if profile.structure_complexity > 0.5:
			score += 0.1
		
		return min(score, 1.0)
	
	async def _analyze_zstd_suitability(self, data: bytes, profile: ContentProfile) -> float:
		"""Analyze suitability for Zstandard compression"""
		
		score = 0.6  # Base score
		
		# Excellent for text and structured data
		if profile.content_type in [ContentType.TEXT, ContentType.JSON, ContentType.XML]:
			score += 0.3
		
		# Good compression ratio for larger content
		if profile.size_bytes > 10240:  # > 10KB
			score += 0.2
		
		return min(score, 1.0)
	
	async def _analyze_brotli_suitability(self, data: bytes, profile: ContentProfile) -> float:
		"""Analyze suitability for Brotli compression"""
		
		score = 0.4  # Lower base score due to compression time
		
		# Excellent for web content
		if profile.content_type in [ContentType.HTML, ContentType.CSS, ContentType.JAVASCRIPT]:
			score += 0.4
		
		# Good for text with high compressibility
		if profile.text_ratio > 0.9 and profile.compressibility_score > 0.8:
			score += 0.3
		
		return min(score, 1.0)
	
	# Entity extraction methods
	
	async def _extract_json_entities(self, data: bytes) -> List[str]:
		"""Extract entities from JSON content"""
		
		entities = []
		try:
			obj = json.loads(data.decode('utf-8'))
			entities.extend(self._extract_json_keys(obj))
		except:
			pass
		
		return entities
	
	async def _extract_text_entities(self, data: bytes) -> List[str]:
		"""Extract entities from text content"""
		
		try:
			text = data.decode('utf-8')
			# Simple entity extraction (would use NLP in production)
			words = re.findall(r'\b[A-Z][a-z]+\b', text)
			return list(set(words))[:10]  # Top 10 capitalized words
		except:
			return []
	
	async def _extract_html_entities(self, data: bytes) -> List[str]:
		"""Extract entities from HTML content"""
		
		try:
			html = data.decode('utf-8')
			# Extract tag names
			tags = re.findall(r'<(\w+)', html)
			# Extract IDs and classes
			ids = re.findall(r'id=["\']([^"\']+)["\']', html)
			classes = re.findall(r'class=["\']([^"\']+)["\']', html)
			
			entities = list(set(tags + ids + classes))
			return entities[:20]  # Top 20 entities
		except:
			return []
	
	async def _extract_api_entities(self, data: bytes) -> List[str]:
		"""Extract entities from API response content"""
		
		try:
			# Try JSON first
			obj = json.loads(data.decode('utf-8'))
			return self._extract_json_keys(obj)
		except:
			# Fall back to text extraction
			return await self._extract_text_entities(data)
	
	def _extract_json_keys(self, obj: Any, depth: int = 0) -> List[str]:
		"""Recursively extract keys from JSON object"""
		
		if depth > 3:  # Limit depth
			return []
		
		keys = []
		if isinstance(obj, dict):
			keys.extend(obj.keys())
			for value in obj.values():
				keys.extend(self._extract_json_keys(value, depth + 1))
		elif isinstance(obj, list) and obj:
			keys.extend(self._extract_json_keys(obj[0], depth + 1))  # Sample first item
		
		return [str(key) for key in keys if isinstance(key, (str, int))]
	
	# Helper methods
	
	async def _detect_language(self, data: bytes) -> Optional[str]:
		"""Detect language of text content (simplified)"""
		
		try:
			text = data.decode('utf-8').lower()
			
			# Simple heuristic language detection
			if any(word in text for word in ['the', 'and', 'is', 'in', 'to', 'of']):
				return 'en'
			elif any(word in text for word in ['le', 'la', 'et', 'est', 'dans', 'de']):
				return 'fr'
			elif any(word in text for word in ['der', 'die', 'das', 'und', 'ist', 'in']):
				return 'de'
			
		except:
			pass
		
		return None
	
	def _categorize_domain(self, key: str) -> Optional[str]:
		"""Categorize content domain from key"""
		
		key_lower = key.lower()
		
		if any(term in key_lower for term in ['user', 'profile', 'account']):
			return 'user_management'
		elif any(term in key_lower for term in ['product', 'catalog', 'inventory']):
			return 'e_commerce'
		elif any(term in key_lower for term in ['api', 'service', 'endpoint']):
			return 'api_services'
		elif any(term in key_lower for term in ['config', 'setting', 'preference']):
			return 'configuration'
		elif any(term in key_lower for term in ['log', 'metric', 'analytics']):
			return 'monitoring'
		
		return 'general'
	
	async def _generate_semantic_tags(self, key: str, data: bytes, profile: ContentProfile) -> List[str]:
		"""Generate semantic tags for content"""
		
		tags = []
		
		# Add content type tag
		tags.append(f"type:{profile.content_type.value}")
		
		# Add size category
		if profile.size_bytes < 1024:
			tags.append("size:small")
		elif profile.size_bytes < 1048576:
			tags.append("size:medium")
		else:
			tags.append("size:large")
		
		# Add compressibility tag
		if profile.compressibility_score > 0.8:
			tags.append("compression:high")
		elif profile.compressibility_score > 0.5:
			tags.append("compression:medium")
		else:
			tags.append("compression:low")
		
		# Add domain tag if available
		if profile.domain_category:
			tags.append(f"domain:{profile.domain_category}")
		
		# Add structural tags
		if profile.structure_complexity > 0.7:
			tags.append("structure:complex")
		elif profile.structure_complexity > 0.3:
			tags.append("structure:moderate")
		else:
			tags.append("structure:simple")
		
		return tags
	
	# Content optimization methods
	
	async def _optimize_json_serialization(self, data: Any) -> Tuple[bytes, str]:
		"""Optimize JSON serialization"""
		
		# Compact JSON without extra whitespace
		json_str = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
		return json_str.encode('utf-8'), "json-compact"
	
	async def _optimize_xml_serialization(self, data: Any) -> Tuple[bytes, str]:
		"""Optimize XML serialization"""
		
		if isinstance(data, bytes):
			return data, "xml-binary"
		elif isinstance(data, str):
			# Remove unnecessary whitespace
			xml_str = re.sub(r'>\s+<', '><', data)
			return xml_str.encode('utf-8'), "xml-compressed"
		else:
			return str(data).encode('utf-8'), "xml-string"
	
	async def _optimize_structured_serialization(self, data: Any) -> Tuple[bytes, str]:
		"""Optimize structured data serialization"""
		
		# Use compact JSON as default for structured data
		return await self._optimize_json_serialization(data)
	
	async def _optimize_text_serialization(self, data: Any) -> Tuple[bytes, str]:
		"""Optimize text serialization"""
		
		if isinstance(data, str):
			# Use UTF-8 encoding
			return data.encode('utf-8'), "utf-8"
		elif isinstance(data, bytes):
			return data, "bytes"
		else:
			return str(data).encode('utf-8'), "utf-8"
	
	# Analysis helper methods
	
	def _sample_large_content(self, data: bytes) -> bytes:
		"""Sample large content for analysis"""
		
		# Take samples from beginning, middle, and end
		size = len(data)
		sample_size = self.max_analysis_size // 3
		
		beginning = data[:sample_size]
		middle_start = size // 2 - sample_size // 2
		middle = data[middle_start:middle_start + sample_size]
		end = data[-sample_size:]
		
		return beginning + middle + end
	
	def _calculate_analysis_confidence(self, profile: ContentProfile) -> float:
		"""Calculate confidence score for content analysis"""
		
		confidence = 0.5  # Base confidence
		
		# Higher confidence for larger content
		if profile.size_bytes > 1024:
			confidence += 0.2
		
		# Higher confidence for recognized content types
		if profile.content_type != ContentType.BINARY:
			confidence += 0.2
		
		# Higher confidence for structured content
		if profile.content_type in [ContentType.JSON, ContentType.XML]:
			confidence += 0.1
		
		# Higher confidence for text content
		if profile.text_ratio > 0.8:
			confidence += 0.1
		
		return min(confidence, 1.0)
	
	def _build_text_patterns(self) -> Dict[str, List[str]]:
		"""Build text content patterns"""
		
		return {
			'log_patterns': [r'\d{4}-\d{2}-\d{2}', r'ERROR', r'INFO', r'DEBUG'],
			'code_patterns': [r'function\s+\w+', r'class\s+\w+', r'import\s+\w+'],
			'data_patterns': [r'\w+@\w+\.\w+', r'\+?1?\d{10}', r'\d{4}-\d{4}-\d{4}-\d{4}']
		}
	
	def _build_structured_patterns(self) -> Dict[str, List[str]]:
		"""Build structured content patterns"""
		
		return {
			'api_patterns': ['status', 'data', 'message', 'error', 'success'],
			'config_patterns': ['host', 'port', 'username', 'password', 'settings'],
			'user_patterns': ['id', 'name', 'email', 'profile', 'preferences']
		}
	
	def _extract_key_pattern(self, key: str) -> Optional[str]:
		"""Extract generalized pattern from cache key"""
		
		# Replace IDs and timestamps with wildcards
		pattern = re.sub(r'\d+', '*', key)
		pattern = re.sub(r'[a-f0-9]{8,}', '*', pattern)  # Replace hex IDs
		
		if pattern != key and '*' in pattern:
			return pattern
		
		return None
	
	def _detect_version_pattern(self, key: str) -> Optional[str]:
		"""Detect version lineage in cache key"""
		
		# Look for version patterns
		version_match = re.search(r'v\d+|version\d+|\d+\.\d+', key.lower())
		if version_match:
			base_key = key[:version_match.start()] + key[version_match.end():]
			return base_key
		
		return None
	
	def _calculate_content_similarity(self, profile1: ContentProfile, profile2: ContentProfile) -> float:
		"""Calculate similarity between two content profiles"""
		
		similarity = 0.0
		
		# Content type similarity
		if profile1.content_type == profile2.content_type:
			similarity += 0.3
		
		# Domain similarity
		if profile1.domain_category == profile2.domain_category:
			similarity += 0.2
		
		# Semantic tag similarity
		common_tags = set(profile1.semantic_tags) & set(profile2.semantic_tags)
		if common_tags:
			tag_similarity = len(common_tags) / max(len(profile1.semantic_tags), len(profile2.semantic_tags), 1)
			similarity += tag_similarity * 0.3
		
		# Size similarity
		size_ratio = min(profile1.size_bytes, profile2.size_bytes) / max(profile1.size_bytes, profile2.size_bytes, 1)
		similarity += size_ratio * 0.2
		
		return similarity
	
	async def _analyze_structural_patterns(self, entries: Dict[str, CacheEntry]) -> Dict[str, List[str]]:
		"""Analyze structural patterns in cache keys"""
		
		relationships = defaultdict(list)
		
		# Group keys by pattern
		pattern_groups = defaultdict(list)
		for key in entries.keys():
			pattern = self._extract_key_pattern(key)
			if pattern:
				pattern_groups[pattern].append(key)
		
		# Find related keys within pattern groups
		for pattern, keys in pattern_groups.items():
			if len(keys) > 1:
				for i, key1 in enumerate(keys):
					for key2 in keys[i+1:]:
						relationships[key1].append(key2)
						relationships[key2].append(key1)
		
		return relationships
	
	async def _analyze_temporal_access_patterns(self, entries: Dict[str, CacheEntry]) -> Dict[str, List[str]]:
		"""Analyze temporal access patterns for relationships"""
		
		relationships = defaultdict(list)
		
		# Find entries accessed within similar time windows
		time_groups = defaultdict(list)
		for key, entry in entries.items():
			if entry.last_accessed:
				# Group by hour
				hour_key = entry.last_accessed.replace(minute=0, second=0, microsecond=0)
				time_groups[hour_key].append(key)
		
		# Create relationships for co-accessed items
		for keys in time_groups.values():
			if len(keys) > 1:
				for i, key1 in enumerate(keys):
					for key2 in keys[i+1:]:
						relationships[key1].append(key2)
						relationships[key2].append(key1)
		
		return relationships
	
	# Recommendation helper methods
	
	async def _predict_compression_ratio(self, profile: ContentProfile) -> float:
		"""Predict compression ratio based on content analysis"""
		
		if profile.compressibility_score > 0.8:
			return 0.3  # Expect 70% reduction
		elif profile.compressibility_score > 0.6:
			return 0.5  # Expect 50% reduction
		elif profile.compressibility_score > 0.4:
			return 0.7  # Expect 30% reduction
		else:
			return 0.9  # Expect 10% reduction
	
	def _recommend_serialization_format(self, profile: ContentProfile) -> str:
		"""Recommend optimal serialization format"""
		
		if profile.content_type == ContentType.JSON:
			return "json-compact"
		elif profile.content_type == ContentType.XML:
			return "xml-compressed"
		elif profile.content_type == ContentType.TEXT:
			return "utf-8"
		else:
			return "binary"
	
	def _calculate_prefetch_priority(self, profile: ContentProfile) -> float:
		"""Calculate prefetch priority based on content analysis"""
		
		priority = 0.5  # Base priority
		
		# Higher priority for structured data
		if profile.content_type in [ContentType.JSON, ContentType.API_RESPONSE]:
			priority += 0.2
		
		# Higher priority for user-related content
		if profile.domain_category == 'user_management':
			priority += 0.2
		
		# Higher priority for smaller content
		if profile.size_bytes < 10240:  # < 10KB
			priority += 0.1
		
		return min(priority, 1.0)
	
	def _recommend_cache_tier(self, profile: ContentProfile) -> str:
		"""Recommend optimal cache tier based on content"""
		
		# Small, frequently accessed content -> L1
		if profile.size_bytes < 1024 and profile.domain_category in ['user_management', 'api_services']:
			return "L1"
		
		# Structured data -> L2
		elif profile.content_type in [ContentType.JSON, ContentType.API_RESPONSE]:
			return "L2"
		
		# Large content -> L3
		elif profile.size_bytes > 1048576:  # > 1MB
			return "L3"
		
		# Default -> L2
		else:
			return "L2"
	
	def _recommend_ttl(self, profile: ContentProfile) -> Optional[int]:
		"""Recommend TTL based on content type and characteristics"""
		
		# Configuration data - longer TTL
		if profile.domain_category == 'configuration':
			return 7200  # 2 hours
		
		# User data - moderate TTL
		elif profile.domain_category == 'user_management':
			return 1800  # 30 minutes
		
		# API responses - shorter TTL
		elif profile.content_type == ContentType.API_RESPONSE:
			return 300  # 5 minutes
		
		# Default
		else:
			return 3600  # 1 hour
	
	async def _calculate_optimization_impact(self, recommendation: OptimizationRecommendation,
											 entry: CacheEntry) -> None:
		"""Calculate expected optimization impact"""
		
		# Size reduction from compression
		recommendation.size_reduction_percent = (1.0 - recommendation.expected_compression_ratio) * 100
		
		# Access speed improvement (simplified calculation)
		if recommendation.content_profile.recommended_strategy == CompressionStrategy.FAST:
			recommendation.access_speed_improvement = 10.0
		elif recommendation.content_profile.recommended_strategy == CompressionStrategy.BALANCED:
			recommendation.access_speed_improvement = 5.0
		else:
			recommendation.access_speed_improvement = 2.0
		
		# Storage cost reduction
		recommendation.storage_cost_reduction = recommendation.size_reduction_percent * 0.5
	
	def _generate_optimization_reasoning(self, recommendation: OptimizationRecommendation) -> str:
		"""Generate human-readable reasoning for optimization recommendation"""
		
		reasons = []
		profile = recommendation.content_profile
		
		# Compression reasoning
		reasons.append(f"Content type {profile.content_type.value} with {profile.compressibility_score:.1%} compressibility")
		
		# Algorithm reasoning
		reasons.append(f"Recommended {recommendation.compression_algorithm.value} compression for optimal balance")
		
		# Strategy reasoning
		if profile.recommended_strategy == CompressionStrategy.AGGRESSIVE:
			reasons.append("Aggressive compression for maximum space saving")
		elif profile.recommended_strategy == CompressionStrategy.FAST:
			reasons.append("Fast compression for frequent access patterns")
		
		return "; ".join(reasons)
	
	def _cleanup_optimization_history(self) -> None:
		"""Clean up optimization history to prevent memory growth"""
		
		if len(self.optimization_history) > 1000:
			self.optimization_history = self.optimization_history[-1000:]


# Export main components
__all__ = [
	'ContentAnalysisEngine',
	'ContentType',
	'CompressionStrategy',
	'ContentProfile',
	'OptimizationRecommendation'
]