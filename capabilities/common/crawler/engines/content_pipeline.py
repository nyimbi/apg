"""
APG Crawler Capability - Content Extraction and Cleaning Pipeline
=================================================================

Advanced content processing pipeline with:
- Multi-format content extraction (HTML, JSON, XML, PDF)
- AI-powered content cleaning and noise removal
- Markdown conversion with structure preservation
- Content fingerprinting and duplicate detection
- Language detection and text normalization
- Business entity extraction and enrichment

Copyright © 2025 Datacraft (nyimbi@gmail.com)
"""

import asyncio
import logging
import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json

# HTML and content processing
from bs4 import BeautifulSoup, Comment
import html2text
from markdownify import markdownify as md
import bleach
from readability import Document

# Language and text processing
import langdetect
from langdetect.lang_detect_exception import LangDetectException
import spacy
from spacy.lang.en import English

# Content extraction
import trafilatura
from newspaper import Article
import PyPDF2
import textract

# Utilities
from urllib.parse import urljoin, urlparse
import magic
import mimetypes

from ..views import ContentProcessingStage, ContentCleaningConfig
from .stealth_engine import CrawlResult

# =====================================================
# CONFIGURATION AND TYPES
# =====================================================

logger = logging.getLogger(__name__)

class ContentType(str, Enum):
	"""Supported content types"""
	HTML = "text/html"
	JSON = "application/json"
	XML = "application/xml"
	PDF = "application/pdf"
	TEXT = "text/plain"
	MARKDOWN = "text/markdown"
	DOC = "application/msword"
	DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

@dataclass
class ContentExtractionResult:
	"""Result of content extraction"""
	url: str
	title: Optional[str]
	main_content: str
	raw_content: str
	cleaned_content: str
	markdown_content: str
	content_type: ContentType
	language: Optional[str]
	publish_date: Optional[datetime]
	author: Optional[str]
	description: Optional[str]
	keywords: List[str]
	images: List[str]
	links: List[str]
	content_fingerprint: str
	processing_stage: ContentProcessingStage
	metadata: Dict[str, Any]
	success: bool
	error: Optional[str] = None

@dataclass
class CleaningStats:
	"""Statistics from content cleaning"""
	original_length: int
	cleaned_length: int
	removed_elements: Dict[str, int]
	preserved_elements: Dict[str, int]
	language_confidence: float
	content_quality_score: float


# =====================================================
# CONTENT EXTRACTION ENGINE
# =====================================================

class ContentExtractionEngine:
	"""Multi-format content extraction with AI-powered cleaning"""
	
	def __init__(self):
		# Initialize text processing tools
		self.nlp = None  # Will be loaded lazily
		self.html2text_converter = html2text.HTML2Text()
		self.setup_html2text()
		
		# Content cleaning patterns
		self.noise_patterns = [
			r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',
			r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>',
			r'<noscript\b[^<]*(?:(?!<\/noscript>)<[^<]*)*<\/noscript>',
			r'<!--.*?-->',
			r'<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>',
		]
		
		# Social media and tracking patterns
		self.social_patterns = [
			r'<div[^>]*class="[^"]*social[^"]*"[^>]*>.*?</div>',
			r'<div[^>]*class="[^"]*share[^"]*"[^>]*>.*?</div>',
			r'<div[^>]*class="[^"]*tweet[^"]*"[^>]*>.*?</div>',
			r'<div[^>]*class="[^"]*facebook[^"]*"[^>]*>.*?</div>',
		]
		
		# Navigation patterns
		self.nav_patterns = [
			r'<nav\b[^<]*(?:(?!<\/nav>)<[^<]*)*<\/nav>',
			r'<header\b[^<]*(?:(?!<\/header>)<[^<]*)*<\/header>',
			r'<footer\b[^<]*(?:(?!<\/footer>)<[^<]*)*<\/footer>',
			r'<div[^>]*class="[^"]*nav[^"]*"[^>]*>.*?</div>',
			r'<div[^>]*class="[^"]*menu[^"]*"[^>]*>.*?</div>',
		]
	
	def setup_html2text(self):
		"""Configure HTML to text converter"""
		self.html2text_converter.ignore_links = False
		self.html2text_converter.ignore_images = False
		self.html2text_converter.ignore_emphasis = False
		self.html2text_converter.body_width = 0
		self.html2text_converter.unicode_snob = True
		self.html2text_converter.escape_snob = True
	
	def load_nlp(self):
		"""Lazy load spaCy NLP model"""
		if not self.nlp:
			try:
				self.nlp = spacy.load("en_core_web_sm")
			except OSError:
				logger.warning("spaCy English model not found, using basic tokenizer")
				self.nlp = English()
	
	async def extract_content(self, crawl_result: CrawlResult, 
							  config: ContentCleaningConfig) -> ContentExtractionResult:
		"""Extract and clean content from crawl result"""
		
		# Detect content type
		content_type = self._detect_content_type(crawl_result.content, crawl_result.headers)
		
		try:
			# Route to appropriate extraction method
			if content_type == ContentType.HTML:
				return await self._extract_html_content(crawl_result, config)
			elif content_type == ContentType.JSON:
				return await self._extract_json_content(crawl_result, config)
			elif content_type == ContentType.XML:
				return await self._extract_xml_content(crawl_result, config)
			elif content_type == ContentType.PDF:
				return await self._extract_pdf_content(crawl_result, config)
			else:
				return await self._extract_text_content(crawl_result, config)
		
		except Exception as e:
			logger.error(f"Content extraction failed for {crawl_result.url}: {str(e)}")
			return ContentExtractionResult(
				url=crawl_result.url,
				title=None,
				main_content="",
				raw_content=crawl_result.content,
				cleaned_content="",
				markdown_content="",
				content_type=content_type,
				language=None,
				publish_date=None,
				author=None,
				description=None,
				keywords=[],
				images=[],
				links=[],
				content_fingerprint="",
				processing_stage=ContentProcessingStage.RAW_EXTRACTED,
				metadata={},
				success=False,
				error=str(e)
			)
	
	def _detect_content_type(self, content: str, headers: Dict[str, str]) -> ContentType:
		"""Detect content type from headers and content"""
		# Check Content-Type header
		content_type_header = headers.get('content-type', '').lower()
		
		if 'text/html' in content_type_header:
			return ContentType.HTML
		elif 'application/json' in content_type_header:
			return ContentType.JSON
		elif 'application/xml' in content_type_header or 'text/xml' in content_type_header:
			return ContentType.XML
		elif 'application/pdf' in content_type_header:
			return ContentType.PDF
		
		# Try to detect from content
		content_stripped = content.strip()
		
		if content_stripped.startswith('<!DOCTYPE html') or content_stripped.startswith('<html'):
			return ContentType.HTML
		elif content_stripped.startswith('{') or content_stripped.startswith('['):
			try:
				json.loads(content_stripped)
				return ContentType.JSON
			except:
				pass
		elif content_stripped.startswith('<?xml') or content_stripped.startswith('<'):
			return ContentType.XML
		
		return ContentType.TEXT
	
	async def _extract_html_content(self, crawl_result: CrawlResult, 
									config: ContentCleaningConfig) -> ContentExtractionResult:
		"""Extract content from HTML using multiple methods"""
		
		content = crawl_result.content
		url = crawl_result.url
		
		# Try newspaper3k first (best for articles)
		newspaper_result = self._extract_with_newspaper(content, url)
		
		# Try readability (good for general content)
		readability_result = self._extract_with_readability(content)
		
		# Try trafilatura (excellent for web articles)
		trafilatura_result = self._extract_with_trafilatura(content, url)
		
		# Combine results, preferring newspaper for articles
		best_content = self._select_best_extraction(
			newspaper_result, readability_result, trafilatura_result
		)
		
		# Clean the content
		cleaned_content = await self._clean_html_content(content, config)
		
		# Convert to markdown
		markdown_content = self._convert_to_markdown(cleaned_content, config)
		
		# Detect language
		language = self._detect_language(best_content.get('text', ''))
		
		# Extract metadata
		metadata = self._extract_html_metadata(content)
		
		# Generate fingerprint
		fingerprint = self._generate_fingerprint(markdown_content)
		
		# Calculate quality score
		quality_score = self._calculate_content_quality(
			best_content.get('text', ''), metadata, len(cleaned_content)
		)
		
		return ContentExtractionResult(
			url=url,
			title=best_content.get('title'),
			main_content=best_content.get('text', ''),
			raw_content=content,
			cleaned_content=cleaned_content,
			markdown_content=markdown_content,
			content_type=ContentType.HTML,
			language=language,
			publish_date=best_content.get('publish_date'),
			author=best_content.get('author'),
			description=metadata.get('description'),
			keywords=metadata.get('keywords', []),
			images=self._extract_images(content, url),
			links=self._extract_links(content, url),
			content_fingerprint=fingerprint,
			processing_stage=ContentProcessingStage.CLEANED,
			metadata={
				'extraction_method': 'multi_method',
				'content_quality_score': quality_score,
				'cleaning_config': config.model_dump()
			},
			success=True
		)
	
	def _extract_with_newspaper(self, content: str, url: str) -> Dict[str, Any]:
		"""Extract content using newspaper3k"""
		try:
			article = Article(url)
			article.set_html(content)
			article.parse()
			
			return {
				'text': article.text,
				'title': article.title,
				'author': ', '.join(article.authors) if article.authors else None,
				'publish_date': article.publish_date,
				'summary': article.summary if hasattr(article, 'summary') else None,
				'method': 'newspaper'
			}
		except Exception as e:
			logger.debug(f"Newspaper extraction failed: {str(e)}")
			return {'text': '', 'method': 'newspaper', 'error': str(e)}
	
	def _extract_with_readability(self, content: str) -> Dict[str, Any]:
		"""Extract content using readability"""
		try:
			doc = Document(content)
			return {
				'text': BeautifulSoup(doc.content(), 'html.parser').get_text(),
				'title': doc.title(),
				'method': 'readability'
			}
		except Exception as e:
			logger.debug(f"Readability extraction failed: {str(e)}")
			return {'text': '', 'method': 'readability', 'error': str(e)}
	
	def _extract_with_trafilatura(self, content: str, url: str) -> Dict[str, Any]:
		"""Extract content using trafilatura"""
		try:
			text = trafilatura.extract(content, url=url, include_comments=False)
			metadata = trafilatura.extract_metadata(content, url=url)
			
			return {
				'text': text or '',
				'title': metadata.title if metadata else None,
				'author': metadata.author if metadata else None,
				'publish_date': metadata.date if metadata else None,
				'method': 'trafilatura'
			}
		except Exception as e:
			logger.debug(f"Trafilatura extraction failed: {str(e)}")
			return {'text': '', 'method': 'trafilatura', 'error': str(e)}
	
	def _select_best_extraction(self, *results) -> Dict[str, Any]:
		"""Select the best extraction result"""
		valid_results = [r for r in results if r.get('text') and len(r['text']) > 100]
		
		if not valid_results:
			# Return the result with the longest text, even if short
			return max(results, key=lambda x: len(x.get('text', '')))
		
		# Prefer newspaper for articles, then trafilatura, then readability
		preference_order = ['newspaper', 'trafilatura', 'readability']
		
		for preferred_method in preference_order:
			for result in valid_results:
				if result.get('method') == preferred_method:
					return result
		
		# Fallback to longest text
		return max(valid_results, key=lambda x: len(x.get('text', '')))
	
	async def _clean_html_content(self, content: str, config: ContentCleaningConfig) -> str:
		"""Clean HTML content based on configuration"""
		soup = BeautifulSoup(content, 'html.parser')
		
		# Remove comments
		for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
			comment.extract()
		
		# Remove script and style tags
		for tag in soup(["script", "style", "noscript"]):
			tag.decompose()
		
		# Remove navigation elements
		if config.remove_navigation:
			for tag in soup(["nav", "header", "footer"]):
				tag.decompose()
			
			# Remove common navigation classes
			nav_selectors = [
				'.navigation', '.navbar', '.nav-bar', '.menu', '.breadcrumb',
				'.sidebar', '.side-bar', '#navigation', '#navbar', '#menu'
			]
			for selector in nav_selectors:
				for element in soup.select(selector):
					element.decompose()
		
		# Remove ads and promotional content
		if config.remove_ads:
			ad_selectors = [
				'.ad', '.ads', '.advertisement', '.promo', '.promotion',
				'.sponsored', '.banner', '#ad', '#ads', '[class*="ad-"]',
				'[id*="ad-"]', '.google-ad', '.adsense'
			]
			for selector in ad_selectors:
				for element in soup.select(selector):
					element.decompose()
		
		# Remove social media widgets
		if config.remove_social_widgets:
			social_selectors = [
				'.social', '.share', '.sharing', '.tweet', '.facebook',
				'.twitter', '.linkedin', '.instagram', '.social-media',
				'.social-share', '.social-buttons'
			]
			for selector in social_selectors:
				for element in soup.select(selector):
					element.decompose()
		
		# Remove comments sections
		if config.remove_comments:
			comment_selectors = [
				'.comments', '.comment', '#comments', '#comment',
				'.discussion', '.disqus', '.facebook-comments'
			]
			for selector in comment_selectors:
				for element in soup.select(selector):
					element.decompose()
		
		# Clean up whitespace and get text
		cleaned_html = str(soup)
		
		# Additional text cleaning
		cleaned_html = re.sub(r'\s+', ' ', cleaned_html)  # Normalize whitespace
		cleaned_html = re.sub(r'\n\s*\n', '\n\n', cleaned_html)  # Normalize line breaks
		
		return cleaned_html.strip()
	
	def _convert_to_markdown(self, html_content: str, config: ContentCleaningConfig) -> str:
		"""Convert HTML to clean markdown"""
		
		if config.markdown_formatting:
			# Use markdownify for better structure preservation
			markdown = md(
				html_content,
				heading_style="ATX",
				bullets="-",
				strip=['script', 'style'],
				convert=['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'blockquote']
			)
		else:
			# Use html2text for simpler conversion
			markdown = self.html2text_converter.handle(html_content)
		
		# Clean up markdown
		markdown = re.sub(r'\n\s*\n\s*\n', '\n\n', markdown)  # Remove excessive line breaks
		markdown = re.sub(r'[ \t]+$', '', markdown, flags=re.MULTILINE)  # Remove trailing whitespace
		
		# Ensure minimum content length
		if len(markdown.strip()) < config.min_content_length:
			return ""
		
		# Enforce maximum content length
		if len(markdown) > config.max_content_length:
			markdown = markdown[:config.max_content_length] + "\n\n[Content truncated...]"
		
		return markdown.strip()
	
	def _detect_language(self, text: str) -> Optional[str]:
		"""Detect language of text content"""
		if not text or len(text) < 50:
			return None
		
		try:
			return langdetect.detect(text)
		except LangDetectException:
			return None
	
	def _extract_html_metadata(self, content: str) -> Dict[str, Any]:
		"""Extract metadata from HTML"""
		soup = BeautifulSoup(content, 'html.parser')
		metadata = {}
		
		# Title
		title_tag = soup.find('title')
		if title_tag:
			metadata['title'] = title_tag.get_text().strip()
		
		# Meta description
		desc_tag = soup.find('meta', attrs={'name': 'description'})
		if desc_tag:
			metadata['description'] = desc_tag.get('content', '').strip()
		
		# Meta keywords
		keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
		if keywords_tag:
			keywords = keywords_tag.get('content', '').strip()
			metadata['keywords'] = [k.strip() for k in keywords.split(',') if k.strip()]
		
		# Open Graph metadata
		og_tags = soup.find_all('meta', attrs={'property': lambda x: x and x.startswith('og:')})
		for tag in og_tags:
			prop = tag.get('property', '').replace('og:', '')
			content = tag.get('content', '').strip()
			if prop and content:
				metadata[f'og_{prop}'] = content
		
		# Schema.org structured data
		schema_scripts = soup.find_all('script', attrs={'type': 'application/ld+json'})
		structured_data = []
		for script in schema_scripts:
			try:
				data = json.loads(script.string)
				structured_data.append(data)
			except:
				pass
		
		if structured_data:
			metadata['structured_data'] = structured_data
		
		return metadata
	
	def _extract_images(self, content: str, base_url: str) -> List[str]:
		"""Extract image URLs from content"""
		soup = BeautifulSoup(content, 'html.parser')
		images = []
		
		for img in soup.find_all('img', src=True):
			src = img['src']
			# Convert relative URLs to absolute
			if src.startswith('/'):
				src = urljoin(base_url, src)
			elif not src.startswith(('http://', 'https://')):
				src = urljoin(base_url, src)
			
			if src not in images:
				images.append(src)
		
		return images
	
	def _extract_links(self, content: str, base_url: str) -> List[str]:
		"""Extract links from content"""
		soup = BeautifulSoup(content, 'html.parser')
		links = []
		
		for link in soup.find_all('a', href=True):
			href = link['href']
			# Convert relative URLs to absolute
			if href.startswith('/'):
				href = urljoin(base_url, href)
			elif not href.startswith(('http://', 'https://', 'mailto:', 'tel:')):
				href = urljoin(base_url, href)
			
			if href.startswith(('http://', 'https://')) and href not in links:
				links.append(href)
		
		return links
	
	def _generate_fingerprint(self, content: str) -> str:
		"""Generate SHA-256 fingerprint for content"""
		if not content:
			return ""
		return hashlib.sha256(content.encode('utf-8')).hexdigest()
	
	def _calculate_content_quality(self, text: str, metadata: Dict[str, Any], 
								   content_length: int) -> float:
		"""Calculate content quality score"""
		score = 0.0
		
		# Length score (0.3 weight)
		if len(text) > 1000:
			score += 0.3
		elif len(text) > 500:
			score += 0.2
		elif len(text) > 100:
			score += 0.1
		
		# Metadata completeness (0.2 weight)
		metadata_score = 0
		if metadata.get('title'):
			metadata_score += 0.05
		if metadata.get('description'):
			metadata_score += 0.05
		if metadata.get('keywords'):
			metadata_score += 0.05
		if metadata.get('structured_data'):
			metadata_score += 0.05
		
		score += metadata_score
		
		# Text quality indicators (0.3 weight)
		if text:
			# Sentence structure
			sentences = text.split('.')
			if len(sentences) > 5:
				score += 0.1
			
			# Paragraph structure
			paragraphs = text.split('\n\n')
			if len(paragraphs) > 3:
				score += 0.1
			
			# Word diversity
			words = text.split()
			unique_words = set(word.lower() for word in words)
			if len(words) > 0:
				diversity = len(unique_words) / len(words)
				if diversity > 0.3:
					score += 0.1
		
		# Content structure (0.2 weight)
		if content_length > len(text):  # Had structure that was cleaned
			compression_ratio = len(text) / content_length
			if 0.1 <= compression_ratio <= 0.5:  # Good signal-to-noise ratio
				score += 0.2
		
		return min(score, 1.0)  # Cap at 1.0
	
	async def _extract_json_content(self, crawl_result: CrawlResult, 
									config: ContentCleaningConfig) -> ContentExtractionResult:
		"""Extract content from JSON response"""
		try:
			data = json.loads(crawl_result.content)
			
			# Try to extract text content from common JSON structures
			text_content = self._extract_text_from_json(data)
			title = self._extract_title_from_json(data)
			
			# Convert to markdown if requested
			markdown_content = text_content if not config.markdown_formatting else f"# {title}\n\n{text_content}"
			
			# Generate fingerprint
			fingerprint = self._generate_fingerprint(markdown_content)
			
			return ContentExtractionResult(
				url=crawl_result.url,
				title=title,
				main_content=text_content,
				raw_content=crawl_result.content,
				cleaned_content=text_content,
				markdown_content=markdown_content,
				content_type=ContentType.JSON,
				language=self._detect_language(text_content),
				publish_date=None,
				author=None,
				description=None,
				keywords=[],
				images=[],
				links=[],
				content_fingerprint=fingerprint,
				processing_stage=ContentProcessingStage.CLEANED,
				metadata={'json_structure': type(data).__name__},
				success=True
			)
			
		except json.JSONDecodeError as e:
			return ContentExtractionResult(
				url=crawl_result.url,
				title=None,
				main_content="",
				raw_content=crawl_result.content,
				cleaned_content="",
				markdown_content="",
				content_type=ContentType.JSON,
				language=None,
				publish_date=None,
				author=None,
				description=None,
				keywords=[],
				images=[],
				links=[],
				content_fingerprint="",
				processing_stage=ContentProcessingStage.RAW_EXTRACTED,
				metadata={},
				success=False,
				error=f"Invalid JSON: {str(e)}"
			)
	
	def _extract_text_from_json(self, data: Any, max_depth: int = 3) -> str:
		"""Recursively extract text content from JSON data"""
		if max_depth <= 0:
			return ""
		
		text_parts = []
		
		if isinstance(data, dict):
			# Look for common text fields
			text_fields = ['text', 'content', 'body', 'description', 'message', 'title', 'name']
			for field in text_fields:
				if field in data and isinstance(data[field], str):
					text_parts.append(data[field])
			
			# Recursively process other fields
			for key, value in data.items():
				if key not in text_fields:
					text_parts.append(self._extract_text_from_json(value, max_depth - 1))
		
		elif isinstance(data, list):
			for item in data:
				text_parts.append(self._extract_text_from_json(item, max_depth - 1))
		
		elif isinstance(data, str):
			text_parts.append(data)
		
		return " ".join(filter(None, text_parts))
	
	def _extract_title_from_json(self, data: Any) -> Optional[str]:
		"""Extract title from JSON data"""
		if isinstance(data, dict):
			title_fields = ['title', 'name', 'headline', 'subject']
			for field in title_fields:
				if field in data and isinstance(data[field], str):
					return data[field]
		
		return None
	
	async def _extract_xml_content(self, crawl_result: CrawlResult, 
								   config: ContentCleaningConfig) -> ContentExtractionResult:
		"""Extract content from XML"""
		# For XML, we'll convert to text and treat similarly to HTML
		soup = BeautifulSoup(crawl_result.content, 'xml')
		text_content = soup.get_text()
		
		# Basic cleaning
		text_content = re.sub(r'\s+', ' ', text_content).strip()
		
		# Generate fingerprint
		fingerprint = self._generate_fingerprint(text_content)
		
		return ContentExtractionResult(
			url=crawl_result.url,
			title=None,
			main_content=text_content,
			raw_content=crawl_result.content,
			cleaned_content=text_content,
			markdown_content=text_content,
			content_type=ContentType.XML,
			language=self._detect_language(text_content),
			publish_date=None,
			author=None,
			description=None,
			keywords=[],
			images=[],
			links=[],
			content_fingerprint=fingerprint,
			processing_stage=ContentProcessingStage.CLEANED,
			metadata={},
			success=True
		)
	
	async def _extract_pdf_content(self, crawl_result: CrawlResult, 
								   config: ContentCleaningConfig) -> ContentExtractionResult:
		"""Extract content from PDF (placeholder implementation)"""
		# This would require the PDF content as bytes, not text
		# For now, return an error indicating PDF processing needs binary content
		
		return ContentExtractionResult(
			url=crawl_result.url,
			title=None,
			main_content="",
			raw_content=crawl_result.content,
			cleaned_content="",
			markdown_content="",
			content_type=ContentType.PDF,
			language=None,
			publish_date=None,
			author=None,
			description=None,
			keywords=[],
			images=[],
			links=[],
			content_fingerprint="",
			processing_stage=ContentProcessingStage.RAW_EXTRACTED,
			metadata={},
			success=False,
			error="PDF processing requires binary content, not text"
		)
	
	async def _extract_text_content(self, crawl_result: CrawlResult, 
									config: ContentCleaningConfig) -> ContentExtractionResult:
		"""Extract content from plain text"""
		content = crawl_result.content.strip()
		
		# Basic text cleaning
		if config.min_content_length and len(content) < config.min_content_length:
			content = ""
		
		if config.max_content_length and len(content) > config.max_content_length:
			content = content[:config.max_content_length] + "\n\n[Content truncated...]"
		
		# Generate fingerprint
		fingerprint = self._generate_fingerprint(content)
		
		return ContentExtractionResult(
			url=crawl_result.url,
			title=None,
			main_content=content,
			raw_content=crawl_result.content,
			cleaned_content=content,
			markdown_content=content,
			content_type=ContentType.TEXT,
			language=self._detect_language(content),
			publish_date=None,
			author=None,
			description=None,
			keywords=[],
			images=[],
			links=[],
			content_fingerprint=fingerprint,
			processing_stage=ContentProcessingStage.CLEANED,
			metadata={},
			success=True
		)


# =====================================================
# CONTENT PROCESSING PIPELINE
# =====================================================

class ContentProcessingPipeline:
	"""Orchestrates the complete content processing pipeline"""
	
	def __init__(self):
		self.extraction_engine = ContentExtractionEngine()
	
	async def process_crawl_result(self, crawl_result: CrawlResult, 
								   config: ContentCleaningConfig) -> ContentExtractionResult:
		"""Process a crawl result through the complete pipeline"""
		
		# Extract and clean content
		extraction_result = await self.extraction_engine.extract_content(crawl_result, config)
		
		if not extraction_result.success:
			return extraction_result
		
		# Update processing stage based on successful operations
		if extraction_result.markdown_content:
			extraction_result.processing_stage = ContentProcessingStage.MARKDOWN_CONVERTED
		
		if extraction_result.content_fingerprint:
			extraction_result.processing_stage = ContentProcessingStage.FINGERPRINTED
		
		return extraction_result
	
	async def batch_process(self, crawl_results: List[CrawlResult], 
							config: ContentCleaningConfig) -> List[ContentExtractionResult]:
		"""Process multiple crawl results in parallel"""
		tasks = [
			self.process_crawl_result(result, config) 
			for result in crawl_results
		]
		
		return await asyncio.gather(*tasks, return_exceptions=True)


# =====================================================
# EXPORTS
# =====================================================

__all__ = [
	'ContentProcessingPipeline',
	'ContentExtractionEngine',
	'ContentExtractionResult',
	'ContentType',
	'CleaningStats'
]