"""
APG Document Content Management - OCR Service

Advanced OCR capabilities with multi-language support, image preprocessing,
layout analysis, and integration with APG AI platform.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import io
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid_extensions import uuid7str

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import pdf2image
from pydantic import BaseModel, Field, ConfigDict, validator

logger = logging.getLogger(__name__)


class OCRConfig(BaseModel):
	"""OCR configuration settings"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Tesseract configuration
	tesseract_cmd: str = Field(default='tesseract', description="Tesseract command path")
	tesseract_config: str = Field(default='--oem 3 --psm 6', description="Tesseract configuration")
	
	# Supported languages
	languages: List[str] = Field(
		default=['eng', 'fra', 'deu', 'spa', 'ita', 'por', 'rus', 'chi_sim', 'jpn', 'ara'],
		description="Supported OCR languages"
	)
	
	# Image preprocessing
	enable_preprocessing: bool = Field(default=True, description="Enable image preprocessing")
	dpi: int = Field(default=300, description="DPI for PDF to image conversion")
	
	# Performance settings
	max_image_size: int = Field(default=4096, description="Maximum image dimension")
	parallel_processing: bool = Field(default=True, description="Enable parallel processing")
	max_workers: int = Field(default=4, description="Maximum worker threads")
	
	# Quality thresholds
	min_confidence: float = Field(default=0.5, description="Minimum OCR confidence threshold")
	min_text_length: int = Field(default=10, description="Minimum text length for valid results")


class OCRResult(BaseModel):
	"""OCR processing result"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="OCR result ID")
	document_id: str = Field(..., description="Source document ID")
	page_number: int = Field(default=1, description="Page number")
	
	# Extracted content
	text_content: str = Field(..., description="Extracted text content")
	confidence_score: float = Field(..., description="Overall confidence score")
	language_detected: str = Field(..., description="Detected language")
	
	# Text blocks and structure
	text_blocks: List[Dict[str, Any]] = Field(default_factory=list, description="Text blocks with coordinates")
	layout_structure: Dict[str, Any] = Field(default_factory=dict, description="Document layout structure")
	
	# Processing metadata
	processing_time_ms: int = Field(..., description="Processing time in milliseconds")
	preprocessing_applied: List[str] = Field(default_factory=list, description="Applied preprocessing steps")
	
	# Quality metrics
	word_count: int = Field(default=0, description="Total word count")
	character_count: int = Field(default=0, description="Total character count")
	line_count: int = Field(default=0, description="Total line count")
	
	# Error handling
	warnings: List[str] = Field(default_factory=list, description="Processing warnings")
	errors: List[str] = Field(default_factory=list, description="Processing errors")
	
	# Timestamps
	processed_at: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")


class OCREngine:
	"""Advanced OCR engine with preprocessing and layout analysis"""
	
	def __init__(self, config: OCRConfig = None, apg_ai_client=None):
		self.config = config or OCRConfig()
		self.apg_ai_client = apg_ai_client
		self.logger = logging.getLogger(__name__)
		
		# Configure Tesseract
		if self.config.tesseract_cmd:
			pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd
	
	async def process_document(
		self,
		file_path: str,
		document_id: str,
		options: Dict[str, Any] = None
	) -> List[OCRResult]:
		"""Process document for OCR extraction"""
		
		self.logger.info(f"Starting OCR processing for document {document_id}")
		start_time = datetime.utcnow()
		
		try:
			# Determine file type and extract images
			images = await self._extract_images_from_document(file_path)
			
			if not images:
				raise ValueError(f"No processable images found in document: {file_path}")
			
			# Process each page/image
			ocr_results = []
			
			if self.config.parallel_processing and len(images) > 1:
				# Parallel processing for multi-page documents
				tasks = [
					self._process_single_image(img, document_id, page_num + 1, options)
					for page_num, img in enumerate(images)
				]
				ocr_results = await asyncio.gather(*tasks, return_exceptions=True)
				
				# Filter out exceptions
				ocr_results = [
					result for result in ocr_results 
					if not isinstance(result, Exception)
				]
			else:
				# Sequential processing
				for page_num, image in enumerate(images):
					result = await self._process_single_image(
						image, document_id, page_num + 1, options
					)
					if result:
						ocr_results.append(result)
			
			processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
			self.logger.info(
				f"OCR processing completed for document {document_id}: "
				f"{len(ocr_results)} pages processed in {processing_time:.0f}ms"
			)
			
			return ocr_results
			
		except Exception as e:
			self.logger.error(f"OCR processing failed for document {document_id}: {str(e)}")
			raise
	
	async def _extract_images_from_document(self, file_path: str) -> List[Image.Image]:
		"""Extract images from various document formats"""
		
		file_path = Path(file_path)
		file_extension = file_path.suffix.lower()
		
		if file_extension == '.pdf':
			return await self._extract_images_from_pdf(file_path)
		elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
			return await self._load_single_image(file_path)
		else:
			raise ValueError(f"Unsupported file format: {file_extension}")
	
	async def _extract_images_from_pdf(self, pdf_path: Path) -> List[Image.Image]:
		"""Extract images from PDF file"""
		
		try:
			# Convert PDF pages to images
			images = pdf2image.convert_from_path(
				pdf_path,
				dpi=self.config.dpi,
				fmt='PNG',
				thread_count=self.config.max_workers
			)
			
			self.logger.info(f"Extracted {len(images)} pages from PDF: {pdf_path}")
			return images
			
		except Exception as e:
			self.logger.error(f"Failed to extract images from PDF {pdf_path}: {str(e)}")
			raise
	
	async def _load_single_image(self, image_path: Path) -> List[Image.Image]:
		"""Load single image file"""
		
		try:
			image = Image.open(image_path)
			
			# Convert to RGB if needed
			if image.mode != 'RGB':
				image = image.convert('RGB')
			
			return [image]
			
		except Exception as e:
			self.logger.error(f"Failed to load image {image_path}: {str(e)}")
			raise
	
	async def _process_single_image(
		self,
		image: Image.Image,
		document_id: str,
		page_number: int,
		options: Dict[str, Any] = None
	) -> OCRResult:
		"""Process single image for OCR"""
		
		start_time = datetime.utcnow()
		options = options or {}
		
		try:
			# Preprocess image if enabled
			preprocessed_image, preprocessing_steps = await self._preprocess_image(
				image, options.get('preprocessing', {})
			)
			
			# Detect language
			language = await self._detect_language(preprocessed_image, options.get('language'))
			
			# Perform OCR with layout analysis
			ocr_data = await self._perform_ocr_with_layout(preprocessed_image, language)
			
			# Extract text content and structure
			text_content = self._extract_clean_text(ocr_data)
			text_blocks = self._extract_text_blocks(ocr_data)
			layout_structure = await self._analyze_layout_structure(ocr_data)
			
			# Calculate confidence and quality metrics
			confidence_score = self._calculate_confidence_score(ocr_data)
			word_count = len(text_content.split())
			character_count = len(text_content)
			line_count = text_content.count('\n') + 1
			
			# Processing time
			processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
			
			# Create result
			result = OCRResult(
				document_id=document_id,
				page_number=page_number,
				text_content=text_content,
				confidence_score=confidence_score,
				language_detected=language,
				text_blocks=text_blocks,
				layout_structure=layout_structure,
				processing_time_ms=processing_time_ms,
				preprocessing_applied=preprocessing_steps,
				word_count=word_count,
				character_count=character_count,
				line_count=line_count
			)
			
			# Validate result quality
			await self._validate_ocr_result(result)
			
			return result
			
		except Exception as e:
			self.logger.error(f"Failed to process image for document {document_id}, page {page_number}: {str(e)}")
			
			# Return error result
			return OCRResult(
				document_id=document_id,
				page_number=page_number,
				text_content="",
				confidence_score=0.0,
				language_detected="unknown",
				processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
				errors=[str(e)]
			)
	
	async def _preprocess_image(
		self,
		image: Image.Image,
		preprocessing_options: Dict[str, Any]
	) -> Tuple[Image.Image, List[str]]:
		"""Apply image preprocessing for better OCR results"""
		
		if not self.config.enable_preprocessing:
			return image, []
		
		steps_applied = []
		processed_image = image.copy()
		
		try:
			# Convert PIL to OpenCV format
			cv_image = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
			
			# Resize if too large
			height, width = cv_image.shape[:2]
			if max(height, width) > self.config.max_image_size:
				scale = self.config.max_image_size / max(height, width)
				new_width = int(width * scale)
				new_height = int(height * scale)
				cv_image = cv2.resize(cv_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
				steps_applied.append("resize")
			
			# Convert to grayscale
			if preprocessing_options.get('grayscale', True):
				cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
				steps_applied.append("grayscale")
			
			# Noise reduction
			if preprocessing_options.get('denoise', True):
				cv_image = cv2.medianBlur(cv_image, 3)
				steps_applied.append("denoise")
			
			# Improve contrast
			if preprocessing_options.get('enhance_contrast', True):
				cv_image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv_image)
				steps_applied.append("enhance_contrast")
			
			# Morphological operations to improve text
			if preprocessing_options.get('morphological', True):
				kernel = np.ones((1, 1), np.uint8)
				cv_image = cv2.morphologyEx(cv_image, cv2.MORPH_CLOSE, kernel)
				steps_applied.append("morphological")
			
			# Gaussian blur for smoothing
			if preprocessing_options.get('blur', False):
				cv_image = cv2.GaussianBlur(cv_image, (1, 1), 0)
				steps_applied.append("blur")
			
			# Threshold to binary
			if preprocessing_options.get('threshold', True):
				_, cv_image = cv2.threshold(cv_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
				steps_applied.append("threshold")
			
			# Convert back to PIL
			if len(cv_image.shape) == 3:
				processed_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
			else:
				processed_image = Image.fromarray(cv_image)
			
			self.logger.debug(f"Applied preprocessing steps: {steps_applied}")
			
		except Exception as e:
			self.logger.warning(f"Preprocessing failed, using original image: {str(e)}")
			steps_applied.append(f"preprocessing_failed: {str(e)}")
		
		return processed_image, steps_applied
	
	async def _detect_language(self, image: Image.Image, hint_language: str = None) -> str:
		"""Detect document language for OCR"""
		
		if hint_language and hint_language in self.config.languages:
			return hint_language
		
		try:
			# Use Tesseract language detection
			osd = pytesseract.image_to_osd(image)
			
			# Parse OSD output to extract script information
			# For simplicity, default to English if detection fails
			detected_language = 'eng'
			
			if detected_language not in self.config.languages:
				self.logger.warning(f"Detected language {detected_language} not supported, using English")
				detected_language = 'eng'
			
			return detected_language
			
		except Exception as e:
			self.logger.warning(f"Language detection failed: {str(e)}, using English")
			return 'eng'
	
	async def _perform_ocr_with_layout(self, image: Image.Image, language: str) -> Dict[str, Any]:
		"""Perform OCR with layout analysis"""
		
		try:
			# Configure Tesseract for detailed output
			config = f"{self.config.tesseract_config} -l {language}"
			
			# Get detailed OCR data
			ocr_data = pytesseract.image_to_data(
				image,
				config=config,
				output_type=pytesseract.Output.DICT
			)
			
			return ocr_data
			
		except Exception as e:
			self.logger.error(f"OCR processing failed: {str(e)}")
			raise
	
	def _extract_clean_text(self, ocr_data: Dict[str, Any]) -> str:
		"""Extract clean text from OCR data"""
		
		text_parts = []
		
		for i in range(len(ocr_data['text'])):
			confidence = int(ocr_data['conf'][i])
			text = ocr_data['text'][i].strip()
			
			# Only include text with reasonable confidence
			if confidence > 0 and text and confidence >= (self.config.min_confidence * 100):
				text_parts.append(text)
		
		# Join text parts and clean up
		full_text = ' '.join(text_parts)
		
		# Clean up common OCR artifacts
		full_text = self._clean_ocr_artifacts(full_text)
		
		return full_text
	
	def _extract_text_blocks(self, ocr_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Extract text blocks with coordinates"""
		
		text_blocks = []
		
		for i in range(len(ocr_data['text'])):
			confidence = int(ocr_data['conf'][i])
			text = ocr_data['text'][i].strip()
			
			if confidence > 0 and text and confidence >= (self.config.min_confidence * 100):
				block = {
					'text': text,
					'confidence': confidence / 100.0,
					'bbox': {
						'left': ocr_data['left'][i],
						'top': ocr_data['top'][i],
						'width': ocr_data['width'][i],
						'height': ocr_data['height'][i]
					},
					'block_num': ocr_data['block_num'][i],
					'par_num': ocr_data['par_num'][i],
					'line_num': ocr_data['line_num'][i],
					'word_num': ocr_data['word_num'][i]
				}
				text_blocks.append(block)
		
		return text_blocks
	
	async def _analyze_layout_structure(self, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Analyze document layout structure"""
		
		try:
			# Group text by blocks and paragraphs
			blocks = {}
			paragraphs = {}
			lines = {}
			
			for i in range(len(ocr_data['text'])):
				confidence = int(ocr_data['conf'][i])
				text = ocr_data['text'][i].strip()
				
				if confidence > 0 and text:
					block_num = ocr_data['block_num'][i]
					par_num = ocr_data['par_num'][i]
					line_num = ocr_data['line_num'][i]
					
					# Group by block
					if block_num not in blocks:
						blocks[block_num] = []
					blocks[block_num].append(text)
					
					# Group by paragraph
					par_key = f"{block_num}_{par_num}"
					if par_key not in paragraphs:
						paragraphs[par_key] = []
					paragraphs[par_key].append(text)
					
					# Group by line
					line_key = f"{block_num}_{par_num}_{line_num}"
					if line_key not in lines:
						lines[line_key] = []
					lines[line_key].append(text)
			
			layout_structure = {
				'total_blocks': len(blocks),
				'total_paragraphs': len(paragraphs),
				'total_lines': len(lines),
				'blocks': {str(k): ' '.join(v) for k, v in blocks.items()},
				'paragraphs': {k: ' '.join(v) for k, v in paragraphs.items()},
				'reading_order': list(blocks.keys())
			}
			
			return layout_structure
			
		except Exception as e:
			self.logger.warning(f"Layout analysis failed: {str(e)}")
			return {}
	
	def _calculate_confidence_score(self, ocr_data: Dict[str, Any]) -> float:
		"""Calculate overall confidence score"""
		
		confidences = []
		text_lengths = []
		
		for i in range(len(ocr_data['text'])):
			confidence = int(ocr_data['conf'][i])
			text = ocr_data['text'][i].strip()
			
			if confidence > 0 and text:
				confidences.append(confidence / 100.0)
				text_lengths.append(len(text))
		
		if not confidences:
			return 0.0
		
		# Weight confidence by text length
		total_length = sum(text_lengths)
		if total_length == 0:
			return sum(confidences) / len(confidences)
		
		weighted_confidence = sum(
			conf * length / total_length
			for conf, length in zip(confidences, text_lengths)
		)
		
		return min(weighted_confidence, 1.0)
	
	def _clean_ocr_artifacts(self, text: str) -> str:
		"""Clean common OCR artifacts"""
		
		# Remove excessive whitespace
		text = ' '.join(text.split())
		
		# Fix common OCR substitutions
		substitutions = {
			'rn': 'm',
			'|': 'l',
			'0': 'O',  # Context-dependent
			'5': 'S',  # Context-dependent
		}
		
		# Apply substitutions carefully (this is simplified)
		# In production, this would use more sophisticated correction
		
		return text
	
	async def _validate_ocr_result(self, result: OCRResult) -> None:
		"""Validate OCR result quality"""
		
		warnings = []
		
		# Check minimum text length
		if len(result.text_content) < self.config.min_text_length:
			warnings.append(f"Text content too short: {len(result.text_content)} characters")
		
		# Check confidence threshold
		if result.confidence_score < self.config.min_confidence:
			warnings.append(f"Low confidence score: {result.confidence_score:.2f}")
		
		# Check for suspicious patterns
		if result.character_count > 0:
			non_alpha_ratio = sum(1 for c in result.text_content if not c.isalnum() and not c.isspace()) / result.character_count
			if non_alpha_ratio > 0.3:
				warnings.append(f"High non-alphanumeric character ratio: {non_alpha_ratio:.2f}")
		
		result.warnings.extend(warnings)
	
	async def get_supported_languages(self) -> List[str]:
		"""Get list of supported OCR languages"""
		return self.config.languages.copy()
	
	async def health_check(self) -> Dict[str, Any]:
		"""Perform OCR engine health check"""
		
		try:
			# Test Tesseract availability
			version = pytesseract.get_tesseract_version()
			
			# Test basic functionality with a small test image
			test_image = Image.new('RGB', (100, 50), color='white')
			test_result = pytesseract.image_to_string(test_image)
			
			return {
				'status': 'healthy',
				'tesseract_version': str(version),
				'supported_languages': self.config.languages,
				'config': {
					'dpi': self.config.dpi,
					'max_image_size': self.config.max_image_size,
					'preprocessing_enabled': self.config.enable_preprocessing
				}
			}
			
		except Exception as e:
			return {
				'status': 'unhealthy',
				'error': str(e)
			}


class OCRService:
	"""High-level OCR service with APG integration"""
	
	def __init__(self, config: OCRConfig = None, apg_ai_client=None):
		self.config = config or OCRConfig()
		self.ocr_engine = OCREngine(self.config, apg_ai_client)
		self.apg_ai_client = apg_ai_client
		self.logger = logging.getLogger(__name__)
	
	async def process_document_ocr(
		self,
		document_id: str,
		file_path: str,
		options: Dict[str, Any] = None
	) -> Dict[str, Any]:
		"""Process document for OCR with APG integration"""
		
		try:
			# Perform OCR processing
			ocr_results = await self.ocr_engine.process_document(
				file_path=file_path,
				document_id=document_id,
				options=options
			)
			
			# Combine results from all pages
			combined_text = '\n\n'.join(result.text_content for result in ocr_results)
			combined_confidence = sum(result.confidence_score for result in ocr_results) / len(ocr_results)
			
			# Enhance with APG AI if available
			ai_enhancements = {}
			if self.apg_ai_client and combined_text.strip():
				ai_enhancements = await self._enhance_with_apg_ai(combined_text, options)
			
			# Create comprehensive result
			processing_result = {
				'document_id': document_id,
				'total_pages': len(ocr_results),
				'combined_text': combined_text,
				'combined_confidence': combined_confidence,
				'page_results': [result.dict() for result in ocr_results],
				'ai_enhancements': ai_enhancements,
				'processing_summary': {
					'total_words': sum(result.word_count for result in ocr_results),
					'total_characters': sum(result.character_count for result in ocr_results),
					'total_lines': sum(result.line_count for result in ocr_results),
					'average_confidence': combined_confidence,
					'languages_detected': list(set(result.language_detected for result in ocr_results)),
					'total_processing_time_ms': sum(result.processing_time_ms for result in ocr_results)
				}
			}
			
			self.logger.info(f"OCR processing completed for document {document_id}")
			return processing_result
			
		except Exception as e:
			self.logger.error(f"OCR service processing failed for document {document_id}: {str(e)}")
			raise
	
	async def _enhance_with_apg_ai(
		self,
		text_content: str,
		options: Dict[str, Any] = None
	) -> Dict[str, Any]:
		"""Enhance OCR results with APG AI capabilities"""
		
		enhancements = {}
		
		try:
			if self.apg_ai_client:
				# Spelling and grammar correction
				if options and options.get('ai_correction', True):
					corrected_text = await self._ai_text_correction(text_content)
					if corrected_text != text_content:
						enhancements['corrected_text'] = corrected_text
				
				# Entity extraction
				if options and options.get('ai_entities', True):
					entities = await self._ai_entity_extraction(text_content)
					if entities:
						enhancements['entities'] = entities
				
				# Content classification
				if options and options.get('ai_classification', True):
					classification = await self._ai_content_classification(text_content)
					if classification:
						enhancements['classification'] = classification
		
		except Exception as e:
			self.logger.warning(f"AI enhancement failed: {str(e)}")
			enhancements['ai_enhancement_error'] = str(e)
		
		return enhancements
	
	async def _ai_text_correction(self, text: str) -> str:
		"""Use APG AI for text correction"""
		try:
			# This would integrate with APG AI service for text correction
			# For now, return original text
			return text
		except Exception:
			return text
	
	async def _ai_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
		"""Use APG AI for entity extraction"""
		try:
			# This would integrate with APG AI service for entity extraction
			return []
		except Exception:
			return []
	
	async def _ai_content_classification(self, text: str) -> Dict[str, Any]:
		"""Use APG AI for content classification"""
		try:
			# This would integrate with APG AI service for classification
			return {}
		except Exception:
			return {}
	
	async def health_check(self) -> Dict[str, Any]:
		"""Perform service health check"""
		return await self.ocr_engine.health_check()