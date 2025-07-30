"""
Natural Language Visual Query Interface - Revolutionary Conversational Computer Vision

Advanced natural language interface that enables users to ask questions about images
in plain English and receive intelligent, contextual responses with visual evidence
and interactive follow-up suggestions.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid_extensions import uuid7str

import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from transformers import (
	pipeline, AutoTokenizer, AutoModel, BlipProcessor, BlipForConditionalGeneration
)
import torch
from sentence_transformers import SentenceTransformer
import spacy

from .models import CVBaseModel, ProcessingType, AnalysisLevel


class QueryIntent(CVBaseModel):
	"""Parsed natural language query intent"""
	
	intent_type: str = Field(..., description="Type of query intent")
	primary_action: str = Field(..., description="Primary action requested")
	target_objects: List[str] = Field(
		default_factory=list, description="Objects mentioned in query"
	)
	attributes: List[str] = Field(
		default_factory=list, description="Attributes to analyze"
	)
	constraints: Dict[str, Any] = Field(
		default_factory=dict, description="Query constraints and filters"
	)
	confidence: float = Field(..., ge=0.0, le=1.0, description="Intent parsing confidence")
	language: str = Field(default="en", description="Query language")


class AnalysisPipeline(CVBaseModel):
	"""Analysis pipeline configuration based on query intent"""
	
	required_processors: List[str] = Field(..., description="Required processing modules")
	optional_processors: List[str] = Field(
		default_factory=list, description="Optional processing modules"
	)
	processing_order: List[str] = Field(..., description="Order of processing operations")
	parameters: Dict[str, Any] = Field(
		default_factory=dict, description="Processing parameters"
	)
	expected_outputs: List[str] = Field(..., description="Expected output types")


class NaturalLanguageResponse(CVBaseModel):
	"""Natural language response to user query"""
	
	answer: str = Field(..., description="Main answer to user query")
	confidence: float = Field(..., ge=0.0, le=1.0, description="Response confidence")
	supporting_evidence: Dict[str, Any] = Field(
		default_factory=dict, description="Visual evidence supporting the answer"
	)
	follow_up_suggestions: List[str] = Field(
		default_factory=list, description="Suggested follow-up questions"
	)
	query_interpretation: str = Field(..., description="How the query was interpreted")
	response_type: str = Field(..., description="Type of response (factual, analytical, etc.)")
	visual_highlights: List[Dict[str, Any]] = Field(
		default_factory=list, description="Visual elements to highlight in response"
	)


class ConversationContext(CVBaseModel):
	"""Conversation context for multi-turn interactions"""
	
	conversation_id: str = Field(default_factory=uuid7str, description="Conversation identifier")
	previous_queries: List[str] = Field(
		default_factory=list, description="Previous queries in conversation"
	)
	previous_responses: List[str] = Field(
		default_factory=list, description="Previous responses in conversation"
	)
	image_context: Dict[str, Any] = Field(
		default_factory=dict, description="Context about current image"
	)
	user_preferences: Dict[str, Any] = Field(
		default_factory=dict, description="User interaction preferences"
	)
	conversation_state: str = Field(
		default="active", description="Current conversation state"
	)


class NaturalLanguageVisualQuery:
	"""
	Revolutionary Natural Language Visual Query Interface
	
	Enables users to interact with computer vision systems using natural language,
	making advanced visual analysis accessible to non-technical users through
	conversational interfaces with intelligent context understanding.
	"""
	
	def __init__(self):
		self.nlp_model = None
		self.vision_language_model = None
		self.sentence_transformer = None
		self.spacy_nlp = None
		
		# Intent classification
		self.intent_classifier = None
		self.intent_patterns = {}
		
		# Query processing
		self.query_cache: Dict[str, Any] = {}
		self.conversation_contexts: Dict[str, ConversationContext] = {}
		
		# Response generation
		self.response_templates = {}
		self.follow_up_generators = {}
	
	async def _log_query_operation(
		self,
		operation: str,
		query_id: Optional[str] = None,
		details: Optional[str] = None
	) -> None:
		"""Log natural language query operations"""
		assert operation is not None, "Operation name must be provided"
		query_ref = f" [Query: {query_id}]" if query_id else ""
		detail_info = f" - {details}" if details else ""
		print(f"Natural Language Query: {operation}{query_ref}{detail_info}")
	
	async def initialize_query_interface(self) -> bool:
		"""
		Initialize the natural language query interface
		
		Returns:
			bool: Success status of initialization
		"""
		try:
			await self._log_query_operation("Initializing natural language query interface")
			
			# Initialize NLP models
			await self._initialize_nlp_models()
			
			# Initialize vision-language models
			await self._initialize_vision_language_models()
			
			# Setup intent classification
			await self._setup_intent_classification()
			
			# Initialize response generation
			await self._initialize_response_generation()
			
			# Setup conversation management
			await self._setup_conversation_management()
			
			await self._log_query_operation("Natural language query interface initialized successfully")
			return True
			
		except Exception as e:
			await self._log_query_operation(
				"Failed to initialize natural language query interface",
				details=str(e)
			)
			return False
	
	async def _initialize_nlp_models(self) -> None:
		"""Initialize NLP models for query understanding"""
		try:
			# Load spaCy model for advanced NLP
			self.spacy_nlp = spacy.load("en_core_web_sm")
			
			# Load sentence transformer for semantic similarity
			self.sentence_transformer = SentenceTransformer(
				'sentence-transformers/all-MiniLM-L6-v2'
			)
			
			# Load question answering model
			self.nlp_model = pipeline(
				"question-answering",
				model="deepset/roberta-base-squad2",
				return_all_scores=True
			)
			
		except Exception as e:
			raise RuntimeError(f"Failed to initialize NLP models: {e}")
	
	async def _initialize_vision_language_models(self) -> None:
		"""Initialize vision-language models for multimodal understanding"""
		try:
			# Load BLIP model for image captioning and VQA
			self.blip_processor = BlipProcessor.from_pretrained(
				"Salesforce/blip-image-captioning-base"
			)
			self.vision_language_model = BlipForConditionalGeneration.from_pretrained(
				"Salesforce/blip-image-captioning-base"
			)
			
		except Exception as e:
			raise RuntimeError(f"Failed to initialize vision-language models: {e}")
	
	async def _setup_intent_classification(self) -> None:
		"""Setup intent classification patterns and models"""
		# Define intent patterns for different query types
		self.intent_patterns = {
			"object_detection": {
				"patterns": [
					r"what.*see",
					r"identify.*object",
					r"what.*in.*image",
					r"detect.*object",
					r"find.*object",
					r"how many.*"
				],
				"keywords": ["what", "identify", "detect", "find", "objects", "see"]
			},
			"quality_assessment": {
				"patterns": [
					r"quality.*good",
					r"defect.*present",
					r"problem.*with",
					r"issue.*found",
					r"acceptable.*quality"
				],
				"keywords": ["quality", "defect", "problem", "issue", "good", "bad"]
			},
			"text_extraction": {
				"patterns": [
					r"read.*text",
					r"extract.*text",
					r"what.*text",
					r"ocr.*extract",
					r"words.*in.*image"
				],
				"keywords": ["text", "read", "extract", "words", "ocr", "document"]
			},
			"measurement": {
				"patterns": [
					r"measure.*dimension",
					r"size.*of",
					r"how.*big",
					r"width.*height",
					r"dimensions.*of"
				],
				"keywords": ["measure", "size", "dimension", "width", "height", "big", "small"]
			},
			"comparison": {
				"patterns": [
					r"compare.*with",
					r"difference.*between",
					r"similar.*to",
					r"match.*specification",
					r"meets.*requirement"
				],
				"keywords": ["compare", "difference", "similar", "match", "meets", "requirement"]
			},
			"compliance": {
				"patterns": [
					r"compliant.*with",
					r"meets.*standard",
					r"regulation.*satisfied",
					r"certification.*valid",
					r"audit.*pass"
				],
				"keywords": ["compliant", "standard", "regulation", "certification", "audit"]
			}
		}
	
	async def _initialize_response_generation(self) -> None:
		"""Initialize response generation templates and patterns"""
		self.response_templates = {
			"object_detection": {
				"positive": "I can see {count} {objects} in the image. {details}",
				"negative": "I don't see any {objects} in this image.",
				"uncertain": "There might be {objects} in the image, but I'm not completely certain."
			},
			"quality_assessment": {
				"good": "The quality appears to be good. {details}",
				"poor": "I've identified some quality issues: {issues}",
				"uncertain": "The quality assessment is inconclusive. {details}"
			},
			"text_extraction": {
				"found": "I found the following text: '{text}'. {details}",
				"not_found": "I couldn't extract any readable text from this image.",
				"partial": "I found some text, but it may be incomplete: '{text}'"
			},
			"measurement": {
				"precise": "The {object} measures approximately {dimensions}. {details}",
				"estimated": "Based on visual estimation, the {object} is roughly {dimensions}.",
				"unavailable": "I cannot accurately measure the {object} from this image."
			}
		}
		
		# Initialize follow-up generators
		self.follow_up_generators = {
			"object_detection": self._generate_object_followups,
			"quality_assessment": self._generate_quality_followups,
			"text_extraction": self._generate_text_followups,
			"measurement": self._generate_measurement_followups
		}
	
	async def _setup_conversation_management(self) -> None:
		"""Setup conversation context management"""
		# Initialize conversation memory patterns
		self.conversation_memory = {
			"max_history": 10,
			"context_decay": 0.9,
			"relevance_threshold": 0.7
		}
	
	async def process_natural_query(
		self,
		query: str,
		image_data: bytes,
		user_context: Dict[str, Any],
		conversation_id: Optional[str] = None
	) -> NaturalLanguageResponse:
		"""
		Process natural language query about visual content
		
		Args:
			query: Natural language query from user
			image_data: Image data to analyze
			user_context: User context and preferences
			conversation_id: Optional conversation ID for multi-turn interactions
			
		Returns:
			NaturalLanguageResponse: Comprehensive response to user query
		"""
		try:
			query_id = uuid7str()
			await self._log_query_operation(
				"Processing natural language query",
				query_id=query_id,
				details=f"Query: '{query[:50]}...'"
			)
			
			# Get or create conversation context
			conversation_context = await self._get_conversation_context(
				conversation_id, user_context
			)
			
			# Parse query intent
			query_intent = await self._parse_query_intent(query, conversation_context)
			
			# Build analysis pipeline
			analysis_pipeline = await self._build_analysis_pipeline(
				query_intent, image_data
			)
			
			# Execute visual analysis
			visual_results = await self._execute_analysis_pipeline(
				image_data, analysis_pipeline
			)
			
			# Generate natural language response
			response = await self._generate_natural_response(
				query, query_intent, visual_results, user_context
			)
			
			# Add follow-up suggestions
			follow_ups = await self._generate_followup_suggestions(
				query_intent, visual_results, conversation_context
			)
			response.follow_up_suggestions = follow_ups
			
			# Update conversation context
			await self._update_conversation_context(
				conversation_context, query, response
			)
			
			await self._log_query_operation(
				"Natural language query processed successfully",
				query_id=query_id,
				details=f"Intent: {query_intent.intent_type}, Confidence: {response.confidence:.2f}"
			)
			
			return response
			
		except Exception as e:
			await self._log_query_operation(
				"Natural language query processing failed",
				query_id=query_id,
				details=str(e)
			)
			
			# Return error response
			return NaturalLanguageResponse(
				tenant_id=user_context.get("tenant_id", "unknown"),
				created_by=user_context.get("user_id", "unknown"),
				answer="I'm sorry, I encountered an error processing your query. Please try rephrasing your question.",
				confidence=0.0,
				query_interpretation=f"Failed to process query: {query}",
				response_type="error"
			)
	
	async def _get_conversation_context(
		self,
		conversation_id: Optional[str],
		user_context: Dict[str, Any]
	) -> ConversationContext:
		"""Get or create conversation context"""
		if conversation_id and conversation_id in self.conversation_contexts:
			return self.conversation_contexts[conversation_id]
		
		# Create new conversation context
		context = ConversationContext(
			conversation_id=conversation_id or uuid7str(),
			tenant_id=user_context.get("tenant_id", "unknown"),
			created_by=user_context.get("user_id", "unknown"),
			user_preferences=user_context.get("preferences", {})
		)
		
		self.conversation_contexts[context.conversation_id] = context
		return context
	
	async def _parse_query_intent(
		self,
		query: str,
		conversation_context: ConversationContext
	) -> QueryIntent:
		"""Parse natural language query to understand intent"""
		query_lower = query.lower()
		doc = self.spacy_nlp(query)
		
		# Initialize intent analysis
		intent_scores = {}
		detected_objects = []
		detected_attributes = []
		
		# Analyze query against intent patterns
		for intent_type, patterns in self.intent_patterns.items():
			score = 0.0
			
			# Pattern matching
			for pattern in patterns["patterns"]:
				if re.search(pattern, query_lower):
					score += 0.4
			
			# Keyword matching
			for keyword in patterns["keywords"]:
				if keyword in query_lower:
					score += 0.1
			
			intent_scores[intent_type] = min(score, 1.0)
		
		# Extract entities and objects
		for ent in doc.ents:
			if ent.label_ in ["PERSON", "ORG", "PRODUCT"]:
				detected_objects.append(ent.text.lower())
		
		# Extract nouns as potential objects
		for token in doc:
			if token.pos_ == "NOUN" and len(token.text) > 2:
				detected_objects.append(token.lemma_.lower())
		
		# Extract adjectives as attributes
		for token in doc:
			if token.pos_ == "ADJ":
				detected_attributes.append(token.lemma_.lower())
		
		# Determine primary intent
		primary_intent = max(intent_scores.items(), key=lambda x: x[1])
		
		# Determine primary action
		primary_action = await self._extract_primary_action(doc, primary_intent[0])
		
		return QueryIntent(
			tenant_id=conversation_context.tenant_id,
			created_by=conversation_context.created_by,
			intent_type=primary_intent[0],
			primary_action=primary_action,
			target_objects=list(set(detected_objects))[:5],  # Limit to 5 most relevant
			attributes=list(set(detected_attributes))[:5],
			confidence=primary_intent[1],
			language="en"
		)
	
	async def _extract_primary_action(self, doc, intent_type: str) -> str:
		"""Extract primary action verb from query"""
		action_verbs = []
		
		for token in doc:
			if token.pos_ == "VERB" and token.dep_ in ["ROOT", "aux"]:
				action_verbs.append(token.lemma_.lower())
		
		if action_verbs:
			return action_verbs[0]
		
		# Default actions by intent type
		default_actions = {
			"object_detection": "detect",
			"quality_assessment": "assess",
			"text_extraction": "extract",
			"measurement": "measure",
			"comparison": "compare",
			"compliance": "verify"
		}
		
		return default_actions.get(intent_type, "analyze")
	
	async def _build_analysis_pipeline(
		self,
		query_intent: QueryIntent,
		image_data: bytes
	) -> AnalysisPipeline:
		"""Build analysis pipeline based on query intent"""
		required_processors = []
		optional_processors = []
		parameters = {}
		
		if query_intent.intent_type == "object_detection":
			required_processors.extend(["object_detection", "classification"])
			if query_intent.target_objects:
				parameters["target_classes"] = query_intent.target_objects
		
		elif query_intent.intent_type == "quality_assessment":
			required_processors.extend(["quality_control", "defect_detection"])
			parameters["quality_threshold"] = 0.8
		
		elif query_intent.intent_type == "text_extraction":
			required_processors.append("ocr")
			parameters["languages"] = ["en"]
		
		elif query_intent.intent_type == "measurement":
			required_processors.extend(["object_detection", "dimensional_analysis"])
			optional_processors.append("scale_reference")
		
		elif query_intent.intent_type == "comparison":
			required_processors.extend(["feature_extraction", "similarity_analysis"])
			optional_processors.append("template_matching")
		
		elif query_intent.intent_type == "compliance":
			required_processors.extend(["compliance_check", "standards_verification"])
			parameters["standards"] = ["iso_9001", "fda_gmp"]
		
		# Always include basic image analysis
		if "image_analysis" not in required_processors:
			optional_processors.append("image_analysis")
		
		return AnalysisPipeline(
			tenant_id=query_intent.tenant_id,
			created_by=query_intent.created_by,
			required_processors=required_processors,
			optional_processors=optional_processors,
			processing_order=required_processors + optional_processors,
			parameters=parameters,
			expected_outputs=["visual_analysis", "metadata", "confidence_scores"]
		)
	
	async def _execute_analysis_pipeline(
		self,
		image_data: bytes,
		analysis_pipeline: AnalysisPipeline
	) -> Dict[str, Any]:
		"""Execute the analysis pipeline on image data"""
		results = {
			"processing_results": {},
			"metadata": {
				"processors_used": analysis_pipeline.required_processors,
				"processing_time_ms": 0,
				"success": True
			}
		}
		
		start_time = datetime.utcnow()
		
		# Simulate processing results based on pipeline
		for processor in analysis_pipeline.required_processors:
			if processor == "object_detection":
				results["processing_results"]["objects"] = await self._simulate_object_detection(
					image_data, analysis_pipeline.parameters
				)
			elif processor == "ocr":
				results["processing_results"]["text"] = await self._simulate_ocr(
					image_data, analysis_pipeline.parameters
				)
			elif processor == "quality_control":
				results["processing_results"]["quality"] = await self._simulate_quality_assessment(
					image_data, analysis_pipeline.parameters
				)
			elif processor == "dimensional_analysis":
				results["processing_results"]["dimensions"] = await self._simulate_dimensional_analysis(
					image_data, analysis_pipeline.parameters
				)
		
		# Calculate processing time
		end_time = datetime.utcnow()
		results["metadata"]["processing_time_ms"] = int(
			(end_time - start_time).total_seconds() * 1000
		)
		
		return results
	
	async def _simulate_object_detection(
		self,
		image_data: bytes,
		parameters: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Simulate object detection results"""
		# In real implementation, this would call actual CV models
		target_classes = parameters.get("target_classes", [])
		
		if "person" in target_classes:
			return {
				"detected_objects": [
					{
						"class_name": "person",
						"confidence": 0.92,
						"bounding_box": {"x": 100, "y": 50, "width": 200, "height": 400},
						"count": 2
					}
				],
				"total_objects": 2,
				"confidence": 0.92
			}
		
		return {
			"detected_objects": [],
			"total_objects": 0,
			"confidence": 0.5
		}
	
	async def _simulate_ocr(
		self,
		image_data: bytes,
		parameters: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Simulate OCR text extraction results"""
		return {
			"extracted_text": "Sample document text extracted from image",
			"confidence": 0.87,
			"language": "en",
			"text_regions": [
				{
					"text": "Sample document text",
					"confidence": 0.87,
					"bounding_box": {"x": 50, "y": 100, "width": 300, "height": 50}
				}
			]
		}
	
	async def _simulate_quality_assessment(
		self,
		image_data: bytes,
		parameters: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Simulate quality assessment results"""
		return {
			"overall_quality_score": 0.85,
			"quality_issues": [],
			"pass_fail_status": "PASS",
			"assessment_details": {
				"surface_quality": 0.9,
				"dimensional_accuracy": 0.8,
				"color_consistency": 0.85
			}
		}
	
	async def _simulate_dimensional_analysis(
		self,
		image_data: bytes,
		parameters: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Simulate dimensional analysis results"""
		return {
			"measurements": {
				"width": {"value": 150, "unit": "mm", "confidence": 0.8},
				"height": {"value": 200, "unit": "mm", "confidence": 0.8}
			},
			"reference_scale": "auto_detected",
			"measurement_accuracy": 0.8
		}
	
	async def _generate_natural_response(
		self,
		query: str,
		query_intent: QueryIntent,
		visual_results: Dict[str, Any],
		user_context: Dict[str, Any]
	) -> NaturalLanguageResponse:
		"""Generate natural language response based on analysis results"""
		intent_type = query_intent.intent_type
		processing_results = visual_results.get("processing_results", {})
		
		# Generate response based on intent type
		if intent_type == "object_detection":
			response = await self._generate_object_detection_response(
				query_intent, processing_results
			)
		elif intent_type == "text_extraction":
			response = await self._generate_text_extraction_response(
				query_intent, processing_results
			)
		elif intent_type == "quality_assessment":
			response = await self._generate_quality_assessment_response(
				query_intent, processing_results
			)
		elif intent_type == "measurement":
			response = await self._generate_measurement_response(
				query_intent, processing_results
			)
		else:
			response = await self._generate_generic_response(
				query_intent, processing_results
			)
		
		# Add metadata
		response.tenant_id = user_context.get("tenant_id", "unknown")
		response.created_by = user_context.get("user_id", "unknown")
		response.supporting_evidence = visual_results
		response.query_interpretation = f"Interpreted as {intent_type} query: {query_intent.primary_action}"
		
		return response
	
	async def _generate_object_detection_response(
		self,
		query_intent: QueryIntent,
		processing_results: Dict[str, Any]
	) -> NaturalLanguageResponse:
		"""Generate response for object detection queries"""
		objects_data = processing_results.get("objects", {})
		detected_objects = objects_data.get("detected_objects", [])
		
		if detected_objects:
			object_counts = {}
			for obj in detected_objects:
				class_name = obj.get("class_name", "object")
				object_counts[class_name] = object_counts.get(class_name, 0) + 1
			
			# Create response
			object_descriptions = []
			for class_name, count in object_counts.items():
				if count == 1:
					object_descriptions.append(f"1 {class_name}")
				else:
					object_descriptions.append(f"{count} {class_name}s")
			
			answer = f"I can see {', '.join(object_descriptions)} in the image."
			confidence = objects_data.get("confidence", 0.0)
			
		else:
			target_objects = query_intent.target_objects
			if target_objects:
				object_list = ", ".join(target_objects)
				answer = f"I don't see any {object_list} in this image."
			else:
				answer = "I don't see any specific objects in this image that match your query."
			confidence = 0.3
		
		return NaturalLanguageResponse(
			answer=answer,
			confidence=confidence,
			response_type="object_detection",
			visual_highlights=[
				{
					"type": "bounding_box",
					"coordinates": obj.get("bounding_box", {}),
					"label": obj.get("class_name", "object")
				}
				for obj in detected_objects
			]
		)
	
	async def _generate_text_extraction_response(
		self,
		query_intent: QueryIntent,
		processing_results: Dict[str, Any]
	) -> NaturalLanguageResponse:
		"""Generate response for text extraction queries"""
		text_data = processing_results.get("text", {})
		extracted_text = text_data.get("extracted_text", "")
		
		if extracted_text.strip():
			answer = f"I found the following text: '{extracted_text}'"
			confidence = text_data.get("confidence", 0.0)
			
			visual_highlights = [
				{
					"type": "text_region",
					"coordinates": region.get("bounding_box", {}),
					"text": region.get("text", "")
				}
				for region in text_data.get("text_regions", [])
			]
		else:
			answer = "I couldn't extract any readable text from this image."
			confidence = 0.2
			visual_highlights = []
		
		return NaturalLanguageResponse(
			answer=answer,
			confidence=confidence,
			response_type="text_extraction",
			visual_highlights=visual_highlights
		)
	
	async def _generate_quality_assessment_response(
		self,
		query_intent: QueryIntent,
		processing_results: Dict[str, Any]
	) -> NaturalLanguageResponse:
		"""Generate response for quality assessment queries"""
		quality_data = processing_results.get("quality", {})
		quality_score = quality_data.get("overall_quality_score", 0.0)
		pass_fail = quality_data.get("pass_fail_status", "UNKNOWN")
		
		if quality_score >= 0.8:
			answer = f"The quality appears to be good with a score of {quality_score:.1%}. The item {pass_fail.lower()}s quality standards."
			confidence = 0.9
		elif quality_score >= 0.6:
			answer = f"The quality is acceptable with a score of {quality_score:.1%}, but there may be minor issues."
			confidence = 0.7
		else:
			issues = quality_data.get("quality_issues", [])
			issue_text = f" Issues found: {', '.join(issues[:3])}" if issues else ""
			answer = f"The quality appears to be poor with a score of {quality_score:.1%}.{issue_text}"
			confidence = 0.8
		
		return NaturalLanguageResponse(
			answer=answer,
			confidence=confidence,
			response_type="quality_assessment"
		)
	
	async def _generate_measurement_response(
		self,
		query_intent: QueryIntent,
		processing_results: Dict[str, Any]
	) -> NaturalLanguageResponse:
		"""Generate response for measurement queries"""
		dimensions_data = processing_results.get("dimensions", {})
		measurements = dimensions_data.get("measurements", {})
		
		if measurements:
			measurement_descriptions = []
			for dimension, data in measurements.items():
				value = data.get("value", 0)
				unit = data.get("unit", "units")
				measurement_descriptions.append(f"{dimension}: {value}{unit}")
			
			answer = f"Based on the image analysis, the measurements are: {', '.join(measurement_descriptions)}."
			confidence = dimensions_data.get("measurement_accuracy", 0.5)
		else:
			answer = "I cannot accurately measure the objects in this image. A reference scale or better image quality may be needed."
			confidence = 0.2
		
		return NaturalLanguageResponse(
			answer=answer,
			confidence=confidence,
			response_type="measurement"
		)
	
	async def _generate_generic_response(
		self,
		query_intent: QueryIntent,
		processing_results: Dict[str, Any]
	) -> NaturalLanguageResponse:
		"""Generate generic response for unclassified queries"""
		answer = "I've analyzed the image, but I'm not sure exactly what you're looking for. Could you please be more specific about what you'd like to know?"
		
		return NaturalLanguageResponse(
			answer=answer,
			confidence=0.3,
			response_type="generic"
		)
	
	async def _generate_followup_suggestions(
		self,
		query_intent: QueryIntent,
		visual_results: Dict[str, Any],
		conversation_context: ConversationContext
	) -> List[str]:
		"""Generate intelligent follow-up question suggestions"""
		intent_type = query_intent.intent_type
		
		if intent_type in self.follow_up_generators:
			return await self.follow_up_generators[intent_type](
				query_intent, visual_results, conversation_context
			)
		
		return [
			"Can you tell me more about what you see?",
			"Would you like me to analyze a different aspect of the image?",
			"Is there something specific you're looking for?"
		]
	
	async def _generate_object_followups(
		self,
		query_intent: QueryIntent,
		visual_results: Dict[str, Any],
		conversation_context: ConversationContext
	) -> List[str]:
		"""Generate follow-up suggestions for object detection"""
		suggestions = [
			"Would you like me to provide more details about the detected objects?",
			"Should I check the quality or condition of these objects?",
			"Would you like to know the approximate sizes of these objects?"
		]
		
		# Add specific suggestions based on detected objects
		objects_data = visual_results.get("processing_results", {}).get("objects", {})
		detected_objects = objects_data.get("detected_objects", [])
		
		if detected_objects:
			object_types = set(obj.get("class_name", "") for obj in detected_objects)
			if "person" in object_types:
				suggestions.append("Would you like me to analyze safety compliance for the people in the image?")
			if any("product" in obj_type.lower() for obj_type in object_types):
				suggestions.append("Should I check if these products meet quality standards?")
		
		return suggestions[:4]  # Limit to 4 suggestions
	
	async def _generate_quality_followups(
		self,
		query_intent: QueryIntent,
		visual_results: Dict[str, Any],
		conversation_context: ConversationContext
	) -> List[str]:
		"""Generate follow-up suggestions for quality assessment"""
		quality_data = visual_results.get("processing_results", {}).get("quality", {})
		quality_score = quality_data.get("overall_quality_score", 0.0)
		
		suggestions = []
		
		if quality_score >= 0.8:
			suggestions.extend([
				"Would you like me to verify compliance with specific standards?",
				"Should I compare this with previous quality assessments?",
				"Would you like detailed quality metrics for reporting?"
			])
		else:
			suggestions.extend([
				"Would you like me to identify specific quality issues?",
				"Should I suggest corrective actions for the quality problems?",
				"Would you like me to assess the severity of these issues?"
			])
		
		return suggestions[:4]
	
	async def _generate_text_followups(
		self,
		query_intent: QueryIntent,
		visual_results: Dict[str, Any],
		conversation_context: ConversationContext
	) -> List[str]:
		"""Generate follow-up suggestions for text extraction"""
		text_data = visual_results.get("processing_results", {}).get("text", {})
		extracted_text = text_data.get("extracted_text", "")
		
		suggestions = []
		
		if extracted_text.strip():
			suggestions.extend([
				"Would you like me to translate this text to another language?",
				"Should I extract specific information from this text?",
				"Would you like me to verify if this text meets document standards?"
			])
		else:
			suggestions.extend([
				"Would you like me to try enhanced text extraction methods?",
				"Should I check if there's text in a different language?",
				"Would you like me to analyze the image quality for text extraction?"
			])
		
		return suggestions[:4]
	
	async def _generate_measurement_followups(
		self,
		query_intent: QueryIntent,
		visual_results: Dict[str, Any],
		conversation_context: ConversationContext
	) -> List[str]:
		"""Generate follow-up suggestions for measurements"""
		return [
			"Would you like me to compare these measurements with specifications?",
			"Should I calculate the area or volume based on these dimensions?",
			"Would you like me to assess if these measurements are within tolerance?",
			"Should I check for dimensional accuracy across multiple samples?"
		]
	
	async def _update_conversation_context(
		self,
		conversation_context: ConversationContext,
		query: str,
		response: NaturalLanguageResponse
	) -> None:
		"""Update conversation context with new query and response"""
		# Add to conversation history
		conversation_context.previous_queries.append(query)
		conversation_context.previous_responses.append(response.answer)
		
		# Limit history length
		max_history = self.conversation_memory["max_history"]
		if len(conversation_context.previous_queries) > max_history:
			conversation_context.previous_queries = conversation_context.previous_queries[-max_history:]
			conversation_context.previous_responses = conversation_context.previous_responses[-max_history:]
		
		# Update timestamp
		conversation_context.updated_at = datetime.utcnow()


# Export main classes
__all__ = [
	"NaturalLanguageVisualQuery",
	"QueryIntent",
	"AnalysisPipeline", 
	"NaturalLanguageResponse",
	"ConversationContext"
]