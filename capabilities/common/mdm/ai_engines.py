#!/usr/bin/env python3
"""
APG Master Data Management (MDM) - AI/ML Engines
Advanced AI-powered data quality, matching, and entity resolution engines using local Ollama models

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import re
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
import logging
from uuid_extensions import uuid7str

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import aiohttp
import phonenumbers
from email_validator import validate_email, EmailNotValidError

from .models import EntityType, MatchConfidence, DataQualityStatus
from .database import MDMDatabaseManager


class AIModelType(str, Enum):
    """Types of AI models used in MDM"""
    ENTITY_RESOLUTION = "entity_resolution"
    DATA_QUALITY = "data_quality"
    SEMANTIC_MATCHING = "semantic_matching"
    ANOMALY_DETECTION = "anomaly_detection"
    TEXT_CLASSIFICATION = "text_classification"


@dataclass
class ModelInferenceRequest:
    """Request for AI model inference"""
    model_type: AIModelType
    input_data: Dict[str, Any]
    model_parameters: Optional[Dict[str, Any]] = None
    confidence_threshold: float = 0.5


@dataclass
class ModelInferenceResult:
    """Result from AI model inference"""
    model_type: AIModelType
    predictions: Dict[str, Any]
    confidence_scores: Dict[str, float]
    inference_time_ms: float
    model_version: str
    metadata: Optional[Dict[str, Any]] = None


class OllamaClient:
    """Client for local Ollama AI model inference"""
    
    def __init__(self, base_url: str = "http://localhost:11434", config: Dict[str, Any] = None):
        self.base_url = base_url.rstrip('/')
        self.config = config or {}
        self.session = None
        self.model_cache = {}
        
        # Default models for different tasks
        self.default_models = {
            AIModelType.ENTITY_RESOLUTION: "llama3.2:3b",
            AIModelType.DATA_QUALITY: "llama3.2:3b", 
            AIModelType.SEMANTIC_MATCHING: "nomic-embed-text:latest",
            AIModelType.ANOMALY_DETECTION: "llama3.2:3b",
            AIModelType.TEXT_CLASSIFICATION: "llama3.2:3b"
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def check_model_availability(self, model_name: str) -> bool:
        """Check if model is available in Ollama"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    available_models = [model.get('name', '') for model in models]
                    return any(model_name in available for available in available_models)
                return False
        except Exception as e:
            print(f"[MDM-AI] Error checking model availability: {str(e)}")
            return False
    
    async def generate_text(self, model: str, prompt: str, 
                           system_prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Generate text using Ollama model"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', 0.1),
                    "top_p": kwargs.get('top_p', 0.9),
                    "num_predict": kwargs.get('max_tokens', 1000)
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            start_time = datetime.utcnow()
            
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    end_time = datetime.utcnow()
                    inference_time = (end_time - start_time).total_seconds() * 1000
                    
                    return {
                        'response': result.get('response', ''),
                        'model': result.get('model', model),
                        'done': result.get('done', True),
                        'inference_time_ms': inference_time,
                        'total_duration': result.get('total_duration', 0),
                        'load_duration': result.get('load_duration', 0)
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error: {response.status} - {error_text}")
                    
        except Exception as e:
            print(f"[MDM-AI] Error in text generation: {str(e)}")
            return {
                'response': '',
                'error': str(e),
                'inference_time_ms': 0
            }
    
    async def get_embeddings(self, model: str, text: str) -> Dict[str, Any]:
        """Get text embeddings using Ollama model"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            payload = {
                "model": model,
                "prompt": text
            }
            
            start_time = datetime.utcnow()
            
            async with self.session.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    end_time = datetime.utcnow()
                    inference_time = (end_time - start_time).total_seconds() * 1000
                    
                    return {
                        'embedding': result.get('embedding', []),
                        'model': result.get('model', model),
                        'inference_time_ms': inference_time
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama embeddings API error: {response.status} - {error_text}")
                    
        except Exception as e:
            print(f"[MDM-AI] Error getting embeddings: {str(e)}")
            return {
                'embedding': [],
                'error': str(e),
                'inference_time_ms': 0
            }


class EntityMatchingEngine:
    """Advanced AI-powered entity matching and duplicate detection"""
    
    def __init__(self, ollama_client: OllamaClient, config: Dict[str, Any] = None):
        self.ollama_client = ollama_client
        self.config = config or {}
        self.matching_cache = {}
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),
            max_features=5000
        )
        
        # Matching weights by entity type
        self.matching_weights = {
            EntityType.CUSTOMER: {
                'name': 0.4, 'email': 0.3, 'phone': 0.2, 'address': 0.1
            },
            EntityType.PRODUCT: {
                'name': 0.3, 'sku': 0.4, 'manufacturer': 0.2, 'model': 0.1
            },
            EntityType.SUPPLIER: {
                'name': 0.4, 'email': 0.2, 'tax_id': 0.3, 'address': 0.1
            },
            'default': {
                'name': 0.5, 'identifier': 0.3, 'description': 0.2
            }
        }
    
    async def find_duplicate_candidates(self, entity_data: Dict[str, Any], 
                                      candidate_entities: List[Dict[str, Any]],
                                      entity_type: str = None) -> List[Dict[str, Any]]:
        """Find potential duplicate candidates using AI-enhanced matching"""
        try:
            if not candidate_entities:
                return []
            
            matching_results = []
            entity_type_enum = EntityType(entity_type) if entity_type else None
            weights = self.matching_weights.get(entity_type_enum, self.matching_weights['default'])
            
            # Prepare entity text representations for semantic matching
            entity_text = self._entity_to_text(entity_data)
            candidate_texts = [self._entity_to_text(candidate) for candidate in candidate_entities]
            
            # Get semantic embeddings if available
            semantic_scores = await self._compute_semantic_similarity(entity_text, candidate_texts)
            
            # Process each candidate
            for i, candidate in enumerate(candidate_entities):
                try:
                    # Calculate multiple similarity scores
                    similarity_scores = await self._calculate_similarity_scores(
                        entity_data, candidate, weights
                    )
                    
                    # Add semantic similarity if available
                    if semantic_scores and i < len(semantic_scores):
                        similarity_scores['semantic'] = semantic_scores[i]
                        weights_with_semantic = {**weights, 'semantic': 0.2}
                        # Normalize other weights
                        total_weight = sum(weights.values())
                        for key in weights:
                            weights_with_semantic[key] = weights[key] * 0.8 / total_weight
                    else:
                        weights_with_semantic = weights
                    
                    # Calculate overall match score
                    overall_score = sum(
                        similarity_scores.get(attr, 0) * weight 
                        for attr, weight in weights_with_semantic.items()
                        if attr in similarity_scores
                    ) * 100
                    
                    # Determine confidence level
                    confidence = self._determine_match_confidence(overall_score)
                    
                    # Determine recommended action
                    recommended_action = self._determine_recommended_action(
                        overall_score, similarity_scores
                    )
                    
                    # Create match explanation
                    explanation = self._generate_match_explanation(
                        similarity_scores, overall_score, weights_with_semantic
                    )
                    
                    matching_results.append({
                        'candidate_id': candidate.get('entity_id', candidate.get('id')),
                        'candidate_name': candidate.get('entity_name', candidate.get('name', '')),
                        'candidate_business_key': candidate.get('business_key', ''),
                        'candidate_source_system': candidate.get('source_system', ''),
                        'match_score': round(overall_score, 2),
                        'confidence': confidence.value,
                        'matching_attributes': [
                            attr for attr, score in similarity_scores.items() 
                            if score > 0.7
                        ],
                        'similarity_details': {
                            attr: round(score * 100, 2) 
                            for attr, score in similarity_scores.items()
                        },
                        'recommended_action': recommended_action,
                        'match_explanation': explanation
                    })
                    
                except Exception as e:
                    print(f"[MDM-AI] Error processing candidate {i}: {str(e)}")
                    continue
            
            # Sort by match score descending
            matching_results.sort(key=lambda x: x['match_score'], reverse=True)
            
            # Filter results based on minimum threshold
            min_threshold = self.config.get('min_match_threshold', 50.0)
            matching_results = [
                result for result in matching_results 
                if result['match_score'] >= min_threshold
            ]
            
            return matching_results[:20]  # Return top 20 matches
            
        except Exception as e:
            print(f"[MDM-AI] Error in duplicate detection: {str(e)}")
            return []
    
    async def _calculate_similarity_scores(self, entity1: Dict[str, Any], 
                                         entity2: Dict[str, Any],
                                         weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate detailed similarity scores between two entities"""
        scores = {}
        
        # Name similarity (fuzzy matching)
        name1 = str(entity1.get('entity_name', entity1.get('name', ''))).lower().strip()
        name2 = str(entity2.get('entity_name', entity2.get('name', ''))).lower().strip()
        if name1 and name2:
            scores['name'] = self._fuzzy_match(name1, name2)
        
        # Attributes-based matching
        attrs1 = entity1.get('attributes', {})
        attrs2 = entity2.get('attributes', {})
        
        # Email matching
        email1 = attrs1.get('email', '')
        email2 = attrs2.get('email', '')
        if email1 and email2:
            scores['email'] = 1.0 if email1.lower() == email2.lower() else 0.0
        
        # Phone number matching (normalized)
        phone1 = self._normalize_phone(attrs1.get('phone', ''))
        phone2 = self._normalize_phone(attrs2.get('phone', ''))
        if phone1 and phone2:
            scores['phone'] = 1.0 if phone1 == phone2 else self._fuzzy_match(phone1, phone2)
        
        # Address matching
        addr1 = str(attrs1.get('address', '')).lower().strip()
        addr2 = str(attrs2.get('address', '')).lower().strip()
        if addr1 and addr2:
            scores['address'] = self._fuzzy_match(addr1, addr2)
        
        # SKU/identifier matching for products
        sku1 = str(attrs1.get('sku', entity1.get('business_key', ''))).upper()
        sku2 = str(attrs2.get('sku', entity2.get('business_key', ''))).upper()
        if sku1 and sku2:
            scores['sku'] = 1.0 if sku1 == sku2 else self._fuzzy_match(sku1, sku2)
        
        # Tax ID matching for suppliers
        tax1 = str(attrs1.get('tax_id', '')).replace('-', '').replace(' ', '')
        tax2 = str(attrs2.get('tax_id', '')).replace('-', '').replace(' ', '')
        if tax1 and tax2:
            scores['tax_id'] = 1.0 if tax1 == tax2 else 0.0
        
        # Manufacturer/brand matching
        mfg1 = str(attrs1.get('manufacturer', attrs1.get('brand', ''))).lower().strip()
        mfg2 = str(attrs2.get('manufacturer', attrs2.get('brand', ''))).lower().strip()
        if mfg1 and mfg2:
            scores['manufacturer'] = self._fuzzy_match(mfg1, mfg2)
        
        return scores
    
    async def _compute_semantic_similarity(self, entity_text: str, 
                                         candidate_texts: List[str]) -> List[float]:
        """Compute semantic similarity using embeddings"""
        try:
            model_name = self.ollama_client.default_models[AIModelType.SEMANTIC_MATCHING]
            
            # Check if embedding model is available
            if not await self.ollama_client.check_model_availability(model_name):
                print(f"[MDM-AI] Embedding model {model_name} not available, using TF-IDF")
                return self._compute_tfidf_similarity(entity_text, candidate_texts)
            
            # Get embedding for main entity
            entity_embedding_result = await self.ollama_client.get_embeddings(model_name, entity_text)
            entity_embedding = entity_embedding_result.get('embedding', [])
            
            if not entity_embedding:
                return self._compute_tfidf_similarity(entity_text, candidate_texts)
            
            similarities = []
            
            # Get embeddings for each candidate and compute similarity
            for candidate_text in candidate_texts:
                try:
                    candidate_embedding_result = await self.ollama_client.get_embeddings(
                        model_name, candidate_text
                    )
                    candidate_embedding = candidate_embedding_result.get('embedding', [])
                    
                    if candidate_embedding and len(candidate_embedding) == len(entity_embedding):
                        # Compute cosine similarity
                        similarity = self._cosine_similarity(entity_embedding, candidate_embedding)
                        similarities.append(similarity)
                    else:
                        similarities.append(0.0)
                        
                except Exception as e:
                    print(f"[MDM-AI] Error computing embedding similarity: {str(e)}")
                    similarities.append(0.0)
            
            return similarities
            
        except Exception as e:
            print(f"[MDM-AI] Error in semantic similarity: {str(e)}")
            return self._compute_tfidf_similarity(entity_text, candidate_texts)
    
    def _compute_tfidf_similarity(self, entity_text: str, candidate_texts: List[str]) -> List[float]:
        """Fallback TF-IDF based similarity computation"""
        try:
            all_texts = [entity_text] + candidate_texts
            
            # Compute TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Compute cosine similarities with the first text (entity)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            
            return similarities[0].tolist()
            
        except Exception as e:
            print(f"[MDM-AI] Error in TF-IDF similarity: {str(e)}")
            return [0.0] * len(candidate_texts)
    
    def _entity_to_text(self, entity: Dict[str, Any]) -> str:
        """Convert entity to text representation for semantic analysis"""
        text_parts = []
        
        # Add name
        name = entity.get('entity_name', entity.get('name', ''))
        if name:
            text_parts.append(name)
        
        # Add description
        description = entity.get('entity_description', entity.get('description', ''))
        if description:
            text_parts.append(description)
        
        # Add key attributes
        attributes = entity.get('attributes', {})
        for key, value in attributes.items():
            if isinstance(value, (str, int, float)) and str(value).strip():
                text_parts.append(f"{key}: {value}")
        
        # Add business key
        business_key = entity.get('business_key', '')
        if business_key:
            text_parts.append(f"key: {business_key}")
        
        return ' '.join(text_parts)
    
    def _fuzzy_match(self, str1: str, str2: str) -> float:
        """Compute fuzzy string similarity"""
        if not str1 or not str2:
            return 0.0
        
        # Use SequenceMatcher for basic fuzzy matching
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number for comparison"""
        if not phone:
            return ""
        
        try:
            # Parse phone number
            parsed = phonenumbers.parse(phone, None)
            if phonenumbers.is_valid_number(parsed):
                return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        except:
            pass
        
        # Fallback: extract digits only
        digits = re.sub(r'\D', '', phone)
        return digits if len(digits) >= 10 else ""
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        try:
            if len(vec1) != len(vec2):
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = math.sqrt(sum(a * a for a in vec1))
            norm2 = math.sqrt(sum(b * b for b in vec2))
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception:
            return 0.0
    
    def _determine_match_confidence(self, score: float) -> MatchConfidence:
        """Determine match confidence based on score"""
        if score >= 95:
            return MatchConfidence.EXACT
        elif score >= 80:
            return MatchConfidence.HIGH
        elif score >= 60:
            return MatchConfidence.MEDIUM
        elif score >= 40:
            return MatchConfidence.LOW
        else:
            return MatchConfidence.UNCERTAIN
    
    def _determine_recommended_action(self, score: float, 
                                    similarity_scores: Dict[str, float]) -> str:
        """Determine recommended action based on match analysis"""
        if score >= 90:
            return "merge"
        elif score >= 70:
            # Check if high-confidence exact matches exist
            exact_matches = [k for k, v in similarity_scores.items() if v >= 0.95]
            if len(exact_matches) >= 2:
                return "merge"
            else:
                return "review"
        elif score >= 50:
            return "review"
        else:
            return "ignore"
    
    def _generate_match_explanation(self, similarity_scores: Dict[str, float],
                                  overall_score: float, 
                                  weights: Dict[str, float]) -> str:
        """Generate human-readable match explanation"""
        explanations = []
        
        # High similarity attributes
        high_sim_attrs = [
            attr for attr, score in similarity_scores.items() 
            if score >= 0.8
        ]
        
        if high_sim_attrs:
            explanations.append(f"High similarity in: {', '.join(high_sim_attrs)}")
        
        # Medium similarity attributes
        medium_sim_attrs = [
            attr for attr, score in similarity_scores.items() 
            if 0.5 <= score < 0.8
        ]
        
        if medium_sim_attrs:
            explanations.append(f"Partial similarity in: {', '.join(medium_sim_attrs)}")
        
        # Overall assessment
        if overall_score >= 80:
            explanations.append("Strong overall match confidence")
        elif overall_score >= 60:
            explanations.append("Moderate match confidence")
        else:
            explanations.append("Low match confidence")
        
        return ". ".join(explanations) + "."


class DataQualityEngine:
    """AI-enhanced data quality assessment engine"""
    
    def __init__(self, ollama_client: OllamaClient, config: Dict[str, Any] = None):
        self.ollama_client = ollama_client
        self.config = config or {}
        
        # Quality assessment prompts
        self.quality_prompts = {
            'completeness': """
            Analyze the completeness of this data record. Consider which fields are missing or incomplete.
            Entity Type: {entity_type}
            Data: {data}
            
            Provide a completeness score from 0-100 and identify missing critical fields.
            Response format: {{"score": <number>, "missing_fields": [<list>], "reasoning": "<explanation>"}}
            """,
            
            'accuracy': """
            Assess the accuracy of this data record. Look for format issues, invalid values, or inconsistencies.
            Entity Type: {entity_type}
            Data: {data}
            
            Provide an accuracy score from 0-100 and identify any accuracy issues.
            Response format: {{"score": <number>, "issues": [<list>], "reasoning": "<explanation>"}}
            """,
            
            'consistency': """
            Evaluate the consistency of data formats and values in this record.
            Entity Type: {entity_type}
            Data: {data}
            
            Check for consistent formatting, naming conventions, and value representations.
            Response format: {{"score": <number>, "inconsistencies": [<list>], "reasoning": "<explanation>"}}
            """
        }
    
    async def assess_data_quality_with_ai(self, entity_data: Dict[str, Any],
                                        entity_type: str = None) -> Dict[str, Any]:
        """Perform AI-enhanced data quality assessment"""
        try:
            model_name = self.ollama_client.default_models[AIModelType.DATA_QUALITY]
            
            # Check if model is available
            if not await self.ollama_client.check_model_availability(model_name):
                print(f"[MDM-AI] Quality model {model_name} not available, using rule-based assessment")
                return await self._fallback_quality_assessment(entity_data, entity_type)
            
            quality_results = {}
            data_str = json.dumps(entity_data, indent=2)
            
            # Assess each quality dimension
            for dimension, prompt_template in self.quality_prompts.items():
                try:
                    prompt = prompt_template.format(
                        entity_type=entity_type or 'unknown',
                        data=data_str
                    )
                    
                    result = await self.ollama_client.generate_text(
                        model=model_name,
                        prompt=prompt,
                        system_prompt="You are a data quality expert. Analyze data records and provide structured assessments.",
                        temperature=0.1,
                        max_tokens=500
                    )
                    
                    if result.get('response'):
                        try:
                            # Parse JSON response
                            parsed_result = json.loads(result['response'])
                            quality_results[dimension] = {
                                'score': max(0, min(100, parsed_result.get('score', 0))),
                                'details': parsed_result,
                                'ai_generated': True,
                                'inference_time_ms': result.get('inference_time_ms', 0)
                            }
                        except json.JSONDecodeError:
                            # Fallback to extracting score from text
                            score = self._extract_score_from_text(result['response'])
                            quality_results[dimension] = {
                                'score': score,
                                'details': {'reasoning': result['response']},
                                'ai_generated': True,
                                'inference_time_ms': result.get('inference_time_ms', 0)
                            }
                    else:
                        # Use fallback assessment for this dimension
                        fallback_result = await self._assess_dimension_fallback(
                            dimension, entity_data, entity_type
                        )
                        quality_results[dimension] = fallback_result
                        
                except Exception as e:
                    print(f"[MDM-AI] Error assessing {dimension}: {str(e)}")
                    # Use fallback for this dimension
                    fallback_result = await self._assess_dimension_fallback(
                        dimension, entity_data, entity_type
                    )
                    quality_results[dimension] = fallback_result
            
            # Calculate overall quality score
            dimension_scores = [
                result['score'] for result in quality_results.values()
                if 'score' in result
            ]
            
            if dimension_scores:
                overall_score = sum(dimension_scores) / len(dimension_scores)
            else:
                overall_score = 0.0
            
            # Combine results
            return {
                'overall_score': round(overall_score, 2),
                'dimension_results': quality_results,
                'assessment_method': 'ai_enhanced',
                'model_used': model_name
            }
            
        except Exception as e:
            print(f"[MDM-AI] Error in AI quality assessment: {str(e)}")
            return await self._fallback_quality_assessment(entity_data, entity_type)
    
    async def _fallback_quality_assessment(self, entity_data: Dict[str, Any],
                                         entity_type: str = None) -> Dict[str, Any]:
        """Fallback rule-based quality assessment"""
        results = {}
        
        # Completeness assessment
        results['completeness'] = await self._assess_dimension_fallback(
            'completeness', entity_data, entity_type
        )
        
        # Accuracy assessment  
        results['accuracy'] = await self._assess_dimension_fallback(
            'accuracy', entity_data, entity_type
        )
        
        # Consistency assessment
        results['consistency'] = await self._assess_dimension_fallback(
            'consistency', entity_data, entity_type
        )
        
        # Calculate overall score
        scores = [result['score'] for result in results.values()]
        overall_score = sum(scores) / len(scores) if scores else 0
        
        return {
            'overall_score': round(overall_score, 2),
            'dimension_results': results,
            'assessment_method': 'rule_based',
            'model_used': 'fallback'
        }
    
    async def _assess_dimension_fallback(self, dimension: str, 
                                       entity_data: Dict[str, Any],
                                       entity_type: str = None) -> Dict[str, Any]:
        """Fallback assessment for specific quality dimensions"""
        if dimension == 'completeness':
            return self._assess_completeness_fallback(entity_data, entity_type)
        elif dimension == 'accuracy':
            return self._assess_accuracy_fallback(entity_data, entity_type)
        elif dimension == 'consistency':
            return self._assess_consistency_fallback(entity_data, entity_type)
        else:
            return {'score': 75.0, 'details': {}, 'ai_generated': False}
    
    def _assess_completeness_fallback(self, entity_data: Dict[str, Any], 
                                    entity_type: str = None) -> Dict[str, Any]:
        """Rule-based completeness assessment"""
        required_fields = {
            'customer': ['entity_name', 'email', 'phone'],
            'product': ['entity_name', 'sku', 'category'],
            'supplier': ['entity_name', 'contact_email'],
            'default': ['entity_name']
        }
        
        fields_to_check = required_fields.get(entity_type, required_fields['default'])
        
        # Check entity-level fields
        completed_fields = 0
        missing_fields = []
        
        for field in fields_to_check:
            if field in entity_data and entity_data[field]:
                completed_fields += 1
            else:
                missing_fields.append(field)
        
        # Check attributes
        attributes = entity_data.get('attributes', {})
        attribute_fields = required_fields.get(entity_type, [])
        
        for field in attribute_fields:
            if field not in fields_to_check:  # Don't double-count
                if field in attributes and attributes[field]:
                    completed_fields += 1
                else:
                    missing_fields.append(field)
        
        total_fields = len(fields_to_check) + len([f for f in attribute_fields if f not in fields_to_check])
        score = (completed_fields / total_fields) * 100 if total_fields > 0 else 100
        
        return {
            'score': round(score, 2),
            'details': {
                'missing_fields': missing_fields,
                'completed_fields': completed_fields,
                'total_fields': total_fields
            },
            'ai_generated': False
        }
    
    def _assess_accuracy_fallback(self, entity_data: Dict[str, Any], 
                                entity_type: str = None) -> Dict[str, Any]:
        """Rule-based accuracy assessment"""
        issues = []
        total_checks = 0
        passed_checks = 0
        
        attributes = entity_data.get('attributes', {})
        
        # Email validation
        email = attributes.get('email', '')
        if email:
            total_checks += 1
            try:
                validate_email(email)
                passed_checks += 1
            except EmailNotValidError:
                issues.append(f"Invalid email format: {email}")
        
        # Phone validation
        phone = attributes.get('phone', '')
        if phone:
            total_checks += 1
            try:
                parsed = phonenumbers.parse(phone, None)
                if phonenumbers.is_valid_number(parsed):
                    passed_checks += 1
                else:
                    issues.append(f"Invalid phone number: {phone}")
            except:
                issues.append(f"Invalid phone number format: {phone}")
        
        # URL validation
        url = attributes.get('website', attributes.get('url', ''))
        if url:
            total_checks += 1
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            if url_pattern.match(url):
                passed_checks += 1
            else:
                issues.append(f"Invalid URL format: {url}")
        
        score = (passed_checks / total_checks) * 100 if total_checks > 0 else 100
        
        return {
            'score': round(score, 2),
            'details': {
                'issues': issues,
                'total_checks': total_checks,
                'passed_checks': passed_checks
            },
            'ai_generated': False
        }
    
    def _assess_consistency_fallback(self, entity_data: Dict[str, Any], 
                                   entity_type: str = None) -> Dict[str, Any]:
        """Rule-based consistency assessment"""
        inconsistencies = []
        consistency_score = 100.0
        
        attributes = entity_data.get('attributes', {})
        
        # Name consistency
        name = entity_data.get('entity_name', '')
        if name:
            # Check for proper capitalization
            if not all(word[0].isupper() for word in name.split() if word):
                inconsistencies.append("Name capitalization inconsistent")
                consistency_score -= 10
        
        # Phone consistency
        phone = attributes.get('phone', '')
        if phone:
            # Check for consistent format
            digits = re.sub(r'\D', '', phone)
            if len(digits) < 10:
                inconsistencies.append("Phone number too short")
                consistency_score -= 15
            elif not re.match(r'^\+?[\d\s\-\(\)]+$', phone):
                inconsistencies.append("Phone number contains invalid characters")
                consistency_score -= 10
        
        # Date format consistency
        for key, value in attributes.items():
            if 'date' in key.lower() and isinstance(value, str):
                # Check for consistent date format
                if not re.match(r'^\d{4}-\d{2}-\d{2}', value):
                    inconsistencies.append(f"Inconsistent date format in {key}")
                    consistency_score -= 5
        
        return {
            'score': max(0, consistency_score),
            'details': {
                'inconsistencies': inconsistencies
            },
            'ai_generated': False
        }
    
    def _extract_score_from_text(self, text: str) -> float:
        """Extract numeric score from AI-generated text"""
        # Look for patterns like "score: 85" or "85%" or just numbers
        patterns = [
            r'score[:\s]+(\d+\.?\d*)',
            r'(\d+\.?\d*)%',
            r'(\d+\.?\d*)/100',
            r'(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    return max(0, min(100, score))
                except ValueError:
                    continue
        
        return 50.0  # Default score if nothing found


class AnomalyDetectionEngine:
    """AI-powered anomaly detection for data quality monitoring"""
    
    def __init__(self, ollama_client: OllamaClient, config: Dict[str, Any] = None):
        self.ollama_client = ollama_client
        self.config = config or {}
        self.statistical_thresholds = {
            'z_score_threshold': 3.0,
            'iqr_multiplier': 1.5
        }
    
    async def detect_anomalies(self, entity_data: Dict[str, Any], 
                             historical_data: List[Dict[str, Any]] = None,
                             entity_type: str = None) -> Dict[str, Any]:
        """Detect data anomalies using AI and statistical methods"""
        try:
            anomalies = []
            
            # Statistical anomaly detection
            if historical_data:
                statistical_anomalies = self._detect_statistical_anomalies(
                    entity_data, historical_data
                )
                anomalies.extend(statistical_anomalies)
            
            # AI-based anomaly detection
            ai_anomalies = await self._detect_ai_anomalies(entity_data, entity_type)
            anomalies.extend(ai_anomalies)
            
            # Pattern-based anomaly detection
            pattern_anomalies = self._detect_pattern_anomalies(entity_data, entity_type)
            anomalies.extend(pattern_anomalies)
            
            # Calculate anomaly score
            anomaly_score = min(100.0, len(anomalies) * 15)  # Scale based on number of anomalies
            
            return {
                'anomaly_score': anomaly_score,
                'anomalies_detected': len(anomalies),
                'anomalies': anomalies,
                'assessment_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f"[MDM-AI] Error in anomaly detection: {str(e)}")
            return {
                'anomaly_score': 0.0,
                'anomalies_detected': 0,
                'anomalies': [],
                'error': str(e)
            }
    
    def _detect_statistical_anomalies(self, entity_data: Dict[str, Any],
                                    historical_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect statistical anomalies using z-score and IQR methods"""
        anomalies = []
        
        try:
            # Extract numeric attributes for statistical analysis
            numeric_attrs = self._extract_numeric_attributes(entity_data)
            
            for attr_name, current_value in numeric_attrs.items():
                # Collect historical values for this attribute
                historical_values = []
                for hist_data in historical_data:
                    hist_attrs = hist_data.get('attributes', {})
                    if attr_name in hist_attrs:
                        try:
                            hist_value = float(hist_attrs[attr_name])
                            historical_values.append(hist_value)
                        except (ValueError, TypeError):
                            continue
                
                if len(historical_values) < 5:  # Need minimum data for statistical analysis
                    continue
                
                # Z-score analysis
                mean = np.mean(historical_values)
                std = np.std(historical_values)
                
                if std > 0:
                    z_score = abs(current_value - mean) / std
                    if z_score > self.statistical_thresholds['z_score_threshold']:
                        anomalies.append({
                            'type': 'statistical_outlier',
                            'attribute': attr_name,
                            'current_value': current_value,
                            'expected_range': f"{mean - 2*std:.2f} to {mean + 2*std:.2f}",
                            'z_score': z_score,
                            'severity': 'high' if z_score > 4 else 'medium'
                        })
                
                # IQR analysis
                q1 = np.percentile(historical_values, 25)
                q3 = np.percentile(historical_values, 75)
                iqr = q3 - q1
                
                lower_bound = q1 - self.statistical_thresholds['iqr_multiplier'] * iqr
                upper_bound = q3 + self.statistical_thresholds['iqr_multiplier'] * iqr
                
                if current_value < lower_bound or current_value > upper_bound:
                    anomalies.append({
                        'type': 'iqr_outlier',
                        'attribute': attr_name,
                        'current_value': current_value,
                        'expected_range': f"{lower_bound:.2f} to {upper_bound:.2f}",
                        'severity': 'medium'
                    })
        
        except Exception as e:
            print(f"[MDM-AI] Error in statistical anomaly detection: {str(e)}")
        
        return anomalies
    
    async def _detect_ai_anomalies(self, entity_data: Dict[str, Any], 
                                 entity_type: str = None) -> List[Dict[str, Any]]:
        """Use AI to detect semantic and contextual anomalies"""
        anomalies = []
        
        try:
            model_name = self.ollama_client.default_models[AIModelType.ANOMALY_DETECTION]
            
            if not await self.ollama_client.check_model_availability(model_name):
                return anomalies  # Return empty if model not available
            
            prompt = f"""
            Analyze this {entity_type or 'entity'} data for anomalies, inconsistencies, or unusual patterns.
            Look for:
            - Values that don't make sense for the entity type
            - Inconsistent or conflicting information
            - Unusual patterns or formats
            - Missing expected relationships between fields
            
            Data: {json.dumps(entity_data, indent=2)}
            
            Respond with JSON format: {{"anomalies": [{{"type": "anomaly_type", "field": "field_name", "issue": "description", "severity": "low/medium/high"}}]}}
            """
            
            result = await self.ollama_client.generate_text(
                model=model_name,
                prompt=prompt,
                system_prompt="You are a data quality expert specializing in anomaly detection.",
                temperature=0.1,
                max_tokens=800
            )
            
            if result.get('response'):
                try:
                    parsed_result = json.loads(result['response'])
                    ai_anomalies = parsed_result.get('anomalies', [])
                    
                    for anomaly in ai_anomalies:
                        anomalies.append({
                            'type': 'ai_detected',
                            'subtype': anomaly.get('type', 'unknown'),
                            'attribute': anomaly.get('field', 'unknown'),
                            'description': anomaly.get('issue', 'No description'),
                            'severity': anomaly.get('severity', 'medium'),
                            'ai_generated': True
                        })
                        
                except json.JSONDecodeError:
                    # Try to extract anomalies from text
                    if 'anomal' in result['response'].lower() or 'inconsist' in result['response'].lower():
                        anomalies.append({
                            'type': 'ai_detected',
                            'subtype': 'general',
                            'attribute': 'multiple',
                            'description': result['response'][:200],
                            'severity': 'medium',
                            'ai_generated': True
                        })
        
        except Exception as e:
            print(f"[MDM-AI] Error in AI anomaly detection: {str(e)}")
        
        return anomalies
    
    def _detect_pattern_anomalies(self, entity_data: Dict[str, Any], 
                                entity_type: str = None) -> List[Dict[str, Any]]:
        """Detect pattern-based anomalies using rule-based methods"""
        anomalies = []
        
        try:
            attributes = entity_data.get('attributes', {})
            
            # Age anomalies
            age = attributes.get('age')
            if age is not None:
                try:
                    age_val = float(age)
                    if age_val < 0 or age_val > 150:
                        anomalies.append({
                            'type': 'value_range',
                            'attribute': 'age',
                            'current_value': age_val,
                            'issue': 'Age outside reasonable range',
                            'severity': 'high'
                        })
                except (ValueError, TypeError):
                    pass
            
            # Price anomalies
            price = attributes.get('price', attributes.get('cost'))
            if price is not None:
                try:
                    price_val = float(price)
                    if price_val < 0:
                        anomalies.append({
                            'type': 'negative_value',
                            'attribute': 'price',
                            'current_value': price_val,
                            'issue': 'Negative price value',
                            'severity': 'high'
                        })
                    elif price_val > 1000000:  # Very high price
                        anomalies.append({
                            'type': 'extreme_value',
                            'attribute': 'price',
                            'current_value': price_val,
                            'issue': 'Extremely high price value',
                            'severity': 'medium'
                        })
                except (ValueError, TypeError):
                    pass
            
            # Email domain anomalies
            email = attributes.get('email', '')
            if email:
                # Check for suspicious domains
                suspicious_domains = ['tempmail', 'throwaway', '10minutemail', 'guerrillamail']
                domain = email.split('@')[-1].lower() if '@' in email else ''
                
                for suspicious in suspicious_domains:
                    if suspicious in domain:
                        anomalies.append({
                            'type': 'suspicious_pattern',
                            'attribute': 'email',
                            'current_value': email,
                            'issue': f'Suspicious email domain: {domain}',
                            'severity': 'medium'
                        })
                        break
            
            # Phone number anomalies
            phone = attributes.get('phone', '')
            if phone:
                # Check for obviously fake numbers
                digits = re.sub(r'\D', '', phone)
                if len(set(digits)) == 1:  # All same digit
                    anomalies.append({
                        'type': 'suspicious_pattern',
                        'attribute': 'phone',
                        'current_value': phone,
                        'issue': 'Phone number has all identical digits',
                        'severity': 'high'
                    })
        
        except Exception as e:
            print(f"[MDM-AI] Error in pattern anomaly detection: {str(e)}")
        
        return anomalies
    
    def _extract_numeric_attributes(self, entity_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract numeric attributes for statistical analysis"""
        numeric_attrs = {}
        attributes = entity_data.get('attributes', {})
        
        for key, value in attributes.items():
            try:
                numeric_value = float(value)
                numeric_attrs[key] = numeric_value
            except (ValueError, TypeError):
                continue
        
        return numeric_attrs


# Export main classes
__all__ = [
    'OllamaClient', 'EntityMatchingEngine', 'DataQualityEngine', 'AnomalyDetectionEngine',
    'AIModelType', 'ModelInferenceRequest', 'ModelInferenceResult'
]