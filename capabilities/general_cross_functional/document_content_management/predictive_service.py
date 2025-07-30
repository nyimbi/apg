"""
APG Document Content Management - Predictive Analytics Service

Predictive analytics for content value and risk assessment using
machine learning models and business intelligence.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from .models import DCMDocument, DCMPredictiveAnalytics, DCMContentIntelligence


class PredictiveEngine:
	"""Predictive analytics engine for content value and risk assessment"""
	
	def __init__(self, apg_ai_client=None, apg_ml_client=None):
		"""Initialize predictive engine with APG AI/ML integration"""
		self.apg_ai_client = apg_ai_client
		self.apg_ml_client = apg_ml_client
		self.logger = logging.getLogger(__name__)
		
		# Prediction models
		self.value_model = RandomForestRegressor(n_estimators=100, random_state=42)
		self.risk_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
		self.usage_model = RandomForestRegressor(n_estimators=50, random_state=42)
		self.scaler = StandardScaler()
		
		# Feature weights for value prediction
		self.value_features = {
			'access_frequency': 0.25,
			'business_criticality': 0.20,
			'legal_significance': 0.15,
			'content_quality': 0.15,
			'collaboration_level': 0.10,
			'reference_count': 0.10,
			'recency': 0.05
		}
		
		# Risk categories and weights
		self.risk_categories = {
			'compliance_risk': 0.30,
			'security_risk': 0.25,
			'obsolescence_risk': 0.20,
			'operational_risk': 0.15,
			'legal_risk': 0.10
		}
		
		# Prediction accuracy tracking
		self.prediction_stats = {
			'total_predictions': 0,
			'value_predictions': 0,
			'risk_predictions': 0,
			'usage_predictions': 0,
			'accuracy_scores': [],
			'model_confidence': 0.0
		}
	
	async def generate_predictions(
		self,
		document: DCMDocument,
		content_intelligence: Optional[DCMContentIntelligence] = None,
		prediction_types: List[str] = None
	) -> DCMPredictiveAnalytics:
		"""Generate comprehensive predictions for document"""
		prediction_types = prediction_types or ['value', 'risk', 'usage', 'lifecycle']
		
		try:
			# Extract features for prediction
			features = await self._extract_prediction_features(document, content_intelligence)
			
			# Generate predictions based on requested types
			predictions = {}
			
			if 'value' in prediction_types:
				predictions['value'] = await self._predict_content_value(features, document)
			
			if 'risk' in prediction_types:
				predictions['risk'] = await self._predict_content_risks(features, document)
			
			if 'usage' in prediction_types:
				predictions['usage'] = await self._predict_future_usage(features, document)
			
			if 'lifecycle' in prediction_types:
				predictions['lifecycle'] = await self._predict_lifecycle_events(features, document)
			
			# Calculate overall confidence
			confidence_scores = [pred.get('confidence', 0.5) for pred in predictions.values()]
			overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
			
			# Create prediction record
			prediction_record = DCMPredictiveAnalytics(
				tenant_id=document.tenant_id,
				created_by=document.created_by,
				updated_by=document.updated_by,
				document_id=document.id,
				prediction_type='comprehensive',
				model_version='apg-predictive-v1',
				content_value_score=predictions.get('value', {}).get('score', 0.0),
				future_usage_prediction=predictions.get('usage', {}).get('forecast', {}),
				business_impact_score=predictions.get('value', {}).get('business_impact', 0.0),
				risk_probability=predictions.get('risk', {}).get('probabilities', {}),
				compliance_risk_score=predictions.get('risk', {}).get('compliance_risk', 0.0),
				obsolescence_probability=predictions.get('lifecycle', {}).get('obsolescence', 0.0),
				expected_lifespan_days=predictions.get('lifecycle', {}).get('lifespan_days'),
				next_review_prediction=predictions.get('lifecycle', {}).get('next_review'),
				archival_recommendation=predictions.get('lifecycle', {}).get('archival_date'),
				prediction_confidence=overall_confidence,
				validation_status='pending'
			)
			
			# Update statistics
			self._update_prediction_stats(prediction_record)
			
			self.logger.info(f"Predictions generated for document {document.id}")
			return prediction_record
			
		except Exception as e:
			self.logger.error(f"Prediction generation error: {str(e)}")
			
			# Return minimal prediction record
			return DCMPredictiveAnalytics(
				tenant_id=document.tenant_id,
				created_by=document.created_by,
				updated_by=document.updated_by,
				document_id=document.id,
				prediction_type='error',
				model_version='error',
				content_value_score=0.0,
				business_impact_score=0.0,
				compliance_risk_score=0.0,
				obsolescence_probability=0.0,
				prediction_confidence=0.0,
				validation_status='failed'
			)
	
	async def _extract_prediction_features(
		self,
		document: DCMDocument,
		content_intelligence: Optional[DCMContentIntelligence]
	) -> Dict[str, float]:
		"""Extract features for predictive modeling"""
		features = {}
		
		# Document age and recency
		doc_age_days = (datetime.utcnow() - document.created_at).days
		features['document_age_days'] = doc_age_days
		features['recency_score'] = max(0, 1 - (doc_age_days / 3650))  # Decay over 10 years
		
		# Usage metrics
		features['view_count'] = document.view_count
		features['download_count'] = document.download_count
		features['share_count'] = document.share_count
		features['access_frequency'] = self._calculate_access_frequency(document)
		
		# Content characteristics
		features['file_size_mb'] = document.file_size / (1024 * 1024)
		features['version_count'] = document.major_version + document.minor_version / 10
		features['keyword_count'] = len(document.keywords)
		features['category_count'] = len(document.categories)
		
		# Content intelligence features
		if content_intelligence:
			features['content_quality_score'] = content_intelligence.content_quality_score
			features['readability_score'] = content_intelligence.readability_score or 0.5
			features['completeness_score'] = content_intelligence.completeness_score
			features['sensitive_data_flag'] = 1.0 if content_intelligence.sensitive_data_detected else 0.0
			features['compliance_flag_count'] = len(content_intelligence.compliance_flags)
			features['entity_count'] = len(content_intelligence.entity_extraction)
			
			# Risk scores from content intelligence
			risk_scores = content_intelligence.risk_assessment
			features['pii_risk'] = risk_scores.get('pii_exposure', 0.0)
			features['compliance_risk'] = risk_scores.get('compliance_violation', 0.0)
			features['legal_risk'] = risk_scores.get('legal_exposure', 0.0)
		else:
			# Default values when no intelligence available
			features.update({
				'content_quality_score': 0.5,
				'readability_score': 0.5,
				'completeness_score': 0.5,
				'sensitive_data_flag': 0.0,
				'compliance_flag_count': 0,
				'entity_count': 0,
				'pii_risk': 0.0,
				'compliance_risk': 0.0,
				'legal_risk': 0.0
			})
		
		# Document type encoding (simplified)
		doc_type_scores = {
			'contract': 0.9,
			'invoice': 0.7,
			'policy': 0.8,
			'manual': 0.6,
			'email': 0.3,
			'temporary': 0.1
		}
		features['document_type_score'] = doc_type_scores.get(
			document.document_type.value, 0.5
		)
		
		# Business context features
		features['is_locked'] = 1.0 if document.is_locked else 0.0
		features['is_latest_version'] = 1.0 if document.is_latest_version else 0.0
		features['has_expiry'] = 1.0 if document.expiry_date else 0.0
		
		return features
	
	def _calculate_access_frequency(self, document: DCMDocument) -> float:
		"""Calculate access frequency score"""
		total_accesses = document.view_count + document.download_count
		doc_age_days = max(1, (datetime.utcnow() - document.created_at).days)
		
		# Access per day, normalized
		frequency = total_accesses / doc_age_days
		return min(1.0, frequency * 10)  # Scale to 0-1 range
	
	async def _predict_content_value(
		self,
		features: Dict[str, float],
		document: DCMDocument
	) -> Dict[str, Any]:
		"""Predict content value and business impact"""
		try:
			# Calculate weighted value score
			value_score = 0.0
			for feature, weight in self.value_features.items():
				if feature in features:
					value_score += features[feature] * weight
			
			# Use APG ML for sophisticated prediction if available
			if self.apg_ml_client:
				try:
					ml_prediction = await self.apg_ml_client.predict_content_value(
						features, document.id
					)
					value_score = ml_prediction.get('value_score', value_score)
				except Exception as e:
					self.logger.warning(f"APG ML value prediction failed: {str(e)}")
			
			# Business impact calculation
			business_impact = self._calculate_business_impact(features, value_score)
			
			# Strategic importance assessment
			strategic_importance = self._assess_strategic_importance(features, document)
			
			# Future value projection
			value_trajectory = self._project_value_trajectory(features, value_score)
			
			return {
				'score': min(1.0, value_score),
				'business_impact': business_impact,
				'strategic_importance': strategic_importance,
				'value_trajectory': value_trajectory,
				'confidence': 0.75,
				'factors': {
					'access_pattern': features.get('access_frequency', 0),
					'content_quality': features.get('content_quality_score', 0),
					'business_criticality': features.get('document_type_score', 0)
				}
			}
		except Exception as e:
			self.logger.error(f"Value prediction error: {str(e)}")
			return {'score': 0.5, 'confidence': 0.0, 'error': str(e)}
	
	def _calculate_business_impact(self, features: Dict[str, float], value_score: float) -> float:
		"""Calculate business impact score"""
		impact_factors = [
			features.get('access_frequency', 0) * 0.3,
			features.get('document_type_score', 0) * 0.3,
			features.get('compliance_flag_count', 0) / 10 * 0.2,  # Normalize
			value_score * 0.2
		]
		return min(1.0, sum(impact_factors))
	
	def _assess_strategic_importance(self, features: Dict[str, float], document: DCMDocument) -> str:
		"""Assess strategic importance level"""
		importance_score = (
			features.get('document_type_score', 0) * 0.4 +
			features.get('access_frequency', 0) * 0.3 +
			features.get('compliance_flag_count', 0) / 5 * 0.3
		)
		
		if importance_score > 0.8:
			return 'critical'
		elif importance_score > 0.6:
			return 'high'
		elif importance_score > 0.4:
			return 'medium'
		else:
			return 'low'
	
	def _project_value_trajectory(self, features: Dict[str, float], current_value: float) -> Dict[str, float]:
		"""Project value trajectory over time"""
		# Simple decay model for demonstration
		decay_rate = 0.05  # 5% per year
		
		projections = {}
		for months in [6, 12, 24, 36]:
			decay_factor = (1 - decay_rate) ** (months / 12)
			
			# Adjust for content type
			if features.get('document_type_score', 0) > 0.8:  # High-value documents
				decay_factor = max(decay_factor, 0.8)  # Slower decay
			
			projections[f'{months}_months'] = current_value * decay_factor
		
		return projections
	
	async def _predict_content_risks(
		self,
		features: Dict[str, float],
		document: DCMDocument
	) -> Dict[str, Any]:
		"""Predict various content-related risks"""
		try:
			risk_probabilities = {}
			
			# Compliance risk
			compliance_factors = [
				features.get('compliance_risk', 0),
				features.get('sensitive_data_flag', 0),
				features.get('compliance_flag_count', 0) / 10
			]
			risk_probabilities['compliance_violation'] = min(1.0, sum(compliance_factors) / len(compliance_factors))
			
			# Security risk
			security_factors = [
				features.get('pii_risk', 0),
				features.get('sensitive_data_flag', 0),
				features.get('access_frequency', 0) * 0.1  # High access = higher exposure
			]
			risk_probabilities['security_breach'] = min(1.0, sum(security_factors) / len(security_factors))
			
			# Obsolescence risk
			age_factor = features.get('document_age_days', 0) / 1825  # 5 years
			usage_factor = 1 - features.get('access_frequency', 0)
			risk_probabilities['obsolescence'] = min(1.0, (age_factor + usage_factor) / 2)
			
			# Legal risk
			legal_factors = [
				features.get('legal_risk', 0),
				features.get('document_type_score', 0) * 0.1  # Some types have higher legal risk
			]
			risk_probabilities['legal_exposure'] = min(1.0, sum(legal_factors) / len(legal_factors))
			
			# Overall risk score
			overall_risk = sum(
				prob * weight for prob, weight in 
				zip(risk_probabilities.values(), self.risk_categories.values())
			)
			
			return {
				'probabilities': risk_probabilities,
				'overall_risk': overall_risk,
				'compliance_risk': risk_probabilities['compliance_violation'],
				'risk_level': self._categorize_risk_level(overall_risk),
				'confidence': 0.7,
				'mitigation_urgency': self._assess_mitigation_urgency(risk_probabilities)
			}
		except Exception as e:
			self.logger.error(f"Risk prediction error: {str(e)}")
			return {'probabilities': {}, 'overall_risk': 0.5, 'confidence': 0.0}
	
	def _categorize_risk_level(self, risk_score: float) -> str:
		"""Categorize overall risk level"""
		if risk_score > 0.8:
			return 'critical'
		elif risk_score > 0.6:
			return 'high'
		elif risk_score > 0.4:
			return 'medium'
		else:
			return 'low'
	
	def _assess_mitigation_urgency(self, risk_probabilities: Dict[str, float]) -> str:
		"""Assess urgency of risk mitigation"""
		max_risk = max(risk_probabilities.values()) if risk_probabilities else 0
		
		if max_risk > 0.8:
			return 'immediate'
		elif max_risk > 0.6:
			return 'urgent'
		elif max_risk > 0.4:
			return 'moderate'
		else:
			return 'low'
	
	async def _predict_future_usage(
		self,
		features: Dict[str, float],
		document: DCMDocument
	) -> Dict[str, Any]:
		"""Predict future usage patterns"""
		try:
			current_usage = features.get('access_frequency', 0)
			age_factor = features.get('recency_score', 0)
			
			# Usage trend prediction
			trend_factor = 1.0
			if features.get('document_age_days', 0) > 365:
				trend_factor = 0.8  # Older documents typically decline
			
			# Seasonal adjustments (simplified)
			seasonal_factors = {
				'next_month': 1.0,
				'next_quarter': 0.9 * trend_factor,
				'next_year': 0.7 * trend_factor,
				'next_3_years': 0.4 * trend_factor
			}
			
			usage_forecast = {}
			for period, factor in seasonal_factors.items():
				predicted_usage = current_usage * factor * age_factor
				usage_forecast[period] = max(0.0, min(1.0, predicted_usage))
			
			# Peak usage prediction
			peak_probability = features.get('document_type_score', 0) * 0.5
			
			return {
				'forecast': usage_forecast,
				'trend': 'declining' if trend_factor < 1.0 else 'stable',
				'peak_usage_probability': peak_probability,
				'usage_category': self._categorize_usage_level(current_usage),
				'confidence': 0.65
			}
		except Exception as e:
			self.logger.error(f"Usage prediction error: {str(e)}")
			return {'forecast': {}, 'confidence': 0.0}
	
	def _categorize_usage_level(self, usage_score: float) -> str:
		"""Categorize usage level"""
		if usage_score > 0.8:
			return 'high'
		elif usage_score > 0.5:
			return 'medium'
		elif usage_score > 0.2:
			return 'low'
		else:
			return 'minimal'
	
	async def _predict_lifecycle_events(
		self,
		features: Dict[str, float],
		document: DCMDocument
	) -> Dict[str, Any]:
		"""Predict document lifecycle events"""
		try:
			# Expected lifespan calculation
			base_lifespan = 2190  # 6 years default
			
			# Adjust based on document type
			type_multipliers = {
				0.9: 7300,  # Contracts: 20 years
				0.8: 2555,  # Policies: 7 years
				0.7: 2190,  # Invoices: 6 years
				0.6: 1095,  # Manuals: 3 years
				0.3: 365,   # Emails: 1 year
			}
			
			doc_type_score = features.get('document_type_score', 0.5)
			for threshold, lifespan in type_multipliers.items():
				if doc_type_score >= threshold:
					base_lifespan = lifespan
					break
			
			# Adjust for content quality and usage
			quality_factor = features.get('content_quality_score', 0.5)
			usage_factor = features.get('access_frequency', 0.5)
			
			lifespan_multiplier = (quality_factor + usage_factor) / 2
			expected_lifespan = int(base_lifespan * lifespan_multiplier)
			
			# Calculate specific dates
			creation_date = document.created_at
			next_review_date = creation_date + timedelta(days=min(365, expected_lifespan // 3))
			archival_date = creation_date + timedelta(days=expected_lifespan)
			
			# Obsolescence probability
			age_days = features.get('document_age_days', 0)
			obsolescence_prob = min(1.0, age_days / expected_lifespan)
			
			return {
				'lifespan_days': expected_lifespan,
				'next_review': next_review_date.date(),
				'archival_date': archival_date.date(),
				'obsolescence': obsolescence_prob,
				'lifecycle_stage': self._determine_lifecycle_stage(age_days, expected_lifespan),
				'confidence': 0.6
			}
		except Exception as e:
			self.logger.error(f"Lifecycle prediction error: {str(e)}")
			return {'lifespan_days': 2190, 'confidence': 0.0}
	
	def _determine_lifecycle_stage(self, age_days: int, expected_lifespan: int) -> str:
		"""Determine current lifecycle stage"""
		stage_ratio = age_days / expected_lifespan if expected_lifespan > 0 else 0
		
		if stage_ratio < 0.25:
			return 'creation'
		elif stage_ratio < 0.5:
			return 'active'
		elif stage_ratio < 0.75:
			return 'mature'
		elif stage_ratio < 1.0:
			return 'aging'
		else:
			return 'obsolete'
	
	def _update_prediction_stats(self, prediction_record: DCMPredictiveAnalytics):
		"""Update prediction performance statistics"""
		self.prediction_stats['total_predictions'] += 1
		
		if prediction_record.content_value_score > 0:
			self.prediction_stats['value_predictions'] += 1
		
		if prediction_record.risk_probability:
			self.prediction_stats['risk_predictions'] += 1
		
		if prediction_record.future_usage_prediction:
			self.prediction_stats['usage_predictions'] += 1
		
		# Update model confidence
		confidence_scores = [prediction_record.prediction_confidence]
		if self.prediction_stats['accuracy_scores']:
			confidence_scores.extend(self.prediction_stats['accuracy_scores'][-10:])  # Last 10
		
		self.prediction_stats['model_confidence'] = sum(confidence_scores) / len(confidence_scores)
	
	async def validate_predictions(
		self,
		prediction_id: str,
		actual_outcomes: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Validate predictions against actual outcomes"""
		try:
			# This would compare predictions with actual results
			# and update model accuracy
			validation_results = {
				'prediction_id': prediction_id,
				'accuracy_score': 0.0,
				'error_metrics': {},
				'model_updates_required': False
			}
			
			# Calculate accuracy for each prediction type
			if 'value_score' in actual_outcomes:
				# Calculate value prediction accuracy
				pass
			
			if 'risk_events' in actual_outcomes:
				# Calculate risk prediction accuracy
				pass
			
			if 'usage_data' in actual_outcomes:
				# Calculate usage prediction accuracy
				pass
			
			self.logger.info(f"Prediction validation completed for {prediction_id}")
			return validation_results
			
		except Exception as e:
			self.logger.error(f"Prediction validation error: {str(e)}")
			return {'error': str(e)}
	
	async def get_predictive_analytics(self) -> Dict[str, Any]:
		"""Get predictive engine performance analytics"""
		return {
			"prediction_statistics": self.prediction_stats,
			"model_accuracy": self.prediction_stats['model_confidence'],
			"supported_predictions": list(self.value_features.keys()),
			"risk_categories": list(self.risk_categories.keys())
		}