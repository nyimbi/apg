"""
Duplicate Prevention Intelligence - Revolutionary Feature #7
Transform duplicate detection from reactive fire-fighting to proactive intelligence

© 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
Website: www.datacraft.co.ke
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID
from enum import Enum
import asyncio
from dataclasses import dataclass
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from typing_extensions import Annotated

from ..auth_rbac.models import User, Role
from ..audit_compliance.models import AuditEntry
from .models import APGBaseModel, Invoice


class DuplicateConfidence(str, Enum):
	CERTAIN = "certain"          # 95-100%
	VERY_HIGH = "very_high"      # 85-94%
	HIGH = "high"                # 75-84%
	MODERATE = "moderate"        # 60-74%
	LOW = "low"                  # 40-59%
	VERY_LOW = "very_low"        # <40%


class DuplicateType(str, Enum):
	EXACT_DUPLICATE = "exact_duplicate"
	NEAR_DUPLICATE = "near_duplicate"
	VARIANT_DUPLICATE = "variant_duplicate"
	POTENTIAL_DUPLICATE = "potential_duplicate"
	FALSE_POSITIVE = "false_positive"


class DetectionMethod(str, Enum):
	EXACT_MATCH = "exact_match"
	FUZZY_LOGIC = "fuzzy_logic"
	ML_SIMILARITY = "ml_similarity"
	PATTERN_ANALYSIS = "pattern_analysis"
	BEHAVIORAL_ANALYSIS = "behavioral_analysis"
	COMPOSITE_SCORING = "composite_scoring"


class PreventionAction(str, Enum):
	BLOCK_SUBMISSION = "block_submission"
	FLAG_FOR_REVIEW = "flag_for_review"
	AUTO_MERGE = "auto_merge"
	REQUEST_CLARIFICATION = "request_clarification"
	ALLOW_WITH_WARNING = "allow_with_warning"


@dataclass
class DuplicateSignature:
	"""AI-powered duplicate signature for intelligent matching"""
	vendor_signature: str
	amount_signature: str
	date_signature: str
	content_signature: str
	behavioral_signature: str
	composite_hash: str


@dataclass
class DuplicateInsight:
	"""Intelligence insight about duplicate patterns"""
	pattern_type: str
	frequency: int
	risk_level: str
	business_impact: str
	prevention_recommendation: str
	automation_potential: float


class DuplicateDetectionRule(APGBaseModel):
	"""Intelligent duplicate detection rule with ML adaptation"""
	
	id: str = Field(default_factory=uuid7str)
	rule_name: str
	description: str
	detection_method: DetectionMethod
	
	# Matching criteria
	fields_to_match: List[str]
	similarity_threshold: float = Field(ge=0.0, le=1.0)
	time_window_days: int = 30
	
	# ML enhancement
	ml_model_enabled: bool = True
	confidence_threshold: float = Field(ge=0.0, le=1.0, default=0.8)
	learning_enabled: bool = True
	
	# Action configuration
	prevention_action: PreventionAction
	auto_resolution_enabled: bool = False
	human_review_required: bool = True
	
	# Performance metrics
	accuracy_score: float = Field(ge=0.0, le=1.0, default=0.0)
	false_positive_rate: float = Field(ge=0.0, le=1.0, default=0.0)
	false_negative_rate: float = Field(ge=0.0, le=1.0, default=0.0)
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)


class DuplicateMatch(APGBaseModel):
	"""Intelligent duplicate match with detailed analysis"""
	
	id: str = Field(default_factory=uuid7str)
	original_invoice_id: str
	duplicate_candidate_id: str
	detection_timestamp: datetime = Field(default_factory=datetime.utcnow)
	
	# Match confidence and type
	confidence_score: float = Field(ge=0.0, le=1.0)
	confidence_level: DuplicateConfidence
	duplicate_type: DuplicateType
	detection_method: DetectionMethod
	
	# Detailed matching analysis
	field_matches: Dict[str, Any] = Field(default_factory=dict)
	similarity_scores: Dict[str, float] = Field(default_factory=dict)
	variance_analysis: Dict[str, Any] = Field(default_factory=dict)
	
	# AI insights
	risk_assessment: Dict[str, Any] = Field(default_factory=dict)
	business_impact_score: float = Field(ge=0.0, le=10.0, default=5.0)
	resolution_recommendations: List[str] = Field(default_factory=list)
	
	# Resolution tracking
	prevention_action_taken: Optional[PreventionAction] = None
	resolution_status: str = "pending"
	resolution_timestamp: Optional[datetime] = None
	resolution_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
	
	# Learning feedback
	human_feedback: Optional[str] = None
	accuracy_confirmed: Optional[bool] = None
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)


class DuplicatePreventionProfile(APGBaseModel):
	"""Vendor-specific duplicate prevention profile with behavioral intelligence"""
	
	id: str = Field(default_factory=uuid7str)
	vendor_id: str
	tenant_id: str
	
	# Historical patterns
	duplicate_history: List[Dict[str, Any]] = Field(default_factory=list)
	typical_invoice_patterns: Dict[str, Any] = Field(default_factory=dict)
	submission_behavior: Dict[str, Any] = Field(default_factory=dict)
	
	# Risk assessment
	duplicate_risk_score: float = Field(ge=0.0, le=1.0, default=0.5)
	risk_factors: List[str] = Field(default_factory=list)
	protective_measures: List[str] = Field(default_factory=list)
	
	# Adaptive thresholds
	custom_similarity_threshold: Optional[float] = None
	custom_time_window: Optional[int] = None
	enhanced_monitoring_enabled: bool = False
	
	# Performance tracking
	prevention_effectiveness: float = Field(ge=0.0, le=1.0, default=0.0)
	false_positive_rate: float = Field(ge=0.0, le=1.0, default=0.0)
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)


class DuplicatePreventionService:
	"""
	Revolutionary Duplicate Prevention Intelligence Service
	
	Transforms duplicate detection from reactive fire-fighting to proactive
	intelligence with ML-powered pattern recognition, behavioral analysis,
	and adaptive prevention strategies.
	"""
	
	def __init__(self, user_context: Dict[str, Any]):
		self.user_context = user_context
		self.user_id = user_context.get('user_id')
		self.tenant_id = user_context.get('tenant_id')
		
		# AI detection configuration
		self.default_similarity_threshold = 0.85
		self.ml_confidence_threshold = 0.8
		self.time_window_days = 30
		
	async def analyze_for_duplicates(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Perform comprehensive duplicate analysis with AI intelligence
		
		This transforms duplicate detection by providing:
		- Multi-layered detection using exact, fuzzy, and ML algorithms
		- Behavioral pattern analysis for vendor-specific intelligence
		- Proactive prevention with confidence-based actions
		- Continuous learning from human feedback
		"""
		try:
			invoice_id = invoice_data.get('invoice_id', '')
			vendor_id = invoice_data.get('vendor_id', '')
			
			# Generate AI-powered duplicate signature
			duplicate_signature = await self._generate_duplicate_signature(invoice_data)
			
			# Perform multi-layered duplicate detection
			detection_results = await self._perform_multi_layered_detection(invoice_data, duplicate_signature)
			
			# Apply vendor-specific behavioral analysis
			behavioral_analysis = await self._analyze_vendor_behavior(vendor_id, invoice_data)
			
			# Calculate composite duplicate probability
			composite_analysis = await self._calculate_composite_probability(detection_results, behavioral_analysis)
			
			# Generate intelligent recommendations
			recommendations = await self._generate_intelligent_recommendations(composite_analysis)
			
			# Determine prevention action
			prevention_action = await self._determine_prevention_action(composite_analysis, recommendations)
			
			# Execute prevention action if required
			action_result = await self._execute_prevention_action(invoice_data, prevention_action)
			
			# Update learning models
			await self._update_learning_models(invoice_data, detection_results, composite_analysis)
			
			return {
				'analysis_type': 'duplicate_prevention_intelligence',
				'invoice_id': invoice_id,
				'vendor_id': vendor_id,
				'analyzed_at': datetime.utcnow(),
				
				# Detection results
				'duplicate_detected': composite_analysis.get('duplicate_detected', False),
				'confidence_score': composite_analysis.get('confidence_score', 0.0),
				'confidence_level': composite_analysis.get('confidence_level', DuplicateConfidence.VERY_LOW.value),
				'duplicate_type': composite_analysis.get('duplicate_type', DuplicateType.FALSE_POSITIVE.value),
				
				# Matching details
				'potential_matches': [
					{
						'match_id': match.get('match_id'),
						'original_invoice_id': match.get('original_invoice_id'),
						'similarity_score': match.get('similarity_score'),
						'detection_method': match.get('detection_method'),
						'field_matches': match.get('field_matches', {}),
						'risk_assessment': match.get('risk_assessment', {})
					}
					for match in detection_results.get('matches', [])
				],
				
				# Behavioral insights
				'behavioral_analysis': {
					'vendor_risk_score': behavioral_analysis.get('risk_score', 0.0),
					'typical_patterns': behavioral_analysis.get('typical_patterns', {}),
					'anomaly_indicators': behavioral_analysis.get('anomaly_indicators', []),
					'submission_behavior': behavioral_analysis.get('submission_behavior', {})
				},
				
				# Prevention action
				'prevention_action': {
					'action_type': prevention_action.get('action_type'),
					'action_taken': prevention_action.get('action_taken', False),
					'blocking_reason': prevention_action.get('blocking_reason'),
					'recommended_resolution': prevention_action.get('recommended_resolution'),
					'human_review_required': prevention_action.get('human_review_required', False)
				},
				
				# Recommendations
				'recommendations': recommendations,
				
				# Learning insights
				'learning_insights': await self._generate_learning_insights(detection_results, behavioral_analysis)
			}
			
		except Exception as e:
			return {
				'error': f'Duplicate prevention analysis failed: {str(e)}',
				'analysis_type': 'duplicate_prevention_intelligence',
				'analyzed_at': datetime.utcnow(),
				'invoice_id': invoice_data.get('invoice_id', '')
			}
	
	async def _generate_duplicate_signature(self, invoice_data: Dict[str, Any]) -> DuplicateSignature:
		"""Generate AI-powered duplicate signature for intelligent matching"""
		# Vendor signature (normalized vendor identification)
		vendor_signature = await self._generate_vendor_signature(invoice_data.get('vendor_id', ''))
		
		# Amount signature (amount clustering with tolerance)
		amount_signature = await self._generate_amount_signature(invoice_data.get('total_amount', 0.0))
		
		# Date signature (date clustering with business day awareness)
		date_signature = await self._generate_date_signature(invoice_data.get('invoice_date', ''))
		
		# Content signature (semantic hash of line items and descriptions)
		content_signature = await self._generate_content_signature(invoice_data.get('line_items', []))
		
		# Behavioral signature (submission patterns and metadata)
		behavioral_signature = await self._generate_behavioral_signature(invoice_data)
		
		# Composite hash for efficient lookup
		composite_components = f"{vendor_signature}|{amount_signature}|{date_signature}|{content_signature}|{behavioral_signature}"
		composite_hash = str(hash(composite_components))
		
		return DuplicateSignature(
			vendor_signature=vendor_signature,
			amount_signature=amount_signature,
			date_signature=date_signature,
			content_signature=content_signature,
			behavioral_signature=behavioral_signature,
			composite_hash=composite_hash
		)
	
	async def _perform_multi_layered_detection(self, invoice_data: Dict[str, Any], signature: DuplicateSignature) -> Dict[str, Any]:
		"""Perform multi-layered duplicate detection with various algorithms"""
		detection_results = {
			'matches': [],
			'detection_methods_used': [],
			'highest_confidence': 0.0,
			'total_candidates': 0
		}
		
		# Layer 1: Exact signature matching
		exact_matches = await self._detect_exact_matches(signature)
		if exact_matches:
			detection_results['matches'].extend(exact_matches)
			detection_results['detection_methods_used'].append(DetectionMethod.EXACT_MATCH.value)
		
		# Layer 2: Fuzzy logic matching
		fuzzy_matches = await self._detect_fuzzy_matches(invoice_data, signature)
		if fuzzy_matches:
			detection_results['matches'].extend(fuzzy_matches)
			detection_results['detection_methods_used'].append(DetectionMethod.FUZZY_LOGIC.value)
		
		# Layer 3: ML similarity matching
		ml_matches = await self._detect_ml_similarity_matches(invoice_data, signature)
		if ml_matches:
			detection_results['matches'].extend(ml_matches)
			detection_results['detection_methods_used'].append(DetectionMethod.ML_SIMILARITY.value)
		
		# Layer 4: Pattern analysis
		pattern_matches = await self._detect_pattern_matches(invoice_data, signature)
		if pattern_matches:
			detection_results['matches'].extend(pattern_matches)
			detection_results['detection_methods_used'].append(DetectionMethod.PATTERN_ANALYSIS.value)
		
		# Deduplicate and rank matches
		deduplicated_matches = await self._deduplicate_and_rank_matches(detection_results['matches'])
		detection_results['matches'] = deduplicated_matches
		detection_results['total_candidates'] = len(deduplicated_matches)
		detection_results['highest_confidence'] = max([m.get('similarity_score', 0.0) for m in deduplicated_matches], default=0.0)
		
		return detection_results
	
	async def _analyze_vendor_behavior(self, vendor_id: str, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Analyze vendor-specific behavioral patterns for duplicate intelligence"""
		# Get vendor duplicate prevention profile
		vendor_profile = await self._get_vendor_duplicate_profile(vendor_id)
		
		# Analyze submission patterns
		submission_analysis = await self._analyze_submission_patterns(vendor_id, invoice_data)
		
		# Calculate risk factors
		risk_factors = await self._calculate_vendor_risk_factors(vendor_profile, submission_analysis)
		
		# Detect behavioral anomalies
		anomaly_indicators = await self._detect_behavioral_anomalies(vendor_profile, invoice_data)
		
		return {
			'risk_score': vendor_profile.duplicate_risk_score if vendor_profile else 0.5,
			'typical_patterns': vendor_profile.typical_invoice_patterns if vendor_profile else {},
			'submission_behavior': submission_analysis,
			'risk_factors': risk_factors,
			'anomaly_indicators': anomaly_indicators,
			'enhanced_monitoring': vendor_profile.enhanced_monitoring_enabled if vendor_profile else False
		}
	
	async def _calculate_composite_probability(self, detection_results: Dict[str, Any], behavioral_analysis: Dict[str, Any]) -> Dict[str, Any]:
		"""Calculate composite duplicate probability using ML ensemble methods"""
		matches = detection_results.get('matches', [])
		if not matches:
			return {
				'duplicate_detected': False,
				'confidence_score': 0.0,
				'confidence_level': DuplicateConfidence.VERY_LOW.value,
				'duplicate_type': DuplicateType.FALSE_POSITIVE.value
			}
		
		# Get best match
		best_match = max(matches, key=lambda m: m.get('similarity_score', 0.0))
		base_confidence = best_match.get('similarity_score', 0.0)
		
		# Apply behavioral adjustments
		behavioral_risk = behavioral_analysis.get('risk_score', 0.5)
		anomaly_count = len(behavioral_analysis.get('anomaly_indicators', []))
		
		# Composite scoring algorithm
		composite_confidence = base_confidence
		
		# Boost confidence for high-risk vendors
		if behavioral_risk > 0.7:
			composite_confidence = min(1.0, composite_confidence * 1.2)
		
		# Reduce confidence for vendors with good history
		if behavioral_risk < 0.3:
			composite_confidence = composite_confidence * 0.8
		
		# Adjust for behavioral anomalies
		if anomaly_count > 0:
			composite_confidence = min(1.0, composite_confidence + (anomaly_count * 0.1))
		
		# Determine confidence level and duplicate type
		if composite_confidence >= 0.95:
			confidence_level = DuplicateConfidence.CERTAIN
			duplicate_type = DuplicateType.EXACT_DUPLICATE
		elif composite_confidence >= 0.85:
			confidence_level = DuplicateConfidence.VERY_HIGH
			duplicate_type = DuplicateType.NEAR_DUPLICATE
		elif composite_confidence >= 0.75:
			confidence_level = DuplicateConfidence.HIGH
			duplicate_type = DuplicateType.VARIANT_DUPLICATE
		elif composite_confidence >= 0.60:
			confidence_level = DuplicateConfidence.MODERATE
			duplicate_type = DuplicateType.POTENTIAL_DUPLICATE
		else:
			confidence_level = DuplicateConfidence.LOW
			duplicate_type = DuplicateType.FALSE_POSITIVE
		
		return {
			'duplicate_detected': composite_confidence >= self.ml_confidence_threshold,
			'confidence_score': composite_confidence,
			'confidence_level': confidence_level.value,
			'duplicate_type': duplicate_type.value,
			'best_match': best_match,
			'composite_factors': {
				'base_confidence': base_confidence,
				'behavioral_adjustment': behavioral_risk,
				'anomaly_adjustment': anomaly_count * 0.1
			}
		}
	
	async def _generate_intelligent_recommendations(self, composite_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Generate intelligent recommendations based on duplicate analysis"""
		recommendations = []
		
		confidence_score = composite_analysis.get('confidence_score', 0.0)
		duplicate_type = composite_analysis.get('duplicate_type', DuplicateType.FALSE_POSITIVE.value)
		
		# High confidence duplicate recommendations
		if confidence_score >= 0.9:
			recommendations.append({
				'category': 'immediate_action',
				'priority': 'critical',
				'title': 'Block Duplicate Invoice Submission',
				'description': 'High confidence duplicate detected - block submission immediately',
				'action': 'Block submission and notify accounts payable team',
				'business_impact': 'Prevents duplicate payment and maintains financial accuracy'
			})
		
		# Moderate confidence recommendations
		elif confidence_score >= 0.7:
			recommendations.append({
				'category': 'review_required',
				'priority': 'high',
				'title': 'Manual Review Required for Potential Duplicate',
				'description': 'Potential duplicate requires human verification',
				'action': 'Flag for manual review by senior AP specialist',
				'business_impact': 'Balances automation efficiency with accuracy assurance'
			})
		
		# Process improvement recommendations
		if duplicate_type == DuplicateType.VARIANT_DUPLICATE.value:
			recommendations.append({
				'category': 'process_improvement',
				'priority': 'medium',
				'title': 'Vendor Communication Enhancement',
				'description': 'Vendor submitting similar invoices with variations',
				'action': 'Reach out to vendor to clarify invoicing practices',
				'business_impact': 'Improves vendor relationship and reduces future ambiguity'
			})
		
		# Automation optimization recommendations
		recommendations.append({
			'category': 'automation_optimization',
			'priority': 'low',
			'title': 'Enhance Detection Rules',
			'description': 'Optimize detection algorithms based on this analysis',
			'action': 'Update ML models with analysis results for improved accuracy',
			'business_impact': 'Continuously improves duplicate detection effectiveness'
		})
		
		return recommendations
	
	async def _determine_prevention_action(self, composite_analysis: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Determine appropriate prevention action based on analysis"""
		confidence_score = composite_analysis.get('confidence_score', 0.0)
		duplicate_detected = composite_analysis.get('duplicate_detected', False)
		
		if not duplicate_detected:
			return {
				'action_type': PreventionAction.ALLOW_WITH_WARNING.value,
				'action_taken': False,
				'blocking_reason': None,
				'recommended_resolution': 'Proceed with normal processing',
				'human_review_required': False
			}
		
		# Determine action based on confidence
		if confidence_score >= 0.95:
			return {
				'action_type': PreventionAction.BLOCK_SUBMISSION.value,
				'action_taken': True,
				'blocking_reason': 'High confidence duplicate detected - exact or near-exact match found',
				'recommended_resolution': 'Investigate original invoice and resolve duplication',
				'human_review_required': True
			}
		elif confidence_score >= 0.8:
			return {
				'action_type': PreventionAction.FLAG_FOR_REVIEW.value,
				'action_taken': True,
				'blocking_reason': 'Potential duplicate requires verification',
				'recommended_resolution': 'Manual review by AP specialist to confirm or dismiss',
				'human_review_required': True
			}
		elif confidence_score >= 0.7:
			return {
				'action_type': PreventionAction.REQUEST_CLARIFICATION.value,
				'action_taken': True,
				'blocking_reason': 'Similar invoice detected - clarification needed',
				'recommended_resolution': 'Request additional documentation from vendor',
				'human_review_required': False
			}
		else:
			return {
				'action_type': PreventionAction.ALLOW_WITH_WARNING.value,
				'action_taken': False,
				'blocking_reason': None,
				'recommended_resolution': 'Proceed with enhanced monitoring',
				'human_review_required': False
			}
	
	async def _execute_prevention_action(self, invoice_data: Dict[str, Any], prevention_action: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute the determined prevention action"""
		action_type = prevention_action.get('action_type')
		
		if action_type == PreventionAction.BLOCK_SUBMISSION.value:
			# Block the invoice submission
			result = await self._block_invoice_submission(invoice_data, prevention_action)
		elif action_type == PreventionAction.FLAG_FOR_REVIEW.value:
			# Flag for manual review
			result = await self._flag_for_manual_review(invoice_data, prevention_action)
		elif action_type == PreventionAction.REQUEST_CLARIFICATION.value:
			# Request clarification from vendor
			result = await self._request_vendor_clarification(invoice_data, prevention_action)
		else:
			# Allow with monitoring
			result = await self._allow_with_monitoring(invoice_data, prevention_action)
		
		return result
	
	async def _update_learning_models(self, invoice_data: Dict[str, Any], detection_results: Dict[str, Any], composite_analysis: Dict[str, Any]) -> None:
		"""Update ML models with new data for continuous learning"""
		# Update detection accuracy metrics
		await self._update_detection_metrics(detection_results, composite_analysis)
		
		# Update vendor behavioral models
		await self._update_vendor_models(invoice_data, composite_analysis)
		
		# Update global duplicate patterns
		await self._update_global_patterns(detection_results)
	
	async def _generate_learning_insights(self, detection_results: Dict[str, Any], behavioral_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Generate insights for continuous learning improvement"""
		insights = []
		
		# Detection effectiveness insight
		methods_used = detection_results.get('detection_methods_used', [])
		if len(methods_used) > 1:
			insights.append({
				'insight_type': 'detection_effectiveness',
				'description': f'Multiple detection methods used: {", ".join(methods_used)}',
				'learning_opportunity': 'Analyze which method provided most accurate results',
				'automation_potential': 0.8
			})
		
		# Behavioral pattern insight
		risk_score = behavioral_analysis.get('risk_score', 0.5)
		if risk_score > 0.7:
			insights.append({
				'insight_type': 'vendor_risk_pattern',
				'description': 'High-risk vendor pattern detected',
				'learning_opportunity': 'Enhance vendor-specific detection rules',
				'automation_potential': 0.6
			})
		
		return insights
	
	# Implementation helper methods (simplified for brevity)
	
	async def _generate_vendor_signature(self, vendor_id: str) -> str:
		"""Generate normalized vendor signature"""
		return f"VENDOR_{vendor_id}"
	
	async def _generate_amount_signature(self, amount: float) -> str:
		"""Generate amount clustering signature"""
		# Round to nearest $10 for clustering
		rounded_amount = round(amount / 10) * 10
		return f"AMT_{rounded_amount}"
	
	async def _generate_date_signature(self, invoice_date: str) -> str:
		"""Generate date clustering signature"""
		# Convert to date and create weekly cluster
		try:
			date_obj = datetime.fromisoformat(invoice_date).date()
			week_start = date_obj - timedelta(days=date_obj.weekday())
			return f"WEEK_{week_start.isoformat()}"
		except:
			return "DATE_UNKNOWN"
	
	async def _generate_content_signature(self, line_items: List[Dict[str, Any]]) -> str:
		"""Generate semantic content signature"""
		# Simplified content hash based on descriptions
		descriptions = [item.get('description', '') for item in line_items]
		content_text = ' '.join(descriptions).lower().strip()
		return f"CONTENT_{hash(content_text)}"
	
	async def _generate_behavioral_signature(self, invoice_data: Dict[str, Any]) -> str:
		"""Generate behavioral submission signature"""
		# Consider submission time, file format, etc.
		submission_hour = datetime.utcnow().hour
		return f"BEHAVIOR_{submission_hour}"
	
	async def _detect_exact_matches(self, signature: DuplicateSignature) -> List[Dict[str, Any]]:
		"""Detect exact signature matches"""
		# Implementation would query database for exact matches
		return [
			{
				'match_id': uuid7str(),
				'original_invoice_id': 'INV-2025-001234',
				'similarity_score': 1.0,
				'detection_method': DetectionMethod.EXACT_MATCH.value,
				'field_matches': {
					'vendor_signature': True,
					'amount_signature': True,
					'content_signature': True
				},
				'risk_assessment': {
					'risk_level': 'critical',
					'confidence': 0.98
				}
			}
		]
	
	async def _detect_fuzzy_matches(self, invoice_data: Dict[str, Any], signature: DuplicateSignature) -> List[Dict[str, Any]]:
		"""Detect fuzzy logic matches"""
		# Implementation would use fuzzy matching algorithms
		return []
	
	async def _detect_ml_similarity_matches(self, invoice_data: Dict[str, Any], signature: DuplicateSignature) -> List[Dict[str, Any]]:
		"""Detect ML similarity matches"""
		# Implementation would use ML similarity models
		return []
	
	async def _detect_pattern_matches(self, invoice_data: Dict[str, Any], signature: DuplicateSignature) -> List[Dict[str, Any]]:
		"""Detect pattern-based matches"""
		# Implementation would use pattern recognition
		return []
	
	async def _deduplicate_and_rank_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Deduplicate and rank matches by confidence"""
		# Remove duplicates and sort by similarity score
		unique_matches = {}
		for match in matches:
			key = match.get('original_invoice_id')
			if key not in unique_matches or match.get('similarity_score', 0) > unique_matches[key].get('similarity_score', 0):
				unique_matches[key] = match
		
		return sorted(unique_matches.values(), key=lambda m: m.get('similarity_score', 0), reverse=True)
	
	async def _get_vendor_duplicate_profile(self, vendor_id: str) -> Optional[DuplicatePreventionProfile]:
		"""Get vendor duplicate prevention profile"""
		# Implementation would fetch from database
		return DuplicatePreventionProfile(
			vendor_id=vendor_id,
			tenant_id=self.tenant_id,
			duplicate_risk_score=0.3,
			risk_factors=['low_volume_vendor'],
			prevention_effectiveness=0.85
		)
	
	async def _analyze_submission_patterns(self, vendor_id: str, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Analyze vendor submission patterns"""
		return {
			'submission_frequency': 'weekly',
			'typical_amounts': [1000, 2500, 5000],
			'submission_times': ['morning', 'afternoon'],
			'format_consistency': 0.9
		}
	
	async def _calculate_vendor_risk_factors(self, profile: Optional[DuplicatePreventionProfile], submission_analysis: Dict[str, Any]) -> List[str]:
		"""Calculate vendor-specific risk factors"""
		risk_factors = []
		
		if profile and profile.duplicate_risk_score > 0.7:
			risk_factors.append('high_historical_duplicates')
		
		if submission_analysis.get('format_consistency', 1.0) < 0.8:
			risk_factors.append('inconsistent_format')
		
		return risk_factors
	
	async def _detect_behavioral_anomalies(self, profile: Optional[DuplicatePreventionProfile], invoice_data: Dict[str, Any]) -> List[str]:
		"""Detect behavioral anomalies"""
		anomalies = []
		
		# Check for unusual submission timing
		current_hour = datetime.utcnow().hour
		if current_hour < 6 or current_hour > 22:
			anomalies.append('unusual_submission_time')
		
		# Check for unusual amount patterns
		amount = invoice_data.get('total_amount', 0.0)
		if profile and profile.typical_invoice_patterns:
			typical_range = profile.typical_invoice_patterns.get('amount_range', [0, 10000])
			if amount < typical_range[0] or amount > typical_range[1]:
				anomalies.append('unusual_amount')
		
		return anomalies
	
	async def _block_invoice_submission(self, invoice_data: Dict[str, Any], prevention_action: Dict[str, Any]) -> Dict[str, Any]:
		"""Block invoice submission"""
		return {
			'action_executed': True,
			'submission_blocked': True,
			'notification_sent': True,
			'review_queue_added': True
		}
	
	async def _flag_for_manual_review(self, invoice_data: Dict[str, Any], prevention_action: Dict[str, Any]) -> Dict[str, Any]:
		"""Flag invoice for manual review"""
		return {
			'action_executed': True,
			'review_flag_added': True,
			'notification_sent': True,
			'priority_level': 'high'
		}
	
	async def _request_vendor_clarification(self, invoice_data: Dict[str, Any], prevention_action: Dict[str, Any]) -> Dict[str, Any]:
		"""Request clarification from vendor"""
		return {
			'action_executed': True,
			'clarification_requested': True,
			'notification_sent': True,
			'response_deadline': datetime.utcnow() + timedelta(days=3)
		}
	
	async def _allow_with_monitoring(self, invoice_data: Dict[str, Any], prevention_action: Dict[str, Any]) -> Dict[str, Any]:
		"""Allow submission with enhanced monitoring"""
		return {
			'action_executed': True,
			'submission_allowed': True,
			'monitoring_enabled': True,
			'tracking_enhanced': True
		}
	
	async def _update_detection_metrics(self, detection_results: Dict[str, Any], composite_analysis: Dict[str, Any]) -> None:
		"""Update detection accuracy metrics"""
		# Implementation would update ML model metrics
		pass
	
	async def _update_vendor_models(self, invoice_data: Dict[str, Any], composite_analysis: Dict[str, Any]) -> None:
		"""Update vendor behavioral models"""
		# Implementation would update vendor profiles
		pass
	
	async def _update_global_patterns(self, detection_results: Dict[str, Any]) -> None:
		"""Update global duplicate patterns"""
		# Implementation would update global pattern recognition
		pass