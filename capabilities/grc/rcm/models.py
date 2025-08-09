"""
APG Governance, Risk & Compliance Models

Revolutionary GRC data models with AI-powered intelligence, following APG patterns
for multi-tenant architecture, audit trails, and seamless ecosystem integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.hybrid import hybrid_property
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from annotated_types import Annotated
import json

from ..auth_rbac.models import BaseMixin, AuditMixin, Model


# ==============================================================================
# ENUM DEFINITIONS
# ==============================================================================

class GRCRiskLevel(str, Enum):
	"""Risk severity levels with quantitative thresholds"""
	CRITICAL = "critical"		# Risk Score: 90-100
	HIGH = "high"				# Risk Score: 70-89
	MEDIUM = "medium"			# Risk Score: 40-69
	LOW = "low"					# Risk Score: 20-39
	MINIMAL = "minimal"			# Risk Score: 0-19


class GRCRiskStatus(str, Enum):
	"""Risk lifecycle status tracking"""
	IDENTIFIED = "identified"
	ASSESSED = "assessed"
	TREATED = "treated"
	MONITORED = "monitored"
	CLOSED = "closed"
	ESCALATED = "escalated"


class GRCComplianceStatus(str, Enum):
	"""Compliance state with AI-powered determination"""
	COMPLIANT = "compliant"
	NON_COMPLIANT = "non_compliant"
	PARTIALLY_COMPLIANT = "partially_compliant"
	NOT_ASSESSED = "not_assessed"
	PENDING_REVIEW = "pending_review"
	EXCEPTION_APPROVED = "exception_approved"


class GRCControlType(str, Enum):
	"""Control classification for intelligent automation"""
	PREVENTIVE = "preventive"
	DETECTIVE = "detective"
	CORRECTIVE = "corrective"
	COMPENSATING = "compensating"
	DIRECTIVE = "directive"


class GRCGovernanceDecisionType(str, Enum):
	"""Governance decision categorization"""
	POLICY_APPROVAL = "policy_approval"
	RISK_ACCEPTANCE = "risk_acceptance"
	BUDGET_ALLOCATION = "budget_allocation"
	STRATEGIC_DIRECTION = "strategic_direction"
	COMPLIANCE_EXCEPTION = "compliance_exception"
	OPERATIONAL_CHANGE = "operational_change"


# ==============================================================================
# CORE RISK MANAGEMENT MODELS
# ==============================================================================

class GRCRisk(Model, AuditMixin, BaseMixin):
	"""
	Core Risk Entity with AI-Powered Intelligence
	
	Revolutionary risk management with predictive analytics, automated
	correlation analysis, and intelligent risk treatment recommendations.
	"""
	__tablename__ = 'grc_risk'
	
	# Identity and Classification
	risk_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Risk Identification
	risk_title = Column(String(200), nullable=False, index=True)
	risk_description = Column(Text, nullable=False)
	risk_category_id = Column(String(36), ForeignKey('grc_risk_category.category_id'), nullable=False, index=True)
	risk_owner_id = Column(String(36), nullable=False, index=True)  # User responsible for risk
	
	# Risk Assessment (AI-Enhanced)
	inherent_probability = Column(Float, nullable=False, default=0.0)  # 0.0-1.0
	inherent_impact = Column(Float, nullable=False, default=0.0)       # 0.0-1.0
	inherent_risk_score = Column(Float, nullable=False, default=0.0)   # Calculated: prob * impact * 100
	
	residual_probability = Column(Float, nullable=False, default=0.0)
	residual_impact = Column(Float, nullable=False, default=0.0)
	residual_risk_score = Column(Float, nullable=False, default=0.0)
	
	# AI-Powered Risk Intelligence
	ai_risk_prediction = Column(JSON, default=dict)  # ML model predictions
	risk_velocity = Column(Float, default=0.0)       # Rate of risk change
	risk_correlation_score = Column(Float, default=0.0)  # Correlation with other risks
	predictive_indicators = Column(JSON, default=list)   # Leading indicators
	
	# Risk Status and Management
	risk_level = Column(String(20), nullable=False, default=GRCRiskLevel.MEDIUM.value, index=True)
	risk_status = Column(String(20), nullable=False, default=GRCRiskStatus.IDENTIFIED.value, index=True)
	risk_appetite_alignment = Column(String(20), default="within_appetite")  # within, above, below
	
	# Temporal Tracking
	risk_identification_date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
	last_assessment_date = Column(DateTime, nullable=True)
	next_review_date = Column(DateTime, nullable=True, index=True)
	risk_materialization_date = Column(DateTime, nullable=True)
	
	# Business Context
	business_process = Column(String(100), nullable=True, index=True)
	geographic_scope = Column(JSON, default=list)  # Geographic locations affected
	stakeholder_impact = Column(JSON, default=dict)  # Impact on different stakeholders
	regulatory_implications = Column(JSON, default=list)  # Regulatory consequences
	
	# Financial Impact (Quantitative)
	financial_impact_min = Column(Float, nullable=True)  # Minimum financial impact
	financial_impact_max = Column(Float, nullable=True)  # Maximum financial impact
	financial_impact_expected = Column(Float, nullable=True)  # Expected value
	currency = Column(String(3), default="USD")
	
	# Operational Impact
	operational_impact = Column(JSON, default=dict)  # Service disruption, downtime, etc.
	reputation_impact = Column(String(20), nullable=True)  # low, medium, high, critical
	strategic_impact = Column(String(20), nullable=True)   # Impact on strategic objectives
	
	# Risk Metadata
	risk_tags = Column(JSON, default=list)  # Flexible tagging system
	external_references = Column(JSON, default=list)  # External risk databases, standards
	risk_escalation_criteria = Column(JSON, default=dict)  # Escalation thresholds
	
	# AI Learning and Improvement
	ml_model_version = Column(String(20), nullable=True)  # Version of ML model used
	prediction_accuracy = Column(Float, nullable=True)    # Historical prediction accuracy
	learning_feedback = Column(JSON, default=dict)       # Feedback for model improvement
	
	# Relationships
	category = relationship("GRCRiskCategory", back_populates="risks")
	assessments = relationship("GRCRiskAssessment", back_populates="risk")
	treatments = relationship("GRCRiskTreatment", back_populates="risk")
	indicators = relationship("GRCRiskIndicator", back_populates="risk")
	
	# Indexes for performance
	__table_args__ = (
		Index('idx_grc_risk_tenant_status', 'tenant_id', 'risk_status'),
		Index('idx_grc_risk_level_score', 'risk_level', 'residual_risk_score'),
		Index('idx_grc_risk_review_date', 'next_review_date'),
		Index('idx_grc_risk_owner_category', 'risk_owner_id', 'risk_category_id'),
	)
	
	def __repr__(self):
		return f"<GRCRisk {self.risk_title}>"
	
	@hybrid_property
	def risk_score_category(self) -> str:
		"""Determine risk level based on residual risk score"""
		if self.residual_risk_score >= 90:
			return GRCRiskLevel.CRITICAL.value
		elif self.residual_risk_score >= 70:
			return GRCRiskLevel.HIGH.value
		elif self.residual_risk_score >= 40:
			return GRCRiskLevel.MEDIUM.value
		elif self.residual_risk_score >= 20:
			return GRCRiskLevel.LOW.value
		else:
			return GRCRiskLevel.MINIMAL.value
	
	def calculate_inherent_risk_score(self) -> float:
		"""Calculate inherent risk score from probability and impact"""
		self.inherent_risk_score = self.inherent_probability * self.inherent_impact * 100
		return self.inherent_risk_score
	
	def calculate_residual_risk_score(self) -> float:
		"""Calculate residual risk score after controls"""
		self.residual_risk_score = self.residual_probability * self.residual_impact * 100
		return self.residual_risk_score
	
	def update_risk_level(self):
		"""Update risk level based on current risk score"""
		self.risk_level = self.risk_score_category
	
	def is_overdue_for_review(self) -> bool:
		"""Check if risk review is overdue"""
		if not self.next_review_date:
			return True
		return datetime.utcnow() > self.next_review_date
	
	def calculate_risk_velocity(self, historical_scores: List[float]) -> float:
		"""Calculate rate of risk change over time"""
		if len(historical_scores) < 2:
			return 0.0
		
		# Simple velocity calculation - can be enhanced with ML
		recent_change = historical_scores[-1] - historical_scores[-2]
		self.risk_velocity = recent_change
		return self.risk_velocity
	
	@validates('inherent_probability', 'inherent_impact', 'residual_probability', 'residual_impact')
	def validate_probability_impact(self, key, value):
		"""Validate probability and impact are between 0 and 1"""
		if value is not None and (value < 0.0 or value > 1.0):
			raise ValueError(f"{key} must be between 0.0 and 1.0")
		return value


class GRCRiskCategory(Model, AuditMixin, BaseMixin):
	"""
	Hierarchical Risk Taxonomy with AI-Powered Classification
	
	Intelligent risk categorization supporting multiple frameworks
	(ISO 31000, COSO ERM, industry-specific taxonomies).
	"""
	__tablename__ = 'grc_risk_category'
	
	# Identity
	category_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Category Definition
	category_name = Column(String(100), nullable=False, index=True)
	category_description = Column(Text, nullable=True)
	category_code = Column(String(20), nullable=True, index=True)  # Short code for reporting
	
	# Hierarchical Structure
	parent_category_id = Column(String(36), ForeignKey('grc_risk_category.category_id'), nullable=True, index=True)
	category_level = Column(Integer, nullable=False, default=1)  # Depth in hierarchy
	category_path = Column(String(500), nullable=True)  # Full path for queries
	
	# Framework Alignment
	iso31000_alignment = Column(String(50), nullable=True)  # ISO 31000 category mapping
	coso_erm_alignment = Column(String(50), nullable=True)  # COSO ERM framework mapping
	nist_alignment = Column(String(50), nullable=True)      # NIST framework mapping
	custom_framework = Column(JSON, default=dict)          # Custom framework mappings
	
	# AI-Powered Classification
	ai_classification_confidence = Column(Float, default=0.0)  # ML confidence in classification
	related_categories = Column(JSON, default=list)           # AI-identified related categories
	risk_pattern_indicators = Column(JSON, default=list)      # Common risk patterns
	
	# Category Intelligence
	typical_probability_range = Column(JSON, default=dict)    # {"min": 0.1, "max": 0.8}
	typical_impact_range = Column(JSON, default=dict)         # Typical impact range
	industry_benchmarks = Column(JSON, default=dict)          # Industry-specific benchmarks
	regulatory_focus_areas = Column(JSON, default=list)       # Regulatory attention areas
	
	# Category Status
	is_active = Column(Boolean, default=True, index=True)
	category_priority = Column(Integer, default=50)           # 1-100 priority scale
	review_frequency_days = Column(Integer, default=90)       # Recommended review frequency
	
	# Usage Analytics
	risk_count = Column(Integer, default=0)                   # Number of risks in category
	average_risk_score = Column(Float, default=0.0)          # Average risk score in category
	category_trend = Column(String(20), default="stable")    # increasing, decreasing, stable
	
	# Relationships
	parent_category = relationship("GRCRiskCategory", remote_side=[category_id])
	subcategories = relationship("GRCRiskCategory", back_populates="parent_category")
	risks = relationship("GRCRisk", back_populates="category")
	
	# Indexes
	__table_args__ = (
		Index('idx_grc_category_tenant_active', 'tenant_id', 'is_active'),
		Index('idx_grc_category_parent_level', 'parent_category_id', 'category_level'),
		Index('idx_grc_category_name_code', 'category_name', 'category_code'),
	)
	
	def __repr__(self):
		return f"<GRCRiskCategory {self.category_name}>"
	
	def get_full_path(self) -> str:
		"""Get full hierarchical path of the category"""
		if self.parent_category:
			return f"{self.parent_category.get_full_path()} > {self.category_name}"
		return self.category_name
	
	def update_category_path(self):
		"""Update the category path for efficient querying"""
		self.category_path = self.get_full_path()
	
	def calculate_category_statistics(self):
		"""Calculate category-level risk statistics"""
		if self.risks:
			self.risk_count = len(self.risks)
			active_risks = [r for r in self.risks if r.risk_status != GRCRiskStatus.CLOSED.value]
			if active_risks:
				self.average_risk_score = sum(r.residual_risk_score for r in active_risks) / len(active_risks)
	
	def get_subcategory_ids(self) -> List[str]:
		"""Get all subcategory IDs recursively"""
		subcategory_ids = [self.category_id]
		for subcategory in self.subcategories:
			subcategory_ids.extend(subcategory.get_subcategory_ids())
		return subcategory_ids


class GRCRiskAssessment(Model, AuditMixin, BaseMixin):
	"""
	AI-Enhanced Risk Assessment with Temporal Tracking
	
	Comprehensive risk assessment supporting both qualitative and
	quantitative approaches with ML-powered prediction capabilities.
	"""
	__tablename__ = 'grc_risk_assessment'
	
	# Identity
	assessment_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	risk_id = Column(String(36), ForeignKey('grc_risk.risk_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Assessment Metadata
	assessment_type = Column(String(20), nullable=False, index=True)  # initial, periodic, event_driven, continuous
	assessment_method = Column(String(30), nullable=False)  # qualitative, quantitative, semi_quantitative, ai_assisted
	assessor_id = Column(String(36), nullable=False, index=True)  # User who performed assessment
	assessment_date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
	
	# Risk Assessment Values
	probability_score = Column(Float, nullable=False)  # 0.0-1.0
	impact_score = Column(Float, nullable=False)       # 0.0-1.0
	overall_risk_score = Column(Float, nullable=False) # Calculated composite score
	confidence_level = Column(Float, default=0.7)     # Assessor confidence (0.0-1.0)
	
	# Detailed Impact Analysis
	financial_impact = Column(Float, nullable=True)        # Monetary impact
	operational_impact_score = Column(Float, nullable=True) # 0.0-1.0
	reputation_impact_score = Column(Float, nullable=True)  # 0.0-1.0
	regulatory_impact_score = Column(Float, nullable=True)  # 0.0-1.0
	strategic_impact_score = Column(Float, nullable=True)   # 0.0-1.0
	
	# AI-Powered Enhancement
	ai_probability_prediction = Column(Float, nullable=True)  # ML-predicted probability
	ai_impact_prediction = Column(Float, nullable=True)      # ML-predicted impact
	ai_confidence_score = Column(Float, nullable=True)       # AI prediction confidence
	ml_model_used = Column(String(50), nullable=True)        # Model identifier
	
	# Scenario Analysis
	best_case_scenario = Column(JSON, default=dict)    # Optimistic scenario
	worst_case_scenario = Column(JSON, default=dict)   # Pessimistic scenario
	most_likely_scenario = Column(JSON, default=dict)  # Expected scenario
	scenario_probabilities = Column(JSON, default=dict) # Probability distribution
	
	# Assessment Context
	assessment_scope = Column(Text, nullable=True)      # Scope and boundaries
	key_assumptions = Column(JSON, default=list)       # Assessment assumptions
	data_sources = Column(JSON, default=list)          # Information sources used
	limitations = Column(Text, nullable=True)          # Assessment limitations
	
	# Evidence and Documentation
	supporting_evidence = Column(JSON, default=list)   # Links to evidence
	assessment_notes = Column(Text, nullable=True)     # Detailed notes
	peer_review_status = Column(String(20), default="pending")  # pending, approved, rejected
	peer_reviewer_id = Column(String(36), nullable=True, index=True)
	
	# Quality Metrics
	assessment_quality_score = Column(Float, nullable=True)  # Quality rating (0.0-1.0)
	completeness_score = Column(Float, nullable=True)       # Data completeness
	consistency_score = Column(Float, nullable=True)        # Consistency with similar risks
	
	# Temporal Tracking
	valid_from = Column(DateTime, nullable=False, default=datetime.utcnow)
	valid_until = Column(DateTime, nullable=True)
	superseded_by = Column(String(36), ForeignKey('grc_risk_assessment.assessment_id'), nullable=True)
	
	# Relationships
	risk = relationship("GRCRisk", back_populates="assessments")
	superseding_assessment = relationship("GRCRiskAssessment", remote_side=[assessment_id])
	
	# Indexes
	__table_args__ = (
		Index('idx_grc_assessment_risk_date', 'risk_id', 'assessment_date'),
		Index('idx_grc_assessment_tenant_type', 'tenant_id', 'assessment_type'),
		Index('idx_grc_assessment_valid_period', 'valid_from', 'valid_until'),
	)
	
	def __repr__(self):
		return f"<GRCRiskAssessment {self.assessment_id} for Risk {self.risk_id}>"
	
	def calculate_overall_risk_score(self) -> float:
		"""Calculate composite risk score from probability and impact"""
		self.overall_risk_score = self.probability_score * self.impact_score * 100
		return self.overall_risk_score
	
	def calculate_composite_impact(self) -> float:
		"""Calculate weighted composite impact from multiple dimensions"""
		impact_scores = [
			self.impact_score,
			self.operational_impact_score or 0.0,
			self.reputation_impact_score or 0.0,
			self.regulatory_impact_score or 0.0,
			self.strategic_impact_score or 0.0
		]
		
		# Filter out None values and calculate weighted average
		valid_scores = [score for score in impact_scores if score is not None]
		if valid_scores:
			return sum(valid_scores) / len(valid_scores)
		return self.impact_score
	
	def is_current(self) -> bool:
		"""Check if assessment is currently valid"""
		now = datetime.utcnow()
		if self.valid_until and now > self.valid_until:
			return False
		return now >= self.valid_from
	
	def compare_with_ai_prediction(self) -> Dict[str, float]:
		"""Compare human assessment with AI prediction"""
		comparison = {}
		if self.ai_probability_prediction is not None:
			comparison['probability_variance'] = abs(self.probability_score - self.ai_probability_prediction)
		if self.ai_impact_prediction is not None:
			comparison['impact_variance'] = abs(self.impact_score - self.ai_impact_prediction)
		return comparison


# ==============================================================================
# COMPLIANCE MANAGEMENT MODELS
# ==============================================================================

class GRCRegulation(Model, AuditMixin, BaseMixin):
	"""
	Comprehensive Regulatory Framework with AI-Powered Monitoring
	
	Global regulatory database with intelligent change detection,
	impact analysis, and compliance requirement mapping.
	"""
	__tablename__ = 'grc_regulation'
	
	# Identity
	regulation_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Regulatory Identification
	regulation_name = Column(String(200), nullable=False, index=True)
	regulation_code = Column(String(50), nullable=True, index=True)  # Official code/number
	short_name = Column(String(100), nullable=True)
	regulation_type = Column(String(50), nullable=False, index=True)  # law, regulation, standard, guideline
	
	# Regulatory Authority
	issuing_authority = Column(String(200), nullable=False, index=True)
	jurisdiction = Column(String(100), nullable=False, index=True)  # Country, state, region
	authority_website = Column(String(500), nullable=True)
	official_publication = Column(String(200), nullable=True)
	
	# Regulatory Content
	regulation_summary = Column(Text, nullable=False)
	regulation_purpose = Column(Text, nullable=True)
	scope_and_applicability = Column(Text, nullable=True)
	key_requirements = Column(JSON, default=list)  # Structured requirements
	
	# Versioning and Changes
	regulation_version = Column(String(20), nullable=True)
	effective_date = Column(DateTime, nullable=False, index=True)
	last_amended_date = Column(DateTime, nullable=True)
	next_review_date = Column(DateTime, nullable=True, index=True)
	sunset_date = Column(DateTime, nullable=True)  # When regulation expires
	
	# AI-Powered Change Detection
	change_detection_enabled = Column(Boolean, default=True)
	last_change_scan = Column(DateTime, nullable=True)
	change_frequency_days = Column(Integer, default=30)  # How often to check for changes
	ai_change_confidence = Column(Float, default=0.0)   # AI confidence in detected changes
	detected_changes = Column(JSON, default=list)       # Recent changes detected
	
	# Business Impact Analysis
	business_processes_affected = Column(JSON, default=list)  # Processes impacted
	compliance_complexity = Column(String(20), default="medium")  # low, medium, high, very_high
	implementation_effort = Column(String(20), default="medium") # low, medium, high, very_high
	estimated_compliance_cost = Column(Float, nullable=True)     # Estimated cost to comply
	
	# Industry and Geographic Scope
	applicable_industries = Column(JSON, default=list)  # Industry codes affected
	geographic_scope = Column(JSON, default=list)       # Geographic applicability
	organization_size_scope = Column(JSON, default=list) # SME, large, all
	
	# Regulatory Intelligence
	related_regulations = Column(JSON, default=list)    # Related regulation IDs
	superseded_regulations = Column(JSON, default=list) # Regulations this replaces
	cross_references = Column(JSON, default=list)       # External references
	
	# Compliance Tracking
	compliance_status = Column(String(20), default=GRCComplianceStatus.NOT_ASSESSED.value, index=True)
	compliance_deadline = Column(DateTime, nullable=True, index=True)
	compliance_percentage = Column(Float, default=0.0)  # 0.0-100.0
	compliance_notes = Column(Text, nullable=True)
	
	# Document Management
	regulation_documents = Column(JSON, default=list)   # Links to regulation documents
	guidance_documents = Column(JSON, default=list)     # Implementation guidance
	interpretation_notes = Column(JSON, default=list)   # Legal interpretations
	
	# Risk and Impact
	non_compliance_penalties = Column(JSON, default=dict)  # Penalty structure
	enforcement_history = Column(JSON, default=list)      # Historical enforcement actions
	risk_rating = Column(String(20), default="medium")    # Regulatory risk level
	
	# Status and Metadata
	is_active = Column(Boolean, default=True, index=True)
	regulation_priority = Column(Integer, default=50)     # 1-100 priority scale
	monitoring_frequency = Column(String(20), default="monthly")  # daily, weekly, monthly, quarterly
	
	# Relationships
	controls = relationship("GRCControl", back_populates="regulation")
	compliance_records = relationship("GRCCompliance", back_populates="regulation")
	
	# Indexes
	__table_args__ = (
		Index('idx_grc_regulation_tenant_active', 'tenant_id', 'is_active'),
		Index('idx_grc_regulation_jurisdiction_type', 'jurisdiction', 'regulation_type'),
		Index('idx_grc_regulation_effective_date', 'effective_date'),
		Index('idx_grc_regulation_compliance_deadline', 'compliance_deadline'),
	)
	
	def __repr__(self):
		return f"<GRCRegulation {self.regulation_name}>"
	
	def is_currently_effective(self) -> bool:
		"""Check if regulation is currently in effect"""
		now = datetime.utcnow()
		if now < self.effective_date:
			return False
		if self.sunset_date and now > self.sunset_date:
			return False
		return True
	
	def days_until_deadline(self) -> Optional[int]:
		"""Calculate days until compliance deadline"""
		if not self.compliance_deadline:
			return None
		delta = self.compliance_deadline - datetime.utcnow()
		return delta.days
	
	def is_overdue_for_review(self) -> bool:
		"""Check if regulation review is overdue"""
		if not self.next_review_date:
			return False
		return datetime.utcnow() > self.next_review_date
	
	def calculate_compliance_risk(self) -> float:
		"""Calculate risk score based on compliance status and penalties"""
		base_risk = 50.0  # Medium baseline risk
		
		# Adjust based on compliance status
		if self.compliance_status == GRCComplianceStatus.NON_COMPLIANT.value:
			base_risk += 30.0
		elif self.compliance_status == GRCComplianceStatus.PARTIALLY_COMPLIANT.value:
			base_risk += 15.0
		elif self.compliance_status == GRCComplianceStatus.COMPLIANT.value:
			base_risk -= 20.0
		
		# Adjust based on deadline proximity
		days_to_deadline = self.days_until_deadline()
		if days_to_deadline is not None:
			if days_to_deadline < 30:
				base_risk += 20.0
			elif days_to_deadline < 90:
				base_risk += 10.0
		
		# Adjust based on penalty severity
		if self.non_compliance_penalties:
			penalty_severity = self.non_compliance_penalties.get('severity', 'medium')
			if penalty_severity == 'high':
				base_risk += 15.0
			elif penalty_severity == 'critical':
				base_risk += 25.0
		
		return min(100.0, max(0.0, base_risk))


class GRCControl(Model, AuditMixin, BaseMixin):
	"""
	Intelligent Control Framework with Self-Testing Capabilities
	
	Advanced control management with AI-powered effectiveness assessment,
	automated testing, and continuous optimization recommendations.
	"""
	__tablename__ = 'grc_control'
	
	# Identity
	control_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Control Definition
	control_name = Column(String(200), nullable=False, index=True)
	control_code = Column(String(50), nullable=True, index=True)  # Control reference number
	control_description = Column(Text, nullable=False)
	control_objective = Column(Text, nullable=False)
	
	# Control Classification
	control_type = Column(String(20), nullable=False, default=GRCControlType.PREVENTIVE.value, index=True)
	control_category = Column(String(50), nullable=False, index=True)  # IT, operational, financial, etc.
	control_family = Column(String(100), nullable=True)  # Related controls grouping
	
	# Regulatory Mapping
	regulation_id = Column(String(36), ForeignKey('grc_regulation.regulation_id'), nullable=True, index=True)
	regulatory_citations = Column(JSON, default=list)  # Specific regulatory references
	framework_mappings = Column(JSON, default=dict)    # NIST, ISO, COBIT mappings
	
	# Control Implementation
	control_owner_id = Column(String(36), nullable=False, index=True)  # Primary responsible person
	process_owner_id = Column(String(36), nullable=True, index=True)   # Process owner
	implementation_status = Column(String(20), default="designed", index=True)  # designed, implemented, operating
	implementation_date = Column(DateTime, nullable=True)
	
	# Control Design
	control_procedures = Column(Text, nullable=False)   # Detailed procedures
	control_frequency = Column(String(20), nullable=False, index=True)  # continuous, daily, weekly, monthly, quarterly, annual
	automation_level = Column(String(20), default="manual")  # manual, semi_automated, fully_automated
	control_triggers = Column(JSON, default=list)      # What triggers control execution
	
	# AI-Powered Self-Testing
	self_testing_enabled = Column(Boolean, default=False)
	last_self_test = Column(DateTime, nullable=True)
	self_test_frequency_hours = Column(Integer, default=24)  # How often to self-test
	self_test_results = Column(JSON, default=dict)     # Latest self-test results
	self_test_success_rate = Column(Float, default=0.0) # Historical success rate
	
	# Control Effectiveness
	design_effectiveness = Column(String(20), default="not_assessed")  # effective, needs_improvement, ineffective
	operating_effectiveness = Column(String(20), default="not_assessed")
	overall_effectiveness_score = Column(Float, default=0.0)  # 0.0-100.0
	effectiveness_trend = Column(String(20), default="stable")  # improving, stable, declining
	
	# Testing and Validation
	testing_methodology = Column(String(50), nullable=True)  # inquiry, observation, re_performance, etc.
	testing_frequency = Column(String(20), default="quarterly")
	last_testing_date = Column(DateTime, nullable=True, index=True)
	next_testing_date = Column(DateTime, nullable=True, index=True)
	testing_results = Column(JSON, default=dict)       # Latest testing results
	
	# Control Monitoring
	key_control_indicators = Column(JSON, default=list)  # KCIs for monitoring
	monitoring_frequency = Column(String(20), default="monthly")
	alert_thresholds = Column(JSON, default=dict)      # Threshold values for alerts
	monitoring_dashboard_url = Column(String(500), nullable=True)
	
	# Risk Mitigation
	risks_mitigated = Column(JSON, default=list)       # Risk IDs this control addresses
	risk_reduction_factor = Column(Float, default=0.0) # How much risk this control reduces (0.0-1.0)
	compensating_controls = Column(JSON, default=list) # Backup/compensating controls
	
	# Control Intelligence
	ai_effectiveness_prediction = Column(Float, nullable=True)  # AI-predicted effectiveness
	failure_prediction_score = Column(Float, default=0.0)      # Likelihood of control failure
	optimization_recommendations = Column(JSON, default=list)   # AI recommendations
	similar_controls = Column(JSON, default=list)              # Similar controls for benchmarking
	
	# Exception Management
	approved_exceptions = Column(JSON, default=list)   # Approved control exceptions
	exception_expiry_dates = Column(JSON, default=dict) # When exceptions expire
	temporary_workarounds = Column(JSON, default=list) # Temporary alternative controls
	
	# Documentation and Evidence
	control_documentation = Column(JSON, default=list) # Links to control documentation
	evidence_requirements = Column(JSON, default=list) # What evidence to collect
	latest_evidence = Column(JSON, default=dict)       # Most recent evidence
	evidence_retention_days = Column(Integer, default=2555)  # 7 years default
	
	# Performance Metrics
	control_cost = Column(Float, nullable=True)        # Cost to operate control
	control_efficiency = Column(Float, default=0.0)   # Cost-effectiveness ratio
	false_positive_rate = Column(Float, default=0.0)  # Rate of false alerts
	false_negative_rate = Column(Float, default=0.0)  # Rate of missed issues
	
	# Status and Lifecycle
	is_active = Column(Boolean, default=True, index=True)
	control_priority = Column(Integer, default=50)     # 1-100 priority scale
	retirement_date = Column(DateTime, nullable=True)  # When control will be retired
	replacement_control_id = Column(String(36), nullable=True) # Replacing control
	
	# Relationships
	regulation = relationship("GRCRegulation", back_populates="controls")
	compliance_records = relationship("GRCCompliance", back_populates="control")
	
	# Indexes
	__table_args__ = (
		Index('idx_grc_control_tenant_active', 'tenant_id', 'is_active'),
		Index('idx_grc_control_owner_category', 'control_owner_id', 'control_category'),
		Index('idx_grc_control_testing_date', 'next_testing_date'),
		Index('idx_grc_control_effectiveness', 'overall_effectiveness_score'),
	)
	
	def __repr__(self):
		return f"<GRCControl {self.control_name}>"
	
	def is_due_for_testing(self) -> bool:
		"""Check if control is due for testing"""
		if not self.next_testing_date:
			return True
		return datetime.utcnow() >= self.next_testing_date
	
	def calculate_next_testing_date(self):
		"""Calculate next testing date based on frequency"""
		if not self.last_testing_date:
			self.last_testing_date = datetime.utcnow()
		
		frequency_map = {
			'daily': 1,
			'weekly': 7,
			'monthly': 30,
			'quarterly': 90,
			'semi_annual': 180,
			'annual': 365
		}
		
		days_to_add = frequency_map.get(self.testing_frequency, 90)
		self.next_testing_date = self.last_testing_date + timedelta(days=days_to_add)
	
	def calculate_effectiveness_score(self) -> float:
		"""Calculate overall control effectiveness score"""
		design_score = {'effective': 100, 'needs_improvement': 60, 'ineffective': 20, 'not_assessed': 50}
		operating_score = {'effective': 100, 'needs_improvement': 60, 'ineffective': 20, 'not_assessed': 50}
		
		design_points = design_score.get(self.design_effectiveness, 50)
		operating_points = operating_score.get(self.operating_effectiveness, 50)
		
		# Weight design 40%, operating 60%
		self.overall_effectiveness_score = (design_points * 0.4) + (operating_points * 0.6)
		return self.overall_effectiveness_score
	
	def should_self_test(self) -> bool:
		"""Check if control should perform self-test"""
		if not self.self_testing_enabled:
			return False
		
		if not self.last_self_test:
			return True
		
		hours_since_test = (datetime.utcnow() - self.last_self_test).total_seconds() / 3600
		return hours_since_test >= self.self_test_frequency_hours
	
	def update_self_test_success_rate(self, test_passed: bool):
		"""Update rolling success rate for self-tests"""
		# Simple rolling average - can be enhanced with time-weighted calculations
		current_rate = self.self_test_success_rate or 0.0
		if test_passed:
			self.self_test_success_rate = min(1.0, current_rate + 0.1)
		else:
			self.self_test_success_rate = max(0.0, current_rate - 0.1)


# ==============================================================================
# GOVERNANCE MODELS
# ==============================================================================

class GRCPolicy(Model, AuditMixin, BaseMixin):
	"""
	AI-Assisted Corporate Policy Management
	
	Comprehensive policy lifecycle management with intelligent
	consistency checking, impact analysis, and stakeholder collaboration.
	"""
	__tablename__ = 'grc_policy'
	
	# Identity
	policy_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Policy Definition
	policy_title = Column(String(200), nullable=False, index=True)
	policy_number = Column(String(50), nullable=True, index=True)  # Policy reference number
	policy_type = Column(String(50), nullable=False, index=True)   # corporate, operational, technical, etc.
	policy_category = Column(String(100), nullable=False, index=True)  # HR, IT, Finance, Risk, etc.
	
	# Policy Content
	policy_purpose = Column(Text, nullable=False)
	policy_scope = Column(Text, nullable=False)
	policy_statement = Column(Text, nullable=False)     # Main policy content
	policy_procedures = Column(Text, nullable=True)     # Detailed procedures
	definitions = Column(JSON, default=dict)            # Key terms and definitions
	
	# Policy Hierarchy
	parent_policy_id = Column(String(36), ForeignKey('grc_policy.policy_id'), nullable=True, index=True)
	policy_level = Column(Integer, default=1)           # Hierarchy level
	related_policies = Column(JSON, default=list)       # Related policy IDs
	superseded_policies = Column(JSON, default=list)    # Policies this replaces
	
	# Governance and Ownership
	policy_owner_id = Column(String(36), nullable=False, index=True)    # Executive owner
	policy_steward_id = Column(String(36), nullable=False, index=True)  # Day-to-day manager
	approving_authority = Column(String(100), nullable=False)           # Board, CEO, etc.
	approval_date = Column(DateTime, nullable=True, index=True)
	
	# Lifecycle Management
	policy_version = Column(String(20), nullable=False, default="1.0")
	effective_date = Column(DateTime, nullable=False, index=True)
	last_review_date = Column(DateTime, nullable=True)
	next_review_date = Column(DateTime, nullable=False, index=True)
	review_frequency_months = Column(Integer, default=12)
	
	# AI-Powered Policy Intelligence
	ai_consistency_score = Column(Float, default=0.0)   # Consistency with related policies
	policy_complexity_score = Column(Float, default=0.0) # Readability and complexity
	compliance_alignment_score = Column(Float, default=0.0) # Alignment with regulations
	policy_gaps_identified = Column(JSON, default=list) # AI-identified gaps
	improvement_suggestions = Column(JSON, default=list) # AI suggestions
	
	# Stakeholder Management
	stakeholder_groups = Column(JSON, default=list)     # Affected stakeholder groups
	training_required = Column(Boolean, default=False)
	training_completion_rate = Column(Float, default=0.0) # 0.0-100.0
	acknowledgment_required = Column(Boolean, default=False)
	acknowledgment_rate = Column(Float, default=0.0)    # 0.0-100.0
	
	# Impact Analysis
	business_impact = Column(String(20), default="medium")  # low, medium, high, critical
	implementation_complexity = Column(String(20), default="medium")
	estimated_implementation_cost = Column(Float, nullable=True)
	affected_processes = Column(JSON, default=list)     # Business processes affected
	system_changes_required = Column(JSON, default=list) # IT system changes needed
	
	# Monitoring and Compliance
	monitoring_requirements = Column(JSON, default=list) # How to monitor compliance
	key_performance_indicators = Column(JSON, default=list) # Policy KPIs
	compliance_metrics = Column(JSON, default=dict)     # Current compliance metrics
	violation_tracking = Column(JSON, default=list)     # Policy violations
	
	# Documentation and Communication
	policy_documents = Column(JSON, default=list)       # Policy document links
	communication_plan = Column(JSON, default=dict)     # How policy is communicated
	training_materials = Column(JSON, default=list)     # Training resources
	frequently_asked_questions = Column(JSON, default=list) # FAQ items
	
	# Exception Management
	exception_criteria = Column(Text, nullable=True)    # When exceptions are allowed
	approved_exceptions = Column(JSON, default=list)    # Current exceptions
	exception_approval_process = Column(Text, nullable=True) # Exception workflow
	
	# Status and Workflow
	policy_status = Column(String(20), default="draft", index=True)  # draft, review, approved, active, retired
	workflow_stage = Column(String(30), nullable=True)  # Current workflow stage
	approval_workflow_id = Column(String(36), nullable=True)  # Workflow instance ID
	
	# Analytics and Usage
	policy_views = Column(Integer, default=0)           # Number of times viewed
	search_frequency = Column(Integer, default=0)       # How often searched
	feedback_score = Column(Float, default=0.0)         # User feedback score
	usage_analytics = Column(JSON, default=dict)        # Detailed usage metrics
	
	# Relationships
	parent_policy = relationship("GRCPolicy", remote_side=[policy_id])
	child_policies = relationship("GRCPolicy", back_populates="parent_policy")
	
	# Indexes
	__table_args__ = (
		Index('idx_grc_policy_tenant_status', 'tenant_id', 'policy_status'),
		Index('idx_grc_policy_category_type', 'policy_category', 'policy_type'),
		Index('idx_grc_policy_review_date', 'next_review_date'),
		Index('idx_grc_policy_owner_steward', 'policy_owner_id', 'policy_steward_id'),
	)
	
	def __repr__(self):
		return f"<GRCPolicy {self.policy_title}>"
	
	def is_due_for_review(self) -> bool:
		"""Check if policy is due for review"""
		return datetime.utcnow() >= self.next_review_date
	
	def calculate_next_review_date(self):
		"""Calculate next review date based on frequency"""
		base_date = self.last_review_date or self.effective_date or datetime.utcnow()
		self.next_review_date = base_date + timedelta(days=self.review_frequency_months * 30)
	
	def is_currently_effective(self) -> bool:
		"""Check if policy is currently in effect"""
		now = datetime.utcnow()
		return (self.policy_status == "active" and 
				self.effective_date <= now and
				self.approval_date is not None)
	
	def calculate_compliance_rate(self) -> float:
		"""Calculate overall policy compliance rate"""
		# Combine training completion and acknowledgment rates
		if self.training_required and self.acknowledgment_required:
			return (self.training_completion_rate + self.acknowledgment_rate) / 2
		elif self.training_required:
			return self.training_completion_rate
		elif self.acknowledgment_required:
			return self.acknowledgment_rate
		else:
			# Use violation tracking to estimate compliance
			if self.violation_tracking:
				# Simple calculation - can be enhanced
				recent_violations = len([v for v in self.violation_tracking if v.get('date', '') > (datetime.utcnow() - timedelta(days=90)).isoformat()])
				return max(0.0, 100.0 - (recent_violations * 5))  # Each violation reduces by 5%
		return 85.0  # Default assumption


class GRCGovernanceDecision(Model, AuditMixin, BaseMixin):
	"""
	Intelligent Governance Decision Management
	
	Comprehensive decision tracking with AI-powered impact analysis,
	stakeholder engagement, and outcome measurement.
	"""
	__tablename__ = 'grc_governance_decision'
	
	# Identity
	decision_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Decision Identification
	decision_title = Column(String(200), nullable=False, index=True)
	decision_type = Column(String(30), nullable=False, default=GRCGovernanceDecisionType.OPERATIONAL_CHANGE.value, index=True)
	decision_category = Column(String(50), nullable=False, index=True)  # strategic, operational, financial, risk
	decision_priority = Column(String(20), default="medium")  # low, medium, high, critical
	
	# Decision Context
	decision_description = Column(Text, nullable=False)
	business_rationale = Column(Text, nullable=False)
	decision_background = Column(Text, nullable=True)
	alternatives_considered = Column(JSON, default=list)    # Alternative options
	
	# Decision Making Process
	decision_maker_id = Column(String(36), nullable=False, index=True)  # Primary decision maker
	decision_committee = Column(String(100), nullable=True)  # Committee if applicable
	stakeholders_involved = Column(JSON, default=list)      # All stakeholders
	decision_criteria = Column(JSON, default=list)          # Decision criteria used
	
	# AI-Powered Decision Support
	ai_impact_analysis = Column(JSON, default=dict)         # AI-generated impact analysis
	ai_risk_assessment = Column(JSON, default=dict)         # AI risk evaluation
	ai_recommendation = Column(Text, nullable=True)         # AI recommendation
	ai_confidence_score = Column(Float, default=0.0)        # AI confidence in analysis
	similar_decisions = Column(JSON, default=list)          # Similar historical decisions
	
	# Decision Timeline
	decision_requested_date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
	decision_deadline = Column(DateTime, nullable=True, index=True)
	decision_made_date = Column(DateTime, nullable=True, index=True)
	implementation_start_date = Column(DateTime, nullable=True)
	implementation_target_date = Column(DateTime, nullable=True)
	
	# Decision Outcome
	decision_status = Column(String(20), default="pending", index=True)  # pending, approved, rejected, deferred, implemented
	decision_outcome = Column(Text, nullable=True)          # Final decision made
	decision_rationale = Column(Text, nullable=True)        # Reasoning for decision
	conditions_and_constraints = Column(JSON, default=list) # Conditions attached to decision
	
	# Impact Assessment
	expected_benefits = Column(JSON, default=list)          # Expected positive outcomes
	expected_risks = Column(JSON, default=list)             # Expected negative outcomes
	resource_requirements = Column(JSON, default=dict)      # Resources needed
	budget_impact = Column(Float, nullable=True)            # Financial impact
	timeline_impact = Column(String(100), nullable=True)    # Impact on timelines
	
	# Stakeholder Analysis
	stakeholder_impact_analysis = Column(JSON, default=dict) # Impact on each stakeholder
	stakeholder_positions = Column(JSON, default=dict)      # Stakeholder support/opposition
	communication_plan = Column(JSON, default=dict)         # How to communicate decision
	change_management_plan = Column(JSON, default=dict)     # Change management approach
	
	# Implementation Tracking
	implementation_plan = Column(JSON, default=list)        # Implementation steps
	implementation_progress = Column(Float, default=0.0)    # 0.0-100.0 completion
	implementation_milestones = Column(JSON, default=list)  # Key milestones
	implementation_issues = Column(JSON, default=list)      # Issues encountered
	
	# Monitoring and Review
	success_metrics = Column(JSON, default=list)            # How to measure success
	monitoring_frequency = Column(String(20), default="monthly") # How often to review
	next_review_date = Column(DateTime, nullable=True, index=True)
	performance_against_expectations = Column(JSON, default=dict) # Actual vs expected
	
	# Documentation and Evidence
	supporting_documents = Column(JSON, default=list)       # Links to supporting docs
	meeting_minutes = Column(JSON, default=list)            # Decision meeting records
	approval_evidence = Column(JSON, default=list)          # Approval documentation
	implementation_evidence = Column(JSON, default=list)    # Implementation proof
	
	# Learning and Improvement
	lessons_learned = Column(JSON, default=list)            # What was learned
	decision_quality_score = Column(Float, nullable=True)   # Quality rating (0.0-100.0)
	stakeholder_satisfaction = Column(Float, nullable=True) # Stakeholder satisfaction score
	would_decide_same_again = Column(Boolean, nullable=True) # Retrospective assessment
	
	# Relationships and Dependencies
	related_risks = Column(JSON, default=list)              # Risk IDs affected
	related_policies = Column(JSON, default=list)           # Policy IDs affected
	related_projects = Column(JSON, default=list)           # Project IDs affected
	dependent_decisions = Column(JSON, default=list)        # Decisions that depend on this
	
	# Indexes
	__table_args__ = (
		Index('idx_grc_decision_tenant_status', 'tenant_id', 'decision_status'),
		Index('idx_grc_decision_maker_date', 'decision_maker_id', 'decision_requested_date'),
		Index('idx_grc_decision_deadline', 'decision_deadline'),
		Index('idx_grc_decision_review_date', 'next_review_date'),
	)
	
	def __repr__(self):
		return f"<GRCGovernanceDecision {self.decision_title}>"
	
	def is_overdue(self) -> bool:
		"""Check if decision is overdue"""
		if not self.decision_deadline:
			return False
		return (self.decision_status == "pending" and 
				datetime.utcnow() > self.decision_deadline)
	
	def days_to_deadline(self) -> Optional[int]:
		"""Calculate days until decision deadline"""
		if not self.decision_deadline:
			return None
		delta = self.decision_deadline - datetime.utcnow()
		return delta.days
	
	def calculate_decision_velocity(self) -> Optional[float]:
		"""Calculate time taken to make decision (in days)"""
		if not self.decision_made_date:
			return None
		delta = self.decision_made_date - self.decision_requested_date
		return delta.total_seconds() / 86400  # Convert to days
	
	def calculate_implementation_progress_rate(self) -> Optional[float]:
		"""Calculate implementation progress rate (% per day)"""
		if not self.implementation_start_date or self.implementation_progress == 0:
			return None
		
		days_elapsed = (datetime.utcnow() - self.implementation_start_date).total_seconds() / 86400
		if days_elapsed <= 0:
			return None
		
		return self.implementation_progress / days_elapsed
	
	def is_due_for_review(self) -> bool:
		"""Check if decision is due for review"""
		if not self.next_review_date:
			return False
		return datetime.utcnow() >= self.next_review_date
	
	def calculate_stakeholder_alignment(self) -> float:
		"""Calculate overall stakeholder alignment score"""
		if not self.stakeholder_positions:
			return 50.0  # Neutral assumption
		
		positions = list(self.stakeholder_positions.values())
		if not positions:
			return 50.0
		
		# Convert positions to numeric scores (support=100, neutral=50, oppose=0)
		position_scores = []
		for position in positions:
			if position.lower() in ['support', 'strongly_support']:
				position_scores.append(100.0)
			elif position.lower() in ['neutral', 'undecided']:
				position_scores.append(50.0)
			elif position.lower() in ['oppose', 'strongly_oppose']:
				position_scores.append(0.0)
			else:
				position_scores.append(50.0)  # Default to neutral
		
		return sum(position_scores) / len(position_scores)


# ==============================================================================
# PYDANTIC MODELS FOR API SERIALIZATION
# ==============================================================================

class GRCRiskCreate(BaseModel):
	"""Pydantic model for creating new risks"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	risk_title: str = Field(..., min_length=1, max_length=200)
	risk_description: str = Field(..., min_length=1)
	risk_category_id: str = Field(..., min_length=36, max_length=36)
	risk_owner_id: str = Field(..., min_length=36, max_length=36)
	
	inherent_probability: Annotated[float, AfterValidator(lambda x: x if 0.0 <= x <= 1.0 else ValueError("Must be between 0.0 and 1.0"))]
	inherent_impact: Annotated[float, AfterValidator(lambda x: x if 0.0 <= x <= 1.0 else ValueError("Must be between 0.0 and 1.0"))]
	
	business_process: Optional[str] = None
	financial_impact_expected: Optional[float] = None
	risk_tags: List[str] = Field(default_factory=list)


class GRCRiskResponse(BaseModel):
	"""Pydantic model for risk API responses"""
	model_config = ConfigDict(from_attributes=True)
	
	risk_id: str
	risk_title: str
	risk_description: str
	risk_level: str
	risk_status: str
	inherent_risk_score: float
	residual_risk_score: float
	risk_owner_id: str
	next_review_date: Optional[datetime]
	created_at: datetime
	updated_at: datetime


class GRCComplianceCreate(BaseModel):
	"""Pydantic model for creating compliance records"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	regulation_id: str = Field(..., min_length=36, max_length=36)
	control_id: str = Field(..., min_length=36, max_length=36)
	compliance_status: str = Field(..., pattern="^(compliant|non_compliant|partially_compliant|not_assessed|pending_review|exception_approved)$")
	assessment_date: datetime
	assessor_id: str = Field(..., min_length=36, max_length=36)
	compliance_evidence: List[str] = Field(default_factory=list)
	compliance_notes: Optional[str] = None


class GRCPolicyCreate(BaseModel):
	"""Pydantic model for creating policies"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	policy_title: str = Field(..., min_length=1, max_length=200)
	policy_type: str = Field(..., min_length=1, max_length=50)
	policy_category: str = Field(..., min_length=1, max_length=100)
	policy_purpose: str = Field(..., min_length=1)
	policy_scope: str = Field(..., min_length=1)
	policy_statement: str = Field(..., min_length=1)
	policy_owner_id: str = Field(..., min_length=36, max_length=36)
	policy_steward_id: str = Field(..., min_length=36, max_length=36)
	effective_date: datetime
	review_frequency_months: int = Field(default=12, ge=1, le=60)


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def _log_pretty_path(path: str) -> str:
	"""Format file path for logging"""
	return path.replace('/Users/nyimbiodero/src/pjs/apg/', 'apg/')

def _log_model_creation(model_name: str, model_id: str, tenant_id: str):
	"""Log model creation for audit purposes"""
	print(f"Created {model_name} {model_id} for tenant {tenant_id}")

def _log_model_update(model_name: str, model_id: str, changes: Dict[str, Any]):
	"""Log model updates for audit purposes"""
	print(f"Updated {model_name} {model_id}: {list(changes.keys())}")

def calculate_risk_correlation(risk1: GRCRisk, risk2: GRCRisk) -> float:
	"""Calculate correlation score between two risks"""
	# Simplified correlation calculation - can be enhanced with ML
	correlation_score = 0.0
	
	# Category similarity
	if risk1.risk_category_id == risk2.risk_category_id:
		correlation_score += 0.3
	
	# Business process similarity
	if risk1.business_process and risk2.business_process:
		if risk1.business_process == risk2.business_process:
			correlation_score += 0.2
	
	# Owner similarity
	if risk1.risk_owner_id == risk2.risk_owner_id:
		correlation_score += 0.1
	
	# Tag similarity
	if risk1.risk_tags and risk2.risk_tags:
		common_tags = set(risk1.risk_tags) & set(risk2.risk_tags)
		tag_similarity = len(common_tags) / max(len(risk1.risk_tags), len(risk2.risk_tags))
		correlation_score += tag_similarity * 0.2
	
	# Geographic scope similarity
	if risk1.geographic_scope and risk2.geographic_scope:
		common_locations = set(risk1.geographic_scope) & set(risk2.geographic_scope)
		location_similarity = len(common_locations) / max(len(risk1.geographic_scope), len(risk2.geographic_scope))
		correlation_score += location_similarity * 0.2
	
	return min(1.0, correlation_score)

def predict_risk_emergence(risk_indicators: List[Dict[str, Any]], 
						   historical_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""Predict potential risk emergence based on indicators and patterns"""
	# Placeholder for ML-based risk emergence prediction
	# In production, this would use trained ML models
	
	emergence_probability = 0.0
	risk_factors = []
	
	# Analyze indicators for risk signals
	for indicator in risk_indicators:
		if indicator.get('trend', '') == 'increasing':
			emergence_probability += 0.1
			risk_factors.append(f"Increasing trend in {indicator.get('name', 'unknown indicator')}")
		
		if indicator.get('threshold_breach', False):
			emergence_probability += 0.2
			risk_factors.append(f"Threshold breach in {indicator.get('name', 'unknown indicator')}")
	
	# Analyze historical patterns
	for pattern in historical_patterns:
		if pattern.get('pattern_strength', 0) > 0.7:
			emergence_probability += 0.15
			risk_factors.append(f"Strong historical pattern: {pattern.get('description', 'unknown pattern')}")
	
	return {
		'emergence_probability': min(1.0, emergence_probability),
		'confidence_score': 0.7,  # Placeholder - would be calculated by ML model
		'risk_factors': risk_factors,
		'recommendation': 'Monitor closely' if emergence_probability > 0.5 else 'Continue standard monitoring'
	}

# Export models for use in other modules
__all__ = [
	'GRCRisk', 'GRCRiskCategory', 'GRCRiskAssessment',
	'GRCRegulation', 'GRCControl', 
	'GRCPolicy', 'GRCGovernanceDecision',
	'GRCRiskCreate', 'GRCRiskResponse', 'GRCComplianceCreate', 'GRCPolicyCreate',
	'GRCRiskLevel', 'GRCRiskStatus', 'GRCComplianceStatus', 'GRCControlType', 'GRCGovernanceDecisionType'
]