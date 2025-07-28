"""
APG Customer Relationship Management - GDPR/CCPA Compliance Framework

Comprehensive privacy regulation compliance system implementing GDPR, CCPA, 
and other global privacy laws with automated compliance monitoring, reporting,
and data subject rights management.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, field_validator
import xml.etree.ElementTree as ET

import asyncpg
import redis.asyncio as redis


logger = logging.getLogger(__name__)


# ================================
# Enums and Constants
# ================================

class PrivacyRegulation(Enum):
	GDPR = "gdpr"  # General Data Protection Regulation (EU)
	CCPA = "ccpa"  # California Consumer Privacy Act (US)
	LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
	PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)
	PDPA_SG = "pdpa_sg"  # Personal Data Protection Act (Singapore)
	PDPA_TH = "pdpa_th"  # Personal Data Protection Act (Thailand)
	DPA_UK = "dpa_uk"  # Data Protection Act (UK)
	APPI = "appi"  # Act on Protection of Personal Information (Japan)


class LegalBasis(Enum):
	# GDPR Article 6 Legal Bases
	CONSENT = "consent"
	CONTRACT = "contract"
	LEGAL_OBLIGATION = "legal_obligation"
	VITAL_INTERESTS = "vital_interests"
	PUBLIC_TASK = "public_task"
	LEGITIMATE_INTERESTS = "legitimate_interests"
	
	# CCPA Business Purposes
	BUSINESS_PURPOSE = "business_purpose"
	COMMERCIAL_PURPOSE = "commercial_purpose"


class DataCategory(Enum):
	# GDPR/CCPA Data Categories
	IDENTIFIERS = "identifiers"
	PERSONAL_INFO = "personal_info"
	PROTECTED_CHARACTERISTICS = "protected_characteristics"
	COMMERCIAL_INFO = "commercial_info"
	BIOMETRIC_INFO = "biometric_info"
	INTERNET_ACTIVITY = "internet_activity"
	GEOLOCATION = "geolocation"
	SENSORY_DATA = "sensory_data"
	PROFESSIONAL_INFO = "professional_info"
	EDUCATION_INFO = "education_info"
	INFERENCES = "inferences"
	SENSITIVE_PERSONAL_INFO = "sensitive_personal_info"


class ProcessingPurpose(Enum):
	# Common Processing Purposes
	CUSTOMER_SERVICE = "customer_service"
	MARKETING = "marketing"
	ANALYTICS = "analytics"
	FRAUD_PREVENTION = "fraud_prevention"
	LEGAL_COMPLIANCE = "legal_compliance"
	SECURITY = "security"
	PRODUCT_IMPROVEMENT = "product_improvement"
	RESEARCH = "research"
	PERSONALIZATION = "personalization"
	COMMUNICATION = "communication"


class DataSubjectRight(Enum):
	# GDPR Rights (Articles 15-22)
	ACCESS = "access"  # Article 15
	RECTIFICATION = "rectification"  # Article 16
	ERASURE = "erasure"  # Article 17 (Right to be forgotten)
	RESTRICT_PROCESSING = "restrict_processing"  # Article 18
	DATA_PORTABILITY = "data_portability"  # Article 20
	OBJECT = "object"  # Article 21
	AUTOMATED_DECISION_MAKING = "automated_decision_making"  # Article 22
	
	# CCPA Rights
	KNOW = "know"  # Right to know
	DELETE = "delete"  # Right to delete
	OPT_OUT = "opt_out"  # Right to opt-out of sale
	NON_DISCRIMINATION = "non_discrimination"  # Right to non-discrimination


class ConsentType(Enum):
	EXPLICIT = "explicit"
	IMPLICIT = "implicit"
	OPT_IN = "opt_in"
	OPT_OUT = "opt_out"
	GRANULAR = "granular"
	BLANKET = "blanket"


class ComplianceStatus(Enum):
	COMPLIANT = "compliant"
	NON_COMPLIANT = "non_compliant"
	PARTIAL_COMPLIANCE = "partial_compliance"
	UNDER_REVIEW = "under_review"
	REMEDIATION_REQUIRED = "remediation_required"


class RiskLevel(Enum):
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


# ================================
# Pydantic Models
# ================================

class DataProcessingActivity(BaseModel):
	"""GDPR Article 30 Processing Activity Record"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	activity_name: str
	description: str
	controller_name: str
	controller_contact: Dict[str, Any] = Field(default_factory=dict)
	dpo_contact: Optional[Dict[str, Any]] = None
	processor_details: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Processing Details
	legal_basis: List[LegalBasis] = Field(default_factory=list)
	purposes: List[ProcessingPurpose] = Field(default_factory=list)
	data_categories: List[DataCategory] = Field(default_factory=list)
	data_subject_categories: List[str] = Field(default_factory=list)
	recipients: List[Dict[str, Any]] = Field(default_factory=list)
	international_transfers: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Retention and Security
	retention_period: str
	security_measures: List[str] = Field(default_factory=list)
	
	# Automated Decision Making
	automated_decision_making: bool = False
	profiling: bool = False
	decision_logic: Optional[str] = None
	
	# Compliance
	dpia_required: bool = False
	dpia_completed: bool = False
	dpia_reference: Optional[str] = None
	
	# Metadata
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	last_reviewed_at: Optional[datetime] = None
	next_review_date: Optional[datetime] = None
	is_active: bool = True
	metadata: Dict[str, Any] = Field(default_factory=dict)


class ConsentManagement(BaseModel):
	"""Enhanced consent management record"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	data_subject_id: str
	data_subject_email: str
	
	# Consent Details
	consent_type: ConsentType
	consent_version: str
	consent_text: str
	purposes: List[ProcessingPurpose] = Field(default_factory=list)
	data_categories: List[DataCategory] = Field(default_factory=list)
	legal_basis: LegalBasis
	
	# Consent Status
	consent_given: bool
	consent_timestamp: datetime = Field(default_factory=datetime.utcnow)
	consent_method: str
	consent_source: str
	consent_evidence: Dict[str, Any] = Field(default_factory=dict)
	
	# Expiry and Withdrawal
	consent_expiry: Optional[datetime] = None
	auto_renewal: bool = False
	withdrawal_allowed: bool = True
	withdrawn_at: Optional[datetime] = None
	withdrawal_reason: Optional[str] = None
	
	# Granular Consent
	granular_consents: Dict[str, bool] = Field(default_factory=dict)
	
	# Compliance
	gdpr_compliant: bool = True
	ccpa_compliant: bool = True
	compliance_notes: Optional[str] = None
	
	# Metadata
	is_active: bool = True
	metadata: Dict[str, Any] = Field(default_factory=dict)


class PrivacyImpactAssessment(BaseModel):
	"""Data Protection Impact Assessment (DPIA)"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	assessment_name: str
	processing_activity_id: str
	
	# Assessment Details
	assessment_date: datetime = Field(default_factory=datetime.utcnow)
	assessed_by: str
	review_date: datetime
	
	# Risk Assessment
	likelihood_scores: Dict[str, int] = Field(default_factory=dict)
	impact_scores: Dict[str, int] = Field(default_factory=dict)
	overall_risk_level: RiskLevel
	risk_description: str
	
	# Identified Risks
	privacy_risks: List[Dict[str, Any]] = Field(default_factory=list)
	technical_risks: List[Dict[str, Any]] = Field(default_factory=list)
	organizational_risks: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Mitigation Measures
	existing_measures: List[str] = Field(default_factory=list)
	additional_measures: List[Dict[str, Any]] = Field(default_factory=list)
	residual_risk_level: RiskLevel
	
	# DPO/Stakeholder Consultation
	dpo_consulted: bool = False
	dpo_opinion: Optional[str] = None
	stakeholder_consultation: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Decision
	processing_approved: bool = False
	approval_conditions: List[str] = Field(default_factory=list)
	monitoring_requirements: List[str] = Field(default_factory=list)
	
	# Status
	status: str = "draft"
	completed_at: Optional[datetime] = None
	metadata: Dict[str, Any] = Field(default_factory=dict)


class DataBreach(BaseModel):
	"""Data breach incident record"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	breach_title: str
	breach_description: str
	
	# Breach Details
	discovered_at: datetime
	occurred_at: Optional[datetime] = None
	discovered_by: str
	breach_type: str
	breach_category: str
	
	# Impact Assessment
	data_subjects_affected: int = 0
	data_categories_affected: List[DataCategory] = Field(default_factory=list)
	likelihood_of_harm: RiskLevel
	severity_of_harm: RiskLevel
	overall_risk_assessment: RiskLevel
	
	# Personal Data Involved
	personal_data_types: List[str] = Field(default_factory=list)
	sensitive_data_involved: bool = False
	data_volume_estimate: str
	
	# Technical Details
	attack_vector: Optional[str] = None
	vulnerabilities_exploited: List[str] = Field(default_factory=list)
	systems_affected: List[str] = Field(default_factory=list)
	
	# Response Actions
	immediate_actions: List[str] = Field(default_factory=list)
	containment_measures: List[str] = Field(default_factory=list)
	recovery_actions: List[str] = Field(default_factory=list)
	
	# Notifications
	authority_notification_required: bool = False
	authority_notified_at: Optional[datetime] = None
	authority_reference: Optional[str] = None
	individual_notification_required: bool = False
	individuals_notified_at: Optional[datetime] = None
	
	# Legal and Regulatory
	regulations_applicable: List[PrivacyRegulation] = Field(default_factory=list)
	potential_fines: Decimal = Decimal('0.00')
	legal_actions: List[str] = Field(default_factory=list)
	
	# Closure
	incident_closed: bool = False
	closed_at: Optional[datetime] = None
	lessons_learned: Optional[str] = None
	preventive_measures: List[str] = Field(default_factory=list)
	
	# Metadata
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	metadata: Dict[str, Any] = Field(default_factory=dict)


class ComplianceAudit(BaseModel):
	"""Privacy compliance audit record"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	audit_name: str
	audit_type: str
	regulations_scope: List[PrivacyRegulation] = Field(default_factory=list)
	
	# Audit Details
	audit_date: datetime = Field(default_factory=datetime.utcnow)
	auditor_name: str
	audit_scope: List[str] = Field(default_factory=list)
	
	# Findings
	compliance_areas: List[Dict[str, Any]] = Field(default_factory=list)
	non_compliance_issues: List[Dict[str, Any]] = Field(default_factory=list)
	recommendations: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Scoring
	overall_compliance_score: Decimal = Decimal('0.00')
	area_scores: Dict[str, Decimal] = Field(default_factory=dict)
	compliance_status: ComplianceStatus
	
	# Actions
	corrective_actions: List[Dict[str, Any]] = Field(default_factory=list)
	follow_up_required: bool = False
	next_audit_date: Optional[datetime] = None
	
	# Metadata
	completed_at: Optional[datetime] = None
	report_generated: bool = False
	metadata: Dict[str, Any] = Field(default_factory=dict)


class DataSubjectRightsRequest(BaseModel):
	"""Enhanced data subject rights request"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	request_type: DataSubjectRight
	regulation: PrivacyRegulation
	
	# Requestor Information
	data_subject_id: str
	data_subject_email: str
	data_subject_name: Optional[str] = None
	requestor_relationship: str = "self"  # self, legal_representative, etc.
	
	# Request Details
	request_description: str
	specific_data_requested: List[str] = Field(default_factory=list)
	request_scope: Dict[str, Any] = Field(default_factory=dict)
	preferred_format: str = "json"
	
	# Verification
	identity_verification_method: str
	identity_verified: bool = False
	verification_documents: List[str] = Field(default_factory=list)
	verified_at: Optional[datetime] = None
	verified_by: Optional[str] = None
	
	# Processing
	received_at: datetime = Field(default_factory=datetime.utcnow)
	due_date: datetime
	assigned_to: Optional[str] = None
	processing_status: str = "received"
	status_updates: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Response
	response_data: Dict[str, Any] = Field(default_factory=dict)
	response_format: str = "json"
	response_sent_at: Optional[datetime] = None
	response_method: Optional[str] = None
	
	# Compliance Timelines
	acknowledgment_sent: bool = False
	acknowledgment_sent_at: Optional[datetime] = None
	extension_requested: bool = False
	extension_reason: Optional[str] = None
	extended_due_date: Optional[datetime] = None
	
	# Legal Considerations
	rejection_reason: Optional[str] = None
	legal_basis_for_rejection: Optional[str] = None
	fee_charged: Decimal = Decimal('0.00')
	fee_justification: Optional[str] = None
	
	# Quality Control
	review_required: bool = True
	reviewed_by: Optional[str] = None
	reviewed_at: Optional[datetime] = None
	quality_score: Optional[int] = None
	
	# Escalation
	escalated: bool = False
	escalation_reason: Optional[str] = None
	escalated_to: Optional[str] = None
	escalated_at: Optional[datetime] = None
	
	# Completion
	completed_at: Optional[datetime] = None
	satisfaction_rating: Optional[int] = None
	complainant_feedback: Optional[str] = None
	
	# Metadata
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	metadata: Dict[str, Any] = Field(default_factory=dict)


# ================================
# GDPR/CCPA Compliance Manager
# ================================

class GDPRCCPAComplianceManager:
	"""Comprehensive privacy regulation compliance system"""
	
	def __init__(self, db_pool: asyncpg.Pool, redis_client: redis.Redis = None):
		self.db_pool = db_pool
		self.redis_client = redis_client
		self._processing_activities = {}
		self._compliance_rules = {}
		self._notification_templates = {}
		self._initialized = False
	
	async def initialize(self):
		"""Initialize the compliance system"""
		try:
			if self._initialized:
				return
			
			logger.info("🏛️ Initializing GDPR/CCPA Compliance Manager...")
			
			# Validate database connection
			async with self.db_pool.acquire() as conn:
				await conn.execute("SELECT 1")
			
			# Load processing activities
			await self._load_processing_activities()
			
			# Load compliance rules
			await self._load_compliance_rules()
			
			# Load notification templates
			await self._load_notification_templates()
			
			# Start background tasks
			asyncio.create_task(self._consent_expiry_monitor())
			asyncio.create_task(self._compliance_monitoring_task())
			asyncio.create_task(self._data_retention_cleanup())
			
			self._initialized = True
			logger.info("✅ GDPR/CCPA Compliance Manager initialized successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to initialize compliance manager: {str(e)}")
			raise
	
	async def _load_processing_activities(self):
		"""Load data processing activities"""
		try:
			async with self.db_pool.acquire() as conn:
				activities = await conn.fetch("""
					SELECT * FROM crm_data_processing_activities 
					WHERE is_active = true
				""")
				
				for activity in activities:
					activity_data = dict(activity)
					activity_data['legal_basis'] = [LegalBasis(lb) for lb in json.loads(activity['legal_basis'] or '[]')]
					activity_data['purposes'] = [ProcessingPurpose(p) for p in json.loads(activity['purposes'] or '[]')]
					activity_data['data_categories'] = [DataCategory(dc) for dc in json.loads(activity['data_categories'] or '[]')]
					activity_data['data_subject_categories'] = json.loads(activity['data_subject_categories'] or '[]')
					activity_data['recipients'] = json.loads(activity['recipients'] or '[]')
					activity_data['international_transfers'] = json.loads(activity['international_transfers'] or '[]')
					activity_data['security_measures'] = json.loads(activity['security_measures'] or '[]')
					activity_data['controller_contact'] = json.loads(activity['controller_contact'] or '{}')
					activity_data['dpo_contact'] = json.loads(activity['dpo_contact'] or '{}') if activity['dpo_contact'] else None
					activity_data['processor_details'] = json.loads(activity['processor_details'] or '[]')
					activity_data['metadata'] = json.loads(activity['metadata'] or '{}')
					
					processing_activity = DataProcessingActivity(**activity_data)
					self._processing_activities[processing_activity.id] = processing_activity
				
				logger.info(f"📋 Loaded {len(self._processing_activities)} processing activities")
				
		except Exception as e:
			logger.error(f"Error loading processing activities: {str(e)}")
	
	async def _load_compliance_rules(self):
		"""Load compliance rules and requirements"""
		try:
			# Define standard compliance rules
			self._compliance_rules = {
				PrivacyRegulation.GDPR: {
					'consent_requirements': {
						'explicit_required': ['marketing', 'profiling', 'sensitive_data'],
						'withdrawal_mechanism': True,
						'granular_consent': True,
						'clear_language': True
					},
					'data_subject_rights': {
						'access': {'timeline_days': 30, 'format': ['json', 'csv', 'xml']},
						'rectification': {'timeline_days': 30, 'verification_required': True},
						'erasure': {'timeline_days': 30, 'exceptions': ['legal_obligation', 'freedom_of_expression']},
						'portability': {'timeline_days': 30, 'structured_format': True},
						'object': {'timeline_days': 30, 'legitimate_interests_assessment': True},
						'restrict_processing': {'timeline_days': 30, 'notification_required': True}
					},
					'breach_notification': {
						'authority_timeline_hours': 72,
						'individual_timeline_days': 30,
						'risk_threshold': 'high',
						'documentation_required': True
					},
					'dpo_requirements': {
						'public_authorities': True,
						'core_activities_monitoring': True,
						'large_scale_sensitive_data': True
					}
				},
				PrivacyRegulation.CCPA: {
					'consumer_rights': {
						'know': {'timeline_days': 45, 'twice_per_year': True},
						'delete': {'timeline_days': 45, 'verification_required': True},
						'opt_out': {'timeline_days': 15, 'global_privacy_control': True},
						'non_discrimination': {'price_differential_allowed': True}
					},
					'disclosure_requirements': {
						'categories_collected': True,
						'sources_of_data': True,
						'purposes_of_use': True,
						'categories_shared': True,
						'retention_periods': True
					},
					'sale_definition': {
						'broad_interpretation': True,
						'includes_advertising': True,
						'monetary_consideration': False
					}
				}
			}
			
			logger.info(f"📋 Loaded compliance rules for {len(self._compliance_rules)} regulations")
			
		except Exception as e:
			logger.error(f"Error loading compliance rules: {str(e)}")
	
	async def _load_notification_templates(self):
		"""Load notification templates"""
		try:
			self._notification_templates = {
				'gdpr_breach_authority': {
					'subject': 'Personal Data Breach Notification - Case #{case_id}',
					'template': 'gdpr_breach_authority_notification.html',
					'required_fields': ['nature_of_breach', 'data_categories', 'affected_individuals', 'consequences', 'measures_taken']
				},
				'gdpr_breach_individual': {
					'subject': 'Important Security Notice - Your Personal Data',
					'template': 'gdpr_breach_individual_notification.html',
					'required_fields': ['nature_of_breach', 'likely_consequences', 'measures_taken', 'contact_information']
				},
				'data_subject_response': {
					'subject': 'Response to Your Data Subject Rights Request',
					'template': 'data_subject_response.html',
					'required_fields': ['request_type', 'response_data', 'next_steps']
				}
			}
			
			logger.info(f"📨 Loaded {len(self._notification_templates)} notification templates")
			
		except Exception as e:
			logger.error(f"Error loading notification templates: {str(e)}")
	
	async def create_processing_activity(
		self,
		tenant_id: str,
		activity_name: str,
		description: str,
		controller_name: str,
		controller_contact: Dict[str, Any],
		legal_basis: List[LegalBasis],
		purposes: List[ProcessingPurpose],
		data_categories: List[DataCategory],
		retention_period: str,
		created_by: str,
		**kwargs
	) -> DataProcessingActivity:
		"""Create data processing activity record (GDPR Article 30)"""
		try:
			activity = DataProcessingActivity(
				tenant_id=tenant_id,
				activity_name=activity_name,
				description=description,
				controller_name=controller_name,
				controller_contact=controller_contact,
				legal_basis=legal_basis,
				purposes=purposes,
				data_categories=data_categories,
				retention_period=retention_period,
				created_by=created_by,
				**kwargs
			)
			
			# Determine if DPIA is required
			activity.dpia_required = self._assess_dpia_requirement(activity)
			
			# Store in database
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_data_processing_activities (
						id, tenant_id, activity_name, description, controller_name,
						controller_contact, dpo_contact, processor_details, legal_basis,
						purposes, data_categories, data_subject_categories, recipients,
						international_transfers, retention_period, security_measures,
						automated_decision_making, profiling, decision_logic,
						dpia_required, dpia_completed, dpia_reference,
						created_by, is_active, metadata
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25)
				""",
				activity.id, activity.tenant_id, activity.activity_name,
				activity.description, activity.controller_name,
				json.dumps(activity.controller_contact),
				json.dumps(activity.dpo_contact) if activity.dpo_contact else None,
				json.dumps(activity.processor_details),
				json.dumps([lb.value for lb in activity.legal_basis]),
				json.dumps([p.value for p in activity.purposes]),
				json.dumps([dc.value for dc in activity.data_categories]),
				json.dumps(activity.data_subject_categories),
				json.dumps(activity.recipients),
				json.dumps(activity.international_transfers),
				activity.retention_period,
				json.dumps(activity.security_measures),
				activity.automated_decision_making,
				activity.profiling,
				activity.decision_logic,
				activity.dpia_required,
				activity.dpia_completed,
				activity.dpia_reference,
				activity.created_by,
				activity.is_active,
				json.dumps(activity.metadata))
			
			# Cache the activity
			self._processing_activities[activity.id] = activity
			
			logger.info(f"📋 Created processing activity: {activity_name}")
			return activity
			
		except Exception as e:
			logger.error(f"Error creating processing activity: {str(e)}")
			raise
	
	def _assess_dpia_requirement(self, activity: DataProcessingActivity) -> bool:
		"""Assess if DPIA is required (GDPR Article 35)"""
		# DPIA is required if processing is likely to result in high risk
		high_risk_indicators = [
			activity.automated_decision_making,
			activity.profiling,
			DataCategory.SENSITIVE_PERSONAL_INFO in activity.data_categories,
			DataCategory.BIOMETRIC_INFO in activity.data_categories,
			len(activity.international_transfers) > 0,
			'large_scale' in activity.description.lower(),
			'systematic_monitoring' in activity.description.lower(),
			'vulnerable_individuals' in activity.description.lower()
		]
		
		return sum(high_risk_indicators) >= 2
	
	async def record_consent(
		self,
		tenant_id: str,
		data_subject_id: str,
		data_subject_email: str,
		consent_type: ConsentType,
		consent_version: str,
		consent_text: str,
		purposes: List[ProcessingPurpose],
		data_categories: List[DataCategory],
		legal_basis: LegalBasis,
		consent_given: bool,
		consent_method: str,
		consent_source: str,
		consent_evidence: Dict[str, Any] = None,
		granular_consents: Dict[str, bool] = None
	) -> ConsentManagement:
		"""Record data subject consent"""
		try:
			consent = ConsentManagement(
				tenant_id=tenant_id,
				data_subject_id=data_subject_id,
				data_subject_email=data_subject_email,
				consent_type=consent_type,
				consent_version=consent_version,
				consent_text=consent_text,
				purposes=purposes,
				data_categories=data_categories,
				legal_basis=legal_basis,
				consent_given=consent_given,
				consent_method=consent_method,
				consent_source=consent_source,
				consent_evidence=consent_evidence or {},
				granular_consents=granular_consents or {}
			)
			
			# Validate consent compliance
			consent.gdpr_compliant = self._validate_gdpr_consent(consent)
			consent.ccpa_compliant = self._validate_ccpa_consent(consent)
			
			# Store in database
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_consent_management (
						id, tenant_id, data_subject_id, data_subject_email,
						consent_type, consent_version, consent_text, purposes,
						data_categories, legal_basis, consent_given, consent_timestamp,
						consent_method, consent_source, consent_evidence, consent_expiry,
						auto_renewal, withdrawal_allowed, granular_consents,
						gdpr_compliant, ccpa_compliant, compliance_notes, is_active, metadata
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
				""",
				consent.id, consent.tenant_id, consent.data_subject_id,
				consent.data_subject_email, consent.consent_type.value,
				consent.consent_version, consent.consent_text,
				json.dumps([p.value for p in consent.purposes]),
				json.dumps([dc.value for dc in consent.data_categories]),
				consent.legal_basis.value, consent.consent_given,
				consent.consent_timestamp, consent.consent_method,
				consent.consent_source, json.dumps(consent.consent_evidence),
				consent.consent_expiry, consent.auto_renewal,
				consent.withdrawal_allowed, json.dumps(consent.granular_consents),
				consent.gdpr_compliant, consent.ccpa_compliant,
				consent.compliance_notes, consent.is_active, json.dumps(consent.metadata))
			
			logger.info(f"📝 Recorded consent: {data_subject_email} ({consent_type.value})")
			return consent
			
		except Exception as e:
			logger.error(f"Error recording consent: {str(e)}")
			raise
	
	def _validate_gdpr_consent(self, consent: ConsentManagement) -> bool:
		"""Validate GDPR consent requirements"""
		# GDPR Article 7 - Conditions for consent
		validations = [
			consent.consent_given is not None,  # Unambiguous indication
			consent.consent_method in ['checkbox', 'button_click', 'verbal', 'written'],
			consent.withdrawal_allowed,  # Right to withdraw
			len(consent.consent_text) > 0,  # Clear and plain language
			consent.consent_evidence is not None  # Evidence of consent
		]
		
		# Additional requirements for explicit consent
		if consent.consent_type == ConsentType.EXPLICIT:
			validations.extend([
				consent.consent_method in ['checkbox', 'button_click', 'written'],
				'explicit' in consent.consent_text.lower(),
				len(consent.granular_consents) > 0  # Granular consent options
			])
		
		return all(validations)
	
	def _validate_ccpa_consent(self, consent: ConsentManagement) -> bool:
		"""Validate CCPA consent requirements"""
		# CCPA focuses more on opt-out rather than opt-in
		validations = [
			consent.consent_given is not None,
			consent.withdrawal_allowed,
			len(consent.consent_text) > 0,
			consent.consent_evidence is not None
		]
		
		return all(validations)
	
	async def create_data_subject_request(
		self,
		tenant_id: str,
		request_type: DataSubjectRight,
		regulation: PrivacyRegulation,
		data_subject_id: str,
		data_subject_email: str,
		request_description: str,
		identity_verification_method: str,
		specific_data_requested: List[str] = None,
		preferred_format: str = "json"
	) -> DataSubjectRightsRequest:
		"""Create data subject rights request"""
		try:
			# Calculate due date based on regulation
			timeline_days = self._get_response_timeline(regulation, request_type)
			due_date = datetime.utcnow() + timedelta(days=timeline_days)
			
			request = DataSubjectRightsRequest(
				tenant_id=tenant_id,
				request_type=request_type,
				regulation=regulation,
				data_subject_id=data_subject_id,
				data_subject_email=data_subject_email,
				request_description=request_description,
				specific_data_requested=specific_data_requested or [],
				preferred_format=preferred_format,
				identity_verification_method=identity_verification_method,
				due_date=due_date
			)
			
			# Store in database
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_data_subject_rights_requests (
						id, tenant_id, request_type, regulation, data_subject_id,
						data_subject_email, data_subject_name, requestor_relationship,
						request_description, specific_data_requested, request_scope,
						preferred_format, identity_verification_method, identity_verified,
						verification_documents, received_at, due_date, processing_status,
						status_updates, response_format, acknowledgment_sent,
						extension_requested, fee_charged, review_required, escalated,
						created_at, metadata
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27)
				""",
				request.id, request.tenant_id, request.request_type.value,
				request.regulation.value, request.data_subject_id,
				request.data_subject_email, request.data_subject_name,
				request.requestor_relationship, request.request_description,
				json.dumps(request.specific_data_requested),
				json.dumps(request.request_scope), request.preferred_format,
				request.identity_verification_method, request.identity_verified,
				json.dumps(request.verification_documents), request.received_at,
				request.due_date, request.processing_status,
				json.dumps(request.status_updates), request.response_format,
				request.acknowledgment_sent, request.extension_requested,
				request.fee_charged, request.review_required, request.escalated,
				request.created_at, json.dumps(request.metadata))
			
			# Send acknowledgment
			await self._send_acknowledgment(request)
			
			logger.info(f"📨 Created data subject request: {request_type.value} for {data_subject_email}")
			return request
			
		except Exception as e:
			logger.error(f"Error creating data subject request: {str(e)}")
			raise
	
	def _get_response_timeline(self, regulation: PrivacyRegulation, request_type: DataSubjectRight) -> int:
		"""Get response timeline based on regulation and request type"""
		timelines = {
			PrivacyRegulation.GDPR: {
				DataSubjectRight.ACCESS: 30,
				DataSubjectRight.RECTIFICATION: 30,
				DataSubjectRight.ERASURE: 30,
				DataSubjectRight.RESTRICT_PROCESSING: 30,
				DataSubjectRight.DATA_PORTABILITY: 30,
				DataSubjectRight.OBJECT: 30
			},
			PrivacyRegulation.CCPA: {
				DataSubjectRight.KNOW: 45,
				DataSubjectRight.DELETE: 45,
				DataSubjectRight.OPT_OUT: 15
			}
		}
		
		return timelines.get(regulation, {}).get(request_type, 30)
	
	async def _send_acknowledgment(self, request: DataSubjectRightsRequest):
		"""Send acknowledgment of data subject request"""
		try:
			# Update acknowledgment status
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					UPDATE crm_data_subject_rights_requests 
					SET acknowledgment_sent = true, acknowledgment_sent_at = NOW()
					WHERE id = $1
				""", request.id)
			
			# In production, this would send actual email
			logger.info(f"📧 Sent acknowledgment for request {request.id}")
			
		except Exception as e:
			logger.error(f"Error sending acknowledgment: {str(e)}")
	
	async def create_data_breach_record(
		self,
		tenant_id: str,
		breach_title: str,
		breach_description: str,
		discovered_at: datetime,
		discovered_by: str,
		breach_type: str,
		data_subjects_affected: int,
		data_categories_affected: List[DataCategory],
		likelihood_of_harm: RiskLevel,
		severity_of_harm: RiskLevel,
		created_by: str,
		**kwargs
	) -> DataBreach:
		"""Create data breach incident record"""
		try:
			# Calculate overall risk assessment
			risk_matrix = {
				(RiskLevel.LOW, RiskLevel.LOW): RiskLevel.LOW,
				(RiskLevel.LOW, RiskLevel.MEDIUM): RiskLevel.MEDIUM,
				(RiskLevel.LOW, RiskLevel.HIGH): RiskLevel.MEDIUM,
				(RiskLevel.MEDIUM, RiskLevel.LOW): RiskLevel.MEDIUM,
				(RiskLevel.MEDIUM, RiskLevel.MEDIUM): RiskLevel.HIGH,
				(RiskLevel.MEDIUM, RiskLevel.HIGH): RiskLevel.HIGH,
				(RiskLevel.HIGH, RiskLevel.LOW): RiskLevel.MEDIUM,
				(RiskLevel.HIGH, RiskLevel.MEDIUM): RiskLevel.HIGH,
				(RiskLevel.HIGH, RiskLevel.HIGH): RiskLevel.CRITICAL
			}
			
			overall_risk = risk_matrix.get((likelihood_of_harm, severity_of_harm), RiskLevel.MEDIUM)
			
			breach = DataBreach(
				tenant_id=tenant_id,
				breach_title=breach_title,
				breach_description=breach_description,
				discovered_at=discovered_at,
				discovered_by=discovered_by,
				breach_type=breach_type,
				data_subjects_affected=data_subjects_affected,
				data_categories_affected=data_categories_affected,
				likelihood_of_harm=likelihood_of_harm,
				severity_of_harm=severity_of_harm,
				overall_risk_assessment=overall_risk,
				created_by=created_by,
				**kwargs
			)
			
			# Determine notification requirements
			breach.authority_notification_required = self._assess_authority_notification_requirement(breach)
			breach.individual_notification_required = self._assess_individual_notification_requirement(breach)
			
			# Store in database
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_data_breaches (
						id, tenant_id, breach_title, breach_description, discovered_at,
						occurred_at, discovered_by, breach_type, breach_category,
						data_subjects_affected, data_categories_affected, likelihood_of_harm,
						severity_of_harm, overall_risk_assessment, personal_data_types,
						sensitive_data_involved, data_volume_estimate, attack_vector,
						vulnerabilities_exploited, systems_affected, immediate_actions,
						containment_measures, recovery_actions, authority_notification_required,
						individual_notification_required, regulations_applicable,
						potential_fines, legal_actions, incident_closed, lessons_learned,
						preventive_measures, created_by, metadata
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33)
				""",
				breach.id, breach.tenant_id, breach.breach_title, breach.breach_description,
				breach.discovered_at, breach.occurred_at, breach.discovered_by,
				breach.breach_type, breach.breach_category, breach.data_subjects_affected,
				json.dumps([dc.value for dc in breach.data_categories_affected]),
				breach.likelihood_of_harm.value, breach.severity_of_harm.value,
				breach.overall_risk_assessment.value, json.dumps(breach.personal_data_types),
				breach.sensitive_data_involved, breach.data_volume_estimate,
				breach.attack_vector, json.dumps(breach.vulnerabilities_exploited),
				json.dumps(breach.systems_affected), json.dumps(breach.immediate_actions),
				json.dumps(breach.containment_measures), json.dumps(breach.recovery_actions),
				breach.authority_notification_required, breach.individual_notification_required,
				json.dumps([r.value for r in breach.regulations_applicable]),
				breach.potential_fines, json.dumps(breach.legal_actions),
				breach.incident_closed, breach.lessons_learned,
				json.dumps(breach.preventive_measures), breach.created_by,
				json.dumps(breach.metadata))
			
			# Trigger notifications if required
			if breach.authority_notification_required:
				await self._schedule_authority_notification(breach)
			
			if breach.individual_notification_required:
				await self._schedule_individual_notifications(breach)
			
			logger.warning(f"🚨 Data breach recorded: {breach_title} (Risk: {overall_risk.value})")
			return breach
			
		except Exception as e:
			logger.error(f"Error creating data breach record: {str(e)}")
			raise
	
	def _assess_authority_notification_requirement(self, breach: DataBreach) -> bool:
		"""Assess if authority notification is required"""
		# GDPR: Notify within 72 hours if likely to result in risk to rights and freedoms
		# CCPA: No general breach notification requirement to authorities
		
		if breach.overall_risk_assessment in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
			return True
		
		# Sensitive data always requires notification
		sensitive_categories = [
			DataCategory.BIOMETRIC_INFO,
			DataCategory.SENSITIVE_PERSONAL_INFO
		]
		
		if any(cat in breach.data_categories_affected for cat in sensitive_categories):
			return True
		
		# Large number of affected individuals
		if breach.data_subjects_affected > 1000:
			return True
		
		return False
	
	def _assess_individual_notification_requirement(self, breach: DataBreach) -> bool:
		"""Assess if individual notification is required"""
		# GDPR: Notify individuals if high risk to rights and freedoms
		# CCPA: Notify individuals if unauthorized access/disclosure
		
		if breach.overall_risk_assessment == RiskLevel.CRITICAL:
			return True
		
		# High risk scenarios
		high_risk_indicators = [
			breach.sensitive_data_involved,
			'identity_theft' in breach.breach_description.lower(),
			'financial' in breach.breach_description.lower(),
			breach.data_subjects_affected > 100
		]
		
		return sum(high_risk_indicators) >= 2
	
	async def _schedule_authority_notification(self, breach: DataBreach):
		"""Schedule authority notification for data breach"""
		try:
			# In production, this would integrate with regulatory systems
			logger.warning(f"📋 Authority notification scheduled for breach {breach.id}")
			
			# Update deadline (72 hours for GDPR)
			notification_deadline = breach.discovered_at + timedelta(hours=72)
			
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					UPDATE crm_data_breaches 
					SET authority_notification_required = true
					WHERE id = $1
				""", breach.id)
			
		except Exception as e:
			logger.error(f"Error scheduling authority notification: {str(e)}")
	
	async def _schedule_individual_notifications(self, breach: DataBreach):
		"""Schedule individual notifications for data breach"""
		try:
			# In production, this would queue individual notifications
			logger.warning(f"📧 Individual notifications scheduled for breach {breach.id}")
			
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					UPDATE crm_data_breaches 
					SET individual_notification_required = true
					WHERE id = $1
				""", breach.id)
			
		except Exception as e:
			logger.error(f"Error scheduling individual notifications: {str(e)}")
	
	async def conduct_compliance_audit(
		self,
		tenant_id: str,
		audit_name: str,
		audit_type: str,
		regulations_scope: List[PrivacyRegulation],
		auditor_name: str,
		audit_scope: List[str]
	) -> ComplianceAudit:
		"""Conduct privacy compliance audit"""
		try:
			audit = ComplianceAudit(
				tenant_id=tenant_id,
				audit_name=audit_name,
				audit_type=audit_type,
				regulations_scope=regulations_scope,
				auditor_name=auditor_name,
				audit_scope=audit_scope
			)
			
			# Perform audit checks
			audit_results = await self._perform_compliance_checks(tenant_id, regulations_scope, audit_scope)
			
			audit.compliance_areas = audit_results['compliance_areas']
			audit.non_compliance_issues = audit_results['non_compliance_issues']
			audit.recommendations = audit_results['recommendations']
			audit.overall_compliance_score = audit_results['overall_score']
			audit.area_scores = audit_results['area_scores']
			audit.compliance_status = audit_results['status']
			
			# Generate corrective actions
			audit.corrective_actions = self._generate_corrective_actions(audit.non_compliance_issues)
			audit.follow_up_required = len(audit.corrective_actions) > 0
			
			# Store in database
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_compliance_audits (
						id, tenant_id, audit_name, audit_type, regulations_scope,
						audit_date, auditor_name, audit_scope, compliance_areas,
						non_compliance_issues, recommendations, overall_compliance_score,
						area_scores, compliance_status, corrective_actions,
						follow_up_required, next_audit_date, report_generated, metadata
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
				""",
				audit.id, audit.tenant_id, audit.audit_name, audit.audit_type,
				json.dumps([r.value for r in audit.regulations_scope]),
				audit.audit_date, audit.auditor_name, json.dumps(audit.audit_scope),
				json.dumps(audit.compliance_areas), json.dumps(audit.non_compliance_issues),
				json.dumps(audit.recommendations), audit.overall_compliance_score,
				json.dumps({k: float(v) for k, v in audit.area_scores.items()}),
				audit.compliance_status.value, json.dumps(audit.corrective_actions),
				audit.follow_up_required, audit.next_audit_date,
				audit.report_generated, json.dumps(audit.metadata))
			
			logger.info(f"📊 Completed compliance audit: {audit_name} (Score: {audit.overall_compliance_score})")
			return audit
			
		except Exception as e:
			logger.error(f"Error conducting compliance audit: {str(e)}")
			raise
	
	async def _perform_compliance_checks(
		self,
		tenant_id: str,
		regulations: List[PrivacyRegulation],
		scope: List[str]
	) -> Dict[str, Any]:
		"""Perform detailed compliance checks"""
		try:
			compliance_areas = []
			non_compliance_issues = []
			recommendations = []
			area_scores = {}
			
			async with self.db_pool.acquire() as conn:
				# Check consent management
				consent_score = await self._audit_consent_management(conn, tenant_id)
				area_scores['consent_management'] = consent_score
				
				if consent_score < 80:
					non_compliance_issues.append({
						'area': 'consent_management',
						'issue': 'Inadequate consent management processes',
						'severity': 'high',
						'regulation': 'GDPR Article 7'
					})
				
				# Check data subject rights
				rights_score = await self._audit_data_subject_rights(conn, tenant_id)
				area_scores['data_subject_rights'] = rights_score
				
				if rights_score < 75:
					non_compliance_issues.append({
						'area': 'data_subject_rights',
						'issue': 'Insufficient data subject rights implementation',
						'severity': 'high',
						'regulation': 'GDPR Articles 15-22'
					})
				
				# Check data protection measures
				protection_score = await self._audit_data_protection(conn, tenant_id)
				area_scores['data_protection'] = protection_score
				
				if protection_score < 85:
					non_compliance_issues.append({
						'area': 'data_protection',
						'issue': 'Insufficient technical and organizational measures',
						'severity': 'medium',
						'regulation': 'GDPR Article 32'
					})
				
				# Check breach management
				breach_score = await self._audit_breach_management(conn, tenant_id)
				area_scores['breach_management'] = breach_score
				
				if breach_score < 70:
					non_compliance_issues.append({
						'area': 'breach_management',
						'issue': 'Inadequate breach detection and response procedures',
						'severity': 'high',
						'regulation': 'GDPR Articles 33-34'
					})
			
			# Calculate overall score
			overall_score = Decimal(str(sum(area_scores.values()) / len(area_scores)))
			
			# Determine status
			if overall_score >= 90:
				status = ComplianceStatus.COMPLIANT
			elif overall_score >= 75:
				status = ComplianceStatus.PARTIAL_COMPLIANCE
			else:
				status = ComplianceStatus.NON_COMPLIANT
			
			return {
				'compliance_areas': compliance_areas,
				'non_compliance_issues': non_compliance_issues,
				'recommendations': recommendations,
				'overall_score': overall_score,
				'area_scores': area_scores,
				'status': status
			}
			
		except Exception as e:
			logger.error(f"Error performing compliance checks: {str(e)}")
			raise
	
	async def _audit_consent_management(self, conn, tenant_id: str) -> Decimal:
		"""Audit consent management compliance"""
		try:
			# Check consent records
			total_consents = await conn.fetchval("""
				SELECT COUNT(*) FROM crm_consent_management 
				WHERE tenant_id = $1
			""", tenant_id)
			
			if total_consents == 0:
				return Decimal('50.0')  # No consent management
			
			# Check GDPR compliance
			gdpr_compliant = await conn.fetchval("""
				SELECT COUNT(*) FROM crm_consent_management 
				WHERE tenant_id = $1 AND gdpr_compliant = true
			""", tenant_id)
			
			# Check withdrawal mechanisms
			withdrawal_enabled = await conn.fetchval("""
				SELECT COUNT(*) FROM crm_consent_management 
				WHERE tenant_id = $1 AND withdrawal_allowed = true
			""", tenant_id)
			
			# Calculate score
			gdpr_ratio = gdpr_compliant / total_consents
			withdrawal_ratio = withdrawal_enabled / total_consents
			
			score = (gdpr_ratio * 60) + (withdrawal_ratio * 40)
			return Decimal(str(score * 100))
			
		except Exception as e:
			logger.error(f"Error auditing consent management: {str(e)}")
			return Decimal('0.0')
	
	async def _audit_data_subject_rights(self, conn, tenant_id: str) -> Decimal:
		"""Audit data subject rights implementation"""
		try:
			# Check if there are any requests
			total_requests = await conn.fetchval("""
				SELECT COUNT(*) FROM crm_data_subject_rights_requests 
				WHERE tenant_id = $1
			""", tenant_id)
			
			if total_requests == 0:
				return Decimal('75.0')  # No requests to evaluate
			
			# Check response times
			on_time_responses = await conn.fetchval("""
				SELECT COUNT(*) FROM crm_data_subject_rights_requests 
				WHERE tenant_id = $1 AND completed_at <= due_date
			""", tenant_id)
			
			completed_requests = await conn.fetchval("""
				SELECT COUNT(*) FROM crm_data_subject_rights_requests 
				WHERE tenant_id = $1 AND completed_at IS NOT NULL
			""", tenant_id)
			
			# Calculate scores
			completion_ratio = completed_requests / total_requests if total_requests > 0 else 0
			timeliness_ratio = on_time_responses / completed_requests if completed_requests > 0 else 0
			
			score = (completion_ratio * 50) + (timeliness_ratio * 50)
			return Decimal(str(score * 100))
			
		except Exception as e:
			logger.error(f"Error auditing data subject rights: {str(e)}")
			return Decimal('0.0')
	
	async def _audit_data_protection(self, conn, tenant_id: str) -> Decimal:
		"""Audit data protection measures"""
		try:
			# Check encryption
			encrypted_fields = await conn.fetchval("""
				SELECT COUNT(*) FROM crm_field_encryption_rules 
				WHERE tenant_id = $1 AND is_active = true
			""", tenant_id)
			
			# Check access controls
			access_controls = await conn.fetchval("""
				SELECT COUNT(*) FROM crm_user_roles 
				WHERE tenant_id = $1 AND is_active = true
			""", tenant_id)
			
			# Check audit logging
			audit_entries = await conn.fetchval("""
				SELECT COUNT(*) FROM crm_audit_entries 
				WHERE tenant_id = $1 AND timestamp > NOW() - INTERVAL '30 days'
			""", tenant_id)
			
			# Score based on presence of controls
			encryption_score = min(encrypted_fields * 10, 40)
			access_score = min(access_controls * 5, 30)
			audit_score = min(audit_entries / 100, 30)
			
			total_score = encryption_score + access_score + audit_score
			return Decimal(str(total_score))
			
		except Exception as e:
			logger.error(f"Error auditing data protection: {str(e)}")
			return Decimal('0.0')
	
	async def _audit_breach_management(self, conn, tenant_id: str) -> Decimal:
		"""Audit breach management procedures"""
		try:
			# Check if breach procedures exist
			total_breaches = await conn.fetchval("""
				SELECT COUNT(*) FROM crm_data_breaches 
				WHERE tenant_id = $1
			""", tenant_id)
			
			if total_breaches == 0:
				return Decimal('80.0')  # No breaches to evaluate (good)
			
			# Check notification compliance
			timely_notifications = await conn.fetchval("""
				SELECT COUNT(*) FROM crm_data_breaches 
				WHERE tenant_id = $1 
				AND authority_notification_required = true 
				AND authority_notified_at <= discovered_at + INTERVAL '72 hours'
			""", tenant_id)
			
			required_notifications = await conn.fetchval("""
				SELECT COUNT(*) FROM crm_data_breaches 
				WHERE tenant_id = $1 AND authority_notification_required = true
			""", tenant_id)
			
			# Check documentation
			documented_breaches = await conn.fetchval("""
				SELECT COUNT(*) FROM crm_data_breaches 
				WHERE tenant_id = $1 AND lessons_learned IS NOT NULL
			""", tenant_id)
			
			# Calculate scores
			notification_ratio = timely_notifications / required_notifications if required_notifications > 0 else 1
			documentation_ratio = documented_breaches / total_breaches if total_breaches > 0 else 0
			
			score = (notification_ratio * 70) + (documentation_ratio * 30)
			return Decimal(str(score * 100))
			
		except Exception as e:
			logger.error(f"Error auditing breach management: {str(e)}")
			return Decimal('0.0')
	
	def _generate_corrective_actions(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Generate corrective actions for compliance issues"""
		actions = []
		
		for issue in issues:
			if issue['area'] == 'consent_management':
				actions.append({
					'action': 'Implement granular consent mechanisms',
					'priority': 'high',
					'timeline_days': 30,
					'responsible_party': 'Privacy Officer',
					'resources_required': ['Legal review', 'Technical implementation']
				})
			
			elif issue['area'] == 'data_subject_rights':
				actions.append({
					'action': 'Establish automated data subject request processing',
					'priority': 'high',
					'timeline_days': 45,
					'responsible_party': 'IT Department',
					'resources_required': ['System integration', 'Process documentation']
				})
			
			elif issue['area'] == 'data_protection':
				actions.append({
					'action': 'Enhance encryption and access controls',
					'priority': 'medium',
					'timeline_days': 60,
					'responsible_party': 'Security Team',
					'resources_required': ['Security assessment', 'Implementation plan']
				})
			
			elif issue['area'] == 'breach_management':
				actions.append({
					'action': 'Develop incident response procedures',
					'priority': 'high',
					'timeline_days': 21,
					'responsible_party': 'CISO',
					'resources_required': ['Policy development', 'Staff training']
				})
		
		return actions
	
	async def _consent_expiry_monitor(self):
		"""Monitor and handle consent expiry"""
		while True:
			try:
				await asyncio.sleep(86400)  # Check daily
				
				async with self.db_pool.acquire() as conn:
					# Find expired consents
					expired_consents = await conn.fetch("""
						SELECT id, data_subject_id, data_subject_email, purposes
						FROM crm_consent_management 
						WHERE consent_expiry < NOW() 
						AND is_active = true
						AND auto_renewal = false
					""")
					
					for consent in expired_consents:
						# Mark as inactive
						await conn.execute("""
							UPDATE crm_consent_management 
							SET is_active = false, updated_at = NOW()
							WHERE id = $1
						""", consent['id'])
						
						# Log audit event
						logger.info(f"⏰ Consent expired for {consent['data_subject_email']}: {consent['id']}")
						
						# In production, would trigger data processing review
			
			except Exception as e:
				logger.error(f"Error in consent expiry monitor: {str(e)}")
	
	async def _compliance_monitoring_task(self):
		"""Continuous compliance monitoring"""
		while True:
			try:
				await asyncio.sleep(3600)  # Check hourly
				
				# Monitor for high-risk activities
				async with self.db_pool.acquire() as conn:
					# Check for unauthorized access patterns
					suspicious_access = await conn.fetch("""
						SELECT user_id, COUNT(*) as access_count
						FROM crm_audit_entries 
						WHERE event_type = 'unauthorized_access'
						AND timestamp > NOW() - INTERVAL '1 hour'
						GROUP BY user_id
						HAVING COUNT(*) > 5
					""")
					
					for access in suspicious_access:
						logger.warning(f"🚨 Suspicious access pattern detected: User {access['user_id']}")
						
						# In production, would trigger security response
			
			except Exception as e:
				logger.error(f"Error in compliance monitoring: {str(e)}")
	
	async def _data_retention_cleanup(self):
		"""Automated data retention cleanup"""
		while True:
			try:
				await asyncio.sleep(86400)  # Run daily
				
				async with self.db_pool.acquire() as conn:
					# Clean up expired data based on retention policies
					cleanup_result = await conn.execute("""
						DELETE FROM crm_audit_entries 
						WHERE timestamp < NOW() - INTERVAL '1 day' * retention_period_days
						AND legal_hold = false
						AND status = 'archived'
					""")
					
					if cleanup_result != "DELETE 0":
						logger.info(f"🗑️ Data retention cleanup: {cleanup_result}")
			
			except Exception as e:
				logger.error(f"Error in data retention cleanup: {str(e)}")
	
	async def generate_compliance_report(
		self,
		tenant_id: str,
		regulation: PrivacyRegulation,
		start_date: datetime,
		end_date: datetime
	) -> Dict[str, Any]:
		"""Generate comprehensive compliance report"""
		try:
			report = {
				'report_id': uuid7str(),
				'tenant_id': tenant_id,
				'regulation': regulation.value,
				'period': {
					'start_date': start_date.isoformat(),
					'end_date': end_date.isoformat()
				},
				'generated_at': datetime.utcnow().isoformat(),
				'sections': {}
			}
			
			async with self.db_pool.acquire() as conn:
				# Processing activities summary
				report['sections']['processing_activities'] = await self._report_processing_activities(conn, tenant_id)
				
				# Consent management summary
				report['sections']['consent_management'] = await self._report_consent_management(conn, tenant_id, start_date, end_date)
				
				# Data subject requests summary
				report['sections']['data_subject_requests'] = await self._report_data_subject_requests(conn, tenant_id, start_date, end_date)
				
				# Data breaches summary
				report['sections']['data_breaches'] = await self._report_data_breaches(conn, tenant_id, start_date, end_date)
				
				# Compliance status
				report['sections']['compliance_status'] = await self._report_compliance_status(conn, tenant_id)
			
			logger.info(f"📊 Generated compliance report for {regulation.value}")
			return report
			
		except Exception as e:
			logger.error(f"Error generating compliance report: {str(e)}")
			raise
	
	async def _report_processing_activities(self, conn, tenant_id: str) -> Dict[str, Any]:
		"""Generate processing activities report section"""
		try:
			activities = await conn.fetch("""
				SELECT activity_name, purposes, data_categories, legal_basis,
					   automated_decision_making, dpia_required, dpia_completed
				FROM crm_data_processing_activities 
				WHERE tenant_id = $1 AND is_active = true
			""", tenant_id)
			
			return {
				'total_activities': len(activities),
				'dpia_required': sum(1 for a in activities if a['dpia_required']),
				'dpia_completed': sum(1 for a in activities if a['dpia_completed']),
				'automated_processing': sum(1 for a in activities if a['automated_decision_making']),
				'activities': [
					{
						'name': a['activity_name'],
						'purposes': json.loads(a['purposes'] or '[]'),
						'data_categories': json.loads(a['data_categories'] or '[]'),
						'legal_basis': json.loads(a['legal_basis'] or '[]')
					}
					for a in activities
				]
			}
			
		except Exception as e:
			logger.error(f"Error reporting processing activities: {str(e)}")
			return {}
	
	async def _report_consent_management(self, conn, tenant_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
		"""Generate consent management report section"""
		try:
			consents = await conn.fetch("""
				SELECT consent_type, consent_given, gdpr_compliant, ccpa_compliant,
					   withdrawn_at, purposes
				FROM crm_consent_management 
				WHERE tenant_id = $1 
				AND consent_timestamp BETWEEN $2 AND $3
			""", tenant_id, start_date, end_date)
			
			total_consents = len(consents)
			consents_given = sum(1 for c in consents if c['consent_given'])
			consents_withdrawn = sum(1 for c in consents if c['withdrawn_at'])
			gdpr_compliant = sum(1 for c in consents if c['gdpr_compliant'])
			
			return {
				'total_consents': total_consents,
				'consents_given': consents_given,
				'consents_withdrawn': consents_withdrawn,
				'withdrawal_rate': (consents_withdrawn / total_consents * 100) if total_consents > 0 else 0,
				'gdpr_compliance_rate': (gdpr_compliant / total_consents * 100) if total_consents > 0 else 100
			}
			
		except Exception as e:
			logger.error(f"Error reporting consent management: {str(e)}")
			return {}
	
	async def _report_data_subject_requests(self, conn, tenant_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
		"""Generate data subject requests report section"""
		try:
			requests = await conn.fetch("""
				SELECT request_type, processing_status, completed_at, due_date,
					   identity_verified, response_provided
				FROM crm_data_subject_rights_requests 
				WHERE tenant_id = $1 
				AND received_at BETWEEN $2 AND $3
			""", tenant_id, start_date, end_date)
			
			total_requests = len(requests)
			completed_requests = sum(1 for r in requests if r['completed_at'])
			on_time_completion = sum(1 for r in requests if r['completed_at'] and r['completed_at'] <= r['due_date'])
			
			return {
				'total_requests': total_requests,
				'completed_requests': completed_requests,
				'completion_rate': (completed_requests / total_requests * 100) if total_requests > 0 else 0,
				'on_time_completion_rate': (on_time_completion / completed_requests * 100) if completed_requests > 0 else 0,
				'request_types': {
					request_type.value: sum(1 for r in requests if r['request_type'] == request_type.value)
					for request_type in DataSubjectRight
				}
			}
			
		except Exception as e:
			logger.error(f"Error reporting data subject requests: {str(e)}")
			return {}
	
	async def _report_data_breaches(self, conn, tenant_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
		"""Generate data breaches report section"""
		try:
			breaches = await conn.fetch("""
				SELECT breach_type, overall_risk_assessment, data_subjects_affected,
					   authority_notification_required, authority_notified_at,
					   individual_notification_required, individuals_notified_at,
					   discovered_at, incident_closed
				FROM crm_data_breaches 
				WHERE tenant_id = $1 
				AND discovered_at BETWEEN $2 AND $3
			""", tenant_id, start_date, end_date)
			
			total_breaches = len(breaches)
			high_risk_breaches = sum(1 for b in breaches if b['overall_risk_assessment'] in ['high', 'critical'])
			authority_notifications = sum(1 for b in breaches if b['authority_notification_required'])
			timely_notifications = sum(1 for b in breaches 
				if b['authority_notification_required'] and b['authority_notified_at'] 
				and b['authority_notified_at'] <= b['discovered_at'] + timedelta(hours=72))
			
			return {
				'total_breaches': total_breaches,
				'high_risk_breaches': high_risk_breaches,
				'authority_notifications_required': authority_notifications,
				'timely_notification_rate': (timely_notifications / authority_notifications * 100) if authority_notifications > 0 else 100,
				'total_affected_individuals': sum(b['data_subjects_affected'] or 0 for b in breaches),
				'breach_types': {}
			}
			
		except Exception as e:
			logger.error(f"Error reporting data breaches: {str(e)}")
			return {}
	
	async def _report_compliance_status(self, conn, tenant_id: str) -> Dict[str, Any]:
		"""Generate compliance status report section"""
		try:
			# Get latest audit results
			latest_audit = await conn.fetchrow("""
				SELECT overall_compliance_score, compliance_status, area_scores
				FROM crm_compliance_audits 
				WHERE tenant_id = $1 
				ORDER BY audit_date DESC 
				LIMIT 1
			""", tenant_id)
			
			if latest_audit:
				return {
					'overall_score': float(latest_audit['overall_compliance_score']),
					'status': latest_audit['compliance_status'],
					'area_scores': json.loads(latest_audit['area_scores'] or '{}'),
					'last_audit_date': latest_audit.get('audit_date', '').isoformat() if latest_audit.get('audit_date') else None
				}
			else:
				return {
					'overall_score': 0,
					'status': 'not_assessed',
					'area_scores': {},
					'last_audit_date': None
				}
			
		except Exception as e:
			logger.error(f"Error reporting compliance status: {str(e)}")
			return {}
	
	async def health_check(self) -> Dict[str, Any]:
		"""Perform health check"""
		try:
			health_status = {
				'status': 'healthy',
				'timestamp': datetime.utcnow().isoformat(),
				'components': {}
			}
			
			# Check database connection
			try:
				async with self.db_pool.acquire() as conn:
					result = await conn.fetchrow("SELECT COUNT(*) as count FROM crm_data_processing_activities")
				health_status['components']['database'] = 'healthy'
				health_status['components']['processing_activities'] = result['count']
			except Exception as e:
				health_status['components']['database'] = f'unhealthy: {str(e)}'
				health_status['status'] = 'degraded'
			
			# Check loaded components
			health_status['components']['loaded_activities'] = len(self._processing_activities)
			health_status['components']['compliance_rules'] = len(self._compliance_rules)
			health_status['components']['notification_templates'] = len(self._notification_templates)
			
			return health_status
			
		except Exception as e:
			return {
				'status': 'unhealthy',
				'error': str(e),
				'timestamp': datetime.utcnow().isoformat()
			}