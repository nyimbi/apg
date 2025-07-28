"""
APG Security Compliance Monitoring - Pydantic Models

Enterprise compliance monitoring models with automated assessments,
regulatory framework support, and continuous compliance tracking.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator
from pydantic import ConfigDict
from uuid_extensions import uuid7str


class ComplianceFramework(str, Enum):
	SOC2 = "soc2"
	ISO27001 = "iso27001"
	NIST_CSF = "nist_csf"
	PCI_DSS = "pci_dss"
	HIPAA = "hipaa"
	GDPR = "gdpr"
	CCPA = "ccpa"
	FISMA = "fisma"
	SOX = "sox"
	COBIT = "cobit"
	CIS_CONTROLS = "cis_controls"
	NIST_800_53 = "nist_800_53"


class ComplianceStatus(str, Enum):
	COMPLIANT = "compliant"
	NON_COMPLIANT = "non_compliant"
	PARTIALLY_COMPLIANT = "partially_compliant"
	NOT_ASSESSED = "not_assessed"
	IN_REMEDIATION = "in_remediation"
	EXCEPTION_GRANTED = "exception_granted"


class AssessmentType(str, Enum):
	AUTOMATED = "automated"
	MANUAL = "manual"
	HYBRID = "hybrid"
	EXTERNAL_AUDIT = "external_audit"
	SELF_ASSESSMENT = "self_assessment"
	CONTINUOUS = "continuous"


class ControlType(str, Enum):
	PREVENTIVE = "preventive"
	DETECTIVE = "detective"
	CORRECTIVE = "corrective"
	DIRECTIVE = "directive"
	COMPENSATING = "compensating"


class RiskLevel(str, Enum):
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	NEGLIGIBLE = "negligible"


class ComplianceControl(BaseModel):
	"""Individual compliance control definition and tracking"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	# Control identification
	control_id: str = Field(description="Unique control identifier")
	control_name: str = Field(description="Control name")
	control_description: str = Field(description="Detailed control description")
	
	# Framework mapping
	framework: ComplianceFramework
	framework_version: str = Field(description="Framework version")
	control_family: str = Field(description="Control family or domain")
	control_category: str = Field(description="Control category")
	
	# Control characteristics
	control_type: ControlType
	control_objective: str = Field(description="Control objective statement")
	control_frequency: str = Field(description="Control execution frequency")
	
	# Implementation details
	implementation_guidance: str = Field(description="Implementation guidance")
	testing_procedures: List[str] = Field(default_factory=list)
	evidence_requirements: List[str] = Field(default_factory=list)
	
	# Current status
	compliance_status: ComplianceStatus = ComplianceStatus.NOT_ASSESSED
	last_assessment_date: Optional[datetime] = None
	next_assessment_due: Optional[datetime] = None
	
	# Risk and priority
	risk_level: RiskLevel = RiskLevel.MEDIUM
	business_criticality: str = Field(default="medium")
	regulatory_requirement: bool = False
	
	# Ownership and responsibility
	control_owner: str = Field(description="Control owner")
	responsible_team: Optional[str] = None
	business_owner: Optional[str] = None
	
	# Automated monitoring
	automated_monitoring: bool = False
	monitoring_tools: List[str] = Field(default_factory=list)
	monitoring_frequency: Optional[str] = None
	
	# Exceptions and compensating controls
	has_exceptions: bool = False
	exception_details: List[str] = Field(default_factory=list)
	compensating_controls: List[str] = Field(default_factory=list)
	
	# Documentation and evidence
	documentation_links: List[str] = Field(default_factory=list)
	evidence_artifacts: List[str] = Field(default_factory=list)
	procedures_documented: bool = False
	
	# Compliance history
	compliance_trend: str = Field(default="stable")  # improving, degrading, stable
	historical_assessments: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Remediation tracking
	remediation_required: bool = False
	remediation_plan_id: Optional[str] = None
	target_remediation_date: Optional[datetime] = None
	
	is_active: bool = True
	created_by: str = Field(description="Control creator")
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class ComplianceAssessment(BaseModel):
	"""Compliance assessment execution and results"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	assessment_name: str = Field(description="Assessment name")
	assessment_description: str = Field(description="Assessment description")
	assessment_type: AssessmentType
	
	# Scope and framework
	framework: ComplianceFramework
	framework_version: str = Field(description="Framework version")
	assessment_scope: List[str] = Field(default_factory=list)
	controls_assessed: List[str] = Field(default_factory=list)
	
	# Assessment execution
	assessment_period_start: datetime
	assessment_period_end: datetime
	actual_start_date: Optional[datetime] = None
	actual_end_date: Optional[datetime] = None
	
	# Team and resources
	lead_assessor: str = Field(description="Lead assessor")
	assessment_team: List[str] = Field(default_factory=list)
	external_assessors: List[str] = Field(default_factory=list)
	
	# Assessment methodology
	assessment_methodology: str = Field(description="Assessment methodology")
	testing_procedures: List[str] = Field(default_factory=list)
	sampling_approach: Optional[str] = None
	
	# Results summary
	total_controls_assessed: int = 0
	controls_compliant: int = 0
	controls_non_compliant: int = 0
	controls_partially_compliant: int = 0
	overall_compliance_score: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Findings and gaps
	critical_findings: int = 0
	high_findings: int = 0
	medium_findings: int = 0
	low_findings: int = 0
	total_findings: int = 0
	
	# Assessment quality
	evidence_sufficiency: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	testing_completeness: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	documentation_quality: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Deliverables
	assessment_report_path: Optional[str] = None
	executive_summary_path: Optional[str] = None
	detailed_findings_path: Optional[str] = None
	remediation_plan_path: Optional[str] = None
	
	# Status and approval
	assessment_status: str = Field(default="planning")
	draft_report_completed: bool = False
	final_report_approved: bool = False
	approved_by: Optional[str] = None
	approval_date: Optional[datetime] = None
	
	# Follow-up and remediation
	remediation_required: bool = False
	remediation_deadline: Optional[datetime] = None
	follow_up_assessment_required: bool = False
	next_assessment_due: Optional[datetime] = None
	
	# External audit context
	external_audit: bool = False
	audit_firm: Optional[str] = None
	audit_partner: Optional[str] = None
	regulatory_filing_required: bool = False
	
	created_by: str = Field(description="Assessment creator")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class ComplianceFinding(BaseModel):
	"""Individual compliance finding and deficiency tracking"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	assessment_id: str = Field(description="Associated assessment")
	control_id: str = Field(description="Associated control")
	
	finding_title: str = Field(description="Finding title")
	finding_description: str = Field(description="Detailed finding description")
	finding_type: str = Field(description="Type of finding")
	
	# Severity and risk
	severity: RiskLevel = RiskLevel.MEDIUM
	risk_rating: Decimal = Field(default=Decimal('50.0'), ge=0, le=100)
	business_impact: str = Field(description="Business impact assessment")
	
	# Root cause analysis
	root_cause: Optional[str] = None
	contributing_factors: List[str] = Field(default_factory=list)
	systemic_issue: bool = False
	
	# Evidence and testing
	testing_performed: str = Field(description="Testing performed")
	evidence_reviewed: List[str] = Field(default_factory=list)
	sample_size: Optional[int] = None
	population_size: Optional[int] = None
	
	# Recommendation and remediation
	recommendation: str = Field(description="Remediation recommendation")
	remediation_priority: RiskLevel = RiskLevel.MEDIUM
	estimated_effort: Optional[str] = None
	target_resolution_date: Optional[datetime] = None
	
	# Assignment and ownership
	assigned_to: Optional[str] = None
	responsible_team: Optional[str] = None
	business_owner: Optional[str] = None
	
	# Remediation tracking
	remediation_status: str = Field(default="open")
	remediation_plan: Optional[str] = None
	remediation_started: Optional[datetime] = None
	remediation_completed: Optional[datetime] = None
	
	# Validation and closure
	management_response: Optional[str] = None
	corrective_action_plan: Optional[str] = None
	validation_required: bool = True
	validated_by: Optional[str] = None
	validation_date: Optional[datetime] = None
	
	# Finding relationships
	related_findings: List[str] = Field(default_factory=list)
	recurring_finding: bool = False
	previous_occurrence: Optional[str] = None
	
	# Regulatory and legal implications
	regulatory_citation: Optional[str] = None
	legal_implications: List[str] = Field(default_factory=list)
	external_reporting_required: bool = False
	
	# Quality metrics
	finding_accuracy: Optional[Decimal] = Field(None, ge=0, le=100)
	stakeholder_agreement: bool = False
	
	created_by: str = Field(description="Finding creator")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class ComplianceException(BaseModel):
	"""Compliance exception and risk acceptance tracking"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	control_id: str = Field(description="Associated control")
	
	exception_title: str = Field(description="Exception title")
	exception_description: str = Field(description="Exception description")
	exception_type: str = Field(description="Type of exception")
	
	# Business justification
	business_justification: str = Field(description="Business justification")
	business_impact_if_compliant: str = Field(description="Impact of full compliance")
	cost_benefit_analysis: Optional[str] = None
	
	# Risk assessment
	residual_risk_level: RiskLevel = RiskLevel.MEDIUM
	risk_mitigation_measures: List[str] = Field(default_factory=list)
	compensating_controls: List[str] = Field(default_factory=list)
	
	# Exception timeline
	exception_start_date: datetime = Field(default_factory=datetime.utcnow)
	exception_end_date: datetime
	review_frequency: str = Field(description="Review frequency")
	next_review_date: Optional[datetime] = None
	
	# Approval and governance
	requested_by: str = Field(description="Exception requester")
	approved_by: str = Field(description="Exception approver")
	approval_date: datetime = Field(default_factory=datetime.utcnow)
	approval_level: str = Field(description="Required approval level")
	
	# Monitoring and oversight
	monitoring_requirements: List[str] = Field(default_factory=list)
	reporting_requirements: List[str] = Field(default_factory=list)
	oversight_committee: Optional[str] = None
	
	# Exception status
	exception_status: str = Field(default="active")
	suspension_reason: Optional[str] = None
	revocation_reason: Optional[str] = None
	revoked_by: Optional[str] = None
	revocation_date: Optional[datetime] = None
	
	# Documentation and evidence
	supporting_documentation: List[str] = Field(default_factory=list)
	legal_review: bool = False
	compliance_review: bool = False
	risk_committee_review: bool = False
	
	# Renewal and extension
	renewable: bool = True
	extension_history: List[Dict[str, Any]] = Field(default_factory=list)
	maximum_extensions: Optional[int] = None
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class ComplianceMetrics(BaseModel):
	"""Compliance monitoring metrics and KPIs"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	metric_period_start: datetime
	metric_period_end: datetime
	
	# Framework coverage
	frameworks_monitored: List[str] = Field(default_factory=list)
	total_controls: int = 0
	controls_assessed: int = 0
	assessment_coverage: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Compliance status
	compliant_controls: int = 0
	non_compliant_controls: int = 0
	partially_compliant_controls: int = 0
	overall_compliance_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Framework-specific compliance
	framework_compliance_rates: Dict[str, Decimal] = Field(default_factory=dict)
	soc2_compliance_rate: Optional[Decimal] = None
	iso27001_compliance_rate: Optional[Decimal] = None
	pci_dss_compliance_rate: Optional[Decimal] = None
	
	# Assessment metrics
	assessments_completed: int = 0
	assessments_in_progress: int = 0
	overdue_assessments: int = 0
	average_assessment_duration: Optional[timedelta] = None
	
	# Findings and remediation
	total_findings: int = 0
	critical_findings: int = 0
	high_findings: int = 0
	medium_findings: int = 0
	low_findings: int = 0
	
	findings_remediated: int = 0
	findings_overdue: int = 0
	mean_time_to_remediation: Optional[timedelta] = None
	remediation_success_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Exception management
	active_exceptions: int = 0
	expired_exceptions: int = 0
	exceptions_requiring_review: int = 0
	exception_approval_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Risk metrics
	total_compliance_risk_score: Decimal = Field(default=Decimal('0.0'))
	average_control_risk_score: Decimal = Field(default=Decimal('0.0'))
	high_risk_controls: int = 0
	
	# Trend analysis
	compliance_trend: str = Field(default="stable")  # improving, degrading, stable
	assessment_quality_trend: str = Field(default="stable")
	finding_reduction_rate: Decimal = Field(default=Decimal('0.0'))
	
	# Automation and efficiency
	automated_controls: int = 0
	automation_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	manual_effort_hours: Decimal = Field(default=Decimal('0.0'))
	cost_per_control: Optional[Decimal] = None
	
	# Stakeholder engagement
	control_owner_participation: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	business_owner_engagement: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	training_completion_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# External audit readiness
	audit_readiness_score: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	external_audit_findings: int = 0
	management_letter_comments: int = 0
	
	# Regulatory compliance
	regulatory_violations: int = 0
	regulatory_fines: Decimal = Field(default=Decimal('0.0'))
	breach_notifications: int = 0
	
	# Business impact
	compliance_cost_avoidance: Optional[Decimal] = None
	business_process_efficiency: Decimal = Field(default=Decimal('0.0'))
	customer_trust_metrics: Optional[Decimal] = None
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class ComplianceProgram(BaseModel):
	"""Overall compliance program definition and management"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	program_name: str = Field(description="Compliance program name")
	program_description: str = Field(description="Program description")
	program_version: str = Field(description="Program version")
	
	# Scope and framework
	applicable_frameworks: List[ComplianceFramework] = Field(default_factory=list)
	business_units_in_scope: List[str] = Field(default_factory=list)
	systems_in_scope: List[str] = Field(default_factory=list)
	data_types_in_scope: List[str] = Field(default_factory=list)
	
	# Program governance
	program_owner: str = Field(description="Program owner")
	compliance_committee: List[str] = Field(default_factory=list)
	board_oversight: bool = False
	external_oversight: List[str] = Field(default_factory=list)
	
	# Policies and procedures
	policy_framework: Dict[str, str] = Field(default_factory=dict)
	procedures_documented: bool = False
	training_program: Optional[str] = None
	awareness_program: Optional[str] = None
	
	# Risk management
	risk_assessment_frequency: str = Field(description="Risk assessment frequency")
	risk_tolerance: Dict[str, str] = Field(default_factory=dict)
	risk_appetite_statement: Optional[str] = None
	
	# Monitoring and measurement
	monitoring_strategy: str = Field(description="Monitoring strategy")
	kpi_framework: Dict[str, Any] = Field(default_factory=dict)
	reporting_frequency: str = Field(description="Reporting frequency")
	dashboard_url: Optional[str] = None
	
	# Assessment schedule
	assessment_calendar: Dict[str, Any] = Field(default_factory=dict)
	continuous_monitoring: bool = False
	automated_testing: bool = False
	
	# Incident and breach response
	incident_response_plan: Optional[str] = None
	breach_notification_procedures: Optional[str] = None
	regulatory_communication_plan: Optional[str] = None
	
	# Vendor and third-party management
	third_party_risk_program: Optional[str] = None
	vendor_assessment_requirements: List[str] = Field(default_factory=list)
	supply_chain_security: bool = False
	
	# Technology and tools
	grc_platform: Optional[str] = None
	monitoring_tools: List[str] = Field(default_factory=list)
	automation_tools: List[str] = Field(default_factory=list)
	
	# Program effectiveness
	maturity_level: str = Field(default="developing")
	effectiveness_rating: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	last_program_review: Optional[datetime] = None
	next_program_review: Optional[datetime] = None
	
	# Budget and resources
	annual_budget: Optional[Decimal] = None
	fte_allocation: Optional[Decimal] = None
	external_consulting_budget: Optional[Decimal] = None
	
	# Communication and reporting
	stakeholder_communication_plan: Dict[str, Any] = Field(default_factory=dict)
	executive_reporting_schedule: Optional[str] = None
	board_reporting_schedule: Optional[str] = None
	
	is_active: bool = True
	created_by: str = Field(description="Program creator")
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None