"""
Enterprise Compliance and Governance Module

Provides comprehensive compliance and governance capabilities:
- Multi-framework compliance (SOX, HIPAA, GDPR, PCI DSS, SOC2, ISO27001)
- Policy management and enforcement
- Risk assessment and mitigation
- Regulatory reporting
- Data governance and classification
- Audit trail management
- Compliance monitoring and alerting

© 2025 Datacraft
Author: Nyimbi Odero
"""

import asyncio
import json
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import re
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, validator
from uuid_extensions import uuid7str

from .enterprise_integration import ComplianceFramework, AuditEvent, enterprise_integration
from .database import get_async_db_session


class RiskLevel(str, Enum):
	"""Risk levels"""
	VERY_LOW = "very_low"
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


class DataClassification(str, Enum):
	"""Data classification levels"""
	PUBLIC = "public"
	INTERNAL = "internal"
	CONFIDENTIAL = "confidential"
	RESTRICTED = "restricted"
	SECRET = "secret"
	TOP_SECRET = "top_secret"


class PolicyStatus(str, Enum):
	"""Policy status"""
	DRAFT = "draft"
	REVIEW = "review"
	APPROVED = "approved"
	ACTIVE = "active"
	DEPRECATED = "deprecated"
	ARCHIVED = "archived"


class ControlType(str, Enum):
	"""Control types"""
	PREVENTIVE = "preventive"
	DETECTIVE = "detective"
	CORRECTIVE = "corrective"
	COMPENSATING = "compensating"


class MonitoringFrequency(str, Enum):
	"""Monitoring frequency"""
	REAL_TIME = "real_time"
	HOURLY = "hourly"
	DAILY = "daily"
	WEEKLY = "weekly"
	MONTHLY = "monthly"
	QUARTERLY = "quarterly"
	ANNUALLY = "annually"


@dataclass
class ComplianceRequirement:
	"""Compliance requirement definition"""
	id: str
	framework: ComplianceFramework
	requirement_id: str
	title: str
	description: str
	control_type: ControlType
	risk_level: RiskLevel
	implementation_guidance: str
	evidence_requirements: List[str]
	testing_procedures: List[str]
	frequency: MonitoringFrequency
	responsible_party: str
	related_requirements: List[str] = field(default_factory=list)


class CompliancePolicy(BaseModel):
	"""Compliance policy model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	description: str
	framework: ComplianceFramework
	version: str = "1.0"
	status: PolicyStatus = PolicyStatus.DRAFT
	effective_date: datetime
	review_date: datetime
	expiry_date: Optional[datetime] = None
	owner: str
	approver: Optional[str] = None
	scope: List[str] = Field(default_factory=list)  # Business units, systems, etc.
	requirements: List[str] = Field(default_factory=list)  # Requirement IDs
	controls: List[Dict[str, Any]] = Field(default_factory=list)
	exceptions: List[Dict[str, Any]] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	tenant_id: Optional[str] = None


class RiskAssessment(BaseModel):
	"""Risk assessment model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	description: str
	asset_type: str  # system, process, data, etc.
	asset_id: str
	risk_category: str
	threat_sources: List[str] = Field(default_factory=list)
	vulnerabilities: List[str] = Field(default_factory=list)
	likelihood: RiskLevel
	impact: RiskLevel
	inherent_risk: RiskLevel
	residual_risk: RiskLevel
	risk_tolerance: RiskLevel
	mitigation_controls: List[str] = Field(default_factory=list)
	treatment_plan: Optional[str] = None
	owner: str
	assessor: str
	assessment_date: datetime = Field(default_factory=datetime.utcnow)
	next_review_date: datetime
	status: str = "open"  # open, in_progress, mitigated, accepted
	tenant_id: Optional[str] = None


class DataInventoryItem(BaseModel):
	"""Data inventory item"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	description: str
	data_type: str  # PII, PHI, financial, etc.
	classification: DataClassification
	location: str  # system, database, file, etc.
	owner: str
	steward: str
	retention_period: Optional[int] = None  # days
	purpose: str
	legal_basis: Optional[str] = None  # for GDPR
	processing_activities: List[str] = Field(default_factory=list)
	sharing_agreements: List[Dict[str, Any]] = Field(default_factory=list)
	protection_measures: List[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	tenant_id: Optional[str] = None


class ComplianceControl(BaseModel):
	"""Compliance control model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	control_id: str
	name: str
	description: str
	framework: ComplianceFramework
	control_type: ControlType
	category: str
	objective: str
	implementation_status: str = "not_implemented"  # not_implemented, partially_implemented, implemented
	effectiveness: Optional[str] = None  # ineffective, partially_effective, effective
	test_frequency: MonitoringFrequency
	last_test_date: Optional[datetime] = None
	next_test_date: Optional[datetime] = None
	test_results: List[Dict[str, Any]] = Field(default_factory=list)
	responsible_party: str
	evidence_location: Optional[str] = None
	automation_level: str = "manual"  # manual, semi_automated, automated
	cost: Optional[float] = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	tenant_id: Optional[str] = None


class ComplianceException(BaseModel):
	"""Compliance exception model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	policy_id: str
	requirement_id: str
	business_justification: str
	risk_assessment: str
	compensating_controls: List[str] = Field(default_factory=list)
	approval_level: str
	approver: str
	approved_date: datetime
	expiry_date: datetime
	status: str = "active"  # active, expired, revoked
	review_date: datetime
	monitoring_plan: Optional[str] = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	tenant_id: Optional[str] = None


class ComplianceIncident(BaseModel):
	"""Compliance incident model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	title: str
	description: str
	incident_type: str  # violation, breach, non_compliance
	severity: RiskLevel
	framework: ComplianceFramework
	affected_systems: List[str] = Field(default_factory=list)
	affected_data: List[str] = Field(default_factory=list)
	root_cause: Optional[str] = None
	discovery_date: datetime = Field(default_factory=datetime.utcnow)
	reported_date: Optional[datetime] = None
	resolution_date: Optional[datetime] = None
	status: str = "open"  # open, investigating, resolved, closed
	assignee: Optional[str] = None
	remediation_actions: List[Dict[str, Any]] = Field(default_factory=list)
	lessons_learned: Optional[str] = None
	regulatory_notification: bool = False
	customer_notification: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	tenant_id: Optional[str] = None


class ComplianceFrameworkManager:
	"""Compliance framework management"""
	
	def __init__(self):
		self.requirements_cache = {}
		self._load_framework_requirements()
	
	def _load_framework_requirements(self):
		"""Load compliance framework requirements"""
		# SOX (Sarbanes-Oxley) requirements
		self.requirements_cache[ComplianceFramework.SOX] = [
			ComplianceRequirement(
				id="sox_302",
				framework=ComplianceFramework.SOX,
				requirement_id="Section 302",
				title="Corporate Responsibility for Financial Reports",
				description="Principal executive and financial officers must certify financial reports",
				control_type=ControlType.PREVENTIVE,
				risk_level=RiskLevel.HIGH,
				implementation_guidance="Implement certification process for financial reports",
				evidence_requirements=["Signed certifications", "Review documentation"],
				testing_procedures=["Review certification process", "Test completeness"],
				frequency=MonitoringFrequency.QUARTERLY,
				responsible_party="CFO"
			),
			ComplianceRequirement(
				id="sox_404",
				framework=ComplianceFramework.SOX,
				requirement_id="Section 404",
				title="Management Assessment of Internal Controls",
				description="Management must assess effectiveness of internal controls over financial reporting",
				control_type=ControlType.DETECTIVE,
				risk_level=RiskLevel.HIGH,
				implementation_guidance="Document and test internal controls over financial reporting",
				evidence_requirements=["Control documentation", "Test results", "Management assessment"],
				testing_procedures=["Control testing", "Walkthrough procedures"],
				frequency=MonitoringFrequency.ANNUALLY,
				responsible_party="Internal Audit"
			)
		]
		
		# HIPAA requirements
		self.requirements_cache[ComplianceFramework.HIPAA] = [
			ComplianceRequirement(
				id="hipaa_164_308",
				framework=ComplianceFramework.HIPAA,
				requirement_id="164.308",
				title="Administrative Safeguards",
				description="Administrative actions and policies to manage security",
				control_type=ControlType.PREVENTIVE,
				risk_level=RiskLevel.HIGH,
				implementation_guidance="Implement administrative safeguards for PHI protection",
				evidence_requirements=["Policies", "Training records", "Access logs"],
				testing_procedures=["Policy review", "Access testing"],
				frequency=MonitoringFrequency.ANNUALLY,
				responsible_party="Security Officer"
			),
			ComplianceRequirement(
				id="hipaa_164_312",
				framework=ComplianceFramework.HIPAA,
				requirement_id="164.312",
				title="Technical Safeguards",
				description="Technology controls to protect PHI",
				control_type=ControlType.PREVENTIVE,
				risk_level=RiskLevel.HIGH,
				implementation_guidance="Implement technical controls for PHI access",
				evidence_requirements=["Access controls", "Encryption", "Audit logs"],
				testing_procedures=["Technical testing", "Penetration testing"],
				frequency=MonitoringFrequency.QUARTERLY,
				responsible_party="IT Security"
			)
		]
		
		# GDPR requirements
		self.requirements_cache[ComplianceFramework.GDPR] = [
			ComplianceRequirement(
				id="gdpr_art_5",
				framework=ComplianceFramework.GDPR,
				requirement_id="Article 5",
				title="Principles of Processing",
				description="Personal data processing principles",
				control_type=ControlType.PREVENTIVE,
				risk_level=RiskLevel.HIGH,
				implementation_guidance="Implement data processing principles",
				evidence_requirements=["Data processing records", "Consent records"],
				testing_procedures=["Data processing review", "Consent validation"],
				frequency=MonitoringFrequency.QUARTERLY,
				responsible_party="Data Protection Officer"
			),
			ComplianceRequirement(
				id="gdpr_art_32",
				framework=ComplianceFramework.GDPR,
				requirement_id="Article 32",
				title="Security of Processing",
				description="Appropriate technical and organizational measures",
				control_type=ControlType.PREVENTIVE,
				risk_level=RiskLevel.HIGH,
				implementation_guidance="Implement security measures for personal data",
				evidence_requirements=["Security controls", "Risk assessments"],
				testing_procedures=["Security testing", "Vulnerability assessment"],
				frequency=MonitoringFrequency.QUARTERLY,
				responsible_party="CISO"
			)
		]
		
		# PCI DSS requirements
		self.requirements_cache[ComplianceFramework.PCI_DSS] = [
			ComplianceRequirement(
				id="pci_req_1",
				framework=ComplianceFramework.PCI_DSS,
				requirement_id="Requirement 1",
				title="Install and maintain a firewall",
				description="Firewall configuration to protect cardholder data",
				control_type=ControlType.PREVENTIVE,
				risk_level=RiskLevel.HIGH,
				implementation_guidance="Configure and maintain firewall rules",
				evidence_requirements=["Firewall configurations", "Change logs"],
				testing_procedures=["Firewall rule review", "Penetration testing"],
				frequency=MonitoringFrequency.QUARTERLY,
				responsible_party="Network Security"
			),
			ComplianceRequirement(
				id="pci_req_3",
				framework=ComplianceFramework.PCI_DSS,
				requirement_id="Requirement 3",
				title="Protect stored cardholder data",
				description="Encryption and protection of stored cardholder data",
				control_type=ControlType.PREVENTIVE,
				risk_level=RiskLevel.CRITICAL,
				implementation_guidance="Encrypt cardholder data storage",
				evidence_requirements=["Encryption standards", "Key management"],
				testing_procedures=["Encryption testing", "Key management review"],
				frequency=MonitoringFrequency.QUARTERLY,
				responsible_party="Security Team"
			)
		]
	
	def get_framework_requirements(self, framework: ComplianceFramework) -> List[ComplianceRequirement]:
		"""Get requirements for compliance framework"""
		return self.requirements_cache.get(framework, [])
	
	def get_requirement_by_id(self, framework: ComplianceFramework, requirement_id: str) -> Optional[ComplianceRequirement]:
		"""Get specific requirement by ID"""
		requirements = self.get_framework_requirements(framework)
		return next((req for req in requirements if req.requirement_id == requirement_id), None)


class PolicyManager:
	"""Compliance policy management"""
	
	async def create_policy(self, policy: CompliancePolicy) -> str:
		"""Create compliance policy"""
		try:
			async with get_async_db_session() as session:
				from sqlalchemy import text
				
				await session.execute(
					text("""
					INSERT INTO wo_compliance_policies (
						id, name, description, framework, version, status,
						effective_date, review_date, expiry_date, owner,
						approver, scope, requirements, controls, exceptions,
						created_at, updated_at, tenant_id
					) VALUES (
						:id, :name, :description, :framework, :version, :status,
						:effective_date, :review_date, :expiry_date, :owner,
						:approver, :scope, :requirements, :controls, :exceptions,
						:created_at, :updated_at, :tenant_id
					)
					"""),
					{
						"id": policy.id,
						"name": policy.name,
						"description": policy.description,
						"framework": policy.framework.value,
						"version": policy.version,
						"status": policy.status.value,
						"effective_date": policy.effective_date,
						"review_date": policy.review_date,
						"expiry_date": policy.expiry_date,
						"owner": policy.owner,
						"approver": policy.approver,
						"scope": json.dumps(policy.scope),
						"requirements": json.dumps(policy.requirements),
						"controls": json.dumps(policy.controls),
						"exceptions": json.dumps(policy.exceptions),
						"created_at": policy.created_at,
						"updated_at": policy.updated_at,
						"tenant_id": policy.tenant_id
					}
				)
				await session.commit()
			
			# Log policy creation
			audit_event = AuditEvent(
				event_type="policy_management",
				action="policy_created",
				result="success",
				resource_type="compliance_policy",
				resource_id=policy.id,
				details={
					"policy_name": policy.name,
					"framework": policy.framework.value,
					"status": policy.status.value
				},
				risk_level="medium",
				compliance_tags=[policy.framework.value],
				tenant_id=policy.tenant_id
			)
			await enterprise_integration.log_audit_event(audit_event)
			
			return policy.id
			
		except Exception as e:
			print(f"Policy creation error: {e}")
			raise
	
	async def update_policy(self, policy_id: str, updates: Dict[str, Any]) -> bool:
		"""Update compliance policy"""
		try:
			updates["updated_at"] = datetime.utcnow()
			
			# Build dynamic update query
			set_clauses = []
			params = {"policy_id": policy_id}
			
			for field, value in updates.items():
				if field in ["scope", "requirements", "controls", "exceptions"]:
					value = json.dumps(value)
				set_clauses.append(f"{field} = :{field}")
				params[field] = value
			
			async with get_async_db_session() as session:
				from sqlalchemy import text
				
				query = f"""
				UPDATE wo_compliance_policies 
				SET {', '.join(set_clauses)}
				WHERE id = :policy_id
				"""
				
				result = await session.execute(text(query), params)
				await session.commit()
				
				if result.rowcount > 0:
					# Log policy update
					audit_event = AuditEvent(
						event_type="policy_management",
						action="policy_updated",
						result="success",
						resource_type="compliance_policy",
						resource_id=policy_id,
						details={"updated_fields": list(updates.keys())},
						risk_level="medium"
					)
					await enterprise_integration.log_audit_event(audit_event)
					return True
			
			return False
			
		except Exception as e:
			print(f"Policy update error: {e}")
			raise
	
	async def get_policies_for_review(self) -> List[Dict[str, Any]]:
		"""Get policies that need review"""
		try:
			async with get_async_db_session() as session:
				from sqlalchemy import text
				
				result = await session.execute(
					text("""
					SELECT id, name, framework, owner, review_date
					FROM wo_compliance_policies 
					WHERE review_date <= :now 
					AND status IN ('active', 'approved')
					ORDER BY review_date ASC
					"""),
					{"now": datetime.utcnow()}
				)
				
				return [
					{
						"id": row[0],
						"name": row[1],
						"framework": row[2],
						"owner": row[3],
						"review_date": row[4]
					}
					for row in result.fetchall()
				]
				
		except Exception as e:
			print(f"Policy review query error: {e}")
			return []


class RiskManager:
	"""Risk assessment and management"""
	
	async def create_risk_assessment(self, assessment: RiskAssessment) -> str:
		"""Create risk assessment"""
		try:
			async with get_async_db_session() as session:
				from sqlalchemy import text
				
				await session.execute(
					text("""
					INSERT INTO wo_risk_assessments (
						id, name, description, asset_type, asset_id, risk_category,
						threat_sources, vulnerabilities, likelihood, impact,
						inherent_risk, residual_risk, risk_tolerance,
						mitigation_controls, treatment_plan, owner, assessor,
						assessment_date, next_review_date, status, tenant_id
					) VALUES (
						:id, :name, :description, :asset_type, :asset_id, :risk_category,
						:threat_sources, :vulnerabilities, :likelihood, :impact,
						:inherent_risk, :residual_risk, :risk_tolerance,
						:mitigation_controls, :treatment_plan, :owner, :assessor,
						:assessment_date, :next_review_date, :status, :tenant_id
					)
					"""),
					{
						"id": assessment.id,
						"name": assessment.name,
						"description": assessment.description,
						"asset_type": assessment.asset_type,
						"asset_id": assessment.asset_id,
						"risk_category": assessment.risk_category,
						"threat_sources": json.dumps(assessment.threat_sources),
						"vulnerabilities": json.dumps(assessment.vulnerabilities),
						"likelihood": assessment.likelihood.value,
						"impact": assessment.impact.value,
						"inherent_risk": assessment.inherent_risk.value,
						"residual_risk": assessment.residual_risk.value,
						"risk_tolerance": assessment.risk_tolerance.value,
						"mitigation_controls": json.dumps(assessment.mitigation_controls),
						"treatment_plan": assessment.treatment_plan,
						"owner": assessment.owner,
						"assessor": assessment.assessor,
						"assessment_date": assessment.assessment_date,
						"next_review_date": assessment.next_review_date,
						"status": assessment.status,
						"tenant_id": assessment.tenant_id
					}
				)
				await session.commit()
			
			return assessment.id
			
		except Exception as e:
			print(f"Risk assessment creation error: {e}")
			raise
	
	async def calculate_risk_score(self, likelihood: RiskLevel, impact: RiskLevel) -> Tuple[RiskLevel, int]:
		"""Calculate risk score based on likelihood and impact"""
		risk_values = {
			RiskLevel.VERY_LOW: 1,
			RiskLevel.LOW: 2,
			RiskLevel.MEDIUM: 3,
			RiskLevel.HIGH: 4,
			RiskLevel.CRITICAL: 5
		}
		
		likelihood_score = risk_values[likelihood]
		impact_score = risk_values[impact]
		
		# Risk matrix calculation
		risk_score = likelihood_score * impact_score
		
		if risk_score <= 2:
			return RiskLevel.VERY_LOW, risk_score
		elif risk_score <= 4:
			return RiskLevel.LOW, risk_score
		elif risk_score <= 9:
			return RiskLevel.MEDIUM, risk_score
		elif risk_score <= 16:
			return RiskLevel.HIGH, risk_score
		else:
			return RiskLevel.CRITICAL, risk_score
	
	async def get_risk_dashboard(self) -> Dict[str, Any]:
		"""Get risk management dashboard data"""
		try:
			async with get_async_db_session() as session:
				from sqlalchemy import text
				
				# Risk distribution
				risk_dist_result = await session.execute(
					text("""
					SELECT residual_risk, COUNT(*) as count
					FROM wo_risk_assessments
					WHERE status = 'open'
					GROUP BY residual_risk
					""")
				)
				
				risk_distribution = {row[0]: row[1] for row in risk_dist_result.fetchall()}
				
				# High-risk items
				high_risk_result = await session.execute(
					text("""
					SELECT name, asset_type, residual_risk, owner
					FROM wo_risk_assessments
					WHERE residual_risk IN ('high', 'critical')
					AND status = 'open'
					ORDER BY assessment_date DESC
					LIMIT 10
					""")
				)
				
				high_risk_items = [
					{
						"name": row[0],
						"asset_type": row[1],
						"risk_level": row[2],
						"owner": row[3]
					}
					for row in high_risk_result.fetchall()
				]
				
				# Overdue assessments
				overdue_result = await session.execute(
					text("""
					SELECT name, owner, next_review_date
					FROM wo_risk_assessments
					WHERE next_review_date < :now
					AND status = 'open'
					ORDER BY next_review_date ASC
					"""),
					{"now": datetime.utcnow()}
				)
				
				overdue_assessments = [
					{
						"name": row[0],
						"owner": row[1],
						"due_date": row[2]
					}
					for row in overdue_result.fetchall()
				]
				
				return {
					"risk_distribution": risk_distribution,
					"high_risk_items": high_risk_items,
					"overdue_assessments": overdue_assessments,
					"total_risks": sum(risk_distribution.values())
				}
				
		except Exception as e:
			print(f"Risk dashboard error: {e}")
			return {}


class DataGovernanceManager:
	"""Data governance and classification"""
	
	async def create_data_inventory_item(self, item: DataInventoryItem) -> str:
		"""Create data inventory item"""
		try:
			async with get_async_db_session() as session:
				from sqlalchemy import text
				
				await session.execute(
					text("""
					INSERT INTO wo_data_inventory (
						id, name, description, data_type, classification,
						location, owner, steward, retention_period, purpose,
						legal_basis, processing_activities, sharing_agreements,
						protection_measures, created_at, updated_at, tenant_id
					) VALUES (
						:id, :name, :description, :data_type, :classification,
						:location, :owner, :steward, :retention_period, :purpose,
						:legal_basis, :processing_activities, :sharing_agreements,
						:protection_measures, :created_at, :updated_at, :tenant_id
					)
					"""),
					{
						"id": item.id,
						"name": item.name,
						"description": item.description,
						"data_type": item.data_type,
						"classification": item.classification.value,
						"location": item.location,
						"owner": item.owner,
						"steward": item.steward,
						"retention_period": item.retention_period,
						"purpose": item.purpose,
						"legal_basis": item.legal_basis,
						"processing_activities": json.dumps(item.processing_activities),
						"sharing_agreements": json.dumps(item.sharing_agreements),
						"protection_measures": json.dumps(item.protection_measures),
						"created_at": item.created_at,
						"updated_at": item.updated_at,
						"tenant_id": item.tenant_id
					}
				)
				await session.commit()
			
			return item.id
			
		except Exception as e:
			print(f"Data inventory creation error: {e}")
			raise
	
	async def classify_data_automatically(self, data_sample: str, location: str) -> Tuple[DataClassification, List[str]]:
		"""Automatically classify data based on content analysis"""
		classification = DataClassification.PUBLIC
		indicators = []
		
		# Convert to lowercase for pattern matching
		data_lower = data_sample.lower()
		
		# PII patterns
		pii_patterns = {
			"ssn": r"\b\d{3}-?\d{2}-?\d{4}\b",
			"credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
			"email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
			"phone": r"\b\d{3}[-.]\d{3}[-.]\d{4}\b",
			"driver_license": r"\b[A-Z]\d{7}\b"
		}
		
		# Health information patterns
		health_patterns = {
			"medical_record": r"\bmrn[\s:]*\d+\b",
			"diagnosis_code": r"\bicd[- ]?\d+\b",
			"prescription": r"\brx[\s:]*\w+\b"
		}
		
		# Financial patterns
		financial_patterns = {
			"account_number": r"\bacc[\s:]*\d{8,}\b",
			"routing_number": r"\b\d{9}\b",
			"swift_code": r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}[A-Z0-9]{3}\b"
		}
		
		# Check for PII
		for indicator, pattern in pii_patterns.items():
			if re.search(pattern, data_sample, re.IGNORECASE):
				indicators.append(f"PII: {indicator}")
				if classification == DataClassification.PUBLIC:
					classification = DataClassification.CONFIDENTIAL
		
		# Check for health information
		for indicator, pattern in health_patterns.items():
			if re.search(pattern, data_sample, re.IGNORECASE):
				indicators.append(f"PHI: {indicator}")
				classification = DataClassification.RESTRICTED
		
		# Check for financial information
		for indicator, pattern in financial_patterns.items():
			if re.search(pattern, data_sample, re.IGNORECASE):
				indicators.append(f"Financial: {indicator}")
				if classification in [DataClassification.PUBLIC, DataClassification.INTERNAL]:
					classification = DataClassification.CONFIDENTIAL
		
		# Check location-based rules
		if "production" in location.lower():
			if classification == DataClassification.PUBLIC:
				classification = DataClassification.INTERNAL
		
		if any(keyword in location.lower() for keyword in ["secure", "vault", "encrypted"]):
			if classification in [DataClassification.PUBLIC, DataClassification.INTERNAL]:
				classification = DataClassification.CONFIDENTIAL
		
		return classification, indicators
	
	async def get_data_governance_metrics(self) -> Dict[str, Any]:
		"""Get data governance metrics"""
		try:
			async with get_async_db_session() as session:
				from sqlalchemy import text
				
				# Classification distribution
				class_dist_result = await session.execute(
					text("""
					SELECT classification, COUNT(*) as count
					FROM wo_data_inventory
					GROUP BY classification
					""")
				)
				
				classification_distribution = {row[0]: row[1] for row in class_dist_result.fetchall()}
				
				# Data types
				type_dist_result = await session.execute(
					text("""
					SELECT data_type, COUNT(*) as count
					FROM wo_data_inventory
					GROUP BY data_type
					""")
				)
				
				data_type_distribution = {row[0]: row[1] for row in type_dist_result.fetchall()}
				
				# Retention compliance
				retention_result = await session.execute(
					text("""
					SELECT 
						COUNT(*) as total,
						COUNT(CASE WHEN retention_period IS NOT NULL THEN 1 END) as with_retention
					FROM wo_data_inventory
					""")
				)
				
				retention_row = retention_result.fetchone()
				retention_compliance = {
					"total": retention_row[0],
					"with_retention_policy": retention_row[1],
					"compliance_rate": (retention_row[1] / retention_row[0] * 100) if retention_row[0] > 0 else 0
				}
				
				return {
					"classification_distribution": classification_distribution,
					"data_type_distribution": data_type_distribution,
					"retention_compliance": retention_compliance,
					"total_data_assets": sum(classification_distribution.values())
				}
				
		except Exception as e:
			print(f"Data governance metrics error: {e}")
			return {}


class ComplianceMonitor:
	"""Compliance monitoring and alerting"""
	
	def __init__(self):
		self.monitoring_rules = {}
		self._setup_monitoring_rules()
	
	def _setup_monitoring_rules(self):
		"""Setup compliance monitoring rules"""
		self.monitoring_rules = {
			"policy_review_overdue": {
				"description": "Policies overdue for review",
				"query": """
				SELECT COUNT(*) FROM wo_compliance_policies 
				WHERE review_date < :now AND status = 'active'
				""",
				"threshold": 0,
				"severity": "medium",
				"frequency": MonitoringFrequency.DAILY
			},
			"high_risk_items": {
				"description": "High/Critical risk items without mitigation",
				"query": """
				SELECT COUNT(*) FROM wo_risk_assessments 
				WHERE residual_risk IN ('high', 'critical') 
				AND status = 'open' 
				AND treatment_plan IS NULL
				""",
				"threshold": 0,
				"severity": "high",
				"frequency": MonitoringFrequency.DAILY
			},
			"failed_control_tests": {
				"description": "Controls with failed tests",
				"query": """
				SELECT COUNT(*) FROM wo_compliance_controls 
				WHERE effectiveness = 'ineffective'
				AND test_results->-1->>'result' = 'failed'
				""",
				"threshold": 0,
				"severity": "high",
				"frequency": MonitoringFrequency.DAILY
			},
			"data_retention_violations": {
				"description": "Data assets exceeding retention period",
				"query": """
				SELECT COUNT(*) FROM wo_data_inventory 
				WHERE retention_period IS NOT NULL 
				AND created_at < NOW() - INTERVAL '1 day' * retention_period
				""",
				"threshold": 0,
				"severity": "medium",
				"frequency": MonitoringFrequency.WEEKLY
			}
		}
	
	async def run_compliance_checks(self) -> List[Dict[str, Any]]:
		"""Run compliance monitoring checks"""
		alerts = []
		
		try:
			async with get_async_db_session() as session:
				from sqlalchemy import text
				
				for rule_name, rule_config in self.monitoring_rules.items():
					try:
						result = await session.execute(
							text(rule_config["query"]),
							{"now": datetime.utcnow()}
						)
						
						count = result.scalar()
						
						if count > rule_config["threshold"]:
							alert = {
								"rule_name": rule_name,
								"description": rule_config["description"],
								"count": count,
								"threshold": rule_config["threshold"],
								"severity": rule_config["severity"],
								"timestamp": datetime.utcnow()
							}
							alerts.append(alert)
							
							# Log compliance alert
							audit_event = AuditEvent(
								event_type="compliance_monitoring",
								action="compliance_alert",
								result="violation_detected",
								details=alert,
								risk_level=rule_config["severity"]
							)
							await enterprise_integration.log_audit_event(audit_event)
							
					except Exception as e:
						print(f"Monitoring rule {rule_name} error: {e}")
				
				return alerts
				
		except Exception as e:
			print(f"Compliance monitoring error: {e}")
			return []
	
	async def generate_compliance_dashboard(self) -> Dict[str, Any]:
		"""Generate compliance dashboard data"""
		try:
			# Run compliance checks
			alerts = await self.run_compliance_checks()
			
			# Get compliance metrics
			async with get_async_db_session() as session:
				from sqlalchemy import text
				
				# Policy status distribution
				policy_status_result = await session.execute(
					text("""
					SELECT status, COUNT(*) as count
					FROM wo_compliance_policies
					GROUP BY status
					""")
				)
				
				policy_status = {row[0]: row[1] for row in policy_status_result.fetchall()}
				
				# Control effectiveness
				control_effectiveness_result = await session.execute(
					text("""
					SELECT effectiveness, COUNT(*) as count
					FROM wo_compliance_controls
					WHERE effectiveness IS NOT NULL
					GROUP BY effectiveness
					""")
				)
				
				control_effectiveness = {row[0]: row[1] for row in control_effectiveness_result.fetchall()}
				
				# Recent incidents
				incidents_result = await session.execute(
					text("""
					SELECT title, severity, status, discovery_date
					FROM wo_compliance_incidents
					WHERE discovery_date >= :since
					ORDER BY discovery_date DESC
					LIMIT 10
					"""),
					{"since": datetime.utcnow() - timedelta(days=30)}
				)
				
				recent_incidents = [
					{
						"title": row[0],
						"severity": row[1],
						"status": row[2],
						"discovery_date": row[3]
					}
					for row in incidents_result.fetchall()
				]
				
				return {
					"alerts": alerts,
					"alert_count": len(alerts),
					"policy_status": policy_status,
					"control_effectiveness": control_effectiveness,
					"recent_incidents": recent_incidents,
					"last_updated": datetime.utcnow()
				}
				
		except Exception as e:
			print(f"Compliance dashboard error: {e}")
			return {}


class ComplianceGovernanceManager:
	"""Main compliance and governance management class"""
	
	def __init__(self):
		self.framework_manager = ComplianceFrameworkManager()
		self.policy_manager = PolicyManager()
		self.risk_manager = RiskManager()
		self.data_governance = DataGovernanceManager()
		self.monitor = ComplianceMonitor()
	
	async def create_policy(self, policy: CompliancePolicy) -> str:
		"""Create compliance policy"""
		return await self.policy_manager.create_policy(policy)
	
	async def create_risk_assessment(self, assessment: RiskAssessment) -> str:
		"""Create risk assessment"""
		return await self.risk_manager.create_risk_assessment(assessment)
	
	async def create_data_inventory_item(self, item: DataInventoryItem) -> str:
		"""Create data inventory item"""
		return await self.data_governance.create_data_inventory_item(item)
	
	async def run_compliance_monitoring(self) -> List[Dict[str, Any]]:
		"""Run compliance monitoring"""
		return await self.monitor.run_compliance_checks()
	
	async def get_compliance_dashboard(self) -> Dict[str, Any]:
		"""Get comprehensive compliance dashboard"""
		dashboard_data = await self.monitor.generate_compliance_dashboard()
		risk_data = await self.risk_manager.get_risk_dashboard()
		data_governance_data = await self.data_governance.get_data_governance_metrics()
		
		dashboard_data.update({
			"risk_management": risk_data,
			"data_governance": data_governance_data
		})
		
		return dashboard_data
	
	async def generate_framework_report(self, framework: ComplianceFramework, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
		"""Generate framework-specific compliance report"""
		# Get framework requirements
		requirements = self.framework_manager.get_framework_requirements(framework)
		
		# Get compliance data from enterprise integration
		compliance_report = await enterprise_integration.generate_compliance_report(
			framework, start_date, end_date
		)
		
		# Add framework-specific analysis
		compliance_report.update({
			"framework_requirements": [
				{
					"id": req.id,
					"requirement_id": req.requirement_id,
					"title": req.title,
					"control_type": req.control_type.value,
					"risk_level": req.risk_level.value
				}
				for req in requirements
			],
			"requirements_count": len(requirements)
		})
		
		return compliance_report


# Global compliance governance manager instance
compliance_governance = ComplianceGovernanceManager()