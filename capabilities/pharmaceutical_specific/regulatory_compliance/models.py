"""
Regulatory Compliance Models

Database models for pharmaceutical regulatory compliance including frameworks,
submissions, audits, deviations, corrective actions, and compliance monitoring.
"""

from datetime import datetime, date
from typing import Dict, List, Any
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint, JSON
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ....auth_rbac.models import BaseMixin, AuditMixin, Model


class PHRCRegulatoryFramework(Model, AuditMixin, BaseMixin):
	"""
	Regulatory frameworks (FDA, EMA, GMP, etc.) that the organization must comply with.
	"""
	__tablename__ = 'ph_rc_regulatory_framework'
	
	# Identity
	framework_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Framework Information
	framework_code = Column(String(20), nullable=False, index=True)  # FDA, EMA, GMP, etc.
	framework_name = Column(String(200), nullable=False)
	region = Column(String(100), nullable=True)  # US, EU, Global, etc.
	description = Column(Text, nullable=True)
	
	# Configuration
	website_url = Column(String(500), nullable=True)
	contact_info = Column(JSON, nullable=True)  # Contact details
	key_regulations = Column(JSON, nullable=True)  # List of key regulations
	submission_types = Column(JSON, nullable=True)  # Supported submission types
	
	# Status
	is_active = Column(Boolean, default=True)
	version = Column(String(50), nullable=True)
	effective_date = Column(Date, nullable=True)
	
	# Relationships
	submissions = relationship("PHRCSubmission", back_populates="framework")
	audits = relationship("PHRCAudit", back_populates="framework") 
	controls = relationship("PHRCComplianceControl", back_populates="framework")
	
	def __repr__(self):
		return f"<PHRCRegulatoryFramework {self.framework_name}>"


class PHRCSubmission(Model, AuditMixin, BaseMixin):
	"""
	Regulatory submissions (IND, NDA, BLA, etc.) to regulatory authorities.
	"""
	__tablename__ = 'ph_rc_submission'
	
	# Identity
	submission_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Submission Information
	submission_number = Column(String(100), nullable=False, unique=True, index=True)
	submission_type = Column(String(50), nullable=False)  # IND, NDA, BLA, etc.
	submission_title = Column(String(500), nullable=False)
	description = Column(Text, nullable=True)
	
	# Product Information
	product_name = Column(String(200), nullable=True)
	active_ingredient = Column(String(200), nullable=True)
	therapeutic_area = Column(String(100), nullable=True)
	indication = Column(Text, nullable=True)
	
	# Regulatory Framework
	framework_id = Column(String(36), ForeignKey('ph_rc_regulatory_framework.framework_id'), nullable=False)
	
	# Submission Status
	status = Column(String(50), nullable=False, default='Draft')  # Draft, Submitted, Under Review, Approved, Rejected
	submission_date = Column(Date, nullable=True)
	target_response_date = Column(Date, nullable=True)
	actual_response_date = Column(Date, nullable=True)
	
	# Review Information
	review_division = Column(String(200), nullable=True)
	reviewer_name = Column(String(200), nullable=True)
	reviewer_contact = Column(String(500), nullable=True)
	
	# Compliance
	regulatory_contact_id = Column(String(36), ForeignKey('ph_rc_regulatory_contact.contact_id'), nullable=True)
	priority_designation = Column(String(50), nullable=True)  # Standard, Priority, Fast Track, etc.
	
	# Metadata
	submission_metadata = Column(JSON, nullable=True)
	fees_paid = Column(DECIMAL(15, 2), nullable=True)
	currency = Column(String(3), default='USD')
	
	# Relationships
	framework = relationship("PHRCRegulatoryFramework", back_populates="submissions")
	documents = relationship("PHRCSubmissionDocument", back_populates="submission")
	regulatory_contact = relationship("PHRCRegulatoryContact", back_populates="submissions")
	
	def __repr__(self):
		return f"<PHRCSubmission {self.submission_number}>"


class PHRCSubmissionDocument(Model, AuditMixin, BaseMixin):
	"""
	Documents associated with regulatory submissions.
	"""
	__tablename__ = 'ph_rc_submission_document'
	
	# Identity
	document_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Document Information
	submission_id = Column(String(36), ForeignKey('ph_rc_submission.submission_id'), nullable=False)
	document_name = Column(String(500), nullable=False)
	document_type = Column(String(100), nullable=False)  # Protocol, CSR, CMC, etc.
	document_version = Column(String(50), nullable=False)
	
	# File Information
	file_path = Column(String(1000), nullable=True)
	file_size = Column(Integer, nullable=True)
	file_hash = Column(String(128), nullable=True)
	mime_type = Column(String(200), nullable=True)
	
	# Status
	status = Column(String(50), nullable=False, default='Draft')
	is_final = Column(Boolean, default=False)
	approval_required = Column(Boolean, default=True)
	approved_by = Column(String(36), nullable=True)  # User ID
	approved_date = Column(DateTime, nullable=True)
	
	# Compliance
	electronic_signature = Column(JSON, nullable=True)
	document_metadata = Column(JSON, nullable=True)
	
	# Relationships
	submission = relationship("PHRCSubmission", back_populates="documents")
	
	def __repr__(self):
		return f"<PHRCSubmissionDocument {self.document_name}>"


class PHRCAudit(Model, AuditMixin, BaseMixin):
	"""
	Regulatory audits and inspections.
	"""
	__tablename__ = 'ph_rc_audit'
	
	# Identity
	audit_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Audit Information
	audit_number = Column(String(100), nullable=False, unique=True, index=True)
	audit_title = Column(String(500), nullable=False)
	audit_type = Column(String(50), nullable=False)  # Internal, External, Self-Assessment
	audit_scope = Column(Text, nullable=True)
	
	# Regulatory Framework
	framework_id = Column(String(36), ForeignKey('ph_rc_regulatory_framework.framework_id'), nullable=False)
	
	# Schedule
	planned_start_date = Column(Date, nullable=True)
	planned_end_date = Column(Date, nullable=True)
	actual_start_date = Column(Date, nullable=True)
	actual_end_date = Column(Date, nullable=True)
	
	# Audit Team
	lead_auditor = Column(String(200), nullable=True)
	audit_team = Column(JSON, nullable=True)  # List of auditor details
	auditee_contact = Column(String(200), nullable=True)
	
	# Status
	status = Column(String(50), nullable=False, default='Planned')  # Planned, In Progress, Completed, Cancelled
	overall_rating = Column(String(50), nullable=True)  # Satisfactory, Needs Improvement, Unsatisfactory
	
	# Results
	executive_summary = Column(Text, nullable=True)
	audit_report_path = Column(String(1000), nullable=True)
	
	# Relationships
	framework = relationship("PHRCRegulatoryFramework", back_populates="audits")
	findings = relationship("PHRCAuditFinding", back_populates="audit")
	
	def __repr__(self):
		return f"<PHRCAudit {self.audit_number}>"


class PHRCAuditFinding(Model, AuditMixin, BaseMixin):
	"""
	Individual findings from regulatory audits.
	"""
	__tablename__ = 'ph_rc_audit_finding'
	
	# Identity
	finding_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Finding Information
	audit_id = Column(String(36), ForeignKey('ph_rc_audit.audit_id'), nullable=False)
	finding_number = Column(String(100), nullable=False, index=True)
	finding_title = Column(String(500), nullable=False)
	description = Column(Text, nullable=False)
	
	# Classification
	severity = Column(String(50), nullable=False)  # Critical, Major, Minor, Observation
	category = Column(String(100), nullable=True)  # Quality System, Documentation, etc.
	regulation_reference = Column(String(200), nullable=True)
	
	# Response
	response_required = Column(Boolean, default=True)
	response_deadline = Column(Date, nullable=True)
	assigned_to = Column(String(36), nullable=True)  # User ID
	
	# Status
	status = Column(String(50), nullable=False, default='Open')  # Open, In Progress, Closed, Verified
	closure_date = Column(Date, nullable=True)
	
	# Evidence
	evidence_documents = Column(JSON, nullable=True)
	
	# Relationships
	audit = relationship("PHRCAudit", back_populates="findings")
	corrective_actions = relationship("PHRCCorrectiveAction", back_populates="finding")
	
	def __repr__(self):
		return f"<PHRCAuditFinding {self.finding_number}>"


class PHRCDeviation(Model, AuditMixin, BaseMixin):
	"""
	Quality deviations and non-conformances.
	"""
	__tablename__ = 'ph_rc_deviation'
	
	# Identity
	deviation_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Deviation Information
	deviation_number = Column(String(100), nullable=False, unique=True, index=True)
	deviation_title = Column(String(500), nullable=False)
	description = Column(Text, nullable=False)
	
	# Classification
	deviation_type = Column(String(100), nullable=False)  # Process, Product, System, etc.
	severity = Column(String(50), nullable=False)  # Critical, Major, Minor
	impact_assessment = Column(Text, nullable=True)
	
	# Context
	process_area = Column(String(200), nullable=True)
	product_affected = Column(String(200), nullable=True)
	batch_lot_affected = Column(String(200), nullable=True)
	
	# Discovery
	discovered_date = Column(Date, nullable=False)
	discovered_by = Column(String(36), nullable=False)  # User ID
	discovery_method = Column(String(200), nullable=True)
	
	# Investigation
	investigation_required = Column(Boolean, default=True)
	investigation_deadline = Column(Date, nullable=True)
	assigned_investigator = Column(String(36), nullable=True)  # User ID
	root_cause = Column(Text, nullable=True)
	
	# Status
	status = Column(String(50), nullable=False, default='Open')  # Open, Under Investigation, CAPA Required, Closed
	closure_date = Column(Date, nullable=True)
	
	# Approval
	approved_by = Column(String(36), nullable=True)  # User ID
	approved_date = Column(DateTime, nullable=True)
	
	# Relationships
	corrective_actions = relationship("PHRCCorrectiveAction", back_populates="deviation")
	
	def __repr__(self):
		return f"<PHRCDeviation {self.deviation_number}>"


class PHRCCorrectiveAction(Model, AuditMixin, BaseMixin):
	"""
	Corrective and Preventive Actions (CAPA).
	"""
	__tablename__ = 'ph_rc_corrective_action'
	
	# Identity
	action_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Action Information
	action_number = Column(String(100), nullable=False, unique=True, index=True)
	action_title = Column(String(500), nullable=False)
	description = Column(Text, nullable=False)
	
	# Source
	source_type = Column(String(50), nullable=False)  # Deviation, Audit Finding, Customer Complaint, etc.
	deviation_id = Column(String(36), ForeignKey('ph_rc_deviation.deviation_id'), nullable=True)
	finding_id = Column(String(36), ForeignKey('ph_rc_audit_finding.finding_id'), nullable=True)
	
	# Classification
	action_type = Column(String(50), nullable=False)  # Corrective, Preventive, Both
	category = Column(String(100), nullable=True)  # Process, Training, System, etc.
	
	# Planning
	planned_start_date = Column(Date, nullable=True)
	planned_completion_date = Column(Date, nullable=False)
	assigned_to = Column(String(36), nullable=False)  # User ID
	
	# Implementation
	actual_start_date = Column(Date, nullable=True)
	actual_completion_date = Column(Date, nullable=True)
	implementation_notes = Column(Text, nullable=True)
	
	# Status
	status = Column(String(50), nullable=False, default='Planned')  # Planned, In Progress, Completed, Verified, Closed
	
	# Effectiveness
	effectiveness_check_required = Column(Boolean, default=True)
	effectiveness_check_date = Column(Date, nullable=True)
	effectiveness_verified = Column(Boolean, default=False)
	effectiveness_notes = Column(Text, nullable=True)
	
	# Approval
	approved_by = Column(String(36), nullable=True)  # User ID
	approved_date = Column(DateTime, nullable=True)
	
	# Relationships
	deviation = relationship("PHRCDeviation", back_populates="corrective_actions")
	finding = relationship("PHRCAuditFinding", back_populates="corrective_actions")
	
	def __repr__(self):
		return f"<PHRCCorrectiveAction {self.action_number}>"


class PHRCComplianceControl(Model, AuditMixin, BaseMixin):
	"""
	Automated compliance controls and monitoring.
	"""
	__tablename__ = 'ph_rc_compliance_control'
	
	# Identity
	control_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Control Information
	control_code = Column(String(50), nullable=False, unique=True, index=True)
	control_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=False)
	
	# Framework
	framework_id = Column(String(36), ForeignKey('ph_rc_regulatory_framework.framework_id'), nullable=False)
	regulation_reference = Column(String(200), nullable=True)
	
	# Configuration
	control_type = Column(String(50), nullable=False)  # Automated, Manual, Hybrid
	frequency = Column(String(50), nullable=True)  # Real-time, Daily, Weekly, etc.
	severity = Column(String(50), nullable=False)  # Critical, High, Medium, Low
	
	# Implementation
	implementation_details = Column(JSON, nullable=True)
	monitoring_query = Column(Text, nullable=True)
	threshold_values = Column(JSON, nullable=True)
	
	# Status
	is_active = Column(Boolean, default=True)
	last_execution = Column(DateTime, nullable=True)
	next_execution = Column(DateTime, nullable=True)
	
	# Results
	last_result = Column(String(50), nullable=True)  # Pass, Fail, Warning
	failure_count = Column(Integer, default=0)
	
	# Relationships
	framework = relationship("PHRCRegulatoryFramework", back_populates="controls")
	
	def __repr__(self):
		return f"<PHRCComplianceControl {self.control_name}>"


class PHRCRegulatoryContact(Model, AuditMixin, BaseMixin):
	"""
	Regulatory authority contacts and relationship management.
	"""
	__tablename__ = 'ph_rc_regulatory_contact'
	
	# Identity
	contact_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Contact Information
	contact_name = Column(String(200), nullable=False)
	title = Column(String(200), nullable=True)
	organization = Column(String(200), nullable=False)
	department = Column(String(200), nullable=True)
	
	# Contact Details
	email = Column(String(320), nullable=True)
	phone = Column(String(50), nullable=True)
	address = Column(JSON, nullable=True)
	
	# Specialization
	expertise_areas = Column(JSON, nullable=True)  # List of expertise areas
	product_types = Column(JSON, nullable=True)  # Types of products they handle
	
	# Relationship
	relationship_type = Column(String(50), nullable=False)  # Primary, Secondary, Emergency
	preferred_communication = Column(String(50), nullable=True)  # Email, Phone, etc.
	notes = Column(Text, nullable=True)
	
	# Status
	is_active = Column(Boolean, default=True)
	
	# Relationships
	submissions = relationship("PHRCSubmission", back_populates="regulatory_contact")
	
	def __repr__(self):
		return f"<PHRCRegulatoryContact {self.contact_name}>"


class PHRCInspection(Model, AuditMixin, BaseMixin):
	"""
	Regulatory inspections and preparation tracking.
	"""
	__tablename__ = 'ph_rc_inspection'
	
	# Identity
	inspection_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Inspection Information
	inspection_number = Column(String(100), nullable=False, unique=True, index=True)
	inspection_type = Column(String(50), nullable=False)  # Pre-approval, Routine, For-cause, etc.
	inspection_scope = Column(Text, nullable=True)
	
	# Authority
	regulatory_authority = Column(String(200), nullable=False)
	lead_inspector = Column(String(200), nullable=True)
	inspection_team = Column(JSON, nullable=True)
	
	# Schedule
	notification_date = Column(Date, nullable=True)
	planned_start_date = Column(Date, nullable=True)
	planned_end_date = Column(Date, nullable=True)
	actual_start_date = Column(Date, nullable=True)
	actual_end_date = Column(Date, nullable=True)
	
	# Preparation
	preparation_status = Column(String(50), nullable=False, default='Not Started')
	preparation_checklist = Column(JSON, nullable=True)
	responsible_team = Column(JSON, nullable=True)
	
	# Results
	status = Column(String(50), nullable=False, default='Scheduled')  # Scheduled, In Progress, Completed
	outcome = Column(String(50), nullable=True)  # No Action Indicated, Voluntary Action, Official Action
	
	# Follow-up
	inspection_report_received = Column(Boolean, default=False)
	response_required = Column(Boolean, default=False)
	response_deadline = Column(Date, nullable=True)
	
	def __repr__(self):
		return f"<PHRCInspection {self.inspection_number}>"


class PHRCRegulatoryReport(Model, AuditMixin, BaseMixin):
	"""
	Regulatory reports and periodic submissions.
	"""
	__tablename__ = 'ph_rc_regulatory_report'
	
	# Identity
	report_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Report Information
	report_number = Column(String(100), nullable=False, unique=True, index=True)
	report_type = Column(String(100), nullable=False)  # Annual Report, PSUR, etc.
	report_title = Column(String(500), nullable=False)
	
	# Period
	reporting_period_start = Column(Date, nullable=False)
	reporting_period_end = Column(Date, nullable=False)
	
	# Submission
	due_date = Column(Date, nullable=False)
	submission_date = Column(Date, nullable=True)
	submitted_by = Column(String(36), nullable=True)  # User ID
	
	# Content
	report_content = Column(JSON, nullable=True)
	attachments = Column(JSON, nullable=True)
	
	# Status
	status = Column(String(50), nullable=False, default='Draft')  # Draft, In Review, Submitted, Accepted
	
	# Approval
	approved_by = Column(String(36), nullable=True)  # User ID
	approved_date = Column(DateTime, nullable=True)
	
	def __repr__(self):
		return f"<PHRCRegulatoryReport {self.report_number}>"