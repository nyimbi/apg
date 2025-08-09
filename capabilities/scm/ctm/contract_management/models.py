"""
Contract Management Models

Database models for contract lifecycle management, amendments, renewals, and compliance.
"""

from datetime import datetime, date
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str

from ...auth_rbac.models import BaseMixin, AuditMixin, Model


class PPCContract(Model, AuditMixin, BaseMixin):
	"""Contract master records"""
	__tablename__ = 'ppc_contract'
	
	# Identity
	contract_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Contract Information
	contract_number = Column(String(50), nullable=False, index=True)
	contract_title = Column(String(200), nullable=False)
	contract_type = Column(String(50), default='Purchase')  # Purchase, Service, Master Agreement, etc.
	description = Column(Text, nullable=True)
	
	# Parties
	vendor_id = Column(String(36), nullable=False, index=True)
	vendor_name = Column(String(200), nullable=False)
	buyer_id = Column(String(36), nullable=False, index=True)
	buyer_name = Column(String(100), nullable=False)
	
	# Contract Dates
	effective_date = Column(Date, nullable=False, index=True)
	expiration_date = Column(Date, nullable=False, index=True)
	execution_date = Column(Date, nullable=True)
	
	# Status
	status = Column(String(20), default='Draft', index=True)  # Draft, Active, Expired, Terminated, Renewed
	
	# Financial Information
	contract_value = Column(DECIMAL(15, 2), default=0.00)
	committed_spend = Column(DECIMAL(15, 2), default=0.00)
	actual_spend = Column(DECIMAL(15, 2), default=0.00)
	currency_code = Column(String(3), default='USD')
	
	# Terms and Conditions
	payment_terms = Column(String(100), nullable=True)
	delivery_terms = Column(String(100), nullable=True)
	performance_terms = Column(Text, nullable=True)
	warranty_terms = Column(Text, nullable=True)
	
	# Legal and Compliance
	governing_law = Column(String(100), nullable=True)
	dispute_resolution = Column(String(100), nullable=True)
	confidentiality_required = Column(Boolean, default=False)
	
	# Renewal Information
	auto_renewal = Column(Boolean, default=False)
	renewal_notice_days = Column(Integer, default=90)
	renewal_terms = Column(Text, nullable=True)
	
	# Risk and Insurance
	insurance_required = Column(Boolean, default=False)
	insurance_amount = Column(DECIMAL(15, 2), default=0.00)
	liability_cap = Column(DECIMAL(15, 2), default=0.00)
	
	# Performance Metrics
	sla_requirements = Column(Text, nullable=True)
	kpi_metrics = Column(Text, nullable=True)
	penalty_clauses = Column(Text, nullable=True)
	
	# Document Management
	contract_document_path = Column(String(500), nullable=True)
	signed_document_path = Column(String(500), nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'contract_number', name='uq_contract_number_tenant'),
	)
	
	# Relationships
	lines = relationship("PPCContractLine", back_populates="contract", cascade="all, delete-orphan")
	amendments = relationship("PPCContractAmendment", back_populates="contract")
	renewals = relationship("PPCContractRenewal", back_populates="contract")
	milestones = relationship("PPCContractMilestone", back_populates="contract")
	documents = relationship("PPCContractDocument", back_populates="contract")
	
	def __repr__(self):
		return f"<PPCContract {self.contract_number} - {self.status} - ${self.contract_value}>"
	
	def is_expiring_soon(self, days: int = 90) -> bool:
		"""Check if contract is expiring within specified days"""
		from datetime import timedelta
		
		if self.status != 'Active':
			return False
		
		cutoff_date = date.today() + timedelta(days=days)
		return self.expiration_date <= cutoff_date
	
	def get_utilization_percentage(self) -> float:
		"""Get contract utilization percentage"""
		if self.contract_value == 0:
			return 0.0
		
		return float((self.actual_spend / self.contract_value) * 100)


class PPCContractLine(Model, AuditMixin, BaseMixin):
	"""Contract line items for detailed terms"""
	__tablename__ = 'ppc_contract_line'
	
	# Identity
	line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	contract_id = Column(String(36), ForeignKey('ppc_contract.contract_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Information
	line_number = Column(Integer, nullable=False)
	description = Column(Text, nullable=False)
	item_category = Column(String(100), nullable=True)
	
	# Quantities and Pricing
	quantity = Column(DECIMAL(12, 4), default=0.0000)
	unit_of_measure = Column(String(20), nullable=True)
	unit_price = Column(DECIMAL(15, 4), default=0.0000)
	line_value = Column(DECIMAL(15, 2), default=0.00)
	
	# Service Level Agreements
	service_level = Column(String(100), nullable=True)
	response_time = Column(String(50), nullable=True)
	uptime_requirement = Column(DECIMAL(5, 2), nullable=True)  # Percentage
	
	# Performance Metrics
	performance_standard = Column(Text, nullable=True)
	measurement_criteria = Column(Text, nullable=True)
	
	# Pricing Terms
	pricing_model = Column(String(50), nullable=True)  # Fixed, Variable, Cost Plus, etc.
	price_escalation = Column(DECIMAL(5, 2), default=0.00)  # Annual percentage
	volume_discounts = Column(Text, nullable=True)
	
	# Relationships
	contract = relationship("PPCContract", back_populates="lines")
	
	def __repr__(self):
		return f"<PPCContractLine {self.line_number}: {self.description}>"


class PPCContractAmendment(Model, AuditMixin, BaseMixin):
	"""Contract amendments and modifications"""
	__tablename__ = 'ppc_contract_amendment'
	
	# Identity
	amendment_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	contract_id = Column(String(36), ForeignKey('ppc_contract.contract_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Amendment Information
	amendment_number = Column(String(50), nullable=False, index=True)
	amendment_title = Column(String(200), nullable=False)
	amendment_type = Column(String(50), nullable=False)  # Price, Term, Scope, etc.
	description = Column(Text, nullable=False)
	
	# Amendment Details
	requested_by = Column(String(36), nullable=False)
	requested_by_name = Column(String(100), nullable=False)
	request_date = Column(Date, default=date.today)
	justification = Column(Text, nullable=False)
	
	# Status and Approval
	status = Column(String(20), default='Draft', index=True)  # Draft, Pending, Approved, Rejected, Executed
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(Date, nullable=True)
	
	# Effective Information
	effective_date = Column(Date, nullable=True)
	executed_date = Column(Date, nullable=True)
	
	# Financial Impact
	original_value = Column(DECIMAL(15, 2), default=0.00)
	amended_value = Column(DECIMAL(15, 2), default=0.00)
	value_change = Column(DECIMAL(15, 2), default=0.00)
	
	# Term Changes
	original_expiration_date = Column(Date, nullable=True)
	new_expiration_date = Column(Date, nullable=True)
	
	# Document Management
	amendment_document_path = Column(String(500), nullable=True)
	signed_amendment_path = Column(String(500), nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'amendment_number', name='uq_amendment_number_tenant'),
	)
	
	# Relationships
	contract = relationship("PPCContract", back_populates="amendments")
	
	def __repr__(self):
		return f"<PPCContractAmendment {self.amendment_number} - {self.status}>"


class PPCContractRenewal(Model, AuditMixin, BaseMixin):
	"""Contract renewal tracking"""
	__tablename__ = 'ppc_contract_renewal'
	
	# Identity
	renewal_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	contract_id = Column(String(36), ForeignKey('ppc_contract.contract_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Renewal Information
	renewal_number = Column(Integer, nullable=False)  # 1st renewal, 2nd renewal, etc.
	renewal_type = Column(String(20), default='Extension')  # Extension, New Term, Renegotiation
	
	# Timeline
	notice_date = Column(Date, nullable=True)
	decision_due_date = Column(Date, nullable=False)
	new_start_date = Column(Date, nullable=True)
	new_end_date = Column(Date, nullable=True)
	
	# Status
	renewal_status = Column(String(20), default='Pending', index=True)  # Pending, Approved, Declined, Executed
	
	# Terms
	new_contract_value = Column(DECIMAL(15, 2), default=0.00)
	price_adjustment = Column(DECIMAL(5, 2), default=0.00)  # Percentage change
	term_changes = Column(Text, nullable=True)
	
	# Decision Information
	decision_maker_id = Column(String(36), nullable=True)
	decision_date = Column(Date, nullable=True)
	decision_rationale = Column(Text, nullable=True)
	
	# Execution
	executed = Column(Boolean, default=False)
	execution_date = Column(Date, nullable=True)
	new_contract_id = Column(String(36), nullable=True)  # If new contract created
	
	# Relationships
	contract = relationship("PPCContract", back_populates="renewals")
	
	def __repr__(self):
		return f"<PPCContractRenewal {self.contract.contract_number} - Renewal #{self.renewal_number}>"


class PPCContractMilestone(Model, AuditMixin, BaseMixin):
	"""Contract milestones and deliverables"""
	__tablename__ = 'ppc_contract_milestone'
	
	# Identity
	milestone_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	contract_id = Column(String(36), ForeignKey('ppc_contract.contract_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Milestone Information
	milestone_name = Column(String(200), nullable=False)
	milestone_type = Column(String(50), default='Deliverable')  # Deliverable, Payment, Review, etc.
	description = Column(Text, nullable=True)
	
	# Timeline
	planned_date = Column(Date, nullable=False)
	actual_date = Column(Date, nullable=True)
	due_date = Column(Date, nullable=True)
	
	# Status
	status = Column(String(20), default='Planned', index=True)  # Planned, In Progress, Completed, Overdue, Cancelled
	completion_percentage = Column(DECIMAL(5, 2), default=0.00)
	
	# Financial Information
	milestone_value = Column(DECIMAL(15, 2), default=0.00)
	payment_trigger = Column(Boolean, default=False)
	
	# Dependencies
	prerequisite_milestone_id = Column(String(36), nullable=True)
	critical_path = Column(Boolean, default=False)
	
	# Acceptance Criteria
	acceptance_criteria = Column(Text, nullable=True)
	acceptance_status = Column(String(20), default='Pending')  # Pending, Accepted, Rejected
	accepted_by = Column(String(36), nullable=True)
	accepted_date = Column(Date, nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Relationships
	contract = relationship("PPCContract", back_populates="milestones")
	
	def __repr__(self):
		return f"<PPCContractMilestone {self.milestone_name} - {self.status}>"
	
	def is_overdue(self) -> bool:
		"""Check if milestone is overdue"""
		if self.status == 'Completed':
			return False
		
		due_date = self.due_date or self.planned_date
		return date.today() > due_date


class PPCContractDocument(Model, AuditMixin, BaseMixin):
	"""Contract-related documents"""
	__tablename__ = 'ppc_contract_document'
	
	# Identity
	document_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	contract_id = Column(String(36), ForeignKey('ppc_contract.contract_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Document Information
	document_name = Column(String(200), nullable=False)
	document_type = Column(String(50), nullable=False)  # Contract, Amendment, SOW, Invoice, etc.
	document_category = Column(String(50), nullable=True)
	description = Column(Text, nullable=True)
	
	# Document Details
	document_version = Column(String(20), default='1.0')
	file_name = Column(String(255), nullable=False)
	file_path = Column(String(500), nullable=False)
	file_size = Column(Integer, default=0)  # Size in bytes
	mime_type = Column(String(100), nullable=True)
	
	# Status
	is_active = Column(Boolean, default=True)
	is_signed = Column(Boolean, default=False)
	requires_signature = Column(Boolean, default=False)
	
	# Upload Information
	uploaded_by = Column(String(36), nullable=False)
	uploaded_by_name = Column(String(100), nullable=False)
	upload_date = Column(DateTime, default=datetime.utcnow)
	
	# Access Control
	is_confidential = Column(Boolean, default=False)
	access_level = Column(String(20), default='Internal')  # Public, Internal, Restricted, Confidential
	
	# Relationships
	contract = relationship("PPCContract", back_populates="documents")
	
	def __repr__(self):
		return f"<PPCContractDocument {self.document_name} - {self.document_type}>"