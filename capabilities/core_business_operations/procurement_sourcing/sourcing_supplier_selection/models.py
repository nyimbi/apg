"""
Sourcing & Supplier Selection Models

Database models for RFQ/RFP processes, bid management, and supplier evaluation.
"""

from datetime import datetime, date
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str

from ...auth_rbac.models import BaseMixin, AuditMixin, Model


class PPSRFQHeader(Model, AuditMixin, BaseMixin):
	"""Request for Quote/Proposal header"""
	__tablename__ = 'pps_rfq_header'
	
	# Identity
	rfq_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# RFQ Information
	rfq_number = Column(String(50), nullable=False, index=True)
	rfq_title = Column(String(200), nullable=False)
	rfq_type = Column(String(20), default='RFQ')  # RFQ, RFP, RFI
	description = Column(Text, nullable=True)
	
	# Sourcing Details
	sourcing_manager_id = Column(String(36), nullable=False, index=True)
	sourcing_manager_name = Column(String(100), nullable=False)
	department = Column(String(50), nullable=True)
	category = Column(String(100), nullable=True)
	
	# Timeline
	issue_date = Column(Date, default=date.today, nullable=False)
	response_due_date = Column(Date, nullable=False, index=True)
	evaluation_complete_date = Column(Date, nullable=True)
	award_date = Column(Date, nullable=True)
	
	# Status
	status = Column(String(20), default='Draft', index=True)  # Draft, Issued, Closed, Awarded, Cancelled
	
	# Evaluation Configuration
	evaluation_method = Column(String(50), default='Lowest Price')  # Lowest Price, Best Value, Technical
	auto_extend_deadline = Column(Boolean, default=False)
	allow_partial_bids = Column(Boolean, default=True)
	
	# Financial Information
	estimated_value = Column(DECIMAL(15, 2), default=0.00)
	currency_code = Column(String(3), default='USD')
	
	# Terms and Conditions
	terms_and_conditions = Column(Text, nullable=True)
	payment_terms = Column(String(50), nullable=True)
	delivery_terms = Column(String(50), nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'rfq_number', name='uq_rfq_number_tenant'),
	)
	
	# Relationships
	lines = relationship("PPSRFQLine", back_populates="rfq_header", cascade="all, delete-orphan")
	bids = relationship("PPSBid", back_populates="rfq_header")
	evaluations = relationship("PPSSupplierEvaluation", back_populates="rfq_header")
	
	def __repr__(self):
		return f"<PPSRFQHeader {self.rfq_number} - {self.status}>"


class PPSRFQLine(Model, AuditMixin, BaseMixin):
	"""RFQ line items"""
	__tablename__ = 'pps_rfq_line'
	
	# Identity
	rfq_line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	rfq_id = Column(String(36), ForeignKey('pps_rfq_header.rfq_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Information
	line_number = Column(Integer, nullable=False)
	description = Column(Text, nullable=False)
	specifications = Column(Text, nullable=True)
	
	# Item Information
	item_code = Column(String(50), nullable=True)
	item_category = Column(String(100), nullable=True)
	
	# Quantity and Units
	quantity_required = Column(DECIMAL(12, 4), default=1.0000)
	unit_of_measure = Column(String(20), default='EA')
	
	# Delivery Requirements
	delivery_date_required = Column(Date, nullable=True)
	delivery_location = Column(String(200), nullable=True)
	
	# Evaluation Criteria
	quality_requirements = Column(Text, nullable=True)
	technical_requirements = Column(Text, nullable=True)
	service_requirements = Column(Text, nullable=True)
	
	# Relationships
	rfq_header = relationship("PPSRFQHeader", back_populates="lines")
	bid_lines = relationship("PPSBidLine", back_populates="rfq_line")
	
	def __repr__(self):
		return f"<PPSRFQLine {self.line_number}: {self.description}>"


class PPSBid(Model, AuditMixin, BaseMixin):
	"""Supplier bids/proposals"""
	__tablename__ = 'pps_bid'
	
	# Identity
	bid_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	rfq_id = Column(String(36), ForeignKey('pps_rfq_header.rfq_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Bid Information
	bid_number = Column(String(50), nullable=False, index=True)
	supplier_id = Column(String(36), nullable=False, index=True)
	supplier_name = Column(String(200), nullable=False)
	
	# Submission Details
	submitted_date = Column(DateTime, nullable=True)
	submitted_by = Column(String(100), nullable=True)
	submission_method = Column(String(50), default='Online')  # Online, Email, Paper
	
	# Status
	status = Column(String(20), default='Draft', index=True)  # Draft, Submitted, Under Review, Accepted, Rejected
	
	# Financial Summary
	total_bid_amount = Column(DECIMAL(15, 2), default=0.00)
	currency_code = Column(String(3), default='USD')
	
	# Evaluation Results
	technical_score = Column(DECIMAL(5, 2), default=0.00)
	commercial_score = Column(DECIMAL(5, 2), default=0.00)
	overall_score = Column(DECIMAL(5, 2), default=0.00)
	rank = Column(Integer, nullable=True)
	
	# Bid Validity
	bid_valid_until = Column(Date, nullable=True)
	
	# Comments and Notes
	supplier_comments = Column(Text, nullable=True)
	internal_notes = Column(Text, nullable=True)
	
	# Award Information
	awarded = Column(Boolean, default=False)
	award_amount = Column(DECIMAL(15, 2), default=0.00)
	award_date = Column(Date, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'bid_number', name='uq_bid_number_tenant'),
	)
	
	# Relationships
	rfq_header = relationship("PPSRFQHeader", back_populates="bids")
	lines = relationship("PPSBidLine", back_populates="bid", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<PPSBid {self.bid_number} - {self.supplier_name} - ${self.total_bid_amount}>"


class PPSBidLine(Model, AuditMixin, BaseMixin):
	"""Bid line items"""
	__tablename__ = 'pps_bid_line'
	
	# Identity
	bid_line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	bid_id = Column(String(36), ForeignKey('pps_bid.bid_id'), nullable=False, index=True)
	rfq_line_id = Column(String(36), ForeignKey('pps_rfq_line.rfq_line_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Information
	line_number = Column(Integer, nullable=False)
	
	# Pricing
	unit_price = Column(DECIMAL(15, 4), default=0.0000)
	extended_price = Column(DECIMAL(15, 2), default=0.00)
	
	# Delivery
	delivery_lead_time = Column(Integer, nullable=True)  # Days
	delivery_date_proposed = Column(Date, nullable=True)
	
	# Technical Response
	technical_compliance = Column(String(20), default='Compliant')  # Compliant, Non-Compliant, Partial
	technical_comments = Column(Text, nullable=True)
	
	# Alternative Proposals
	is_alternative = Column(Boolean, default=False)
	alternative_description = Column(Text, nullable=True)
	
	# Relationships
	bid = relationship("PPSBid", back_populates="lines")
	rfq_line = relationship("PPSRFQLine", back_populates="bid_lines")
	
	def __repr__(self):
		return f"<PPSBidLine {self.line_number}: ${self.unit_price}>"


class PPSSupplierEvaluation(Model, AuditMixin, BaseMixin):
	"""Supplier evaluation records"""
	__tablename__ = 'pps_supplier_evaluation'
	
	# Identity
	evaluation_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	rfq_id = Column(String(36), ForeignKey('pps_rfq_header.rfq_id'), nullable=False, index=True)
	supplier_id = Column(String(36), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Evaluation Information
	evaluator_id = Column(String(36), nullable=False)
	evaluator_name = Column(String(100), nullable=False)
	evaluation_date = Column(Date, default=date.today)
	
	# Scores
	technical_score = Column(DECIMAL(5, 2), default=0.00)
	commercial_score = Column(DECIMAL(5, 2), default=0.00)
	delivery_score = Column(DECIMAL(5, 2), default=0.00)
	quality_score = Column(DECIMAL(5, 2), default=0.00)
	service_score = Column(DECIMAL(5, 2), default=0.00)
	overall_score = Column(DECIMAL(5, 2), default=0.00)
	
	# Recommendations
	recommendation = Column(String(20), default='Under Review')  # Recommend, Do Not Recommend, Conditional
	recommendation_reason = Column(Text, nullable=True)
	
	# Risk Assessment
	risk_level = Column(String(20), default='Medium')  # Low, Medium, High
	risk_factors = Column(Text, nullable=True)
	
	# Comments
	strengths = Column(Text, nullable=True)
	weaknesses = Column(Text, nullable=True)
	additional_comments = Column(Text, nullable=True)
	
	# Relationships
	rfq_header = relationship("PPSRFQHeader", back_populates="evaluations")
	
	def __repr__(self):
		return f"<PPSSupplierEvaluation {self.supplier_id} - Score: {self.overall_score}>"


class PPSEvaluationCriteria(Model, AuditMixin, BaseMixin):
	"""Evaluation criteria configuration"""
	__tablename__ = 'pps_evaluation_criteria'
	
	# Identity
	criteria_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Criteria Information
	criteria_name = Column(String(100), nullable=False)
	criteria_description = Column(Text, nullable=True)
	criteria_category = Column(String(50), nullable=False)  # Technical, Commercial, Quality, etc.
	
	# Weighting
	weight_percentage = Column(DECIMAL(5, 2), default=0.00)  # Percentage of total score
	max_points = Column(DECIMAL(5, 2), default=100.00)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	is_mandatory = Column(Boolean, default=False)
	
	# Scoring Guidelines
	scoring_guidelines = Column(Text, nullable=True)
	
	def __repr__(self):
		return f"<PPSEvaluationCriteria {self.criteria_name} - {self.weight_percentage}%>"


class PPSAwardRecommendation(Model, AuditMixin, BaseMixin):
	"""Award recommendations"""
	__tablename__ = 'pps_award_recommendation'
	
	# Identity
	recommendation_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	rfq_id = Column(String(36), ForeignKey('pps_rfq_header.rfq_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Recommendation Information
	recommended_supplier_id = Column(String(36), nullable=False)
	recommended_supplier_name = Column(String(200), nullable=False)
	recommended_bid_id = Column(String(36), ForeignKey('pps_bid.bid_id'), nullable=False)
	
	# Recommendation Details
	recommender_id = Column(String(36), nullable=False)
	recommender_name = Column(String(100), nullable=False)
	recommendation_date = Column(Date, default=date.today)
	
	# Financial Information
	recommended_amount = Column(DECIMAL(15, 2), default=0.00)
	estimated_savings = Column(DECIMAL(15, 2), default=0.00)
	savings_percentage = Column(DECIMAL(5, 2), default=0.00)
	
	# Justification
	award_justification = Column(Text, nullable=False)
	key_differentiators = Column(Text, nullable=True)
	risk_mitigation = Column(Text, nullable=True)
	
	# Approval Status
	status = Column(String(20), default='Pending', index=True)  # Pending, Approved, Rejected
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(Date, nullable=True)
	
	# Implementation
	implemented = Column(Boolean, default=False)
	implementation_date = Column(Date, nullable=True)
	
	# Relationships
	rfq_header = relationship("PPSRFQHeader")
	recommended_bid = relationship("PPSBid")
	
	def __repr__(self):
		return f"<PPSAwardRecommendation {self.recommended_supplier_name} - ${self.recommended_amount}>"