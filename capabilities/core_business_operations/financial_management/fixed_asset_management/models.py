"""
Fixed Asset Management Models

Database models for the Fixed Asset Management sub-capability including
asset master data, categories, depreciation, acquisitions, disposals,
transfers, maintenance, insurance, valuations, and lease tracking.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ...auth_rbac.models import BaseMixin, AuditMixin, Model


class CFAMAssetCategory(Model, AuditMixin, BaseMixin):
	"""
	Asset categories for classification and default settings.
	
	Defines categories like Buildings, Equipment, Vehicles with
	default depreciation methods and GL account mappings.
	"""
	__tablename__ = 'cf_fam_asset_category'
	
	# Identity
	category_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Category Information
	category_code = Column(String(20), nullable=False, index=True)
	category_name = Column(String(100), nullable=False, index=True)
	description = Column(Text, nullable=True)
	
	# Hierarchy
	parent_category_id = Column(String(36), ForeignKey('cf_fam_asset_category.category_id'), nullable=True, index=True)
	
	# Default Settings
	default_useful_life_years = Column(Integer, default=5)
	default_useful_life_months = Column(Integer, default=0)
	default_depreciation_method_id = Column(String(36), ForeignKey('cf_fam_depreciation_method.method_id'), nullable=True, index=True)
	default_salvage_percent = Column(DECIMAL(5, 2), default=0.00)  # % of cost
	
	# GL Account Mappings
	gl_asset_account_id = Column(String(36), nullable=True)  # FK to CFGLAccount
	gl_depreciation_account_id = Column(String(36), nullable=True)  # Accumulated Depreciation
	gl_expense_account_id = Column(String(36), nullable=True)  # Depreciation Expense
	
	# Configuration
	is_active = Column(Boolean, default=True)
	allow_depreciation = Column(Boolean, default=True)
	require_location = Column(Boolean, default=True)
	require_custodian = Column(Boolean, default=False)
	minimum_cost = Column(DECIMAL(15, 2), default=0.00)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'category_code', name='uq_asset_category_code_tenant'),
	)
	
	# Relationships
	parent_category = relationship("CFAMAssetCategory", remote_side=[category_id])
	child_categories = relationship("CFAMAssetCategory")
	default_depreciation_method = relationship("CFAMDepreciationMethod", foreign_keys=[default_depreciation_method_id])
	assets = relationship("CFAMAsset", back_populates="category")
	
	def __repr__(self):
		return f"<CFAMAssetCategory {self.category_code} - {self.category_name}>"
	
	def get_full_path(self) -> str:
		"""Get full category path"""
		if self.parent_category:
			return f"{self.parent_category.get_full_path()} > {self.category_name}"
		return self.category_name


class CFAMDepreciationMethod(Model, AuditMixin, BaseMixin):
	"""
	Depreciation methods for asset depreciation calculations.
	
	Supports various depreciation methods including straight-line,
	declining balance, sum of years digits, and units of production.
	"""
	__tablename__ = 'cf_fam_depreciation_method'
	
	# Identity
	method_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Method Information
	method_code = Column(String(20), nullable=False, index=True)
	method_name = Column(String(100), nullable=False)
	description = Column(Text, nullable=True)
	
	# Method Configuration
	formula = Column(String(50), nullable=False)  # straight_line, declining_balance, etc.
	depreciation_rate = Column(DECIMAL(5, 2), nullable=True)  # For declining balance methods
	convention = Column(String(20), default='half_year')  # half_year, full_year, mid_month
	
	# Status
	is_active = Column(Boolean, default=True)
	is_system = Column(Boolean, default=False)  # System-defined methods
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'method_code', name='uq_depreciation_method_code_tenant'),
	)
	
	# Relationships
	categories = relationship("CFAMAssetCategory", back_populates="default_depreciation_method")
	assets = relationship("CFAMAsset", back_populates="depreciation_method")
	depreciation_records = relationship("CFAMDepreciation", back_populates="method")
	
	def __repr__(self):
		return f"<CFAMDepreciationMethod {self.method_code} - {self.method_name}>"
	
	def calculate_depreciation(self, cost: Decimal, salvage_value: Decimal, 
							  useful_life_years: int, period_number: int = 1,
							  units_produced: int = None, total_units: int = None) -> Decimal:
		"""Calculate depreciation for a period"""
		depreciable_base = cost - salvage_value
		
		if self.formula == 'straight_line':
			return depreciable_base / (useful_life_years * 12)  # Monthly depreciation
		
		elif self.formula == 'declining_balance':
			rate = self.depreciation_rate / 100 if self.depreciation_rate else (1 / useful_life_years)
			remaining_value = cost * ((1 - rate) ** (period_number - 1))
			return remaining_value * rate
		
		elif self.formula == 'double_declining_balance':
			rate = 2 / useful_life_years
			remaining_value = cost * ((1 - rate) ** (period_number - 1))
			return remaining_value * rate
		
		elif self.formula == 'sum_of_years_digits':
			sum_of_years = sum(range(1, useful_life_years + 1))
			years_remaining = useful_life_years - period_number + 1
			return depreciable_base * (years_remaining / sum_of_years) / 12  # Monthly
		
		elif self.formula == 'units_of_production':
			if units_produced and total_units:
				return depreciable_base * (units_produced / total_units)
			return Decimal('0.00')
		
		elif self.formula == 'none':
			return Decimal('0.00')
		
		else:
			return depreciable_base / (useful_life_years * 12)  # Default to straight line


class CFAMAsset(Model, AuditMixin, BaseMixin):
	"""
	Fixed asset master data.
	
	Core asset information including identification, financial data,
	location, custodian, and lifecycle status.
	"""
	__tablename__ = 'cf_fam_asset'
	
	# Identity
	asset_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Asset Identification
	asset_number = Column(String(50), nullable=False, index=True)
	asset_tag = Column(String(50), nullable=True, index=True)  # Physical tag/barcode
	asset_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	
	# Classification
	category_id = Column(String(36), ForeignKey('cf_fam_asset_category.category_id'), nullable=False, index=True)
	
	# Financial Information
	acquisition_cost = Column(DECIMAL(15, 2), nullable=False)
	salvage_value = Column(DECIMAL(15, 2), default=0.00)
	current_book_value = Column(DECIMAL(15, 2), default=0.00)
	accumulated_depreciation = Column(DECIMAL(15, 2), default=0.00)
	fair_market_value = Column(DECIMAL(15, 2), nullable=True)
	replacement_cost = Column(DECIMAL(15, 2), nullable=True)
	
	# Currency
	currency_code = Column(String(3), default='USD')
	
	# Depreciation Settings
	depreciation_method_id = Column(String(36), ForeignKey('cf_fam_depreciation_method.method_id'), nullable=True, index=True)
	useful_life_years = Column(Integer, nullable=True)
	useful_life_months = Column(Integer, nullable=True)
	salvage_percent = Column(DECIMAL(5, 2), default=0.00)
	
	# Dates
	acquisition_date = Column(Date, nullable=False, index=True)
	placed_in_service_date = Column(Date, nullable=True, index=True)
	disposal_date = Column(Date, nullable=True, index=True)
	last_depreciation_date = Column(Date, nullable=True)
	
	# Location and Assignment
	location = Column(String(100), nullable=True)
	department = Column(String(50), nullable=True)
	cost_center = Column(String(20), nullable=True)
	custodian = Column(String(100), nullable=True)  # Employee responsible
	custodian_employee_id = Column(String(36), nullable=True)
	
	# Physical Details
	manufacturer = Column(String(100), nullable=True)
	model = Column(String(100), nullable=True)
	serial_number = Column(String(100), nullable=True, index=True)
	year_manufactured = Column(Integer, nullable=True)
	condition = Column(String(20), default='Good')  # Excellent, Good, Fair, Poor
	
	# Status
	status = Column(String(20), default='Active', index=True)  # Active, Inactive, Disposed, etc.
	is_depreciable = Column(Boolean, default=True)
	is_fully_depreciated = Column(Boolean, default=False)
	
	# Lease Information (for leased assets)
	is_leased = Column(Boolean, default=False)
	lease_id = Column(String(36), ForeignKey('cf_fam_asset_lease.lease_id'), nullable=True, index=True)
	
	# Insurance
	is_insured = Column(Boolean, default=False)
	insurance_value = Column(DECIMAL(15, 2), nullable=True)
	
	# Maintenance
	last_maintenance_date = Column(Date, nullable=True)
	next_maintenance_date = Column(Date, nullable=True)
	maintenance_cost_ytd = Column(DECIMAL(15, 2), default=0.00)
	
	# GL Account Overrides (if different from category defaults)
	gl_asset_account_id = Column(String(36), nullable=True)
	gl_depreciation_account_id = Column(String(36), nullable=True)
	gl_expense_account_id = Column(String(36), nullable=True)
	
	# Component Tracking
	parent_asset_id = Column(String(36), ForeignKey('cf_fam_asset.asset_id'), nullable=True, index=True)
	is_component = Column(Boolean, default=False)
	component_percentage = Column(DECIMAL(5, 2), nullable=True)  # % of parent cost
	
	# Tax Information
	tax_book_value = Column(DECIMAL(15, 2), nullable=True)
	tax_accumulated_depreciation = Column(DECIMAL(15, 2), default=0.00)
	
	# Attachments and Notes
	photo_url = Column(String(500), nullable=True)
	document_attachments = Column(Text, nullable=True)  # JSON array of document URLs
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'asset_number', name='uq_asset_number_tenant'),
		UniqueConstraint('tenant_id', 'asset_tag', name='uq_asset_tag_tenant'),
	)
	
	# Relationships
	category = relationship("CFAMAssetCategory", back_populates="assets")
	depreciation_method = relationship("CFAMDepreciationMethod", back_populates="assets")
	parent_asset = relationship("CFAMAsset", remote_side=[asset_id])
	component_assets = relationship("CFAMAsset")
	lease = relationship("CFAMAssetLease", back_populates="assets")
	
	# Related records
	acquisitions = relationship("CFAMAssetAcquisition", back_populates="asset")
	disposals = relationship("CFAMAssetDisposal", back_populates="asset")
	transfers = relationship("CFAMAssetTransfer", back_populates="asset")
	maintenance_records = relationship("CFAMAssetMaintenance", back_populates="asset")
	insurance_records = relationship("CFAMAssetInsurance", back_populates="asset")
	valuations = relationship("CFAMAssetValuation", back_populates="asset")
	depreciation_records = relationship("CFAMDepreciation", back_populates="asset")
	
	def __repr__(self):
		return f"<CFAMAsset {self.asset_number} - {self.asset_name}>"
	
	def calculate_current_book_value(self) -> Decimal:
		"""Calculate current book value"""
		return self.acquisition_cost - self.accumulated_depreciation
	
	def calculate_depreciation_rate(self) -> Decimal:
		"""Calculate annual depreciation rate"""
		if self.useful_life_years and self.useful_life_years > 0:
			return Decimal('1.00') / self.useful_life_years
		return Decimal('0.00')
	
	def is_disposal_eligible(self) -> bool:
		"""Check if asset can be disposed"""
		return self.status in ['Active', 'Inactive'] and not self.disposal_date
	
	def get_monthly_depreciation(self) -> Decimal:
		"""Get monthly depreciation amount"""
		if not self.is_depreciable or self.is_fully_depreciated:
			return Decimal('0.00')
		
		if self.depreciation_method:
			return self.depreciation_method.calculate_depreciation(
				self.acquisition_cost,
				self.salvage_value,
				self.useful_life_years or 5
			)
		
		# Default straight-line
		depreciable_base = self.acquisition_cost - self.salvage_value
		if self.useful_life_years:
			return depreciable_base / (self.useful_life_years * 12)
		return Decimal('0.00')


class CFAMAssetAcquisition(Model, AuditMixin, BaseMixin):
	"""
	Asset acquisition records.
	
	Tracks how assets were acquired including purchase details,
	vendor information, and GL posting.
	"""
	__tablename__ = 'cf_fam_asset_acquisition'
	
	# Identity
	acquisition_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Asset Reference
	asset_id = Column(String(36), ForeignKey('cf_fam_asset.asset_id'), nullable=False, index=True)
	
	# Acquisition Information
	acquisition_number = Column(String(50), nullable=False, index=True)
	acquisition_type = Column(String(20), default='Purchase')  # Purchase, Donation, Construction, etc.
	acquisition_date = Column(Date, nullable=False, index=True)
	
	# Financial Details
	gross_cost = Column(DECIMAL(15, 2), nullable=False)
	freight_cost = Column(DECIMAL(15, 2), default=0.00)
	installation_cost = Column(DECIMAL(15, 2), default=0.00)
	other_costs = Column(DECIMAL(15, 2), default=0.00)
	total_cost = Column(DECIMAL(15, 2), nullable=False)
	
	# Source Information
	vendor_id = Column(String(36), nullable=True)  # FK to AP Vendor
	vendor_name = Column(String(200), nullable=True)
	purchase_order_number = Column(String(50), nullable=True)
	invoice_number = Column(String(50), nullable=True)
	invoice_date = Column(Date, nullable=True)
	
	# Funding
	funding_source = Column(String(50), nullable=True)  # Budget, Grant, Loan, etc.
	project_id = Column(String(36), nullable=True)
	department = Column(String(50), nullable=True)
	cost_center = Column(String(20), nullable=True)
	
	# GL Posting
	is_posted = Column(Boolean, default=False)
	posted_date = Column(DateTime, nullable=True)
	journal_entry_id = Column(String(36), nullable=True)  # FK to GL Journal Entry
	
	# Approval
	requires_approval = Column(Boolean, default=True)
	approved = Column(Boolean, default=False)
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	
	# Documentation
	receipt_number = Column(String(50), nullable=True)
	contract_number = Column(String(50), nullable=True)
	warranty_start_date = Column(Date, nullable=True)
	warranty_end_date = Column(Date, nullable=True)
	warranty_terms = Column(Text, nullable=True)
	
	# Notes
	description = Column(Text, nullable=True)
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'acquisition_number', name='uq_acquisition_number_tenant'),
	)
	
	# Relationships
	asset = relationship("CFAMAsset", back_populates="acquisitions")
	
	def __repr__(self):
		return f"<CFAMAssetAcquisition {self.acquisition_number}>"
	
	def calculate_total_cost(self):
		"""Calculate total acquisition cost"""
		self.total_cost = (
			self.gross_cost + 
			self.freight_cost + 
			self.installation_cost + 
			self.other_costs
		)


class CFAMAssetDisposal(Model, AuditMixin, BaseMixin):
	"""
	Asset disposal records.
	
	Tracks asset disposals including sale, scrap, donation,
	and trade-in transactions with gain/loss calculations.
	"""
	__tablename__ = 'cf_fam_asset_disposal'
	
	# Identity
	disposal_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Asset Reference
	asset_id = Column(String(36), ForeignKey('cf_fam_asset.asset_id'), nullable=False, index=True)
	
	# Disposal Information
	disposal_number = Column(String(50), nullable=False, index=True)
	disposal_date = Column(Date, nullable=False, index=True)
	disposal_method = Column(String(20), nullable=False)  # Sale, Scrap, Trade, Donation, etc.
	disposal_reason = Column(String(50), nullable=True)
	
	# Financial Details
	book_value_at_disposal = Column(DECIMAL(15, 2), nullable=False)
	accumulated_depreciation_at_disposal = Column(DECIMAL(15, 2), nullable=False)
	disposal_proceeds = Column(DECIMAL(15, 2), default=0.00)
	disposal_costs = Column(DECIMAL(15, 2), default=0.00)  # Removal, transportation, etc.
	net_proceeds = Column(DECIMAL(15, 2), default=0.00)
	
	# Gain/Loss Calculation
	gain_loss_amount = Column(DECIMAL(15, 2), default=0.00)
	is_gain = Column(Boolean, nullable=True)  # True = gain, False = loss, None = break-even
	
	# Purchaser/Recipient Information
	purchaser_name = Column(String(200), nullable=True)
	purchaser_contact = Column(String(200), nullable=True)
	sales_agreement_number = Column(String(50), nullable=True)
	
	# Trade-in Information
	trade_in_asset_id = Column(String(36), nullable=True)  # New asset received in trade
	trade_in_allowance = Column(DECIMAL(15, 2), nullable=True)
	
	# GL Posting
	is_posted = Column(Boolean, default=False)
	posted_date = Column(DateTime, nullable=True)
	journal_entry_id = Column(String(36), nullable=True)  # FK to GL Journal Entry
	
	# Approval
	requires_approval = Column(Boolean, default=True)
	approved = Column(Boolean, default=False)
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	
	# Documentation
	disposal_authorization_number = Column(String(50), nullable=True)
	disposal_certificate = Column(String(500), nullable=True)  # URL to certificate
	environmental_clearance = Column(Boolean, default=False)
	
	# Notes
	description = Column(Text, nullable=True)
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'disposal_number', name='uq_disposal_number_tenant'),
	)
	
	# Relationships
	asset = relationship("CFAMAsset", back_populates="disposals")
	
	def __repr__(self):
		return f"<CFAMAssetDisposal {self.disposal_number}>"
	
	def calculate_gain_loss(self):
		"""Calculate gain or loss on disposal"""
		self.net_proceeds = self.disposal_proceeds - self.disposal_costs
		self.gain_loss_amount = self.net_proceeds - self.book_value_at_disposal
		
		if self.gain_loss_amount > 0:
			self.is_gain = True
		elif self.gain_loss_amount < 0:
			self.is_gain = False
		else:
			self.is_gain = None  # Break-even


class CFAMAssetTransfer(Model, AuditMixin, BaseMixin):
	"""
	Asset transfer records.
	
	Tracks transfers between locations, departments, or custodians
	for asset accountability and cost center allocation.
	"""
	__tablename__ = 'cf_fam_asset_transfer'
	
	# Identity
	transfer_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Asset Reference
	asset_id = Column(String(36), ForeignKey('cf_fam_asset.asset_id'), nullable=False, index=True)
	
	# Transfer Information
	transfer_number = Column(String(50), nullable=False, index=True)
	transfer_date = Column(Date, nullable=False, index=True)
	transfer_type = Column(String(20), default='Location')  # Location, Department, Custodian, etc.
	reason = Column(String(100), nullable=True)
	
	# From Information
	from_location = Column(String(100), nullable=True)
	from_department = Column(String(50), nullable=True)
	from_cost_center = Column(String(20), nullable=True)
	from_custodian = Column(String(100), nullable=True)
	from_custodian_employee_id = Column(String(36), nullable=True)
	
	# To Information
	to_location = Column(String(100), nullable=True)
	to_department = Column(String(50), nullable=True)
	to_cost_center = Column(String(20), nullable=True)
	to_custodian = Column(String(100), nullable=True)
	to_custodian_employee_id = Column(String(36), nullable=True)
	
	# Transfer Details
	effective_date = Column(Date, nullable=True)  # When transfer becomes effective
	transfer_cost = Column(DECIMAL(15, 2), default=0.00)  # Cost of transfer (moving, etc.)
	
	# Approval
	requires_approval = Column(Boolean, default=False)
	approved = Column(Boolean, default=True)  # Default approved unless required
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	
	# Status
	status = Column(String(20), default='Completed')  # Pending, Completed, Cancelled
	completed_date = Column(DateTime, nullable=True)
	
	# Documentation
	transfer_authorization = Column(String(50), nullable=True)
	receiving_signature = Column(String(100), nullable=True)
	condition_at_transfer = Column(String(20), nullable=True)
	
	# Notes
	description = Column(Text, nullable=True)
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'transfer_number', name='uq_transfer_number_tenant'),
	)
	
	# Relationships
	asset = relationship("CFAMAsset", back_populates="transfers")
	
	def __repr__(self):
		return f"<CFAMAssetTransfer {self.transfer_number}>"


class CFAMAssetMaintenance(Model, AuditMixin, BaseMixin):
	"""
	Asset maintenance records.
	
	Tracks maintenance activities including preventive, corrective,
	and emergency maintenance with cost tracking and scheduling.
	"""
	__tablename__ = 'cf_fam_asset_maintenance'
	
	# Identity
	maintenance_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Asset Reference
	asset_id = Column(String(36), ForeignKey('cf_fam_asset.asset_id'), nullable=False, index=True)
	
	# Maintenance Information
	maintenance_number = Column(String(50), nullable=False, index=True)
	maintenance_type = Column(String(20), nullable=False)  # Preventive, Corrective, Emergency, etc.
	maintenance_date = Column(Date, nullable=False, index=True)
	scheduled_date = Column(Date, nullable=True, index=True)
	completed_date = Column(Date, nullable=True)
	
	# Service Information
	service_provider = Column(String(200), nullable=True)  # Internal or external
	technician_name = Column(String(100), nullable=True)
	work_order_number = Column(String(50), nullable=True)
	
	# Maintenance Details
	description = Column(Text, nullable=False)
	work_performed = Column(Text, nullable=True)
	parts_replaced = Column(Text, nullable=True)  # JSON array of parts
	findings = Column(Text, nullable=True)
	recommendations = Column(Text, nullable=True)
	
	# Cost Information
	labor_hours = Column(DECIMAL(8, 2), default=0.00)
	labor_rate = Column(DECIMAL(10, 2), default=0.00)
	labor_cost = Column(DECIMAL(15, 2), default=0.00)
	parts_cost = Column(DECIMAL(15, 2), default=0.00)
	other_costs = Column(DECIMAL(15, 2), default=0.00)
	total_cost = Column(DECIMAL(15, 2), default=0.00)
	
	# Status and Priority
	status = Column(String(20), default='Scheduled')  # Scheduled, In Progress, Completed, Cancelled
	priority = Column(String(20), default='Normal')  # Low, Normal, High, Emergency
	urgency = Column(String(20), default='Routine')  # Routine, Urgent, Critical
	
	# Scheduling
	next_maintenance_date = Column(Date, nullable=True)
	maintenance_interval_days = Column(Integer, nullable=True)
	recurring = Column(Boolean, default=False)
	
	# Quality and Safety
	quality_check_passed = Column(Boolean, nullable=True)
	safety_check_passed = Column(Boolean, nullable=True)
	warranty_work = Column(Boolean, default=False)
	warranty_claim_number = Column(String(50), nullable=True)
	
	# Documentation
	photos = Column(Text, nullable=True)  # JSON array of photo URLs
	documents = Column(Text, nullable=True)  # JSON array of document URLs
	inspection_checklist = Column(Text, nullable=True)  # JSON checklist
	
	# Performance Impact
	downtime_hours = Column(DECIMAL(8, 2), default=0.00)
	production_impact = Column(Text, nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	internal_notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'maintenance_number', name='uq_maintenance_number_tenant'),
	)
	
	# Relationships
	asset = relationship("CFAMAsset", back_populates="maintenance_records")
	
	def __repr__(self):
		return f"<CFAMAssetMaintenance {self.maintenance_number}>"
	
	def calculate_total_cost(self):
		"""Calculate total maintenance cost"""
		self.labor_cost = self.labor_hours * self.labor_rate
		self.total_cost = self.labor_cost + self.parts_cost + self.other_costs


class CFAMAssetInsurance(Model, AuditMixin, BaseMixin):
	"""
	Asset insurance tracking.
	
	Tracks insurance policies, coverage amounts, and renewal dates
	for asset protection and risk management.
	"""
	__tablename__ = 'cf_fam_asset_insurance'
	
	# Identity
	insurance_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Asset Reference
	asset_id = Column(String(36), ForeignKey('cf_fam_asset.asset_id'), nullable=False, index=True)
	
	# Policy Information
	policy_number = Column(String(50), nullable=False, index=True)
	insurance_company = Column(String(200), nullable=False)
	policy_type = Column(String(50), nullable=False)  # Property, Liability, Comprehensive, etc.
	coverage_type = Column(String(50), nullable=True)  # Replacement Cost, Actual Cash Value, etc.
	
	# Coverage Details
	coverage_amount = Column(DECIMAL(15, 2), nullable=False)
	deductible_amount = Column(DECIMAL(15, 2), default=0.00)
	premium_amount = Column(DECIMAL(15, 2), nullable=False)
	
	# Policy Dates
	policy_start_date = Column(Date, nullable=False, index=True)
	policy_end_date = Column(Date, nullable=False, index=True)
	last_appraisal_date = Column(Date, nullable=True)
	next_appraisal_date = Column(Date, nullable=True)
	
	# Status
	is_active = Column(Boolean, default=True)
	auto_renew = Column(Boolean, default=False)
	
	# Contact Information
	agent_name = Column(String(100), nullable=True)
	agent_phone = Column(String(50), nullable=True)
	agent_email = Column(String(100), nullable=True)
	
	# Coverage Details
	perils_covered = Column(Text, nullable=True)  # JSON array of covered perils
	exclusions = Column(Text, nullable=True)  # JSON array of exclusions
	special_conditions = Column(Text, nullable=True)
	
	# Claims History
	claims_count = Column(Integer, default=0)
	last_claim_date = Column(Date, nullable=True)
	total_claims_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Documentation
	policy_documents = Column(Text, nullable=True)  # JSON array of document URLs
	certificates = Column(Text, nullable=True)  # JSON array of certificate URLs
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'policy_number', name='uq_insurance_policy_tenant'),
	)
	
	# Relationships
	asset = relationship("CFAMAsset", back_populates="insurance_records")
	
	def __repr__(self):
		return f"<CFAMAssetInsurance {self.policy_number}>"
	
	def is_renewal_due(self, days_ahead: int = 30) -> bool:
		"""Check if renewal is due within specified days"""
		if self.policy_end_date:
			days_until_expiry = (self.policy_end_date - date.today()).days
			return days_until_expiry <= days_ahead
		return False
	
	def get_coverage_ratio(self) -> Decimal:
		"""Get coverage ratio vs asset current value"""
		if self.asset and self.asset.current_book_value > 0:
			return self.coverage_amount / self.asset.current_book_value
		return Decimal('0.00')


class CFAMAssetValuation(Model, AuditMixin, BaseMixin):
	"""
	Asset valuation and revaluation records.
	
	Tracks asset revaluations, impairment testing, and fair value
	assessments for financial reporting and compliance.
	"""
	__tablename__ = 'cf_fam_asset_valuation'
	
	# Identity
	valuation_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Asset Reference
	asset_id = Column(String(36), ForeignKey('cf_fam_asset.asset_id'), nullable=False, index=True)
	
	# Valuation Information
	valuation_date = Column(Date, nullable=False, index=True)
	valuation_type = Column(String(20), nullable=False)  # Revaluation, Impairment, Fair Value, etc.
	valuation_method = Column(String(50), nullable=False)  # Market, Cost, Income approach
	valuation_purpose = Column(String(50), nullable=True)  # Financial Reporting, Insurance, Sale, etc.
	
	# Valuation Results
	appraised_value = Column(DECIMAL(15, 2), nullable=False)
	book_value_at_valuation = Column(DECIMAL(15, 2), nullable=False)
	revaluation_surplus_deficit = Column(DECIMAL(15, 2), default=0.00)
	impairment_loss = Column(DECIMAL(15, 2), default=0.00)
	
	# Valuation Details
	valuation_basis = Column(String(100), nullable=True)  # Basis used for valuation
	market_conditions = Column(Text, nullable=True)
	assumptions = Column(Text, nullable=True)  # Key assumptions used
	limitations = Column(Text, nullable=True)  # Limitations of valuation
	
	# Appraiser Information
	appraiser_name = Column(String(200), nullable=True)
	appraiser_firm = Column(String(200), nullable=True)
	appraiser_license = Column(String(50), nullable=True)
	appraiser_contact = Column(String(200), nullable=True)
	
	# Effective Dates
	effective_date = Column(Date, nullable=True)  # When valuation becomes effective
	next_valuation_date = Column(Date, nullable=True)  # Next scheduled valuation
	
	# Approval and Posting
	requires_approval = Column(Boolean, default=True)
	approved = Column(Boolean, default=False)
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	
	is_posted = Column(Boolean, default=False)
	posted_date = Column(DateTime, nullable=True)
	journal_entry_id = Column(String(36), nullable=True)  # FK to GL Journal Entry
	
	# Documentation
	valuation_report = Column(String(500), nullable=True)  # URL to valuation report
	supporting_documents = Column(Text, nullable=True)  # JSON array of document URLs
	photos = Column(Text, nullable=True)  # JSON array of photo URLs
	
	# Compliance
	ifrs_compliant = Column(Boolean, default=True)
	gaap_compliant = Column(Boolean, default=True)
	tax_compliant = Column(Boolean, default=True)
	
	# Notes
	description = Column(Text, nullable=True)
	notes = Column(Text, nullable=True)
	
	# Relationships
	asset = relationship("CFAMAsset", back_populates="valuations")
	
	def __repr__(self):
		return f"<CFAMAssetValuation {self.asset.asset_number} - {self.valuation_date}>"
	
	def calculate_revaluation_impact(self):
		"""Calculate revaluation surplus/deficit and impairment"""
		self.revaluation_surplus_deficit = self.appraised_value - self.book_value_at_valuation
		
		if self.revaluation_surplus_deficit < 0:
			self.impairment_loss = abs(self.revaluation_surplus_deficit)
		else:
			self.impairment_loss = Decimal('0.00')


class CFAMAssetLease(Model, AuditMixin, BaseMixin):
	"""
	Asset lease tracking for ASC 842 / IFRS 16 compliance.
	
	Tracks lease agreements, payments, and accounting entries
	for right-of-use assets and lease liabilities.
	"""
	__tablename__ = 'cf_fam_asset_lease'
	
	# Identity
	lease_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Lease Information
	lease_number = Column(String(50), nullable=False, index=True)
	lease_name = Column(String(200), nullable=False)
	lease_type = Column(String(20), nullable=False)  # Finance, Operating
	lease_classification = Column(String(20), nullable=True)  # ASC 842 classification
	
	# Lessor Information
	lessor_name = Column(String(200), nullable=False)
	lessor_contact = Column(String(200), nullable=True)
	lessor_address = Column(Text, nullable=True)
	
	# Lease Terms
	lease_start_date = Column(Date, nullable=False, index=True)
	lease_end_date = Column(Date, nullable=False, index=True)
	lease_term_months = Column(Integer, nullable=False)
	
	# Renewal and Purchase Options
	renewal_options = Column(Integer, default=0)  # Number of renewal periods
	renewal_period_months = Column(Integer, nullable=True)
	purchase_option = Column(Boolean, default=False)
	purchase_option_price = Column(DECIMAL(15, 2), nullable=True)
	guaranteed_residual_value = Column(DECIMAL(15, 2), default=0.00)
	
	# Payment Information
	base_monthly_payment = Column(DECIMAL(15, 2), nullable=False)
	escalation_rate = Column(DECIMAL(5, 2), default=0.00)  # Annual escalation %
	payment_frequency = Column(String(20), default='Monthly')  # Monthly, Quarterly, etc.
	payment_timing = Column(String(20), default='Advance')  # Advance, Arrears
	
	# Initial Measurement
	initial_lease_liability = Column(DECIMAL(15, 2), default=0.00)
	initial_rou_asset = Column(DECIMAL(15, 2), default=0.00)
	initial_direct_costs = Column(DECIMAL(15, 2), default=0.00)
	prepaid_lease_payments = Column(DECIMAL(15, 2), default=0.00)
	lease_incentives_received = Column(DECIMAL(15, 2), default=0.00)
	
	# Current Balances
	current_lease_liability = Column(DECIMAL(15, 2), default=0.00)
	current_rou_asset = Column(DECIMAL(15, 2), default=0.00)
	accumulated_amortization = Column(DECIMAL(15, 2), default=0.00)
	
	# Discount Rate
	incremental_borrowing_rate = Column(DECIMAL(5, 4), nullable=True)  # IBR used
	lessor_implicit_rate = Column(DECIMAL(5, 4), nullable=True)  # If known
	discount_rate_used = Column(DECIMAL(5, 4), nullable=False)  # Rate actually used
	
	# Variable Payments
	has_variable_payments = Column(Boolean, default=False)
	variable_payment_basis = Column(String(100), nullable=True)  # Index, usage, performance
	
	# GL Account Mappings
	gl_rou_asset_account_id = Column(String(36), nullable=True)
	gl_lease_liability_account_id = Column(String(36), nullable=True)
	gl_lease_expense_account_id = Column(String(36), nullable=True)
	gl_amortization_expense_account_id = Column(String(36), nullable=True)
	gl_interest_expense_account_id = Column(String(36), nullable=True)
	
	# Status
	is_active = Column(Boolean, default=True)
	early_termination_date = Column(Date, nullable=True)
	termination_penalty = Column(DECIMAL(15, 2), nullable=True)
	
	# Asset Details
	leased_asset_description = Column(Text, nullable=True)
	leased_asset_location = Column(String(200), nullable=True)
	leased_asset_condition = Column(String(20), nullable=True)
	
	# Documentation
	lease_agreement = Column(String(500), nullable=True)  # URL to lease document
	amendments = Column(Text, nullable=True)  # JSON array of amendments
	related_documents = Column(Text, nullable=True)  # JSON array of documents
	
	# Notes
	notes = Column(Text, nullable=True)
	accounting_notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'lease_number', name='uq_lease_number_tenant'),
	)
	
	# Relationships
	assets = relationship("CFAMAsset", back_populates="lease")
	
	def __repr__(self):
		return f"<CFAMAssetLease {self.lease_number} - {self.lease_name}>"
	
	def calculate_lease_term_months(self):
		"""Calculate lease term in months"""
		if self.lease_start_date and self.lease_end_date:
			months = (self.lease_end_date.year - self.lease_start_date.year) * 12
			months += self.lease_end_date.month - self.lease_start_date.month
			self.lease_term_months = months
	
	def is_finance_lease(self) -> bool:
		"""Determine if lease should be classified as finance lease"""
		# ASC 842 criteria for finance lease classification
		lease_term_years = self.lease_term_months / 12 if self.lease_term_months else 0
		
		# Title transfer test
		if self.purchase_option and self.purchase_option_price and self.purchase_option_price <= 1:
			return True
		
		# Economic life test (75% rule removed in ASC 842, but commonly used)
		# Present value test (90% rule removed in ASC 842, but commonly used)
		# Specialized asset test
		
		return self.lease_type == 'Finance'
	
	def get_monthly_payment(self, payment_date: date = None) -> Decimal:
		"""Get monthly payment amount with escalations"""
		if not payment_date:
			payment_date = date.today()
		
		# Calculate years since lease start
		years_elapsed = (payment_date.year - self.lease_start_date.year)
		if payment_date.month < self.lease_start_date.month:
			years_elapsed -= 1
		
		# Apply escalation
		escalated_payment = self.base_monthly_payment
		if self.escalation_rate > 0 and years_elapsed > 0:
			escalation_factor = (1 + (self.escalation_rate / 100)) ** years_elapsed
			escalated_payment = self.base_monthly_payment * escalation_factor
		
		return escalated_payment


class CFAMDepreciation(Model, AuditMixin, BaseMixin):
	"""
	Depreciation calculation and history records.
	
	Stores calculated depreciation amounts by period for each asset
	including book and tax depreciation tracking.
	"""
	__tablename__ = 'cf_fam_depreciation'
	
	# Identity
	depreciation_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Asset and Method References
	asset_id = Column(String(36), ForeignKey('cf_fam_asset.asset_id'), nullable=False, index=True)
	method_id = Column(String(36), ForeignKey('cf_fam_depreciation_method.method_id'), nullable=False, index=True)
	
	# Period Information
	depreciation_date = Column(Date, nullable=False, index=True)
	fiscal_year = Column(Integer, nullable=False, index=True)
	fiscal_period = Column(Integer, nullable=False, index=True)  # 1-12 for months
	period_name = Column(String(20), nullable=False)  # "2024-01" for Jan 2024
	
	# Depreciation Calculation
	beginning_book_value = Column(DECIMAL(15, 2), nullable=False)
	depreciation_amount = Column(DECIMAL(15, 2), nullable=False)
	accumulated_depreciation_before = Column(DECIMAL(15, 2), nullable=False)
	accumulated_depreciation_after = Column(DECIMAL(15, 2), nullable=False)
	ending_book_value = Column(DECIMAL(15, 2), nullable=False)
	
	# Tax Depreciation (if different)
	tax_depreciation_amount = Column(DECIMAL(15, 2), nullable=True)
	tax_accumulated_depreciation_before = Column(DECIMAL(15, 2), nullable=True)
	tax_accumulated_depreciation_after = Column(DECIMAL(15, 2), nullable=True)
	tax_book_value = Column(DECIMAL(15, 2), nullable=True)
	
	# Calculation Details
	depreciable_base = Column(DECIMAL(15, 2), nullable=False)
	salvage_value = Column(DECIMAL(15, 2), default=0.00)
	useful_life_remaining_months = Column(Integer, nullable=True)
	depreciation_rate = Column(DECIMAL(8, 6), nullable=True)  # Rate used for calculation
	
	# Units of Production (if applicable)
	units_produced = Column(Integer, nullable=True)
	cumulative_units = Column(Integer, nullable=True)
	total_estimated_units = Column(Integer, nullable=True)
	
	# Status and Control
	is_calculated = Column(Boolean, default=True)
	is_posted = Column(Boolean, default=False)
	is_adjustment = Column(Boolean, default=False)  # True for catch-up or adjustment entries
	adjustment_reason = Column(String(200), nullable=True)
	
	# GL Posting Information
	posted_date = Column(DateTime, nullable=True)
	journal_entry_id = Column(String(36), nullable=True)  # FK to GL Journal Entry
	posted_by = Column(String(36), nullable=True)
	
	# Calculation Metadata
	calculation_method = Column(String(50), nullable=True)  # For audit trail
	calculation_date = Column(DateTime, default=datetime.utcnow)
	calculated_by = Column(String(36), nullable=True)
	
	# Convention Applied
	convention_applied = Column(String(20), nullable=True)  # half_year, mid_month, etc.
	proration_factor = Column(DECIMAL(8, 6), default=1.000000)  # For partial periods
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'asset_id', 'depreciation_date', name='uq_depreciation_asset_date'),
	)
	
	# Relationships
	asset = relationship("CFAMAsset", back_populates="depreciation_records")
	method = relationship("CFAMDepreciationMethod", back_populates="depreciation_records")
	
	def __repr__(self):
		return f"<CFAMDepreciation {self.asset.asset_number} - {self.period_name}>"
	
	def validate_calculation(self) -> bool:
		"""Validate depreciation calculation"""
		# Check that ending book value = beginning book value - depreciation
		expected_ending = self.beginning_book_value - self.depreciation_amount
		return abs(self.ending_book_value - expected_ending) < 0.01
	
	def create_gl_entries(self) -> List[Dict[str, Any]]:
		"""Create GL journal entries for depreciation"""
		if self.depreciation_amount <= 0:
			return []
		
		entries = []
		
		# Depreciation Expense (Debit)
		entries.append({
			'account_id': self.asset.gl_expense_account_id or self.asset.category.gl_expense_account_id,
			'debit_amount': self.depreciation_amount,
			'credit_amount': 0.00,
			'description': f'Depreciation expense - {self.asset.asset_name}',
			'reference': f'Asset {self.asset.asset_number}'
		})
		
		# Accumulated Depreciation (Credit)
		entries.append({
			'account_id': self.asset.gl_depreciation_account_id or self.asset.category.gl_depreciation_account_id,
			'debit_amount': 0.00,
			'credit_amount': self.depreciation_amount,
			'description': f'Accumulated depreciation - {self.asset.asset_name}',
			'reference': f'Asset {self.asset.asset_number}'
		})
		
		return entries