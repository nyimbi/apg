"""
Cost Accounting Models

Database models for the Cost Accounting sub-capability including cost centers,
cost categories, activity-based costing, job costing, and variance analysis.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ...auth_rbac.models import BaseMixin, AuditMixin, Model


class CFCACostCenter(Model, AuditMixin, BaseMixin):
	"""
	Cost Centers for organizing and tracking costs by responsibility areas.
	
	Represents organizational units responsible for managing costs
	such as departments, divisions, or production lines.
	"""
	__tablename__ = 'cf_ca_cost_center'
	
	# Identity
	cost_center_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Cost Center Information
	center_code = Column(String(20), nullable=False, index=True)
	center_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	
	# Hierarchy
	parent_center_id = Column(String(36), ForeignKey('cf_ca_cost_center.cost_center_id'), nullable=True, index=True)
	level = Column(Integer, default=0)
	path = Column(String(500), nullable=True)
	
	# Classification
	center_type = Column(String(50), nullable=False, index=True)  # Production, Service, Administrative, etc.
	responsibility_type = Column(String(50), nullable=False)  # Cost, Profit, Investment
	
	# Manager & Responsibility
	manager_name = Column(String(200), nullable=True)
	manager_email = Column(String(200), nullable=True)
	department = Column(String(100), nullable=True)
	location = Column(String(200), nullable=True)
	
	# Budget & Control
	annual_budget = Column(DECIMAL(15, 2), default=0.00)
	ytd_actual = Column(DECIMAL(15, 2), default=0.00)
	ytd_budget = Column(DECIMAL(15, 2), default=0.00)
	
	# Status
	is_active = Column(Boolean, default=True)
	effective_date = Column(Date, nullable=False)
	end_date = Column(Date, nullable=True)
	
	# Configuration
	allow_cost_allocation = Column(Boolean, default=True)
	require_job_number = Column(Boolean, default=False)
	default_currency = Column(String(3), default='USD')
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'center_code', name='uq_cost_center_code_tenant'),
		Index('ix_cost_center_type_active', 'center_type', 'is_active'),
	)
	
	# Relationships
	parent_center = relationship("CFCACostCenter", remote_side=[cost_center_id])
	child_centers = relationship("CFCACostCenter")
	cost_allocations = relationship("CFCACostAllocation", foreign_keys="CFCACostAllocation.source_center_id")
	job_costs = relationship("CFCAJobCost", back_populates="cost_center")
	product_costs = relationship("CFCAProductCost", back_populates="cost_center")
	
	def __repr__(self):
		return f"<CFCACostCenter {self.center_code} - {self.center_name}>"
	
	def get_full_path(self) -> str:
		"""Get full cost center path"""
		if self.parent_center:
			return f"{self.parent_center.get_full_path()} > {self.center_name}"
		return self.center_name
	
	def calculate_budget_variance(self) -> Dict[str, Decimal]:
		"""Calculate budget variance"""
		variance = self.ytd_actual - self.ytd_budget
		variance_percent = (variance / self.ytd_budget * 100) if self.ytd_budget else Decimal('0')
		
		return {
			'variance_amount': variance,
			'variance_percent': variance_percent,
			'is_favorable': variance < 0,  # Under budget is favorable for costs
			'is_significant': abs(variance_percent) > Decimal('5')
		}


class CFCACostCategory(Model, AuditMixin, BaseMixin):
	"""
	Cost Categories for classifying different types of costs.
	
	Hierarchical structure for organizing costs by type, nature,
	and behavior (fixed/variable, direct/indirect).
	"""
	__tablename__ = 'cf_ca_cost_category'
	
	# Identity
	category_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Category Information
	category_code = Column(String(20), nullable=False, index=True)
	category_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	
	# Hierarchy
	parent_category_id = Column(String(36), ForeignKey('cf_ca_cost_category.category_id'), nullable=True, index=True)
	level = Column(Integer, default=0)
	path = Column(String(500), nullable=True)
	
	# Cost Classification
	cost_type = Column(String(50), nullable=False, index=True)  # Direct, Indirect, Period
	cost_behavior = Column(String(50), nullable=False)  # Fixed, Variable, Mixed
	cost_nature = Column(String(50), nullable=False)  # Material, Labor, Overhead, Administrative
	
	# Cost Behavior Parameters
	is_variable = Column(Boolean, default=True)
	is_traceable = Column(Boolean, default=True)  # Can be traced to cost objects
	is_controllable = Column(Boolean, default=True)  # Can be controlled by management
	
	# GL Integration
	gl_account_code = Column(String(20), nullable=True, index=True)
	gl_account_id = Column(String(36), nullable=True, index=True)
	
	# Status
	is_active = Column(Boolean, default=True)
	effective_date = Column(Date, nullable=False)
	end_date = Column(Date, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'category_code', name='uq_cost_category_code_tenant'),
		Index('ix_cost_category_type_behavior', 'cost_type', 'cost_behavior'),
	)
	
	# Relationships
	parent_category = relationship("CFCACostCategory", remote_side=[category_id])
	child_categories = relationship("CFCACostCategory")
	activity_costs = relationship("CFCAActivityCost", back_populates="cost_category")
	product_costs = relationship("CFCAProductCost", back_populates="cost_category")
	job_costs = relationship("CFCAJobCost", back_populates="cost_category")
	standard_costs = relationship("CFCAStandardCost", back_populates="cost_category")
	
	def __repr__(self):
		return f"<CFCACostCategory {self.category_code} - {self.category_name}>"
	
	def get_full_path(self) -> str:
		"""Get full category path"""
		if self.parent_category:
			return f"{self.parent_category.get_full_path()} > {self.category_name}"
		return self.category_name
	
	def is_direct_cost(self) -> bool:
		"""Check if this is a direct cost category"""
		return self.cost_type == 'Direct'
	
	def is_overhead_cost(self) -> bool:
		"""Check if this is an overhead cost category"""
		return self.cost_type == 'Indirect' and self.cost_nature == 'Overhead'


class CFCACostDriver(Model, AuditMixin, BaseMixin):
	"""
	Cost Drivers for activity-based costing and cost allocation.
	
	Defines measurable factors that cause costs to be incurred,
	used for allocating costs to products, services, or cost objects.
	"""
	__tablename__ = 'cf_ca_cost_driver'
	
	# Identity
	driver_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Driver Information
	driver_code = Column(String(20), nullable=False, index=True)
	driver_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	
	# Measurement
	unit_of_measure = Column(String(50), nullable=False)  # Hours, Units, Setups, etc.
	driver_type = Column(String(50), nullable=False, index=True)  # Volume, Activity, Transaction, Facility
	
	# Behavior
	is_volume_based = Column(Boolean, default=True)
	is_activity_based = Column(Boolean, default=False)
	requires_measurement = Column(Boolean, default=True)
	
	# Calculation
	calculation_method = Column(String(100), nullable=True)  # Formula or method description
	calculation_frequency = Column(String(50), default='Monthly')  # Daily, Weekly, Monthly, etc.
	
	# Default Values
	default_rate = Column(DECIMAL(12, 4), nullable=True)  # Default cost per unit
	current_capacity = Column(DECIMAL(15, 2), nullable=True)  # Current capacity in units
	practical_capacity = Column(DECIMAL(15, 2), nullable=True)  # Practical capacity
	
	# Status
	is_active = Column(Boolean, default=True)
	effective_date = Column(Date, nullable=False)
	end_date = Column(Date, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'driver_code', name='uq_cost_driver_code_tenant'),
		Index('ix_cost_driver_type_active', 'driver_type', 'is_active'),
	)
	
	# Relationships
	cost_allocations = relationship("CFCACostAllocation", back_populates="cost_driver")
	activities = relationship("CFCAActivity", back_populates="primary_cost_driver")
	activity_costs = relationship("CFCAActivityCost", back_populates="cost_driver")
	
	def __repr__(self):
		return f"<CFCACostDriver {self.driver_code} - {self.driver_name}>"
	
	def calculate_rate(self, total_cost: Decimal, total_activity: Decimal) -> Decimal:
		"""Calculate cost rate per unit of driver"""
		if total_activity and total_activity > 0:
			return total_cost / total_activity
		return Decimal('0')
	
	def get_capacity_utilization(self, actual_activity: Decimal) -> Dict[str, Any]:
		"""Calculate capacity utilization metrics"""
		if not self.practical_capacity or self.practical_capacity <= 0:
			return {'utilization_percent': None, 'unused_capacity': None}
		
		utilization = (actual_activity / self.practical_capacity) * 100
		unused_capacity = self.practical_capacity - actual_activity
		
		return {
			'utilization_percent': utilization,
			'unused_capacity': unused_capacity,
			'is_over_capacity': actual_activity > self.practical_capacity,
			'efficiency_rating': 'High' if utilization >= 85 else 'Medium' if utilization >= 70 else 'Low'
		}


class CFCACostAllocation(Model, AuditMixin, BaseMixin):
	"""
	Cost Allocation Rules for distributing costs across cost centers,
	products, or other cost objects.
	"""
	__tablename__ = 'cf_ca_cost_allocation'
	
	# Identity
	allocation_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Allocation Information
	allocation_code = Column(String(20), nullable=False, index=True)
	allocation_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Source and Target
	source_center_id = Column(String(36), ForeignKey('cf_ca_cost_center.cost_center_id'), nullable=False, index=True)
	target_center_id = Column(String(36), ForeignKey('cf_ca_cost_center.cost_center_id'), nullable=True, index=True)
	cost_category_id = Column(String(36), ForeignKey('cf_ca_cost_category.category_id'), nullable=True, index=True)
	
	# Allocation Method
	allocation_method = Column(String(100), nullable=False, index=True)  # Direct, Step-down, Reciprocal, ABC
	cost_driver_id = Column(String(36), ForeignKey('cf_ca_cost_driver.driver_id'), nullable=True, index=True)
	allocation_basis = Column(String(100), nullable=True)  # Specific basis description
	
	# Allocation Parameters
	allocation_percent = Column(DECIMAL(5, 2), nullable=True)  # Fixed percentage
	allocation_formula = Column(Text, nullable=True)  # Complex formula
	
	# Period and Frequency
	allocation_frequency = Column(String(50), default='Monthly')  # Daily, Weekly, Monthly, Quarterly
	effective_date = Column(Date, nullable=False)
	end_date = Column(Date, nullable=True)
	
	# Status and Control
	is_active = Column(Boolean, default=True)
	is_automatic = Column(Boolean, default=False)  # Automatic vs manual allocation
	requires_approval = Column(Boolean, default=True)
	last_allocation_date = Column(Date, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'allocation_code', name='uq_cost_allocation_code_tenant'),
		Index('ix_cost_allocation_source_target', 'source_center_id', 'target_center_id'),
	)
	
	# Relationships
	source_center = relationship("CFCACostCenter", foreign_keys=[source_center_id])
	target_center = relationship("CFCACostCenter", foreign_keys=[target_center_id])
	cost_category = relationship("CFCACostCategory")
	cost_driver = relationship("CFCACostDriver", back_populates="cost_allocations")
	
	def __repr__(self):
		return f"<CFCACostAllocation {self.allocation_code} - {self.allocation_name}>"
	
	def calculate_allocation_amount(self, total_cost: Decimal, driver_quantity: Decimal, total_driver_quantity: Decimal) -> Decimal:
		"""Calculate allocated cost amount"""
		if self.allocation_method == 'Percentage' and self.allocation_percent:
			return total_cost * (self.allocation_percent / 100)
		elif self.allocation_method == 'Driver_Based' and total_driver_quantity > 0:
			return total_cost * (driver_quantity / total_driver_quantity)
		elif self.allocation_method == 'Equal':
			# Equal distribution - would need to know number of targets
			return total_cost  # Simplified
		else:
			return Decimal('0')
	
	def is_reciprocal_allocation(self) -> bool:
		"""Check if this is a reciprocal allocation method"""
		return self.allocation_method in ['Reciprocal', 'Simultaneous']


class CFCACostPool(Model, AuditMixin, BaseMixin):
	"""
	Cost Pools for activity-based costing.
	
	Groups costs that will be allocated to cost objects using
	the same cost driver or allocation method.
	"""
	__tablename__ = 'cf_ca_cost_pool'
	
	# Identity
	pool_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Pool Information
	pool_code = Column(String(20), nullable=False, index=True)
	pool_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	
	# Classification
	pool_type = Column(String(50), nullable=False, index=True)  # Production, Setup, Quality, Administrative, etc.
	cost_behavior = Column(String(50), nullable=False)  # Fixed, Variable, Mixed
	
	# Assignment
	cost_center_id = Column(String(36), ForeignKey('cf_ca_cost_center.cost_center_id'), nullable=True, index=True)
	primary_driver_id = Column(String(36), ForeignKey('cf_ca_cost_driver.driver_id'), nullable=True, index=True)
	
	# Pool Totals
	budgeted_cost = Column(DECIMAL(15, 2), default=0.00)
	actual_cost = Column(DECIMAL(15, 2), default=0.00)
	allocated_cost = Column(DECIMAL(15, 2), default=0.00)
	
	# Activity Measures
	budgeted_activity = Column(DECIMAL(15, 2), default=0.00)
	actual_activity = Column(DECIMAL(15, 2), default=0.00)
	
	# Rates
	budgeted_rate = Column(DECIMAL(12, 4), default=0.00)  # Budgeted cost per unit
	actual_rate = Column(DECIMAL(12, 4), default=0.00)  # Actual cost per unit
	
	# Status
	is_active = Column(Boolean, default=True)
	effective_date = Column(Date, nullable=False)
	end_date = Column(Date, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'pool_code', name='uq_cost_pool_code_tenant'),
		Index('ix_cost_pool_type_active', 'pool_type', 'is_active'),
	)
	
	# Relationships
	cost_center = relationship("CFCACostCenter")
	primary_driver = relationship("CFCACostDriver")
	activities = relationship("CFCAActivity", back_populates="cost_pool")
	activity_costs = relationship("CFCAActivityCost", back_populates="cost_pool")
	
	def __repr__(self):
		return f"<CFCACostPool {self.pool_code} - {self.pool_name}>"
	
	def calculate_rates(self) -> Dict[str, Decimal]:
		"""Calculate budgeted and actual rates"""
		budgeted_rate = self.budgeted_cost / self.budgeted_activity if self.budgeted_activity > 0 else Decimal('0')
		actual_rate = self.actual_cost / self.actual_activity if self.actual_activity > 0 else Decimal('0')
		
		return {
			'budgeted_rate': budgeted_rate,
			'actual_rate': actual_rate,
			'rate_variance': actual_rate - budgeted_rate,
			'cost_variance': self.actual_cost - self.budgeted_cost,
			'activity_variance': self.actual_activity - self.budgeted_activity
		}
	
	def get_utilization_metrics(self) -> Dict[str, Any]:
		"""Get cost pool utilization metrics"""
		if self.budgeted_activity <= 0:
			return {'utilization': None, 'efficiency': None}
		
		utilization = (self.actual_activity / self.budgeted_activity) * 100
		efficiency = (self.budgeted_cost / self.actual_cost) * 100 if self.actual_cost > 0 else 100
		
		return {
			'activity_utilization': utilization,
			'cost_efficiency': efficiency,
			'under_utilized': utilization < 80,
			'over_budget': self.actual_cost > self.budgeted_cost
		}


class CFCAActivity(Model, AuditMixin, BaseMixin):
	"""
	Activities for Activity-Based Costing (ABC).
	
	Defines work processes that consume resources and
	are performed to produce products or services.
	"""
	__tablename__ = 'cf_ca_activity'
	
	# Identity
	activity_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Activity Information
	activity_code = Column(String(20), nullable=False, index=True)
	activity_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	
	# Classification
	activity_type = Column(String(50), nullable=False, index=True)  # Primary, Support, Sustaining
	value_category = Column(String(50), nullable=False)  # Value-Added, Non-Value-Added, Business-Value-Added
	
	# Assignment
	cost_pool_id = Column(String(36), ForeignKey('cf_ca_cost_pool.pool_id'), nullable=True, index=True)
	cost_center_id = Column(String(36), ForeignKey('cf_ca_cost_center.cost_center_id'), nullable=True, index=True)
	primary_driver_id = Column(String(36), ForeignKey('cf_ca_cost_driver.driver_id'), nullable=True, index=True)
	
	# Activity Measures
	capacity_measure = Column(String(100), nullable=True)  # How capacity is measured
	practical_capacity = Column(DECIMAL(15, 2), nullable=True)
	current_capacity = Column(DECIMAL(15, 2), nullable=True)
	
	# Resources and Cost
	resource_requirements = Column(Text, nullable=True)  # JSON describing resource needs
	estimated_cost_per_unit = Column(DECIMAL(12, 4), nullable=True)
	setup_time_minutes = Column(Integer, nullable=True)
	processing_time_minutes = Column(Integer, nullable=True)
	
	# Performance
	quality_rating = Column(DECIMAL(3, 2), nullable=True)  # 0.00 to 1.00
	efficiency_rating = Column(DECIMAL(3, 2), nullable=True)  # 0.00 to 1.00
	cycle_time_minutes = Column(Integer, nullable=True)
	
	# Status
	is_active = Column(Boolean, default=True)
	is_value_added = Column(Boolean, default=True)
	effective_date = Column(Date, nullable=False)
	end_date = Column(Date, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'activity_code', name='uq_activity_code_tenant'),
		Index('ix_activity_type_value', 'activity_type', 'value_category'),
	)
	
	# Relationships
	cost_pool = relationship("CFCACostPool", back_populates="activities")
	cost_center = relationship("CFCACostCenter")
	primary_cost_driver = relationship("CFCACostDriver", back_populates="activities")
	activity_costs = relationship("CFCAActivityCost", back_populates="activity")
	
	def __repr__(self):
		return f"<CFCAActivity {self.activity_code} - {self.activity_name}>"
	
	def is_bottleneck_activity(self, demand: Decimal) -> bool:
		"""Check if activity is a bottleneck based on demand vs capacity"""
		if not self.practical_capacity:
			return False
		return demand > self.practical_capacity
	
	def calculate_activity_rate(self, total_cost: Decimal, total_activity: Decimal) -> Decimal:
		"""Calculate cost rate per activity unit"""
		if total_activity and total_activity > 0:
			return total_cost / total_activity
		return self.estimated_cost_per_unit or Decimal('0')
	
	def get_resource_requirements(self) -> Dict[str, Any]:
		"""Parse resource requirements from JSON"""
		if self.resource_requirements:
			try:
				return json.loads(self.resource_requirements)
			except (json.JSONDecodeError, TypeError):
				return {}
		return {}


class CFCAActivityCost(Model, AuditMixin, BaseMixin):
	"""
	Activity Costs for tracking costs consumed by activities.
	
	Links cost categories to activities and tracks the consumption
	of resources by different activities.
	"""
	__tablename__ = 'cf_ca_activity_cost'
	
	# Identity
	activity_cost_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Assignment
	activity_id = Column(String(36), ForeignKey('cf_ca_activity.activity_id'), nullable=False, index=True)
	cost_category_id = Column(String(36), ForeignKey('cf_ca_cost_category.category_id'), nullable=False, index=True)
	cost_pool_id = Column(String(36), ForeignKey('cf_ca_cost_pool.pool_id'), nullable=True, index=True)
	cost_driver_id = Column(String(36), ForeignKey('cf_ca_cost_driver.driver_id'), nullable=True, index=True)
	
	# Period
	cost_period = Column(String(7), nullable=False, index=True)  # YYYY-MM format
	fiscal_year = Column(Integer, nullable=False, index=True)
	fiscal_period = Column(Integer, nullable=False, index=True)
	
	# Cost Amounts
	budgeted_cost = Column(DECIMAL(15, 2), default=0.00)
	actual_cost = Column(DECIMAL(15, 2), default=0.00)
	allocated_cost = Column(DECIMAL(15, 2), default=0.00)
	
	# Activity Quantities
	budgeted_quantity = Column(DECIMAL(15, 2), default=0.00)
	actual_quantity = Column(DECIMAL(15, 2), default=0.00)
	
	# Rates
	budgeted_rate = Column(DECIMAL(12, 4), default=0.00)
	actual_rate = Column(DECIMAL(12, 4), default=0.00)
	
	# Allocation Details
	allocation_method = Column(String(100), nullable=True)
	allocation_basis = Column(String(200), nullable=True)
	driver_quantity = Column(DECIMAL(15, 2), nullable=True)
	driver_rate = Column(DECIMAL(12, 4), nullable=True)
	
	# Status
	is_posted = Column(Boolean, default=False)
	is_allocated = Column(Boolean, default=False)
	posting_date = Column(Date, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'activity_id', 'cost_category_id', 'cost_period', name='uq_activity_cost_period'),
		Index('ix_activity_cost_period_posted', 'cost_period', 'is_posted'),
	)
	
	# Relationships
	activity = relationship("CFCAActivity", back_populates="activity_costs")
	cost_category = relationship("CFCACostCategory", back_populates="activity_costs")
	cost_pool = relationship("CFCACostPool", back_populates="activity_costs")
	cost_driver = relationship("CFCACostDriver", back_populates="activity_costs")
	
	def __repr__(self):
		return f"<CFCAActivityCost {self.activity_id} - {self.cost_category_id} - {self.cost_period}>"
	
	def calculate_variances(self) -> Dict[str, Decimal]:
		"""Calculate cost and efficiency variances"""
		cost_variance = self.actual_cost - self.budgeted_cost
		quantity_variance = self.actual_quantity - self.budgeted_quantity
		rate_variance = self.actual_rate - self.budgeted_rate
		
		# Efficiency variance (quantity variance * budgeted rate)
		efficiency_variance = quantity_variance * self.budgeted_rate
		
		# Rate variance (rate variance * actual quantity)  
		rate_variance_amount = rate_variance * self.actual_quantity
		
		return {
			'total_cost_variance': cost_variance,
			'efficiency_variance': efficiency_variance,
			'rate_variance': rate_variance_amount,
			'quantity_variance': quantity_variance,
			'is_favorable': cost_variance < 0
		}
	
	def get_cost_per_unit(self) -> Decimal:
		"""Get actual cost per unit of activity"""
		if self.actual_quantity and self.actual_quantity > 0:
			return self.actual_cost / self.actual_quantity
		return Decimal('0')


class CFCAProductCost(Model, AuditMixin, BaseMixin):
	"""
	Product Costing for tracking costs by product or service.
	
	Accumulates direct and indirect costs for products using
	various costing methods (job order, process, ABC).
	"""
	__tablename__ = 'cf_ca_product_cost'
	
	# Identity
	product_cost_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Product Information
	product_code = Column(String(50), nullable=False, index=True)
	product_name = Column(String(200), nullable=False)
	product_category = Column(String(100), nullable=True, index=True)
	
	# Assignment
	cost_center_id = Column(String(36), ForeignKey('cf_ca_cost_center.cost_center_id'), nullable=True, index=True)
	cost_category_id = Column(String(36), ForeignKey('cf_ca_cost_category.category_id'), nullable=False, index=True)
	
	# Period
	cost_period = Column(String(7), nullable=False, index=True)  # YYYY-MM format
	fiscal_year = Column(Integer, nullable=False, index=True)
	fiscal_period = Column(Integer, nullable=False, index=True)
	
	# Costing Method
	costing_method = Column(String(50), nullable=False, index=True)  # Standard, Actual, Normal, ABC
	
	# Quantities
	production_quantity = Column(DECIMAL(15, 4), default=0.00)
	completed_quantity = Column(DECIMAL(15, 4), default=0.00)
	spoiled_quantity = Column(DECIMAL(15, 4), default=0.00)
	
	# Direct Costs
	direct_material_cost = Column(DECIMAL(15, 2), default=0.00)
	direct_labor_cost = Column(DECIMAL(15, 2), default=0.00)
	direct_expense_cost = Column(DECIMAL(15, 2), default=0.00)
	
	# Indirect Costs
	allocated_overhead = Column(DECIMAL(15, 2), default=0.00)
	allocated_admin = Column(DECIMAL(15, 2), default=0.00)
	allocated_selling = Column(DECIMAL(15, 2), default=0.00)
	
	# Total Costs
	total_cost = Column(DECIMAL(15, 2), default=0.00)
	unit_cost = Column(DECIMAL(12, 4), default=0.00)
	
	# Standard Cost Comparison (if applicable)
	standard_cost = Column(DECIMAL(12, 4), nullable=True)
	cost_variance = Column(DECIMAL(15, 2), nullable=True)
	
	# Work in Process
	beginning_wip = Column(DECIMAL(15, 2), default=0.00)
	ending_wip = Column(DECIMAL(15, 2), default=0.00)
	
	# Status
	is_completed = Column(Boolean, default=False)
	is_posted = Column(Boolean, default=False)
	completion_date = Column(Date, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'product_code', 'cost_category_id', 'cost_period', name='uq_product_cost_period'),
		Index('ix_product_cost_method_period', 'costing_method', 'cost_period'),
	)
	
	# Relationships
	cost_center = relationship("CFCACostCenter", back_populates="product_costs")
	cost_category = relationship("CFCACostCategory", back_populates="product_costs")
	
	def __repr__(self):
		return f"<CFCAProductCost {self.product_code} - {self.cost_period}>"
	
	def calculate_total_cost(self) -> Decimal:
		"""Calculate total product cost"""
		direct_costs = (self.direct_material_cost + self.direct_labor_cost + self.direct_expense_cost)
		indirect_costs = (self.allocated_overhead + self.allocated_admin + self.allocated_selling)
		return direct_costs + indirect_costs
	
	def calculate_unit_cost(self) -> Decimal:
		"""Calculate cost per unit"""
		total_cost = self.calculate_total_cost()
		if self.completed_quantity and self.completed_quantity > 0:
			return total_cost / self.completed_quantity
		elif self.production_quantity and self.production_quantity > 0:
			return total_cost / self.production_quantity
		return Decimal('0')
	
	def get_cost_breakdown(self) -> Dict[str, Any]:
		"""Get detailed cost breakdown"""
		total_cost = self.calculate_total_cost()
		unit_cost = self.calculate_unit_cost()
		
		if total_cost == 0:
			return {'error': 'No costs recorded'}
		
		return {
			'direct_costs': {
				'material': self.direct_material_cost,
				'labor': self.direct_labor_cost,
				'expense': self.direct_expense_cost,
				'total': self.direct_material_cost + self.direct_labor_cost + self.direct_expense_cost,
				'percentage': ((self.direct_material_cost + self.direct_labor_cost + self.direct_expense_cost) / total_cost) * 100
			},
			'indirect_costs': {
				'overhead': self.allocated_overhead,
				'admin': self.allocated_admin,
				'selling': self.allocated_selling,
				'total': self.allocated_overhead + self.allocated_admin + self.allocated_selling,
				'percentage': ((self.allocated_overhead + self.allocated_admin + self.allocated_selling) / total_cost) * 100
			},
			'totals': {
				'total_cost': total_cost,
				'unit_cost': unit_cost,
				'completed_units': self.completed_quantity,
				'cost_per_completed_unit': total_cost / self.completed_quantity if self.completed_quantity > 0 else 0
			}
		}


class CFCAJobCost(Model, AuditMixin, BaseMixin):
	"""
	Job Costing for project-based cost tracking.
	
	Tracks costs for specific jobs, projects, or custom orders
	with detailed cost accumulation by category.
	"""
	__tablename__ = 'cf_ca_job_cost'
	
	# Identity
	job_cost_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Job Information
	job_number = Column(String(50), nullable=False, index=True)
	job_name = Column(String(200), nullable=False)
	job_description = Column(Text, nullable=True)
	
	# Assignment
	cost_center_id = Column(String(36), ForeignKey('cf_ca_cost_center.cost_center_id'), nullable=False, index=True)
	cost_category_id = Column(String(36), ForeignKey('cf_ca_cost_category.category_id'), nullable=False, index=True)
	
	# Customer/Project
	customer_code = Column(String(50), nullable=True, index=True)
	customer_name = Column(String(200), nullable=True)
	project_code = Column(String(50), nullable=True, index=True)
	contract_number = Column(String(50), nullable=True)
	
	# Dates
	start_date = Column(Date, nullable=False)
	planned_completion_date = Column(Date, nullable=True)
	actual_completion_date = Column(Date, nullable=True)
	
	# Budget Information
	budgeted_cost = Column(DECIMAL(15, 2), default=0.00)
	budgeted_hours = Column(DECIMAL(10, 2), default=0.00)
	contract_value = Column(DECIMAL(15, 2), nullable=True)
	
	# Actual Costs
	actual_material_cost = Column(DECIMAL(15, 2), default=0.00)
	actual_labor_cost = Column(DECIMAL(15, 2), default=0.00)
	actual_overhead_cost = Column(DECIMAL(15, 2), default=0.00)
	actual_other_cost = Column(DECIMAL(15, 2), default=0.00)
	
	# Actual Hours/Quantities
	actual_labor_hours = Column(DECIMAL(10, 2), default=0.00)
	actual_machine_hours = Column(DECIMAL(10, 2), default=0.00)
	
	# Committed Costs (POs, commitments)
	committed_material_cost = Column(DECIMAL(15, 2), default=0.00)
	committed_labor_cost = Column(DECIMAL(15, 2), default=0.00)
	committed_other_cost = Column(DECIMAL(15, 2), default=0.00)
	
	# Billing Information
	billed_to_date = Column(DECIMAL(15, 2), default=0.00)
	percent_complete = Column(DECIMAL(5, 2), default=0.00)
	billing_method = Column(String(50), nullable=True)  # Fixed, T&M, Cost-Plus, etc.
	
	# Status
	job_status = Column(String(50), nullable=False, default='Active', index=True)  # Active, Hold, Complete, Cancelled
	is_billable = Column(Boolean, default=True)
	is_closed = Column(Boolean, default=False)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'job_number', 'cost_category_id', name='uq_job_cost_category'),
		Index('ix_job_cost_status_dates', 'job_status', 'start_date'),
	)
	
	# Relationships
	cost_center = relationship("CFCACostCenter", back_populates="job_costs")
	cost_category = relationship("CFCACostCategory", back_populates="job_costs")
	
	def __repr__(self):
		return f"<CFCAJobCost {self.job_number} - {self.cost_category.category_name}>"
	
	def calculate_total_actual_cost(self) -> Decimal:
		"""Calculate total actual cost"""
		return (self.actual_material_cost + self.actual_labor_cost + 
				self.actual_overhead_cost + self.actual_other_cost)
	
	def calculate_total_committed_cost(self) -> Decimal:
		"""Calculate total committed cost"""
		return (self.committed_material_cost + self.committed_labor_cost + self.committed_other_cost)
	
	def calculate_cost_to_complete(self) -> Decimal:
		"""Calculate estimated cost to complete"""
		if self.percent_complete >= 100:
			return Decimal('0')
		
		if self.percent_complete > 0:
			total_estimated_cost = self.calculate_total_actual_cost() / (self.percent_complete / 100)
			return total_estimated_cost - self.calculate_total_actual_cost()
		
		return self.budgeted_cost - self.calculate_total_actual_cost()
	
	def get_job_profitability(self) -> Dict[str, Any]:
		"""Calculate job profitability metrics"""
		total_cost = self.calculate_total_actual_cost()
		committed_cost = self.calculate_total_committed_cost()
		estimated_final_cost = total_cost + self.calculate_cost_to_complete()
		
		profit = (self.contract_value or 0) - estimated_final_cost
		profit_margin = (profit / self.contract_value * 100) if self.contract_value else 0
		
		return {
			'contract_value': self.contract_value or 0,
			'actual_cost_to_date': total_cost,
			'committed_costs': committed_cost,
			'estimated_cost_to_complete': self.calculate_cost_to_complete(),
			'estimated_final_cost': estimated_final_cost,
			'estimated_profit': profit,
			'profit_margin_percent': profit_margin,
			'cost_variance': (self.budgeted_cost - total_cost) if self.budgeted_cost else 0,
			'is_profitable': profit > 0,
			'is_over_budget': total_cost > self.budgeted_cost if self.budgeted_cost else False
		}
	
	def is_on_schedule(self) -> bool:
		"""Check if job is on schedule"""
		if not self.planned_completion_date:
			return True
		
		if self.actual_completion_date:
			return self.actual_completion_date <= self.planned_completion_date
		
		# Check current progress vs time elapsed
		from datetime import date
		today = date.today()
		if today > self.planned_completion_date and not self.is_closed:
			return False
		
		return True


class CFCAStandardCost(Model, AuditMixin, BaseMixin):
	"""
	Standard Costs for variance analysis and performance measurement.
	
	Maintains standard costs for products, services, and activities
	used for budgeting, pricing, and variance analysis.
	"""
	__tablename__ = 'cf_ca_standard_cost'
	
	# Identity
	standard_cost_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Cost Object
	cost_object_type = Column(String(50), nullable=False, index=True)  # Product, Service, Activity, Job
	cost_object_code = Column(String(50), nullable=False, index=True)
	cost_object_name = Column(String(200), nullable=False)
	
	# Cost Classification
	cost_category_id = Column(String(36), ForeignKey('cf_ca_cost_category.category_id'), nullable=False, index=True)
	cost_center_id = Column(String(36), ForeignKey('cf_ca_cost_center.cost_center_id'), nullable=True, index=True)
	
	# Standard Cost Information
	standard_cost_per_unit = Column(DECIMAL(12, 4), nullable=False)
	standard_quantity_per_unit = Column(DECIMAL(10, 4), default=1.0000)
	standard_rate_per_quantity = Column(DECIMAL(12, 4), nullable=False)
	
	# Units and Measures
	unit_of_measure = Column(String(50), nullable=False)
	quantity_unit_of_measure = Column(String(50), nullable=True)  # For quantity standards
	
	# Effective Period
	effective_date = Column(Date, nullable=False)
	end_date = Column(Date, nullable=True)
	fiscal_year = Column(Integer, nullable=False, index=True)
	version = Column(String(20), default='1.0')
	
	# Standard Type
	standard_type = Column(String(50), nullable=False, index=True)  # Ideal, Attainable, Current, Historical
	revision_reason = Column(String(200), nullable=True)
	
	# Tolerance Levels
	favorable_variance_threshold = Column(DECIMAL(5, 2), default=5.00)  # Percentage
	unfavorable_variance_threshold = Column(DECIMAL(5, 2), default=5.00)  # Percentage
	
	# Status
	is_active = Column(Boolean, default=True)
	is_approved = Column(Boolean, default=False)
	approved_by = Column(String(200), nullable=True)
	approved_date = Column(Date, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'cost_object_type', 'cost_object_code', 'cost_category_id', 'effective_date', 
						name='uq_standard_cost_object_date'),
		Index('ix_standard_cost_effective_active', 'effective_date', 'is_active'),
	)
	
	# Relationships
	cost_category = relationship("CFCACostCategory", back_populates="standard_costs")
	cost_center = relationship("CFCACostCenter")
	variance_analyses = relationship("CFCAVarianceAnalysis", back_populates="standard_cost")
	
	def __repr__(self):
		return f"<CFCAStandardCost {self.cost_object_code} - {self.cost_category.category_name}>"
	
	def is_current_standard(self, as_of_date: Optional[date] = None) -> bool:
		"""Check if this standard is current for the given date"""
		check_date = as_of_date or date.today()
		
		if check_date < self.effective_date:
			return False
		
		if self.end_date and check_date > self.end_date:
			return False
		
		return self.is_active
	
	def calculate_standard_cost(self, quantity: Decimal = Decimal('1')) -> Decimal:
		"""Calculate total standard cost for given quantity"""
		return self.standard_cost_per_unit * quantity
	
	def is_variance_significant(self, actual_cost: Decimal, quantity: Decimal = Decimal('1')) -> Dict[str, Any]:
		"""Check if variance from actual cost is significant"""
		standard_total = self.calculate_standard_cost(quantity)
		variance = actual_cost - standard_total
		variance_percent = (variance / standard_total * 100) if standard_total > 0 else Decimal('0')
		
		is_favorable = variance < 0
		threshold = self.favorable_variance_threshold if is_favorable else self.unfavorable_variance_threshold
		is_significant = abs(variance_percent) > threshold
		
		return {
			'variance_amount': variance,
			'variance_percent': variance_percent,
			'is_favorable': is_favorable,
			'is_significant': is_significant,
			'threshold_used': threshold,
			'standard_cost': standard_total,
			'actual_cost': actual_cost
		}


class CFCAVarianceAnalysis(Model, AuditMixin, BaseMixin):
	"""
	Variance Analysis for comparing actual costs to standard costs.
	
	Analyzes variances by type (price, quantity, efficiency, volume)
	and provides insights for cost control and performance measurement.
	"""
	__tablename__ = 'cf_ca_variance_analysis'
	
	# Identity
	variance_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Reference
	standard_cost_id = Column(String(36), ForeignKey('cf_ca_standard_cost.standard_cost_id'), nullable=False, index=True)
	cost_center_id = Column(String(36), ForeignKey('cf_ca_cost_center.cost_center_id'), nullable=True, index=True)
	
	# Analysis Period
	analysis_period = Column(String(7), nullable=False, index=True)  # YYYY-MM format
	fiscal_year = Column(Integer, nullable=False, index=True)
	fiscal_period = Column(Integer, nullable=False, index=True)
	analysis_date = Column(Date, nullable=False)
	
	# Cost Object
	cost_object_type = Column(String(50), nullable=False, index=True)
	cost_object_code = Column(String(50), nullable=False, index=True)
	cost_object_name = Column(String(200), nullable=False)
	
	# Actual vs Standard
	actual_quantity = Column(DECIMAL(15, 4), nullable=False)
	actual_cost = Column(DECIMAL(15, 2), nullable=False)
	actual_rate = Column(DECIMAL(12, 4), nullable=False)
	
	standard_quantity = Column(DECIMAL(15, 4), nullable=False)
	standard_cost = Column(DECIMAL(15, 2), nullable=False)
	standard_rate = Column(DECIMAL(12, 4), nullable=False)
	
	# Variance Components
	total_variance = Column(DECIMAL(15, 2), nullable=False)  # Actual - Standard
	price_rate_variance = Column(DECIMAL(15, 2), default=0.00)  # (Actual Rate - Standard Rate) * Actual Quantity
	quantity_efficiency_variance = Column(DECIMAL(15, 2), default=0.00)  # (Actual Qty - Standard Qty) * Standard Rate
	volume_variance = Column(DECIMAL(15, 2), default=0.00)  # For overhead variances
	
	# Variance Classifications
	is_favorable = Column(Boolean, nullable=False)  # Negative variance (under budget)
	is_significant = Column(Boolean, nullable=False)  # Exceeds threshold
	variance_percent = Column(DECIMAL(8, 4), nullable=False)  # Variance as % of standard
	
	# Analysis
	variance_type = Column(String(50), nullable=False, index=True)  # Material, Labor, Overhead, etc.
	primary_cause = Column(String(200), nullable=True)
	secondary_causes = Column(Text, nullable=True)  # JSON array of contributing factors
	
	# Management Response
	requires_action = Column(Boolean, default=False)
	action_taken = Column(Text, nullable=True)
	action_date = Column(Date, nullable=True)
	responsible_party = Column(String(200), nullable=True)
	
	# Follow-up
	follow_up_required = Column(Boolean, default=False)
	follow_up_date = Column(Date, nullable=True)
	resolution_status = Column(String(50), default='Open')  # Open, In Progress, Resolved, Closed
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'standard_cost_id', 'cost_object_code', 'analysis_period', 
						name='uq_variance_analysis_period'),
		Index('ix_variance_analysis_period_type', 'analysis_period', 'variance_type'),
	)
	
	# Relationships
	standard_cost = relationship("CFCAStandardCost", back_populates="variance_analyses")
	cost_center = relationship("CFCACostCenter")
	
	def __repr__(self):
		return f"<CFCAVarianceAnalysis {self.cost_object_code} - {self.analysis_period}>"
	
	def calculate_variance_components(self) -> Dict[str, Decimal]:
		"""Calculate detailed variance components"""
		# Price/Rate Variance: (Actual Rate - Standard Rate) × Actual Quantity
		price_variance = (self.actual_rate - self.standard_rate) * self.actual_quantity
		
		# Quantity/Efficiency Variance: (Actual Quantity - Standard Quantity) × Standard Rate
		quantity_variance = (self.actual_quantity - self.standard_quantity) * self.standard_rate
		
		# Volume Variance (for overhead): Based on capacity utilization
		# Simplified calculation - would be more complex in practice
		budgeted_quantity = self.standard_quantity
		volume_variance = (budgeted_quantity - self.actual_quantity) * self.standard_rate
		
		return {
			'price_rate_variance': price_variance,
			'quantity_efficiency_variance': quantity_variance,
			'volume_variance': volume_variance,
			'total_variance': price_variance + quantity_variance,
			'favorable_price': price_variance < 0,
			'favorable_quantity': quantity_variance < 0,
			'favorable_total': self.total_variance < 0
		}
	
	def get_variance_significance(self) -> str:
		"""Get variance significance rating"""
		abs_percent = abs(self.variance_percent)
		
		if abs_percent >= 20:
			return 'Critical'
		elif abs_percent >= 10:
			return 'High'
		elif abs_percent >= 5:
			return 'Medium'
		else:
			return 'Low'
	
	def get_potential_causes(self) -> List[str]:
		"""Get potential causes based on variance type and pattern"""
		causes = []
		
		if self.variance_type == 'Material':
			if self.price_rate_variance != 0:
				causes.extend([
					'Material price changes',
					'Supplier changes',
					'Purchase quantity discounts',
					'Market price fluctuations'
				])
			if self.quantity_efficiency_variance != 0:
				causes.extend([
					'Material waste/spoilage',
					'Process inefficiencies',
					'Quality issues',
					'Design changes'
				])
		
		elif self.variance_type == 'Labor':
			if self.price_rate_variance != 0:
				causes.extend([
					'Wage rate changes',
					'Overtime premiums',
					'Skills mix differences',
					'Union contract changes'
				])
			if self.quantity_efficiency_variance != 0:
				causes.extend([
					'Training issues',
					'Equipment problems',
					'Process improvements',
					'Worker experience levels'
				])
		
		elif self.variance_type == 'Overhead':
			causes.extend([
				'Volume changes',
				'Spending variances',
				'Capacity utilization',
				'Fixed cost changes',
				'Activity level changes'
			])
		
		return causes
	
	def recommend_actions(self) -> List[str]:
		"""Recommend management actions based on variance analysis"""
		actions = []
		
		if not self.is_significant:
			return ['Monitor - variance within acceptable limits']
		
		if self.is_favorable:
			actions.extend([
				'Investigate cause of favorable variance',
				'Consider updating standards if sustainable',
				'Share best practices with other areas',
				'Evaluate if quality was compromised'
			])
		else:
			actions.extend([
				'Investigate root cause immediately',
				'Implement corrective actions',
				'Review and update processes',
				'Consider training needs'
			])
		
		if abs(self.variance_percent) >= 20:
			actions.extend([
				'Escalate to senior management',
				'Require immediate action plan',
				'Daily monitoring until resolved'
			])
		
		return actions