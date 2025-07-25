"""
Enterprise Asset Management Models

Database models for comprehensive enterprise asset management including
asset master data, work orders, maintenance, inventory, contracts, and 
performance tracking with full APG platform integration.

Integration Points:
- auth_rbac: BaseMixin, AuditMixin, multi-tenant security
- audit_compliance: Complete audit trails and regulatory compliance
- fixed_asset_management: Financial asset synchronization
- predictive_maintenance: Asset health and failure prediction integration
- digital_twin_marketplace: Real-time asset mirroring and simulation
- document_management: Asset documentation and compliance certificates
- notification_engine: Automated alerts and stakeholder communications
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint, Index, JSON
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ...auth_rbac.models import BaseMixin, AuditMixin, Model


class EAAsset(Model, AuditMixin, BaseMixin):
	"""
	Enterprise Asset Master - Comprehensive asset registry with APG integration.
	
	Central repository for all enterprise assets with complete lifecycle tracking,
	financial integration, operational monitoring, and regulatory compliance.
	Integrates with APG Fixed Asset Management, Predictive Maintenance, and Digital Twin.
	"""
	__tablename__ = 'ea_asset'
	
	# Identity and Classification
	asset_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Asset Identification
	asset_number = Column(String(50), nullable=False, index=True)
	asset_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	asset_tag = Column(String(50), nullable=True, index=True)  # Physical barcode/QR code
	serial_number = Column(String(100), nullable=True, index=True)
	
	# Classification and Hierarchy
	asset_type = Column(String(50), nullable=False, index=True)  # equipment, facility, vehicle, etc.
	asset_category = Column(String(50), nullable=False, index=True)  # production, support, infrastructure
	asset_class = Column(String(50), nullable=True, index=True)  # rotating, static, electrical, etc.
	criticality_level = Column(String(20), default='medium', index=True)  # low, medium, high, critical
	
	# Hierarchy and Relationships
	parent_asset_id = Column(String(36), ForeignKey('ea_asset.asset_id'), nullable=True, index=True)
	system_id = Column(String(36), nullable=True, index=True)  # System/subsystem grouping
	facility_id = Column(String(36), nullable=True, index=True)  # Physical facility
	
	# Manufacturer and Technical Details
	manufacturer = Column(String(100), nullable=True, index=True)
	model_number = Column(String(100), nullable=True)
	year_manufactured = Column(Integer, nullable=True)
	specifications = Column(JSON, default=dict)  # Technical specifications
	capacity_rating = Column(String(100), nullable=True)  # Nameplate capacity
	
	# Location and Physical Details
	location_id = Column(String(36), ForeignKey('ea_location.location_id'), nullable=True, index=True)
	precise_location = Column(String(200), nullable=True)  # Room, floor, coordinates
	gps_latitude = Column(DECIMAL(10, 8), nullable=True)
	gps_longitude = Column(DECIMAL(11, 8), nullable=True)
	elevation = Column(Float, nullable=True)  # Meters above sea level
	
	# Operational Status and Lifecycle
	status = Column(String(20), default='active', index=True)  # active, inactive, maintenance, retired, disposed
	operational_status = Column(String(20), default='operational', index=True)  # operational, down, standby, testing
	lifecycle_stage = Column(String(20), default='operational', index=True)  # planning, construction, operational, end_of_life
	
	# Installation and Service Dates
	acquisition_date = Column(Date, nullable=True, index=True)
	installation_date = Column(Date, nullable=True)
	commissioning_date = Column(Date, nullable=True, index=True)
	warranty_start_date = Column(Date, nullable=True)
	warranty_end_date = Column(Date, nullable=True)
	expected_retirement_date = Column(Date, nullable=True)
	
	# Financial Information (Integration with Fixed Asset Management)
	purchase_cost = Column(DECIMAL(15, 2), nullable=True)
	current_book_value = Column(DECIMAL(15, 2), nullable=True)
	replacement_cost = Column(DECIMAL(15, 2), nullable=True)
	annual_operating_cost = Column(DECIMAL(15, 2), nullable=True)
	currency_code = Column(String(3), default='USD')
	
	# APG Fixed Asset Management Integration
	fixed_asset_id = Column(String(36), nullable=True, index=True)  # FK to CFAMAsset
	is_capitalized = Column(Boolean, default=True)
	
	# Performance and Reliability Metrics
	availability_target = Column(DECIMAL(5, 2), default=95.00)  # Target availability %
	current_availability = Column(DECIMAL(5, 2), nullable=True)  # Current availability %
	mtbf_hours = Column(Float, nullable=True)  # Mean Time Between Failures
	mttr_hours = Column(Float, nullable=True)  # Mean Time To Repair
	total_operating_hours = Column(Float, default=0.0)
	total_downtime_hours = Column(Float, default=0.0)
	
	# Maintenance Configuration
	maintenance_strategy = Column(String(50), default='predictive', index=True)  # reactive, preventive, predictive, condition_based
	maintenance_frequency_days = Column(Integer, nullable=True)
	maintenance_frequency_hours = Column(Float, nullable=True)
	last_maintenance_date = Column(Date, nullable=True)
	next_maintenance_due = Column(Date, nullable=True, index=True)
	
	# APG Predictive Maintenance Integration
	predictive_asset_id = Column(String(36), nullable=True, index=True)  # FK to PMAsset
	health_score = Column(Float, nullable=True, index=True)  # 0-100 health score
	condition_status = Column(String(20), nullable=True, index=True)  # excellent, good, fair, poor, critical
	
	# Digital Twin Integration
	digital_twin_id = Column(String(36), nullable=True, index=True)  # FK to Digital Twin
	has_digital_twin = Column(Boolean, default=False)
	iot_enabled = Column(Boolean, default=False)
	sensor_count = Column(Integer, default=0)
	
	# Environmental and Safety
	environmental_conditions = Column(JSON, default=dict)  # Temperature, humidity, etc.
	safety_classification = Column(String(50), nullable=True)  # Safety integrity level
	hazardous_area_rating = Column(String(20), nullable=True)  # Ex rating, Class/Division
	environmental_impact = Column(String(20), nullable=True)  # low, medium, high
	
	# Regulatory and Compliance
	regulatory_requirements = Column(JSON, default=list)  # List of applicable regulations
	inspection_requirements = Column(JSON, default=list)  # Required inspections
	certifications = Column(JSON, default=list)  # Certifications and approvals
	compliance_status = Column(String(20), default='compliant', index=True)
	
	# Energy and Sustainability
	energy_consumption_kwh = Column(Float, nullable=True)  # Annual energy consumption
	co2_emissions_tons = Column(Float, nullable=True)  # Annual CO2 emissions
	sustainability_rating = Column(String(20), nullable=True)  # A, B, C, D, E rating
	energy_efficiency_class = Column(String(10), nullable=True)  # A+++, A++, A+, A, B, C, D
	
	# Assignment and Responsibility
	custodian_employee_id = Column(String(36), nullable=True, index=True)
	department = Column(String(50), nullable=True, index=True)
	cost_center = Column(String(20), nullable=True, index=True)
	business_unit = Column(String(50), nullable=True)
	responsible_engineer_id = Column(String(36), nullable=True)
	
	# Vendor and Service Information
	primary_vendor_id = Column(String(36), nullable=True, index=True)
	service_provider_id = Column(String(36), nullable=True)
	support_contract_id = Column(String(36), nullable=True)
	support_level = Column(String(20), nullable=True)  # basic, standard, premium
	
	# Documentation and Media
	documentation_links = Column(JSON, default=list)  # Links to manuals, drawings, etc.
	photo_urls = Column(JSON, default=list)  # Asset photos
	video_urls = Column(JSON, default=list)  # Training/maintenance videos
	qr_code_url = Column(String(500), nullable=True)  # Generated QR code for mobile access
	
	# Custom Fields and Metadata
	custom_attributes = Column(JSON, default=dict)  # Tenant-specific custom fields
	tags = Column(JSON, default=list)  # Searchable tags
	external_system_ids = Column(JSON, default=dict)  # Integration with external systems
	
	# Audit and Change Tracking (Enhanced)
	version_number = Column(Integer, default=1)
	change_reason = Column(String(200), nullable=True)
	approved_by = Column(String(36), nullable=True)
	approval_date = Column(DateTime, nullable=True)
	
	# Performance Optimization
	search_vector = Column(Text, nullable=True)  # Full-text search optimization
	last_indexed = Column(DateTime, nullable=True)
	
	# Constraints and Indexes
	__table_args__ = (
		UniqueConstraint('tenant_id', 'asset_number', name='uq_ea_asset_number_tenant'),
		UniqueConstraint('tenant_id', 'asset_tag', name='uq_ea_asset_tag_tenant'),
		Index('idx_ea_asset_location_status', 'location_id', 'status'),
		Index('idx_ea_asset_type_criticality', 'asset_type', 'criticality_level'),
		Index('idx_ea_asset_maintenance_due', 'next_maintenance_due', 'status'),
		Index('idx_ea_asset_health_condition', 'health_score', 'condition_status'),
		Index('idx_ea_asset_search', 'search_vector'),
	)
	
	# Relationships
	parent_asset = relationship("EAAsset", remote_side=[asset_id])
	child_assets = relationship("EAAsset", back_populates="parent_asset")
	location = relationship("EALocation", back_populates="assets")
	
	# Work Orders and Maintenance
	work_orders = relationship("EAWorkOrder", back_populates="asset")
	maintenance_records = relationship("EAMaintenanceRecord", back_populates="asset")
	
	# Performance and Analytics
	performance_records = relationship("EAPerformanceRecord", back_populates="asset")
	
	# Contracts and Agreements
	contracts = relationship("EAContract", back_populates="asset", secondary="ea_asset_contract")
	
	def __repr__(self):
		return f"<EAAsset {self.asset_number} - {self.asset_name}>"
	
	async def _log_asset_creation(self) -> str:
		"""Log asset creation for APG audit compliance"""
		return f"Created asset {self.asset_number} - {self.asset_name} in {self.location.location_name if self.location else 'Unknown Location'}"
	
	async def _log_status_change(self, old_status: str, new_status: str) -> str:
		"""Log status changes for APG audit compliance"""
		return f"Asset {self.asset_number} status changed from {old_status} to {new_status}"
	
	def calculate_availability(self) -> Decimal:
		"""Calculate current availability percentage"""
		if self.total_operating_hours <= 0:
			return Decimal('100.00')
		
		total_time = self.total_operating_hours + self.total_downtime_hours
		if total_time <= 0:
			return Decimal('100.00')
		
		availability = (self.total_operating_hours / total_time) * 100
		self.current_availability = Decimal(str(round(availability, 2)))
		return self.current_availability
	
	def get_maintenance_status(self) -> str:
		"""Get maintenance status based on due date"""
		if not self.next_maintenance_due:
			return 'no_schedule'
		
		today = date.today()
		days_until_due = (self.next_maintenance_due - today).days
		
		if days_until_due < 0:
			return 'overdue'
		elif days_until_due <= 7:
			return 'due_soon'
		elif days_until_due <= 30:
			return 'upcoming'
		else:
			return 'current'
	
	def get_health_status(self) -> str:
		"""Get overall health status"""
		if self.health_score is None:
			return 'unknown'
		elif self.health_score >= 90:
			return 'excellent'
		elif self.health_score >= 75:
			return 'good'
		elif self.health_score >= 50:
			return 'fair'
		elif self.health_score >= 25:
			return 'poor'
		else:
			return 'critical'
	
	def is_critical_asset(self) -> bool:
		"""Check if asset is business critical"""
		return self.criticality_level in ['high', 'critical']
	
	def get_full_hierarchy_path(self) -> str:
		"""Get full asset hierarchy path"""
		if self.parent_asset:
			return f"{self.parent_asset.get_full_hierarchy_path()} > {self.asset_name}"
		return self.asset_name
	
	def update_search_vector(self):
		"""Update search vector for full-text search"""
		search_terms = [
			self.asset_number or '',
			self.asset_name or '',
			self.description or '',
			self.manufacturer or '',
			self.model_number or '',
			self.serial_number or '',
			self.asset_type or '',
			self.asset_category or '',
			' '.join(self.tags or [])
		]
		self.search_vector = ' '.join(filter(None, search_terms)).lower()
		self.last_indexed = datetime.utcnow()


class EALocation(Model, AuditMixin, BaseMixin):
	"""
	Asset Location Hierarchy - Physical and logical asset locations.
	
	Hierarchical location structure supporting unlimited depth with GPS coordinates,
	facility integration, and environmental condition tracking.
	"""
	__tablename__ = 'ea_location'
	
	# Identity
	location_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Location Information
	location_code = Column(String(50), nullable=False, index=True)
	location_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	location_type = Column(String(50), nullable=False, index=True)  # site, building, floor, room, area, zone
	
	# Hierarchy
	parent_location_id = Column(String(36), ForeignKey('ea_location.location_id'), nullable=True, index=True)
	hierarchy_level = Column(Integer, default=0, index=True)  # 0=root, 1=level1, etc.
	hierarchy_path = Column(Text, nullable=True)  # Full path for efficient queries
	
	# Geographic Information
	address = Column(Text, nullable=True)
	city = Column(String(100), nullable=True)
	state_province = Column(String(50), nullable=True)
	postal_code = Column(String(20), nullable=True)
	country_code = Column(String(2), nullable=True)
	
	# GPS Coordinates
	gps_latitude = Column(DECIMAL(10, 8), nullable=True)
	gps_longitude = Column(DECIMAL(11, 8), nullable=True)
	elevation = Column(Float, nullable=True)  # Meters above sea level
	coordinate_system = Column(String(20), default='WGS84')
	
	# Physical Characteristics
	floor_area_sqm = Column(Float, nullable=True)  # Square meters
	volume_cum = Column(Float, nullable=True)  # Cubic meters
	max_capacity = Column(Integer, nullable=True)  # Maximum occupancy/capacity
	current_utilization = Column(Float, nullable=True)  # Current utilization %
	
	# Environmental Conditions
	temperature_range_min = Column(Float, nullable=True)  # Celsius
	temperature_range_max = Column(Float, nullable=True)  # Celsius
	humidity_range_min = Column(Float, nullable=True)  # Percentage
	humidity_range_max = Column(Float, nullable=True)  # Percentage
	environmental_class = Column(String(20), nullable=True)  # Clean room class, etc.
	
	# Safety and Security
	safety_zone = Column(String(50), nullable=True)  # Safety classification
	security_level = Column(String(20), nullable=True)  # Security clearance required
	hazardous_area_rating = Column(String(20), nullable=True)  # Explosion rating
	emergency_assembly_point = Column(String(200), nullable=True)
	
	# Access and Infrastructure
	access_requirements = Column(JSON, default=list)  # Required access permissions
	available_utilities = Column(JSON, default=list)  # Power, water, gas, compressed air, etc.
	network_availability = Column(JSON, default=dict)  # IT/OT network access
	crane_coverage = Column(Boolean, default=False)
	fork_lift_access = Column(Boolean, default=False)
	
	# Status and Operations
	is_active = Column(Boolean, default=True, index=True)
	operational_hours = Column(JSON, default=dict)  # Operating schedule
	access_restrictions = Column(JSON, default=list)  # Time/personnel restrictions
	maintenance_window = Column(String(100), nullable=True)  # Preferred maintenance time
	
	# Cost Centers and Ownership
	cost_center = Column(String(20), nullable=True, index=True)
	business_unit = Column(String(50), nullable=True)
	facility_manager_id = Column(String(36), nullable=True)
	contact_information = Column(JSON, default=dict)  # Emergency contacts, etc.
	
	# Documentation and Media
	layout_drawing_url = Column(String(500), nullable=True)
	photo_urls = Column(JSON, default=list)
	virtual_tour_url = Column(String(500), nullable=True)
	
	# Custom Attributes
	custom_attributes = Column(JSON, default=dict)
	tags = Column(JSON, default=list)
	
	# Constraints and Indexes
	__table_args__ = (
		UniqueConstraint('tenant_id', 'location_code', name='uq_ea_location_code_tenant'),
		Index('idx_ea_location_type_active', 'location_type', 'is_active'),
		Index('idx_ea_location_hierarchy', 'hierarchy_level', 'parent_location_id'),
		Index('idx_ea_location_gps', 'gps_latitude', 'gps_longitude'),
	)
	
	# Relationships
	parent_location = relationship("EALocation", remote_side=[location_id])
	child_locations = relationship("EALocation", back_populates="parent_location")
	assets = relationship("EAAsset", back_populates="location")
	
	def __repr__(self):
		return f"<EALocation {self.location_code} - {self.location_name}>"
	
	async def _log_location_creation(self) -> str:
		"""Log location creation for APG audit compliance"""
		return f"Created location {self.location_code} - {self.location_name} at level {self.hierarchy_level}"
	
	def get_full_path(self) -> str:
		"""Get full location hierarchy path"""
		if self.parent_location:
			return f"{self.parent_location.get_full_path()} > {self.location_name}"
		return self.location_name
	
	def calculate_utilization(self) -> float:
		"""Calculate current utilization percentage"""
		if not self.max_capacity or not self.assets:
			return 0.0
		
		active_assets = len([a for a in self.assets if a.status == 'active'])
		utilization = (active_assets / self.max_capacity) * 100
		self.current_utilization = round(utilization, 2)
		return self.current_utilization
	
	def get_distance_to(self, other_location: 'EALocation') -> float:
		"""Calculate distance to another location in kilometers"""
		if not all([self.gps_latitude, self.gps_longitude, 
				   other_location.gps_latitude, other_location.gps_longitude]):
			return 0.0
		
		# Haversine formula for great circle distance
		from math import radians, cos, sin, asin, sqrt
		
		lat1, lon1 = radians(float(self.gps_latitude)), radians(float(self.gps_longitude))
		lat2, lon2 = radians(float(other_location.gps_latitude)), radians(float(other_location.gps_longitude))
		
		dlat = lat2 - lat1
		dlon = lon2 - lon1
		a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
		c = 2 * asin(sqrt(a))
		r = 6371  # Radius of earth in kilometers
		
		return round(c * r, 2)
	
	def update_hierarchy_path(self):
		"""Update hierarchy path for efficient queries"""
		path_parts = []
		current = self
		while current:
			path_parts.insert(0, current.location_code)
			current = current.parent_location
		self.hierarchy_path = ' > '.join(path_parts)


class EAWorkOrder(Model, AuditMixin, BaseMixin):
	"""
	Work Order Management - Comprehensive work order tracking with APG integration.
	
	Manages all types of work orders including maintenance, projects, and inspections
	with integration to APG Real-time Collaboration and Notification Engine.
	"""
	__tablename__ = 'ea_work_order'
	
	# Identity
	work_order_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Work Order Information
	work_order_number = Column(String(50), nullable=False, index=True)
	title = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=False)
	work_type = Column(String(50), nullable=False, index=True)  # maintenance, repair, inspection, project, emergency
	priority = Column(String(20), default='medium', index=True)  # low, medium, high, urgent, emergency
	
	# Asset and Location References
	asset_id = Column(String(36), ForeignKey('ea_asset.asset_id'), nullable=True, index=True)
	location_id = Column(String(36), ForeignKey('ea_location.location_id'), nullable=True, index=True)
	system_id = Column(String(36), nullable=True, index=True)  # System/subsystem
	
	# Classification and Categories
	work_category = Column(String(50), nullable=True, index=True)  # mechanical, electrical, civil, etc.
	maintenance_type = Column(String(50), nullable=True)  # preventive, corrective, predictive, condition_based
	safety_category = Column(String(20), nullable=True)  # routine, permit_required, confined_space, hot_work
	skill_requirements = Column(JSON, default=list)  # Required skills/certifications
	
	# Scheduling and Timing
	requested_date = Column(DateTime, nullable=False, index=True)
	scheduled_start = Column(DateTime, nullable=True, index=True)
	scheduled_end = Column(DateTime, nullable=True, index=True)
	actual_start = Column(DateTime, nullable=True, index=True)
	actual_end = Column(DateTime, nullable=True, index=True)
	
	# Duration and Effort
	estimated_hours = Column(Float, nullable=True)
	actual_hours = Column(Float, nullable=True)
	estimated_cost = Column(DECIMAL(15, 2), nullable=True)
	actual_cost = Column(DECIMAL(15, 2), nullable=True)
	
	# Status and Workflow
	status = Column(String(20), default='draft', index=True)  # draft, planned, scheduled, in_progress, on_hold, completed, cancelled, closed
	workflow_stage = Column(String(50), nullable=True, index=True)  # planning, approval, execution, review, closure
	completion_percentage = Column(Integer, default=0)  # 0-100%
	
	# Assignment and Resources
	assigned_to = Column(String(36), nullable=True, index=True)  # Primary assignee
	assigned_team = Column(JSON, default=list)  # Team members
	supervisor_id = Column(String(36), nullable=True)
	coordinator_id = Column(String(36), nullable=True)
	required_crew_size = Column(Integer, default=1)
	
	# External Resources
	contractor_id = Column(String(36), nullable=True, index=True)
	vendor_id = Column(String(36), nullable=True)
	external_work_order = Column(String(50), nullable=True)  # Contractor's WO number
	
	# Parts and Materials
	parts_required = Column(JSON, default=list)  # Required parts list
	parts_reserved = Column(JSON, default=list)  # Reserved parts
	parts_issued = Column(JSON, default=list)  # Issued parts
	total_parts_cost = Column(DECIMAL(15, 2), default=0.00)
	
	# Tools and Equipment
	tools_required = Column(JSON, default=list)  # Required tools/equipment
	special_tools = Column(JSON, default=list)  # Special/rented tools
	equipment_downtime_required = Column(Boolean, default=False)
	isolation_required = Column(Boolean, default=False)
	
	# Safety and Permits
	safety_requirements = Column(JSON, default=list)  # Safety procedures
	permits_required = Column(JSON, default=list)  # Work permits
	hazard_assessment = Column(Text, nullable=True)
	safety_precautions = Column(Text, nullable=True)
	
	# Quality and Compliance
	quality_requirements = Column(JSON, default=list)  # Quality standards
	inspection_points = Column(JSON, default=list)  # Inspection checkpoints
	regulatory_requirements = Column(JSON, default=list)  # Applicable regulations
	compliance_verification = Column(Boolean, default=False)
	
	# Work Instructions and Procedures
	work_instructions = Column(Text, nullable=True)
	procedure_references = Column(JSON, default=list)  # Links to procedures
	checklist = Column(JSON, default=list)  # Work checklist
	completion_criteria = Column(Text, nullable=True)
	
	# Failure and Problem Details
	failure_mode = Column(String(100), nullable=True)  # How it failed
	failure_cause = Column(String(200), nullable=True)  # Why it failed
	problem_description = Column(Text, nullable=True)
	root_cause_analysis = Column(Text, nullable=True)
	
	# Work Completion and Results
	work_performed = Column(Text, nullable=True)
	completion_notes = Column(Text, nullable=True)
	recommendations = Column(Text, nullable=True)
	follow_up_required = Column(Boolean, default=False)
	follow_up_work_orders = Column(JSON, default=list)  # Related WOs
	
	# Financial Tracking
	budget_code = Column(String(20), nullable=True)
	cost_center = Column(String(20), nullable=True)
	project_code = Column(String(20), nullable=True)
	currency_code = Column(String(3), default='USD')
	labor_cost = Column(DECIMAL(15, 2), default=0.00)
	contractor_cost = Column(DECIMAL(15, 2), default=0.00)
	overhead_cost = Column(DECIMAL(15, 2), default=0.00)
	
	# Performance Metrics
	response_time_hours = Column(Float, nullable=True)  # Time to start work
	resolution_time_hours = Column(Float, nullable=True)  # Time to complete
	first_time_fix = Column(Boolean, nullable=True)  # Fixed on first attempt
	rework_required = Column(Boolean, default=False)
	
	# Approval and Authorization
	requires_approval = Column(Boolean, default=False)
	approval_level = Column(String(20), nullable=True)  # supervisor, manager, director
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	approval_notes = Column(Text, nullable=True)
	
	# Integration with APG Capabilities
	predictive_maintenance_alert_id = Column(String(36), nullable=True)  # Triggered by PM alert
	notification_sent = Column(Boolean, default=False)
	collaboration_room_id = Column(String(36), nullable=True)  # APG Real-time Collaboration
	
	# Documentation and Attachments
	attachments = Column(JSON, default=list)  # Document/photo attachments
	before_photos = Column(JSON, default=list)  # Before photos
	after_photos = Column(JSON, default=list)  # After photos
	video_links = Column(JSON, default=list)  # Training/reference videos
	
	# Mobile and Field Support
	mobile_accessible = Column(Boolean, default=True)
	offline_capable = Column(Boolean, default=True)
	gps_check_in_required = Column(Boolean, default=False)
	barcode_scan_required = Column(Boolean, default=False)
	
	# Customer and Stakeholder Communication
	customer_notification_required = Column(Boolean, default=False)
	stakeholder_updates = Column(JSON, default=list)  # Communication log
	customer_satisfaction_score = Column(Integer, nullable=True)  # 1-5 rating
	
	# Constraints and Indexes
	__table_args__ = (
		UniqueConstraint('tenant_id', 'work_order_number', name='uq_ea_work_order_number_tenant'),
		Index('idx_ea_work_order_asset_status', 'asset_id', 'status'),
		Index('idx_ea_work_order_scheduled', 'scheduled_start', 'scheduled_end'),
		Index('idx_ea_work_order_priority_status', 'priority', 'status'),
		Index('idx_ea_work_order_assigned', 'assigned_to', 'status'),
		Index('idx_ea_work_order_type_category', 'work_type', 'work_category'),
	)
	
	# Relationships
	asset = relationship("EAAsset", back_populates="work_orders")
	location = relationship("EALocation")
	maintenance_records = relationship("EAMaintenanceRecord", back_populates="work_order")
	
	def __repr__(self):
		return f"<EAWorkOrder {self.work_order_number} - {self.title}>"
	
	async def _log_work_order_creation(self) -> str:
		"""Log work order creation for APG audit compliance"""
		return f"Created work order {self.work_order_number} for asset {self.asset.asset_number if self.asset else 'N/A'}"
	
	async def _log_status_change(self, old_status: str, new_status: str) -> str:
		"""Log status changes for APG audit compliance"""
		return f"Work order {self.work_order_number} status changed from {old_status} to {new_status}"
	
	def calculate_duration(self) -> timedelta:
		"""Calculate actual work duration"""
		if self.actual_start and self.actual_end:
			return self.actual_end - self.actual_start
		return timedelta(0)
	
	def calculate_cost_variance(self) -> Decimal:
		"""Calculate cost variance (actual vs estimated)"""
		if not self.estimated_cost or not self.actual_cost:
			return Decimal('0.00')
		return self.actual_cost - self.estimated_cost
	
	def calculate_schedule_variance_hours(self) -> float:
		"""Calculate schedule variance in hours"""
		if not self.scheduled_end or not self.actual_end:
			return 0.0
		delta = self.actual_end - self.scheduled_end
		return delta.total_seconds() / 3600
	
	def is_overdue(self) -> bool:
		"""Check if work order is overdue"""
		if not self.scheduled_end or self.status in ['completed', 'cancelled', 'closed']:
			return False
		return datetime.utcnow() > self.scheduled_end
	
	def get_urgency_level(self) -> str:
		"""Calculate urgency based on priority and due date"""
		if self.priority == 'emergency':
			return 'immediate'
		elif self.priority == 'urgent':
			return 'urgent'
		elif self.is_overdue():
			return 'overdue'
		elif self.scheduled_start and (self.scheduled_start - datetime.utcnow()).days <= 1:
			return 'due_soon'
		else:
			return 'normal'
	
	def update_completion_percentage(self):
		"""Update completion percentage based on checklist"""
		if not self.checklist:
			return
		
		completed_items = sum(1 for item in self.checklist if item.get('completed', False))
		total_items = len(self.checklist)
		
		if total_items > 0:
			self.completion_percentage = int((completed_items / total_items) * 100)


class EAMaintenanceRecord(Model, AuditMixin, BaseMixin):
	"""
	Maintenance Record - Comprehensive maintenance activity tracking.
	
	Detailed records of all maintenance activities with integration to
	APG Predictive Maintenance and performance analytics.
	"""
	__tablename__ = 'ea_maintenance_record'
	
	# Identity
	record_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# References
	asset_id = Column(String(36), ForeignKey('ea_asset.asset_id'), nullable=False, index=True)
	work_order_id = Column(String(36), ForeignKey('ea_work_order.work_order_id'), nullable=True, index=True)
	location_id = Column(String(36), ForeignKey('ea_location.location_id'), nullable=True, index=True)
	
	# Maintenance Details
	maintenance_number = Column(String(50), nullable=False, index=True)
	maintenance_type = Column(String(50), nullable=False, index=True)  # preventive, corrective, predictive, emergency
	maintenance_category = Column(String(50), nullable=False)  # routine, repair, overhaul, replacement, inspection
	description = Column(Text, nullable=False)
	
	# Timing and Duration
	started_at = Column(DateTime, nullable=False, index=True)
	completed_at = Column(DateTime, nullable=True, index=True)
	duration_hours = Column(Float, nullable=True)
	downtime_hours = Column(Float, nullable=True)  # Asset unavailable time
	
	# Work Details
	work_performed = Column(Text, nullable=False)
	parts_replaced = Column(JSON, default=list)  # Detailed parts information
	consumables_used = Column(JSON, default=list)  # Oils, filters, etc.
	tools_used = Column(JSON, default=list)  # Tools and equipment used
	
	# Personnel and Resources
	technician_id = Column(String(36), nullable=True, index=True)
	supervisor_id = Column(String(36), nullable=True)
	team_members = Column(JSON, default=list)  # Additional team members
	contractor_id = Column(String(36), nullable=True)
	labor_hours = Column(Float, nullable=True)
	
	# Cost Tracking
	labor_cost = Column(DECIMAL(15, 2), default=0.00)
	parts_cost = Column(DECIMAL(15, 2), default=0.00)
	contractor_cost = Column(DECIMAL(15, 2), default=0.00)
	other_costs = Column(DECIMAL(15, 2), default=0.00)
	total_cost = Column(DECIMAL(15, 2), default=0.00)
	currency_code = Column(String(3), default='USD')
	
	# Quality and Outcomes
	outcome = Column(String(20), nullable=False, index=True)  # successful, partial, failed, deferred
	quality_rating = Column(Integer, nullable=True)  # 1-5 rating
	effectiveness_score = Column(Float, nullable=True)  # Calculated effectiveness
	first_time_fix = Column(Boolean, nullable=True)
	
	# Asset Condition Before/After
	condition_before = Column(String(20), nullable=True)  # excellent, good, fair, poor, critical
	condition_after = Column(String(20), nullable=True)
	health_score_before = Column(Float, nullable=True)  # 0-100
	health_score_after = Column(Float, nullable=True)  # 0-100
	performance_improvement = Column(Float, nullable=True)  # % improvement
	
	# Findings and Analysis
	findings = Column(Text, nullable=True)
	root_cause = Column(Text, nullable=True)
	failure_mode = Column(String(100), nullable=True)
	wear_patterns = Column(Text, nullable=True)
	recommendations = Column(Text, nullable=True)
	
	# Follow-up and Future Actions
	follow_up_required = Column(Boolean, default=False)
	follow_up_date = Column(Date, nullable=True)
	next_maintenance_adjustment = Column(Integer, nullable=True)  # Days adjustment
	parts_to_order = Column(JSON, default=list)
	
	# Safety and Compliance
	safety_incidents = Column(JSON, default=list)  # Any safety incidents
	permits_used = Column(JSON, default=list)  # Work permits
	regulatory_compliance = Column(Boolean, default=True)
	inspection_results = Column(JSON, default=dict)  # Inspection outcomes
	
	# Environmental Impact
	waste_generated = Column(JSON, default=list)  # Waste materials
	environmental_impact = Column(Text, nullable=True)
	recycling_actions = Column(JSON, default=list)
	
	# Documentation
	before_photos = Column(JSON, default=list)
	after_photos = Column(JSON, default=list)
	measurement_data = Column(JSON, default=dict)  # Vibration, temperature, etc.
	test_results = Column(JSON, default=dict)  # Performance tests
	certificates = Column(JSON, default=list)  # Calibration certificates, etc.
	
	# Warranty and Guarantees
	warranty_work = Column(Boolean, default=False)
	warranty_claim_number = Column(String(50), nullable=True)
	work_guarantee_period = Column(Integer, nullable=True)  # Days
	guarantee_expires = Column(Date, nullable=True)
	
	# Predictive Maintenance Integration
	triggered_by_prediction = Column(Boolean, default=False)
	prediction_accuracy = Column(Float, nullable=True)  # If triggered by prediction
	actual_failure_mode = Column(String(100), nullable=True)
	prediction_feedback = Column(Text, nullable=True)
	
	# Performance Metrics
	mean_time_to_repair = Column(Float, nullable=True)  # MTTR for this maintenance
	reliability_improvement = Column(Float, nullable=True)  # Reliability change
	availability_impact = Column(Float, nullable=True)  # Availability impact
	
	# Knowledge Management
	lessons_learned = Column(Text, nullable=True)
	best_practices = Column(Text, nullable=True)
	knowledge_articles = Column(JSON, default=list)  # Links to knowledge base
	
	# Constraints and Indexes
	__table_args__ = (
		UniqueConstraint('tenant_id', 'maintenance_number', name='uq_ea_maintenance_number_tenant'),
		Index('idx_ea_maintenance_asset_type', 'asset_id', 'maintenance_type'),
		Index('idx_ea_maintenance_date_outcome', 'started_at', 'outcome'),
		Index('idx_ea_maintenance_technician', 'technician_id', 'started_at'),
		Index('idx_ea_maintenance_cost', 'total_cost', 'maintenance_type'),
	)
	
	# Relationships
	asset = relationship("EAAsset", back_populates="maintenance_records")
	work_order = relationship("EAWorkOrder", back_populates="maintenance_records")
	location = relationship("EALocation")
	
	def __repr__(self):
		return f"<EAMaintenanceRecord {self.maintenance_number} - {self.asset.asset_number if self.asset else 'N/A'}>"
	
	async def _log_maintenance_completion(self) -> str:
		"""Log maintenance completion for APG audit compliance"""
		return f"Completed {self.maintenance_type} maintenance {self.maintenance_number} on asset {self.asset.asset_number if self.asset else 'N/A'}"
	
	def calculate_total_cost(self):
		"""Calculate total maintenance cost"""
		self.total_cost = (
			self.labor_cost + 
			self.parts_cost + 
			self.contractor_cost + 
			self.other_costs
		)
	
	def calculate_effectiveness_score(self) -> float:
		"""Calculate maintenance effectiveness score (0-100)"""
		score = 0.0
		factors = 0
		
		# Outcome factor (40% weight)
		if self.outcome == 'successful':
			score += 40
		elif self.outcome == 'partial':
			score += 20
		factors += 1
		
		# First time fix factor (20% weight)
		if self.first_time_fix is True:
			score += 20
		elif self.first_time_fix is False:
			score += 5
		else:
			score += 10  # Unknown
		factors += 1
		
		# Health improvement factor (25% weight)
		if self.health_score_before and self.health_score_after:
			improvement = self.health_score_after - self.health_score_before
			score += min(25, max(0, improvement / 4))  # Scale to 0-25
		else:
			score += 12.5  # Neutral if unknown
		factors += 1
		
		# Quality rating factor (15% weight)
		if self.quality_rating:
			score += (self.quality_rating / 5) * 15
		else:
			score += 7.5  # Neutral if unknown
		factors += 1
		
		self.effectiveness_score = round(score, 1)
		return self.effectiveness_score
	
	def calculate_cost_per_hour(self) -> Decimal:
		"""Calculate cost per hour of maintenance"""
		if not self.duration_hours or self.duration_hours <= 0:
			return Decimal('0.00')
		return self.total_cost / Decimal(str(self.duration_hours))
	
	def update_asset_condition(self):
		"""Update asset condition based on maintenance results"""
		if self.asset and self.condition_after:
			self.asset.condition_status = self.condition_after
		
		if self.asset and self.health_score_after:
			self.asset.health_score = self.health_score_after


class EAInventory(Model, AuditMixin, BaseMixin):
	"""
	Inventory Management - Parts, materials, and consumables tracking.
	
	Comprehensive inventory management with integration to APG Procurement
	for automated ordering and vendor management.
	"""
	__tablename__ = 'ea_inventory'
	
	# Identity
	inventory_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Item Information
	part_number = Column(String(50), nullable=False, index=True)
	description = Column(String(200), nullable=False, index=True)
	item_type = Column(String(50), nullable=False, index=True)  # spare_part, consumable, tool, material
	category = Column(String(50), nullable=False, index=True)  # mechanical, electrical, instrumentation, etc.
	subcategory = Column(String(50), nullable=True)
	
	# Classification and Specifications
	manufacturer = Column(String(100), nullable=True, index=True)
	manufacturer_part_number = Column(String(100), nullable=True)
	model_number = Column(String(100), nullable=True)
	specifications = Column(JSON, default=dict)  # Technical specifications
	dimensions = Column(JSON, default=dict)  # Length, width, height, weight
	
	# Inventory Levels
	current_stock = Column(Integer, default=0, index=True)
	minimum_stock = Column(Integer, default=0)
	maximum_stock = Column(Integer, default=0)
	reorder_point = Column(Integer, default=0)
	economic_order_quantity = Column(Integer, default=1)
	safety_stock = Column(Integer, default=0)
	
	# Storage and Location
	location_id = Column(String(36), ForeignKey('ea_location.location_id'), nullable=True, index=True)
	storage_location = Column(String(100), nullable=True)  # Specific storage location
	bin_location = Column(String(50), nullable=True)  # Bin/shelf identifier
	storage_requirements = Column(JSON, default=dict)  # Temperature, humidity, etc.
	
	# Cost and Valuation
	unit_cost = Column(DECIMAL(15, 2), nullable=True)
	average_cost = Column(DECIMAL(15, 2), nullable=True)
	last_cost = Column(DECIMAL(15, 2), nullable=True)
	total_value = Column(DECIMAL(15, 2), nullable=True)
	currency_code = Column(String(3), default='USD')
	
	# Vendor and Procurement
	primary_vendor_id = Column(String(36), nullable=True, index=True)
	alternate_vendors = Column(JSON, default=list)  # Alternative vendor IDs
	lead_time_days = Column(Integer, nullable=True)
	minimum_order_quantity = Column(Integer, default=1)
	purchase_unit = Column(String(20), default='EA')  # Each, box, meter, etc.
	
	# Asset Compatibility
	compatible_assets = Column(JSON, default=list)  # Asset IDs this part fits
	compatible_asset_types = Column(JSON, default=list)  # Asset types
	interchangeable_parts = Column(JSON, default=list)  # Alternative part numbers
	superseded_by = Column(String(36), nullable=True)  # Replacement part ID
	supersedes = Column(JSON, default=list)  # Parts this replaces
	
	# Usage and Consumption
	annual_usage = Column(Integer, default=0)  # Annual consumption
	last_issue_date = Column(Date, nullable=True)
	last_receipt_date = Column(Date, nullable=True)
	total_issued_ytd = Column(Integer, default=0)
	total_received_ytd = Column(Integer, default=0)
	
	# Lifecycle and Status
	status = Column(String(20), default='active', index=True)  # active, inactive, obsolete, discontinued
	lifecycle_stage = Column(String(20), nullable=True)  # introduction, growth, maturity, decline, obsolete
	criticality = Column(String(20), default='medium', index=True)  # low, medium, high, critical
	
	# Quality and Condition
	condition = Column(String(20), default='new', index=True)  # new, used, refurbished, damaged
	quality_grade = Column(String(10), nullable=True)  # A, B, C grade
	inspection_required = Column(Boolean, default=False)
	shelf_life_days = Column(Integer, nullable=True)
	expiry_tracking = Column(Boolean, default=False)
	
	# Physical Characteristics
	unit_of_measure = Column(String(20), default='EA')
	weight_kg = Column(Float, nullable=True)
	volume_cubic_meters = Column(Float, nullable=True)
	hazardous_material = Column(Boolean, default=False)
	hazard_class = Column(String(20), nullable=True)  # Hazardous material classification
	
	# Tracking and Identification
	barcode = Column(String(100), nullable=True, index=True)
	rfid_tag = Column(String(100), nullable=True)
	serial_number_tracking = Column(Boolean, default=False)
	lot_tracking = Column(Boolean, default=False)
	
	# Automated Management
	auto_reorder = Column(Boolean, default=False)
	auto_reorder_vendor_id = Column(String(36), nullable=True)
	last_auto_order_date = Column(Date, nullable=True)
	demand_forecast = Column(JSON, default=dict)  # Forecasted demand
	
	# Documentation and Media
	photo_urls = Column(JSON, default=list)
	datasheet_url = Column(String(500), nullable=True)
	installation_guide_url = Column(String(500), nullable=True)
	safety_datasheet_url = Column(String(500), nullable=True)
	
	# Custom Attributes
	custom_attributes = Column(JSON, default=dict)
	tags = Column(JSON, default=list)
	
	# Constraints and Indexes
	__table_args__ = (
		UniqueConstraint('tenant_id', 'part_number', name='uq_ea_inventory_part_number_tenant'),
		Index('idx_ea_inventory_stock', 'current_stock', 'minimum_stock'),
		Index('idx_ea_inventory_type_category', 'item_type', 'category'),
		Index('idx_ea_inventory_vendor', 'primary_vendor_id', 'status'),
		Index('idx_ea_inventory_criticality', 'criticality', 'current_stock'),
	)
	
	# Relationships
	location = relationship("EALocation")
	
	def __repr__(self):
		return f"<EAInventory {self.part_number} - {self.description}>"
	
	async def _log_stock_change(self, old_stock: int, new_stock: int, reason: str) -> str:
		"""Log stock changes for APG audit compliance"""
		return f"Stock for {self.part_number} changed from {old_stock} to {new_stock} - {reason}"
	
	def calculate_total_value(self):
		"""Calculate total inventory value"""
		if self.current_stock and self.average_cost:
			self.total_value = Decimal(str(self.current_stock)) * self.average_cost
		else:
			self.total_value = Decimal('0.00')
	
	def is_reorder_required(self) -> bool:
		"""Check if reorder is required"""
		return self.current_stock <= self.reorder_point
	
	def is_stockout(self) -> bool:
		"""Check if item is out of stock"""
		return self.current_stock <= 0
	
	def is_overstocked(self) -> bool:
		"""Check if item is overstocked"""
		return self.current_stock > self.maximum_stock
	
	def calculate_stock_days(self) -> float:
		"""Calculate days of stock remaining"""
		if not self.annual_usage or self.annual_usage <= 0:
			return float('inf')
		
		daily_usage = self.annual_usage / 365
		if daily_usage <= 0:
			return float('inf')
		
		return self.current_stock / daily_usage
	
	def get_reorder_quantity(self) -> int:
		"""Calculate optimal reorder quantity"""
		if self.economic_order_quantity > 0:
			return self.economic_order_quantity
		
		# Simple reorder calculation
		shortage = max(0, self.maximum_stock - self.current_stock)
		return max(shortage, self.minimum_order_quantity)
	
	def update_average_cost(self, new_cost: Decimal, quantity: int):
		"""Update average cost with new receipt"""
		if self.current_stock <= 0:
			self.average_cost = new_cost
		else:
			total_value = (self.average_cost * self.current_stock) + (new_cost * quantity)
			total_quantity = self.current_stock + quantity
			self.average_cost = total_value / total_quantity


class EAContract(Model, AuditMixin, BaseMixin):
	"""
	Contract Management - Service contracts and agreements.
	
	Manages maintenance contracts, service agreements, and warranties
	with integration to APG CRM and Document Management systems.
	"""
	__tablename__ = 'ea_contract'
	
	# Identity
	contract_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Contract Information
	contract_number = Column(String(50), nullable=False, index=True)
	contract_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	contract_type = Column(String(50), nullable=False, index=True)  # maintenance, service, warranty, lease, support
	
	# Parties and Relationships
	contractor_id = Column(String(36), nullable=False, index=True)  # Service provider
	customer_contact_id = Column(String(36), nullable=True)  # Internal contact
	contractor_contact_id = Column(String(36), nullable=True)  # External contact
	
	# Contract Terms
	start_date = Column(Date, nullable=False, index=True)
	end_date = Column(Date, nullable=False, index=True)
	duration_months = Column(Integer, nullable=True)
	auto_renewal = Column(Boolean, default=False)
	renewal_notice_days = Column(Integer, default=30)
	
	# Financial Terms
	contract_value = Column(DECIMAL(15, 2), nullable=True)
	currency_code = Column(String(3), default='USD')
	payment_terms = Column(String(50), nullable=True)  # net_30, net_60, etc.
	billing_frequency = Column(String(20), nullable=True)  # monthly, quarterly, annually
	
	# Service Level Agreements
	response_time_hours = Column(Float, nullable=True)  # Guaranteed response time
	resolution_time_hours = Column(Float, nullable=True)  # Guaranteed resolution time
	availability_target = Column(DECIMAL(5, 2), nullable=True)  # Target availability %
	penalty_clauses = Column(JSON, default=list)  # SLA penalties
	
	# Scope and Coverage
	service_scope = Column(Text, nullable=True)
	inclusions = Column(JSON, default=list)  # What's included
	exclusions = Column(JSON, default=list)  # What's excluded
	coverage_hours = Column(String(50), nullable=True)  # 24x7, business_hours, etc.
	emergency_support = Column(Boolean, default=False)
	
	# Performance Metrics
	performance_kpis = Column(JSON, default=list)  # Key performance indicators
	reporting_requirements = Column(JSON, default=list)  # Required reports
	review_frequency = Column(String(20), nullable=True)  # monthly, quarterly, annually
	
	# Status and Lifecycle
	status = Column(String(20), default='draft', index=True)  # draft, active, suspended, expired, terminated
	approval_status = Column(String(20), default='pending')  # pending, approved, rejected
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	
	# Documentation
	contract_document_url = Column(String(500), nullable=True)
	amendments = Column(JSON, default=list)  # Contract amendments
	related_documents = Column(JSON, default=list)  # Related document URLs
	
	# Renewal and Termination
	renewal_options = Column(JSON, default=list)  # Renewal terms
	termination_clauses = Column(JSON, default=list)  # Termination conditions
	notice_of_termination_date = Column(Date, nullable=True)
	early_termination_penalty = Column(DECIMAL(15, 2), nullable=True)
	
	# Performance Tracking
	actual_response_time = Column(Float, nullable=True)  # Average response time
	actual_resolution_time = Column(Float, nullable=True)  # Average resolution time
	actual_availability = Column(DECIMAL(5, 2), nullable=True)  # Actual availability
	customer_satisfaction_score = Column(Float, nullable=True)  # 1-5 rating
	
	# Financial Tracking
	invoiced_amount_ytd = Column(DECIMAL(15, 2), default=0.00)
	paid_amount_ytd = Column(DECIMAL(15, 2), default=0.00)
	outstanding_amount = Column(DECIMAL(15, 2), default=0.00)
	budget_variance = Column(DECIMAL(15, 2), default=0.00)
	
	# Constraints and Indexes
	__table_args__ = (
		UniqueConstraint('tenant_id', 'contract_number', name='uq_ea_contract_number_tenant'),
		Index('idx_ea_contract_dates', 'start_date', 'end_date'),
		Index('idx_ea_contract_contractor', 'contractor_id', 'status'),
		Index('idx_ea_contract_type_status', 'contract_type', 'status'),
		Index('idx_ea_contract_expiry', 'end_date', 'auto_renewal'),
	)
	
	# Relationships
	assets = relationship("EAAsset", secondary="ea_asset_contract", back_populates="contracts")
	
	def __repr__(self):
		return f"<EAContract {self.contract_number} - {self.contract_name}>"
	
	async def _log_contract_change(self, change_type: str, details: str) -> str:
		"""Log contract changes for APG audit compliance"""
		return f"Contract {self.contract_number} {change_type}: {details}"
	
	def is_active(self) -> bool:
		"""Check if contract is currently active"""
		if self.status != 'active':
			return False
		
		today = date.today()
		return self.start_date <= today <= self.end_date
	
	def is_expiring_soon(self, days_ahead: int = 90) -> bool:
		"""Check if contract is expiring within specified days"""
		if not self.end_date:
			return False
		
		days_until_expiry = (self.end_date - date.today()).days
		return 0 <= days_until_expiry <= days_ahead
	
	def calculate_remaining_days(self) -> int:
		"""Calculate remaining days on contract"""
		if not self.end_date:
			return 0
		
		remaining = (self.end_date - date.today()).days
		return max(0, remaining)
	
	def calculate_utilization(self) -> float:
		"""Calculate contract utilization percentage"""
		if not self.contract_value or self.contract_value <= 0:
			return 0.0
		
		return float((self.invoiced_amount_ytd / self.contract_value) * 100)
	
	def get_renewal_notification_date(self) -> date:
		"""Get date when renewal notification should be sent"""
		if not self.end_date or not self.renewal_notice_days:
			return self.end_date
		
		return self.end_date - timedelta(days=self.renewal_notice_days)


class EAPerformanceRecord(Model, AuditMixin, BaseMixin):
	"""
	Performance Analytics - Asset performance tracking and KPIs.
	
	Comprehensive performance analytics with integration to APG
	AI Orchestration for predictive insights and optimization.
	"""
	__tablename__ = 'ea_performance_record'
	
	# Identity
	record_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Asset Reference
	asset_id = Column(String(36), ForeignKey('ea_asset.asset_id'), nullable=False, index=True)
	
	# Time Period
	measurement_date = Column(Date, nullable=False, index=True)
	measurement_period = Column(String(20), nullable=False, index=True)  # daily, weekly, monthly, quarterly, yearly
	period_start = Column(DateTime, nullable=False)
	period_end = Column(DateTime, nullable=False)
	
	# Availability Metrics
	planned_uptime_hours = Column(Float, nullable=True)
	actual_uptime_hours = Column(Float, nullable=True)
	downtime_hours = Column(Float, nullable=True)
	availability_percentage = Column(DECIMAL(5, 2), nullable=True)
	
	# Reliability Metrics
	failure_count = Column(Integer, default=0)
	mean_time_between_failures = Column(Float, nullable=True)  # Hours
	mean_time_to_repair = Column(Float, nullable=True)  # Hours
	mean_time_to_failure = Column(Float, nullable=True)  # Hours
	
	# Performance Metrics
	performance_efficiency = Column(DECIMAL(5, 2), nullable=True)  # % of rated capacity
	throughput_actual = Column(Float, nullable=True)  # Actual output
	throughput_target = Column(Float, nullable=True)  # Target output
	throughput_variance = Column(DECIMAL(8, 2), nullable=True)  # Variance %
	
	# Quality Metrics
	quality_rate = Column(DECIMAL(5, 2), nullable=True)  # % good quality output
	defect_rate = Column(DECIMAL(5, 2), nullable=True)  # % defective output
	rework_rate = Column(DECIMAL(5, 2), nullable=True)  # % requiring rework
	first_pass_yield = Column(DECIMAL(5, 2), nullable=True)  # % first pass success
	
	# Overall Equipment Effectiveness (OEE)
	oee_availability = Column(DECIMAL(5, 2), nullable=True)
	oee_performance = Column(DECIMAL(5, 2), nullable=True)
	oee_quality = Column(DECIMAL(5, 2), nullable=True)
	oee_overall = Column(DECIMAL(5, 2), nullable=True)
	
	# Cost Metrics
	operating_cost = Column(DECIMAL(15, 2), nullable=True)
	maintenance_cost = Column(DECIMAL(15, 2), nullable=True)
	energy_cost = Column(DECIMAL(15, 2), nullable=True)
	total_cost = Column(DECIMAL(15, 2), nullable=True)
	cost_per_unit = Column(DECIMAL(10, 4), nullable=True)
	
	# Energy and Sustainability
	energy_consumption = Column(Float, nullable=True)  # kWh
	energy_efficiency = Column(DECIMAL(5, 2), nullable=True)  # % efficiency
	co2_emissions = Column(Float, nullable=True)  # kg CO2
	water_consumption = Column(Float, nullable=True)  # Liters
	waste_generated = Column(Float, nullable=True)  # kg
	
	# Condition and Health
	health_score = Column(Float, nullable=True, index=True)  # 0-100
	condition_score = Column(Float, nullable=True)  # 0-100
	vibration_level = Column(Float, nullable=True)  # RMS value
	temperature_average = Column(Float, nullable=True)  # Celsius
	temperature_max = Column(Float, nullable=True)  # Celsius
	
	# Maintenance Metrics
	maintenance_hours = Column(Float, nullable=True)
	preventive_maintenance_compliance = Column(DECIMAL(5, 2), nullable=True)  # %
	corrective_maintenance_ratio = Column(DECIMAL(5, 2), nullable=True)  # %
	maintenance_effectiveness = Column(Float, nullable=True)  # 0-100 score
	
	# Operator and Human Factors
	operator_efficiency = Column(DECIMAL(5, 2), nullable=True)  # %
	training_compliance = Column(DECIMAL(5, 2), nullable=True)  # %
	safety_incidents = Column(Integer, default=0)
	near_miss_incidents = Column(Integer, default=0)
	
	# Benchmarking and Targets
	industry_benchmark = Column(Float, nullable=True)  # Industry average
	target_performance = Column(Float, nullable=True)  # Target value
	performance_gap = Column(Float, nullable=True)  # Gap to target
	improvement_potential = Column(Float, nullable=True)  # Potential improvement
	
	# Trend Analysis
	trend_direction = Column(String(20), nullable=True, index=True)  # improving, stable, declining
	trend_magnitude = Column(Float, nullable=True)  # Rate of change
	seasonal_factor = Column(Float, nullable=True)  # Seasonal adjustment
	
	# External Factors
	weather_impact = Column(Float, nullable=True)  # Weather impact score
	market_conditions = Column(String(50), nullable=True)  # Market impact
	regulatory_changes = Column(JSON, default=list)  # Regulatory impacts
	
	# Raw Data and Analytics
	raw_sensor_data = Column(JSON, default=dict)  # Raw sensor measurements
	calculated_metrics = Column(JSON, default=dict)  # Derived metrics
	statistical_summary = Column(JSON, default=dict)  # Statistical analysis
	
	# Alerts and Thresholds
	threshold_violations = Column(JSON, default=list)  # Threshold breaches
	alert_count = Column(Integer, default=0)
	critical_alerts = Column(Integer, default=0)
	
	# Constraints and Indexes
	__table_args__ = (
		Index('idx_ea_performance_asset_date', 'asset_id', 'measurement_date'),
		Index('idx_ea_performance_period', 'measurement_period', 'measurement_date'),
		Index('idx_ea_performance_oee', 'oee_overall', 'measurement_date'),
		Index('idx_ea_performance_health', 'health_score', 'measurement_date'),
		Index('idx_ea_performance_trend', 'trend_direction', 'measurement_date'),
	)
	
	# Relationships
	asset = relationship("EAAsset", back_populates="performance_records")
	
	def __repr__(self):
		return f"<EAPerformanceRecord {self.asset.asset_number if self.asset else 'N/A'} - {self.measurement_date}>"
	
	def calculate_oee(self):
		"""Calculate Overall Equipment Effectiveness"""
		if all([self.oee_availability, self.oee_performance, self.oee_quality]):
			self.oee_overall = (
				self.oee_availability * 
				self.oee_performance * 
				self.oee_quality
			) / 10000  # Convert from percentage
	
	def calculate_availability(self):
		"""Calculate availability percentage"""
		if self.planned_uptime_hours and self.planned_uptime_hours > 0:
			uptime = self.actual_uptime_hours or 0
			self.availability_percentage = Decimal(
				str((uptime / self.planned_uptime_hours) * 100)
			)
	
	def calculate_mtbf(self):
		"""Calculate Mean Time Between Failures"""
		if self.failure_count and self.failure_count > 0 and self.actual_uptime_hours:
			self.mean_time_between_failures = self.actual_uptime_hours / self.failure_count
	
	def get_performance_grade(self) -> str:
		"""Get performance grade based on OEE"""
		if not self.oee_overall:
			return 'Unknown'
		elif self.oee_overall >= 85:
			return 'World Class'
		elif self.oee_overall >= 70:
			return 'Good'
		elif self.oee_overall >= 50:
			return 'Fair'
		else:
			return 'Poor'


# Association Tables for Many-to-Many Relationships
class EAAssetContract(Model):
	"""Association table for Asset-Contract many-to-many relationship"""
	__tablename__ = 'ea_asset_contract'
	
	asset_id = Column(String(36), ForeignKey('ea_asset.asset_id'), primary_key=True)
	contract_id = Column(String(36), ForeignKey('ea_contract.contract_id'), primary_key=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Relationship Details
	coverage_start_date = Column(Date, nullable=True)
	coverage_end_date = Column(Date, nullable=True)
	coverage_type = Column(String(50), nullable=True)  # full, partial, emergency_only
	priority_level = Column(String(20), default='normal')  # high, normal, low
	
	# Constraints
	__table_args__ = (
		Index('idx_ea_asset_contract_tenant', 'tenant_id'),
	)