"""
Industry Templates Manager

Provides industry-specific templates and pre-configured compositions for quick
deployment of ERP solutions tailored to specific business domains.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict, validator
from uuid_extensions import uuid7str

logger = logging.getLogger(__name__)

class IndustryType(Enum):
	"""Supported industry types."""
	MANUFACTURING = "manufacturing"
	PHARMACEUTICAL = "pharmaceutical"
	SERVICE_COMPANY = "service_company"
	RETAIL = "retail"
	HEALTHCARE = "healthcare"
	AUTOMOTIVE = "automotive"
	AEROSPACE = "aerospace"
	FOOD_BEVERAGE = "food_beverage"
	CHEMICALS = "chemicals"
	TECHNOLOGY = "technology"
	CONSULTING = "consulting"
	CONSTRUCTION = "construction"
	ENERGY = "energy"
	MINING = "mining"
	LOGISTICS = "logistics"
	EDUCATION = "education"
	GOVERNMENT = "government"
	NON_PROFIT = "non_profit"

class TemplateSize(Enum):
	"""Template deployment sizes."""
	SMALL = "small"  # <100 users
	MEDIUM = "medium"  # 100-1000 users
	LARGE = "large"  # 1000-10000 users
	ENTERPRISE = "enterprise"  # 10000+ users

class ComplianceFramework(Enum):
	"""Compliance frameworks supported."""
	FDA_CFR_21 = "fda_cfr_21"  # FDA 21 CFR Part 11
	ISO_13485 = "iso_13485"  # Medical devices
	ISO_9001 = "iso_9001"  # Quality management
	ISO_14001 = "iso_14001"  # Environmental management
	SOX = "sox"  # Sarbanes-Oxley
	GDPR = "gdpr"  # General Data Protection Regulation
	HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
	GMP = "gmp"  # Good Manufacturing Practice
	HACCP = "haccp"  # Hazard Analysis Critical Control Points
	ITAR = "itar"  # International Traffic in Arms Regulations

class TemplateConfiguration(BaseModel):
	"""Configuration for an industry template."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	# Core configuration
	default_capabilities: list[str] = Field(..., description="Default capabilities to include")
	required_capabilities: list[str] = Field(default_factory=list, description="Mandatory capabilities")
	optional_capabilities: list[str] = Field(default_factory=list, description="Optional capabilities")
	excluded_capabilities: list[str] = Field(default_factory=list, description="Capabilities to exclude")
	
	# Sub-capability configuration
	capability_subcapabilities: dict[str, list[str]] = Field(default_factory=dict, description="Sub-capabilities per capability")
	
	# Configuration overrides
	configuration_overrides: dict[str, Any] = Field(default_factory=dict, description="Configuration overrides")
	security_settings: dict[str, Any] = Field(default_factory=dict, description="Security settings")
	workflow_settings: dict[str, Any] = Field(default_factory=dict, description="Workflow settings")
	
	# UI/UX customization
	branding: dict[str, Any] = Field(default_factory=dict, description="Branding configuration")
	menu_structure: dict[str, Any] = Field(default_factory=dict, description="Custom menu structure")
	dashboard_layout: dict[str, Any] = Field(default_factory=dict, description="Dashboard layout")
	
	# Performance tuning
	performance_settings: dict[str, Any] = Field(default_factory=dict, description="Performance settings")
	caching_strategy: dict[str, Any] = Field(default_factory=dict, description="Caching configuration")
	
	# Integration settings
	default_integrations: list[str] = Field(default_factory=list, description="Default third-party integrations")
	api_settings: dict[str, Any] = Field(default_factory=dict, description="API configuration")

class IndustryTemplate(BaseModel):
	"""Industry-specific template definition."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	# Template identity
	id: str = Field(default_factory=uuid7str, description="Unique template ID")
	code: str = Field(..., description="Template code")
	name: str = Field(..., description="Template display name")
	description: str = Field(..., description="Template description")
	version: str = Field(default="1.0.0", description="Template version")
	
	# Industry classification
	industry_type: IndustryType = Field(..., description="Primary industry type")
	sub_industries: list[str] = Field(default_factory=list, description="Sub-industry classifications")
	business_model: str = Field(default="", description="Business model (B2B, B2C, etc.)")
	
	# Deployment characteristics
	template_size: TemplateSize = Field(default=TemplateSize.MEDIUM, description="Recommended deployment size")
	complexity_level: str = Field(default="medium", description="Implementation complexity")
	deployment_time_days: int = Field(default=30, description="Estimated deployment time")
	
	# Compliance and regulations
	compliance_frameworks: list[ComplianceFramework] = Field(default_factory=list, description="Supported compliance frameworks")
	regulatory_requirements: list[str] = Field(default_factory=list, description="Regulatory requirements")
	audit_requirements: dict[str, Any] = Field(default_factory=dict, description="Audit requirements")
	
	# Configuration
	configuration: TemplateConfiguration = Field(..., description="Template configuration")
	
	# Business processes
	core_processes: list[str] = Field(default_factory=list, description="Core business processes")
	supporting_processes: list[str] = Field(default_factory=list, description="Supporting processes")
	kpi_metrics: list[str] = Field(default_factory=list, description="Key performance indicators")
	
	# Metadata
	created_by: str = Field(default="system", description="Template creator")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	tags: list[str] = Field(default_factory=list, description="Template tags")
	
	# Template assets
	documentation_urls: list[str] = Field(default_factory=list, description="Documentation URLs")
	training_materials: list[str] = Field(default_factory=list, description="Training material URLs")
	best_practices: list[str] = Field(default_factory=list, description="Best practices guides")

class TemplateManager:
	"""
	Manager for industry-specific templates and pre-configured compositions.
	
	Provides templates for rapid deployment of industry-specific ERP solutions
	with optimized configurations, compliance settings, and business processes.
	"""
	
	def __init__(self):
		"""Initialize the template manager."""
		self.templates: Dict[str, IndustryTemplate] = {}
		self._load_built_in_templates()
		
		logger.info("TemplateManager initialized")
	
	def _load_built_in_templates(self) -> None:
		"""Load built-in industry templates."""
		self.templates.update({
			"manufacturing_erp": self._create_manufacturing_template(),
			"pharmaceutical_erp": self._create_pharmaceutical_template(),
			"service_company_erp": self._create_service_company_template(),
			"retail_erp": self._create_retail_template(),
			"automotive_erp": self._create_automotive_template(),
			"aerospace_erp": self._create_aerospace_template(),
			"food_beverage_erp": self._create_food_beverage_template(),
			"healthcare_erp": self._create_healthcare_template(),
			"technology_erp": self._create_technology_template(),
			"consulting_erp": self._create_consulting_template(),
			"construction_erp": self._create_construction_template(),
			"energy_erp": self._create_energy_template(),
			"mining_erp": self._create_mining_template(),
		})
	
	def _create_manufacturing_template(self) -> IndustryTemplate:
		"""Create manufacturing ERP template."""
		return IndustryTemplate(
			code="manufacturing_erp",
			name="Manufacturing ERP",
			description="Comprehensive ERP solution for discrete and process manufacturing companies",
			industry_type=IndustryType.MANUFACTURING,
			sub_industries=["discrete_manufacturing", "process_manufacturing", "make_to_order", "make_to_stock"],
			business_model="B2B",
			template_size=TemplateSize.LARGE,
			complexity_level="high",
			deployment_time_days=45,
			compliance_frameworks=[ComplianceFramework.ISO_9001, ComplianceFramework.ISO_14001],
			core_processes=[
				"production_planning",
				"material_requirements_planning",
				"shop_floor_control",
				"quality_management",
				"inventory_management",
				"procurement",
				"financial_management"
			],
			configuration=TemplateConfiguration(
				default_capabilities=[
					"MANUFACTURING",
					"INVENTORY_MANAGEMENT", 
					"PROCUREMENT_PURCHASING",
					"CORE_FINANCIALS",
					"SUPPLY_CHAIN_MANAGEMENT",
					"HR",
					"AUTH_RBAC"
				],
				required_capabilities=[
					"MANUFACTURING",
					"INVENTORY_MANAGEMENT",
					"CORE_FINANCIALS"
				],
				capability_subcapabilities={
					"MANUFACTURING": [
						"production_planning",
						"material_requirements_planning",
						"manufacturing_execution_system",
						"shop_floor_control",
						"quality_management",
						"bill_of_materials"
					],
					"INVENTORY_MANAGEMENT": [
						"stock_tracking_control",
						"batch_lot_tracking",
						"replenishment_reordering"
					],
					"PROCUREMENT_PURCHASING": [
						"purchase_order_management",
						"vendor_management",
						"sourcing_supplier_selection"
					]
				},
				configuration_overrides={
					"manufacturing": {
						"enable_work_orders": True,
						"enable_routing": True,
						"enable_capacity_planning": True,
						"quality_control_points": "all_operations"
					},
					"inventory": {
						"enable_lot_tracking": True,
						"enable_serial_numbers": True,
						"reorder_point_calculation": "dynamic"
					}
				},
				security_settings={
					"require_approval_workflows": True,
					"enable_electronic_signatures": True,
					"audit_trail_level": "detailed"
				}
			),
			kpi_metrics=[
				"overall_equipment_effectiveness",
				"production_efficiency",
				"quality_yield",
				"inventory_turnover",
				"on_time_delivery"
			]
		)
	
	def _create_pharmaceutical_template(self) -> IndustryTemplate:
		"""Create pharmaceutical ERP template."""
		return IndustryTemplate(
			code="pharmaceutical_erp",
			name="Pharmaceutical ERP",
			description="GMP-compliant ERP solution for pharmaceutical and life sciences companies",
			industry_type=IndustryType.PHARMACEUTICAL,
			sub_industries=["drug_manufacturing", "medical_devices", "biotechnology", "clinical_research"],
			business_model="B2B",
			template_size=TemplateSize.ENTERPRISE,
			complexity_level="high",
			deployment_time_days=60,
			compliance_frameworks=[
				ComplianceFramework.FDA_CFR_21,
				ComplianceFramework.GMP,
				ComplianceFramework.ISO_13485
			],
			regulatory_requirements=[
				"21_cfr_part_11_compliance",
				"batch_record_integrity",
				"electronic_signatures",
				"audit_trails",
				"validation_documentation"
			],
			core_processes=[
				"batch_manufacturing",
				"quality_control",
				"regulatory_compliance",
				"clinical_trial_management",
				"batch_genealogy",
				"deviation_management"
			],
			configuration=TemplateConfiguration(
				default_capabilities=[
					"PHARMACEUTICAL_SPECIFIC",
					"MANUFACTURING",
					"INVENTORY_MANAGEMENT",
					"CORE_FINANCIALS",
					"AUDIT_COMPLIANCE",
					"AUTH_RBAC"
				],
				required_capabilities=[
					"PHARMACEUTICAL_SPECIFIC",
					"AUDIT_COMPLIANCE",
					"AUTH_RBAC"
				],
				capability_subcapabilities={
					"PHARMACEUTICAL_SPECIFIC": [
						"regulatory_compliance"
					],
					"MANUFACTURING": [
						"recipe_formula_management",
						"quality_management",
						"batch_lot_tracking"
					]
				},
				configuration_overrides={
					"pharmaceutical": {
						"enable_21_cfr_part_11": True,
						"electronic_signatures_required": True,
						"batch_genealogy_tracking": True,
						"deviation_workflow": True
					},
					"security": {
						"password_complexity": "high",
						"session_timeout_minutes": 30,
						"mfa_required": True
					}
				}
			),
			kpi_metrics=[
				"batch_cycle_time",
				"first_pass_yield",
				"regulatory_compliance_score",
				"deviation_resolution_time",
				"audit_readiness"
			]
		)
	
	def _create_service_company_template(self) -> IndustryTemplate:
		"""Create service company ERP template."""
		return IndustryTemplate(
			code="service_company_erp",
			name="Service Company ERP",
			description="ERP solution optimized for professional services and consulting companies",
			industry_type=IndustryType.SERVICE_COMPANY,
			sub_industries=["professional_services", "consulting", "maintenance_services", "field_services"],
			business_model="B2B",
			template_size=TemplateSize.MEDIUM,
			complexity_level="medium",
			deployment_time_days=30,
			core_processes=[
				"project_management",
				"time_tracking",
				"resource_scheduling",
				"client_management",
				"billing_invoicing",
				"service_delivery"
			],
			configuration=TemplateConfiguration(
				default_capabilities=[
					"SERVICE_SPECIFIC",
					"CORE_FINANCIALS",
					"HR",
					"GENERAL_CROSS_FUNCTIONAL",
					"AUTH_RBAC"
				],
				required_capabilities=[
					"SERVICE_SPECIFIC",
					"CORE_FINANCIALS"
				],
				capability_subcapabilities={
					"SERVICE_SPECIFIC": [
						"project_management",
						"time_expense_tracking",
						"resource_scheduling",
						"field_service_management",
						"professional_services_automation"
					],
					"GENERAL_CROSS_FUNCTIONAL": [
						"customer_relationship_management"
					]
				},
				configuration_overrides={
					"services": {
						"enable_project_billing": True,
						"time_tracking_method": "detailed",
						"resource_utilization_tracking": True,
						"mobile_field_service": True
					}
				}
			),
			kpi_metrics=[
				"resource_utilization",
				"project_profitability",
				"client_satisfaction",
				"project_delivery_time",
				"billable_hours_ratio"
			]
		)
	
	def _create_retail_template(self) -> IndustryTemplate:
		"""Create retail ERP template."""
		return IndustryTemplate(
			code="retail_erp",
			name="Retail ERP",
			description="Multi-channel retail ERP with e-commerce integration",
			industry_type=IndustryType.RETAIL,
			sub_industries=["fashion_retail", "electronics_retail", "grocery_retail", "specialty_retail"],
			business_model="B2C",
			template_size=TemplateSize.LARGE,
			configuration=TemplateConfiguration(
				default_capabilities=[
					"SALES_ORDER_MANAGEMENT",
					"INVENTORY_MANAGEMENT",
					"PLATFORM_SERVICES",
					"CORE_FINANCIALS",
					"GENERAL_CROSS_FUNCTIONAL",
					"AUTH_RBAC"
				],
				capability_subcapabilities={
					"PLATFORM_SERVICES": [
						"digital_storefront_management",
						"product_catalog_management",
						"payment_gateway_integration"
					],
					"SALES_ORDER_MANAGEMENT": [
						"order_entry",
						"order_processing",
						"pricing_discounts"
					]
				},
				configuration_overrides={
					"retail": {
						"multi_channel_inventory": True,
						"real_time_pricing": True,
						"promotion_engine": True,
						"omnichannel_fulfillment": True
					}
				}
			),
			kpi_metrics=[
				"inventory_turnover",
				"same_store_sales_growth",
				"gross_margin",
				"customer_acquisition_cost",
				"order_fulfillment_time"
			]
		)
	
	def _create_automotive_template(self) -> IndustryTemplate:
		"""Create automotive ERP template."""
		return IndustryTemplate(
			code="automotive_erp",
			name="Automotive ERP",
			description="ERP solution for automotive manufacturers and suppliers",
			industry_type=IndustryType.AUTOMOTIVE,
			compliance_frameworks=[ComplianceFramework.ISO_9001, ComplianceFramework.ITAR],
			configuration=TemplateConfiguration(
				default_capabilities=[
					"MANUFACTURING",
					"INVENTORY_MANAGEMENT",
					"PROCUREMENT_PURCHASING",
					"SUPPLY_CHAIN_MANAGEMENT",
					"CORE_FINANCIALS",
					"AUTH_RBAC"
				],
				configuration_overrides={
					"automotive": {
						"supplier_portal": True,
						"just_in_time_delivery": True,
						"quality_ppap_tracking": True,
						"change_order_management": True
					}
				}
			)
		)
	
	def _create_aerospace_template(self) -> IndustryTemplate:
		"""Create aerospace ERP template."""
		return IndustryTemplate(
			code="aerospace_erp",
			name="Aerospace ERP",
			description="High-compliance ERP for aerospace and defense contractors",
			industry_type=IndustryType.AEROSPACE,
			compliance_frameworks=[ComplianceFramework.ITAR, ComplianceFramework.ISO_9001],
			template_size=TemplateSize.ENTERPRISE,
			complexity_level="high",
			configuration=TemplateConfiguration(
				default_capabilities=[
					"MANUFACTURING",
					"INVENTORY_MANAGEMENT",
					"PROCUREMENT_PURCHASING",
					"CORE_FINANCIALS",
					"AUDIT_COMPLIANCE",
					"AUTH_RBAC"
				],
				configuration_overrides={
					"aerospace": {
						"export_control_compliance": True,
						"serialized_inventory": True,
						"configuration_management": True,
						"as9100_compliance": True
					}
				}
			)
		)
	
	def _create_food_beverage_template(self) -> IndustryTemplate:
		"""Create food & beverage ERP template."""
		return IndustryTemplate(
			code="food_beverage_erp",
			name="Food & Beverage ERP",
			description="Food safety compliant ERP for food and beverage manufacturers",
			industry_type=IndustryType.FOOD_BEVERAGE,
			compliance_frameworks=[ComplianceFramework.HACCP, ComplianceFramework.ISO_9001],
			configuration=TemplateConfiguration(
				default_capabilities=[
					"MANUFACTURING",
					"INVENTORY_MANAGEMENT",
					"CORE_FINANCIALS",
					"AUTH_RBAC"
				],
				capability_subcapabilities={
					"INVENTORY_MANAGEMENT": [
						"batch_lot_tracking",
						"expiry_date_management"
					]
				},
				configuration_overrides={
					"food_safety": {
						"enable_haccp": True,
						"allergen_tracking": True,
						"nutritional_labeling": True,
						"expiry_date_tracking": True
					}
				}
			)
		)
	
	def _create_healthcare_template(self) -> IndustryTemplate:
		"""Create healthcare ERP template."""
		return IndustryTemplate(
			code="healthcare_erp",
			name="Healthcare ERP",
			description="HIPAA-compliant ERP for healthcare organizations",
			industry_type=IndustryType.HEALTHCARE,
			compliance_frameworks=[ComplianceFramework.HIPAA],
			configuration=TemplateConfiguration(
				default_capabilities=[
					"HR",
					"CORE_FINANCIALS",
					"INVENTORY_MANAGEMENT",
					"AUTH_RBAC",
					"AUDIT_COMPLIANCE"
				]
			)
		)
	
	def _create_technology_template(self) -> IndustryTemplate:
		"""Create technology company ERP template."""
		return IndustryTemplate(
			code="technology_erp",
			name="Technology ERP",
			description="Agile ERP solution for software and technology companies",
			industry_type=IndustryType.TECHNOLOGY,
			template_size=TemplateSize.MEDIUM,
			configuration=TemplateConfiguration(
				default_capabilities=[
					"SERVICE_SPECIFIC",
					"SALES_ORDER_MANAGEMENT",
					"CORE_FINANCIALS",
					"HR",
					"AUTH_RBAC"
				]
			)
		)
	
	def _create_consulting_template(self) -> IndustryTemplate:
		"""Create consulting ERP template."""
		return IndustryTemplate(
			code="consulting_erp",
			name="Consulting ERP",
			description="Project-based ERP for consulting firms",
			industry_type=IndustryType.CONSULTING,
			configuration=TemplateConfiguration(
				default_capabilities=[
					"SERVICE_SPECIFIC",
					"CORE_FINANCIALS",
					"HR",
					"GENERAL_CROSS_FUNCTIONAL",
					"AUTH_RBAC"
				]
			)
		)
	
	def _create_construction_template(self) -> IndustryTemplate:
		"""Create construction ERP template."""
		return IndustryTemplate(
			code="construction_erp",
			name="Construction ERP", 
			description="Project-centric ERP for construction companies",
			industry_type=IndustryType.CONSTRUCTION,
			configuration=TemplateConfiguration(
				default_capabilities=[
					"SERVICE_SPECIFIC",
					"INVENTORY_MANAGEMENT",
					"PROCUREMENT_PURCHASING",
					"CORE_FINANCIALS",
					"HR",
					"AUTH_RBAC"
				]
			)
		)
	
	def _create_energy_template(self) -> IndustryTemplate:
		"""Create energy ERP template."""
		return IndustryTemplate(
			code="energy_erp",
			name="Energy ERP",
			description="ERP solution for energy and utilities companies",
			industry_type=IndustryType.ENERGY,
			configuration=TemplateConfiguration(
				default_capabilities=[
					"MANUFACTURING",
					"INVENTORY_MANAGEMENT",
					"CORE_FINANCIALS",
					"PREDICTIVE_MAINTENANCE",
					"AUTH_RBAC"
				]
			)
		)
	
	def _create_mining_template(self) -> IndustryTemplate:
		"""Create mining ERP template."""
		return IndustryTemplate(
			code="mining_erp",
			name="Mining ERP",
			description="ERP solution for mining and extractive industries",
			industry_type=IndustryType.MINING,
			configuration=TemplateConfiguration(
				default_capabilities=[
					"MINING_SPECIFIC",
					"INVENTORY_MANAGEMENT",
					"CORE_FINANCIALS",
					"HR",
					"AUTH_RBAC"
				]
			)
		)
	
	def list_templates(self) -> List[IndustryTemplate]:
		"""Get all available templates."""
		return list(self.templates.values())
	
	def get_template(self, template_code: str) -> Optional[IndustryTemplate]:
		"""Get a specific template by code."""
		return self.templates.get(template_code)
	
	def get_templates_by_industry(self, industry_type: IndustryType) -> List[IndustryTemplate]:
		"""Get templates for a specific industry."""
		return [
			template for template in self.templates.values()
			if template.industry_type == industry_type
		]
	
	def get_templates_by_size(self, template_size: TemplateSize) -> List[IndustryTemplate]:
		"""Get templates for a specific deployment size."""
		return [
			template for template in self.templates.values()
			if template.template_size == template_size
		]
	
	def get_templates_by_compliance(self, compliance_framework: ComplianceFramework) -> List[IndustryTemplate]:
		"""Get templates supporting a specific compliance framework."""
		return [
			template for template in self.templates.values()
			if compliance_framework in template.compliance_frameworks
		]
	
	def search_templates(self, 
						industry: Optional[str] = None,
						size: Optional[str] = None,
						compliance: Optional[str] = None,
						keyword: Optional[str] = None) -> List[IndustryTemplate]:
		"""Search templates by various criteria."""
		results = list(self.templates.values())
		
		if industry:
			try:
				industry_enum = IndustryType(industry.lower())
				results = [t for t in results if t.industry_type == industry_enum]
			except ValueError:
				# Invalid industry type
				return []
		
		if size:
			try:
				size_enum = TemplateSize(size.lower())
				results = [t for t in results if t.template_size == size_enum]
			except ValueError:
				pass
		
		if compliance:
			try:
				compliance_enum = ComplianceFramework(compliance.lower())
				results = [t for t in results if compliance_enum in t.compliance_frameworks]
			except ValueError:
				pass
		
		if keyword:
			keyword_lower = keyword.lower()
			results = [
				t for t in results
				if (keyword_lower in t.name.lower() or 
					keyword_lower in t.description.lower() or
					any(keyword_lower in tag.lower() for tag in t.tags))
			]
		
		return results
	
	def create_custom_template(self, 
							  base_template_code: str,
							  custom_name: str,
							  custom_config: TemplateConfiguration) -> IndustryTemplate:
		"""Create a custom template based on an existing template."""
		base_template = self.get_template(base_template_code)
		if not base_template:
			raise ValueError(f"Base template '{base_template_code}' not found")
		
		# Create new template based on base template
		custom_template = IndustryTemplate(
			code=f"custom_{uuid7str()[:8]}",
			name=custom_name,
			description=f"Custom template based on {base_template.name}",
			industry_type=base_template.industry_type,
			template_size=base_template.template_size,
			complexity_level=base_template.complexity_level,
			compliance_frameworks=base_template.compliance_frameworks.copy(),
			configuration=custom_config,
			tags=["custom"] + base_template.tags
		)
		
		return custom_template
	
	def validate_template_configuration(self, template: IndustryTemplate) -> Dict[str, Any]:
		"""Validate a template configuration."""
		validation_result = {
			"valid": True,
			"errors": [],
			"warnings": []
		}
		
		# Check that required capabilities are in default capabilities
		for req_cap in template.configuration.required_capabilities:
			if req_cap not in template.configuration.default_capabilities:
				validation_result["errors"].append(
					f"Required capability '{req_cap}' not in default capabilities"
				)
				validation_result["valid"] = False
		
		# Check for excluded capabilities in defaults
		for excl_cap in template.configuration.excluded_capabilities:
			if excl_cap in template.configuration.default_capabilities:
				validation_result["errors"].append(
					f"Excluded capability '{excl_cap}' found in default capabilities"
				)
				validation_result["valid"] = False
		
		# Warn about complexity vs size mismatches
		if template.complexity_level == "high" and template.template_size == TemplateSize.SMALL:
			validation_result["warnings"].append(
				"High complexity template may not be suitable for small deployments"
			)
		
		return validation_result

# Global template manager instance
_template_manager_instance: Optional[TemplateManager] = None

def get_template_manager() -> TemplateManager:
	"""Get the global template manager instance."""
	global _template_manager_instance
	if _template_manager_instance is None:
		_template_manager_instance = TemplateManager()
	return _template_manager_instance