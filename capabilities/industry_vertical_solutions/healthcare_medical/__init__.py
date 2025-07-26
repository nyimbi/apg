"""
APG Healthcare Medical Industry Capability

Comprehensive healthcare and medical industry capabilities including
HIPAA compliance, patient management, medical device integration,
and healthcare-specific workflows.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import List, Dict, Any
from enum import Enum

# Healthcare Medical Metadata
__version__ = "1.0.0"
__capability_id__ = "healthcare_medical"
__description__ = "Healthcare and medical industry management platform"

class HCComplianceFramework(str, Enum):
	"""Healthcare compliance frameworks."""
	HIPAA = "hipaa"
	HITECH = "hitech"
	FDA_21CFR = "fda_21cfr"
	GDPR_HEALTH = "gdpr_health"
	SOX_HEALTHCARE = "sox_healthcare"
	ICD_10 = "icd_10"
	CPT = "cpt"
	SNOMED_CT = "snomed_ct"

class HCFacilityType(str, Enum):
	"""Types of healthcare facilities."""
	HOSPITAL = "hospital"
	CLINIC = "clinic"
	PHARMACY = "pharmacy"
	LABORATORY = "laboratory"
	IMAGING_CENTER = "imaging_center"
	SURGERY_CENTER = "surgery_center"
	REHABILITATION = "rehabilitation"
	LONG_TERM_CARE = "long_term_care"

# Sub-capability Registry
SUBCAPABILITIES = [
	"patient_management",
	"medical_records_management",
	"compliance_privacy_management",
	"medical_device_integration",
	"healthcare_analytics",
	"telemedicine_platform",
	"pharmacy_management",
	"laboratory_information_system"
]

# APG Composition Engine Registration
CAPABILITY_METADATA = {
	"capability_id": "industry_vertical_solutions.healthcare_medical",
	"version": __version__,
	"category": "healthcare_industry",
	"subcapabilities": SUBCAPABILITIES,
	"dependencies": [
		"auth_rbac",
		"audit_compliance",
		"general_cross_functional.governance_risk_compliance",
		"general_cross_functional.document_content_management",
		"core_business_operations.financial_management"
	],
	"provides_services": [
		"patient_lifecycle_management",
		"medical_records_system",
		"hipaa_compliance_management",
		"medical_device_monitoring",
		"healthcare_reporting_analytics",
		"telemedicine_services"
	],
	"compliance_frameworks": [
		HCComplianceFramework.HIPAA,
		HCComplianceFramework.HITECH,
		HCComplianceFramework.FDA_21CFR
	],
	"data_models": ["HCPatient", "HCMedicalRecord", "HCDevice", "HCProvider", "HCFacility"],
	"integration_points": [
		"hl7_fhir",
		"epic_integration",
		"cerner_integration", 
		"medical_device_apis",
		"pharmacy_systems"
	]
}

def get_capability_info() -> Dict[str, Any]:
	"""Get healthcare medical capability information."""
	return CAPABILITY_METADATA

def list_subcapabilities() -> List[str]:
	"""List all available subcapabilities."""
	return SUBCAPABILITIES.copy()

def get_compliance_frameworks() -> List[HCComplianceFramework]:
	"""Get supported compliance frameworks."""
	return list(HCComplianceFramework)

def get_facility_types() -> List[HCFacilityType]:
	"""Get supported facility types."""
	return list(HCFacilityType)

__all__ = [
	"HCComplianceFramework",
	"HCFacilityType",
	"SUBCAPABILITIES",
	"CAPABILITY_METADATA",
	"get_capability_info",
	"list_subcapabilities",
	"get_compliance_frameworks",
	"get_facility_types"
]