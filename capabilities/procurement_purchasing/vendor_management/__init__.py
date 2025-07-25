"""
Vendor Management Sub-capability

Comprehensive vendor master data management, performance tracking,
and vendor relationship management.
"""

from typing import Dict, Any

# Sub-capability metadata
SUBCAPABILITY_INFO = {
	'code': 'PPV',
	'name': 'Vendor Management',
	'description': 'Vendor master data, performance tracking, and relationships',
	'version': '1.0.0',
	'models': [
		'PPVVendor', 'PPVVendorContact', 'PPVVendorPerformance',
		'PPVVendorCategory', 'PPVVendorQualification', 'PPVVendorInsurance'
	]
}

def get_subcapability_info() -> Dict[str, Any]:
	return SUBCAPABILITY_INFO.copy()
