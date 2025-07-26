"""
Contract Management Sub-capability

Comprehensive contract lifecycle management including contract creation,
amendment tracking, renewal management, and compliance monitoring.
"""

from typing import Dict, Any

# Sub-capability metadata
SUBCAPABILITY_INFO = {
	'code': 'PPC',
	'name': 'Contract Management',
	'description': 'Contract lifecycle, terms, compliance, and renewals',
	'version': '1.0.0',
	'models': [
		'PPCContract', 'PPCContractLine', 'PPCContractAmendment',
		'PPCContractRenewal', 'PPCContractMilestone', 'PPCContractDocument'
	]
}

def get_subcapability_info() -> Dict[str, Any]:
	return SUBCAPABILITY_INFO.copy()