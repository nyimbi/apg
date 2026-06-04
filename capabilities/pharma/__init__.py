"""
Pharmaceutical & Life Sciences Capabilities

Industry-specific ERP functionality for pharmaceutical and life sciences companies.
Manages regulatory compliance, clinical trials, R&D, product serialization, and batch release processes.
"""

from typing import Dict, List, Any

# Sub-capability IDs
CAPABILITY_IDS = [
	"pharma_com",
	"pharma_ctr",
	"pharma_dis",
	"pharma_mfg",
	"pharma_pvi",
	"pharma_qms",
	"pharma_rec",
	"pharma_reg",
	"pharma_sup",
]

# Capability metadata (legacy, kept for backward compatibility)
CAPABILITY_META = {
	'name': 'Pharmaceutical Specific',
	'code': 'PH',
	'version': '2.0.0',
	'description': 'Industry-specific ERP functionality for pharmaceutical and life sciences companies',
	'industry_focus': 'Pharmaceutical',
	'regulatory_frameworks': ['FDA', 'EMA', 'GMP', 'GxP', '21 CFR Part 11', 'ICH'],
	'subcapabilities': CAPABILITY_IDS,
	'database_tables_prefix': 'ph_',
	'api_prefix': '/api/pharmaceutical',
	'permissions_prefix': 'ph.',
}


def get_capability_ids() -> List[str]:
	"""Return all pharma sub-capability IDs."""
	return list(CAPABILITY_IDS)


def get_capability_info() -> Dict[str, Any]:
	"""Get pharmaceutical capability information."""
	return CAPABILITY_META
