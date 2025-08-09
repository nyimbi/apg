"""
Sourcing & Supplier Selection Sub-capability

Strategic sourcing and supplier selection with RFQ/RFP processes,
bid evaluation, and supplier qualification management.
"""

from typing import Dict, Any

# Sub-capability metadata
SUBCAPABILITY_INFO = {
	'code': 'PPS',
	'name': 'Sourcing & Supplier Selection',
	'description': 'Supplier evaluation, RFQ/RFP processes, and selection',
	'version': '1.0.0',
	'models': [
		'PPSRFQHeader', 'PPSRFQLine', 'PPSBid', 'PPSBidLine',
		'PPSSupplierEvaluation', 'PPSEvaluationCriteria', 'PPSAwardRecommendation'
	]
}

def get_subcapability_info() -> Dict[str, Any]:
	return SUBCAPABILITY_INFO.copy()