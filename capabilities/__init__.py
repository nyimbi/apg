"""
APG Capabilities Registry

Central registry and import hub for all APG (Application Programming Generation) capabilities.
Each capability is organized in its own directory with standardized structure:

- service.py: Core business logic and data models
- views.py: Flask-AppBuilder views and web interfaces  
- models.py: Database models (if different from service.py)
- events.py: Event system integration
- exceptions.py: Custom exception classes
- __init__.py: Capability metadata and exports

Capability Composition:
Capabilities can be composed together using standardized keywords and event systems.
Each capability declares its composition keywords and integration points.
"""

from typing import Dict, List, Any

# Capability metadata registry
CAPABILITY_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_capability(capability_code: str, capability_info: Dict[str, Any]):
	"""Register a capability in the global registry"""
	CAPABILITY_REGISTRY[capability_code] = capability_info

def get_capability(capability_code: str) -> Dict[str, Any]:
	"""Get capability information by code"""
	return CAPABILITY_REGISTRY.get(capability_code)

def list_capabilities() -> List[Dict[str, Any]]:
	"""List all registered capabilities"""
	return list(CAPABILITY_REGISTRY.values())

def get_capabilities_by_keyword(keyword: str) -> List[Dict[str, Any]]:
	"""Get capabilities that support a specific composition keyword"""
	matching = []
	for capability in CAPABILITY_REGISTRY.values():
		if keyword in capability.get('composition_keywords', []):
			matching.append(capability)
	return matching

# Import all capabilities to trigger registration
try:
	from . import ai_orchestration
	register_capability(ai_orchestration.__capability_code__, {
		'code': ai_orchestration.__capability_code__,
		'name': ai_orchestration.__capability_name__,
		'version': ai_orchestration.__version__,
		'composition_keywords': ai_orchestration.__composition_keywords__,
		'module': ai_orchestration
	})
except ImportError:
	pass

try:
	from . import audio_processing
	register_capability(audio_processing.__capability_code__, {
		'code': audio_processing.__capability_code__,
		'name': audio_processing.__capability_name__,
		'version': audio_processing.__version__,
		'composition_keywords': audio_processing.__composition_keywords__,
		'module': audio_processing
	})
except ImportError:
	pass

try:
	from . import audit_compliance
	register_capability(audit_compliance.__capability_code__, {
		'code': audit_compliance.__capability_code__,
		'name': audit_compliance.__capability_name__,
		'version': audit_compliance.__version__,
		'composition_keywords': audit_compliance.__composition_keywords__,
		'module': audit_compliance
	})
except ImportError:
	pass

try:
	from . import blockchain_security
	register_capability(blockchain_security.__capability_code__, {
		'code': blockchain_security.__capability_code__,
		'name': blockchain_security.__capability_name__,
		'version': blockchain_security.__version__,
		'composition_keywords': blockchain_security.__composition_keywords__,
		'module': blockchain_security
	})
except ImportError:
	pass

try:
	from . import computer_vision
	register_capability(computer_vision.__capability_code__, {
		'code': computer_vision.__capability_code__,
		'name': computer_vision.__capability_name__,
		'version': computer_vision.__version__,
		'composition_keywords': computer_vision.__composition_keywords__,
		'module': computer_vision
	})
except ImportError:
	pass

try:
	from . import digital_twin
	register_capability(digital_twin.__capability_code__, {
		'code': digital_twin.__capability_code__,
		'name': digital_twin.__capability_name__,
		'version': digital_twin.__version__,
		'composition_keywords': digital_twin.__composition_keywords__,
		'module': digital_twin
	})
except ImportError:
	pass

try:
	from . import digital_twin_marketplace
	register_capability(digital_twin_marketplace.__capability_code__, {
		'code': digital_twin_marketplace.__capability_code__,
		'name': digital_twin_marketplace.__capability_name__,
		'version': digital_twin_marketplace.__version__,
		'composition_keywords': digital_twin_marketplace.__composition_keywords__,
		'module': digital_twin_marketplace
	})
except ImportError:
	pass

try:
	from . import distributed_computing
	register_capability(distributed_computing.__capability_code__, {
		'code': distributed_computing.__capability_code__,
		'name': distributed_computing.__capability_name__,
		'version': distributed_computing.__version__,
		'composition_keywords': distributed_computing.__composition_keywords__,
		'module': distributed_computing
	})
except ImportError:
	pass

try:
	from . import edge_computing
	register_capability(edge_computing.__capability_code__, {
		'code': edge_computing.__capability_code__,
		'name': edge_computing.__capability_name__,
		'version': edge_computing.__version__,
		'composition_keywords': edge_computing.__composition_keywords__,
		'module': edge_computing
	})
except ImportError:
	pass

try:
	from . import federated_learning
	register_capability(federated_learning.__capability_code__, {
		'code': federated_learning.__capability_code__,
		'name': federated_learning.__capability_name__,
		'version': federated_learning.__version__,
		'composition_keywords': federated_learning.__composition_keywords__,
		'module': federated_learning
	})
except ImportError:
	pass

try:
	from . import financial_management
	register_capability(financial_management.__capability_code__, {
		'code': financial_management.__capability_code__,
		'name': financial_management.__capability_name__,
		'version': financial_management.__version__,
		'composition_keywords': financial_management.__composition_keywords__,
		'module': financial_management
	})
except ImportError:
	pass

try:
	from . import intelligent_orchestration
	register_capability(intelligent_orchestration.__capability_code__, {
		'code': intelligent_orchestration.__capability_code__,
		'name': intelligent_orchestration.__capability_name__,
		'version': intelligent_orchestration.__version__,
		'composition_keywords': intelligent_orchestration.__composition_keywords__,
		'module': intelligent_orchestration
	})
except ImportError:
	pass

try:
	from . import iot_management
	register_capability(iot_management.__capability_code__, {
		'code': iot_management.__capability_code__,
		'name': iot_management.__capability_name__,
		'version': iot_management.__version__,
		'composition_keywords': iot_management.__composition_keywords__,
		'module': iot_management
	})
except ImportError:
	pass

try:
	from . import multi_tenant_enterprise
	register_capability(multi_tenant_enterprise.__capability_code__, {
		'code': multi_tenant_enterprise.__capability_code__,
		'name': multi_tenant_enterprise.__capability_name__,
		'version': multi_tenant_enterprise.__version__,
		'composition_keywords': multi_tenant_enterprise.__composition_keywords__,
		'module': multi_tenant_enterprise
	})
except ImportError:
	pass

try:
	from . import predictive_maintenance
	register_capability(predictive_maintenance.__capability_code__, {
		'code': predictive_maintenance.__capability_code__,
		'name': predictive_maintenance.__capability_name__,
		'version': predictive_maintenance.__version__,
		'composition_keywords': predictive_maintenance.__composition_keywords__,
		'module': predictive_maintenance
	})
except ImportError:
	pass

try:
	from . import product_catalog
	register_capability(product_catalog.__capability_code__, {
		'code': product_catalog.__capability_code__,
		'name': product_catalog.__capability_name__,
		'version': product_catalog.__version__,
		'composition_keywords': product_catalog.__composition_keywords__,
		'module': product_catalog
	})
except ImportError:
	pass

try:
	from . import profile_management
	register_capability(profile_management.__capability_code__, {
		'code': profile_management.__capability_code__,
		'name': profile_management.__capability_name__,
		'version': profile_management.__version__,
		'composition_keywords': profile_management.__composition_keywords__,
		'module': profile_management
	})
except ImportError:
	pass

try:
	from . import quantum_computing
	register_capability(quantum_computing.__capability_code__, {
		'code': quantum_computing.__capability_code__,
		'name': quantum_computing.__capability_name__,
		'version': quantum_computing.__version__,
		'composition_keywords': quantum_computing.__composition_keywords__,
		'module': quantum_computing
	})
except ImportError:
	pass

try:
	from . import real_time_collaboration
	register_capability(real_time_collaboration.__capability_code__, {
		'code': real_time_collaboration.__capability_code__,
		'name': real_time_collaboration.__capability_name__,
		'version': real_time_collaboration.__version__,
		'composition_keywords': real_time_collaboration.__composition_keywords__,
		'module': real_time_collaboration
	})
except ImportError:
	pass

try:
	from . import time_series_analytics
	register_capability(time_series_analytics.__capability_code__, {
		'code': time_series_analytics.__capability_code__,
		'name': time_series_analytics.__capability_name__,
		'version': time_series_analytics.__version__,
		'composition_keywords': time_series_analytics.__composition_keywords__,
		'module': time_series_analytics
	})
except ImportError:
	pass

try:
	from . import visualization_3d
	register_capability(visualization_3d.__capability_code__, {
		'code': visualization_3d.__capability_code__,
		'name': visualization_3d.__capability_name__,
		'version': visualization_3d.__version__,
		'composition_keywords': visualization_3d.__composition_keywords__,
		'module': visualization_3d
	})
except ImportError:
	pass

# Export main interfaces
__all__ = [
	'CAPABILITY_REGISTRY',
	'register_capability',
	'get_capability',
	'list_capabilities',
	'get_capabilities_by_keyword'
]