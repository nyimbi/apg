"""
Workflow Management / Business Process Management (BPM) Sub-Capability

Automates and optimizes business processes for efficiency and standardization
across the organization.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Workflow Management / Business Process Management (BPM)',
	'code': 'WF',
	'version': '1.0.0',
	'capability': 'general_cross_functional',
	'description': 'Automates and optimizes business processes for efficiency and standardization',
	'industry_focus': 'All Industries',
	'dependencies': [],
	'optional_dependencies': ['document_management', 'business_intelligence_analytics'],
	'database_tables': [
		'gc_wf_process_definition',
		'gc_wf_process_instance',
		'gc_wf_task',
		'gc_wf_workflow_step',
		'gc_wf_approval_rule',
		'gc_wf_escalation_rule',
		'gc_wf_process_variable',
		'gc_wf_performance_metric'
	],
	'configuration': {
		'enable_visual_designer': True,
		'enable_parallel_processing': True,
		'enable_conditional_routing': True,
		'enable_escalation_rules': True,
		'enable_sla_monitoring': True,
		'default_task_timeout_hours': 72,
		'enable_process_mining': True,
		'enable_api_integration': True
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META