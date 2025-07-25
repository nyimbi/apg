"""
Business Intelligence (BI) / Analytics Sub-Capability

Provides real-time insights through customizable reports, dashboards,
and data analysis for informed decision-making.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Business Intelligence (BI) / Analytics',
	'code': 'BI',
	'version': '1.0.0',
	'capability': 'general_cross_functional',
	'description': 'Provides real-time insights through customizable reports, dashboards, and data analysis for informed decision-making',
	'industry_focus': 'All Industries',
	'dependencies': [],
	'optional_dependencies': ['document_management'],
	'database_tables': [
		'gc_bi_dashboard',
		'gc_bi_report',
		'gc_bi_data_source',
		'gc_bi_metric',
		'gc_bi_kpi',
		'gc_bi_alert',
		'gc_bi_data_cube',
		'gc_bi_visualization'
	],
	'configuration': {
		'enable_real_time_refresh': True,
		'default_cache_duration_minutes': 15,
		'enable_scheduled_reports': True,
		'enable_drill_down': True,
		'enable_data_export': True,
		'max_concurrent_queries': 50,
		'enable_predictive_analytics': True
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META