"""
Customer Relationship Management Sub-Capability

Comprehensive CRM system for managing customer relationships, sales pipelines,
marketing campaigns, customer service, and business development activities.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Customer Relationship Management',
	'code': 'CRM',
	'version': '1.0.0',
	'capability': 'general_cross_functional',
	'description': 'Comprehensive customer relationship management system with sales pipeline, marketing automation, and customer service',
	'industry_focus': 'All',
	'dependencies': [],
	'optional_dependencies': ['document_management', 'business_intelligence', 'workflow_management'],
	'database_tables': [
		'gc_crm_customer',
		'gc_crm_contact',
		'gc_crm_account',
		'gc_crm_lead',
		'gc_crm_opportunity',
		'gc_crm_sales_stage',
		'gc_crm_activity',
		'gc_crm_task',
		'gc_crm_appointment',
		'gc_crm_campaign',
		'gc_crm_campaign_member',
		'gc_crm_marketing_list',
		'gc_crm_email_template',
		'gc_crm_case',
		'gc_crm_case_comment',
		'gc_crm_product',
		'gc_crm_price_list',
		'gc_crm_quote',
		'gc_crm_quote_line',
		'gc_crm_territory',
		'gc_crm_team',
		'gc_crm_forecast',
		'gc_crm_dashboard_widget',
		'gc_crm_report',
		'gc_crm_integration_log',
		'gc_crm_audit_log'
	],
	'api_endpoints': [
		'/api/general_cross_functional/crm/customers',
		'/api/general_cross_functional/crm/contacts',
		'/api/general_cross_functional/crm/accounts',
		'/api/general_cross_functional/crm/leads',
		'/api/general_cross_functional/crm/opportunities',
		'/api/general_cross_functional/crm/activities',
		'/api/general_cross_functional/crm/campaigns',
		'/api/general_cross_functional/crm/cases',
		'/api/general_cross_functional/crm/quotes',
		'/api/general_cross_functional/crm/dashboard',
		'/api/general_cross_functional/crm/reports',
		'/api/general_cross_functional/crm/pipeline',
		'/api/general_cross_functional/crm/forecast'
	],
	'views': [
		'GCCRMCustomerModelView',
		'GCCRMContactModelView',
		'GCCRMAccountModelView',
		'GCCRMLeadModelView',
		'GCCRMOpportunityModelView',
		'GCCRMActivityModelView',
		'GCCRMCampaignModelView',
		'GCCRMCaseModelView',
		'GCCRMQuoteModelView',
		'GCCRMPipelineView',
		'GCCRMDashboardView',
		'GCCRMReportView',
		'GCCRMForecastView'
	],
	'permissions': [
		'crm.read',
		'crm.write',
		'crm.delete',
		'crm.admin',
		'crm.lead_manage',
		'crm.opportunity_manage',
		'crm.customer_manage',
		'crm.campaign_manage',
		'crm.case_manage',
		'crm.quote_manage',
		'crm.report_view',
		'crm.forecast_view',
		'crm.territory_manage',
		'crm.team_manage',
		'crm.integration_manage'
	],
	'menu_items': [
		{
			'name': 'Customers',
			'endpoint': 'GCCRMCustomerModelView.list',
			'icon': 'fa-users',
			'permission': 'crm.read'
		},
		{
			'name': 'Contacts',
			'endpoint': 'GCCRMContactModelView.list',
			'icon': 'fa-user',
			'permission': 'crm.read'
		},
		{
			'name': 'Leads',
			'endpoint': 'GCCRMLeadModelView.list',
			'icon': 'fa-user-plus',
			'permission': 'crm.lead_manage'
		},
		{
			'name': 'Opportunities',
			'endpoint': 'GCCRMOpportunityModelView.list',
			'icon': 'fa-star',
			'permission': 'crm.opportunity_manage'
		},
		{
			'name': 'Sales Pipeline',
			'endpoint': 'GCCRMPipelineView.index',
			'icon': 'fa-filter',
			'permission': 'crm.read'
		},
		{
			'name': 'Activities',
			'endpoint': 'GCCRMActivityModelView.list',
			'icon': 'fa-calendar',
			'permission': 'crm.read'
		},
		{
			'name': 'Campaigns',
			'endpoint': 'GCCRMCampaignModelView.list',
			'icon': 'fa-bullhorn',
			'permission': 'crm.campaign_manage'
		},
		{
			'name': 'Cases',
			'endpoint': 'GCCRMCaseModelView.list',
			'icon': 'fa-ticket',
			'permission': 'crm.case_manage'
		},
		{
			'name': 'Quotes',
			'endpoint': 'GCCRMQuoteModelView.list',
			'icon': 'fa-file-text',
			'permission': 'crm.quote_manage'
		},
		{
			'name': 'Sales Forecast',
			'endpoint': 'GCCRMForecastView.index',
			'icon': 'fa-line-chart',
			'permission': 'crm.forecast_view'
		},
		{
			'name': 'CRM Dashboard',
			'endpoint': 'GCCRMDashboardView.index',
			'icon': 'fa-dashboard',
			'permission': 'crm.read'
		},
		{
			'name': 'CRM Reports',
			'endpoint': 'GCCRMReportView.index',
			'icon': 'fa-bar-chart',
			'permission': 'crm.report_view'
		}
	],
	'configuration': {
		# Sales Configuration
		'default_sales_stages': [
			'Prospecting',
			'Qualification',
			'Needs Analysis', 
			'Proposal',
			'Negotiation',
			'Closed Won',
			'Closed Lost'
		],
		'opportunity_probability_mapping': {
			'Prospecting': 10,
			'Qualification': 25,
			'Needs Analysis': 50,
			'Proposal': 75,
			'Negotiation': 90,
			'Closed Won': 100,
			'Closed Lost': 0
		},
		
		# Lead Configuration
		'lead_sources': [
			'Website',
			'Social Media',
			'Email Campaign',
			'Trade Show',
			'Referral',
			'Cold Call',
			'Advertisement',
			'Partner',
			'Other'
		],
		'lead_statuses': [
			'New',
			'Contacted',
			'Qualified',
			'Unqualified',
			'Converted',
			'Lost'
		],
		'lead_ratings': [
			'Hot',
			'Warm',
			'Cold'
		],
		
		# Activity Configuration
		'activity_types': [
			'Call',
			'Email',
			'Meeting',
			'Task',
			'Note',
			'Demo',
			'Proposal',
			'Follow-up'
		],
		'activity_priorities': [
			'Low',
			'Normal',
			'High',
			'Urgent'
		],
		
		# Campaign Configuration
		'campaign_types': [
			'Email Marketing',
			'Direct Mail',
			'Telemarketing',
			'Web Campaign',
			'Social Media',
			'Trade Show',
			'Advertisement',
			'Webinar',
			'Event'
		],
		'campaign_statuses': [
			'Planning',
			'Active',
			'Completed',
			'Cancelled',
			'Paused'
		],
		
		# Case Configuration
		'case_types': [
			'Question',
			'Problem',
			'Feature Request',
			'Complaint',
			'Bug Report',
			'Service Request'
		],
		'case_priorities': [
			'Low',
			'Normal',
			'High',
			'Critical'
		],
		'case_statuses': [
			'New',
			'In Progress',
			'Pending',
			'Resolved',
			'Closed',
			'Cancelled'
		],
		
		# Customer Configuration
		'customer_types': [
			'Individual',
			'Small Business',
			'Medium Business',
			'Enterprise',
			'Government',
			'Non-Profit'
		],
		'customer_statuses': [
			'Active',
			'Inactive',
			'Prospect',
			'Former Customer'
		],
		'industry_types': [
			'Technology',
			'Healthcare',
			'Finance',
			'Manufacturing',
			'Retail',
			'Education',
			'Government',
			'Non-Profit',
			'Other'
		],
		
		# Quote Configuration
		'quote_statuses': [
			'Draft',
			'Sent',
			'Reviewed',
			'Accepted',
			'Rejected',
			'Expired'
		],
		'quote_validity_days': 30,
		
		# Integration Configuration
		'email_integration_enabled': True,
		'calendar_sync_enabled': True,
		'phone_integration_enabled': False,
		'social_media_integration_enabled': True,
		'accounting_integration_enabled': True,
		'marketing_automation_enabled': True,
		
		# Automation Configuration
		'auto_lead_assignment': True,
		'auto_follow_up_tasks': True,
		'auto_lead_scoring': True,
		'auto_duplicate_detection': True,
		'auto_activity_logging': True,
		
		# Performance Configuration
		'max_records_per_page': 100,
		'search_results_limit': 1000,
		'dashboard_refresh_minutes': 15,
		'report_cache_hours': 4,
		'bulk_operation_limit': 500
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META

def validate_dependencies(available_subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate dependencies are met"""
	errors = []
	warnings = []
	
	# No hard dependencies for CRM
	
	# Check for useful optional dependencies
	if 'document_management' not in available_subcapabilities:
		warnings.append("Document Management integration not available - limited document attachment capabilities")
	
	if 'business_intelligence' not in available_subcapabilities:
		warnings.append("Business Intelligence integration not available - limited advanced reporting")
	
	if 'workflow_management' not in available_subcapabilities:
		warnings.append("Workflow Management integration not available - limited process automation")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def get_default_sales_stages() -> List[Dict[str, Any]]:
	"""Get default sales stages configuration"""
	stages = SUBCAPABILITY_META['configuration']['default_sales_stages']
	probability_mapping = SUBCAPABILITY_META['configuration']['opportunity_probability_mapping']
	
	result = []
	for i, stage in enumerate(stages):
		result.append({
			'name': stage,
			'order': i + 1,
			'probability': probability_mapping.get(stage, 50),
			'is_closed': stage in ['Closed Won', 'Closed Lost'],
			'is_won': stage == 'Closed Won',
			'color': _get_stage_color(stage),
			'description': _get_stage_description(stage)
		})
	
	return result

def _get_stage_color(stage: str) -> str:
	"""Get color for sales stage"""
	color_mapping = {
		'Prospecting': '#6c757d',
		'Qualification': '#17a2b8',
		'Needs Analysis': '#ffc107',
		'Proposal': '#fd7e14',
		'Negotiation': '#dc3545',
		'Closed Won': '#28a745',
		'Closed Lost': '#343a40'
	}
	return color_mapping.get(stage, '#6c757d')

def _get_stage_description(stage: str) -> str:
	"""Get description for sales stage"""
	descriptions = {
		'Prospecting': 'Initial research and identification of potential customers',
		'Qualification': 'Determining if the prospect has budget, authority, need, and timeline',
		'Needs Analysis': 'Understanding customer requirements and pain points',
		'Proposal': 'Presenting solution and formal proposal to customer',
		'Negotiation': 'Finalizing terms, pricing, and contract details',
		'Closed Won': 'Opportunity successfully closed and won',
		'Closed Lost': 'Opportunity lost to competitor or cancelled'
	}
	return descriptions.get(stage, '')

def get_default_lead_sources() -> List[Dict[str, Any]]:
	"""Get default lead sources configuration"""
	sources = SUBCAPABILITY_META['configuration']['lead_sources']
	
	result = []
	for source in sources:
		result.append({
			'name': source,
			'is_active': True,
			'tracking_enabled': True,
			'cost_per_lead': 0.0,
			'conversion_rate': 0.0
		})
	
	return result

def get_default_activity_types() -> List[Dict[str, Any]]:
	"""Get default activity types configuration"""
	types = SUBCAPABILITY_META['configuration']['activity_types']
	
	result = []
	for activity_type in types:
		result.append({
			'name': activity_type,
			'is_active': True,
			'requires_duration': activity_type in ['Call', 'Meeting', 'Demo'],
			'allows_participants': activity_type in ['Meeting', 'Demo', 'Webinar'],
			'default_duration_minutes': _get_default_duration(activity_type),
			'icon': _get_activity_icon(activity_type)
		})
	
	return result

def _get_default_duration(activity_type: str) -> int:
	"""Get default duration for activity type"""
	durations = {
		'Call': 30,
		'Meeting': 60,
		'Demo': 90,
		'Task': 0,
		'Email': 0,
		'Note': 0,
		'Proposal': 0,
		'Follow-up': 15
	}
	return durations.get(activity_type, 30)

def _get_activity_icon(activity_type: str) -> str:
	"""Get icon for activity type"""
	icons = {
		'Call': 'fa-phone',
		'Email': 'fa-envelope',
		'Meeting': 'fa-users',
		'Task': 'fa-tasks',
		'Note': 'fa-sticky-note',
		'Demo': 'fa-desktop',
		'Proposal': 'fa-file-text',
		'Follow-up': 'fa-repeat'
	}
	return icons.get(activity_type, 'fa-calendar')

def get_crm_kpis() -> List[Dict[str, Any]]:
	"""Get default CRM KPIs and metrics"""
	return [
		{
			'name': 'Lead Conversion Rate',
			'description': 'Percentage of leads converted to opportunities',
			'formula': '(Converted Leads / Total Leads) * 100',
			'target': 15.0,
			'unit': 'percentage'
		},
		{
			'name': 'Opportunity Win Rate',
			'description': 'Percentage of opportunities won',
			'formula': '(Won Opportunities / Total Opportunities) * 100',
			'target': 25.0,
			'unit': 'percentage'
		},
		{
			'name': 'Average Deal Size',
			'description': 'Average value of won opportunities',
			'formula': 'Sum(Won Opportunity Values) / Count(Won Opportunities)',
			'target': 50000.0,
			'unit': 'currency'
		},
		{
			'name': 'Sales Cycle Length',
			'description': 'Average time from opportunity creation to close',
			'formula': 'Average(Close Date - Create Date)',
			'target': 90.0,
			'unit': 'days'
		},
		{
			'name': 'Pipeline Velocity',
			'description': 'Speed at which opportunities move through pipeline',
			'formula': '(Number of Deals * Average Deal Size * Win Rate) / Sales Cycle Length',
			'target': 10000.0,
			'unit': 'currency_per_day'
		}
	]