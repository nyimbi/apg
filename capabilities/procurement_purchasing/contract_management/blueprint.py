"""
Contract Management Blueprint
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder
from typing import Dict, Any

from .views import (
	ContractView, ContractLineView, ContractAmendmentView,
	ContractRenewalView, ContractMilestoneView, ContractDocumentView, ContractDashboardView
)


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Contract Management sub-capability views"""
	
	appbuilder.add_view(
		ContractView,
		"Contracts",
		icon="fa-file-text-o",
		category="Contract Management",
		category_icon="fa-file-text-o"
	)
	
	appbuilder.add_view(
		ContractLineView,
		"Contract Lines",
		icon="fa-list",
		category="Contract Management"
	)
	
	appbuilder.add_view(
		ContractAmendmentView,
		"Amendments",
		icon="fa-edit",
		category="Contract Management"
	)
	
	appbuilder.add_view(
		ContractRenewalView,
		"Renewals",
		icon="fa-refresh",
		category="Contract Management"
	)
	
	appbuilder.add_view(
		ContractMilestoneView,
		"Milestones",
		icon="fa-flag-checkered",
		category="Contract Management"
	)
	
	appbuilder.add_view(
		ContractDocumentView,
		"Documents",
		icon="fa-folder",
		category="Contract Management"
	)
	
	appbuilder.add_view(
		ContractDashboardView,
		"Dashboard",
		icon="fa-dashboard",
		category="Contract Management"
	)


def get_menu_structure() -> Dict[str, Any]:
	"""Get menu structure for Contract Management sub-capability"""
	
	return {
		'name': 'Contract Management',
		'icon': 'fa-file-text-o',
		'items': [
			{
				'name': 'Dashboard',
				'href': '/contract_management/dashboard/',
				'icon': 'fa-dashboard'
			},
			{
				'name': 'Contracts',
				'href': '/contractview/list/',
				'icon': 'fa-file-text-o'
			},
			{
				'name': 'Amendments',
				'href': '/contractamendmentview/list/',
				'icon': 'fa-edit'
			},
			{
				'name': 'Renewals',
				'href': '/contractrenewalview/list/',
				'icon': 'fa-refresh'
			},
			{
				'name': 'Milestones',
				'href': '/contractmilestoneview/list/',
				'icon': 'fa-flag-checkered'
			}
		]
	}