"""
Sourcing & Supplier Selection Blueprint
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder
from typing import Dict, Any

from .views import RFQHeaderView, BidView, SupplierEvaluationView, AwardRecommendationView, SourcingDashboardView


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Sourcing & Supplier Selection sub-capability views"""
	
	appbuilder.add_view(
		RFQHeaderView,
		"RFQ/RFP",
		icon="fa-file-text",
		category="Sourcing & Supplier Selection",
		category_icon="fa-search"
	)
	
	appbuilder.add_view(
		BidView,
		"Bids",
		icon="fa-handshake-o",
		category="Sourcing & Supplier Selection"
	)
	
	appbuilder.add_view(
		SupplierEvaluationView,
		"Supplier Evaluations",
		icon="fa-star",
		category="Sourcing & Supplier Selection"
	)
	
	appbuilder.add_view(
		AwardRecommendationView,
		"Award Recommendations",
		icon="fa-trophy",
		category="Sourcing & Supplier Selection"
	)
	
	appbuilder.add_view(
		SourcingDashboardView,
		"Dashboard",
		icon="fa-dashboard",
		category="Sourcing & Supplier Selection"
	)


def get_menu_structure() -> Dict[str, Any]:
	"""Get menu structure for Sourcing & Supplier Selection sub-capability"""
	
	return {
		'name': 'Sourcing & Supplier Selection',
		'icon': 'fa-search',
		'items': [
			{
				'name': 'Dashboard',
				'href': '/sourcing_supplier_selection/dashboard/',
				'icon': 'fa-dashboard'
			},
			{
				'name': 'RFQ/RFP',
				'href': '/rfqheaderview/list/',
				'icon': 'fa-file-text'
			},
			{
				'name': 'Bids',
				'href': '/bidview/list/',
				'icon': 'fa-handshake-o'
			},
			{
				'name': 'Evaluations',
				'href': '/supplierevaluationview/list/',
				'icon': 'fa-star'
			}
		]
	}