"""
Replenishment & Reordering Blueprint

Blueprint registration for Replenishment & Reordering sub-capability views and API endpoints.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder
from typing import Dict, List, Any

from .views import (
	IMRRSupplierView, IMRRReplenishmentRuleView, IMRRReplenishmentSuggestionView,
	IMRRPurchaseOrderView, ReplenishmentDashboardView
)


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Replenishment & Reordering sub-capability with Flask-AppBuilder"""
	
	# Register model views
	appbuilder.add_view(
		IMRRSupplierView,
		"Suppliers",
		icon="fa-truck",
		category="Replenishment"
	)
	
	appbuilder.add_view(
		IMRRReplenishmentRuleView,
		"Replenishment Rules",
		icon="fa-cogs",
		category="Replenishment"
	)
	
	appbuilder.add_view(
		IMRRReplenishmentSuggestionView,
		"Replenishment Suggestions",
		icon="fa-lightbulb",
		category="Replenishment"
	)
	
	appbuilder.add_view(
		IMRRPurchaseOrderView,
		"Purchase Orders",
		icon="fa-shopping-cart",
		category="Replenishment"
	)
	
	# Register dashboard
	appbuilder.add_view_no_menu(ReplenishmentDashboardView)
	appbuilder.add_link(
		"Replenishment Dashboard",
		href="/replenishment/dashboard/",
		icon="fa-tachometer-alt",
		category="Replenishment"
	)
	
	print("Replenishment & Reordering sub-capability initialized")


def get_menu_structure() -> Dict[str, Any]:
	"""Get menu structure for Replenishment & Reordering"""
	
	return {
		'name': 'Replenishment & Reordering',
		'items': [
			{
				'name': 'Replenishment Dashboard',
				'href': '/replenishment/dashboard/',
				'icon': 'fa-tachometer-alt'
			},
			{
				'name': 'Suppliers',
				'href': '/imrrsupplierview/list/',
				'icon': 'fa-truck'
			},
			{
				'name': 'Replenishment Rules',
				'href': '/imrrreplenishmentruleview/list/',
				'icon': 'fa-cogs'
			},
			{
				'name': 'Suggestions',
				'href': '/imrrreplenishmentsuggestionview/list/',
				'icon': 'fa-lightbulb'
			},
			{
				'name': 'Purchase Orders',
				'href': '/imrrpurchaseorderview/list/',
				'icon': 'fa-shopping-cart'
			}
		]
	}