"""
Expiry Date Management Blueprint

Blueprint registration for Expiry Date Management sub-capability views and API endpoints.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder
from typing import Dict, List, Any

from .views import (
	IMEDMExpiryItemView, IMEDMDispositionView, IMEDMExpiryAlertView,
	ExpiryDateDashboardView
)


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Expiry Date Management sub-capability with Flask-AppBuilder"""
	
	# Register model views
	appbuilder.add_view(
		IMEDMExpiryItemView,
		"Expiry Items",
		icon="fa-calendar-times",
		category="Expiry Management"
	)
	
	appbuilder.add_view(
		IMEDMDispositionView,
		"Dispositions",
		icon="fa-trash-alt",
		category="Expiry Management"
	)
	
	appbuilder.add_view(
		IMEDMExpiryAlertView,
		"Expiry Alerts",
		icon="fa-bell",
		category="Expiry Management"
	)
	
	# Register dashboard and reports
	appbuilder.add_view_no_menu(ExpiryDateDashboardView)
	appbuilder.add_link(
		"Expiry Dashboard",
		href="/expiry/dashboard/",
		icon="fa-tachometer-alt",
		category="Expiry Management"
	)
	
	appbuilder.add_link(
		"FEFO Report",
		href="/expiry/dashboard/fefo_report",
		icon="fa-sort-numeric-down",
		category="Expiry Management"
	)
	
	print("Expiry Date Management sub-capability initialized")


def get_menu_structure() -> Dict[str, Any]:
	"""Get menu structure for Expiry Date Management"""
	
	return {
		'name': 'Expiry Date Management',
		'items': [
			{
				'name': 'Expiry Dashboard',
				'href': '/expiry/dashboard/',
				'icon': 'fa-tachometer-alt'
			},
			{
				'name': 'Expiry Items',
				'href': '/imedmexpiryitemview/list/',
				'icon': 'fa-calendar-times'
			},
			{
				'name': 'Dispositions',
				'href': '/imedmdispositionview/list/',
				'icon': 'fa-trash-alt'
			},
			{
				'name': 'Expiry Alerts',
				'href': '/imedmexpiryalertview/list/',
				'icon': 'fa-bell'
			},
			{
				'name': 'FEFO Report',
				'href': '/expiry/dashboard/fefo_report',
				'icon': 'fa-sort-numeric-down'
			}
		]
	}