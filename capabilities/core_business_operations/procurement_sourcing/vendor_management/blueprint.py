"""
Vendor Management Blueprint
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder
from typing import Dict, Any

from .views import VendorView, VendorContactView, VendorPerformanceView, VendorDashboardView


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Vendor Management sub-capability views"""
	
	appbuilder.add_view(
		VendorView,
		"Vendors",
		icon="fa-building",
		category="Vendor Management",
		category_icon="fa-building"
	)
	
	appbuilder.add_view(
		VendorContactView,
		"Vendor Contacts",
		icon="fa-users",
		category="Vendor Management"
	)
	
	appbuilder.add_view(
		VendorPerformanceView,
		"Vendor Performance",
		icon="fa-bar-chart",
		category="Vendor Management"
	)
	
	appbuilder.add_view(
		VendorDashboardView,
		"Dashboard",
		icon="fa-dashboard",
		category="Vendor Management"
	)


def get_menu_structure() -> Dict[str, Any]:
	"""Get menu structure for Vendor Management sub-capability"""
	
	return {
		'name': 'Vendor Management',
		'icon': 'fa-building',
		'items': [
			{
				'name': 'Dashboard',
				'href': '/vendor_management/dashboard/',
				'icon': 'fa-dashboard'
			},
			{
				'name': 'Vendors',
				'href': '/vendorview/list/',
				'icon': 'fa-building'
			},
			{
				'name': 'Vendor Performance',
				'href': '/vendorperformanceview/list/',
				'icon': 'fa-bar-chart'
			}
		]
	}