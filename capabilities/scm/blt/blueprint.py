"""
Batch & Lot Tracking Blueprint

Blueprint registration for Batch & Lot Tracking sub-capability views and API endpoints.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder
from typing import Dict, List, Any

from .views import (
	IMBLTBatchView, IMBLTQualityTestView, IMBLTRecallEventView,
	BatchLotDashboardView
)


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Batch & Lot Tracking sub-capability with Flask-AppBuilder"""
	
	# Register model views
	appbuilder.add_view(
		IMBLTBatchView,
		"Batches/Lots",
		icon="fa-barcode",
		category="Batch & Lot Tracking"
	)
	
	appbuilder.add_view(
		IMBLTQualityTestView,
		"Quality Tests",
		icon="fa-flask",
		category="Batch & Lot Tracking"
	)
	
	appbuilder.add_view(
		IMBLTRecallEventView,
		"Recall Events",
		icon="fa-exclamation-triangle",
		category="Batch & Lot Tracking"
	)
	
	# Register dashboard
	appbuilder.add_view_no_menu(BatchLotDashboardView)
	appbuilder.add_link(
		"Batch & Lot Dashboard",
		href="/batch_lot/dashboard/",
		icon="fa-tachometer-alt",
		category="Batch & Lot Tracking"
	)
	
	print("Batch & Lot Tracking sub-capability initialized")


def get_menu_structure() -> Dict[str, Any]:
	"""Get menu structure for Batch & Lot Tracking"""
	
	return {
		'name': 'Batch & Lot Tracking',
		'items': [
			{
				'name': 'Batch & Lot Dashboard',
				'href': '/batch_lot/dashboard/',
				'icon': 'fa-tachometer-alt'
			},
			{
				'name': 'Batches/Lots',
				'href': '/imbltbatchview/list/',
				'icon': 'fa-barcode'
			},
			{
				'name': 'Quality Tests',
				'href': '/imbltqualitytestview/list/',
				'icon': 'fa-flask'
			},
			{
				'name': 'Recall Events',
				'href': '/imbltrecalleventview/list/',
				'icon': 'fa-exclamation-triangle'
			}
		]
	}