"""
Stock Tracking & Control Blueprint

Blueprint registration for Stock Tracking & Control sub-capability views and API endpoints.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder
from typing import Dict, List, Any

from .views import (
	IMSTCItemCategoryView, IMSTCUnitOfMeasureView, IMSTCWarehouseView,
	IMSTCLocationView, IMSTCItemView, IMSTCStockLevelView,
	IMSTCStockMovementView, IMSTCStockAlertView, StockTrackingDashboardView,
	StockMovementChartView
)


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Stock Tracking & Control sub-capability with Flask-AppBuilder"""
	
	# Register model views
	appbuilder.add_view(
		IMSTCItemCategoryView,
		"Item Categories",
		icon="fa-tags",
		category="Stock Tracking"
	)
	
	appbuilder.add_view(
		IMSTCUnitOfMeasureView,
		"Units of Measure",
		icon="fa-balance-scale",
		category="Stock Tracking"
	)
	
	appbuilder.add_view(
		IMSTCWarehouseView,
		"Warehouses",
		icon="fa-warehouse",
		category="Stock Tracking"
	)
	
	appbuilder.add_view(
		IMSTCLocationView,
		"Storage Locations",
		icon="fa-map-marker-alt",
		category="Stock Tracking"
	)
	
	appbuilder.add_view(
		IMSTCItemView,
		"Inventory Items",
		icon="fa-cube",
		category="Stock Tracking"
	)
	
	appbuilder.add_view(
		IMSTCStockLevelView,
		"Current Stock Levels",
		icon="fa-layer-group",
		category="Stock Tracking"
	)
	
	appbuilder.add_view(
		IMSTCStockMovementView,
		"Stock Movement History",
		icon="fa-exchange-alt",
		category="Stock Tracking"
	)
	
	appbuilder.add_view(
		IMSTCStockAlertView,
		"Stock Alerts",
		icon="fa-exclamation-triangle",
		category="Stock Tracking"
	)
	
	# Register dashboard and reports
	appbuilder.add_view_no_menu(StockTrackingDashboardView)
	appbuilder.add_link(
		"Stock Tracking Dashboard",
		href="/stock_tracking/dashboard/",
		icon="fa-tachometer-alt",
		category="Stock Tracking"
	)
	
	appbuilder.add_link(
		"Low Stock Report",
		href="/stock_tracking/dashboard/low_stock_report",
		icon="fa-exclamation-circle",
		category="Stock Tracking"
	)
	
	appbuilder.add_link(
		"Inventory Valuation",
		href="/stock_tracking/dashboard/inventory_valuation",
		icon="fa-dollar-sign",
		category="Stock Tracking"
	)
	
	appbuilder.add_link(
		"ABC Analysis",
		href="/stock_tracking/dashboard/abc_analysis",
		icon="fa-chart-pie",
		category="Stock Tracking"
	)
	
	# Register chart views
	appbuilder.add_view(
		StockMovementChartView,
		"Movement Chart",
		icon="fa-chart-bar",
		category="Stock Tracking"
	)
	
	print("Stock Tracking & Control sub-capability initialized")


def get_menu_structure() -> Dict[str, Any]:
	"""Get menu structure for Stock Tracking & Control"""
	
	return {
		'name': 'Stock Tracking & Control',
		'items': [
			{
				'name': 'Stock Tracking Dashboard',
				'href': '/stock_tracking/dashboard/',
				'icon': 'fa-tachometer-alt'
			},
			{
				'name': 'Inventory Items',
				'href': '/imstcitemview/list/',
				'icon': 'fa-cube'
			},
			{
				'name': 'Current Stock Levels',
				'href': '/imstcstocklevelview/list/',
				'icon': 'fa-layer-group'
			},
			{
				'name': 'Stock Movement History',
				'href': '/imstcstockmovementview/list/',
				'icon': 'fa-exchange-alt'
			},
			{
				'name': 'Stock Alerts',
				'href': '/imstcstockalertview/list/',
				'icon': 'fa-exclamation-triangle'
			},
			{
				'name': 'Reports',
				'icon': 'fa-chart-line',
				'submenu': [
					{
						'name': 'Low Stock Report',
						'href': '/stock_tracking/dashboard/low_stock_report',
						'icon': 'fa-exclamation-circle'
					},
					{
						'name': 'Inventory Valuation',
						'href': '/stock_tracking/dashboard/inventory_valuation',
						'icon': 'fa-dollar-sign'
					},
					{
						'name': 'ABC Analysis',
						'href': '/stock_tracking/dashboard/abc_analysis',
						'icon': 'fa-chart-pie'
					}
				]
			}
		]
	}