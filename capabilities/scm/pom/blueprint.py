"""
Purchase Order Management Blueprint
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder
from typing import Dict, Any

from .views import (
	PurchaseOrderView, PurchaseOrderLineView, ReceiptView,
	ReceiptLineView, ThreeWayMatchView, ChangeOrderView, PurchaseOrderDashboardView
)


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Purchase Order Management sub-capability views"""
	
	appbuilder.add_view(
		PurchaseOrderView,
		"Purchase Orders",
		icon="fa-shopping-cart",
		category="Purchase Order Management",
		category_icon="fa-shopping-cart"
	)
	
	appbuilder.add_view(
		PurchaseOrderLineView,
		"PO Lines",
		icon="fa-list",
		category="Purchase Order Management"
	)
	
	appbuilder.add_view(
		ReceiptView,
		"Receipts",
		icon="fa-inbox",
		category="Purchase Order Management"
	)
	
	appbuilder.add_view(
		ThreeWayMatchView,
		"Three-Way Matching",
		icon="fa-check-square",
		category="Purchase Order Management"
	)
	
	appbuilder.add_view(
		ChangeOrderView,
		"Change Orders",
		icon="fa-edit",
		category="Purchase Order Management"
	)
	
	appbuilder.add_view(
		PurchaseOrderDashboardView,
		"Dashboard",
		icon="fa-dashboard",
		category="Purchase Order Management"
	)


def get_menu_structure() -> Dict[str, Any]:
	"""Get menu structure for Purchase Order Management sub-capability"""
	
	return {
		'name': 'Purchase Order Management',
		'icon': 'fa-shopping-cart',
		'items': [
			{
				'name': 'Dashboard',
				'href': '/purchase_order_management/dashboard/',
				'icon': 'fa-dashboard'
			},
			{
				'name': 'Purchase Orders',
				'href': '/purchaseorderview/list/',
				'icon': 'fa-shopping-cart'
			},
			{
				'name': 'Receipts',
				'href': '/receiptview/list/',
				'icon': 'fa-inbox'
			},
			{
				'name': 'Three-Way Matching',
				'href': '/threewaymatchview/list/',
				'icon': 'fa-check-square'
			}
		]
	}
