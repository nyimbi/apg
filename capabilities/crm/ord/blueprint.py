"""
Order Entry Blueprint Registration

Flask blueprint registration for Order Entry sub-capability.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder

# Create blueprint
order_entry_bp = Blueprint(
	'order_entry',
	__name__,
	url_prefix='/sales_order_management/order_entry'
)

def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Order Entry sub-capability with Flask-AppBuilder"""
	
	# Import views
	from .views import (
		SOECustomerView, SOEShipToAddressView, SOESalesOrderView,
		SOEOrderLineView, SOEOrderChargeView, SOEPriceLevelView,
		SOEOrderTemplateView, SOEOrderSequenceView, OrderEntryDashboardView
	)
	
	# Register views with AppBuilder
	appbuilder.add_view(
		SOECustomerView,
		"Customers",
		icon="fa-users",
		category="Order Entry",
		category_icon="fa-shopping-cart"
	)
	
	appbuilder.add_view(
		SOEShipToAddressView,
		"Ship-To Addresses",
		icon="fa-map-marker",
		category="Order Entry"
	)
	
	appbuilder.add_view(
		SOESalesOrderView,
		"Sales Orders",
		icon="fa-file-text",
		category="Order Entry"
	)
	
	appbuilder.add_view(
		SOEOrderLineView,
		"Order Lines",
		icon="fa-list",
		category="Order Entry"
	)
	
	appbuilder.add_view(
		SOEOrderChargeView,
		"Order Charges",
		icon="fa-dollar-sign",
		category="Order Entry"
	)
	
	appbuilder.add_view(
		SOEPriceLevelView,
		"Price Levels",
		icon="fa-tags",
		category="Order Entry"
	)
	
	appbuilder.add_view(
		SOEOrderTemplateView,
		"Order Templates",
		icon="fa-copy",
		category="Order Entry"
	)
	
	appbuilder.add_view(
		SOEOrderSequenceView,
		"Order Sequences",
		icon="fa-sort-numeric-asc",
		category="Order Entry"
	)
	
	appbuilder.add_view(
		OrderEntryDashboardView,
		"Dashboard",
		icon="fa-dashboard",
		category="Order Entry"
	)
	
	# Register blueprint
	appbuilder.get_app.register_blueprint(order_entry_bp)