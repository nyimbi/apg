"""
Order Processing Blueprint Registration

Flask blueprint registration for Order Processing sub-capability.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder

# Create blueprint
order_processing_bp = Blueprint(
	'order_processing',
	__name__,
	url_prefix='/sales_order_management/order_processing'
)

def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Order Processing sub-capability with Flask-AppBuilder"""
	
	# Import views (would be implemented)
	from .views import (
		SOPFulfillmentTaskView, SOPShipmentView, SOPOrderStatusView,
		SOPOrderWorkflowView
	)
	
	# Register views with AppBuilder
	appbuilder.add_view(
		SOPFulfillmentTaskView,
		"Fulfillment Tasks",
		icon="fa-tasks",
		category="Order Processing",
		category_icon="fa-cogs"
	)
	
	appbuilder.add_view(
		SOPShipmentView,
		"Shipments",
		icon="fa-truck",
		category="Order Processing"
	)
	
	appbuilder.add_view(
		SOPOrderStatusView,
		"Order Status Config",
		icon="fa-flag",
		category="Order Processing"
	)
	
	appbuilder.add_view(
		SOPOrderWorkflowView,
		"Workflows",
		icon="fa-sitemap",
		category="Order Processing"
	)
	
	# Register blueprint
	appbuilder.get_app.register_blueprint(order_processing_bp)