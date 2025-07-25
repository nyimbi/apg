"""
Quotations Blueprint Registration

Flask blueprint registration for Quotations sub-capability.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder

# Create blueprint
quotations_bp = Blueprint(
	'quotations',
	__name__,
	url_prefix='/sales_order_management/quotations'
)

def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Quotations sub-capability with Flask-AppBuilder"""
	
	# Import views (would be implemented)
	from .views import (
		SOQQuotationView, SOQQuoteTemplateView
	)
	
	# Register views with AppBuilder
	appbuilder.add_view(
		SOQQuotationView,
		"Quotations",
		icon="fa-file-text-o",
		category="Quotations",
		category_icon="fa-quote-left"
	)
	
	appbuilder.add_view(
		SOQQuoteTemplateView,
		"Quote Templates",
		icon="fa-copy",
		category="Quotations"
	)
	
	# Register blueprint
	appbuilder.get_app.register_blueprint(quotations_bp)