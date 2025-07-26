"""
Quotations Views

Flask-AppBuilder views for customer quotation and proposal management.
"""

from flask_appbuilder import ModelView
from flask_appbuilder.models.sqla.interface import SQLAInterface

from .models import SOQQuotation, SOQQuoteTemplate


class SOQQuotationView(ModelView):
	"""Customer quotation management view"""
	
	datamodel = SQLAInterface(SOQQuotation)
	
	list_columns = [
		'quote_number', 'customer_name', 'quote_date', 
		'valid_until_date', 'total_amount', 'status'
	]
	search_columns = ['quote_number', 'customer_name', 'quote_name']
	list_filters = ['status', 'quote_date', 'sales_rep_id']
	
	formatters_columns = {
		'total_amount': lambda x: f"${x:,.2f}" if x else "$0.00"
	}
	
	base_order = ('quote_date', 'desc')


class SOQQuoteTemplateView(ModelView):
	"""Quote template management view"""
	
	datamodel = SQLAInterface(SOQQuoteTemplate)
	
	list_columns = [
		'template_name', 'template_type', 'usage_count',
		'is_active', 'is_public'
	]
	search_columns = ['template_name', 'description']
	list_filters = ['template_type', 'is_active', 'is_public']
	base_order = ('template_name', 'asc')