"""
Sales & Order Management Blueprint Registration

Flask blueprint registration and initialization for Sales & Order Management
capability and all its sub-capabilities.
"""

from typing import List, Dict, Any
from flask import Blueprint

# Create main blueprint
sales_order_management_bp = Blueprint(
	'sales_order_management',
	__name__,
	url_prefix='/sales_order_management'
)

def init_capability(appbuilder, subcapabilities: List[str] = None) -> Dict[str, Any]:
	"""
	Initialize Sales & Order Management capability with Flask-AppBuilder
	
	Args:
		appbuilder: Flask-AppBuilder instance
		subcapabilities: List of sub-capabilities to initialize
		
	Returns:
		Dictionary with initialization results
	"""
	from . import CAPABILITY_META
	
	if subcapabilities is None:
		subcapabilities = CAPABILITY_META['implemented_subcapabilities']
	
	initialized = []
	errors = []
	
	# Initialize Order Entry
	if 'order_entry' in subcapabilities:
		try:
			from .order_entry.blueprint import init_subcapability
			init_subcapability(appbuilder)
			initialized.append('order_entry')
		except Exception as e:
			errors.append(f"Error initializing Order Entry: {str(e)}")
	
	# Initialize Order Processing
	if 'order_processing' in subcapabilities:
		try:
			from .order_processing.blueprint import init_subcapability
			init_subcapability(appbuilder)
			initialized.append('order_processing')
		except Exception as e:
			errors.append(f"Error initializing Order Processing: {str(e)}")
	
	# Initialize Pricing & Discounts
	if 'pricing_discounts' in subcapabilities:
		try:
			from .pricing_discounts.blueprint import init_subcapability
			init_subcapability(appbuilder)
			initialized.append('pricing_discounts')
		except Exception as e:
			errors.append(f"Error initializing Pricing & Discounts: {str(e)}")
	
	# Initialize Quotations
	if 'quotations' in subcapabilities:
		try:
			from .quotations.blueprint import init_subcapability
			init_subcapability(appbuilder)
			initialized.append('quotations')
		except Exception as e:
			errors.append(f"Error initializing Quotations: {str(e)}")
	
	# Initialize Sales Forecasting
	if 'sales_forecasting' in subcapabilities:
		try:
			from .sales_forecasting.blueprint import init_subcapability
			init_subcapability(appbuilder)
			initialized.append('sales_forecasting')
		except Exception as e:
			errors.append(f"Error initializing Sales Forecasting: {str(e)}")
	
	# Register main blueprint
	try:
		appbuilder.get_app.register_blueprint(sales_order_management_bp)
	except Exception as e:
		errors.append(f"Error registering main blueprint: {str(e)}")
	
	return {
		'capability': 'Sales & Order Management',
		'code': 'SO',
		'initialized_subcapabilities': initialized,
		'requested_subcapabilities': subcapabilities,
		'errors': errors,
		'success': len(errors) == 0
	}