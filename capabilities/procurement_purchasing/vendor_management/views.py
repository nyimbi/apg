"""
Vendor Management Views

Flask-AppBuilder views for vendor management.
"""

from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface

from .models import PPVVendor, PPVVendorContact, PPVVendorPerformance, PPVVendorCategory
from .service import VendorManagementService


class VendorView(ModelView):
	"""Main view for vendor management"""
	
	datamodel = SQLAInterface(PPVVendor)
	
	list_columns = [
		'vendor_number', 'vendor_name', 'vendor_type', 'industry',
		'is_active', 'is_approved', 'is_preferred', 'overall_rating'
	]
	
	search_columns = [
		'vendor_number', 'vendor_name', 'legal_name', 'tax_id', 'industry'
	]
	
	show_columns = [
		'vendor_number', 'vendor_name', 'legal_name', 'vendor_type',
		'tax_id', 'duns_number', 'website', 'industry', 'business_size',
		'address_line1', 'city', 'state_province', 'country',
		'credit_rating', 'credit_limit', 'payment_terms',
		'is_active', 'is_approved', 'is_preferred',
		'is_minority_owned', 'is_woman_owned', 'is_veteran_owned',
		'overall_rating', 'quality_rating', 'delivery_rating', 'service_rating'
	]
	
	formatters_columns = {
		'credit_limit': lambda x: f"${x:,.2f}" if x else "$0.00",
		'overall_rating': lambda x: f"{x:.2f}/5.00" if x else "Not Rated",
		'quality_rating': lambda x: f"{x:.2f}/5.00" if x else "Not Rated",
		'delivery_rating': lambda x: f"{x:.2f}/5.00" if x else "Not Rated",
		'service_rating': lambda x: f"{x:.2f}/5.00" if x else "Not Rated",
		'is_active': lambda x: '✓' if x else '✗',
		'is_approved': lambda x: '✓' if x else '✗',
		'is_preferred': lambda x: '✓' if x else '✗'
	}


class VendorContactView(ModelView):
	"""View for vendor contacts"""
	
	datamodel = SQLAInterface(PPVVendorContact)
	
	list_columns = [
		'vendor.vendor_name', 'contact_type', 'first_name', 'last_name',
		'title', 'email', 'phone', 'is_primary'
	]
	
	formatters_columns = {
		'is_primary': lambda x: '✓' if x else '✗'
	}


class VendorPerformanceView(ModelView):
	"""View for vendor performance"""
	
	datamodel = SQLAInterface(PPVVendorPerformance)
	
	list_columns = [
		'vendor.vendor_name', 'period_start', 'period_end',
		'total_orders', 'total_value', 'delivery_performance', 'overall_score'
	]
	
	formatters_columns = {
		'total_value': lambda x: f"${x:,.2f}" if x else "$0.00",
		'delivery_performance': lambda x: f"{x:.1f}%" if x else "0.0%",
		'overall_score': lambda x: f"{x:.2f}/5.00" if x else "Not Rated"
	}


class VendorDashboardView(BaseView):
	"""Dashboard view for vendor management"""
	
	route_base = "/vendor_management/dashboard"
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Main dashboard view"""
		
		service = VendorManagementService(self.get_tenant_id())
		
		dashboard_data = {
			'active_vendors': service.get_active_vendor_count(),
			'top_vendors': service.get_top_vendors_by_spend(),
			'avg_performance': service.get_average_performance_score(),
			'new_vendors_this_month': service.get_new_vendor_count(),
			'poor_performers': len(service.get_poor_performing_vendors())
		}
		
		return self.render_template(
			'vendor_dashboard.html',
			dashboard_data=dashboard_data,
			title="Vendor Management Dashboard"
		)
	
	def get_tenant_id(self) -> str:
		return "default_tenant"