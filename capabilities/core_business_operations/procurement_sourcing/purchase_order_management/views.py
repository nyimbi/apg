"""
Purchase Order Management Views

Flask-AppBuilder views for purchase order management.
"""

from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface

from .models import PPOPurchaseOrder, PPOPurchaseOrderLine, PPOReceipt, PPOReceiptLine, PPOThreeWayMatch, PPOChangeOrder
from .service import PurchaseOrderService


class PurchaseOrderView(ModelView):
	"""Main view for purchase order management"""
	
	datamodel = SQLAInterface(PPOPurchaseOrder)
	
	# List view configuration
	list_columns = [
		'po_number', 'title', 'vendor_name', 'buyer_name',
		'po_date', 'required_date', 'status', 'total_amount'
	]
	
	search_columns = [
		'po_number', 'title', 'vendor_name', 'buyer_name', 'status'
	]
	
	# Show view configuration
	show_columns = [
		'po_number', 'title', 'description', 'vendor_name', 'vendor_contact',
		'buyer_name', 'department', 'po_date', 'required_date', 'promised_date',
		'status', 'currency_code', 'subtotal_amount', 'tax_amount', 'freight_amount', 'total_amount',
		'ship_to_location', 'delivery_terms', 'payment_terms',
		'approved', 'approved_by', 'approved_date',
		'received_amount', 'invoiced_amount',
		'special_instructions', 'notes'
	]
	
	# Edit/Add view configuration
	edit_columns = [
		'title', 'description', 'vendor_id', 'vendor_name',
		'required_date', 'promised_date', 'ship_to_location',
		'delivery_terms', 'payment_terms', 'freight_terms',
		'special_instructions', 'notes'
	]
	
	add_columns = [
		'title', 'description', 'vendor_id', 'vendor_name', 'vendor_contact',
		'buyer_name', 'department', 'required_date',
		'ship_to_location', 'delivery_terms', 'payment_terms',
		'special_instructions', 'notes'
	]
	
	# Formatters
	formatters_columns = {
		'total_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'subtotal_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'tax_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'freight_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'received_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'invoiced_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'status': lambda x: self._format_status_badge(x),
		'approved': lambda x: '✓' if x else '✗'
	}
	
	def _format_status_badge(self, status: str) -> str:
		"""Format status as colored badge"""
		color_map = {
			'Draft': 'secondary',
			'Approved': 'success',
			'Open': 'info',
			'Closed': 'primary',
			'Cancelled': 'danger'
		}
		color = color_map.get(status, 'secondary')
		return f'<span class="badge badge-{color}">{status}</span>'


class PurchaseOrderLineView(ModelView):
	"""View for purchase order line items"""
	
	datamodel = SQLAInterface(PPOPurchaseOrderLine)
	
	list_columns = [
		'purchase_order.po_number', 'line_number', 'description',
		'item_code', 'quantity_ordered', 'unit_price', 'line_amount'
	]
	
	show_columns = [
		'purchase_order.po_number', 'line_number', 'description',
		'item_code', 'item_description', 'quantity_ordered', 'quantity_received', 'quantity_invoiced',
		'unit_of_measure', 'unit_price', 'line_amount',
		'tax_code', 'tax_rate', 'tax_amount',
		'gl_account_id', 'cost_center', 'project_id',
		'required_date', 'promised_date', 'line_status'
	]
	
	formatters_columns = {
		'line_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'unit_price': lambda x: f"${x:,.4f}" if x else "$0.0000",
		'quantity_ordered': lambda x: f"{x:,.4f}" if x else "0.0000",
		'quantity_received': lambda x: f"{x:,.4f}" if x else "0.0000",
		'quantity_invoiced': lambda x: f"{x:,.4f}" if x else "0.0000"
	}


class ReceiptView(ModelView):
	"""View for goods receipts"""
	
	datamodel = SQLAInterface(PPOReceipt)
	
	list_columns = [
		'receipt_number', 'purchase_order.po_number', 'receipt_date',
		'received_by_name', 'status'
	]
	
	show_columns = [
		'receipt_number', 'purchase_order.po_number', 'receipt_date',
		'received_by_name', 'status', 'packing_slip_number',
		'carrier', 'tracking_number', 'notes'
	]
	
	formatters_columns = {
		'status': lambda x: self._format_status_badge(x)
	}
	
	def _format_status_badge(self, status: str) -> str:
		"""Format status as colored badge"""
		color_map = {
			'Draft': 'secondary',
			'Posted': 'success',
			'Cancelled': 'danger'
		}
		color = color_map.get(status, 'secondary')
		return f'<span class="badge badge-{color}">{status}</span>'


class ReceiptLineView(ModelView):
	"""View for receipt line items"""
	
	datamodel = SQLAInterface(PPOReceiptLine)
	
	list_columns = [
		'receipt.receipt_number', 'line_number', 'quantity_received',
		'unit_of_measure', 'quality_status'
	]
	
	formatters_columns = {
		'quantity_received': lambda x: f"{x:,.4f}" if x else "0.0000"
	}


class ThreeWayMatchView(ModelView):
	"""View for three-way matching"""
	
	datamodel = SQLAInterface(PPOThreeWayMatch)
	
	list_columns = [
		'purchase_order.po_number', 'receipt.receipt_number',
		'match_date', 'match_status', 'has_exceptions'
	]
	
	formatters_columns = {
		'match_status': lambda x: self._format_match_status_badge(x),
		'has_exceptions': lambda x: '⚠️' if x else '✓'
	}
	
	def _format_match_status_badge(self, status: str) -> str:
		"""Format match status as colored badge"""
		color_map = {
			'Matched': 'success',
			'Exception': 'warning',
			'Resolved': 'primary'
		}
		color = color_map.get(status, 'secondary')
		return f'<span class="badge badge-{color}">{status}</span>'


class ChangeOrderView(ModelView):
	"""View for change orders"""
	
	datamodel = SQLAInterface(PPOChangeOrder)
	
	list_columns = [
		'change_order_number', 'purchase_order.po_number', 'change_type',
		'requested_date', 'status', 'amount_difference'
	]
	
	formatters_columns = {
		'amount_difference': lambda x: f"${x:,.2f}" if x else "$0.00",
		'status': lambda x: self._format_change_status_badge(x)
	}
	
	def _format_change_status_badge(self, status: str) -> str:
		"""Format change status as colored badge"""
		color_map = {
			'Pending': 'warning',
			'Approved': 'success',
			'Rejected': 'danger',
			'Implemented': 'primary'
		}
		color = color_map.get(status, 'secondary')
		return f'<span class="badge badge-{color}">{status}</span>'


class PurchaseOrderDashboardView(BaseView):
	"""Dashboard view for purchase order management"""
	
	route_base = "/purchase_order_management/dashboard"
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Main dashboard view"""
		
		service = PurchaseOrderService(self.get_tenant_id())
		
		# Get dashboard data
		dashboard_data = self._get_dashboard_data(service)
		
		return self.render_template(
			'purchase_order_dashboard.html',
			dashboard_data=dashboard_data,
			title="Purchase Order Management Dashboard"
		)
	
	def _get_dashboard_data(self, service: PurchaseOrderService) -> dict:
		"""Get data for dashboard"""
		
		return {
			'open_pos': len(service.get_purchase_orders_by_status('Open')),
			'pending_receipt': len(service.get_purchase_orders_needing_receipt()),
			'total_po_value_ytd': float(service.get_total_po_value_ytd()),
			'avg_processing_time': service.get_avg_processing_time(),
			'overdue_receipts': len(service.get_overdue_receipts())
		}
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"
