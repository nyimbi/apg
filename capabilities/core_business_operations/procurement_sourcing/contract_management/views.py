"""
Contract Management Views

Flask-AppBuilder views for contract management.
"""

from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface

from .models import PPCContract, PPCContractLine, PPCContractAmendment, PPCContractRenewal, PPCContractMilestone, PPCContractDocument
from .service import ContractManagementService


class ContractView(ModelView):
	"""Main view for contract management"""
	
	datamodel = SQLAInterface(PPCContract)
	
	list_columns = [
		'contract_number', 'contract_title', 'contract_type', 'vendor_name',
		'effective_date', 'expiration_date', 'status', 'contract_value'
	]
	
	search_columns = [
		'contract_number', 'contract_title', 'vendor_name', 'buyer_name', 'status'
	]
	
	show_columns = [
		'contract_number', 'contract_title', 'contract_type', 'description',
		'vendor_name', 'buyer_name', 'effective_date', 'expiration_date', 'execution_date',
		'status', 'contract_value', 'committed_spend', 'actual_spend', 'currency_code',
		'payment_terms', 'delivery_terms', 'performance_terms', 'warranty_terms',
		'governing_law', 'dispute_resolution', 'confidentiality_required',
		'auto_renewal', 'renewal_notice_days', 'renewal_terms',
		'insurance_required', 'insurance_amount', 'liability_cap',
		'sla_requirements', 'kpi_metrics', 'penalty_clauses', 'notes'
	]
	
	formatters_columns = {
		'contract_value': lambda x: f"${x:,.2f}" if x else "$0.00",
		'committed_spend': lambda x: f"${x:,.2f}" if x else "$0.00",
		'actual_spend': lambda x: f"${x:,.2f}" if x else "$0.00",
		'insurance_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'liability_cap': lambda x: f"${x:,.2f}" if x else "$0.00",
		'status': lambda x: self._format_status_badge(x),
		'auto_renewal': lambda x: '✓' if x else '✗',
		'confidentiality_required': lambda x: '✓' if x else '✗',
		'insurance_required': lambda x: '✓' if x else '✗'
	}
	
	def _format_status_badge(self, status: str) -> str:
		"""Format status as colored badge"""
		color_map = {
			'Draft': 'secondary',
			'Active': 'success',
			'Expired': 'warning',
			'Terminated': 'danger',
			'Renewed': 'primary'
		}
		color = color_map.get(status, 'secondary')
		return f'<span class="badge badge-{color}">{status}</span>'


class ContractLineView(ModelView):
	"""View for contract line items"""
	
	datamodel = SQLAInterface(PPCContractLine)
	
	list_columns = [
		'contract.contract_number', 'line_number', 'description',
		'quantity', 'unit_price', 'line_value', 'pricing_model'
	]
	
	show_columns = [
		'contract.contract_number', 'line_number', 'description', 'item_category',
		'quantity', 'unit_of_measure', 'unit_price', 'line_value',
		'service_level', 'response_time', 'uptime_requirement',
		'performance_standard', 'measurement_criteria',
		'pricing_model', 'price_escalation', 'volume_discounts'
	]
	
	formatters_columns = {
		'unit_price': lambda x: f"${x:,.4f}" if x else "$0.0000",
		'line_value': lambda x: f"${x:,.2f}" if x else "$0.00",
		'quantity': lambda x: f"{x:,.4f}" if x else "0.0000",
		'price_escalation': lambda x: f"{x:.2f}%" if x else "0.00%",
		'uptime_requirement': lambda x: f"{x:.2f}%" if x else "N/A"
	}


class ContractAmendmentView(ModelView):
	"""View for contract amendments"""
	
	datamodel = SQLAInterface(PPCContractAmendment)
	
	list_columns = [
		'amendment_number', 'contract.contract_number', 'amendment_title',
		'amendment_type', 'request_date', 'status', 'value_change'
	]
	
	show_columns = [
		'amendment_number', 'contract.contract_number', 'amendment_title',
		'amendment_type', 'description', 'requested_by_name', 'request_date', 'justification',
		'status', 'approved_by', 'approved_date', 'effective_date', 'executed_date',
		'original_value', 'amended_value', 'value_change',
		'original_expiration_date', 'new_expiration_date'
	]
	
	formatters_columns = {
		'original_value': lambda x: f"${x:,.2f}" if x else "$0.00",
		'amended_value': lambda x: f"${x:,.2f}" if x else "$0.00",
		'value_change': lambda x: f"${x:,.2f}" if x else "$0.00",
		'status': lambda x: self._format_amendment_status_badge(x)
	}
	
	def _format_amendment_status_badge(self, status: str) -> str:
		"""Format amendment status as colored badge"""
		color_map = {
			'Draft': 'secondary',
			'Pending': 'warning',
			'Approved': 'success',
			'Rejected': 'danger',
			'Executed': 'primary'
		}
		color = color_map.get(status, 'secondary')
		return f'<span class="badge badge-{color}">{status}</span>'


class ContractRenewalView(ModelView):
	"""View for contract renewals"""
	
	datamodel = SQLAInterface(PPCContractRenewal)
	
	list_columns = [
		'contract.contract_number', 'renewal_number', 'renewal_type',
		'decision_due_date', 'renewal_status', 'new_contract_value'
	]
	
	show_columns = [
		'contract.contract_number', 'renewal_number', 'renewal_type',
		'notice_date', 'decision_due_date', 'new_start_date', 'new_end_date',
		'renewal_status', 'new_contract_value', 'price_adjustment', 'term_changes',
		'decision_maker_id', 'decision_date', 'decision_rationale',
		'executed', 'execution_date', 'new_contract_id'
	]
	
	formatters_columns = {
		'new_contract_value': lambda x: f"${x:,.2f}" if x else "$0.00",
		'price_adjustment': lambda x: f"{x:.2f}%" if x else "0.00%",
		'renewal_status': lambda x: self._format_renewal_status_badge(x),
		'executed': lambda x: '✓' if x else '✗'
	}
	
	def _format_renewal_status_badge(self, status: str) -> str:
		"""Format renewal status as colored badge"""
		color_map = {
			'Pending': 'warning',
			'Approved': 'success',
			'Declined': 'danger',
			'Executed': 'primary'
		}
		color = color_map.get(status, 'secondary')
		return f'<span class="badge badge-{color}">{status}</span>'


class ContractMilestoneView(ModelView):
	"""View for contract milestones"""
	
	datamodel = SQLAInterface(PPCContractMilestone)
	
	list_columns = [
		'contract.contract_number', 'milestone_name', 'milestone_type',
		'planned_date', 'actual_date', 'status', 'completion_percentage'
	]
	
	show_columns = [
		'contract.contract_number', 'milestone_name', 'milestone_type', 'description',
		'planned_date', 'actual_date', 'due_date', 'status', 'completion_percentage',
		'milestone_value', 'payment_trigger', 'critical_path',
		'acceptance_criteria', 'acceptance_status', 'accepted_by', 'accepted_date', 'notes'
	]
	
	formatters_columns = {
		'milestone_value': lambda x: f"${x:,.2f}" if x else "$0.00",
		'completion_percentage': lambda x: f"{x:.1f}%" if x else "0.0%",
		'status': lambda x: self._format_milestone_status_badge(x),
		'acceptance_status': lambda x: self._format_acceptance_status_badge(x),
		'payment_trigger': lambda x: '✓' if x else '✗',
		'critical_path': lambda x: '✓' if x else '✗'
	}
	
	def _format_milestone_status_badge(self, status: str) -> str:
		"""Format milestone status as colored badge"""
		color_map = {
			'Planned': 'secondary',
			'In Progress': 'info',
			'Completed': 'success',
			'Overdue': 'danger',
			'Cancelled': 'warning'
		}
		color = color_map.get(status, 'secondary')
		return f'<span class="badge badge-{color}">{status}</span>'
	
	def _format_acceptance_status_badge(self, status: str) -> str:
		"""Format acceptance status as colored badge"""
		color_map = {
			'Pending': 'warning',
			'Accepted': 'success',
			'Rejected': 'danger'
		}
		color = color_map.get(status, 'secondary')
		return f'<span class="badge badge-{color}">{status}</span>'


class ContractDocumentView(ModelView):
	"""View for contract documents"""
	
	datamodel = SQLAInterface(PPCContractDocument)
	
	list_columns = [
		'contract.contract_number', 'document_name', 'document_type',
		'document_version', 'uploaded_by_name', 'upload_date', 'is_signed'
	]
	
	formatters_columns = {
		'is_signed': lambda x: '✓' if x else '✗',
		'is_active': lambda x: '✓' if x else '✗',
		'requires_signature': lambda x: '✓' if x else '✗',
		'is_confidential': lambda x: '✓' if x else '✗'
	}


class ContractDashboardView(BaseView):
	"""Dashboard view for contract management"""
	
	route_base = "/contract_management/dashboard"
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Main dashboard view"""
		
		service = ContractManagementService(self.get_tenant_id())
		
		# Get dashboard data
		dashboard_data = {
			'active_contracts': service.get_active_contract_count(),
			'expiring_soon': len(service.get_contracts_expiring_soon()),
			'total_contract_value': float(service.get_total_contract_value()),
			'renewal_rate': service.get_renewal_rate(),
			'overdue_milestones': len(service.get_overdue_milestones()),
			'utilization_summary': service.get_contract_utilization_summary()
		}
		
		return self.render_template(
			'contract_dashboard.html',
			dashboard_data=dashboard_data,
			title="Contract Management Dashboard"
		)
	
	def get_tenant_id(self) -> str:
		return "default_tenant"