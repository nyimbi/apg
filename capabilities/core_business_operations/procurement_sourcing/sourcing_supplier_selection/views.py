"""
Sourcing & Supplier Selection Views

Flask-AppBuilder views for RFQ/RFP management and supplier evaluation.
"""

from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface

from .models import PPSRFQHeader, PPSRFQLine, PPSBid, PPSBidLine, PPSSupplierEvaluation, PPSAwardRecommendation
from .service import SourcingSupplierSelectionService


class RFQHeaderView(ModelView):
	"""Main view for RFQ management"""
	
	datamodel = SQLAInterface(PPSRFQHeader)
	
	list_columns = [
		'rfq_number', 'rfq_title', 'rfq_type', 'sourcing_manager_name',
		'issue_date', 'response_due_date', 'status', 'estimated_value'
	]
	
	search_columns = [
		'rfq_number', 'rfq_title', 'sourcing_manager_name', 'category', 'status'
	]
	
	show_columns = [
		'rfq_number', 'rfq_title', 'rfq_type', 'description',
		'sourcing_manager_name', 'department', 'category',
		'issue_date', 'response_due_date', 'evaluation_complete_date', 'award_date',
		'status', 'evaluation_method', 'estimated_value', 'currency_code',
		'terms_and_conditions', 'payment_terms', 'delivery_terms'
	]
	
	formatters_columns = {
		'estimated_value': lambda x: f"${x:,.2f}" if x else "$0.00",
		'status': lambda x: self._format_status_badge(x)
	}
	
	def _format_status_badge(self, status: str) -> str:
		"""Format status as colored badge"""
		color_map = {
			'Draft': 'secondary',
			'Issued': 'info',
			'Closed': 'warning',
			'Awarded': 'success',
			'Cancelled': 'danger'
		}
		color = color_map.get(status, 'secondary')
		return f'<span class="badge badge-{color}">{status}</span>'


class BidView(ModelView):
	"""View for supplier bids"""
	
	datamodel = SQLAInterface(PPSBid)
	
	list_columns = [
		'bid_number', 'rfq_header.rfq_number', 'supplier_name',
		'submitted_date', 'status', 'total_bid_amount', 'overall_score', 'rank'
	]
	
	search_columns = [
		'bid_number', 'supplier_name', 'status'
	]
	
	show_columns = [
		'bid_number', 'rfq_header.rfq_number', 'supplier_name',
		'submitted_date', 'submitted_by', 'submission_method', 'status',
		'total_bid_amount', 'currency_code',
		'technical_score', 'commercial_score', 'overall_score', 'rank',
		'bid_valid_until', 'supplier_comments', 'internal_notes',
		'awarded', 'award_amount', 'award_date'
	]
	
	formatters_columns = {
		'total_bid_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'award_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'technical_score': lambda x: f"{x:.2f}" if x else "0.00",
		'commercial_score': lambda x: f"{x:.2f}" if x else "0.00",
		'overall_score': lambda x: f"{x:.2f}" if x else "0.00",
		'status': lambda x: self._format_bid_status_badge(x),
		'awarded': lambda x: '✓' if x else '✗'
	}
	
	def _format_bid_status_badge(self, status: str) -> str:
		"""Format bid status as colored badge"""
		color_map = {
			'Draft': 'secondary',
			'Submitted': 'info',
			'Under Review': 'warning',
			'Accepted': 'success',
			'Rejected': 'danger'
		}
		color = color_map.get(status, 'secondary')
		return f'<span class="badge badge-{color}">{status}</span>'


class SupplierEvaluationView(ModelView):
	"""View for supplier evaluations"""
	
	datamodel = SQLAInterface(PPSSupplierEvaluation)
	
	list_columns = [
		'rfq_header.rfq_number', 'supplier_id', 'evaluator_name',
		'evaluation_date', 'overall_score', 'recommendation'
	]
	
	show_columns = [
		'rfq_header.rfq_number', 'supplier_id', 'evaluator_name', 'evaluation_date',
		'technical_score', 'commercial_score', 'delivery_score', 'quality_score', 'service_score', 'overall_score',
		'recommendation', 'recommendation_reason', 'risk_level', 'risk_factors',
		'strengths', 'weaknesses', 'additional_comments'
	]
	
	formatters_columns = {
		'technical_score': lambda x: f"{x:.2f}" if x else "0.00",
		'commercial_score': lambda x: f"{x:.2f}" if x else "0.00",
		'delivery_score': lambda x: f"{x:.2f}" if x else "0.00",
		'quality_score': lambda x: f"{x:.2f}" if x else "0.00",
		'service_score': lambda x: f"{x:.2f}" if x else "0.00",
		'overall_score': lambda x: f"{x:.2f}" if x else "0.00",
		'recommendation': lambda x: self._format_recommendation_badge(x),
		'risk_level': lambda x: self._format_risk_badge(x)
	}
	
	def _format_recommendation_badge(self, recommendation: str) -> str:
		"""Format recommendation as colored badge"""
		color_map = {
			'Recommend': 'success',
			'Do Not Recommend': 'danger',
			'Conditional': 'warning',
			'Under Review': 'info'
		}
		color = color_map.get(recommendation, 'secondary')
		return f'<span class="badge badge-{color}">{recommendation}</span>'
	
	def _format_risk_badge(self, risk_level: str) -> str:
		"""Format risk level as colored badge"""
		color_map = {
			'Low': 'success',
			'Medium': 'warning',
			'High': 'danger'
		}
		color = color_map.get(risk_level, 'secondary')
		return f'<span class="badge badge-{color}">{risk_level}</span>'


class AwardRecommendationView(ModelView):
	"""View for award recommendations"""
	
	datamodel = SQLAInterface(PPSAwardRecommendation)
	
	list_columns = [
		'rfq_header.rfq_number', 'recommended_supplier_name', 'recommender_name',
		'recommendation_date', 'recommended_amount', 'status'
	]
	
	show_columns = [
		'rfq_header.rfq_number', 'recommended_supplier_name', 'recommender_name', 'recommendation_date',
		'recommended_amount', 'estimated_savings', 'savings_percentage',
		'award_justification', 'key_differentiators', 'risk_mitigation',
		'status', 'approved_by', 'approved_date', 'implemented', 'implementation_date'
	]
	
	formatters_columns = {
		'recommended_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'estimated_savings': lambda x: f"${x:,.2f}" if x else "$0.00",
		'savings_percentage': lambda x: f"{x:.1f}%" if x else "0.0%",
		'status': lambda x: self._format_status_badge(x),
		'implemented': lambda x: '✓' if x else '✗'
	}
	
	def _format_status_badge(self, status: str) -> str:
		"""Format status as colored badge"""
		color_map = {
			'Pending': 'warning',
			'Approved': 'success',
			'Rejected': 'danger'
		}
		color = color_map.get(status, 'secondary')
		return f'<span class="badge badge-{color}">{status}</span>'


class SourcingDashboardView(BaseView):
	"""Dashboard view for sourcing and supplier selection"""
	
	route_base = "/sourcing_supplier_selection/dashboard"
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Main dashboard view"""
		
		service = SourcingSupplierSelectionService(self.get_tenant_id())
		
		dashboard_data = {
			'active_rfqs': len(service.get_active_rfqs()),
			'pending_evaluations': len(service.get_pending_evaluations()),
			'total_sourcing_value_ytd': float(service.get_total_sourcing_value_ytd()),
			'avg_bid_count': service.get_avg_bid_count_per_rfq(),
			'avg_evaluation_time': service.get_avg_evaluation_time(),
			'cost_savings_ytd': float(service.get_cost_savings_ytd())
		}
		
		return self.render_template(
			'sourcing_dashboard.html',
			dashboard_data=dashboard_data,
			title="Sourcing & Supplier Selection Dashboard"
		)
	
	def get_tenant_id(self) -> str:
		return "default_tenant"