"""
APG Core Financials - Accounts Payable Blueprint

Flask-AppBuilder blueprint with complete APG platform integration including
composition engine registration, authentication, audit compliance, and
capability orchestration.

© 2025 Datacraft. All rights reserved.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from flask import Blueprint, request, jsonify, render_template, flash, redirect, url_for
from flask_appbuilder import BaseView, ModelView, expose, has_access
from flask_appbuilder.actions import action
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from werkzeug.exceptions import BadRequest, NotFound

from .models import (
	APVendor, APInvoice, APPayment, APApprovalWorkflow, APExpenseReport,
	VendorStatus, VendorType, InvoiceStatus, PaymentStatus, PaymentMethod,
	ApprovalStatus, MatchingStatus
)
from .service import (
	APVendorService, APInvoiceService, APPaymentService,
	APWorkflowService, APAnalyticsService
)
from .views import (
	APDashboardSummary, APVendorSummaryView, APVendorDetailView,
	APInvoiceSummaryView, APInvoiceDetailView, APPaymentSummaryView,
	APApprovalWorkflowView, APAgingReportView, APCashFlowForecastView,
	APVendorCreateForm, APInvoiceProcessingForm, APPaymentCreateForm
)
from .api import create_ap_api


class APGAccountsPayableBlueprint:
	"""APG Accounts Payable capability blueprint with platform integration"""
	
	def __init__(self, appbuilder):
		self.appbuilder = appbuilder
		self.capability_id = "core_financials.accounts_payable"
		self.version = "2.0.0"
		
		# APG service integrations
		self.auth_service = None
		self.audit_service = None
		self.ai_service = None
		self.collaboration_service = None
		self.document_service = None
		
		# AP service instances
		self.vendor_service = APVendorService()
		self.invoice_service = APInvoiceService()
		self.payment_service = APPaymentService()
		self.workflow_service = APWorkflowService()
		self.analytics_service = APAnalyticsService()
	
	def register_with_apg(self) -> None:
		"""Register capability with APG composition engine"""
		try:
			# Register capability metadata
			self.appbuilder.composition_engine.register_capability(
				capability_id=self.capability_id,
				version=self.version,
				name="Accounts Payable",
				description="Enterprise-grade accounts payable automation with AI integration",
				category="core_financials",
				dependencies=self._get_dependencies(),
				permissions=self._get_permissions(),
				menu_items=self._get_menu_items(),
				api_endpoints=self._get_api_endpoints(),
				configuration_schema=self._get_configuration_schema()
			)
			
			# Setup APG integrations
			self._setup_auth_integration()
			self._setup_audit_integration()
			self._setup_ai_integration()
			self._setup_collaboration_integration()
			self._setup_document_integration()
			
			# Register Flask views
			self._register_views()
			
			# Register API endpoints
			self._register_api()
			
			await self._log_blueprint_initialization(self.capability_id, "registered successfully")
			
		except Exception as e:
			await self._log_blueprint_initialization(self.capability_id, f"registration failed: {str(e)}")
			raise
	
	def _get_dependencies(self) -> List[Dict[str, Any]]:
		"""Get capability dependencies"""
		return [
			{
				"capability_id": "auth_rbac",
				"version": ">=1.0.0",
				"required": True,
				"integration_points": [
					"user_authentication",
					"role_based_permissions",
					"multi_tenant_isolation"
				]
			},
			{
				"capability_id": "audit_compliance",
				"version": ">=1.0.0",
				"required": True,
				"integration_points": [
					"transaction_audit_trails",
					"compliance_reporting",
					"data_retention_policies"
				]
			},
			{
				"capability_id": "core_financials.general_ledger",
				"version": ">=1.0.0",
				"required": True,
				"integration_points": [
					"journal_entry_posting",
					"account_balance_updates",
					"period_end_closing"
				]
			},
			{
				"capability_id": "document_management",
				"version": ">=1.0.0",
				"required": True,
				"integration_points": [
					"invoice_document_storage",
					"version_control",
					"electronic_signatures"
				]
			},
			{
				"capability_id": "ai_orchestration",
				"version": ">=1.0.0",
				"required": False,
				"integration_points": [
					"intelligent_invoice_processing",
					"gl_code_prediction",
					"fraud_detection"
				]
			},
			{
				"capability_id": "computer_vision",
				"version": ">=1.0.0",
				"required": False,
				"integration_points": [
					"invoice_ocr_processing",
					"document_field_extraction",
					"image_enhancement"
				]
			},
			{
				"capability_id": "federated_learning",
				"version": ">=1.0.0",
				"required": False,
				"integration_points": [
					"predictive_analytics",
					"pattern_recognition",
					"continuous_model_improvement"
				]
			},
			{
				"capability_id": "real_time_collaboration",
				"version": ">=1.0.0",
				"required": False,
				"integration_points": [
					"approval_workflow_notifications",
					"real_time_status_updates",
					"collaborative_document_review"
				]
			}
		]
	
	def _get_permissions(self) -> List[Dict[str, Any]]:
		"""Get capability-specific permissions"""
		return [
			{
				"permission": "ap.read",
				"name": "AP Read Access",
				"description": "View accounts payable data and reports"
			},
			{
				"permission": "ap.write",
				"name": "AP Write Access",
				"description": "Create and modify AP transactions"
			},
			{
				"permission": "ap.approve_invoice",
				"name": "Invoice Approval",
				"description": "Approve invoices within authorization limits"
			},
			{
				"permission": "ap.process_payment",
				"name": "Payment Processing",
				"description": "Process vendor payments"
			},
			{
				"permission": "ap.vendor_admin",
				"name": "Vendor Administration",
				"description": "Manage vendor master data"
			},
			{
				"permission": "ap.admin",
				"name": "AP Administration",
				"description": "Full administrative access to AP module"
			}
		]
	
	def _get_menu_items(self) -> List[Dict[str, Any]]:
		"""Get menu items for APG navigation"""
		return [
			{
				"name": "Accounts Payable",
				"category": "Financials",
				"icon": "fas fa-file-invoice-dollar",
				"order": 100,
				"children": [
					{
						"name": "Dashboard",
						"href": "/ap/dashboard",
						"permission": "ap.read",
						"icon": "fas fa-tachometer-alt"
					},
					{
						"name": "Vendors",
						"href": "/ap/vendors",
						"permission": "ap.read",
						"icon": "fas fa-users"
					},
					{
						"name": "Invoices",
						"href": "/ap/invoices",
						"permission": "ap.read",
						"icon": "fas fa-file-invoice"
					},
					{
						"name": "Payments",
						"href": "/ap/payments",
						"permission": "ap.read",
						"icon": "fas fa-credit-card"
					},
					{
						"name": "Approval Workflows",
						"href": "/ap/workflows",
						"permission": "ap.approve_invoice",
						"icon": "fas fa-tasks"
					},
					{
						"name": "Reports",
						"href": "/ap/reports",
						"permission": "ap.read",
						"icon": "fas fa-chart-bar",
						"children": [
							{
								"name": "Aging Report",
								"href": "/ap/reports/aging",
								"permission": "ap.read"
							},
							{
								"name": "Cash Flow Forecast",
								"href": "/ap/reports/cash_flow",
								"permission": "ap.read"
							},
							{
								"name": "Vendor Performance",
								"href": "/ap/reports/vendor_performance",
								"permission": "ap.read"
							}
						]
					}
				]
			}
		]
	
	def _get_api_endpoints(self) -> List[Dict[str, Any]]:
		"""Get API endpoint definitions"""
		return [
			{
				"path": "/api/v1/core_financials/accounts_payable/vendors",
				"methods": ["GET", "POST"],
				"permission": "ap.read",
				"rate_limit": "1000/hour"
			},
			{
				"path": "/api/v1/core_financials/accounts_payable/invoices",
				"methods": ["GET", "POST"],
				"permission": "ap.read",
				"rate_limit": "1000/hour"
			},
			{
				"path": "/api/v1/core_financials/accounts_payable/payments",
				"methods": ["GET", "POST"],
				"permission": "ap.read",
				"rate_limit": "1000/hour"
			},
			{
				"path": "/api/v1/core_financials/accounts_payable/workflows",
				"methods": ["GET", "POST"],
				"permission": "ap.read",
				"rate_limit": "500/hour"
			},
			{
				"path": "/api/v1/core_financials/accounts_payable/analytics/*",
				"methods": ["GET"],
				"permission": "ap.read",
				"rate_limit": "100/hour"
			}
		]
	
	def _get_configuration_schema(self) -> Dict[str, Any]:
		"""Get capability configuration schema"""
		return {
			"type": "object",
			"properties": {
				"invoice_processing": {
					"type": "object",
					"properties": {
						"auto_ocr_threshold": {
							"type": "number",
							"minimum": 0.0,
							"maximum": 1.0,
							"default": 0.95,
							"description": "OCR confidence threshold for automatic processing"
						},
						"three_way_matching_required": {
							"type": "boolean",
							"default": True,
							"description": "Require three-way matching for all invoices"
						},
						"duplicate_detection_enabled": {
							"type": "boolean",
							"default": True,
							"description": "Enable AI-powered duplicate invoice detection"
						}
					}
				},
				"payment_processing": {
					"type": "object",
					"properties": {
						"default_payment_method": {
							"type": "string",
							"enum": ["ach", "wire", "check", "virtual_card"],
							"default": "ach",
							"description": "Default payment method for new payments"
						},
						"early_discount_optimization": {
							"type": "boolean",
							"default": True,
							"description": "Automatically optimize for early payment discounts"
						},
						"multi_currency_enabled": {
							"type": "boolean",
							"default": True,
							"description": "Enable multi-currency payment processing"
						}
					}
				},
				"approval_workflows": {
					"type": "object",
					"properties": {
						"default_approval_threshold": {
							"type": "number",
							"minimum": 0,
							"default": 10000,
							"description": "Default approval threshold amount"
						},
						"escalation_timeout_hours": {
							"type": "integer",
							"minimum": 1,
							"default": 24,
							"description": "Hours before workflow escalation"
						},
						"mobile_approvals_enabled": {
							"type": "boolean",
							"default": True,
							"description": "Enable mobile approval capabilities"
						}
					}
				}
			}
		}
	
	def _setup_auth_integration(self) -> None:
		"""Configure APG auth_rbac integration"""
		try:
			self.auth_service = self.appbuilder.get_capability("auth_rbac")
			
			if self.auth_service:
				# Register AP-specific permissions
				permissions = [
					"ap.read", "ap.write", "ap.approve_invoice",
					"ap.process_payment", "ap.vendor_admin", "ap.admin"
				]
				
				for permission in permissions:
					self.auth_service.register_permission(
						permission=permission,
						capability_id=self.capability_id
					)
				
				# Register role-based access patterns
				self._register_rbac_patterns()
				
				print("APG auth_rbac integration configured successfully")
			else:
				print("Warning: APG auth_rbac capability not available")
				
		except Exception as e:
			print(f"Failed to setup auth integration: {str(e)}")
	
	def _setup_audit_integration(self) -> None:
		"""Configure APG audit_compliance integration"""
		try:
			self.audit_service = self.appbuilder.get_capability("audit_compliance")
			
			if self.audit_service:
				# Register audit event types
				audit_events = [
					"vendor.created", "vendor.updated", "vendor.suspended",
					"invoice.received", "invoice.processed", "invoice.approved", "invoice.rejected",
					"payment.created", "payment.processed", "payment.completed", "payment.failed",
					"workflow.initiated", "workflow.approved", "workflow.escalated"
				]
				
				for event_type in audit_events:
					self.audit_service.register_event_type(
						event_type=event_type,
						capability_id=self.capability_id,
						retention_days=2555  # 7 years for financial records
					)
				
				# Configure compliance rules
				self._register_compliance_rules()
				
				print("APG audit_compliance integration configured successfully")
			else:
				print("Warning: APG audit_compliance capability not available")
				
		except Exception as e:
			print(f"Failed to setup audit integration: {str(e)}")
	
	def _setup_ai_integration(self) -> None:
		"""Configure APG AI capabilities integration"""
		try:
			self.ai_service = self.appbuilder.get_capability("ai_orchestration")
			computer_vision_service = self.appbuilder.get_capability("computer_vision")
			federated_learning_service = self.appbuilder.get_capability("federated_learning")
			
			if self.ai_service:
				# Register AI workflows
				ai_workflows = [
					{
						"workflow_id": "invoice_processing",
						"name": "AI-Powered Invoice Processing",
						"description": "Extract and validate invoice data using computer vision and NLP",
						"input_types": ["image", "pdf"],
						"output_schema": "invoice_extraction_result",
						"confidence_threshold": 0.95
					},
					{
						"workflow_id": "gl_code_prediction",
						"name": "GL Code Prediction",
						"description": "Predict appropriate GL codes based on invoice content and history",
						"input_types": ["invoice_line_item"],
						"output_schema": "gl_code_prediction",
						"confidence_threshold": 0.85
					},
					{
						"workflow_id": "fraud_detection",
						"name": "Invoice Fraud Detection",
						"description": "Detect potentially fraudulent invoices using pattern analysis",
						"input_types": ["invoice_data"],
						"output_schema": "fraud_risk_assessment",
						"confidence_threshold": 0.75
					}
				]
				
				for workflow in ai_workflows:
					self.ai_service.register_workflow(
						workflow_id=workflow["workflow_id"],
						capability_id=self.capability_id,
						**workflow
					)
				
				print("APG AI integration configured successfully")
			else:
				print("Warning: APG AI capabilities not available")
				
		except Exception as e:
			print(f"Failed to setup AI integration: {str(e)}")
	
	def _setup_collaboration_integration(self) -> None:
		"""Configure APG real_time_collaboration integration"""
		try:
			self.collaboration_service = self.appbuilder.get_capability("real_time_collaboration")
			
			if self.collaboration_service:
				# Register collaboration channels
				channels = [
					{
						"channel_id": "ap_approvals",
						"name": "AP Approvals",
						"description": "Real-time approval workflow notifications",
						"participants": ["ap_managers", "ap_approvers"]
					},
					{
						"channel_id": "ap_exceptions",
						"name": "AP Exceptions",
						"description": "Invoice processing exceptions and alerts",
						"participants": ["ap_processors", "ap_managers"]
					},
					{
						"channel_id": "vendor_communications",
						"name": "Vendor Communications",
						"description": "Direct communication with vendor portals",
						"participants": ["ap_team", "vendors"]
					}
				]
				
				for channel in channels:
					self.collaboration_service.register_channel(
						channel_id=channel["channel_id"],
						capability_id=self.capability_id,
						**channel
					)
				
				print("APG collaboration integration configured successfully")
			else:
				print("Warning: APG real_time_collaboration capability not available")
				
		except Exception as e:
			print(f"Failed to setup collaboration integration: {str(e)}")
	
	def _setup_document_integration(self) -> None:
		"""Configure APG document_management integration"""
		try:
			self.document_service = self.appbuilder.get_capability("document_management")
			
			if self.document_service:
				# Register document types
				document_types = [
					{
						"type_id": "vendor_invoice",
						"name": "Vendor Invoice",
						"description": "Invoices received from vendors",
						"retention_years": 7,
						"encryption_required": True,
						"version_control": True
					},
					{
						"type_id": "payment_voucher",
						"name": "Payment Voucher",
						"description": "Payment authorization documents",
						"retention_years": 7,
						"encryption_required": True,
						"version_control": True
					},
					{
						"type_id": "vendor_contract",
						"name": "Vendor Contract",
						"description": "Legal agreements with vendors",
						"retention_years": 10,
						"encryption_required": True,
						"version_control": True
					}
				]
				
				for doc_type in document_types:
					self.document_service.register_document_type(
						type_id=doc_type["type_id"],
						capability_id=self.capability_id,
						**doc_type
					)
				
				print("APG document_management integration configured successfully")
			else:
				print("Warning: APG document_management capability not available")
				
		except Exception as e:
			print(f"Failed to setup document integration: {str(e)}")
	
	def _register_rbac_patterns(self) -> None:
		"""Register role-based access control patterns"""
		if not self.auth_service:
			return
		
		try:
			# Define AP-specific roles
			roles = [
				{
					"role": "ap_clerk",
					"permissions": ["ap.read", "ap.write"],
					"description": "AP data entry and basic operations"
				},
				{
					"role": "ap_processor",
					"permissions": ["ap.read", "ap.write", "ap.vendor_admin"],
					"description": "Invoice processing and vendor management"
				},
				{
					"role": "ap_approver",
					"permissions": ["ap.read", "ap.approve_invoice"],
					"description": "Invoice approval authority",
					"constraints": {
						"max_approval_amount": 50000
					}
				},
				{
					"role": "ap_manager",
					"permissions": ["ap.read", "ap.write", "ap.approve_invoice", "ap.process_payment", "ap.vendor_admin"],
					"description": "Full AP management authority"
				},
				{
					"role": "ap_admin",
					"permissions": ["ap.admin"],
					"description": "Complete AP administrative access"
				}
			]
			
			for role in roles:
				self.auth_service.register_role(
					role=role["role"],
					capability_id=self.capability_id,
					**role
				)
			
			# Define attribute-based access patterns
			abac_policies = [
				{
					"policy_id": "amount_based_approval",
					"description": "Approval required based on invoice amount",
					"condition": "invoice.total_amount > user.approval_limit",
					"effect": "require_approval"
				},
				{
					"policy_id": "vendor_restriction",
					"description": "Restrict access to specific vendors by department",
					"condition": "vendor.department != user.department AND user.role != 'ap_admin'",
					"effect": "deny"
				},
				{
					"policy_id": "payment_method_restriction",
					"description": "Restrict certain payment methods by role",
					"condition": "payment.method IN ('wire', 'check') AND user.role NOT IN ('ap_manager', 'ap_admin')",
					"effect": "deny"
				}
			]
			
			for policy in abac_policies:
				self.auth_service.register_abac_policy(
					policy_id=policy["policy_id"],
					capability_id=self.capability_id,
					**policy
				)
			
		except Exception as e:
			print(f"Failed to register RBAC patterns: {str(e)}")
	
	def _register_compliance_rules(self) -> None:
		"""Register compliance monitoring rules"""
		if not self.audit_service:
			return
		
		try:
			compliance_rules = [
				{
					"rule_id": "sox_segregation_of_duties",
					"name": "SOX Segregation of Duties",
					"description": "Ensure proper segregation of duties for financial transactions",
					"regulation": "SOX",
					"rule_type": "segregation_of_duties",
					"conditions": [
						"invoice.created_by != invoice.approved_by",
						"payment.created_by != payment.approved_by"
					]
				},
				{
					"rule_id": "gdpr_data_retention",
					"name": "GDPR Data Retention",
					"description": "Ensure proper data retention for vendor personal information",
					"regulation": "GDPR",
					"rule_type": "data_retention",
					"retention_period": "7_years",
					"conditions": [
						"vendor.personal_data_exists = true"
					]
				},
				{
					"rule_id": "payment_authorization_limit",
					"name": "Payment Authorization Limits",
					"description": "Enforce payment authorization limits",
					"regulation": "Internal Controls",
					"rule_type": "authorization_limit",
					"conditions": [
						"payment.amount > user.authorization_limit REQUIRES additional_approval"
					]
				}
			]
			
			for rule in compliance_rules:
				self.audit_service.register_compliance_rule(
					rule_id=rule["rule_id"],
					capability_id=self.capability_id,
					**rule
				)
			
		except Exception as e:
			print(f"Failed to register compliance rules: {str(e)}")
	
	def _register_views(self) -> None:
		"""Register Flask-AppBuilder views"""
		try:
			# Register main AP views
			self.appbuilder.add_view(
				APDashboardView,
				"Dashboard",
				icon="fas fa-tachometer-alt",
				category="Accounts Payable",
				category_icon="fas fa-file-invoice-dollar"
			)
			
			self.appbuilder.add_view(
				APVendorModelView,
				"Vendors",
				icon="fas fa-users",
				category="Accounts Payable"
			)
			
			self.appbuilder.add_view(
				APInvoiceModelView,
				"Invoices",
				icon="fas fa-file-invoice",
				category="Accounts Payable"
			)
			
			self.appbuilder.add_view(
				APPaymentModelView,
				"Payments",
				icon="fas fa-credit-card",
				category="Accounts Payable"
			)
			
			self.appbuilder.add_view(
				APWorkflowModelView,
				"Workflows",
				icon="fas fa-tasks",
				category="Accounts Payable"
			)
			
			self.appbuilder.add_view(
				APReportsView,
				"Reports",
				icon="fas fa-chart-bar",
				category="Accounts Payable"
			)
			
			print("Flask-AppBuilder views registered successfully")
			
		except Exception as e:
			print(f"Failed to register views: {str(e)}")
	
	def _register_api(self) -> None:
		"""Register FastAPI endpoints"""
		try:
			# Create and register FastAPI app
			ap_api = create_ap_api()
			
			# Register with main Flask app
			self.appbuilder.app.register_blueprint(
				Blueprint('ap_api', __name__, url_prefix='/api/v1/core_financials/accounts_payable')
			)
			
			# Mount FastAPI app
			from werkzeug.middleware.dispatcher import DispatcherMiddleware
			self.appbuilder.app.wsgi_app = DispatcherMiddleware(
				self.appbuilder.app.wsgi_app,
				{'/api/v1/core_financials/accounts_payable': ap_api}
			)
			
			print("FastAPI endpoints registered successfully")
			
		except Exception as e:
			print(f"Failed to register API: {str(e)}")
	
	def get_capability_info(self) -> Dict[str, Any]:
		"""Get capability information for APG registry"""
		return {
			"capability_id": self.capability_id,
			"version": self.version,
			"name": "Accounts Payable",
			"description": "Enterprise-grade accounts payable automation with AI integration",
			"category": "core_financials",
			"status": "active",
			"health_check_url": "/api/v1/core_financials/accounts_payable/health",
			"documentation_url": "/ap/docs",
			"support_contact": "ap-support@datacraft.co.ke",
			"dependencies": self._get_dependencies(),
			"configuration": self._get_configuration_schema(),
			"metrics": {
				"invoices_processed_monthly": 0,
				"payments_processed_monthly": 0,
				"average_processing_time_hours": 0.0,
				"touchless_processing_rate": 0.0
			}
		}
	
	async def _log_blueprint_initialization(self, capability_id: str, status: str) -> None:
		"""Log blueprint initialization for monitoring"""
		print(f"APG Blueprint: {capability_id} - {status}")


# Flask-AppBuilder View Classes (Placeholder implementations for integration)

class APDashboardView(BaseView):
	"""AP Dashboard with real-time metrics and insights"""
	
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Display AP dashboard with key metrics"""
		try:
			# Placeholder dashboard implementation
			dashboard_data = {
				"total_vendors": 0,
				"pending_invoices": 0,
				"overdue_amount": 0.0,
				"title": "Accounts Payable Dashboard"
			}
			
			return self.render_template(
				'ap/dashboard.html',
				**dashboard_data
			)
			
		except Exception as e:
			flash(f"Error loading dashboard: {str(e)}", "error")
			return self.render_template('ap/error.html', error=str(e))


class APVendorModelView(ModelView):
	"""Vendor management view with APG integration"""
	
	list_title = "Vendor Management"
	show_title = "Vendor Details"
	add_title = "Add Vendor"
	edit_title = "Edit Vendor"


class APInvoiceModelView(ModelView):
	"""Invoice management view with AI processing"""
	
	list_title = "Invoice Management"
	show_title = "Invoice Details"
	add_title = "Add Invoice"
	edit_title = "Edit Invoice"


class APPaymentModelView(ModelView):
	"""Payment management view with multi-method support"""
	
	list_title = "Payment Management"
	show_title = "Payment Details"
	add_title = "Add Payment"
	edit_title = "Edit Payment"


class APWorkflowModelView(ModelView):
	"""Workflow management view with real-time collaboration"""
	
	list_title = "Approval Workflows"
	show_title = "Workflow Details"


class APReportsView(BaseView):
	"""Reports view with advanced analytics"""
	
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Display available reports"""
		return self.render_template('ap/reports.html', title="AP Reports")
	
	@expose('/aging')
	@has_access
	def aging_report(self):
		"""Display AP aging report"""
		try:
			# Placeholder aging report implementation
			aging_data = {
				"aging_records": [],
				"title": "AP Aging Report"
			}
			
			return self.render_template(
				'ap/aging_report.html',
				**aging_data
			)
			
		except Exception as e:
			flash(f"Error generating aging report: {str(e)}", "error")
			return self.render_template('ap/error.html', error=str(e))


# Export the blueprint class
__all__ = ["APGAccountsPayableBlueprint"]