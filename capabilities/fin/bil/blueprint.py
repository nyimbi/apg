"""
APG Billing Flask-AppBuilder Blueprint

Flask-AppBuilder blueprint registration for the billing capability
with comprehensive admin interface and customer portal.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder

from .views import (
	BillingCustomerModelView,
	BillingPlanModelView,
	BillingSubscriptionModelView,
	BillingInvoiceModelView,
	BillingPaymentModelView,
	BillingUsageModelView,
	BillingDashboardView,
	BillingReportsView,
	BillingCustomerPortalView,
	RevenueByMonthChartView,
	SubscriptionStatusChartView
)


def create_billing_blueprint(appbuilder: AppBuilder) -> Blueprint:
	"""
	Create and configure the billing blueprint with all views
	
	Args:
		appbuilder: Flask-AppBuilder instance
		
	Returns:
		Configured Flask blueprint
	"""
	
	# Create blueprint
	billing_bp = Blueprint(
		'billing',
		__name__,
		url_prefix='/billing',
		template_folder='templates',
		static_folder='static'
	)
	
	# Register model views
	appbuilder.add_view(
		BillingCustomerModelView,
		"Customers",
		icon="fa-users",
		category="Billing",
		category_icon="fa-credit-card"
	)
	
	appbuilder.add_view(
		BillingPlanModelView,
		"Plans",
		icon="fa-list-alt",
		category="Billing"
	)
	
	appbuilder.add_view(
		BillingSubscriptionModelView,
		"Subscriptions", 
		icon="fa-refresh",
		category="Billing"
	)
	
	appbuilder.add_view(
		BillingInvoiceModelView,
		"Invoices",
		icon="fa-file-text",
		category="Billing"
	)
	
	appbuilder.add_view(
		BillingPaymentModelView,
		"Payments",
		icon="fa-credit-card",
		category="Billing"
	)
	
	appbuilder.add_view(
		BillingUsageModelView,
		"Usage",
		icon="fa-bar-chart",
		category="Billing"
	)
	
	# Register dashboard and analytics views
	appbuilder.add_view_no_menu(BillingDashboardView())
	appbuilder.add_link(
		"Dashboard",
		href="/billing/billingdashboardview/",
		icon="fa-dashboard",
		category="Billing"
	)
	
	appbuilder.add_view_no_menu(BillingReportsView())
	appbuilder.add_link(
		"Reports",
		href="/billing/billingreportsview/revenue",
		icon="fa-bar-chart",
		category="Billing"
	)
	
	# Register customer portal
	appbuilder.add_view_no_menu(BillingCustomerPortalView())
	appbuilder.add_link(
		"Customer Portal",
		href="/billing/billingcustomerportalview/",
		icon="fa-user",
		category="Customer"
	)
	
	# Register chart views
	appbuilder.add_view(
		RevenueByMonthChartView,
		"Revenue Chart",
		icon="fa-line-chart",
		category="Analytics",
		category_icon="fa-bar-chart"
	)
	
	appbuilder.add_view(
		SubscriptionStatusChartView,
		"Subscription Chart",
		icon="fa-pie-chart", 
		category="Analytics"
	)
	
	# Add separator and menu sections
	appbuilder.add_separator("Billing")
	appbuilder.add_separator("Analytics")
	
	return billing_bp


def register_billing_permissions(appbuilder: AppBuilder) -> None:
	"""
	Register billing-specific permissions and roles
	
	Args:
		appbuilder: Flask-AppBuilder instance
	"""
	
	# Define billing permissions
	billing_permissions = [
		# Customer permissions
		('can_list', 'BillingCustomerModelView'),
		('can_show', 'BillingCustomerModelView'),
		('can_add', 'BillingCustomerModelView'),
		('can_edit', 'BillingCustomerModelView'),
		('can_delete', 'BillingCustomerModelView'),
		
		# Plan permissions
		('can_list', 'BillingPlanModelView'),
		('can_show', 'BillingPlanModelView'),
		('can_add', 'BillingPlanModelView'),
		('can_edit', 'BillingPlanModelView'),
		('can_delete', 'BillingPlanModelView'),
		
		# Subscription permissions
		('can_list', 'BillingSubscriptionModelView'),
		('can_show', 'BillingSubscriptionModelView'),
		('can_edit', 'BillingSubscriptionModelView'),
		('can_cancel_subscription', 'BillingSubscriptionModelView'),
		
		# Invoice permissions
		('can_list', 'BillingInvoiceModelView'),
		('can_show', 'BillingInvoiceModelView'),
		('can_edit', 'BillingInvoiceModelView'),
		('can_send_invoice', 'BillingInvoiceModelView'),
		('can_mark_paid', 'BillingInvoiceModelView'),
		
		# Payment permissions
		('can_list', 'BillingPaymentModelView'),
		('can_show', 'BillingPaymentModelView'),
		('can_refund_payment', 'BillingPaymentModelView'),
		
		# Usage permissions
		('can_list', 'BillingUsageModelView'),
		('can_show', 'BillingUsageModelView'),
		
		# Dashboard and reports
		('can_dashboard', 'BillingDashboardView'),
		('can_revenue_report', 'BillingReportsView'),
		('can_customer_report', 'BillingReportsView'),
		('can_subscription_report', 'BillingReportsView'),
		
		# Customer portal
		('can_portal', 'BillingCustomerPortalView'),
		('can_customer_invoices', 'BillingCustomerPortalView'),
		('can_customer_usage', 'BillingCustomerPortalView'),
	]
	
	# Register permissions
	for permission, view in billing_permissions:
		appbuilder.sm.add_permission_view_menu(permission, view)
	
	# Create billing roles
	billing_admin_role = appbuilder.sm.add_role("BillingAdmin")
	billing_user_role = appbuilder.sm.add_role("BillingUser")
	customer_role = appbuilder.sm.add_role("Customer")
	
	# Assign permissions to roles
	
	# Billing Admin - full access
	admin_permissions = [perm for perm, _ in billing_permissions]
	for permission, view in billing_permissions:
		perm = appbuilder.sm.find_permission_view_menu(permission, view)
		if perm:
			appbuilder.sm.add_permission_role(billing_admin_role, perm)
	
	# Billing User - read access and limited actions
	user_permissions = [
		('can_list', 'BillingCustomerModelView'),
		('can_show', 'BillingCustomerModelView'),
		('can_list', 'BillingPlanModelView'),
		('can_show', 'BillingPlanModelView'),
		('can_list', 'BillingSubscriptionModelView'),
		('can_show', 'BillingSubscriptionModelView'),
		('can_list', 'BillingInvoiceModelView'),
		('can_show', 'BillingInvoiceModelView'),
		('can_list', 'BillingPaymentModelView'),
		('can_show', 'BillingPaymentModelView'),
		('can_list', 'BillingUsageModelView'),
		('can_show', 'BillingUsageModelView'),
		('can_dashboard', 'BillingDashboardView'),
		('can_revenue_report', 'BillingReportsView'),
		('can_customer_report', 'BillingReportsView'),
		('can_subscription_report', 'BillingReportsView'),
	]
	
	for permission, view in user_permissions:
		perm = appbuilder.sm.find_permission_view_menu(permission, view)
		if perm:
			appbuilder.sm.add_permission_role(billing_user_role, perm)
	
	# Customer - portal access only
	customer_permissions = [
		('can_portal', 'BillingCustomerPortalView'),
		('can_customer_invoices', 'BillingCustomerPortalView'),
		('can_customer_usage', 'BillingCustomerPortalView'),
	]
	
	for permission, view in customer_permissions:
		perm = appbuilder.sm.find_permission_view_menu(permission, view)
		if perm:
			appbuilder.sm.add_permission_role(customer_role, perm)


def init_billing_data(appbuilder: AppBuilder) -> None:
	"""
	Initialize billing capability with default data
	
	Args:
		appbuilder: Flask-AppBuilder instance
	"""
	from .service import get_billing_service
	
	billing_service = get_billing_service()
	
	# Create default plans
	default_plans = [
		{
			"name": "Starter",
			"description": "Perfect for small teams getting started",
			"pricing_model": "flat_rate",
			"base_price": "29.99",
			"currency": "USD",
			"billing_period": "monthly",
			"features": ["Basic features", "Email support", "5 users"],
			"trial_period_days": 14
		},
		{
			"name": "Professional",
			"description": "For growing businesses with advanced needs",
			"pricing_model": "flat_rate",
			"base_price": "99.99", 
			"currency": "USD",
			"billing_period": "monthly",
			"features": ["Advanced features", "Priority support", "25 users", "Analytics"],
			"trial_period_days": 14
		},
		{
			"name": "Enterprise",
			"description": "For large organizations with custom requirements",
			"pricing_model": "hybrid",
			"base_price": "299.99",
			"currency": "USD", 
			"billing_period": "monthly",
			"features": ["All features", "24/7 support", "Unlimited users", "Custom integrations"],
			"trial_period_days": 30
		}
	]
	
	# Create system user for initialization
	system_user = "system"
	
	try:
		for plan_data in default_plans:
			plan_data["tenant_id"] = "default"
			# billing_service.create_plan(system_user, plan_data)
			
		print("✅ Billing capability initialized with default plans")
		
	except Exception as e:
		print(f"⚠️  Error initializing billing data: {e}")


# Blueprint factory function
def create_billing_capability(app, appbuilder: AppBuilder) -> Blueprint:
	"""
	Complete billing capability setup
	
	Args:
		app: Flask application instance
		appbuilder: Flask-AppBuilder instance
		
	Returns:
		Configured billing blueprint
	"""
	
	# Create blueprint
	blueprint = create_billing_blueprint(appbuilder)
	
	# Register permissions and roles
	register_billing_permissions(appbuilder)
	
	# Initialize default data
	init_billing_data(appbuilder)
	
	# Register blueprint with app
	app.register_blueprint(blueprint)
	
	print("✅ APG Billing capability registered successfully")
	
	return blueprint


__all__ = [
	"create_billing_blueprint",
	"register_billing_permissions", 
	"init_billing_data",
	"create_billing_capability"
]