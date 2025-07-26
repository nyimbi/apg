"""
APG API Service Mesh - Flask-AppBuilder Blueprint

Blueprint registration and configuration for the service mesh web interface
with comprehensive view integration and menu structure.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from flask_appbuilder import AppBuilder
from flask import Blueprint

from .views import (
	SMServiceView,
	SMEndpointView,
	SMRouteView,
	SMLoadBalancerView,
	SMPolicyView,
	SMMetricsView,
	SMHealthCheckView,
	SMTopologyView,
	ServiceMeshDashboardView,
	ServiceMeshChartsView
)

def create_blueprint(appbuilder: AppBuilder) -> Blueprint:
	"""
	Create and configure the API Service Mesh blueprint.
	
	Args:
		appbuilder: Flask-AppBuilder instance
		
	Returns:
		Configured Blueprint instance
	"""
	
	# Register main dashboard view
	appbuilder.add_view(
		ServiceMeshDashboardView,
		"Dashboard",
		icon="fa-tachometer",
		category="Service Mesh",
		category_icon="fa-sitemap"
	)
	
	# Register service management views
	appbuilder.add_view(
		SMServiceView,
		"Services",
		icon="fa-server",
		category="Service Mesh"
	)
	
	appbuilder.add_view(
		SMEndpointView,
		"Endpoints",
		icon="fa-plug",
		category="Service Mesh"
	)
	
	# Register traffic management views
	appbuilder.add_view(
		SMRouteView,
		"Routes",
		icon="fa-route",
		category="Traffic Management",
		category_icon="fa-exchange"
	)
	
	appbuilder.add_view(
		SMLoadBalancerView,
		"Load Balancers",
		icon="fa-balance-scale",
		category="Traffic Management"
	)
	
	appbuilder.add_view(
		SMPolicyView,
		"Policies",
		icon="fa-shield",
		category="Traffic Management"
	)
	
	# Register monitoring views
	appbuilder.add_view(
		SMMetricsView,
		"Metrics",
		icon="fa-chart-line",
		category="Monitoring",
		category_icon="fa-chart-bar"
	)
	
	appbuilder.add_view(
		SMHealthCheckView,
		"Health Checks",
		icon="fa-heartbeat",
		category="Monitoring"
	)
	
	appbuilder.add_view(
		SMTopologyView,
		"Topology",
		icon="fa-project-diagram",
		category="Monitoring"
	)
	
	appbuilder.add_view(
		ServiceMeshChartsView,
		"Analytics",
		icon="fa-analytics",
		category="Monitoring"
	)
	
	# Add links to dashboard sections
	appbuilder.add_link(
		"Service Topology",
		href="/servicemeshdashboardview/topology/",
		icon="fa-sitemap",
		category="Service Mesh"
	)
	
	appbuilder.add_link(
		"Real-time Monitoring",
		href="/servicemeshdashboardview/monitoring/",
		icon="fa-monitor",
		category="Service Mesh"
	)
	
	# Create the blueprint (Flask-AppBuilder handles this internally)
	# We return None as Flask-AppBuilder manages blueprints automatically
	return None

def register_menu_items(appbuilder: AppBuilder):
	"""
	Register additional menu items and organize the service mesh interface.
	
	Args:
		appbuilder: Flask-AppBuilder instance
	"""
	
	# Add separator for better organization
	appbuilder.add_separator("Service Mesh")
	
	# Add quick action links
	appbuilder.add_link(
		"Register New Service",
		href="/smserviceview/add/",
		icon="fa-plus-circle",
		category="Quick Actions",
		category_icon="fa-bolt"
	)
	
	appbuilder.add_link(
		"Create Route",
		href="/smrouteview/add/",
		icon="fa-plus-circle",
		category="Quick Actions"
	)
	
	appbuilder.add_link(
		"Setup Load Balancer",
		href="/smloadbalancerview/add/",
		icon="fa-plus-circle",
		category="Quick Actions"
	)
	
	# Add external documentation links
	appbuilder.add_link(
		"Service Mesh Documentation",
		href="/service-mesh/docs/",
		icon="fa-book",
		category="Documentation",
		category_icon="fa-book-open"
	)
	
	appbuilder.add_link(
		"API Reference",
		href="/api/docs",
		icon="fa-code",
		category="Documentation"
	)

# Menu configuration
MENU_CONFIG = {
	"Service Mesh": {
		"icon": "fa-sitemap",
		"items": [
			{
				"name": "Dashboard",
				"href": "/servicemeshdashboardview/dashboard/",
				"icon": "fa-tachometer"
			},
			{
				"name": "Services",
				"href": "/smserviceview/list/",
				"icon": "fa-server"
			},
			{
				"name": "Endpoints", 
				"href": "/smendpointview/list/",
				"icon": "fa-plug"
			},
			{
				"name": "Topology",
				"href": "/servicemeshdashboardview/topology/",
				"icon": "fa-project-diagram"
			}
		]
	},
	"Traffic Management": {
		"icon": "fa-exchange",
		"items": [
			{
				"name": "Routes",
				"href": "/smrouteview/list/",
				"icon": "fa-route"
			},
			{
				"name": "Load Balancers",
				"href": "/smloadbalancerview/list/",
				"icon": "fa-balance-scale"
			},
			{
				"name": "Policies",
				"href": "/smpolicyview/list/",
				"icon": "fa-shield"
			}
		]
	},
	"Monitoring": {
		"icon": "fa-chart-bar",
		"items": [
			{
				"name": "Real-time Dashboard",
				"href": "/servicemeshdashboardview/monitoring/",
				"icon": "fa-monitor"
			},
			{
				"name": "Metrics",
				"href": "/smmetricsview/list/",
				"icon": "fa-chart-line"
			},
			{
				"name": "Health Checks",
				"href": "/smhealthcheckview/list/",
				"icon": "fa-heartbeat"
			},
			{
				"name": "Analytics",
				"href": "/servicemeshchartsview/chart/",
				"icon": "fa-analytics"
			}
		]
	}
}

# Permissions configuration
PERMISSIONS_CONFIG = {
	"Service Management": [
		"can_list_SMServiceView",
		"can_show_SMServiceView", 
		"can_add_SMServiceView",
		"can_edit_SMServiceView",
		"can_delete_SMServiceView",
		"can_list_SMEndpointView",
		"can_show_SMEndpointView",
		"can_add_SMEndpointView",
		"can_edit_SMEndpointView",
		"can_delete_SMEndpointView"
	],
	"Traffic Management": [
		"can_list_SMRouteView",
		"can_show_SMRouteView",
		"can_add_SMRouteView", 
		"can_edit_SMRouteView",
		"can_delete_SMRouteView",
		"can_list_SMLoadBalancerView",
		"can_show_SMLoadBalancerView",
		"can_add_SMLoadBalancerView",
		"can_edit_SMLoadBalancerView",
		"can_delete_SMLoadBalancerView",
		"can_list_SMPolicyView",
		"can_show_SMPolicyView",
		"can_add_SMPolicyView",
		"can_edit_SMPolicyView",
		"can_delete_SMPolicyView"
	],
	"Monitoring": [
		"can_list_SMMetricsView",
		"can_show_SMMetricsView",
		"can_list_SMHealthCheckView", 
		"can_show_SMHealthCheckView",
		"can_list_SMTopologyView",
		"can_show_SMTopologyView",
		"can_chart_ServiceMeshChartsView"
	],
	"Dashboard": [
		"can_dashboard_ServiceMeshDashboardView",
		"can_topology_ServiceMeshDashboardView",
		"can_monitoring_ServiceMeshDashboardView",
		"can_dashboard_data_ServiceMeshDashboardView",
		"can_topology_data_ServiceMeshDashboardView"
	]
}

def setup_security_roles(appbuilder: AppBuilder):
	"""
	Setup security roles for service mesh functionality.
	
	Args:
		appbuilder: Flask-AppBuilder instance
	"""
	
	# Service Mesh Administrator Role
	admin_role = appbuilder.sm.find_role("ServiceMeshAdmin")
	if not admin_role:
		admin_role = appbuilder.sm.add_role("ServiceMeshAdmin")
	
	# Add all permissions to admin role
	for category, permissions in PERMISSIONS_CONFIG.items():
		for permission in permissions:
			perm = appbuilder.sm.find_permission_on_view(permission.split('_')[-1], permission.replace(f"_{permission.split('_')[-1]}", ""))
			if perm:
				appbuilder.sm.add_permission_role(admin_role, perm)
	
	# Service Mesh Operator Role (read/write but no delete)
	operator_role = appbuilder.sm.find_role("ServiceMeshOperator")
	if not operator_role:
		operator_role = appbuilder.sm.add_role("ServiceMeshOperator")
	
	# Add limited permissions to operator role
	operator_permissions = [
		"can_list_SMServiceView", "can_show_SMServiceView", "can_add_SMServiceView", "can_edit_SMServiceView",
		"can_list_SMEndpointView", "can_show_SMEndpointView", "can_add_SMEndpointView", "can_edit_SMEndpointView",
		"can_list_SMRouteView", "can_show_SMRouteView", "can_add_SMRouteView", "can_edit_SMRouteView",
		"can_list_SMLoadBalancerView", "can_show_SMLoadBalancerView", "can_add_SMLoadBalancerView", "can_edit_SMLoadBalancerView",
		"can_list_SMPolicyView", "can_show_SMPolicyView", "can_add_SMPolicyView", "can_edit_SMPolicyView"
	] + PERMISSIONS_CONFIG["Monitoring"] + PERMISSIONS_CONFIG["Dashboard"]
	
	for permission in operator_permissions:
		perm = appbuilder.sm.find_permission_on_view(permission.split('_')[-1], permission.replace(f"_{permission.split('_')[-1]}", ""))
		if perm:
			appbuilder.sm.add_permission_role(operator_role, perm)
	
	# Service Mesh Viewer Role (read-only)
	viewer_role = appbuilder.sm.find_role("ServiceMeshViewer")
	if not viewer_role:
		viewer_role = appbuilder.sm.add_role("ServiceMeshViewer")
	
	# Add read-only permissions to viewer role
	viewer_permissions = [
		"can_list_SMServiceView", "can_show_SMServiceView",
		"can_list_SMEndpointView", "can_show_SMEndpointView", 
		"can_list_SMRouteView", "can_show_SMRouteView",
		"can_list_SMLoadBalancerView", "can_show_SMLoadBalancerView",
		"can_list_SMPolicyView", "can_show_SMPolicyView"
	] + PERMISSIONS_CONFIG["Monitoring"] + PERMISSIONS_CONFIG["Dashboard"]
	
	for permission in viewer_permissions:
		perm = appbuilder.sm.find_permission_on_view(permission.split('_')[-1], permission.replace(f"_{permission.split('_')[-1]}", ""))
		if perm:
			appbuilder.sm.add_permission_role(viewer_role, perm)

# Blueprint metadata
BLUEPRINT_INFO = {
	"name": "api_service_mesh",
	"url_prefix": "/service-mesh",
	"template_folder": "templates",
	"static_folder": "static",
	"description": "APG API Service Mesh management interface",
	"version": "1.0.0",
	"author": "Nyimbi Odero",
	"menu_config": MENU_CONFIG,
	"permissions_config": PERMISSIONS_CONFIG
}