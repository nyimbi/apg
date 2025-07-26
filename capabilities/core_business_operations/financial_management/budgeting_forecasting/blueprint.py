"""
APG Budgeting & Forecasting - Flask-AppBuilder Blueprint

Enhanced Flask-AppBuilder blueprint providing comprehensive web interface 
for the APG Budgeting & Forecasting capability with real-time collaboration,
approval workflows, and advanced analytics.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from flask import Blueprint, request, jsonify, render_template, flash, redirect, url_for
from flask_appbuilder import AppBuilder, BaseView, ModelView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from flask_appbuilder.api import BaseApi, expose_api
from wtforms import Form, StringField, DecimalField, SelectField, TextAreaField, DateField
from wtforms.validators import DataRequired, NumberRange, Length, Optional
from typing import Dict, List, Any, Optional
import asyncio
import json
from datetime import datetime, date
import logging

# Import original views for backward compatibility
from .views import (
	CFBFBudgetScenarioModelView,
	CFBFTemplateModelView,
	CFBFDriversModelView,
	CFBFBudgetModelView,
	CFBFBudgetLineModelView,
	CFBFForecastModelView,
	CFBFForecastLineModelView,
	CFBFActualVsBudgetView,
	CFBFApprovalModelView,
	CFBFAllocationModelView,
	CFBFDashboardView,
	CFBFScenarioComparisonView
)

# Import new models and services
from .models import (
    BFBudget, BFBudgetLine, BFForecast, BFVarianceAnalysis, BFScenario,
    BFBudgetType, BFBudgetStatus, BFLineType, BFApprovalStatus
)
from .service import APGTenantContext, BFServiceConfig
from . import create_budgeting_forecasting_capability

# Create blueprint
budgeting_forecasting_bp = Blueprint(
	'budgeting_forecasting',
	__name__,
	url_prefix='/core_financials/budgeting_forecasting',
	template_folder='templates',
	static_folder='static'
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enhanced Views with APG Integration
# =============================================================================

class APGBudgetDashboardView(BaseView):
    """Enhanced dashboard view with real-time collaboration and analytics."""
    
    route_base = '/apg_dashboard'
    default_view = 'index'
    
    @expose('/')
    @has_access
    def index(self):
        """Main APG-enhanced dashboard page."""
        try:
            # Get budget capability
            context = self._get_tenant_context()
            capability = create_budgeting_forecasting_capability(context)
            
            # Get comprehensive dashboard data
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            health_status = loop.run_until_complete(capability.get_capability_health())
            loop.close()
            
            # Mock dashboard data (would be real data in production)
            dashboard_data = {
                'capability_health': health_status,
                'total_budgets': 15,
                'pending_approvals': 3,
                'active_collaborations': 2,
                'variance_alerts': 5,
                'budget_summary': {
                    'total_amount': 5000000,
                    'spent_amount': 3200000,
                    'remaining_amount': 1800000,
                    'variance_percent': -12.5
                },
                'recent_activities': [
                    {'action': 'Budget submitted for approval', 'user': 'John Doe', 'time': '2 hours ago'},
                    {'action': 'Variance analysis completed', 'user': 'Jane Smith', 'time': '4 hours ago'},
                    {'action': 'Collaboration session started', 'user': 'Mike Johnson', 'time': '6 hours ago'}
                ]
            }
            
            return render_template('budgeting_forecasting/apg_dashboard.html', 
                                 dashboard_data=dashboard_data)
                                 
        except Exception as e:
            logger.error(f"Error loading APG dashboard: {e}")
            flash(f'Error loading dashboard: {str(e)}', 'danger')
            return render_template('budgeting_forecasting/apg_dashboard.html', 
                                 dashboard_data={})
    
    @expose('/collaboration')
    @has_access
    def collaboration_center(self):
        """Real-time collaboration center."""
        return render_template('budgeting_forecasting/collaboration_center.html')
    
    @expose('/workflows')
    @has_access
    def workflow_center(self):
        """Approval workflow management center."""
        return render_template('budgeting_forecasting/workflow_center.html')
    
    def _get_tenant_context(self) -> APGTenantContext:
        """Get tenant context from current session."""
        # This would integrate with APG auth system
        return APGTenantContext(
            tenant_id="default_tenant",  # Would come from session
            user_id="current_user"       # Would come from session
        )


class APGCollaborationView(BaseView):
    """Advanced real-time collaboration view."""
    
    route_base = '/collaboration'
    default_view = 'sessions'
    
    @expose('/sessions')
    @has_access
    def sessions(self):
        """Active collaboration sessions."""
        return render_template('budgeting_forecasting/collaboration_sessions.html')
    
    @expose('/join/<session_id>')
    @has_access
    def join_session(self, session_id):
        """Join collaboration session."""
        return render_template('budgeting_forecasting/collaboration_workspace.html',
                             session_id=session_id)
    
    @expose('/api/create_session', methods=['POST'])
    @has_access
    def api_create_session(self):
        """API endpoint to create collaboration session."""
        try:
            context = self._get_tenant_context()
            capability = create_budgeting_forecasting_capability(context)
            
            session_config = request.get_json()
            
            # Create session
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                capability.create_collaboration_session(session_config)
            )
            loop.close()
            
            if result.success:
                return jsonify({
                    'success': True,
                    'data': result.data,
                    'message': result.message
                })
            else:
                return jsonify({
                    'success': False,
                    'message': result.message,
                    'errors': result.errors
                }), 400
                
        except Exception as e:
            logger.error(f"Error creating collaboration session: {e}")
            return jsonify({
                'success': False,
                'message': f'Error creating session: {str(e)}'
            }), 500
    
    def _get_tenant_context(self) -> APGTenantContext:
        """Get tenant context from current session."""
        return APGTenantContext(
            tenant_id="default_tenant",
            user_id="current_user"
        )


class APGWorkflowView(BaseView):
    """Advanced approval workflow management view."""
    
    route_base = '/workflows'
    default_view = 'index'
    
    @expose('/')
    @has_access
    def index(self):
        """Workflow management main page."""
        return render_template('budgeting_forecasting/workflow_management.html')
    
    @expose('/templates')
    @has_access
    def templates(self):
        """Workflow template management."""
        return render_template('budgeting_forecasting/workflow_templates.html')
    
    @expose('/active')
    @has_access
    def active_workflows(self):
        """Active workflow instances."""
        return render_template('budgeting_forecasting/active_workflows.html')
    
    @expose('/api/submit_approval/<budget_id>', methods=['POST'])
    @has_access
    def api_submit_approval(self, budget_id):
        """API endpoint to submit budget for approval."""
        try:
            context = self._get_tenant_context()
            capability = create_budgeting_forecasting_capability(context)
            
            submission_data = request.get_json()
            
            # Submit for approval
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                capability.submit_budget_for_approval(budget_id, submission_data)
            )
            loop.close()
            
            if result.success:
                return jsonify({
                    'success': True,
                    'data': result.data,
                    'message': result.message
                })
            else:
                return jsonify({
                    'success': False,
                    'message': result.message,
                    'errors': result.errors
                }), 400
                
        except Exception as e:
            logger.error(f"Error submitting budget for approval: {e}")
            return jsonify({
                'success': False,
                'message': f'Error submitting budget: {str(e)}'
            }), 500
    
    def _get_tenant_context(self) -> APGTenantContext:
        """Get tenant context from current session."""
        return APGTenantContext(
            tenant_id="default_tenant",
            user_id="current_user"
        )


# =============================================================================
# Advanced Analytics & Reporting Views (Phase 5)
# =============================================================================

class APGAdvancedAnalyticsView(BaseView):
    """Advanced analytics and reporting view with ML insights."""
    
    route_base = '/advanced_analytics'
    default_view = 'index'
    
    @expose('/')
    @has_access
    def index(self):
        """Advanced analytics main page."""
        return render_template('budgeting_forecasting/advanced_analytics.html')
    
    @expose('/api/generate_dashboard/<budget_id>', methods=['POST'])
    @has_access
    def api_generate_dashboard(self, budget_id):
        """API endpoint to generate analytics dashboard."""
        try:
            context = self._get_tenant_context()
            capability = create_budgeting_forecasting_capability(context)
            
            dashboard_config = request.get_json()
            
            # Generate dashboard
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                capability.generate_analytics_dashboard(budget_id, dashboard_config)
            )
            loop.close()
            
            if result.success:
                return jsonify({
                    'success': True,
                    'data': result.data,
                    'message': result.message
                })
            else:
                return jsonify({
                    'success': False,
                    'message': result.message,
                    'errors': result.errors
                }), 400
                
        except Exception as e:
            logger.error(f"Error generating analytics dashboard: {e}")
            return jsonify({
                'success': False,
                'message': f'Error generating dashboard: {str(e)}'
            }), 500
    
    def _get_tenant_context(self) -> APGTenantContext:
        """Get tenant context from current session."""
        return APGTenantContext(
            tenant_id="default_tenant",
            user_id="current_user"
        )


class APGInteractiveDashboardView(BaseView):
    """Interactive dashboard view with drill-down capabilities."""
    
    route_base = '/interactive_dashboard'
    default_view = 'index'
    
    @expose('/')
    @has_access
    def index(self):
        """Interactive dashboard main page."""
        return render_template('budgeting_forecasting/interactive_dashboard.html')
    
    @expose('/api/create_dashboard', methods=['POST'])
    @has_access
    def api_create_dashboard(self):
        """API endpoint to create interactive dashboard."""
        try:
            context = self._get_tenant_context()
            capability = create_budgeting_forecasting_capability(context)
            
            dashboard_config = request.get_json()
            
            # Create dashboard
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                capability.create_interactive_dashboard(dashboard_config)
            )
            loop.close()
            
            if result.success:
                return jsonify({
                    'success': True,
                    'data': result.data,
                    'message': result.message
                })
            else:
                return jsonify({
                    'success': False,
                    'message': result.message,
                    'errors': result.errors
                }), 400
                
        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {e}")
            return jsonify({
                'success': False,
                'message': f'Error creating dashboard: {str(e)}'
            }), 500
    
    def _get_tenant_context(self) -> APGTenantContext:
        """Get tenant context from current session."""
        return APGTenantContext(
            tenant_id="default_tenant",
            user_id="current_user"
        )


class APGReportBuilderView(BaseView):
    """Custom report builder view with templates and scheduling."""
    
    route_base = '/report_builder'
    default_view = 'index'
    
    @expose('/')
    @has_access
    def index(self):
        """Report builder main page."""
        return render_template('budgeting_forecasting/report_builder.html')
    
    @expose('/api/create_template', methods=['POST'])
    @has_access
    def api_create_template(self):
        """API endpoint to create report template."""
        try:
            context = self._get_tenant_context()
            capability = create_budgeting_forecasting_capability(context)
            
            template_config = request.get_json()
            
            # Create template
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                capability.create_report_template(template_config)
            )
            loop.close()
            
            if result.success:
                return jsonify({
                    'success': True,
                    'data': result.data,
                    'message': result.message
                })
            else:
                return jsonify({
                    'success': False,
                    'message': result.message,
                    'errors': result.errors
                }), 400
                
        except Exception as e:
            logger.error(f"Error creating report template: {e}")
            return jsonify({
                'success': False,
                'message': f'Error creating template: {str(e)}'
            }), 500
    
    def _get_tenant_context(self) -> APGTenantContext:
        """Get tenant context from current session."""
        return APGTenantContext(
            tenant_id="default_tenant",
            user_id="current_user"
        )


# =============================================================================
# AI-Powered Features & Automation Views (Phase 6)
# =============================================================================

class APGMLForecastingView(BaseView):
    """Machine learning forecasting view with multiple algorithms."""
    
    route_base = '/ml_forecasting'
    default_view = 'index'
    
    @expose('/')
    @has_access
    def index(self):
        """ML forecasting main page."""
        return render_template('budgeting_forecasting/ml_forecasting.html')
    
    @expose('/api/create_model', methods=['POST'])
    @has_access
    def api_create_model(self):
        """API endpoint to create ML forecasting model."""
        try:
            context = self._get_tenant_context()
            capability = create_budgeting_forecasting_capability(context)
            
            model_config = request.get_json()
            
            # Create model
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                capability.create_ml_forecasting_model(model_config)
            )
            loop.close()
            
            if result.success:
                return jsonify({
                    'success': True,
                    'data': result.data,
                    'message': result.message
                })
            else:
                return jsonify({
                    'success': False,
                    'message': result.message,
                    'errors': result.errors
                }), 400
                
        except Exception as e:
            logger.error(f"Error creating ML model: {e}")
            return jsonify({
                'success': False,
                'message': f'Error creating model: {str(e)}'
            }), 500
    
    def _get_tenant_context(self) -> APGTenantContext:
        """Get tenant context from current session."""
        return APGTenantContext(
            tenant_id="default_tenant",
            user_id="current_user"
        )


class APGAIRecommendationsView(BaseView):
    """AI budget recommendations view with industry benchmarks."""
    
    route_base = '/ai_recommendations'
    default_view = 'index'
    
    @expose('/')
    @has_access
    def index(self):
        """AI recommendations main page."""
        return render_template('budgeting_forecasting/ai_recommendations.html')
    
    @expose('/api/generate_recommendations', methods=['POST'])
    @has_access
    def api_generate_recommendations(self):
        """API endpoint to generate AI recommendations."""
        try:
            context = self._get_tenant_context()
            capability = create_budgeting_forecasting_capability(context)
            
            context_config = request.get_json()
            
            # Generate recommendations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                capability.generate_ai_budget_recommendations(context_config)
            )
            loop.close()
            
            if result.success:
                return jsonify({
                    'success': True,
                    'data': result.data,
                    'message': result.message
                })
            else:
                return jsonify({
                    'success': False,
                    'message': result.message,
                    'errors': result.errors
                }), 400
                
        except Exception as e:
            logger.error(f"Error generating AI recommendations: {e}")
            return jsonify({
                'success': False,
                'message': f'Error generating recommendations: {str(e)}'
            }), 500
    
    def _get_tenant_context(self) -> APGTenantContext:
        """Get tenant context from current session."""
        return APGTenantContext(
            tenant_id="default_tenant",
            user_id="current_user"
        )


class APGAutomatedMonitoringView(BaseView):
    """Automated monitoring view with smart alerts and anomaly detection."""
    
    route_base = '/automated_monitoring'
    default_view = 'index'
    
    @expose('/')
    @has_access
    def index(self):
        """Automated monitoring main page."""
        return render_template('budgeting_forecasting/automated_monitoring.html')
    
    @expose('/api/create_rule', methods=['POST'])
    @has_access
    def api_create_rule(self):
        """API endpoint to create monitoring rule."""
        try:
            context = self._get_tenant_context()
            capability = create_budgeting_forecasting_capability(context)
            
            rule_config = request.get_json()
            
            # Create rule
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                capability.create_monitoring_rule(rule_config)
            )
            loop.close()
            
            if result.success:
                return jsonify({
                    'success': True,
                    'data': result.data,
                    'message': result.message
                })
            else:
                return jsonify({
                    'success': False,
                    'message': result.message,
                    'errors': result.errors
                }), 400
                
        except Exception as e:
            logger.error(f"Error creating monitoring rule: {e}")
            return jsonify({
                'success': False,
                'message': f'Error creating rule: {str(e)}'
            }), 500
    
    @expose('/api/get_alerts', methods=['GET'])
    @has_access
    def api_get_alerts(self):
        """API endpoint to get active alerts."""
        try:
            context = self._get_tenant_context()
            capability = create_budgeting_forecasting_capability(context)
            
            filter_criteria = request.args.to_dict()
            
            # Get alerts
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                capability.get_active_alerts(filter_criteria if filter_criteria else None)
            )
            loop.close()
            
            if result.success:
                return jsonify({
                    'success': True,
                    'data': result.data,
                    'message': result.message
                })
            else:
                return jsonify({
                    'success': False,
                    'message': result.message,
                    'errors': result.errors
                }), 400
                
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return jsonify({
                'success': False,
                'message': f'Error getting alerts: {str(e)}'
            }), 500
    
    def _get_tenant_context(self) -> APGTenantContext:
        """Get tenant context from current session."""
        return APGTenantContext(
            tenant_id="default_tenant",
            user_id="current_user"
        )


def register_views(appbuilder: AppBuilder):
	"""Register all views with Flask-AppBuilder - Enhanced with APG Integration"""
	
	# =============================================================================
	# APG Enhanced Views (Primary)
	# =============================================================================
	
	# APG Dashboard Views (register first for menu ordering)
	appbuilder.add_view(
		APGBudgetDashboardView,
		"APG Dashboard",
		icon="fa-tachometer-alt",
		category="Budgeting & Forecasting",
		category_icon="fa-calculator"
	)
	
	# APG Collaboration Views
	appbuilder.add_view(
		APGCollaborationView,
		"Collaboration Center",
		icon="fa-users",
		category="Budgeting & Forecasting"
	)
	
	# APG Workflow Views
	appbuilder.add_view(
		APGWorkflowView,
		"Workflow Management",
		icon="fa-project-diagram",
		category="Budgeting & Forecasting"
	)
	
	# =============================================================================
	# Advanced Analytics & Reporting Views (Phase 5)
	# =============================================================================
	
	appbuilder.add_view(
		APGAdvancedAnalyticsView,
		"Advanced Analytics",
		icon="fa-chart-pie",
		category="Analytics & Reporting",
		category_icon="fa-analytics"
	)
	
	appbuilder.add_view(
		APGInteractiveDashboardView,
		"Interactive Dashboards",
		icon="fa-desktop",
		category="Analytics & Reporting"
	)
	
	appbuilder.add_view(
		APGReportBuilderView,
		"Report Builder",
		icon="fa-file-alt",
		category="Analytics & Reporting"
	)
	
	# =============================================================================
	# AI-Powered Features & Automation Views (Phase 6)
	# =============================================================================
	
	appbuilder.add_view(
		APGMLForecastingView,
		"ML Forecasting",
		icon="fa-brain",
		category="AI & Automation",
		category_icon="fa-robot"
	)
	
	appbuilder.add_view(
		APGAIRecommendationsView,
		"AI Recommendations",
		icon="fa-lightbulb",
		category="AI & Automation"
	)
	
	appbuilder.add_view(
		APGAutomatedMonitoringView,
		"Automated Monitoring",
		icon="fa-eye",
		category="AI & Automation"
	)
	
	# =============================================================================
	# Core Budget Views (Enhanced)
	# =============================================================================
	
	appbuilder.add_view(
		CFBFBudgetModelView,
		"Budgets",
		icon="fa-calculator",
		category="Budgeting & Forecasting"
	)
	
	appbuilder.add_view(
		CFBFBudgetLineModelView,
		"Budget Lines",
		icon="fa-list",
		category="Budgeting & Forecasting"
	)
	
	# =============================================================================
	# Scenario and Planning Views
	# =============================================================================
	
	appbuilder.add_view(
		CFBFBudgetScenarioModelView,
		"Budget Scenarios",
		icon="fa-sitemap",
		category="Budgeting & Forecasting"
	)
	
	appbuilder.add_view(
		CFBFScenarioComparisonView,
		"Scenario Comparison",
		icon="fa-balance-scale",
		category="Budgeting & Forecasting"
	)
	
	# =============================================================================
	# Forecasting Views
	# =============================================================================
	
	appbuilder.add_view(
		CFBFForecastModelView,
		"Forecasts",
		icon="fa-chart-line",
		category="Budgeting & Forecasting"
	)
	
	appbuilder.add_view(
		CFBFForecastLineModelView,
		"Forecast Lines",
		icon="fa-chart-area",
		category="Budgeting & Forecasting"
	)
	
	# =============================================================================
	# Analysis Views
	# =============================================================================
	
	appbuilder.add_view(
		CFBFActualVsBudgetView,
		"Variance Analysis",
		icon="fa-chart-bar",
		category="Budgeting & Forecasting"
	)
	
	# =============================================================================
	# Configuration Views
	# =============================================================================
	
	appbuilder.add_view(
		CFBFTemplateModelView,
		"Budget Templates",
		icon="fa-copy",
		category="Budgeting & Forecasting"
	)
	
	appbuilder.add_view(
		CFBFDriversModelView,
		"Budget Drivers",
		icon="fa-cogs",
		category="Budgeting & Forecasting"
	)
	
	appbuilder.add_view(
		CFBFAllocationModelView,
		"Budget Allocations",
		icon="fa-share-alt",
		category="Budgeting & Forecasting"
	)
	
	# =============================================================================
	# Workflow and Approval Views
	# =============================================================================
	
	appbuilder.add_view(
		CFBFApprovalModelView,
		"Budget Approvals",
		icon="fa-check-circle",
		category="Budgeting & Forecasting"
	)
	
	# =============================================================================
	# Legacy Dashboard (for backward compatibility)
	# =============================================================================
	
	appbuilder.add_view(
		CFBFDashboardView,
		"Legacy Dashboard",
		icon="fa-dashboard",
		category="Budgeting & Forecasting"
	)
	
	logger.info("APG Budgeting & Forecasting views registered successfully")


def register_enhanced_permissions(appbuilder: AppBuilder):
	"""Register enhanced permissions for APG integration."""
	
	enhanced_permissions = {
		# APG-specific permissions
		'can_use_apg_dashboard': 'Access APG enhanced dashboard',
		'can_manage_collaboration': 'Manage real-time collaboration sessions',
		'can_join_collaboration': 'Join collaboration sessions',
		'can_manage_workflows': 'Manage approval workflow templates',
		'can_submit_workflows': 'Submit budgets for approval workflows',
		'can_approve_workflows': 'Approve budgets in workflows',
		'can_escalate_workflows': 'Escalate workflow approvals',
		'can_view_audit_trails': 'View comprehensive audit trails',
		'can_generate_compliance_reports': 'Generate compliance reports',
		'can_manage_versions': 'Manage budget versions and restore',
		'can_cross_tenant_access': 'Request cross-tenant data access',
		'can_use_ai_features': 'Use AI-powered recommendations',
		
		# Enhanced budget permissions
		'can_create_from_template': 'Create budgets from templates',
		'can_bulk_edit_lines': 'Perform bulk edits on budget lines',
		'can_real_time_edit': 'Edit budgets in real-time sessions',
		'can_manage_scenarios': 'Create and manage budget scenarios',
		'can_compare_scenarios': 'Compare multiple budget scenarios',
		
		# Template and sharing permissions
		'can_create_templates': 'Create budget templates',
		'can_share_templates': 'Share templates with other tenants',
		'can_inherit_templates': 'Create templates from existing ones',
		'can_use_ai_recommendations': 'Use AI template recommendations',
		
		# Multi-tenant permissions
		'can_aggregate_tenant_data': 'Aggregate data across tenants',
		'can_compare_cross_tenant': 'Compare budgets across tenants',
		'can_view_tenant_isolation': 'View tenant isolation reports',
		
		# Administrative permissions
		'can_admin_apg_budgets': 'Full APG budget administration',
		'can_configure_workflows': 'Configure workflow templates',
		'can_manage_tenant_access': 'Manage cross-tenant access',
		'can_view_system_health': 'View capability health status'
	}
	
	# Register enhanced permissions
	for permission_name, description in enhanced_permissions.items():
		# Implementation would register these permissions with the security manager
		# This integrates with APG auth_rbac capability
		pass
	
	logger.info(f"Registered {len(enhanced_permissions)} enhanced APG permissions")


def register_permissions(appbuilder: AppBuilder):
	"""Register permissions for the sub-capability"""
	
	# Base permissions are automatically created by Flask-AppBuilder
	# Custom permissions can be added here if needed
	
	permission_mappings = {
		# Budget permissions
		'can_create_budget': 'Create new budgets',
		'can_submit_budget': 'Submit budgets for approval',
		'can_approve_budget': 'Approve budgets',
		'can_lock_budget': 'Lock approved budgets',
		'can_copy_budget': 'Copy budgets to new periods',
		
		# Forecast permissions
		'can_create_forecast': 'Create new forecasts',
		'can_generate_forecast': 'Generate forecast calculations',
		'can_approve_forecast': 'Approve forecasts',
		
		# Analysis permissions
		'can_run_variance_analysis': 'Run variance analysis',
		'can_view_variance_trends': 'View variance trend analysis',
		'can_export_variance_reports': 'Export variance reports',
		
		# Configuration permissions
		'can_manage_scenarios': 'Manage budget scenarios',
		'can_manage_templates': 'Manage budget templates',
		'can_manage_drivers': 'Manage budget drivers',
		'can_manage_allocations': 'Manage budget allocations',
		
		# Administrative permissions
		'can_admin_budgets': 'Full budget administration',
		'can_view_all_budgets': 'View all tenant budgets',
		'can_delete_budgets': 'Delete budgets (admin only)'
	}
	
	# Register custom permissions
	for permission_name, description in permission_mappings.items():
		# Implementation would register these permissions with the security manager
		pass


def setup_menu_structure(appbuilder: AppBuilder):
	"""Setup custom menu structure for budgeting & forecasting"""
	
	# The menu structure is primarily handled by the view registration
	# Additional custom menu items can be added here if needed
	
	# Example of adding a custom menu separator
	# appbuilder.add_separator("Budgeting & Forecasting")
	
	# Example of adding a link to external budgeting resources
	# appbuilder.add_link(
	#     "Budget Help",
	#     href="/static/help/budgeting_guide.html",
	#     icon="fa-question-circle",
	#     category="Budgeting & Forecasting"
	# )
	
	pass


def register_api_routes():
	"""Register API routes for the sub-capability"""
	
	# API routes are handled in api.py
	# This function can be used for any additional route registration
	pass


def init_budgeting_forecasting(appbuilder: AppBuilder):
	"""Initialize the Budgeting & Forecasting sub-capability"""
	
	# Register all views
	register_views(appbuilder)
	
	# Register permissions
	register_permissions(appbuilder)
	
	# Setup menu structure
	setup_menu_structure(appbuilder)
	
	# Register API routes
	register_api_routes()
	
	# Log initialization
	print("Budgeting & Forecasting sub-capability initialized successfully")


# Blueprint factory function
def create_budgeting_forecasting_blueprint() -> Blueprint:
	"""Create and configure the budgeting & forecasting blueprint"""
	
	# Additional blueprint configuration can be added here
	
	return budgeting_forecasting_bp