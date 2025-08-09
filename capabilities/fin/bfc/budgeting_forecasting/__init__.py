"""
APG Budgeting & Forecasting Capability - Main Module

Enterprise-grade budgeting and forecasting capability with comprehensive
APG platform integration, real-time collaboration, and advanced analytics.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, List, Any, Optional
import logging
import asyncio
from datetime import datetime

# Core models and enums
from .models import (
    # Core Models
    APGBaseModel, BFBudget, BFBudgetLine, BFForecast, BFForecastDataPoint,
    BFVarianceAnalysis, BFScenario,
    
    # Enumerations
    BFBudgetType, BFBudgetStatus, BFLineType, BFApprovalStatus,
    BFForecastType, BFForecastMethod, BFVarianceType, BFScenarioType,
    
    # Value Types
    PositiveAmount, CurrencyCode, FiscalYear, NonEmptyString
)

# Core services
from .service import (
    APGTenantContext, BFServiceConfig, ServiceResponse, APGServiceBase,
    BudgetingService, ForecastingService, VarianceAnalysisService,
    ScenarioService, create_budgeting_service, create_forecasting_service,
    create_variance_service, create_scenario_service
)

# Advanced services
from .budget_management import (
    BudgetTemplate, BudgetVersion, BudgetCollaboration, AdvancedBudgetService,
    create_advanced_budget_service
)

from .template_system import (
    TemplateCategory, TemplateScope, TemplateComplexity, BudgetTemplateModel,
    TemplateLineItem, TemplateUsageHistory, TemplateManagementService,
    create_template_service
)

from .multitenant_operations import (
    TenantPermissionLevel, CrossTenantScope, AggregationLevel,
    TenantBudgetAccess, CrossTenantComparison, TenantAggregation,
    MultiTenantOperationsService, create_multitenant_service
)

from .realtime_collaboration import (
    CollaborationEventType, UserPresenceStatus, ConflictResolutionStrategy,
    CollaborationSession, CollaborationParticipant, CollaborationEvent,
    BudgetComment, ChangeRequest, RealTimeCollaborationService,
    create_realtime_collaboration_service
)

from .approval_workflows import (
    ApprovalAction, WorkflowStepType, EscalationTrigger, WorkflowTemplate,
    ApprovalWorkflowInstance, WorkflowEscalation, ApprovalWorkflowService,
    create_approval_workflow_service
)

from .version_control_audit import (
    ChangeType, AuditEventType, ComplianceLevel, BudgetVersion as AuditBudgetVersion,
    AuditEvent, ComplianceReport, VersionControlAuditService,
    create_version_control_audit_service
)

# Advanced Analytics & Reporting Services (Phase 5)
from .advanced_analytics import (
    AnalyticsMetricType, AnalyticsPeriod, VarianceSignificance, AnalyticsMetric,
    BudgetAnalyticsDashboard, VarianceAnalysisReport, PredictiveAnalyticsModel,
    AdvancedAnalyticsService, create_advanced_analytics_service
)

from .interactive_dashboard import (
    DashboardType, VisualizationType, DrillDownLevel, DashboardWidget,
    InteractiveDashboard, DashboardTheme, InteractiveDashboardService,
    create_interactive_dashboard_service
)

from .custom_report_builder import (
    ReportType, ReportFormat, ReportFrequency, ReportField, ReportTemplate,
    ReportSchedule, ReportExecution, CustomReportBuilderService,
    create_custom_report_builder_service
)

# AI-Powered Features & Automation Services (Phase 6)
from .ml_forecasting_engine import (
    ForecastAlgorithm, ForecastHorizon, ModelStatus, ForecastFeature,
    MLForecastingModel, ForecastPrediction, ForecastScenario, ModelEnsemble,
    MLForecastingEngineService, create_ml_forecasting_engine_service
)

from .ai_budget_recommendations import (
    RecommendationType, RecommendationPriority, ConfidenceLevel, AIRecommendation,
    RecommendationBundle, BenchmarkData, AIBudgetRecommendationsService,
    create_ai_budget_recommendations_service
)

from .automated_monitoring import (
    AlertType, AlertSeverity, MonitoringFrequency, MonitoringRule, BudgetAlert,
    MonitoringDashboard, AnomalyDetection, AutomatedBudgetMonitoringService,
    create_automated_budget_monitoring_service
)


# =============================================================================
# Unified Capability Interface
# =============================================================================

class BudgetingForecastingCapability:
    """
    Unified interface for the APG Budgeting & Forecasting capability.
    Provides centralized access to all budgeting and forecasting services.
    """
    
    def __init__(self, context: APGTenantContext, config: Optional[BFServiceConfig] = None):
        self.context = context
        self.config = config or BFServiceConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize all services
        self._services = {}
        self._initialize_services()
    
    def _initialize_services(self) -> None:
        """Initialize all budgeting and forecasting services."""
        try:
            # Core services
            self._services['budgeting'] = create_budgeting_service(self.context, self.config)
            self._services['forecasting'] = create_forecasting_service(self.context, self.config)
            self._services['variance'] = create_variance_service(self.context, self.config)
            self._services['scenario'] = create_scenario_service(self.context, self.config)
            
            # Advanced services
            self._services['advanced_budget'] = create_advanced_budget_service(self.context, self.config)
            self._services['templates'] = create_template_service(self.context, self.config)
            self._services['multitenant'] = create_multitenant_service(self.context, self.config)
            self._services['collaboration'] = create_realtime_collaboration_service(self.context, self.config)
            self._services['workflows'] = create_approval_workflow_service(self.context, self.config)
            self._services['audit'] = create_version_control_audit_service(self.context, self.config)
            
            # Advanced Analytics & Reporting Services (Phase 5)
            self._services['analytics'] = create_advanced_analytics_service(self.context, self.config)
            self._services['dashboard'] = create_interactive_dashboard_service(self.context, self.config)
            self._services['reports'] = create_custom_report_builder_service(self.context, self.config)
            
            # AI-Powered Features & Automation Services (Phase 6)
            self._services['ml_forecasting'] = create_ml_forecasting_engine_service(self.context, self.config)
            self._services['ai_recommendations'] = create_ai_budget_recommendations_service(self.context, self.config)
            self._services['monitoring'] = create_automated_budget_monitoring_service(self.context, self.config)
            
            self.logger.info(f"Initialized {len(self._services)} budgeting & forecasting services")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize budgeting & forecasting services: {e}")
            raise
    
    # =============================================================================
    # Core Budgeting Operations
    # =============================================================================
    
    async def create_budget(self, budget_data: Dict[str, Any]) -> ServiceResponse:
        """Create a new budget with comprehensive validation."""
        return await self._services['budgeting'].create_budget(budget_data)
    
    async def create_budget_from_template(self, template_id: str, budget_data: Dict[str, Any]) -> ServiceResponse:
        """Create budget from template with advanced customization."""
        return await self._services['advanced_budget'].create_budget_from_template(template_id, budget_data)
    
    async def update_budget(self, budget_id: str, update_data: Dict[str, Any]) -> ServiceResponse:
        """Update budget with version control and audit tracking."""
        return await self._services['budgeting'].update_budget(budget_id, update_data)
    
    async def get_budget(self, budget_id: str, include_lines: bool = True) -> ServiceResponse:
        """Get budget with optional line items."""
        return await self._services['budgeting'].get_budget(budget_id, include_lines)
    
    async def delete_budget(self, budget_id: str, soft_delete: bool = True) -> ServiceResponse:
        """Delete budget with audit trail."""
        return await self._services['budgeting'].delete_budget(budget_id, soft_delete)
    
    # =============================================================================
    # Real-Time Collaboration Operations
    # =============================================================================
    
    async def create_collaboration_session(self, session_config: Dict[str, Any]) -> ServiceResponse:
        """Create real-time collaboration session."""
        return await self._services['collaboration'].create_collaboration_session(session_config)
    
    async def join_collaboration_session(self, session_id: str, join_config: Dict[str, Any]) -> ServiceResponse:
        """Join collaboration session."""
        return await self._services['collaboration'].join_collaboration_session(session_id, join_config)
    
    # =============================================================================
    # Approval Workflow Operations
    # =============================================================================
    
    async def submit_budget_for_approval(self, budget_id: str, submission_data: Dict[str, Any]) -> ServiceResponse:
        """Submit budget for approval workflow."""
        return await self._services['workflows'].submit_budget_for_approval(budget_id, submission_data)
    
    async def process_approval_action(self, workflow_instance_id: str, action_data: Dict[str, Any]) -> ServiceResponse:
        """Process approval action (approve/reject/delegate)."""
        return await self._services['workflows'].process_approval_action(workflow_instance_id, action_data)
    
    # =============================================================================
    # Advanced Analytics & Reporting Operations (Phase 5)
    # =============================================================================
    
    async def generate_analytics_dashboard(self, budget_id: str, dashboard_config: Dict[str, Any]) -> ServiceResponse:
        """Generate comprehensive analytics dashboard."""
        return await self._services['analytics'].generate_analytics_dashboard(budget_id, dashboard_config)
    
    async def perform_advanced_variance_analysis(self, budget_id: str, analysis_config: Dict[str, Any]) -> ServiceResponse:
        """Perform advanced variance analysis with ML insights."""
        return await self._services['analytics'].perform_advanced_variance_analysis(budget_id, analysis_config)
    
    async def create_interactive_dashboard(self, dashboard_config: Dict[str, Any]) -> ServiceResponse:
        """Create interactive dashboard with real-time capabilities."""
        return await self._services['dashboard'].create_interactive_dashboard(dashboard_config)
    
    async def perform_dashboard_drill_down(self, dashboard_id: str, drill_config: Dict[str, Any]) -> ServiceResponse:
        """Perform drill-down operation on dashboard."""
        return await self._services['dashboard'].perform_drill_down(dashboard_id, drill_config)
    
    async def create_report_template(self, template_config: Dict[str, Any]) -> ServiceResponse:
        """Create custom report template."""
        return await self._services['reports'].create_report_template(template_config)
    
    async def generate_report(self, template_id: str, generation_config: Dict[str, Any]) -> ServiceResponse:
        """Generate report from template."""
        return await self._services['reports'].generate_report(template_id, generation_config)
    
    async def create_report_schedule(self, schedule_config: Dict[str, Any]) -> ServiceResponse:
        """Create automated report schedule."""
        return await self._services['reports'].create_report_schedule(schedule_config)
    
    # =============================================================================
    # AI-Powered Features & Automation Operations (Phase 6)
    # =============================================================================
    
    async def create_ml_forecasting_model(self, model_config: Dict[str, Any]) -> ServiceResponse:
        """Create machine learning forecasting model."""
        return await self._services['ml_forecasting'].create_forecasting_model(model_config)
    
    async def train_forecasting_model(self, model_id: str, training_config: Optional[Dict[str, Any]] = None) -> ServiceResponse:
        """Train ML forecasting model."""
        return await self._services['ml_forecasting'].train_forecasting_model(model_id, training_config)
    
    async def generate_ml_forecast(self, model_id: str, forecast_config: Dict[str, Any]) -> ServiceResponse:
        """Generate forecast using trained ML model."""
        return await self._services['ml_forecasting'].generate_forecast(model_id, forecast_config)
    
    async def create_model_ensemble(self, ensemble_config: Dict[str, Any]) -> ServiceResponse:
        """Create ensemble of multiple forecasting models."""
        return await self._services['ml_forecasting'].create_model_ensemble(ensemble_config)
    
    async def generate_ai_budget_recommendations(self, context_config: Dict[str, Any]) -> ServiceResponse:
        """Generate AI-powered budget recommendations."""
        return await self._services['ai_recommendations'].generate_budget_recommendations(context_config)
    
    async def implement_recommendation(self, recommendation_id: str, implementation_config: Dict[str, Any]) -> ServiceResponse:
        """Implement a specific AI recommendation."""
        return await self._services['ai_recommendations'].implement_recommendation(recommendation_id, implementation_config)
    
    async def track_recommendation_performance(self, recommendation_id: str) -> ServiceResponse:
        """Track performance of implemented recommendation."""
        return await self._services['ai_recommendations'].track_recommendation_performance(recommendation_id)
    
    async def create_monitoring_rule(self, rule_config: Dict[str, Any]) -> ServiceResponse:
        """Create automated monitoring rule."""
        return await self._services['monitoring'].create_monitoring_rule(rule_config)
    
    async def start_automated_monitoring(self) -> ServiceResponse:
        """Start automated budget monitoring processes."""
        return await self._services['monitoring'].start_automated_monitoring()
    
    async def get_active_alerts(self, filter_criteria: Optional[Dict[str, Any]] = None) -> ServiceResponse:
        """Get active budget monitoring alerts."""
        return await self._services['monitoring'].get_active_alerts(filter_criteria)
    
    async def perform_anomaly_detection(self, detection_config: Dict[str, Any]) -> ServiceResponse:
        """Perform anomaly detection on budget data."""
        return await self._services['monitoring'].perform_anomaly_detection(detection_config)
    
    async def get_capability_health(self) -> Dict[str, Any]:
        """Get health status of all services."""
        health_status = {
            'capability': 'budgeting_forecasting',
            'status': 'healthy',
            'timestamp': datetime.utcnow(),
            'tenant_id': self.context.tenant_id,
            'services': {}
        }
        
        for service_name, service in self._services.items():
            try:
                service_health = {
                    'status': 'healthy',
                    'initialized': service is not None,
                    'last_check': datetime.utcnow()
                }
            except Exception as e:
                service_health = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'last_check': datetime.utcnow()
                }
            
            health_status['services'][service_name] = service_health
        
        return health_status


# =============================================================================
# Capability Factory Function
# =============================================================================

def create_budgeting_forecasting_capability(
    context: APGTenantContext, 
    config: Optional[BFServiceConfig] = None
) -> BudgetingForecastingCapability:
    """
    Factory function to create the complete Budgeting & Forecasting capability.
    
    Args:
        context: APG tenant context with user and tenant information
        config: Optional service configuration
    
    Returns:
        BudgetingForecastingCapability: Fully initialized capability instance
    """
    return BudgetingForecastingCapability(context, config)


# =============================================================================
# Legacy Compatibility Layer
# =============================================================================

# Sub-capability metadata for backward compatibility
SUBCAPABILITY_META = {
	'name': 'Budgeting & Forecasting',
	'code': 'BF',
	'version': '2.0.0',  # Updated version
	'capability': 'core_financials',
	'description': 'Enterprise-grade budgeting and forecasting with APG platform integration, real-time collaboration, and advanced analytics.',
	'industry_focus': 'All',
	'dependencies': ['auth_rbac', 'audit_compliance', 'workflow_engine'],
	'optional_dependencies': ['ai_orchestration', 'notification_engine', 'document_management', 'real_time_collaboration'],
	'apg_integrations': [
		'auth_rbac',
		'audit_compliance', 
		'workflow_engine',
		'real_time_collaboration',
		'ai_orchestration',
		'notification_engine',
		'document_management'
	],
	'features': [
		'Real-time collaborative budget editing',
		'Flexible approval workflows with escalation',
		'Comprehensive version control and audit trails',
		'Multi-tenant operations with cross-tenant comparison',
		'AI-powered template recommendations',
		'Advanced forecasting with scenario planning',
		'Variance analysis with automated alerts',
		'Template inheritance and sharing',
		'Integration with all APG platform capabilities',
		'ML-powered variance detection and predictive insights',
		'Interactive dashboards with drill-down capabilities',
		'Custom report builder with scheduling',
		'Multi-algorithm ML forecasting engine',
		'AI budget recommendations with industry benchmarks',
		'Automated monitoring with smart alerts and anomaly detection'
	]
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META

def validate_dependencies(available_subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate dependencies are met"""
	errors = []
	warnings = []
	
	# Check required dependencies
	if 'general_ledger' not in available_subcapabilities:
		errors.append("General Ledger sub-capability is required for budget/actual comparisons")
	
	# Check optional dependencies
	if 'accounts_payable' not in available_subcapabilities:
		warnings.append("Accounts Payable integration not available - expense budgeting may be limited")
	
	if 'accounts_receivable' not in available_subcapabilities:
		warnings.append("Accounts Receivable integration not available - revenue forecasting may be limited")
		
	if 'cash_management' not in available_subcapabilities:
		warnings.append("Cash Management integration not available - cash flow forecasting not available")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def get_default_budget_categories() -> List[Dict[str, Any]]:
	"""Get default budget categories"""
	return [
		{'code': 'REVENUE', 'name': 'Revenue', 'description': 'Revenue and income budgets', 'type': 'Revenue'},
		{'code': 'SALARIES', 'name': 'Salaries & Wages', 'description': 'Personnel costs', 'type': 'Expense'},
		{'code': 'BENEFITS', 'name': 'Employee Benefits', 'description': 'Benefits and payroll taxes', 'type': 'Expense'},
		{'code': 'MARKETING', 'name': 'Marketing & Advertising', 'description': 'Marketing and promotional expenses', 'type': 'Expense'},
		{'code': 'OPERATIONS', 'name': 'Operations', 'description': 'Operational expenses', 'type': 'Expense'},
		{'code': 'FACILITIES', 'name': 'Facilities', 'description': 'Rent, utilities, maintenance', 'type': 'Expense'},
		{'code': 'TECHNOLOGY', 'name': 'Technology', 'description': 'IT expenses and software', 'type': 'Expense'},
		{'code': 'CAPEX', 'name': 'Capital Expenditure', 'description': 'Capital investments', 'type': 'Asset'},
		{'code': 'OTHER', 'name': 'Other', 'description': 'Other budget items', 'type': 'Other'}
	]

def get_default_budget_drivers() -> List[Dict[str, Any]]:
	"""Get default budget drivers"""
	return [
		{'code': 'HEADCOUNT', 'name': 'Headcount', 'description': 'Number of employees', 'unit': 'Count'},
		{'code': 'SALES_VOLUME', 'name': 'Sales Volume', 'description': 'Units sold', 'unit': 'Units'},
		{'code': 'CUSTOMERS', 'name': 'Customer Count', 'description': 'Number of customers', 'unit': 'Count'},
		{'code': 'SQ_FOOTAGE', 'name': 'Square Footage', 'description': 'Office/facility space', 'unit': 'Sq Ft'},
		{'code': 'INFLATION', 'name': 'Inflation Rate', 'description': 'Annual inflation rate', 'unit': 'Percent'},
		{'code': 'GROWTH_RATE', 'name': 'Growth Rate', 'description': 'Revenue growth rate', 'unit': 'Percent'},
		{'code': 'CAPACITY', 'name': 'Capacity Utilization', 'description': 'Production capacity usage', 'unit': 'Percent'},
		{'code': 'CONVERSION', 'name': 'Conversion Rate', 'description': 'Sales conversion rate', 'unit': 'Percent'}
	]

def get_default_budget_scenarios() -> List[Dict[str, Any]]:
	"""Get default budget scenarios"""
	return [
		{'code': 'BASE', 'name': 'Base Case', 'description': 'Most likely scenario', 'probability': 60.0},
		{'code': 'OPTIMISTIC', 'name': 'Optimistic', 'description': 'Best case scenario', 'probability': 20.0},
		{'code': 'PESSIMISTIC', 'name': 'Pessimistic', 'description': 'Worst case scenario', 'probability': 20.0},
		{'code': 'CONSERVATIVE', 'name': 'Conservative', 'description': 'Conservative estimates', 'probability': 0.0},
		{'code': 'AGGRESSIVE', 'name': 'Aggressive', 'description': 'Aggressive growth targets', 'probability': 0.0}
	]

def get_default_allocation_methods() -> List[Dict[str, Any]]:
	"""Get default allocation methods"""
	return [
		{
			'code': 'DIRECT',
			'name': 'Direct Allocation',
			'description': 'Direct assignment to cost centers',
			'formula': None
		},
		{
			'code': 'PERCENTAGE',
			'name': 'Percentage Allocation',
			'description': 'Allocate based on fixed percentages',
			'formula': 'amount * percentage / 100'
		},
		{
			'code': 'HEADCOUNT',
			'name': 'Headcount-Based',
			'description': 'Allocate based on headcount',
			'formula': 'amount * dept_headcount / total_headcount'
		},
		{
			'code': 'REVENUE',
			'name': 'Revenue-Based',
			'description': 'Allocate based on revenue',
			'formula': 'amount * dept_revenue / total_revenue'
		},
		{
			'code': 'SQUARE_FOOTAGE',
			'name': 'Square Footage',
			'description': 'Allocate based on space usage',
			'formula': 'amount * dept_sqft / total_sqft'
		}
	]

def get_default_gl_account_mappings() -> Dict[str, str]:
	"""Get default GL account mappings for budgeting"""
	return {
		'budget_variance': '5900',      # Budget Variance account
		'forecast_adjustment': '5910',  # Forecast Adjustment account
		'budget_reserve': '2900',       # Budget Reserve/Contingency
		'commitment_control': '2910',   # Budget Commitment Control
		'encumbrance': '2920'           # Budget Encumbrance
	}

def get_variance_analysis_rules() -> List[Dict[str, Any]]:
	"""Get default variance analysis rules"""
	return [
		{
			'name': 'Significant Variance',
			'description': 'Variance exceeds threshold amount or percentage',
			'condition': 'abs(variance_amount) > threshold_amount OR abs(variance_percent) > threshold_percent',
			'alert_level': 'Warning',
			'auto_notify': True
		},
		{
			'name': 'Unfavorable Revenue Variance',
			'description': 'Revenue is significantly below budget',
			'condition': 'account_type = "Revenue" AND variance_amount < -threshold_amount',
			'alert_level': 'Critical',
			'auto_notify': True
		},
		{
			'name': 'Unfavorable Expense Variance',
			'description': 'Expenses are significantly over budget',
			'condition': 'account_type = "Expense" AND variance_amount > threshold_amount',
			'alert_level': 'Warning',
			'auto_notify': True
		},
		{
			'name': 'Budget Overrun',
			'description': 'Cumulative spending exceeds annual budget',
			'condition': 'ytd_actual > annual_budget',
			'alert_level': 'Critical',
			'auto_notify': True
		}
	]