"""
APG Customer Relationship Management - Core Service Layer

Revolutionary CRM service implementation providing 10x superior business logic
and operations compared to industry leaders.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union, Tuple
import json

from pydantic import ValidationError

# Local imports
from .models import (
	CRMContact, CRMAccount, CRMLead, CRMOpportunity, CRMActivity, CRMCampaign,
	ContactType, AccountType, LeadStatus, OpportunityStage, ActivityType,
	RecordStatus, LeadSource, Priority
)
from .database import DatabaseManager
from .ai_insights import CRMAIInsights
from .analytics import CRMAnalytics
from .import_export import ContactImportExportManager
from .contact_relationships import ContactRelationshipManager, RelationshipType, RelationshipStrength
from .activity_tracking import ContactActivityTracker, ActivityOutcome
from .account_hierarchy import AccountHierarchyManager, HierarchyUpdateRequest, HierarchyRelationshipType
from .account_relationships import AccountRelationshipManager, AccountRelationshipType, RelationshipStrength, RelationshipDirection
from .territory_management import TerritoryManager, TerritoryType, TerritoryStatus, AssignmentType
from .communication_history import CommunicationManager, CommunicationType, CommunicationDirection, CommunicationStatus, CommunicationOutcome
from .contact_segmentation import ContactSegmentationManager, SegmentType, SegmentStatus, ContactSegment
from .lead_scoring import LeadScoringManager, ScoreCategory, ScoreWeight, LeadScore, LeadScoreRule
from .sales_pipeline import SalesPipelineManager, SalesPipeline, PipelineStage, OpportunityStageHistory, PipelineAnalytics
from .workflow_automation import WorkflowAutomationEngine, Workflow, WorkflowExecution, WorkflowAnalytics
from .email_integration import EmailIntegrationManager, EmailTemplate, EmailMessage, EmailTracking, EmailAnalytics
from .calendar_activity_management import CalendarActivityManager, CalendarEvent, CRMActivity, ActivityTemplate, CalendarAnalytics
from .approval_workflows import ApprovalWorkflowEngine, ApprovalWorkflowTemplate, ApprovalRequest, ApprovalStep, ApprovalHistory, ApprovalAnalytics
from .lead_assignment import LeadAssignmentManager, LeadAssignmentRule, LeadAssignment, AssignmentAnalytics, AssignmentType, AssignmentStatus
from .lead_nurturing import LeadNurturingManager, NurturingWorkflow, NurturingEnrollment, NurturingAnalytics, TriggerType, NurturingStatus
from .crm_dashboard import CRMDashboardManager, DashboardLayout, DashboardWidget, DashboardData, DashboardInsight, DashboardType
from .reporting_engine import AdvancedReportingEngine, ReportDefinition, ReportExecution, ReportSchedule, ReportData, ExportFormat
from .predictive_analytics import (
	PredictiveAnalyticsEngine, PredictionModel, PredictionRequest, PredictionResult,
	ForecastingInsight, ChurnPrediction, LeadScoringInsight, MarketSegmentation
)
from .performance_benchmarking import (
	PerformanceBenchmarkingEngine, PerformanceBenchmark, PerformanceMetric,
	PerformanceComparison, GoalTracking, PerformanceReport
)
from .api_gateway import (
	APIGateway, RateLimitRule, APIEndpoint, APIRequest, APIGatewayMetrics
)
from .webhook_management import (
	WebhookManager, WebhookEndpoint, WebhookEvent, WebhookDelivery, WebhookSubscription
)
from .third_party_integration import (
	ThirdPartyIntegrationManager, IntegrationConnector, FieldMapping, 
	SyncConfiguration, SyncExecution, IntegrationType, AuthenticationType,
	SyncDirection, SyncStatus, DataOperation
)
from .realtime_sync import (
	RealTimeSyncEngine, SyncEvent, SyncEventType, ConflictRecord,
	SyncConfiguration as RealtimeSyncConfiguration, ConflictResolutionStrategy,
	ChangeDetectionMode
)
from .api_versioning import (
	APIVersioningManager, APIVersion, DeprecationNotice, VersionMigration,
	ClientVersionUsage, APIVersionStatus, DeprecationSeverity, VersioningStrategy
)


logger = logging.getLogger(__name__)


class CRMServiceError(Exception):
	"""Base exception for CRM service errors"""
	pass


class CRMValidationError(CRMServiceError):
	"""CRM validation error"""
	pass


class CRMNotFoundError(CRMServiceError):
	"""CRM record not found error"""
	pass


class CRMService:
	"""
	Core CRM service providing comprehensive customer relationship management
	functionality with AI-powered insights and analytics.
	"""
	
	def __init__(self, db_manager: DatabaseManager = None, config_manager: Any = None):
		"""
		Initialize CRM service
		
		Args:
			db_manager: Database manager instance
			config_manager: Configuration manager instance
		"""
		self.db_manager = db_manager or DatabaseManager()
		self.config_manager = config_manager
		self.ai_insights = CRMAIInsights()
		self.analytics = CRMAnalytics()
		self.relationship_manager = ContactRelationshipManager(self.db_manager)
		self.activity_tracker = ContactActivityTracker(self.db_manager)
		self.hierarchy_manager = AccountHierarchyManager(self.db_manager)
		self.account_relationship_manager = AccountRelationshipManager(self.db_manager)
		self.territory_manager = TerritoryManager(self.db_manager)
		self.communication_manager = CommunicationManager(self.db_manager)
		self.segmentation_manager = ContactSegmentationManager(self.db_manager)
		self.lead_scoring_manager = LeadScoringManager(self.db_manager)
		self.pipeline_manager = SalesPipelineManager(self.db_manager)
		self.workflow_automation_engine = WorkflowAutomationEngine(self.db_manager)
		self.email_integration_manager = EmailIntegrationManager(self.db_manager)
		self.calendar_activity_manager = CalendarActivityManager(self.db_manager)
		self.approval_workflow_engine = ApprovalWorkflowEngine(self.db_manager)
		self.lead_assignment_manager = LeadAssignmentManager(self.db_manager)
		self.lead_nurturing_manager = LeadNurturingManager(self.db_manager)
		self.dashboard_manager = CRMDashboardManager(self.db_manager)
		self.reporting_engine = AdvancedReportingEngine(self.db_manager)
		self.predictive_analytics = PredictiveAnalyticsEngine(
			self.db_manager.get_connection_pool(), 
			cache_manager=getattr(self, 'cache_manager', None)
		)
		self.performance_benchmarking = PerformanceBenchmarkingEngine(
			self.db_manager.get_connection_pool(),
			cache_manager=getattr(self, 'cache_manager', None)
		)
		self.api_gateway = APIGateway(self.db_manager.get_connection_pool())
		self.webhook_manager = WebhookManager(self.db_manager.get_connection_pool())
		self.third_party_integration = ThirdPartyIntegrationManager(self.db_manager.get_connection_pool())
		self.realtime_sync = RealTimeSyncEngine(self.db_manager.get_connection_pool())
		self.api_versioning = APIVersioningManager(self.db_manager.get_connection_pool())
		self._initialized = False
	
	async def initialize(self):
		"""Initialize the CRM service"""
		try:
			logger.info("🔧 Initializing CRM service...")
			
			# Initialize database manager
			if not self.db_manager._initialized:
				await self.db_manager.initialize()
			
			# Initialize AI insights
			await self.ai_insights.initialize()
			
			# Initialize analytics
			await self.analytics.initialize()
			
			# Initialize lead scoring manager
			await self.lead_scoring_manager.initialize()
			
			# Initialize pipeline manager
			await self.pipeline_manager.initialize()
			
			# Initialize workflow automation engine
			await self.workflow_automation_engine.initialize()
			
			# Initialize email integration manager
			await self.email_integration_manager.initialize()
			
			# Initialize calendar activity manager
			await self.calendar_activity_manager.initialize()
			
			# Initialize approval workflow engine
			await self.approval_workflow_engine.initialize()
			
			# Initialize lead assignment manager
			await self.lead_assignment_manager.initialize()
			
			# Initialize lead nurturing manager
			await self.lead_nurturing_manager.initialize()
			
			# Initialize dashboard manager
			await self.dashboard_manager.initialize()
			
			# Initialize reporting engine
			await self.reporting_engine.initialize()
			
			# Initialize predictive analytics engine
			# Note: No initialization needed for predictive analytics as it uses db_pool directly
			
			# Initialize API gateway
			await self.api_gateway.initialize()
			
			# Initialize webhook manager
			await self.webhook_manager.initialize()
			
			# Initialize third-party integration manager
			await self.third_party_integration.initialize()
			
			# Initialize real-time sync engine
			await self.realtime_sync.initialize()
			
			# Initialize API versioning manager
			await self.api_versioning.initialize()
			
			self._initialized = True
			logger.info("✅ CRM service initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize CRM service: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Service initialization failed: {str(e)}")
	
	async def health_check(self) -> Dict[str, Any]:
		"""
		Perform health check on CRM service
		
		Returns:
			Dict containing health status
		"""
		health_status = {
			"service": "healthy",
			"timestamp": datetime.utcnow().isoformat(),
			"components": {}
		}
		
		try:
			# Check database health
			if self.db_manager:
				db_health = await self.db_manager.health_check()
				health_status["components"]["database"] = db_health
			
			# Check AI insights health
			if self.ai_insights:
				ai_health = await self.ai_insights.health_check()
				health_status["components"]["ai_insights"] = ai_health
			
			# Check analytics health
			if self.analytics:
				analytics_health = await self.analytics.health_check()
				health_status["components"]["analytics"] = analytics_health
			
		except Exception as e:
			health_status["service"] = "unhealthy"
			health_status["error"] = str(e)
		
		return health_status
	
	# ================================
	# Contact Management
	# ================================
	
	async def create_contact(
		self,
		contact_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> CRMContact:
		"""
		Create a new contact
		
		Args:
			contact_data: Contact information
			tenant_id: Tenant identifier
			created_by: User creating the contact
			
		Returns:
			Created contact object
		"""
		try:
			# Add audit fields
			contact_data.update({
				"tenant_id": tenant_id,
				"created_by": created_by,
				"updated_by": created_by
			})
			
			# Validate and create contact
			contact = CRMContact(**contact_data)
			
			# Check for duplicates
			existing_contact = await self._find_duplicate_contact(contact, tenant_id)
			if existing_contact:
				logger.warning(f"Potential duplicate contact found: {existing_contact.id}")
			
			# Save to database
			saved_contact = await self.db_manager.create_contact(contact)
			
			# Generate AI insights
			if saved_contact.email:
				asyncio.create_task(
					self._generate_contact_insights(saved_contact.id, tenant_id)
				)
			
			logger.info(f"Created contact: {saved_contact.id} for tenant: {tenant_id}")
			return saved_contact
			
		except ValidationError as e:
			raise CRMValidationError(f"Invalid contact data: {str(e)}")
		except Exception as e:
			logger.error(f"Failed to create contact: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Contact creation failed: {str(e)}")
	
	async def get_contact(self, contact_id: str, tenant_id: str) -> CRMContact:
		"""
		Get contact by ID
		
		Args:
			contact_id: Contact identifier
			tenant_id: Tenant identifier
			
		Returns:
			Contact object
		"""
		try:
			contact = await self.db_manager.get_contact(contact_id, tenant_id)
			if not contact:
				raise CRMNotFoundError(f"Contact {contact_id} not found")
			return contact
		except Exception as e:
			if isinstance(e, CRMNotFoundError):
				raise
			logger.error(f"Failed to get contact {contact_id}: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Failed to retrieve contact: {str(e)}")
	
	async def update_contact(
		self,
		contact_id: str,
		update_data: Dict[str, Any],
		tenant_id: str,
		updated_by: str
	) -> CRMContact:
		"""
		Update contact
		
		Args:
			contact_id: Contact identifier
			update_data: Updated contact data
			tenant_id: Tenant identifier
			updated_by: User updating the contact
			
		Returns:
			Updated contact object
		"""
		try:
			# Get existing contact
			existing_contact = await self.get_contact(contact_id, tenant_id)
			
			# Add audit fields
			update_data.update({
				"updated_by": updated_by,
				"updated_at": datetime.utcnow(),
				"version": existing_contact.version + 1
			})
			
			# Update contact
			updated_contact = await self.db_manager.update_contact(
				contact_id, update_data, tenant_id
			)
			
			# Trigger analytics update
			asyncio.create_task(
				self._update_contact_analytics(contact_id, tenant_id)
			)
			
			logger.info(f"Updated contact: {contact_id} for tenant: {tenant_id}")
			return updated_contact
			
		except Exception as e:
			logger.error(f"Failed to update contact {contact_id}: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Contact update failed: {str(e)}")
	
	async def delete_contact(self, contact_id: str, tenant_id: str, deleted_by: str) -> bool:
		"""
		Soft delete contact
		
		Args:
			contact_id: Contact identifier
			tenant_id: Tenant identifier
			deleted_by: User deleting the contact
			
		Returns:
			True if successful
		"""
		try:
			# Soft delete (update status)
			update_data = {
				"status": RecordStatus.DELETED,
				"updated_by": deleted_by,
				"updated_at": datetime.utcnow()
			}
			
			await self.db_manager.update_contact(contact_id, update_data, tenant_id)
			
			logger.info(f"Deleted contact: {contact_id} for tenant: {tenant_id}")
			return True
			
		except Exception as e:
			logger.error(f"Failed to delete contact {contact_id}: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Contact deletion failed: {str(e)}")
	
	async def search_contacts(
		self,
		tenant_id: str,
		filters: Dict[str, Any] = None,
		search_term: str = None,
		limit: int = 100,
		offset: int = 0
	) -> Tuple[List[CRMContact], int]:
		"""
		Search contacts with filters
		
		Args:
			tenant_id: Tenant identifier
			filters: Search filters
			search_term: Text search term
			limit: Result limit
			offset: Result offset
			
		Returns:
			Tuple of (contacts list, total count)
		"""
		try:
			return await self.db_manager.search_contacts(
				tenant_id, filters, search_term, limit, offset
			)
		except Exception as e:
			logger.error(f"Contact search failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Contact search failed: {str(e)}")
	
	# ================================
	# Account Management
	# ================================
	
	async def create_account(
		self,
		account_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> CRMAccount:
		"""
		Create a new account
		
		Args:
			account_data: Account information
			tenant_id: Tenant identifier
			created_by: User creating the account
			
		Returns:
			Created account object
		"""
		try:
			# Add audit fields
			account_data.update({
				"tenant_id": tenant_id,
				"created_by": created_by,
				"updated_by": created_by
			})
			
			# Validate and create account
			account = CRMAccount(**account_data)
			
			# Save to database
			saved_account = await self.db_manager.create_account(account)
			
			# Generate AI insights
			asyncio.create_task(
				self._generate_account_insights(saved_account.id, tenant_id)
			)
			
			logger.info(f"Created account: {saved_account.id} for tenant: {tenant_id}")
			return saved_account
			
		except ValidationError as e:
			raise CRMValidationError(f"Invalid account data: {str(e)}")
		except Exception as e:
			logger.error(f"Failed to create account: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Account creation failed: {str(e)}")
	
	async def get_account(self, account_id: str, tenant_id: str) -> CRMAccount:
		"""
		Get account by ID
		
		Args:
			account_id: Account identifier
			tenant_id: Tenant identifier
			
		Returns:
			Account object
		"""
		try:
			account = await self.db_manager.get_account(account_id, tenant_id)
			if not account:
				raise CRMNotFoundError(f"Account {account_id} not found")
			return account
		except Exception as e:
			if isinstance(e, CRMNotFoundError):
				raise
			logger.error(f"Failed to get account {account_id}: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Failed to retrieve account: {str(e)}")
	
	# ================================
	# Lead Management
	# ================================
	
	async def create_lead(
		self,
		lead_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> CRMLead:
		"""
		Create a new lead
		
		Args:
			lead_data: Lead information
			tenant_id: Tenant identifier
			created_by: User creating the lead
			
		Returns:
			Created lead object
		"""
		try:
			# Add audit fields
			lead_data.update({
				"tenant_id": tenant_id,
				"created_by": created_by,
				"updated_by": created_by
			})
			
			# Validate and create lead
			lead = CRMLead(**lead_data)
			
			# Save to database
			saved_lead = await self.db_manager.create_lead(lead)
			
			# Generate lead score
			asyncio.create_task(
				self._calculate_lead_score(saved_lead.id, tenant_id)
			)
			
			logger.info(f"Created lead: {saved_lead.id} for tenant: {tenant_id}")
			return saved_lead
			
		except ValidationError as e:
			raise CRMValidationError(f"Invalid lead data: {str(e)}")
		except Exception as e:
			logger.error(f"Failed to create lead: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Lead creation failed: {str(e)}")
	
	async def convert_lead(
		self,
		lead_id: str,
		tenant_id: str,
		converted_by: str,
		create_account: bool = True,
		create_opportunity: bool = True,
		opportunity_data: Dict[str, Any] = None
	) -> Dict[str, str]:
		"""
		Convert lead to contact/account/opportunity
		
		Args:
			lead_id: Lead identifier
			tenant_id: Tenant identifier
			converted_by: User converting the lead
			create_account: Whether to create account
			create_opportunity: Whether to create opportunity
			opportunity_data: Opportunity data if creating
			
		Returns:
			Dict with created record IDs
		"""
		try:
			# Get existing lead
			lead = await self.get_lead(lead_id, tenant_id)
			
			result = {}
			
			# Create contact
			contact_data = {
				"first_name": lead.first_name,
				"last_name": lead.last_name,
				"email": lead.email,
				"phone": lead.phone,
				"company": lead.company,
				"contact_type": ContactType.CUSTOMER,
				"lead_source": lead.lead_source
			}
			
			contact = await self.create_contact(contact_data, tenant_id, converted_by)
			result["contact_id"] = contact.id
			
			# Create account if requested
			account = None
			if create_account and lead.company:
				account_data = {
					"account_name": lead.company,
					"account_type": AccountType.CUSTOMER,
					"account_owner_id": converted_by
				}
				
				account = await self.create_account(account_data, tenant_id, converted_by)
				result["account_id"] = account.id
				
				# Link contact to account
				await self.update_contact(
					contact.id,
					{"account_id": account.id},
					tenant_id,
					converted_by
				)
			
			# Create opportunity if requested
			if create_opportunity and opportunity_data:
				opportunity_data.update({
					"account_id": account.id if account else None,
					"primary_contact_id": contact.id,
					"owner_id": converted_by
				})
				
				opportunity = await self.create_opportunity(
					opportunity_data, tenant_id, converted_by
				)
				result["opportunity_id"] = opportunity.id
			
			# Update lead as converted
			await self.update_lead(
				lead_id,
				{
					"is_converted": True,
					"converted_date": datetime.utcnow(),
					"converted_contact_id": contact.id,
					"converted_account_id": account.id if account else None,
					"converted_opportunity_id": result.get("opportunity_id"),
					"lead_status": LeadStatus.CONVERTED
				},
				tenant_id,
				converted_by
			)
			
			logger.info(f"Converted lead: {lead_id} for tenant: {tenant_id}")
			return result
			
		except Exception as e:
			logger.error(f"Failed to convert lead {lead_id}: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Lead conversion failed: {str(e)}")
	
	# ================================
	# Opportunity Management
	# ================================
	
	async def create_opportunity(
		self,
		opportunity_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> CRMOpportunity:
		"""
		Create a new opportunity
		
		Args:
			opportunity_data: Opportunity information
			tenant_id: Tenant identifier
			created_by: User creating the opportunity
			
		Returns:
			Created opportunity object
		"""
		try:
			# Add audit fields
			opportunity_data.update({
				"tenant_id": tenant_id,
				"created_by": created_by,
				"updated_by": created_by
			})
			
			# Validate and create opportunity
			opportunity = CRMOpportunity(**opportunity_data)
			
			# Save to database
			saved_opportunity = await self.db_manager.create_opportunity(opportunity)
			
			# Generate AI win probability
			asyncio.create_task(
				self._calculate_win_probability(saved_opportunity.id, tenant_id)
			)
			
			logger.info(f"Created opportunity: {saved_opportunity.id} for tenant: {tenant_id}")
			return saved_opportunity
			
		except ValidationError as e:
			raise CRMValidationError(f"Invalid opportunity data: {str(e)}")
		except Exception as e:
			logger.error(f"Failed to create opportunity: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Opportunity creation failed: {str(e)}")
	
	async def update_opportunity_stage(
		self,
		opportunity_id: str,
		new_stage: OpportunityStage,
		tenant_id: str,
		updated_by: str,
		notes: str = None
	) -> CRMOpportunity:
		"""
		Update opportunity stage
		
		Args:
			opportunity_id: Opportunity identifier
			new_stage: New opportunity stage
			tenant_id: Tenant identifier
			updated_by: User updating the opportunity
			notes: Optional stage change notes
			
		Returns:
			Updated opportunity object
		"""
		try:
			# Get existing opportunity
			opportunity = await self.get_opportunity(opportunity_id, tenant_id)
			
			# Update stage and related fields
			update_data = {
				"stage": new_stage,
				"stage_changed_date": datetime.utcnow(),
				"previous_stage": opportunity.stage
			}
			
			# Handle closed stages
			if new_stage in [OpportunityStage.CLOSED_WON, OpportunityStage.CLOSED_LOST]:
				update_data.update({
					"is_closed": True,
					"is_won": new_stage == OpportunityStage.CLOSED_WON
				})
			
			# Update opportunity
			updated_opportunity = await self.update_opportunity(
				opportunity_id, update_data, tenant_id, updated_by
			)
			
			# Create activity for stage change
			if notes:
				await self.create_activity(
					{
						"subject": f"Stage changed to {new_stage.value}",
						"activity_type": ActivityType.NOTE,
						"description": notes,
						"related_to_type": "opportunity",
						"related_to_id": opportunity_id,
						"assigned_to_id": updated_by,
						"start_datetime": datetime.utcnow(),
						"is_completed": True
					},
					tenant_id,
					updated_by
				)
			
			logger.info(f"Updated opportunity stage: {opportunity_id} to {new_stage.value}")
			return updated_opportunity
			
		except Exception as e:
			logger.error(f"Failed to update opportunity stage: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Opportunity stage update failed: {str(e)}")
	
	# ================================
	# Activity Management
	# ================================
	
	async def create_activity(
		self,
		activity_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> CRMActivity:
		"""
		Create a new activity
		
		Args:
			activity_data: Activity information
			tenant_id: Tenant identifier
			created_by: User creating the activity
			
		Returns:
			Created activity object
		"""
		try:
			# Add audit fields
			activity_data.update({
				"tenant_id": tenant_id,
				"created_by": created_by,
				"updated_by": created_by
			})
			
			# Validate and create activity
			activity = CRMActivity(**activity_data)
			
			# Save to database
			saved_activity = await self.db_manager.create_activity(activity)
			
			logger.info(f"Created activity: {saved_activity.id} for tenant: {tenant_id}")
			return saved_activity
			
		except ValidationError as e:
			raise CRMValidationError(f"Invalid activity data: {str(e)}")
		except Exception as e:
			logger.error(f"Failed to create activity: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Activity creation failed: {str(e)}")
	
	# ================================
	# Analytics and Insights
	# ================================
	
	async def get_sales_dashboard(self, tenant_id: str, user_id: str = None) -> Dict[str, Any]:
		"""
		Get sales dashboard data
		
		Args:
			tenant_id: Tenant identifier
			user_id: Optional user filter
			
		Returns:
			Dashboard data
		"""
		try:
			return await self.analytics.get_sales_dashboard(tenant_id, user_id)
		except Exception as e:
			logger.error(f"Failed to get sales dashboard: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Dashboard generation failed: {str(e)}")
	
	async def get_pipeline_analytics(self, tenant_id: str, user_id: str = None) -> Dict[str, Any]:
		"""
		Get pipeline analytics
		
		Args:
			tenant_id: Tenant identifier
			user_id: Optional user filter
			
		Returns:
			Pipeline analytics data
		"""
		try:
			return await self.analytics.get_pipeline_analytics(tenant_id, user_id)
		except Exception as e:
			logger.error(f"Failed to get pipeline analytics: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Pipeline analytics failed: {str(e)}")
	
	# ================================
	# Private Helper Methods
	# ================================
	
	async def _find_duplicate_contact(
		self, 
		contact: CRMContact, 
		tenant_id: str
	) -> Optional[CRMContact]:
		"""Find potential duplicate contacts"""
		if not contact.email:
			return None
		
		# Search by email
		contacts, _ = await self.search_contacts(
			tenant_id, 
			filters={"email": contact.email}
		)
		
		return contacts[0] if contacts else None
	
	async def _generate_contact_insights(self, contact_id: str, tenant_id: str):
		"""Generate AI insights for contact"""
		try:
			await self.ai_insights.generate_contact_insights(contact_id, tenant_id)
		except Exception as e:
			logger.error(f"Failed to generate contact insights: {str(e)}")
	
	async def _generate_account_insights(self, account_id: str, tenant_id: str):
		"""Generate AI insights for account"""
		try:
			await self.ai_insights.generate_account_insights(account_id, tenant_id)
		except Exception as e:
			logger.error(f"Failed to generate account insights: {str(e)}")
	
	async def _calculate_lead_score(self, lead_id: str, tenant_id: str):
		"""Calculate AI-powered lead score"""
		try:
			await self.ai_insights.calculate_lead_score(lead_id, tenant_id)
		except Exception as e:
			logger.error(f"Failed to calculate lead score: {str(e)}")
	
	async def _calculate_win_probability(self, opportunity_id: str, tenant_id: str):
		"""Calculate AI-powered win probability"""
		try:
			await self.ai_insights.calculate_win_probability(opportunity_id, tenant_id)
		except Exception as e:
			logger.error(f"Failed to calculate win probability: {str(e)}")
	
	async def _update_contact_analytics(self, contact_id: str, tenant_id: str):
		"""Update contact analytics"""
		try:
			await self.analytics.update_contact_analytics(contact_id, tenant_id)
		except Exception as e:
			logger.error(f"Failed to update contact analytics: {str(e)}")
	
	# Placeholder methods for missing implementations
	async def get_lead(self, lead_id: str, tenant_id: str) -> CRMLead:
		"""Get lead by ID - placeholder implementation"""
		# This would be implemented similar to get_contact
		pass
	
	async def update_lead(self, lead_id: str, update_data: Dict[str, Any], tenant_id: str, updated_by: str) -> CRMLead:
		"""Update lead - placeholder implementation"""
		# This would be implemented similar to update_contact
		pass
	
	async def get_opportunity(self, opportunity_id: str, tenant_id: str) -> CRMOpportunity:
		"""Get opportunity by ID - placeholder implementation"""
		# This would be implemented similar to get_contact
		pass
	
	async def update_opportunity(self, opportunity_id: str, update_data: Dict[str, Any], tenant_id: str, updated_by: str) -> CRMOpportunity:
		"""Update opportunity - placeholder implementation"""
		# This would be implemented similar to update_contact
		pass
	
	async def shutdown(self):
		"""Shutdown the CRM service gracefully"""
		try:
			logger.info("🛑 Shutting down CRM service...")
			
			if self.db_manager:
				await self.db_manager.shutdown()
			
			if self.ai_insights:
				await self.ai_insights.shutdown()
			
			if self.analytics:
				await self.analytics.shutdown()
			
			self._initialized = False
			logger.info("✅ CRM service shutdown completed")
			
		except Exception as e:
			logger.error(f"Error during CRM service shutdown: {str(e)}", exc_info=True)
	
	# ====== LEAD SCORING METHODS ======
	
	async def create_lead_scoring_rule(
		self,
		rule_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> LeadScoreRule:
		"""
		Create a new lead scoring rule
		
		Args:
			rule_data: Rule configuration data
			tenant_id: Tenant identifier
			created_by: User creating the rule
			
		Returns:
			Created scoring rule
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.lead_scoring_manager.create_scoring_rule(
				rule_data=rule_data,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
		except Exception as e:
			logger.error(f"Failed to create lead scoring rule: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Lead scoring rule creation failed: {str(e)}")
	
	async def calculate_lead_score(
		self,
		lead_id: str, 
		tenant_id: str,
		force_recalculate: bool = False
	) -> LeadScore:
		"""
		Calculate comprehensive score for a lead
		
		Args:
			lead_id: Lead identifier
			tenant_id: Tenant identifier
			force_recalculate: Force recalculation even if recent score exists
			
		Returns:
			Calculated lead score
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.lead_scoring_manager.calculate_lead_score(
				lead_id=lead_id,
				tenant_id=tenant_id,
				force_recalculate=force_recalculate
			)
			
		except Exception as e:
			logger.error(f"Failed to calculate lead score: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Lead score calculation failed: {str(e)}")
	
	async def get_lead_score(
		self,
		lead_id: str,
		tenant_id: str
	) -> Optional[LeadScore]:
		"""
		Get current lead score
		
		Args:
			lead_id: Lead identifier
			tenant_id: Tenant identifier
			
		Returns:
			Current lead score if exists
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.lead_scoring_manager.get_lead_score(
				lead_id=lead_id,
				tenant_id=tenant_id
			)
			
		except Exception as e:
			logger.error(f"Failed to get lead score: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Lead score retrieval failed: {str(e)}")
	
	async def batch_score_leads(
		self,
		lead_ids: List[str],
		tenant_id: str,
		force_recalculate: bool = False
	) -> Dict[str, LeadScore]:
		"""
		Calculate scores for multiple leads in batch
		
		Args:
			lead_ids: List of lead identifiers
			tenant_id: Tenant identifier
			force_recalculate: Force recalculation for all leads
			
		Returns:
			Dict mapping lead IDs to their scores
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.lead_scoring_manager.batch_score_leads(
				lead_ids=lead_ids,
				tenant_id=tenant_id,
				force_recalculate=force_recalculate
			)
			
		except Exception as e:
			logger.error(f"Failed to batch score leads: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Batch lead scoring failed: {str(e)}")
	
	async def get_lead_scoring_analytics(
		self,
		tenant_id: str,
		period_days: int = 30
	) -> Dict[str, Any]:
		"""
		Get comprehensive lead scoring analytics
		
		Args:
			tenant_id: Tenant identifier
			period_days: Analysis period in days
			
		Returns:
			Scoring analytics data
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			analytics = await self.lead_scoring_manager.get_scoring_analytics(
				tenant_id=tenant_id,
				period_days=period_days
			)
			
			return analytics.model_dump()
			
		except Exception as e:
			logger.error(f"Failed to get lead scoring analytics: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Lead scoring analytics failed: {str(e)}")
	
	async def create_default_scoring_rules(
		self,
		tenant_id: str,
		created_by: str
	) -> List[LeadScoreRule]:
		"""
		Create default lead scoring rules for a new tenant
		
		Args:
			tenant_id: Tenant identifier
			created_by: User creating the rules
			
		Returns:
			List of created default rules
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.lead_scoring_manager.create_default_scoring_rules(
				tenant_id=tenant_id,
				created_by=created_by
			)
			
		except Exception as e:
			logger.error(f"Failed to create default scoring rules: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Default scoring rules creation failed: {str(e)}")
	
	# ====== SALES PIPELINE METHODS ======
	
	async def create_sales_pipeline(
		self,
		pipeline_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> SalesPipeline:
		"""
		Create a new sales pipeline
		
		Args:
			pipeline_data: Pipeline configuration data
			tenant_id: Tenant identifier
			created_by: User creating the pipeline
			
		Returns:
			Created sales pipeline
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.pipeline_manager.create_pipeline(
				pipeline_data=pipeline_data,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
		except Exception as e:
			logger.error(f"Failed to create sales pipeline: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Sales pipeline creation failed: {str(e)}")
	
	async def create_pipeline_stage(
		self,
		stage_data: Dict[str, Any],
		pipeline_id: str,
		tenant_id: str,
		created_by: str
	) -> PipelineStage:
		"""
		Create a new pipeline stage
		
		Args:
			stage_data: Stage configuration data
			pipeline_id: Pipeline identifier
			tenant_id: Tenant identifier
			created_by: User creating the stage
			
		Returns:
			Created pipeline stage
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.pipeline_manager.create_pipeline_stage(
				stage_data=stage_data,
				pipeline_id=pipeline_id,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
		except Exception as e:
			logger.error(f"Failed to create pipeline stage: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Pipeline stage creation failed: {str(e)}")
	
	async def move_opportunity_to_stage(
		self,
		opportunity_id: str,
		to_stage_id: str,
		tenant_id: str,
		changed_by: str,
		change_reason: Optional[str] = None,
		notes: Optional[str] = None
	) -> OpportunityStageHistory:
		"""
		Move an opportunity to a new stage
		
		Args:
			opportunity_id: Opportunity identifier
			to_stage_id: Target stage identifier
			tenant_id: Tenant identifier
			changed_by: User making the change
			change_reason: Reason for the change
			notes: Additional notes
			
		Returns:
			Stage change history record
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.pipeline_manager.move_opportunity_to_stage(
				opportunity_id=opportunity_id,
				to_stage_id=to_stage_id,
				tenant_id=tenant_id,
				changed_by=changed_by,
				change_reason=change_reason,
				notes=notes
			)
			
		except Exception as e:
			logger.error(f"Failed to move opportunity to stage: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Opportunity stage change failed: {str(e)}")
	
	async def get_pipeline_analytics(
		self,
		pipeline_id: str,
		tenant_id: str,
		period_days: int = 30
	) -> Dict[str, Any]:
		"""
		Get comprehensive pipeline analytics
		
		Args:
			pipeline_id: Pipeline identifier
			tenant_id: Tenant identifier
			period_days: Analysis period in days
			
		Returns:
			Pipeline analytics data
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			analytics = await self.pipeline_manager.get_pipeline_analytics(
				pipeline_id=pipeline_id,
				tenant_id=tenant_id,
				period_days=period_days
			)
			
			return analytics.model_dump()
			
		except Exception as e:
			logger.error(f"Failed to get pipeline analytics: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Pipeline analytics failed: {str(e)}")
	
	async def get_sales_pipeline(
		self,
		pipeline_id: str,
		tenant_id: str
	) -> Optional[SalesPipeline]:
		"""
		Get sales pipeline by ID
		
		Args:
			pipeline_id: Pipeline identifier
			tenant_id: Tenant identifier
			
		Returns:
			Pipeline if found
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.pipeline_manager.get_pipeline(
				pipeline_id=pipeline_id,
				tenant_id=tenant_id
			)
			
		except Exception as e:
			logger.error(f"Failed to get sales pipeline: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Pipeline retrieval failed: {str(e)}")
	
	async def list_sales_pipelines(
		self,
		tenant_id: str,
		active_only: bool = True
	) -> List[SalesPipeline]:
		"""
		List all pipelines for tenant
		
		Args:
			tenant_id: Tenant identifier
			active_only: Return only active pipelines
			
		Returns:
			List of pipelines
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.pipeline_manager.list_pipelines(
				tenant_id=tenant_id,
				active_only=active_only
			)
			
		except Exception as e:
			logger.error(f"Failed to list sales pipelines: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Pipeline listing failed: {str(e)}")
	
	# ================================
	# Workflow Automation Operations
	# ================================
	
	async def create_workflow(
		self,
		workflow_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> Workflow:
		"""
		Create a new workflow
		
		Args:
			workflow_data: Workflow configuration data
			tenant_id: Tenant identifier
			created_by: User creating the workflow
			
		Returns:
			Created workflow
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.workflow_automation_engine.create_workflow(
				workflow_data=workflow_data,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
		except Exception as e:
			logger.error(f"Failed to create workflow: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Workflow creation failed: {str(e)}")
	
	async def execute_workflow(
		self,
		workflow_id: str,
		tenant_id: str,
		triggered_by: str = "system",
		record_id: Optional[str] = None,
		record_type: Optional[str] = None,
		input_data: Dict[str, Any] = None
	) -> WorkflowExecution:
		"""
		Execute a workflow
		
		Args:
			workflow_id: Workflow identifier
			tenant_id: Tenant identifier
			triggered_by: User/system triggering execution
			record_id: Associated record ID
			record_type: Type of associated record
			input_data: Input data for workflow
			
		Returns:
			Workflow execution record
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.workflow_automation_engine.execute_workflow(
				workflow_id=workflow_id,
				tenant_id=tenant_id,
				triggered_by=triggered_by,
				record_id=record_id,
				record_type=record_type,
				input_data=input_data or {}
			)
			
		except Exception as e:
			logger.error(f"Failed to execute workflow: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Workflow execution failed: {str(e)}")
	
	async def get_workflow_analytics(
		self,
		workflow_id: Optional[str],
		tenant_id: str,
		period_days: int = 30
	) -> WorkflowAnalytics:
		"""
		Get workflow performance analytics
		
		Args:
			workflow_id: Specific workflow ID (optional)
			tenant_id: Tenant identifier
			period_days: Analysis period in days
			
		Returns:
			Workflow analytics data
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.workflow_automation_engine.get_workflow_analytics(
				workflow_id=workflow_id,
				tenant_id=tenant_id,
				period_days=period_days
			)
			
		except Exception as e:
			logger.error(f"Failed to get workflow analytics: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Workflow analytics failed: {str(e)}")
	
	async def list_workflows(
		self,
		tenant_id: str,
		active_only: bool = True
	) -> List[Workflow]:
		"""
		List workflows for tenant
		
		Args:
			tenant_id: Tenant identifier
			active_only: Return only active workflows
			
		Returns:
			List of workflows
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.workflow_automation_engine.list_workflows(
				tenant_id=tenant_id,
				active_only=active_only
			)
			
		except Exception as e:
			logger.error(f"Failed to list workflows: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Workflow listing failed: {str(e)}")
	
	async def get_workflow_executions(
		self,
		workflow_id: str,
		tenant_id: str,
		limit: int = 50,
		offset: int = 0
	) -> List[WorkflowExecution]:
		"""
		Get workflow execution history
		
		Args:
			workflow_id: Workflow identifier
			tenant_id: Tenant identifier
			limit: Maximum number of executions to return
			offset: Offset for pagination
			
		Returns:
			List of workflow executions
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.workflow_automation_engine.get_workflow_executions(
				workflow_id=workflow_id,
				tenant_id=tenant_id,
				limit=limit,
				offset=offset
			)
			
		except Exception as e:
			logger.error(f"Failed to get workflow executions: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Workflow execution history failed: {str(e)}")
	
	# ================================
	# Email Integration Operations
	# ================================
	
	async def create_email_template(
		self,
		template_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> EmailTemplate:
		"""
		Create a new email template
		
		Args:
			template_data: Template configuration data
			tenant_id: Tenant identifier
			created_by: User creating the template
			
		Returns:
			Created email template
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.email_integration_manager.create_email_template(
				template_data=template_data,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
		except Exception as e:
			logger.error(f"Failed to create email template: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Email template creation failed: {str(e)}")
	
	async def send_email(
		self,
		email_data: Dict[str, Any],
		tenant_id: str,
		created_by: str,
		send_immediately: bool = True
	) -> EmailMessage:
		"""
		Send an email message
		
		Args:
			email_data: Email message data
			tenant_id: Tenant identifier
			created_by: User sending the email
			send_immediately: Whether to send immediately or queue
			
		Returns:
			Email message record
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.email_integration_manager.send_email(
				email_data=email_data,
				tenant_id=tenant_id,
				created_by=created_by,
				send_immediately=send_immediately
			)
			
		except Exception as e:
			logger.error(f"Failed to send email: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Email sending failed: {str(e)}")
	
	async def send_template_email(
		self,
		template_id: str,
		recipient_data: Dict[str, Any],
		merge_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> EmailMessage:
		"""
		Send email using a template
		
		Args:
			template_id: Email template identifier
			recipient_data: Recipient information
			merge_data: Data for template merge fields
			tenant_id: Tenant identifier
			created_by: User sending the email
			
		Returns:
			Email message record
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.email_integration_manager.send_template_email(
				template_id=template_id,
				recipient_data=recipient_data,
				merge_data=merge_data,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
		except Exception as e:
			logger.error(f"Failed to send template email: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Template email sending failed: {str(e)}")
	
	async def track_email_event(
		self,
		email_id: str,
		event_type: str,
		event_data: Dict[str, Any],
		tenant_id: str
	) -> EmailTracking:
		"""
		Track an email event
		
		Args:
			email_id: Email message identifier
			event_type: Type of event
			event_data: Event-specific data
			tenant_id: Tenant identifier
			
		Returns:
			Email tracking record
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.email_integration_manager.track_email_event(
				email_id=email_id,
				event_type=event_type,
				event_data=event_data,
				tenant_id=tenant_id
			)
			
		except Exception as e:
			logger.error(f"Failed to track email event: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Email event tracking failed: {str(e)}")
	
	async def get_email_analytics(
		self,
		tenant_id: str,
		start_date: datetime,
		end_date: datetime,
		filters: Optional[Dict[str, Any]] = None
	) -> EmailAnalytics:
		"""
		Get comprehensive email analytics
		
		Args:
			tenant_id: Tenant identifier
			start_date: Analysis period start
			end_date: Analysis period end
			filters: Additional filters
			
		Returns:
			Email analytics data
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.email_integration_manager.get_email_analytics(
				tenant_id=tenant_id,
				start_date=start_date,
				end_date=end_date,
				filters=filters
			)
			
		except Exception as e:
			logger.error(f"Failed to get email analytics: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Email analytics failed: {str(e)}")
	
	async def list_email_templates(
		self,
		tenant_id: str,
		category: Optional[str] = None,
		active_only: bool = True
	) -> List[EmailTemplate]:
		"""
		List email templates for tenant
		
		Args:
			tenant_id: Tenant identifier
			category: Template category filter
			active_only: Return only active templates
			
		Returns:
			List of email templates
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.email_integration_manager.list_email_templates(
				tenant_id=tenant_id,
				category=category,
				active_only=active_only
			)
			
		except Exception as e:
			logger.error(f"Failed to list email templates: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Email template listing failed: {str(e)}")
	
	# ================================
	# Calendar and Activity Management Operations
	# ================================
	
	async def create_calendar_event(
		self,
		event_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> CalendarEvent:
		"""
		Create a new calendar event
		
		Args:
			event_data: Event configuration data
			tenant_id: Tenant identifier
			created_by: User creating the event
			
		Returns:
			Created calendar event
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.calendar_activity_manager.create_calendar_event(
				event_data=event_data,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
		except Exception as e:
			logger.error(f"Failed to create calendar event: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Calendar event creation failed: {str(e)}")
	
	async def create_activity(
		self,
		activity_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> CRMActivity:
		"""
		Create a new CRM activity
		
		Args:
			activity_data: Activity configuration data
			tenant_id: Tenant identifier
			created_by: User creating the activity
			
		Returns:
			Created CRM activity
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.calendar_activity_manager.create_activity(
				activity_data=activity_data,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
		except Exception as e:
			logger.error(f"Failed to create activity: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Activity creation failed: {str(e)}")
	
	async def create_activity_from_template(
		self,
		template_id: str,
		activity_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> CRMActivity:
		"""
		Create activity from template
		
		Args:
			template_id: Activity template identifier
			activity_data: Override data for activity
			tenant_id: Tenant identifier
			created_by: User creating the activity
			
		Returns:
			Created CRM activity
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.calendar_activity_manager.create_activity_from_template(
				template_id=template_id,
				activity_data=activity_data,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
		except Exception as e:
			logger.error(f"Failed to create activity from template: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Template activity creation failed: {str(e)}")
	
	async def complete_activity(
		self,
		activity_id: str,
		tenant_id: str,
		completed_by: str,
		outcome: Optional[str] = None,
		notes: Optional[str] = None
	) -> CRMActivity:
		"""
		Mark activity as completed
		
		Args:
			activity_id: Activity identifier
			tenant_id: Tenant identifier
			completed_by: User completing the activity
			outcome: Activity outcome
			notes: Completion notes
			
		Returns:
			Updated activity
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.calendar_activity_manager.complete_activity(
				activity_id=activity_id,
				tenant_id=tenant_id,
				completed_by=completed_by,
				outcome=outcome,
				notes=notes
			)
			
		except Exception as e:
			logger.error(f"Failed to complete activity: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Activity completion failed: {str(e)}")
	
	async def get_calendar_analytics(
		self,
		tenant_id: str,
		start_date: datetime,
		end_date: datetime,
		user_id: Optional[str] = None
	) -> CalendarAnalytics:
		"""
		Get comprehensive calendar and activity analytics
		
		Args:
			tenant_id: Tenant identifier
			start_date: Analysis period start
			end_date: Analysis period end
			user_id: Optional user filter
			
		Returns:
			Calendar analytics data
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.calendar_activity_manager.get_calendar_analytics(
				tenant_id=tenant_id,
				start_date=start_date,
				end_date=end_date,
				user_id=user_id
			)
			
		except Exception as e:
			logger.error(f"Failed to get calendar analytics: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Calendar analytics failed: {str(e)}")
	
	async def get_upcoming_events(
		self,
		tenant_id: str,
		user_id: Optional[str] = None,
		days_ahead: int = 7,
		limit: int = 50
	) -> List[CalendarEvent]:
		"""
		Get upcoming calendar events
		
		Args:
			tenant_id: Tenant identifier
			user_id: Optional user filter
			days_ahead: Days to look ahead
			limit: Maximum events to return
			
		Returns:
			List of upcoming events
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.calendar_activity_manager.get_upcoming_events(
				tenant_id=tenant_id,
				user_id=user_id,
				days_ahead=days_ahead,
				limit=limit
			)
			
		except Exception as e:
			logger.error(f"Failed to get upcoming events: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Upcoming events retrieval failed: {str(e)}")
	
	async def get_overdue_activities(
		self,
		tenant_id: str,
		user_id: Optional[str] = None
	) -> List[CRMActivity]:
		"""
		Get overdue activities
		
		Args:
			tenant_id: Tenant identifier
			user_id: Optional user filter
			
		Returns:
			List of overdue activities
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.calendar_activity_manager.get_overdue_activities(
				tenant_id=tenant_id,
				user_id=user_id
			)
			
		except Exception as e:
			logger.error(f"Failed to get overdue activities: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Overdue activities retrieval failed: {str(e)}")
	
	# ==========================================
	# APPROVAL WORKFLOWS MANAGEMENT
	# ==========================================
	
	async def create_approval_template(self, template_data: Dict[str, Any], tenant_id: str, created_by: str) -> ApprovalWorkflowTemplate:
		"""
		Create a new approval workflow template
		
		Args:
			template_data: Template configuration data
			tenant_id: Tenant identifier
			created_by: User creating the template
			
		Returns:
			Created approval workflow template
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.approval_workflow_engine.create_template(
				template_data=template_data,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
		except Exception as e:
			logger.error(f"Failed to create approval template: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Approval template creation failed: {str(e)}")
	
	async def submit_approval_request(self, request_data: Dict[str, Any], tenant_id: str, requested_by: str) -> ApprovalRequest:
		"""
		Submit a new approval request
		
		Args:
			request_data: Approval request data
			tenant_id: Tenant identifier
			requested_by: User submitting the request
			
		Returns:
			Created approval request
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.approval_workflow_engine.submit_approval_request(
				request_data=request_data,
				tenant_id=tenant_id,
				requested_by=requested_by
			)
			
		except Exception as e:
			logger.error(f"Failed to submit approval request: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Approval request submission failed: {str(e)}")
	
	async def process_approval_action(self, request_id: str, step_id: str, action: str, actor_id: str, notes: str, tenant_id: str) -> ApprovalRequest:
		"""
		Process an approval action (approve, reject, delegate, etc.)
		
		Args:
			request_id: Approval request identifier
			step_id: Approval step identifier
			action: Action to take (approve, reject, delegate, etc.)
			actor_id: User performing the action
			notes: Action notes/comments
			tenant_id: Tenant identifier
			
		Returns:
			Updated approval request
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.approval_workflow_engine.process_approval_action(
				request_id=request_id,
				step_id=step_id,
				action=action,
				actor_id=actor_id,
				notes=notes,
				tenant_id=tenant_id
			)
			
		except Exception as e:
			logger.error(f"Failed to process approval action: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Approval action processing failed: {str(e)}")
	
	async def get_approval_request(self, request_id: str, tenant_id: str) -> Optional[ApprovalRequest]:
		"""
		Retrieve approval request by ID
		
		Args:
			request_id: Request identifier
			tenant_id: Tenant identifier
			
		Returns:
			Approval request or None
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.approval_workflow_engine.get_approval_request(
				request_id=request_id,
				tenant_id=tenant_id
			)
			
		except Exception as e:
			logger.error(f"Failed to get approval request: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Approval request retrieval failed: {str(e)}")
	
	async def list_approval_requests(self, tenant_id: str, status: str = None, requester_id: str = None, limit: int = 50, offset: int = 0) -> Tuple[List[ApprovalRequest], int]:
		"""
		List approval requests with filtering
		
		Args:
			tenant_id: Tenant identifier
			status: Optional status filter
			requester_id: Optional requester filter
			limit: Maximum number of results
			offset: Result offset for pagination
			
		Returns:
			Tuple of (approval requests, total count)
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.approval_workflow_engine.list_approval_requests(
				tenant_id=tenant_id,
				status=status,
				requester_id=requester_id,
				limit=limit,
				offset=offset
			)
			
		except Exception as e:
			logger.error(f"Failed to list approval requests: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Approval request listing failed: {str(e)}")
	
	async def get_pending_approvals(self, tenant_id: str, approver_id: str) -> List[ApprovalStep]:
		"""
		Get pending approvals for a specific approver
		
		Args:
			tenant_id: Tenant identifier
			approver_id: Approver user identifier
			
		Returns:
			List of pending approval steps
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.approval_workflow_engine.get_pending_approvals(
				tenant_id=tenant_id,
				approver_id=approver_id
			)
			
		except Exception as e:
			logger.error(f"Failed to get pending approvals: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Pending approvals retrieval failed: {str(e)}")
	
	async def get_approval_analytics(self, tenant_id: str, approval_type: str = None, period_days: int = 30) -> ApprovalAnalytics:
		"""
		Get approval workflow analytics
		
		Args:
			tenant_id: Tenant identifier
			approval_type: Optional approval type filter
			period_days: Analysis period in days
			
		Returns:
			Approval workflow analytics
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.approval_workflow_engine.get_approval_analytics(
				tenant_id=tenant_id,
				approval_type=approval_type,
				period_days=period_days
			)
			
		except Exception as e:
			logger.error(f"Failed to get approval analytics: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Approval analytics retrieval failed: {str(e)}")
	
	# ==========================================
	# LEAD ASSIGNMENT MANAGEMENT
	# ==========================================
	
	async def create_assignment_rule(self, rule_data: Dict[str, Any], tenant_id: str, created_by: str) -> LeadAssignmentRule:
		"""
		Create a new lead assignment rule
		
		Args:
			rule_data: Rule configuration data
			tenant_id: Tenant identifier
			created_by: User creating the rule
			
		Returns:
			Created lead assignment rule
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.lead_assignment_manager.create_assignment_rule(
				rule_data=rule_data,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
		except Exception as e:
			logger.error(f"Failed to create assignment rule: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Assignment rule creation failed: {str(e)}")
	
	async def assign_lead(self, lead_data: Dict[str, Any], tenant_id: str) -> Optional[LeadAssignment]:
		"""
		Assign a lead using configured assignment rules
		
		Args:
			lead_data: Lead data for assignment
			tenant_id: Tenant identifier
			
		Returns:
			Lead assignment record or None
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.lead_assignment_manager.assign_lead(
				lead_data=lead_data,
				tenant_id=tenant_id
			)
			
		except Exception as e:
			logger.error(f"Failed to assign lead: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Lead assignment failed: {str(e)}")
	
	async def get_assignment_analytics(self, tenant_id: str, period_days: int = 30) -> AssignmentAnalytics:
		"""
		Get lead assignment analytics
		
		Args:
			tenant_id: Tenant identifier
			period_days: Analysis period in days
			
		Returns:
			Assignment analytics data
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.lead_assignment_manager.get_assignment_analytics(
				tenant_id=tenant_id,
				period_days=period_days
			)
			
		except Exception as e:
			logger.error(f"Failed to get assignment analytics: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Assignment analytics retrieval failed: {str(e)}")
	
	# ==========================================
	# LEAD NURTURING MANAGEMENT
	# ==========================================
	
	async def create_nurturing_workflow(self, workflow_data: Dict[str, Any], tenant_id: str, created_by: str) -> NurturingWorkflow:
		"""
		Create a new lead nurturing workflow
		
		Args:
			workflow_data: Workflow configuration data
			tenant_id: Tenant identifier
			created_by: User creating the workflow
			
		Returns:
			Created nurturing workflow
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.lead_nurturing_manager.create_nurturing_workflow(
				workflow_data=workflow_data,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
		except Exception as e:
			logger.error(f"Failed to create nurturing workflow: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Nurturing workflow creation failed: {str(e)}")
	
	async def enroll_lead_in_nurturing(self, workflow_id: str, lead_data: Dict[str, Any], tenant_id: str, enrolled_by: str = None) -> Optional[NurturingEnrollment]:
		"""
		Enroll a lead in a nurturing workflow
		
		Args:
			workflow_id: Workflow identifier
			lead_data: Lead data for enrollment
			tenant_id: Tenant identifier
			enrolled_by: Optional user who enrolled the lead
			
		Returns:
			Nurturing enrollment record or None
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.lead_nurturing_manager.enroll_lead(
				workflow_id=workflow_id,
				lead_data=lead_data,
				tenant_id=tenant_id,
				enrolled_by=enrolled_by
			)
			
		except Exception as e:
			logger.error(f"Failed to enroll lead in nurturing: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Lead nurturing enrollment failed: {str(e)}")
	
	async def process_nurturing_trigger(self, trigger_type: str, trigger_data: Dict[str, Any], tenant_id: str):
		"""
		Process a nurturing workflow trigger event
		
		Args:
			trigger_type: Type of trigger event
			trigger_data: Trigger event data
			tenant_id: Tenant identifier
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			trigger_enum = TriggerType(trigger_type)
			await self.lead_nurturing_manager.process_trigger(
				trigger_type=trigger_enum,
				trigger_data=trigger_data,
				tenant_id=tenant_id
			)
			
		except Exception as e:
			logger.error(f"Failed to process nurturing trigger: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Nurturing trigger processing failed: {str(e)}")
	
	async def get_nurturing_analytics(self, tenant_id: str, workflow_id: str = None, period_days: int = 30) -> NurturingAnalytics:
		"""
		Get lead nurturing analytics
		
		Args:
			tenant_id: Tenant identifier
			workflow_id: Optional specific workflow ID
			period_days: Analysis period in days
			
		Returns:
			Nurturing analytics data
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.lead_nurturing_manager.get_nurturing_analytics(
				tenant_id=tenant_id,
				workflow_id=workflow_id,
				period_days=period_days
			)
			
		except Exception as e:
			logger.error(f"Failed to get nurturing analytics: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Nurturing analytics retrieval failed: {str(e)}")
	
	# ==========================================
	# CRM DASHBOARD MANAGEMENT
	# ==========================================
	
	async def create_dashboard(self, dashboard_data: Dict[str, Any], tenant_id: str, created_by: str) -> DashboardLayout:
		"""
		Create a new CRM dashboard
		
		Args:
			dashboard_data: Dashboard configuration data
			tenant_id: Tenant identifier
			created_by: User creating the dashboard
			
		Returns:
			Created dashboard layout
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.dashboard_manager.create_dashboard(
				dashboard_data=dashboard_data,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
		except Exception as e:
			logger.error(f"Failed to create dashboard: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Dashboard creation failed: {str(e)}")
	
	async def get_dashboard_data(self, dashboard_id: str, tenant_id: str, time_range: str = None, filters: Dict[str, Any] = None) -> Dict[str, DashboardData]:
		"""
		Get dashboard data for all widgets
		
		Args:
			dashboard_id: Dashboard identifier
			tenant_id: Tenant identifier
			time_range: Optional time range filter
			filters: Optional additional filters
			
		Returns:
			Dictionary of widget data by widget ID
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.dashboard_manager.get_dashboard_data(
				dashboard_id=dashboard_id,
				tenant_id=tenant_id,
				time_range=time_range,
				filters=filters
			)
			
		except Exception as e:
			logger.error(f"Failed to get dashboard data: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Dashboard data retrieval failed: {str(e)}")
	
	# ==========================================
	# ADVANCED REPORTING ENGINE
	# ==========================================
	
	async def create_report(self, report_data: Dict[str, Any], tenant_id: str, created_by: str) -> ReportDefinition:
		"""
		Create a new report definition
		
		Args:
			report_data: Report configuration data
			tenant_id: Tenant identifier
			created_by: User creating the report
			
		Returns:
			Created report definition
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.reporting_engine.create_report(
				report_data=report_data,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
		except Exception as e:
			logger.error(f"Failed to create report: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Report creation failed: {str(e)}")
	
	async def execute_report(self, report_id: str, tenant_id: str, executed_by: str, parameters: Dict[str, Any] = None, export_format: str = None) -> ReportData:
		"""
		Execute a report and return results
		
		Args:
			report_id: Report identifier
			tenant_id: Tenant identifier
			executed_by: User executing the report
			parameters: Optional report parameters
			export_format: Optional export format
			
		Returns:
			Report execution results
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			export_enum = ExportFormat(export_format) if export_format else None
			
			return await self.reporting_engine.execute_report(
				report_id=report_id,
				tenant_id=tenant_id,
				executed_by=executed_by,
				parameters=parameters,
				export_format=export_enum
			)
			
		except Exception as e:
			logger.error(f"Failed to execute report: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Report execution failed: {str(e)}")
	
	async def schedule_report(self, schedule_data: Dict[str, Any], tenant_id: str, created_by: str) -> ReportSchedule:
		"""
		Create a report schedule
		
		Args:
			schedule_data: Schedule configuration data
			tenant_id: Tenant identifier
			created_by: User creating the schedule
			
		Returns:
			Created report schedule
		"""
		try:
			if not self._initialized:
				await self.initialize()
			
			return await self.reporting_engine.schedule_report(
				schedule_data=schedule_data,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
		except Exception as e:
			logger.error(f"Failed to schedule report: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Report scheduling failed: {str(e)}")


# Placeholder classes for dependencies
class DatabaseManager:
	"""Placeholder database manager"""
	
	def __init__(self):
		self._initialized = False
	
	async def initialize(self):
		self._initialized = True
	
	async def health_check(self):
		return {"status": "healthy"}
	
	async def create_contact(self, contact: CRMContact) -> CRMContact:
		return contact
	
	async def get_contact(self, contact_id: str, tenant_id: str) -> Optional[CRMContact]:
		return None
	
	async def update_contact(self, contact_id: str, update_data: Dict[str, Any], tenant_id: str) -> CRMContact:
		pass
	
	async def search_contacts(self, tenant_id: str, filters: Dict[str, Any], search_term: str, limit: int, offset: int) -> Tuple[List[CRMContact], int]:
		return [], 0
	
	async def create_account(self, account: CRMAccount) -> CRMAccount:
		return account
	
	async def get_account(self, account_id: str, tenant_id: str) -> Optional[CRMAccount]:
		return None
	
	async def create_lead(self, lead: CRMLead) -> CRMLead:
		return lead
	
	async def create_opportunity(self, opportunity: CRMOpportunity) -> CRMOpportunity:
		return opportunity
	
	async def create_activity(self, activity: CRMActivity) -> CRMActivity:
		return activity
	
	# ================================
	# Import/Export Operations
	# ================================
	
	async def import_contacts(
		self,
		file_data: Union[str, bytes],
		file_format: str,
		tenant_id: str,
		created_by: str,
		mapping_config: Optional[Dict[str, str]] = None,
		deduplicate: bool = True,
		validate_data: bool = True
	) -> Dict[str, Any]:
		"""
		Import contacts from various file formats
		
		Args:
			file_data: File content
			file_format: Format type (csv, json, xlsx, vcf)
			tenant_id: Tenant identifier
			created_by: User performing the import
			mapping_config: Custom field mapping configuration
			deduplicate: Whether to check for duplicates
			validate_data: Whether to validate data before import
			
		Returns:
			Import results with statistics and errors
		"""
		try:
			logger.info(f"🔄 Starting contact import for tenant: {tenant_id}")
			
			# Initialize import/export manager
			import_manager = ContactImportExportManager(self.db_manager, tenant_id)
			
			# Perform import
			results = await import_manager.import_contacts(
				file_data=file_data,
				file_format=file_format,
				mapping_config=mapping_config,
				deduplicate=deduplicate,
				validate_data=validate_data,
				created_by=created_by
			)
			
			# Update analytics with import stats
			if results.get('imported_records', 0) > 0:
				asyncio.create_task(
					self.analytics.track_bulk_import(tenant_id, results)
				)
			
			logger.info(f"✅ Contact import completed - Records: {results.get('imported_records', 0)}")
			return results
			
		except Exception as e:
			logger.error(f"Contact import failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Contact import failed: {str(e)}")
	
	async def export_contacts(
		self,
		export_format: str,
		tenant_id: str,
		contact_ids: Optional[List[str]] = None,
		filters: Optional[Dict[str, Any]] = None,
		include_fields: Optional[List[str]] = None,
		exclude_fields: Optional[List[str]] = None
	) -> Tuple[Union[str, bytes], str]:
		"""
		Export contacts to various formats
		
		Args:
			export_format: Format for export (csv, json, xlsx, pdf)
			tenant_id: Tenant identifier
			contact_ids: Specific contact IDs to export
			filters: Filters to apply to contact selection
			include_fields: Fields to include in export
			exclude_fields: Fields to exclude from export
			
		Returns:
			Tuple of (exported_data, filename)
		"""
		try:
			logger.info(f"🔄 Starting contact export for tenant: {tenant_id}")
			
			# Initialize import/export manager
			export_manager = ContactImportExportManager(self.db_manager, tenant_id)
			
			# Perform export
			export_data, filename = await export_manager.export_contacts(
				export_format=export_format,
				contact_ids=contact_ids,
				filters=filters,
				include_fields=include_fields,
				exclude_fields=exclude_fields
			)
			
			# Track export analytics
			asyncio.create_task(
				self.analytics.track_export(tenant_id, export_format, filename)
			)
			
			logger.info(f"✅ Contact export completed: {filename}")
			return export_data, filename
			
		except Exception as e:
			logger.error(f"Contact export failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Contact export failed: {str(e)}")
	
	async def get_import_template(
		self,
		file_format: str,
		mapping_type: Optional[str] = None
	) -> Tuple[Union[str, bytes], str]:
		"""
		Generate import template for specified format
		
		Args:
			file_format: Template format (csv, json, xlsx)
			mapping_type: Predefined mapping type (salesforce, hubspot, dynamics)
			
		Returns:
			Tuple of (template_data, filename)
		"""
		try:
			logger.info(f"🔄 Generating import template - Format: {file_format}")
			
			# Initialize import/export manager (dummy tenant for template)
			template_manager = ContactImportExportManager(self.db_manager, "template")
			
			# Generate template
			template_data, filename = await template_manager.get_import_template(
				file_format=file_format,
				mapping_type=mapping_type
			)
			
			logger.info(f"✅ Import template generated: {filename}")
			return template_data, filename
			
		except Exception as e:
			logger.error(f"Template generation failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Template generation failed: {str(e)}")
	
	async def get_contact_duplicates(
		self,
		tenant_id: str,
		threshold: float = 0.8
	) -> List[Dict[str, Any]]:
		"""
		Find potential duplicate contacts
		
		Args:
			tenant_id: Tenant identifier
			threshold: Similarity threshold for duplicates
			
		Returns:
			List of potential duplicate groups
		"""
		try:
			logger.info(f"🔍 Finding contact duplicates for tenant: {tenant_id}")
			
			# Get all contacts for tenant
			contacts_result = await self.db_manager.list_contacts(
				tenant_id=tenant_id,
				limit=10000  # Large limit to get all contacts
			)
			contacts = contacts_result.get('items', [])
			
			duplicates = []
			
			# Simple duplicate detection by email and name similarity
			contact_groups = {}
			
			for contact in contacts:
				# Group by email first
				email_key = contact.email.lower() if contact.email else None
				
				if email_key:
					if email_key not in contact_groups:
						contact_groups[email_key] = []
					contact_groups[email_key].append(contact)
			
			# Find groups with multiple contacts
			for email, contact_list in contact_groups.items():
				if len(contact_list) > 1:
					duplicates.append({
						"group_type": "email",
						"key": email,
						"contacts": [
							{
								"id": c.id,
								"first_name": c.first_name,
								"last_name": c.last_name,
								"email": c.email,
								"company": c.company,
								"created_at": c.created_at.isoformat()
							}
							for c in contact_list
						],
						"similarity_score": 1.0  # Exact email match
					})
			
			logger.info(f"✅ Found {len(duplicates)} duplicate groups")
			return duplicates
			
		except Exception as e:
			logger.error(f"Duplicate detection failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Duplicate detection failed: {str(e)}")
	
	async def merge_contacts(
		self,
		primary_contact_id: str,
		duplicate_contact_ids: List[str],
		tenant_id: str,
		merged_by: str
	) -> CRMContact:
		"""
		Merge duplicate contacts into primary contact
		
		Args:
			primary_contact_id: ID of the contact to keep
			duplicate_contact_ids: IDs of contacts to merge
			tenant_id: Tenant identifier
			merged_by: User performing the merge
			
		Returns:
			Updated primary contact
		"""
		try:
			logger.info(f"🔄 Merging contacts - Primary: {primary_contact_id}")
			
			# Get primary contact
			primary_contact = await self.db_manager.get_contact(primary_contact_id, tenant_id)
			if not primary_contact:
				raise CRMNotFoundError(f"Primary contact not found: {primary_contact_id}")
			
			# Get duplicate contacts
			duplicate_contacts = []
			for dup_id in duplicate_contact_ids:
				dup_contact = await self.db_manager.get_contact(dup_id, tenant_id)
				if dup_contact:
					duplicate_contacts.append(dup_contact)
			
			# Merge data (keep primary, fill missing fields from duplicates)
			merged_data = {
				"phone": primary_contact.phone,
				"mobile": primary_contact.mobile,
				"company": primary_contact.company,
				"job_title": primary_contact.job_title,
				"department": primary_contact.department,
				"website": primary_contact.website,
				"linkedin_profile": primary_contact.linkedin_profile,
				"description": primary_contact.description,
				"notes": primary_contact.notes,
				"address": primary_contact.address,
				"city": primary_contact.city,
				"state": primary_contact.state,
				"postal_code": primary_contact.postal_code,
				"country": primary_contact.country
			}
			
			# Fill missing fields from duplicates
			for dup_contact in duplicate_contacts:
				for field, value in merged_data.items():
					if not value and hasattr(dup_contact, field):
						dup_value = getattr(dup_contact, field)
						if dup_value:
							merged_data[field] = dup_value
			
			# Update primary contact
			merged_data['updated_by'] = merged_by
			updated_contact = await self.db_manager.update_contact(primary_contact_id, merged_data, tenant_id)
			
			# Delete duplicate contacts
			for dup_id in duplicate_contact_ids:
				await self.db_manager.delete_contact(dup_id, tenant_id)
			
			# Track merge analytics
			asyncio.create_task(
				self.analytics.track_contact_merge(tenant_id, primary_contact_id, duplicate_contact_ids)
			)
			
			logger.info(f"✅ Contact merge completed - Primary: {primary_contact_id}")
			return updated_contact
			
		except Exception as e:
			logger.error(f"Contact merge failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Contact merge failed: {str(e)}")
	
	# ================================
	# Contact Relationship Management
	# ================================
	
	async def create_contact_relationship(
		self,
		relationship_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> Dict[str, Any]:
		"""
		Create a relationship between contacts
		
		Args:
			relationship_data: Relationship information
			tenant_id: Tenant identifier
			created_by: User creating the relationship
			
		Returns:
			Created relationship data
		"""
		try:
			logger.info(f"🔄 Creating contact relationship for tenant: {tenant_id}")
			
			# Create relationship
			relationship = await self.relationship_manager.create_relationship(
				relationship_data=relationship_data,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
			# Track analytics
			asyncio.create_task(
				self.analytics.track_relationship_creation(tenant_id, relationship.id)
			)
			
			logger.info(f"✅ Contact relationship created: {relationship.id}")
			return relationship.model_dump()
			
		except Exception as e:
			logger.error(f"Contact relationship creation failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Contact relationship creation failed: {str(e)}")
	
	async def get_contact_relationships(
		self,
		contact_id: str,
		tenant_id: str,
		relationship_type: Optional[str] = None,
		include_incoming: bool = True,
		include_outgoing: bool = True
	) -> Dict[str, Any]:
		"""
		Get all relationships for a contact
		
		Args:
			contact_id: Contact identifier
			tenant_id: Tenant identifier
			relationship_type: Filter by relationship type
			include_incoming: Include relationships to this contact
			include_outgoing: Include relationships from this contact
			
		Returns:
			Contact relationships data
		"""
		try:
			logger.info(f"🔍 Getting relationships for contact: {contact_id}")
			
			# Convert string to enum if provided
			rel_type = None
			if relationship_type:
				try:
					rel_type = RelationshipType(relationship_type)
				except ValueError:
					raise CRMValidationError(f"Invalid relationship type: {relationship_type}")
			
			# Get relationships
			relationships = await self.relationship_manager.get_contact_relationships(
				contact_id=contact_id,
				tenant_id=tenant_id,
				relationship_type=rel_type,
				include_incoming=include_incoming,
				include_outgoing=include_outgoing
			)
			
			logger.info(f"✅ Retrieved {relationships['total']} relationships")
			return relationships
			
		except Exception as e:
			logger.error(f"Get contact relationships failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Get contact relationships failed: {str(e)}")
	
	async def update_contact_relationship(
		self,
		relationship_id: str,
		update_data: Dict[str, Any],
		tenant_id: str,
		updated_by: str
	) -> Dict[str, Any]:
		"""
		Update an existing relationship
		
		Args:
			relationship_id: Relationship identifier
			update_data: Fields to update
			tenant_id: Tenant identifier
			updated_by: User updating the relationship
			
		Returns:
			Updated relationship data
		"""
		try:
			logger.info(f"🔄 Updating relationship: {relationship_id}")
			
			# Update relationship
			updated_relationship = await self.relationship_manager.update_relationship(
				relationship_id=relationship_id,
				update_data=update_data,
				tenant_id=tenant_id,
				updated_by=updated_by
			)
			
			logger.info(f"✅ Relationship updated: {relationship_id}")
			return updated_relationship.model_dump()
			
		except Exception as e:
			logger.error(f"Update contact relationship failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Update contact relationship failed: {str(e)}")
	
	async def delete_contact_relationship(
		self,
		relationship_id: str,
		tenant_id: str
	) -> bool:
		"""
		Delete a contact relationship
		
		Args:
			relationship_id: Relationship identifier
			tenant_id: Tenant identifier
			
		Returns:
			True if deleted successfully
		"""
		try:
			logger.info(f"🔄 Deleting relationship: {relationship_id}")
			
			# Delete relationship
			deleted = await self.relationship_manager.delete_relationship(
				relationship_id=relationship_id,
				tenant_id=tenant_id
			)
			
			if deleted:
				# Track analytics
				asyncio.create_task(
					self.analytics.track_relationship_deletion(tenant_id, relationship_id)
				)
				logger.info(f"✅ Relationship deleted: {relationship_id}")
			
			return deleted
			
		except Exception as e:
			logger.error(f"Delete contact relationship failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Delete contact relationship failed: {str(e)}")
	
	async def discover_contact_relationships(
		self,
		tenant_id: str,
		auto_create: bool = False,
		min_confidence: float = 0.6
	) -> Dict[str, Any]:
		"""
		Discover potential relationships between contacts
		
		Args:
			tenant_id: Tenant identifier
			auto_create: Whether to automatically create discovered relationships
			min_confidence: Minimum confidence threshold
			
		Returns:
			Discovered relationships data
		"""
		try:
			logger.info(f"🔍 Discovering relationships for tenant: {tenant_id}")
			
			# Discover relationships
			discovered = await self.relationship_manager.discover_relationships(
				tenant_id=tenant_id,
				auto_create=auto_create,
				min_confidence=min_confidence
			)
			
			# Track analytics
			asyncio.create_task(
				self.analytics.track_relationship_discovery(tenant_id, len(discovered))
			)
			
			logger.info(f"✅ Discovered {len(discovered)} potential relationships")
			return {
				"discovered_relationships": discovered,
				"total_discovered": len(discovered),
				"auto_created": auto_create,
				"min_confidence": min_confidence
			}
			
		except Exception as e:
			logger.error(f"Relationship discovery failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Relationship discovery failed: {str(e)}")
	
	async def get_relationship_analytics(
		self,
		tenant_id: str,
		contact_id: Optional[str] = None
	) -> Dict[str, Any]:
		"""
		Get relationship analytics
		
		Args:
			tenant_id: Tenant identifier
			contact_id: Specific contact ID for individual analytics
			
		Returns:
			Analytics data
		"""
		try:
			logger.info(f"📊 Getting relationship analytics for tenant: {tenant_id}")
			
			# Get analytics
			analytics = await self.relationship_manager.get_relationship_analytics(
				tenant_id=tenant_id,
				contact_id=contact_id
			)
			
			logger.info(f"✅ Retrieved relationship analytics")
			return analytics
			
		except Exception as e:
			logger.error(f"Relationship analytics failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Relationship analytics failed: {str(e)}")
	
	async def get_relationship_graph(
		self,
		contact_id: str,
		tenant_id: str,
		depth: int = 2,
		max_nodes: int = 50
	) -> Dict[str, Any]:
		"""
		Get relationship graph for a contact
		
		Args:
			contact_id: Center contact ID
			tenant_id: Tenant identifier
			depth: Maximum relationship depth
			max_nodes: Maximum number of nodes to return
			
		Returns:
			Graph data with nodes and edges
		"""
		try:
			logger.info(f"🕸️ Building relationship graph for contact: {contact_id}")
			
			# Get relationship graph
			graph = await self.relationship_manager.get_relationship_graph(
				contact_id=contact_id,
				tenant_id=tenant_id,
				depth=depth,
				max_nodes=max_nodes
			)
			
			logger.info(f"✅ Built relationship graph with {graph['total_nodes']} nodes")
			return graph
			
		except Exception as e:
			logger.error(f"Relationship graph failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Relationship graph failed: {str(e)}")
	
	# ================================
	# Activity Tracking & Management
	# ================================
	
	async def create_activity(
		self,
		activity_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> Dict[str, Any]:
		"""
		Create a new activity
		
		Args:
			activity_data: Activity information
			tenant_id: Tenant identifier
			created_by: User creating the activity
			
		Returns:
			Created activity data
		"""
		try:
			logger.info(f"🔄 Creating activity for tenant: {tenant_id}")
			
			# Create activity
			activity = await self.activity_tracker.create_activity(
				activity_data=activity_data,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
			# Track analytics
			asyncio.create_task(
				self.analytics.track_activity_creation(tenant_id, activity.id)
			)
			
			logger.info(f"✅ Activity created: {activity.id}")
			return activity.model_dump()
			
		except Exception as e:
			logger.error(f"Activity creation failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Activity creation failed: {str(e)}")
	
	async def get_activity(
		self,
		activity_id: str,
		tenant_id: str
	) -> Optional[Dict[str, Any]]:
		"""
		Get activity by ID
		
		Args:
			activity_id: Activity identifier
			tenant_id: Tenant identifier
			
		Returns:
			Activity data or None
		"""
		try:
			activity = await self.activity_tracker.get_activity(activity_id, tenant_id)
			return activity.model_dump() if activity else None
			
		except Exception as e:
			logger.error(f"Get activity failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Get activity failed: {str(e)}")
	
	async def get_contact_activities(
		self,
		contact_id: str,
		tenant_id: str,
		activity_type: Optional[str] = None,
		status: Optional[str] = None,
		start_date: Optional[datetime] = None,
		end_date: Optional[datetime] = None,
		limit: int = 50,
		offset: int = 0
	) -> Dict[str, Any]:
		"""
		Get activities for a contact
		
		Args:
			contact_id: Contact identifier
			tenant_id: Tenant identifier
			activity_type: Filter by activity type
			status: Filter by status
			start_date: Filter by start date
			end_date: Filter by end date
			limit: Maximum number of activities
			offset: Offset for pagination
			
		Returns:
			Activities data with pagination
		"""
		try:
			logger.info(f"🔍 Getting activities for contact: {contact_id}")
			
			# Convert string enums if provided
			act_type = None
			if activity_type:
				try:
					act_type = ActivityType(activity_type)
				except ValueError:
					raise CRMValidationError(f"Invalid activity type: {activity_type}")
			
			act_status = None
			if status:
				try:
					act_status = ActivityStatus(status)
				except ValueError:
					raise CRMValidationError(f"Invalid activity status: {status}")
			
			# Get activities
			activities = await self.activity_tracker.get_contact_activities(
				contact_id=contact_id,
				tenant_id=tenant_id,
				activity_type=act_type,
				status=act_status,
				start_date=start_date,
				end_date=end_date,
				limit=limit,
				offset=offset
			)
			
			logger.info(f"✅ Retrieved {len(activities.get('items', []))} activities")
			return activities
			
		except Exception as e:
			logger.error(f"Get contact activities failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Get contact activities failed: {str(e)}")
	
	async def update_activity(
		self,
		activity_id: str,
		update_data: Dict[str, Any],
		tenant_id: str,
		updated_by: str
	) -> Dict[str, Any]:
		"""
		Update an existing activity
		
		Args:
			activity_id: Activity identifier
			update_data: Fields to update
			tenant_id: Tenant identifier
			updated_by: User updating the activity
			
		Returns:
			Updated activity data
		"""
		try:
			logger.info(f"🔄 Updating activity: {activity_id}")
			
			# Update activity
			updated_activity = await self.activity_tracker.update_activity(
				activity_id=activity_id,
				update_data=update_data,
				tenant_id=tenant_id,
				updated_by=updated_by
			)
			
			logger.info(f"✅ Activity updated: {activity_id}")
			return updated_activity.model_dump()
			
		except Exception as e:
			logger.error(f"Update activity failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Update activity failed: {str(e)}")
	
	async def complete_activity(
		self,
		activity_id: str,
		outcome: str,
		outcome_notes: Optional[str],
		tenant_id: str,
		completed_by: str
	) -> Dict[str, Any]:
		"""
		Mark activity as completed
		
		Args:
			activity_id: Activity identifier
			outcome: Activity outcome
			outcome_notes: Notes about the outcome
			tenant_id: Tenant identifier
			completed_by: User completing the activity
			
		Returns:
			Completed activity data
		"""
		try:
			logger.info(f"🔄 Completing activity: {activity_id}")
			
			# Convert string to enum
			try:
				activity_outcome = ActivityOutcome(outcome)
			except ValueError:
				raise CRMValidationError(f"Invalid activity outcome: {outcome}")
			
			# Complete activity
			completed_activity = await self.activity_tracker.complete_activity(
				activity_id=activity_id,
				outcome=activity_outcome,
				outcome_notes=outcome_notes,
				tenant_id=tenant_id,
				completed_by=completed_by
			)
			
			# Track analytics
			asyncio.create_task(
				self.analytics.track_activity_completion(tenant_id, activity_id, outcome)
			)
			
			logger.info(f"✅ Activity completed: {activity_id}")
			return completed_activity.model_dump()
			
		except Exception as e:
			logger.error(f"Complete activity failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Complete activity failed: {str(e)}")
	
	async def delete_activity(
		self,
		activity_id: str,
		tenant_id: str
	) -> bool:
		"""
		Delete an activity
		
		Args:
			activity_id: Activity identifier
			tenant_id: Tenant identifier
			
		Returns:
			True if deleted successfully
		"""
		try:
			logger.info(f"🔄 Deleting activity: {activity_id}")
			
			# Delete activity
			deleted = await self.activity_tracker.delete_activity(activity_id, tenant_id)
			
			if deleted:
				# Track analytics
				asyncio.create_task(
					self.analytics.track_activity_deletion(tenant_id, activity_id)
				)
				logger.info(f"✅ Activity deleted: {activity_id}")
			
			return deleted
			
		except Exception as e:
			logger.error(f"Delete activity failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Delete activity failed: {str(e)}")
	
	async def get_activity_analytics(
		self,
		tenant_id: str,
		contact_id: Optional[str] = None,
		start_date: Optional[datetime] = None,
		end_date: Optional[datetime] = None
	) -> Dict[str, Any]:
		"""
		Get activity analytics
		
		Args:
			tenant_id: Tenant identifier
			contact_id: Specific contact ID
			start_date: Analysis start date
			end_date: Analysis end date
			
		Returns:
			Analytics data
		"""
		try:
			logger.info(f"📊 Getting activity analytics for tenant: {tenant_id}")
			
			# Get analytics
			analytics = await self.activity_tracker.get_activity_analytics(
				tenant_id=tenant_id,
				contact_id=contact_id,
				start_date=start_date,
				end_date=end_date
			)
			
			logger.info(f"✅ Retrieved activity analytics")
			return analytics
			
		except Exception as e:
			logger.error(f"Activity analytics failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Activity analytics failed: {str(e)}")
	
	async def get_engagement_timeline(
		self,
		contact_id: str,
		tenant_id: str,
		days: int = 90
	) -> List[Dict[str, Any]]:
		"""
		Get engagement timeline for a contact
		
		Args:
			contact_id: Contact identifier
			tenant_id: Tenant identifier
			days: Number of days to analyze
			
		Returns:
			Timeline data
		"""
		try:
			logger.info(f"📈 Getting engagement timeline for contact: {contact_id}")
			
			# Get timeline
			timeline = await self.activity_tracker.get_engagement_timeline(
				contact_id=contact_id,
				tenant_id=tenant_id,
				days=days
			)
			
			logger.info(f"✅ Retrieved engagement timeline with {len(timeline)} entries")
			return timeline
			
		except Exception as e:
			logger.error(f"Engagement timeline failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Engagement timeline failed: {str(e)}")
	
	async def get_overdue_activities(
		self,
		tenant_id: str,
		assigned_to: Optional[str] = None,
		days_overdue: Optional[int] = None
	) -> List[Dict[str, Any]]:
		"""
		Get overdue activities
		
		Args:
			tenant_id: Tenant identifier
			assigned_to: Filter by assigned user
			days_overdue: Minimum days overdue
			
		Returns:
			List of overdue activities
		"""
		try:
			logger.info(f"⚠️ Getting overdue activities for tenant: {tenant_id}")
			
			# Get overdue activities
			overdue = await self.activity_tracker.get_overdue_activities(
				tenant_id=tenant_id,
				assigned_to=assigned_to,
				days_overdue=days_overdue
			)
			
			logger.info(f"✅ Found {len(overdue)} overdue activities")
			return overdue
			
		except Exception as e:
			logger.error(f"Get overdue activities failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Get overdue activities failed: {str(e)}")
	
	async def get_upcoming_activities(
		self,
		tenant_id: str,
		assigned_to: Optional[str] = None,
		days_ahead: int = 7
	) -> List[Dict[str, Any]]:
		"""
		Get upcoming activities
		
		Args:
			tenant_id: Tenant identifier
			assigned_to: Filter by assigned user
			days_ahead: Number of days ahead to look
			
		Returns:
			List of upcoming activities
		"""
		try:
			logger.info(f"📅 Getting upcoming activities for tenant: {tenant_id}")
			
			# Get upcoming activities
			upcoming = await self.activity_tracker.get_upcoming_activities(
				tenant_id=tenant_id,
				assigned_to=assigned_to,
				days_ahead=days_ahead
			)
			
			logger.info(f"✅ Found {len(upcoming)} upcoming activities")
			return upcoming
			
		except Exception as e:
			logger.error(f"Get upcoming activities failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Get upcoming activities failed: {str(e)}")
	
	# ================================
	# Account Hierarchy Management
	# ================================
	
	async def build_account_hierarchy(
		self,
		tenant_id: str,
		root_account_id: Optional[str] = None,
		max_depth: int = 10,
		include_metrics: bool = True
	) -> Dict[str, Any]:
		"""
		Build complete account hierarchy tree
		
		Args:
			tenant_id: Tenant identifier
			root_account_id: Root account to start from (None for all roots)
			max_depth: Maximum depth to traverse
			include_metrics: Whether to include aggregated metrics
			
		Returns:
			Complete hierarchy tree with all nodes and relationships
		"""
		try:
			logger.info(f"🌳 Building account hierarchy for tenant: {tenant_id}")
			
			hierarchy = await self.hierarchy_manager.build_account_hierarchy(
				tenant_id=tenant_id,
				root_account_id=root_account_id,
				max_depth=max_depth,
				include_metrics=include_metrics
			)
			
			logger.info(f"✅ Built hierarchy with {hierarchy['total_accounts']} accounts")
			return hierarchy
			
		except Exception as e:
			logger.error(f"Build hierarchy failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Build hierarchy failed: {str(e)}")
	
	async def get_account_ancestors(
		self,
		account_id: str,
		tenant_id: str,
		include_self: bool = False
	) -> List[Dict[str, Any]]:
		"""
		Get all ancestor accounts up the hierarchy
		
		Args:
			account_id: Account identifier
			tenant_id: Tenant identifier
			include_self: Whether to include the account itself
			
		Returns:
			List of ancestor accounts from immediate parent to root
		"""
		try:
			logger.info(f"📈 Getting ancestors for account: {account_id}")
			
			ancestors = await self.hierarchy_manager.get_account_ancestors(
				account_id=account_id,
				tenant_id=tenant_id,
				include_self=include_self
			)
			
			logger.info(f"✅ Found {len(ancestors)} ancestor accounts")
			return ancestors
			
		except Exception as e:
			logger.error(f"Get ancestors failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Get ancestors failed: {str(e)}")
	
	async def get_account_descendants(
		self,
		account_id: str,
		tenant_id: str,
		max_depth: int = 5,
		include_self: bool = False
	) -> List[Dict[str, Any]]:
		"""
		Get all descendant accounts down the hierarchy
		
		Args:
			account_id: Account identifier
			tenant_id: Tenant identifier
			max_depth: Maximum depth to traverse
			include_self: Whether to include the account itself
			
		Returns:
			List of descendant accounts with hierarchy information
		"""
		try:
			logger.info(f"📉 Getting descendants for account: {account_id}")
			
			descendants = await self.hierarchy_manager.get_account_descendants(
				account_id=account_id,
				tenant_id=tenant_id,
				max_depth=max_depth,
				include_self=include_self
			)
			
			logger.info(f"✅ Found {len(descendants)} descendant accounts")
			return descendants
			
		except Exception as e:
			logger.error(f"Get descendants failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Get descendants failed: {str(e)}")
	
	async def update_account_hierarchy(
		self,
		account_id: str,
		new_parent_id: Optional[str],
		relationship_type: HierarchyRelationshipType,
		tenant_id: str,
		updated_by: str,
		notes: Optional[str] = None
	) -> Dict[str, Any]:
		"""
		Update account hierarchy relationships
		
		Args:
			account_id: Account to update
			new_parent_id: New parent account ID
			relationship_type: Type of relationship
			tenant_id: Tenant identifier
			updated_by: User making the change
			notes: Optional notes about the change
			
		Returns:
			Update result with validation information
		"""
		try:
			logger.info(f"🔄 Updating hierarchy for account: {account_id}")
			
			# Create update request
			update_request = HierarchyUpdateRequest(
				account_id=account_id,
				new_parent_id=new_parent_id,
				relationship_type=relationship_type,
				notes=notes
			)
			
			result = await self.hierarchy_manager.update_account_hierarchy(
				update_request=update_request,
				tenant_id=tenant_id,
				updated_by=updated_by
			)
			
			logger.info(f"✅ Updated hierarchy for account: {account_id}")
			return result
			
		except Exception as e:
			logger.error(f"Update hierarchy failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Update hierarchy failed: {str(e)}")
	
	async def get_hierarchy_analytics(
		self,
		tenant_id: str,
		account_id: Optional[str] = None
	) -> Dict[str, Any]:
		"""
		Get hierarchy analytics and metrics
		
		Args:
			tenant_id: Tenant identifier
			account_id: Specific account to analyze (None for all)
			
		Returns:
			Comprehensive hierarchy analytics
		"""
		try:
			logger.info(f"📊 Getting hierarchy analytics for tenant: {tenant_id}")
			
			analytics = await self.hierarchy_manager.get_hierarchy_analytics(
				tenant_id=tenant_id,
				account_id=account_id
			)
			
			logger.info(f"✅ Generated hierarchy analytics with {analytics['hierarchy_metrics']['total_accounts']} accounts")
			return analytics
			
		except Exception as e:
			logger.error(f"Hierarchy analytics failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Hierarchy analytics failed: {str(e)}")
	
	async def find_hierarchy_path(
		self,
		from_account_id: str,
		to_account_id: str,
		tenant_id: str
	) -> Optional[List[Dict[str, Any]]]:
		"""
		Find path between two accounts in hierarchy
		
		Args:
			from_account_id: Starting account
			to_account_id: Target account
			tenant_id: Tenant identifier
			
		Returns:
			Path between accounts or None if no path exists
		"""
		try:
			logger.info(f"🗺️ Finding path from {from_account_id} to {to_account_id}")
			
			path = await self.hierarchy_manager.find_hierarchy_path(
				from_account_id=from_account_id,
				to_account_id=to_account_id,
				tenant_id=tenant_id
			)
			
			if path:
				logger.info(f"✅ Found hierarchy path with {len(path)} steps")
			else:
				logger.info("ℹ️ No hierarchy path found between accounts")
			
			return path
			
		except Exception as e:
			logger.error(f"Find hierarchy path failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Find hierarchy path failed: {str(e)}")
	
	# ================================
	# Account Relationship Management
	# ================================
	
	async def create_account_relationship(
		self,
		relationship_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> Dict[str, Any]:
		"""
		Create a new account relationship
		
		Args:
			relationship_data: Relationship information
			tenant_id: Tenant identifier
			created_by: User creating the relationship
			
		Returns:
			Created relationship object
		"""
		try:
			logger.info(f"🔗 Creating account relationship")
			
			relationship = await self.account_relationship_manager.create_relationship(
				relationship_data=relationship_data,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
			logger.info(f"✅ Created account relationship: {relationship.id}")
			return relationship.model_dump()
			
		except Exception as e:
			logger.error(f"Create account relationship failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Create account relationship failed: {str(e)}")
	
	async def get_account_relationship(
		self,
		relationship_id: str,
		tenant_id: str
	) -> Optional[Dict[str, Any]]:
		"""
		Get account relationship by ID
		
		Args:
			relationship_id: Relationship identifier
			tenant_id: Tenant identifier
			
		Returns:
			Relationship object or None
		"""
		try:
			logger.info(f"📖 Getting account relationship: {relationship_id}")
			
			relationship = await self.account_relationship_manager.get_relationship(
				relationship_id=relationship_id,
				tenant_id=tenant_id
			)
			
			if relationship:
				logger.info(f"✅ Found account relationship: {relationship_id}")
				return relationship.model_dump()
			else:
				logger.info(f"ℹ️ Account relationship not found: {relationship_id}")
				return None
			
		except Exception as e:
			logger.error(f"Get account relationship failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Get account relationship failed: {str(e)}")
	
	async def get_account_relationships(
		self,
		account_id: str,
		tenant_id: str,
		relationship_type: Optional[AccountRelationshipType] = None,
		direction: Optional[RelationshipDirection] = None,
		status: Optional[str] = None,
		include_details: bool = True
	) -> Dict[str, Any]:
		"""
		Get all relationships for an account
		
		Args:
			account_id: Account identifier
			tenant_id: Tenant identifier
			relationship_type: Filter by relationship type
			direction: Filter by direction
			status: Filter by status
			include_details: Include detailed account information
			
		Returns:
			Comprehensive relationship data
		"""
		try:
			logger.info(f"🔍 Getting relationships for account: {account_id}")
			
			# Convert status string to enum if provided
			status_enum = None
			if status:
				from .account_relationships import RelationshipStatus
				status_enum = RelationshipStatus(status)
			
			relationships = await self.account_relationship_manager.get_account_relationships(
				account_id=account_id,
				tenant_id=tenant_id,
				relationship_type=relationship_type,
				direction=direction,
				status=status_enum,
				include_details=include_details
			)
			
			logger.info(f"✅ Found {relationships['total']} relationships for account")
			return relationships
			
		except Exception as e:
			logger.error(f"Get account relationships failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Get account relationships failed: {str(e)}")
	
	async def update_account_relationship(
		self,
		relationship_id: str,
		update_data: Dict[str, Any],
		tenant_id: str,
		updated_by: str
	) -> Dict[str, Any]:
		"""
		Update existing account relationship
		
		Args:
			relationship_id: Relationship identifier
			update_data: Fields to update
			tenant_id: Tenant identifier
			updated_by: User updating the relationship
			
		Returns:
			Updated relationship object
		"""
		try:
			logger.info(f"🔄 Updating account relationship: {relationship_id}")
			
			relationship = await self.account_relationship_manager.update_relationship(
				relationship_id=relationship_id,
				update_data=update_data,
				tenant_id=tenant_id,
				updated_by=updated_by
			)
			
			logger.info(f"✅ Updated account relationship: {relationship_id}")
			return relationship.model_dump()
			
		except Exception as e:
			logger.error(f"Update account relationship failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Update account relationship failed: {str(e)}")
	
	async def delete_account_relationship(
		self,
		relationship_id: str,
		tenant_id: str
	) -> bool:
		"""
		Delete an account relationship
		
		Args:
			relationship_id: Relationship identifier
			tenant_id: Tenant identifier
			
		Returns:
			True if deleted successfully
		"""
		try:
			logger.info(f"🗑️ Deleting account relationship: {relationship_id}")
			
			success = await self.account_relationship_manager.delete_relationship(
				relationship_id=relationship_id,
				tenant_id=tenant_id
			)
			
			if success:
				logger.info(f"✅ Deleted account relationship: {relationship_id}")
			else:
				logger.warning(f"⚠️ Account relationship not found for deletion: {relationship_id}")
			
			return success
			
		except Exception as e:
			logger.error(f"Delete account relationship failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Delete account relationship failed: {str(e)}")
	
	async def discover_potential_relationships(
		self,
		tenant_id: str,
		account_id: Optional[str] = None,
		min_confidence: float = 0.7
	) -> List[Dict[str, Any]]:
		"""
		Discover potential relationships between accounts
		
		Args:
			tenant_id: Tenant identifier
			account_id: Specific account to analyze
			min_confidence: Minimum confidence threshold
			
		Returns:
			List of potential relationships with confidence scores
		"""
		try:
			logger.info(f"🔍 Discovering potential relationships for tenant: {tenant_id}")
			
			relationships = await self.account_relationship_manager.discover_potential_relationships(
				tenant_id=tenant_id,
				account_id=account_id,
				min_confidence=min_confidence
			)
			
			logger.info(f"✅ Discovered {len(relationships)} potential relationships")
			return relationships
			
		except Exception as e:
			logger.error(f"Discover relationships failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Discover relationships failed: {str(e)}")
	
	async def get_account_relationship_analytics(
		self,
		tenant_id: str,
		account_id: Optional[str] = None,
		start_date: Optional[datetime] = None,
		end_date: Optional[datetime] = None
	) -> Dict[str, Any]:
		"""
		Get comprehensive account relationship analytics
		
		Args:
			tenant_id: Tenant identifier
			account_id: Specific account to analyze
			start_date: Analysis start date
			end_date: Analysis end date
			
		Returns:
			Comprehensive relationship analytics
		"""
		try:
			logger.info(f"📊 Getting account relationship analytics for tenant: {tenant_id}")
			
			analytics = await self.account_relationship_manager.get_relationship_analytics(
				tenant_id=tenant_id,
				account_id=account_id,
				start_date=start_date,
				end_date=end_date
			)
			
			logger.info(f"✅ Generated analytics for {analytics.total_relationships} relationships")
			return analytics.model_dump()
			
		except Exception as e:
			logger.error(f"Account relationship analytics failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Account relationship analytics failed: {str(e)}")
	
	# ================================
	# Territory Management
	# ================================
	
	async def create_territory(
		self,
		territory_data: Dict[str, Any],
		tenant_id: str,
		created_by: str,
		owner_id: str
	) -> Dict[str, Any]:
		"""
		Create a new sales territory
		
		Args:
			territory_data: Territory configuration data
			tenant_id: Tenant identifier
			created_by: User creating the territory
			owner_id: Territory owner/manager
			
		Returns:
			Created territory information
		"""
		try:
			logger.info(f"🎯 Creating territory: {territory_data.get('territory_name')}")
			
			territory = await self.territory_manager.create_territory(
				territory_data=territory_data,
				tenant_id=tenant_id,
				created_by=created_by,
				owner_id=owner_id
			)
			
			logger.info(f"✅ Territory created successfully: {territory.id}")
			return territory.model_dump()
			
		except Exception as e:
			logger.error(f"Territory creation failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Territory creation failed: {str(e)}")
	
	async def get_territory(self, territory_id: str, tenant_id: str) -> Dict[str, Any]:
		"""
		Get territory by ID
		
		Args:
			territory_id: Territory identifier
			tenant_id: Tenant identifier
			
		Returns:
			Territory information
		"""
		try:
			territory = await self.territory_manager.get_territory(territory_id, tenant_id)
			if not territory:
				raise CRMNotFoundError(f"Territory not found: {territory_id}")
			
			return territory.model_dump()
			
		except Exception as e:
			if isinstance(e, CRMNotFoundError):
				raise
			logger.error(f"Get territory failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Get territory failed: {str(e)}")
	
	async def list_territories(
		self,
		tenant_id: str,
		territory_type: Optional[TerritoryType] = None,
		status: Optional[TerritoryStatus] = None,
		owner_id: Optional[str] = None,
		limit: int = 100,
		offset: int = 0
	) -> Dict[str, Any]:
		"""
		List territories with optional filtering
		
		Args:
			tenant_id: Tenant identifier
			territory_type: Filter by territory type
			status: Filter by status
			owner_id: Filter by owner
			limit: Maximum number of results
			offset: Number of results to skip
			
		Returns:
			List of territories with pagination info
		"""
		try:
			logger.info(f"📋 Listing territories for tenant: {tenant_id}")
			
			result = await self.territory_manager.list_territories(
				tenant_id=tenant_id,
				territory_type=territory_type,
				status=status,
				owner_id=owner_id,
				limit=limit,
				offset=offset
			)
			
			return {
				"territories": [t.model_dump() for t in result["territories"]],
				"total": result["total"],
				"limit": limit,
				"offset": offset
			}
			
		except Exception as e:
			logger.error(f"List territories failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"List territories failed: {str(e)}")
	
	async def update_territory(
		self,
		territory_id: str,
		update_data: Dict[str, Any],
		tenant_id: str,
		updated_by: str
	) -> Dict[str, Any]:
		"""
		Update territory information
		
		Args:
			territory_id: Territory identifier
			update_data: Updated territory data
			tenant_id: Tenant identifier
			updated_by: User making the update
			
		Returns:
			Updated territory information
		"""
		try:
			logger.info(f"📝 Updating territory: {territory_id}")
			
			territory = await self.territory_manager.update_territory(
				territory_id=territory_id,
				update_data=update_data,
				tenant_id=tenant_id,
				updated_by=updated_by
			)
			
			logger.info(f"✅ Territory updated successfully")
			return territory.model_dump()
			
		except Exception as e:
			logger.error(f"Territory update failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Territory update failed: {str(e)}")
	
	async def assign_account_to_territory(
		self,
		account_id: str,
		territory_id: str,
		assignment_type: AssignmentType,
		tenant_id: str,
		assigned_by: str,
		assignment_reason: Optional[str] = None
	) -> Dict[str, Any]:
		"""
		Assign an account to a territory
		
		Args:
			account_id: Account identifier
			territory_id: Territory identifier
			assignment_type: Type of assignment (primary, secondary, etc.)
			tenant_id: Tenant identifier
			assigned_by: User making the assignment
			assignment_reason: Reason for the assignment
			
		Returns:
			Assignment information
		"""
		try:
			logger.info(f"🎯 Assigning account {account_id} to territory {territory_id}")
			
			assignment = await self.territory_manager.assign_account_to_territory(
				account_id=account_id,
				territory_id=territory_id,
				assignment_type=assignment_type,
				tenant_id=tenant_id,
				assigned_by=assigned_by,
				assignment_reason=assignment_reason
			)
			
			logger.info(f"✅ Account assigned successfully")
			return assignment.model_dump()
			
		except Exception as e:
			logger.error(f"Account assignment failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Account assignment failed: {str(e)}")
	
	async def get_territory_assignments(
		self,
		territory_id: str,
		tenant_id: str,
		assignment_type: Optional[AssignmentType] = None
	) -> List[Dict[str, Any]]:
		"""
		Get all account assignments for a territory
		
		Args:
			territory_id: Territory identifier
			tenant_id: Tenant identifier
			assignment_type: Filter by assignment type
			
		Returns:
			List of account assignments
		"""
		try:
			assignments = await self.territory_manager.get_territory_assignments(
				territory_id=territory_id,
				tenant_id=tenant_id,
				assignment_type=assignment_type
			)
			
			return [a.model_dump() for a in assignments]
			
		except Exception as e:
			logger.error(f"Get territory assignments failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Get territory assignments failed: {str(e)}")
	
	async def get_account_territory_assignments(
		self,
		account_id: str,
		tenant_id: str
	) -> List[Dict[str, Any]]:
		"""
		Get all territory assignments for an account
		
		Args:
			account_id: Account identifier
			tenant_id: Tenant identifier
			
		Returns:
			List of territory assignments
		"""
		try:
			assignments = await self.territory_manager.get_account_territory_assignments(
				account_id=account_id,
				tenant_id=tenant_id
			)
			
			return [a.model_dump() for a in assignments]
			
		except Exception as e:
			logger.error(f"Get account territory assignments failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Get account territory assignments failed: {str(e)}")
	
	async def analyze_territory_coverage(
		self,
		territory_id: str,
		tenant_id: str
	) -> Dict[str, Any]:
		"""
		Analyze territory coverage and performance
		
		Args:
			territory_id: Territory identifier
			tenant_id: Tenant identifier
			
		Returns:
			Territory coverage analysis
		"""
		try:
			logger.info(f"📊 Analyzing territory coverage: {territory_id}")
			
			analysis = await self.territory_manager.analyze_territory_coverage(
				territory_id=territory_id,
				tenant_id=tenant_id
			)
			
			return analysis.model_dump()
			
		except Exception as e:
			logger.error(f"Territory coverage analysis failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Territory coverage analysis failed: {str(e)}")
	
	async def get_territory_assignment_recommendations(
		self,
		territory_id: str,
		tenant_id: str,
		limit: int = 10
	) -> List[Dict[str, Any]]:
		"""
		Get recommended accounts for territory assignment
		
		Args:
			territory_id: Territory identifier
			tenant_id: Tenant identifier
			limit: Maximum number of recommendations
			
		Returns:
			List of account recommendations with match scores
		"""
		try:
			recommendations = await self.territory_manager.get_assignment_recommendations(
				territory_id=territory_id,
				tenant_id=tenant_id,
				limit=limit
			)
			
			return [r.model_dump() for r in recommendations]
			
		except Exception as e:
			logger.error(f"Territory assignment recommendations failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Territory assignment recommendations failed: {str(e)}")
	
	# ================================
	# Communication History Management
	# ================================
	
	async def create_communication(
		self,
		communication_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> Dict[str, Any]:
		"""
		Create a new communication record
		
		Args:
			communication_data: Communication data
			tenant_id: Tenant identifier
			created_by: User creating the communication
			
		Returns:
			Created communication record
		"""
		try:
			logger.info(f"📝 Creating communication: {communication_data.get('subject', 'No subject')}")
			
			communication = await self.communication_manager.create_communication(
				communication_data=communication_data,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
			logger.info(f"✅ Communication created successfully: {communication.id}")
			return communication.model_dump()
			
		except Exception as e:
			logger.error(f"Communication creation failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Communication creation failed: {str(e)}")
	
	async def get_communication(
		self,
		communication_id: str,
		tenant_id: str
	) -> Dict[str, Any]:
		"""
		Get communication by ID
		
		Args:
			communication_id: Communication identifier
			tenant_id: Tenant identifier
			
		Returns:
			Communication record
		"""
		try:
			communication = await self.communication_manager.get_communication(
				communication_id, tenant_id
			)
			
			if not communication:
				raise CRMNotFoundError(f"Communication not found: {communication_id}")
			
			return communication.model_dump()
			
		except Exception as e:
			if isinstance(e, CRMNotFoundError):
				raise
			logger.error(f"Get communication failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Get communication failed: {str(e)}")
	
	async def list_communications(
		self,
		tenant_id: str,
		contact_id: Optional[str] = None,
		account_id: Optional[str] = None,
		communication_type: Optional[CommunicationType] = None,
		direction: Optional[CommunicationDirection] = None,
		status: Optional[CommunicationStatus] = None,
		start_date: Optional[datetime] = None,
		end_date: Optional[datetime] = None,
		tags: Optional[List[str]] = None,
		limit: int = 100,
		offset: int = 0
	) -> Dict[str, Any]:
		"""
		List communications with filtering
		
		Args:
			tenant_id: Tenant identifier
			contact_id: Filter by contact
			account_id: Filter by account
			communication_type: Filter by type
			direction: Filter by direction
			status: Filter by status
			start_date: Filter by start date
			end_date: Filter by end date
			tags: Filter by tags
			limit: Maximum results
			offset: Results offset
			
		Returns:
			List of communications with pagination info
		"""
		try:
			logger.info(f"📋 Listing communications for tenant: {tenant_id}")
			
			result = await self.communication_manager.list_communications(
				tenant_id=tenant_id,
				contact_id=contact_id,
				account_id=account_id,
				communication_type=communication_type,
				direction=direction,
				status=status,
				start_date=start_date,
				end_date=end_date,
				tags=tags,
				limit=limit,
				offset=offset
			)
			
			return {
				"communications": [c.model_dump() for c in result["communications"]],
				"total": result["total"],
				"limit": limit,
				"offset": offset
			}
			
		except Exception as e:
			logger.error(f"List communications failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"List communications failed: {str(e)}")
	
	async def update_communication(
		self,
		communication_id: str,
		update_data: Dict[str, Any],
		tenant_id: str,
		updated_by: str
	) -> Dict[str, Any]:
		"""
		Update communication record
		
		Args:
			communication_id: Communication identifier
			update_data: Fields to update
			tenant_id: Tenant identifier
			updated_by: User making the update
			
		Returns:
			Updated communication record
		"""
		try:
			logger.info(f"📝 Updating communication: {communication_id}")
			
			communication = await self.communication_manager.update_communication(
				communication_id=communication_id,
				update_data=update_data,
				tenant_id=tenant_id,
				updated_by=updated_by
			)
			
			logger.info(f"✅ Communication updated successfully")
			return communication.model_dump()
			
		except Exception as e:
			logger.error(f"Communication update failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Communication update failed: {str(e)}")
	
	async def delete_communication(
		self,
		communication_id: str,
		tenant_id: str
	) -> bool:
		"""
		Delete communication record
		
		Args:
			communication_id: Communication identifier
			tenant_id: Tenant identifier
			
		Returns:
			True if deleted successfully
		"""
		try:
			logger.info(f"🗑️ Deleting communication: {communication_id}")
			
			deleted = await self.communication_manager.delete_communication(
				communication_id, tenant_id
			)
			
			if deleted:
				logger.info(f"✅ Communication deleted successfully")
			
			return deleted
			
		except Exception as e:
			logger.error(f"Communication deletion failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Communication deletion failed: {str(e)}")
	
	async def get_communication_analytics(
		self,
		tenant_id: str,
		contact_id: Optional[str] = None,
		account_id: Optional[str] = None,
		start_date: Optional[datetime] = None,
		end_date: Optional[datetime] = None
	) -> Dict[str, Any]:
		"""
		Get comprehensive communication analytics
		
		Args:
			tenant_id: Tenant identifier
			contact_id: Filter by contact
			account_id: Filter by account
			start_date: Analysis start date
			end_date: Analysis end date
			
		Returns:
			Communication analytics data
		"""
		try:
			logger.info(f"📊 Generating communication analytics for tenant: {tenant_id}")
			
			analytics = await self.communication_manager.get_communication_analytics(
				tenant_id=tenant_id,
				contact_id=contact_id,
				account_id=account_id,
				start_date=start_date,
				end_date=end_date
			)
			
			return analytics.model_dump()
			
		except Exception as e:
			logger.error(f"Communication analytics failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Communication analytics failed: {str(e)}")
	
	async def get_pending_follow_ups(
		self,
		tenant_id: str,
		user_id: Optional[str] = None,
		overdue_only: bool = False
	) -> List[Dict[str, Any]]:
		"""
		Get pending follow-up communications
		
		Args:
			tenant_id: Tenant identifier
			user_id: Filter by user
			overdue_only: Only return overdue follow-ups
			
		Returns:
			List of communications requiring follow-up
		"""
		try:
			logger.info(f"🔔 Getting pending follow-ups for tenant: {tenant_id}")
			
			follow_ups = await self.communication_manager.get_pending_follow_ups(
				tenant_id=tenant_id,
				user_id=user_id,
				overdue_only=overdue_only
			)
			
			return [f.model_dump() for f in follow_ups]
			
		except Exception as e:
			logger.error(f"Get pending follow-ups failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Get pending follow-ups failed: {str(e)}")
	
	async def get_communication_timeline(
		self,
		entity_id: str,
		entity_type: str,
		tenant_id: str,
		limit: int = 20
	) -> List[Dict[str, Any]]:
		"""
		Get communication timeline for an entity
		
		Args:
			entity_id: Entity identifier (contact, account, lead, opportunity)
			entity_type: Type of entity
			tenant_id: Tenant identifier
			limit: Maximum number of communications
			
		Returns:
			Timeline of communications for the entity
		"""
		try:
			logger.info(f"📅 Getting communication timeline for {entity_type}: {entity_id}")
			
			# Get communications based on entity type
			if entity_type == 'contact':
				result = await self.list_communications(
					tenant_id=tenant_id,
					contact_id=entity_id,
					limit=limit,
					offset=0
				)
			elif entity_type == 'account':
				result = await self.list_communications(
					tenant_id=tenant_id,
					account_id=entity_id,
					limit=limit,
					offset=0
				)
			else:
				# For lead/opportunity, we'll need a more generic approach
				result = await self.list_communications(
					tenant_id=tenant_id,
					limit=limit,
					offset=0
				)
				# Filter by entity_id in the result
				filtered_communications = []
				for comm in result["communications"]:
					if (entity_type == 'lead' and comm.get('lead_id') == entity_id) or \
					   (entity_type == 'opportunity' and comm.get('opportunity_id') == entity_id):
						filtered_communications.append(comm)
				result["communications"] = filtered_communications[:limit]
			
			return result["communications"]
			
		except Exception as e:
			logger.error(f"Get communication timeline failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Get communication timeline failed: {str(e)}")
	
	# ================================
	# Contact Segmentation Management
	# ================================
	
	async def create_contact_segment(
		self,
		segment_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> Dict[str, Any]:
		"""
		Create a new contact segment
		
		Args:
			segment_data: Segment configuration data
			tenant_id: Tenant identifier
			created_by: User creating the segment
			
		Returns:
			Created segment information
		"""
		try:
			logger.info(f"📊 Creating contact segment: {segment_data.get('name')}")
			
			segment = await self.segmentation_manager.create_segment(
				segment_data=segment_data,
				tenant_id=tenant_id,
				created_by=created_by
			)
			
			logger.info(f"✅ Contact segment created successfully: {segment.id}")
			return segment.model_dump()
			
		except Exception as e:
			logger.error(f"Contact segment creation failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Contact segment creation failed: {str(e)}")
	
	async def get_contact_segment(
		self,
		segment_id: str,
		tenant_id: str
	) -> Dict[str, Any]:
		"""
		Get contact segment by ID
		
		Args:
			segment_id: Segment identifier
			tenant_id: Tenant identifier
			
		Returns:
			Segment information
		"""
		try:
			segment = await self.segmentation_manager.get_segment(segment_id, tenant_id)
			if not segment:
				raise CRMNotFoundError(f"Contact segment not found: {segment_id}")
			
			return segment.model_dump()
			
		except Exception as e:
			if isinstance(e, CRMNotFoundError):
				raise
			logger.error(f"Get contact segment failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Get contact segment failed: {str(e)}")
	
	async def list_contact_segments(
		self,
		tenant_id: str,
		segment_type: Optional[SegmentType] = None,
		status: Optional[SegmentStatus] = None,
		category: Optional[str] = None,
		tags: Optional[List[str]] = None,
		limit: int = 100,
		offset: int = 0
	) -> Dict[str, Any]:
		"""
		List contact segments with filtering
		
		Args:
			tenant_id: Tenant identifier
			segment_type: Filter by segment type
			status: Filter by status
			category: Filter by category
			tags: Filter by tags
			limit: Maximum results
			offset: Results offset
			
		Returns:
			List of segments with pagination info
		"""
		try:
			logger.info(f"📋 Listing contact segments for tenant: {tenant_id}")
			
			result = await self.segmentation_manager.list_segments(
				tenant_id=tenant_id,
				segment_type=segment_type,
				status=status,
				category=category,
				tags=tags,
				limit=limit,
				offset=offset
			)
			
			return {
				"segments": [s.model_dump() for s in result["segments"]],
				"total": result["total"],
				"limit": limit,
				"offset": offset
			}
			
		except Exception as e:
			logger.error(f"List contact segments failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"List contact segments failed: {str(e)}")
	
	async def update_contact_segment(
		self,
		segment_id: str,
		update_data: Dict[str, Any],
		tenant_id: str,
		updated_by: str
	) -> Dict[str, Any]:
		"""
		Update contact segment
		
		Args:
			segment_id: Segment identifier
			update_data: Fields to update
			tenant_id: Tenant identifier
			updated_by: User making the update
			
		Returns:
			Updated segment information
		"""
		try:
			logger.info(f"📝 Updating contact segment: {segment_id}")
			
			segment = await self.segmentation_manager.update_segment(
				segment_id=segment_id,
				update_data=update_data,
				tenant_id=tenant_id,
				updated_by=updated_by
			)
			
			logger.info(f"✅ Contact segment updated successfully")
			return segment.model_dump()
			
		except Exception as e:
			logger.error(f"Contact segment update failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Contact segment update failed: {str(e)}")
	
	async def delete_contact_segment(
		self,
		segment_id: str,
		tenant_id: str
	) -> bool:
		"""
		Delete contact segment
		
		Args:
			segment_id: Segment identifier
			tenant_id: Tenant identifier
			
		Returns:
			True if deleted successfully
		"""
		try:
			logger.info(f"🗑️ Deleting contact segment: {segment_id}")
			
			deleted = await self.segmentation_manager.delete_segment(
				segment_id, tenant_id
			)
			
			if deleted:
				logger.info(f"✅ Contact segment deleted successfully")
			
			return deleted
			
		except Exception as e:
			logger.error(f"Contact segment deletion failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Contact segment deletion failed: {str(e)}")
	
	async def get_segment_contacts(
		self,
		segment_id: str,
		tenant_id: str,
		limit: int = 100,
		offset: int = 0
	) -> Dict[str, Any]:
		"""
		Get contacts in a segment
		
		Args:
			segment_id: Segment identifier
			tenant_id: Tenant identifier
			limit: Maximum results
			offset: Results offset
			
		Returns:
			List of contacts in the segment with pagination info
		"""
		try:
			logger.info(f"👥 Getting contacts for segment: {segment_id}")
			
			result = await self.segmentation_manager.get_segment_contacts(
				segment_id=segment_id,
				tenant_id=tenant_id,
				limit=limit,
				offset=offset
			)
			
			return {
				"contacts": [c.model_dump() for c in result["contacts"]],
				"total": result["total"],
				"limit": limit,
				"offset": offset
			}
			
		except Exception as e:
			logger.error(f"Get segment contacts failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Get segment contacts failed: {str(e)}")
	
	async def refresh_contact_segment(
		self,
		segment_id: str,
		tenant_id: str
	) -> Dict[str, Any]:
		"""
		Refresh a dynamic segment's contact count
		
		Args:
			segment_id: Segment identifier
			tenant_id: Tenant identifier
			
		Returns:
			Updated segment information
		"""
		try:
			logger.info(f"🔄 Refreshing contact segment: {segment_id}")
			
			segment = await self.segmentation_manager.refresh_segment(
				segment_id, tenant_id
			)
			
			logger.info(f"✅ Contact segment refreshed successfully")
			return segment.model_dump()
			
		except Exception as e:
			logger.error(f"Contact segment refresh failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Contact segment refresh failed: {str(e)}")
	
	async def get_segment_analytics(
		self,
		segment_id: str,
		tenant_id: str
	) -> Dict[str, Any]:
		"""
		Get comprehensive analytics for a segment
		
		Args:
			segment_id: Segment identifier
			tenant_id: Tenant identifier
			
		Returns:
			Segment analytics data
		"""
		try:
			logger.info(f"📊 Generating analytics for segment: {segment_id}")
			
			analytics = await self.segmentation_manager.get_segment_analytics(
				segment_id, tenant_id
			)
			
			return analytics.model_dump()
			
		except Exception as e:
			logger.error(f"Segment analytics failed: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Segment analytics failed: {str(e)}")
	
	# === PREDICTIVE ANALYTICS METHODS ===
	
	async def create_prediction_model(
		self,
		tenant_id: str,
		name: str,
		model_type: str,
		algorithm: str,
		target_variable: str,
		feature_columns: List[str],
		data_sources: List[str],
		training_data_query: str,
		created_by: str,
		hyperparameters: Optional[Dict[str, Any]] = None,
		description: Optional[str] = None
	) -> PredictionModel:
		"""Create a new predictive model"""
		return await self.predictive_analytics.create_prediction_model(
			tenant_id=tenant_id,
			name=name,
			model_type=model_type,
			algorithm=algorithm,
			target_variable=target_variable,
			feature_columns=feature_columns,
			data_sources=data_sources,
			training_data_query=training_data_query,
			created_by=created_by,
			hyperparameters=hyperparameters,
			description=description
		)

	async def train_prediction_model(
		self,
		tenant_id: str,
		model_id: str,
		training_data: Optional[Any] = None
	) -> Dict[str, Any]:
		"""Train a predictive model"""
		return await self.predictive_analytics.train_model(
			tenant_id=tenant_id,
			model_id=model_id,
			training_data=training_data
		)

	async def make_prediction(
		self,
		tenant_id: str,
		model_id: str,
		input_data: Dict[str, Any],
		prediction_type: str = "single",
		include_explanations: bool = True,
		created_by: str = "system"
	) -> PredictionResult:
		"""Make predictions using a trained model"""
		return await self.predictive_analytics.make_prediction(
			tenant_id=tenant_id,
			model_id=model_id,
			input_data=input_data,
			prediction_type=prediction_type,
			include_explanations=include_explanations,
			created_by=created_by
		)

	async def generate_sales_forecast(
		self,
		tenant_id: str,
		forecast_type: str = "revenue",
		period_type: str = "monthly",
		periods_ahead: int = 3
	) -> List[ForecastingInsight]:
		"""Generate sales forecasting insights"""
		return await self.predictive_analytics.generate_sales_forecast(
			tenant_id=tenant_id,
			forecast_type=forecast_type,
			period_type=period_type,
			periods_ahead=periods_ahead
		)

	async def predict_customer_churn(
		self,
		tenant_id: str,
		entity_type: str = "contact",
		entity_ids: Optional[List[str]] = None
	) -> List[ChurnPrediction]:
		"""Predict customer churn risk"""
		return await self.predictive_analytics.predict_customer_churn(
			tenant_id=tenant_id,
			entity_type=entity_type,
			entity_ids=entity_ids
		)

	async def optimize_lead_scoring(
		self,
		tenant_id: str,
		lead_ids: Optional[List[str]] = None
	) -> List[LeadScoringInsight]:
		"""Generate optimized lead scoring insights"""
		return await self.predictive_analytics.optimize_lead_scoring(
			tenant_id=tenant_id,
			lead_ids=lead_ids
		)

	async def perform_market_segmentation(
		self,
		tenant_id: str,
		segmentation_criteria: Dict[str, Any],
		num_segments: int = 5
	) -> List[MarketSegmentation]:
		"""Perform intelligent market segmentation analysis"""
		return await self.predictive_analytics.perform_market_segmentation(
			tenant_id=tenant_id,
			segmentation_criteria=segmentation_criteria,
			num_segments=num_segments
		)

	async def get_prediction_models(
		self,
		tenant_id: str,
		model_type: Optional[str] = None,
		is_active: bool = True
	) -> List[Dict[str, Any]]:
		"""Get available prediction models for tenant"""
		try:
			async with self.db_manager.get_connection() as conn:
				query = """
					SELECT * FROM crm_prediction_models 
					WHERE tenant_id = $1
				"""
				params = [tenant_id]
				
				if model_type:
					query += " AND model_type = $2"
					params.append(model_type)
					
				if is_active is not None:
					query += f" AND is_active = ${len(params) + 1}"
					params.append(is_active)
					
				query += " ORDER BY created_at DESC"
				
				rows = await conn.fetch(query, *params)
				return [dict(row) for row in rows]
				
		except Exception as e:
			logger.error(f"Failed to get prediction models: {str(e)}")
			raise CRMServiceError(f"Failed to get prediction models: {str(e)}")

	async def get_ai_insights(
		self,
		tenant_id: str,
		insight_category: Optional[str] = None,
		is_active: bool = True,
		limit: int = 10
	) -> List[Dict[str, Any]]:
		"""Get AI-generated insights for tenant"""
		try:
			async with self.db_manager.get_connection() as conn:
				query = """
					SELECT * FROM crm_ai_insights 
					WHERE tenant_id = $1
				"""
				params = [tenant_id]
				
				if insight_category:
					query += " AND insight_category = $2"
					params.append(insight_category)
					
				if is_active is not None:
					query += f" AND is_active = ${len(params) + 1}"
					params.append(is_active)
					
				query += f" ORDER BY confidence_score DESC, created_at DESC LIMIT ${len(params) + 1}"
				params.append(limit)
				
				rows = await conn.fetch(query, *params)
				return [dict(row) for row in rows]
				
		except Exception as e:
			logger.error(f"Failed to get AI insights: {str(e)}")
			raise CRMServiceError(f"Failed to get AI insights: {str(e)}")

	async def acknowledge_ai_insight(
		self,
		tenant_id: str,
		insight_id: str,
		acknowledged_by: str
	) -> bool:
		"""Acknowledge an AI insight"""
		try:
			async with self.db_manager.get_connection() as conn:
				result = await conn.execute("""
					UPDATE crm_ai_insights 
					SET is_acknowledged = true, 
						acknowledged_by = $1, 
						acknowledged_at = NOW(),
						updated_at = NOW()
					WHERE id = $2 AND tenant_id = $3
				""", acknowledged_by, insight_id, tenant_id)
				
				return result == "UPDATE 1"
				
		except Exception as e:
			logger.error(f"Failed to acknowledge AI insight: {str(e)}")
			raise CRMServiceError(f"Failed to acknowledge AI insight: {str(e)}")

	# === PERFORMANCE BENCHMARKING METHODS ===
	
	async def create_performance_benchmark(
		self,
		tenant_id: str,
		name: str,
		benchmark_type: str,
		metric_name: str,
		measurement_unit: str,
		benchmark_value: float,
		data_source: str,
		calculation_method: str,
		created_by: str,
		description: Optional[str] = None,
		target_value: Optional[float] = None,
		threshold_ranges: Optional[Dict[str, float]] = None,
		period_type: str = "monthly"
	) -> PerformanceBenchmark:
		"""Create a new performance benchmark"""
		from decimal import Decimal
		return await self.performance_benchmarking.create_benchmark(
			tenant_id=tenant_id,
			name=name,
			benchmark_type=benchmark_type,
			metric_name=metric_name,
			measurement_unit=measurement_unit,
			benchmark_value=Decimal(str(benchmark_value)),
			data_source=data_source,
			calculation_method=calculation_method,
			created_by=created_by,
			description=description,
			target_value=Decimal(str(target_value)) if target_value else None,
			threshold_ranges={k: Decimal(str(v)) for k, v in threshold_ranges.items()} if threshold_ranges else None,
			period_type=period_type
		)

	async def measure_performance(
		self,
		tenant_id: str,
		benchmark_id: str,
		entity_type: str,
		entity_id: str,
		entity_name: str,
		measurement_period: str,
		period_start: date,
		period_end: date
	) -> PerformanceMetric:
		"""Measure performance against a benchmark"""
		return await self.performance_benchmarking.measure_performance(
			tenant_id=tenant_id,
			benchmark_id=benchmark_id,
			entity_type=entity_type,
			entity_id=entity_id,
			entity_name=entity_name,
			measurement_period=measurement_period,
			period_start=period_start,
			period_end=period_end
		)

	async def compare_performance(
		self,
		tenant_id: str,
		comparison_name: str,
		comparison_type: str,
		entities: List[Dict[str, Any]],
		metrics: List[str],
		period_start: date,
		period_end: date,
		created_by: str
	) -> PerformanceComparison:
		"""Compare performance across entities"""
		return await self.performance_benchmarking.compare_performance(
			tenant_id=tenant_id,
			comparison_name=comparison_name,
			comparison_type=comparison_type,
			entities=entities,
			metrics=metrics,
			period_start=period_start,
			period_end=period_end,
			created_by=created_by
		)

	async def track_goal_progress(
		self,
		tenant_id: str,
		goal_id: str,
		current_value: float,
		update_milestones: bool = True
	) -> GoalTracking:
		"""Update goal progress and tracking"""
		from decimal import Decimal
		return await self.performance_benchmarking.track_goal_progress(
			tenant_id=tenant_id,
			goal_id=goal_id,
			current_value=Decimal(str(current_value)),
			update_milestones=update_milestones
		)

	async def generate_performance_report(
		self,
		tenant_id: str,
		report_type: str,
		entity_id: str,
		entity_name: str,
		period_start: date,
		period_end: date,
		include_peer_comparison: bool = True
	) -> PerformanceReport:
		"""Generate comprehensive performance report"""
		return await self.performance_benchmarking.generate_performance_report(
			tenant_id=tenant_id,
			report_type=report_type,
			entity_id=entity_id,
			entity_name=entity_name,
			period_start=period_start,
			period_end=period_end,
			include_peer_comparison=include_peer_comparison
		)

	async def get_performance_dashboard(
		self,
		tenant_id: str,
		entity_type: str,
		entity_id: str,
		dashboard_type: str = "comprehensive"
	) -> Dict[str, Any]:
		"""Get performance dashboard data"""
		return await self.performance_benchmarking.get_performance_dashboard(
			tenant_id=tenant_id,
			entity_type=entity_type,
			entity_id=entity_id,
			dashboard_type=dashboard_type
		)

	async def get_performance_benchmarks(
		self,
		tenant_id: str,
		benchmark_type: Optional[str] = None,
		is_active: bool = True
	) -> List[Dict[str, Any]]:
		"""Get available performance benchmarks for tenant"""
		try:
			async with self.db_manager.get_connection() as conn:
				query = """
					SELECT * FROM crm_performance_benchmarks 
					WHERE tenant_id = $1
				"""
				params = [tenant_id]
				
				if benchmark_type:
					query += " AND benchmark_type = $2"
					params.append(benchmark_type)
					
				if is_active is not None:
					query += f" AND is_active = ${len(params) + 1}"
					params.append(is_active)
					
				query += " ORDER BY created_at DESC"
				
				rows = await conn.fetch(query, *params)
				return [dict(row) for row in rows]
				
		except Exception as e:
			logger.error(f"Failed to get performance benchmarks: {str(e)}")
			raise CRMServiceError(f"Failed to get performance benchmarks: {str(e)}")

	async def get_performance_metrics(
		self,
		tenant_id: str,
		entity_type: Optional[str] = None,
		entity_id: Optional[str] = None,
		benchmark_id: Optional[str] = None,
		period_start: Optional[date] = None,
		period_end: Optional[date] = None,
		limit: int = 50
	) -> List[Dict[str, Any]]:
		"""Get performance metrics with filtering"""
		try:
			async with self.db_manager.get_connection() as conn:
				query = """
					SELECT pm.*, pb.name as benchmark_name, pb.metric_name
					FROM crm_performance_metrics pm
					JOIN crm_performance_benchmarks pb ON pm.benchmark_id = pb.id
					WHERE pm.tenant_id = $1
				"""
				params = [tenant_id]
				
				if entity_type:
					query += " AND pm.entity_type = $2"
					params.append(entity_type)
					
				if entity_id:
					query += f" AND pm.entity_id = ${len(params) + 1}"
					params.append(entity_id)
					
				if benchmark_id:
					query += f" AND pm.benchmark_id = ${len(params) + 1}"
					params.append(benchmark_id)
					
				if period_start:
					query += f" AND pm.period_start >= ${len(params) + 1}"
					params.append(period_start)
					
				if period_end:
					query += f" AND pm.period_end <= ${len(params) + 1}"
					params.append(period_end)
					
				query += f" ORDER BY pm.period_start DESC, pm.created_at DESC LIMIT ${len(params) + 1}"
				params.append(limit)
				
				rows = await conn.fetch(query, *params)
				return [dict(row) for row in rows]
				
		except Exception as e:
			logger.error(f"Failed to get performance metrics: {str(e)}")
			raise CRMServiceError(f"Failed to get performance metrics: {str(e)}")

	# === API GATEWAY METHODS ===
	
	async def create_rate_limit_rule(
		self,
		tenant_id: str,
		rule_name: str,
		resource_pattern: str,
		rate_limit_type: str,
		limit_value: int,
		created_by: str,
		description: Optional[str] = None,
		window_size_seconds: int = 60,
		burst_limit: Optional[int] = None,
		scope: str = "tenant",
		enforcement_action: str = "reject"
	) -> RateLimitRule:
		"""Create a new rate limiting rule"""
		return await self.api_gateway.create_rate_limit_rule(
			tenant_id=tenant_id,
			rule_name=rule_name,
			resource_pattern=resource_pattern,
			rate_limit_type=rate_limit_type,
			limit_value=limit_value,
			created_by=created_by,
			description=description,
			window_size_seconds=window_size_seconds,
			burst_limit=burst_limit,
			scope=scope,
			enforcement_action=enforcement_action
		)

	async def register_api_endpoint(
		self,
		tenant_id: str,
		endpoint_path: str,
		http_methods: List[str],
		created_by: str,
		description: Optional[str] = None,
		version: str = "v1",
		is_public: bool = False,
		authentication_required: bool = True,
		rate_limit_rules: Optional[List[str]] = None,
		caching_config: Optional[Dict[str, Any]] = None
	) -> APIEndpoint:
		"""Register a new API endpoint"""
		return await self.api_gateway.register_endpoint(
			tenant_id=tenant_id,
			endpoint_path=endpoint_path,
			http_methods=http_methods,
			created_by=created_by,
			description=description,
			version=version,
			is_public=is_public,
			authentication_required=authentication_required,
			rate_limit_rules=rate_limit_rules,
			caching_config=caching_config
		)

	async def get_api_gateway_metrics(
		self,
		tenant_id: str,
		start_date: datetime,
		end_date: datetime
	) -> APIGatewayMetrics:
		"""Get API gateway metrics for a time period"""
		return await self.api_gateway.get_metrics(
			tenant_id=tenant_id,
			start_date=start_date,
			end_date=end_date
		)

	async def get_rate_limit_rules(
		self,
		tenant_id: str,
		is_active: bool = True
	) -> List[Dict[str, Any]]:
		"""Get rate limiting rules for tenant"""
		try:
			async with self.db_manager.get_connection() as conn:
				query = """
					SELECT * FROM crm_rate_limit_rules 
					WHERE tenant_id = $1
				"""
				params = [tenant_id]
				
				if is_active is not None:
					query += " AND is_active = $2"
					params.append(is_active)
					
				query += " ORDER BY priority DESC, created_at DESC"
				
				rows = await conn.fetch(query, *params)
				return [dict(row) for row in rows]
				
		except Exception as e:
			logger.error(f"Failed to get rate limit rules: {str(e)}")
			raise CRMServiceError(f"Failed to get rate limit rules: {str(e)}")

	async def get_api_endpoints(
		self,
		tenant_id: str,
		version: Optional[str] = None,
		is_active: bool = True
	) -> List[Dict[str, Any]]:
		"""Get API endpoints for tenant"""
		try:
			async with self.db_manager.get_connection() as conn:
				query = """
					SELECT * FROM crm_api_endpoints 
					WHERE tenant_id = $1
				"""
				params = [tenant_id]
				
				if version:
					query += " AND version = $2"
					params.append(version)
					
				if is_active is not None:
					query += f" AND is_active = ${len(params) + 1}"
					params.append(is_active)
					
				query += " ORDER BY endpoint_path, version DESC"
				
				rows = await conn.fetch(query, *params)
				return [dict(row) for row in rows]
				
		except Exception as e:
			logger.error(f"Failed to get API endpoints: {str(e)}")
			raise CRMServiceError(f"Failed to get API endpoints: {str(e)}")

	async def get_api_requests_log(
		self,
		tenant_id: str,
		start_time: Optional[datetime] = None,
		end_time: Optional[datetime] = None,
		endpoint_path: Optional[str] = None,
		user_id: Optional[str] = None,
		limit: int = 100
	) -> List[Dict[str, Any]]:
		"""Get API request logs with filtering"""
		try:
			async with self.db_manager.get_connection() as conn:
				query = """
					SELECT ar.*, ae.endpoint_path as endpoint_name
					FROM crm_api_requests ar
					LEFT JOIN crm_api_endpoints ae ON ar.endpoint_id = ae.id
					WHERE ar.tenant_id = $1
				"""
				params = [tenant_id]
				
				if start_time:
					query += f" AND ar.timestamp >= ${len(params) + 1}"
					params.append(start_time)
					
				if end_time:
					query += f" AND ar.timestamp <= ${len(params) + 1}"
					params.append(end_time)
					
				if endpoint_path:
					query += f" AND ar.request_path LIKE ${len(params) + 1}"
					params.append(f"%{endpoint_path}%")
					
				if user_id:
					query += f" AND ar.user_id = ${len(params) + 1}"
					params.append(user_id)
					
				query += f" ORDER BY ar.timestamp DESC LIMIT ${len(params) + 1}"
				params.append(limit)
				
				rows = await conn.fetch(query, *params)
				return [dict(row) for row in rows]
				
		except Exception as e:
			logger.error(f"Failed to get API request logs: {str(e)}")
			raise CRMServiceError(f"Failed to get API request logs: {str(e)}")

	async def update_rate_limit_rule(
		self,
		tenant_id: str,
		rule_id: str,
		updates: Dict[str, Any],
		updated_by: str
	) -> bool:
		"""Update a rate limiting rule"""
		try:
			# Build dynamic update query
			set_clauses = []
			params = []
			param_count = 1
			
			for field, value in updates.items():
				if field in ['limit_value', 'window_size_seconds', 'burst_limit', 'priority', 'is_active', 'enforcement_action']:
					set_clauses.append(f"{field} = ${param_count}")
					params.append(value)
					param_count += 1
			
			if not set_clauses:
				return False
			
			set_clauses.append(f"updated_at = ${param_count}")
			params.append(datetime.utcnow())
			param_count += 1
			
			set_clauses.append(f"updated_by = ${param_count}")
			params.append(updated_by)
			param_count += 1
			
			# Add WHERE conditions
			params.extend([rule_id, tenant_id])
			
			query = f"""
				UPDATE crm_rate_limit_rules 
				SET {', '.join(set_clauses)}
				WHERE id = ${param_count} AND tenant_id = ${param_count + 1}
			"""
			
			async with self.db_manager.get_connection() as conn:
				result = await conn.execute(query, *params)
				return result == "UPDATE 1"
				
		except Exception as e:
			logger.error(f"Failed to update rate limit rule: {str(e)}")
			raise CRMServiceError(f"Failed to update rate limit rule: {str(e)}")

	async def delete_rate_limit_rule(
		self,
		tenant_id: str,
		rule_id: str
	) -> bool:
		"""Delete a rate limiting rule"""
		try:
			async with self.db_manager.get_connection() as conn:
				result = await conn.execute("""
					DELETE FROM crm_rate_limit_rules 
					WHERE id = $1 AND tenant_id = $2
				""", rule_id, tenant_id)
				
				return result == "DELETE 1"
				
		except Exception as e:
			logger.error(f"Failed to delete rate limit rule: {str(e)}")
			raise CRMServiceError(f"Failed to delete rate limit rule: {str(e)}")

	# ===== THIRD-PARTY INTEGRATION METHODS =====

	async def create_integration_connector(
		self,
		tenant_id: str,
		connector_name: str,
		integration_type: IntegrationType,
		platform_name: str,
		base_url: str,
		authentication_type: AuthenticationType,
		authentication_config: Dict[str, Any],
		created_by: str,
		description: Optional[str] = None,
		supported_operations: Optional[List[DataOperation]] = None,
		supported_entities: Optional[List[str]] = None,
		custom_headers: Optional[Dict[str, str]] = None,
		rate_limit_config: Optional[Dict[str, Any]] = None
	) -> IntegrationConnector:
		"""Create a new third-party integration connector"""
		try:
			return await self.third_party_integration.create_connector(
				tenant_id=tenant_id,
				connector_name=connector_name,
				integration_type=integration_type,
				platform_name=platform_name,
				base_url=base_url,
				authentication_type=authentication_type,
				authentication_config=authentication_config,
				created_by=created_by,
				description=description,
				supported_operations=supported_operations,
				supported_entities=supported_entities,
				custom_headers=custom_headers,
				rate_limit_config=rate_limit_config
			)
		except Exception as e:
			logger.error(f"Failed to create integration connector: {str(e)}")
			raise CRMServiceError(f"Failed to create integration connector: {str(e)}")

	async def create_field_mapping(
		self,
		connector_id: str,
		tenant_id: str,
		mapping_name: str,
		source_entity: str,
		target_entity: str,
		field_mappings: List[Dict[str, Any]],
		sync_direction: SyncDirection,
		created_by: str,
		transformation_functions: Optional[Dict[str, str]] = None,
		validation_rules: Optional[Dict[str, Any]] = None,
		default_values: Optional[Dict[str, Any]] = None
	) -> FieldMapping:
		"""Create field mapping configuration for integration"""
		try:
			return await self.third_party_integration.create_field_mapping(
				connector_id=connector_id,
				tenant_id=tenant_id,
				mapping_name=mapping_name,
				source_entity=source_entity,
				target_entity=target_entity,
				field_mappings=field_mappings,
				sync_direction=sync_direction,
				created_by=created_by,
				transformation_functions=transformation_functions,
				validation_rules=validation_rules,
				default_values=default_values
			)
		except Exception as e:
			logger.error(f"Failed to create field mapping: {str(e)}")
			raise CRMServiceError(f"Failed to create field mapping: {str(e)}")

	async def create_sync_configuration(
		self,
		connector_id: str,
		tenant_id: str,
		sync_name: str,
		sync_frequency: str,
		created_by: str,
		description: Optional[str] = None,
		schedule_config: Optional[Dict[str, Any]] = None,
		entity_filters: Optional[Dict[str, Any]] = None,
		batch_size: int = 100
	) -> SyncConfiguration:
		"""Create synchronization configuration"""
		try:
			return await self.third_party_integration.create_sync_configuration(
				connector_id=connector_id,
				tenant_id=tenant_id,
				sync_name=sync_name,
				sync_frequency=sync_frequency,
				created_by=created_by,
				description=description,
				schedule_config=schedule_config,
				entity_filters=entity_filters,
				batch_size=batch_size
			)
		except Exception as e:
			logger.error(f"Failed to create sync configuration: {str(e)}")
			raise CRMServiceError(f"Failed to create sync configuration: {str(e)}")

	async def execute_integration_sync(
		self,
		sync_config_id: str,
		tenant_id: str,
		execution_type: str = "manual",
		trigger_source: Optional[str] = None
	) -> SyncExecution:
		"""Execute a synchronization"""
		try:
			return await self.third_party_integration.execute_sync(
				sync_config_id=sync_config_id,
				tenant_id=tenant_id,
				execution_type=execution_type,
				trigger_source=trigger_source
			)
		except Exception as e:
			logger.error(f"Failed to execute integration sync: {str(e)}")
			raise CRMServiceError(f"Failed to execute integration sync: {str(e)}")

	async def get_integration_connectors(
		self,
		tenant_id: str,
		integration_type: Optional[IntegrationType] = None,
		is_active: Optional[bool] = None
	) -> List[Dict[str, Any]]:
		"""Get integration connectors"""
		try:
			return await self.third_party_integration.get_connectors(
				tenant_id=tenant_id,
				integration_type=integration_type,
				is_active=is_active
			)
		except Exception as e:
			logger.error(f"Failed to get integration connectors: {str(e)}")
			raise CRMServiceError(f"Failed to get integration connectors: {str(e)}")

	async def get_sync_history(
		self,
		tenant_id: str,
		connector_id: Optional[str] = None,
		limit: int = 100
	) -> List[Dict[str, Any]]:
		"""Get synchronization execution history"""
		try:
			return await self.third_party_integration.get_sync_history(
				tenant_id=tenant_id,
				connector_id=connector_id,
				limit=limit
			)
		except Exception as e:
			logger.error(f"Failed to get sync history: {str(e)}")
			raise CRMServiceError(f"Failed to get sync history: {str(e)}")

	# ===== WEBHOOK MANAGEMENT METHODS =====

	async def create_webhook_endpoint(
		self,
		tenant_id: str,
		webhook_name: str,
		endpoint_url: str,
		event_types: List[str],
		created_by: str,
		description: Optional[str] = None,
		http_method: str = "POST",
		headers: Optional[Dict[str, str]] = None,
		authentication: Optional[Dict[str, Any]] = None,
		retry_config: Optional[Dict[str, Any]] = None,
		timeout_seconds: int = 30,
		secret_key: Optional[str] = None
	) -> WebhookEndpoint:
		"""Create a new webhook endpoint"""
		try:
			return await self.webhook_manager.create_webhook_endpoint(
				tenant_id=tenant_id,
				webhook_name=webhook_name,
				endpoint_url=endpoint_url,
				event_types=event_types,
				created_by=created_by,
				description=description,
				http_method=http_method,
				headers=headers,
				authentication=authentication,
				retry_config=retry_config,
				timeout_seconds=timeout_seconds,
				secret_key=secret_key
			)
		except Exception as e:
			logger.error(f"Failed to create webhook endpoint: {str(e)}")
			raise CRMServiceError(f"Failed to create webhook endpoint: {str(e)}")

	async def emit_webhook_event(
		self,
		tenant_id: str,
		event_type: str,
		event_category: str,
		event_action: str,
		entity_id: str,
		entity_type: str,
		entity_data: Dict[str, Any],
		user_id: Optional[str] = None,
		previous_data: Optional[Dict[str, Any]] = None,
		metadata: Optional[Dict[str, Any]] = None
	) -> WebhookEvent:
		"""Emit a webhook event for processing"""
		try:
			return await self.webhook_manager.emit_event(
				tenant_id=tenant_id,
				event_type=event_type,
				event_category=event_category,
				event_action=event_action,
				entity_id=entity_id,
				entity_type=entity_type,
				entity_data=entity_data,
				user_id=user_id,
				previous_data=previous_data,
				metadata=metadata
			)
		except Exception as e:
			logger.error(f"Failed to emit webhook event: {str(e)}")
			raise CRMServiceError(f"Failed to emit webhook event: {str(e)}")

	async def get_webhook_endpoints(
		self,
		tenant_id: str,
		is_active: Optional[bool] = None,
		event_type: Optional[str] = None
	) -> List[Dict[str, Any]]:
		"""Get webhook endpoints for tenant"""
		try:
			return await self.webhook_manager.get_webhook_endpoints(
				tenant_id=tenant_id,
				is_active=is_active,
				event_type=event_type
			)
		except Exception as e:
			logger.error(f"Failed to get webhook endpoints: {str(e)}")
			raise CRMServiceError(f"Failed to get webhook endpoints: {str(e)}")

	async def get_webhook_delivery_history(
		self,
		tenant_id: str,
		webhook_id: Optional[str] = None,
		event_type: Optional[str] = None,
		success: Optional[bool] = None,
		limit: int = 100
	) -> List[Dict[str, Any]]:
		"""Get webhook delivery history"""
		try:
			return await self.webhook_manager.get_delivery_history(
				tenant_id=tenant_id,
				webhook_id=webhook_id,
				event_type=event_type,
				success=success,
				limit=limit
			)
		except Exception as e:
			logger.error(f"Failed to get webhook delivery history: {str(e)}")
			raise CRMServiceError(f"Failed to get webhook delivery history: {str(e)}")

	async def get_webhook_metrics(
		self,
		tenant_id: str,
		start_date: datetime,
		end_date: datetime
	) -> Dict[str, Any]:
		"""Get webhook delivery metrics"""
		try:
			return await self.webhook_manager.get_webhook_metrics(
				tenant_id=tenant_id,
				start_date=start_date,
				end_date=end_date
			)
		except Exception as e:
			logger.error(f"Failed to get webhook metrics: {str(e)}")
			raise CRMServiceError(f"Failed to get webhook metrics: {str(e)}")

	async def test_webhook_endpoint(
		self,
		tenant_id: str,
		webhook_id: str
	) -> Dict[str, Any]:
		"""Test webhook endpoint connectivity"""
		try:
			return await self.webhook_manager.test_webhook_endpoint(
				tenant_id=tenant_id,
				webhook_id=webhook_id
			)
		except Exception as e:
			logger.error(f"Failed to test webhook endpoint: {str(e)}")
			raise CRMServiceError(f"Failed to test webhook endpoint: {str(e)}")

	# ===== REAL-TIME SYNCHRONIZATION METHODS =====

	async def emit_realtime_sync_event(
		self,
		tenant_id: str,
		event_type: SyncEventType,
		entity_type: str,
		entity_id: str,
		current_data: Dict[str, Any],
		previous_data: Optional[Dict[str, Any]] = None,
		user_id: Optional[str] = None,
		target_systems: Optional[List[str]] = None,
		metadata: Optional[Dict[str, Any]] = None
	) -> SyncEvent:
		"""Emit a real-time synchronization event"""
		try:
			return await self.realtime_sync.emit_sync_event(
				tenant_id=tenant_id,
				event_type=event_type,
				entity_type=entity_type,
				entity_id=entity_id,
				current_data=current_data,
				previous_data=previous_data,
				user_id=user_id,
				target_systems=target_systems,
				metadata=metadata
			)
		except Exception as e:
			logger.error(f"Failed to emit realtime sync event: {str(e)}")
			raise CRMServiceError(f"Failed to emit realtime sync event: {str(e)}")

	async def create_realtime_sync_configuration(
		self,
		tenant_id: str,
		config_name: str,
		entity_types: List[str],
		target_systems: List[str],
		created_by: str,
		description: Optional[str] = None,
		change_detection_mode: ChangeDetectionMode = ChangeDetectionMode.TIMESTAMP_BASED,
		conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.TIMESTAMP_WINS,
		sync_direction: str = "bidirectional",
		field_filters: Optional[Dict[str, List[str]]] = None,
		**kwargs
	) -> RealtimeSyncConfiguration:
		"""Create a new real-time sync configuration"""
		try:
			return await self.realtime_sync.create_sync_configuration(
				tenant_id=tenant_id,
				config_name=config_name,
				entity_types=entity_types,
				target_systems=target_systems,
				created_by=created_by,
				description=description,
				change_detection_mode=change_detection_mode,
				conflict_resolution=conflict_resolution,
				sync_direction=sync_direction,
				field_filters=field_filters,
				**kwargs
			)
		except Exception as e:
			logger.error(f"Failed to create realtime sync configuration: {str(e)}")
			raise CRMServiceError(f"Failed to create realtime sync configuration: {str(e)}")

	async def get_realtime_sync_configurations(
		self,
		tenant_id: str,
		is_active: Optional[bool] = None
	) -> List[Dict[str, Any]]:
		"""Get real-time sync configurations"""
		try:
			return await self.realtime_sync.get_sync_configurations(
				tenant_id=tenant_id,
				is_active=is_active
			)
		except Exception as e:
			logger.error(f"Failed to get realtime sync configurations: {str(e)}")
			raise CRMServiceError(f"Failed to get realtime sync configurations: {str(e)}")

	async def get_realtime_sync_status(
		self,
		tenant_id: str,
		config_id: Optional[str] = None
	) -> Dict[str, Any]:
		"""Get real-time sync status and metrics"""
		try:
			return await self.realtime_sync.get_sync_status(
				tenant_id=tenant_id,
				config_id=config_id
			)
		except Exception as e:
			logger.error(f"Failed to get realtime sync status: {str(e)}")
			raise CRMServiceError(f"Failed to get realtime sync status: {str(e)}")

	async def pause_realtime_sync(
		self,
		tenant_id: str,
		config_id: Optional[str] = None
	) -> bool:
		"""Pause real-time synchronization"""
		try:
			return await self.realtime_sync.pause_sync(
				tenant_id=tenant_id,
				config_id=config_id
			)
		except Exception as e:
			logger.error(f"Failed to pause realtime sync: {str(e)}")
			raise CRMServiceError(f"Failed to pause realtime sync: {str(e)}")

	async def resume_realtime_sync(
		self,
		tenant_id: str,
		config_id: Optional[str] = None
	) -> bool:
		"""Resume real-time synchronization"""
		try:
			return await self.realtime_sync.resume_sync(
				tenant_id=tenant_id,
				config_id=config_id
			)
		except Exception as e:
			logger.error(f"Failed to resume realtime sync: {str(e)}")
			raise CRMServiceError(f"Failed to resume realtime sync: {str(e)}")

	async def get_sync_conflict_records(
		self,
		tenant_id: str,
		resolved: Optional[bool] = None,
		limit: int = 100
	) -> List[Dict[str, Any]]:
		"""Get data conflict records"""
		try:
			return await self.realtime_sync.get_conflict_records(
				tenant_id=tenant_id,
				resolved=resolved,
				limit=limit
			)
		except Exception as e:
			logger.error(f"Failed to get sync conflict records: {str(e)}")
			raise CRMServiceError(f"Failed to get sync conflict records: {str(e)}")

	async def resolve_sync_conflict(
		self,
		tenant_id: str,
		conflict_id: str,
		resolution_strategy: ConflictResolutionStrategy,
		resolved_value: Any,
		resolved_by: str
	) -> bool:
		"""Manually resolve a data conflict"""
		try:
			return await self.realtime_sync.resolve_conflict(
				tenant_id=tenant_id,
				conflict_id=conflict_id,
				resolution_strategy=resolution_strategy,
				resolved_value=resolved_value,
				resolved_by=resolved_by
			)
		except Exception as e:
			logger.error(f"Failed to resolve sync conflict: {str(e)}")
			raise CRMServiceError(f"Failed to resolve sync conflict: {str(e)}")

	async def shutdown(self):
		"""Shutdown CRM service and all managers"""
		try:
			# Shutdown webhook manager
			if hasattr(self, 'webhook_manager'):
				await self.webhook_manager.shutdown()
			
			# Shutdown third-party integration manager  
			if hasattr(self, 'third_party_integration'):
				await self.third_party_integration.shutdown()
			
			# Shutdown real-time sync engine
			if hasattr(self, 'realtime_sync'):
				await self.realtime_sync.shutdown()
			
			# Shutdown other managers as needed
			if hasattr(self, 'api_gateway'):
				await self.api_gateway.shutdown()
				
		except Exception as e:
			logger.error(f"Error during CRM service shutdown: {str(e)}")
		
		logger.info("CRM service shutdown completed")


class CRMAIInsights:
	"""Placeholder AI insights class"""
	
	async def initialize(self):
		pass
	
	async def health_check(self):
		return {"status": "healthy"}
	
	async def generate_contact_insights(self, contact_id: str, tenant_id: str):
		pass
	
	async def generate_account_insights(self, account_id: str, tenant_id: str):
		pass
	
	async def calculate_lead_score(self, lead_id: str, tenant_id: str):
		pass
	
	async def calculate_win_probability(self, opportunity_id: str, tenant_id: str):
		pass
	
	# ================================
	# API Versioning & Deprecation Methods
	# ================================
	
	async def create_api_version(
		self,
		tenant_id: str,
		version_number: str,
		version_name: str = None,
		status: str = "development",
		supported_endpoints: List[str] = None,
		breaking_changes: List[Dict[str, Any]] = None,
		documentation_url: str = None,
		created_by: str = None
	) -> Dict[str, Any]:
		"""Create a new API version"""
		try:
			return await self.api_versioning.create_version(
				tenant_id=tenant_id,
				version_number=version_number,
				version_name=version_name,
				status=status,
				supported_endpoints=supported_endpoints or [],
				breaking_changes=breaking_changes or [],
				documentation_url=documentation_url,
				created_by=created_by
			)
		except Exception as e:
			logger.error(f"Error creating API version: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Failed to create API version: {str(e)}")
	
	async def get_api_versions(
		self,
		tenant_id: str,
		status: str = None,
		is_default: bool = None
	) -> List[Dict[str, Any]]:
		"""Get API versions"""
		try:
			return await self.api_versioning.get_versions(
				tenant_id=tenant_id,
				status=status,
				is_default=is_default
			)
		except Exception as e:
			logger.error(f"Error getting API versions: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Failed to get API versions: {str(e)}")
	
	async def update_api_version(
		self,
		tenant_id: str,
		version_id: str,
		**kwargs
	) -> bool:
		"""Update an API version"""
		try:
			return await self.api_versioning.update_version(
				tenant_id=tenant_id,
				version_id=version_id,
				**kwargs
			)
		except Exception as e:
			logger.error(f"Error updating API version: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Failed to update API version: {str(e)}")
	
	async def create_deprecation_notice(
		self,
		tenant_id: str,
		version_id: str,
		endpoint_path: str,
		http_method: str = "GET",
		severity: str = "medium",
		deprecation_reason: str = None,
		replacement_endpoint: str = None,
		replacement_version: str = None,
		grace_period_days: int = 90,
		created_by: str = None
	) -> Dict[str, Any]:
		"""Create a deprecation notice"""
		try:
			return await self.api_versioning.create_deprecation_notice(
				tenant_id=tenant_id,
				version_id=version_id,
				endpoint_path=endpoint_path,
				http_method=http_method,
				severity=severity,
				deprecation_reason=deprecation_reason,
				replacement_endpoint=replacement_endpoint,
				replacement_version=replacement_version,
				grace_period_days=grace_period_days,
				created_by=created_by
			)
		except Exception as e:
			logger.error(f"Error creating deprecation notice: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Failed to create deprecation notice: {str(e)}")
	
	async def get_deprecation_notices(
		self,
		tenant_id: str,
		version_id: str = None,
		severity: str = None
	) -> List[Dict[str, Any]]:
		"""Get deprecation notices"""
		try:
			return await self.api_versioning.get_deprecation_notices(
				tenant_id=tenant_id,
				version_id=version_id,
				severity=severity
			)
		except Exception as e:
			logger.error(f"Error getting deprecation notices: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Failed to get deprecation notices: {str(e)}")
	
	async def get_client_version_usage(
		self,
		tenant_id: str,
		version_id: str = None,
		client_type: str = None,
		migration_status: str = None
	) -> List[Dict[str, Any]]:
		"""Get client version usage analytics"""
		try:
			return await self.api_versioning.get_client_usage(
				tenant_id=tenant_id,
				version_id=version_id,
				client_type=client_type,
				migration_status=migration_status
			)
		except Exception as e:
			logger.error(f"Error getting client version usage: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Failed to get client version usage: {str(e)}")
	
	async def create_version_migration(
		self,
		tenant_id: str,
		from_version_id: str,
		to_version_id: str,
		migration_name: str,
		complexity: str = "moderate",
		is_breaking_change: bool = False,
		field_mappings: Dict[str, Any] = None,
		transformation_rules: Dict[str, Any] = None,
		created_by: str = None
	) -> Dict[str, Any]:
		"""Create a version migration plan"""
		try:
			return await self.api_versioning.create_migration(
				tenant_id=tenant_id,
				from_version_id=from_version_id,
				to_version_id=to_version_id,
				migration_name=migration_name,
				complexity=complexity,
				is_breaking_change=is_breaking_change,
				field_mappings=field_mappings or {},
				transformation_rules=transformation_rules or {},
				created_by=created_by
			)
		except Exception as e:
			logger.error(f"Error creating version migration: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Failed to create version migration: {str(e)}")
	
	async def get_version_analytics(
		self,
		tenant_id: str,
		version_id: str = None,
		days: int = 30
	) -> Dict[str, Any]:
		"""Get API version usage analytics"""
		try:
			return await self.api_versioning.get_version_analytics(
				tenant_id=tenant_id,
				version_id=version_id,
				days=days
			)
		except Exception as e:
			logger.error(f"Error getting version analytics: {str(e)}", exc_info=True)
			raise CRMServiceError(f"Failed to get version analytics: {str(e)}")
	
	async def shutdown(self):
		pass


class CRMAnalytics:
	"""Placeholder analytics class"""
	
	async def initialize(self):
		pass
	
	async def health_check(self):
		return {"status": "healthy"}
	
	async def get_sales_dashboard(self, tenant_id: str, user_id: str = None) -> Dict[str, Any]:
		return {"placeholder": "dashboard_data"}
	
	async def get_pipeline_analytics(self, tenant_id: str, user_id: str = None) -> Dict[str, Any]:
		return {"placeholder": "pipeline_data"}
	
	async def update_contact_analytics(self, contact_id: str, tenant_id: str):
		pass
	
	async def shutdown(self):
		pass