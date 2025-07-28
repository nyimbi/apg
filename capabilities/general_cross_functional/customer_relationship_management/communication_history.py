"""
APG Customer Relationship Management - Communication History Module

Advanced communication tracking system that maintains comprehensive records
of all interactions with contacts and accounts, providing valuable context
for relationship management and sales processes.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from enum import Enum
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ValidationError

from .models import CRMContact, CRMAccount
from .database import DatabaseManager


logger = logging.getLogger(__name__)


class CommunicationType(str, Enum):
	"""Types of communications"""
	EMAIL = "email"
	PHONE_CALL = "phone_call"
	VIDEO_CALL = "video_call"
	MEETING = "meeting"
	SMS = "sms"
	CHAT = "chat"
	SOCIAL_MEDIA = "social_media"
	LETTER = "letter"
	FAX = "fax"
	WEBSITE_VISIT = "website_visit"
	WEBINAR = "webinar"
	EVENT = "event"
	OTHER = "other"


class CommunicationDirection(str, Enum):
	"""Direction of communication"""
	INBOUND = "inbound"
	OUTBOUND = "outbound"
	INTERNAL = "internal"


class CommunicationStatus(str, Enum):
	"""Status of communication"""
	SCHEDULED = "scheduled"
	COMPLETED = "completed"
	CANCELLED = "cancelled"
	NO_SHOW = "no_show"
	RESCHEDULED = "rescheduled"


class CommunicationPriority(str, Enum):
	"""Priority level of communication"""
	LOW = "low"
	NORMAL = "normal"
	HIGH = "high"
	URGENT = "urgent"


class CommunicationOutcome(str, Enum):
	"""Outcome of communication"""
	SUCCESSFUL = "successful"
	UNSUCCESSFUL = "unsuccessful"
	FOLLOW_UP_REQUIRED = "follow_up_required"
	MEETING_SCHEDULED = "meeting_scheduled"
	PROPOSAL_REQUESTED = "proposal_requested"
	DECISION_PENDING = "decision_pending"
	CLOSED_WON = "closed_won"
	CLOSED_LOST = "closed_lost"
	NO_INTEREST = "no_interest"


class Communication(BaseModel):
	"""Communication record model"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	
	# Related entities
	contact_id: Optional[str] = None
	account_id: Optional[str] = None
	lead_id: Optional[str] = None
	opportunity_id: Optional[str] = None
	
	# Communication details
	communication_type: CommunicationType
	direction: CommunicationDirection
	status: CommunicationStatus = CommunicationStatus.COMPLETED
	priority: CommunicationPriority = CommunicationPriority.NORMAL
	
	# Content
	subject: str
	content: Optional[str] = None
	summary: Optional[str] = None
	
	# Participants
	from_address: Optional[str] = None
	to_addresses: List[str] = Field(default_factory=list)
	cc_addresses: List[str] = Field(default_factory=list)
	bcc_addresses: List[str] = Field(default_factory=list)
	participants: List[str] = Field(default_factory=list)
	
	# Timing
	scheduled_at: Optional[datetime] = None
	started_at: Optional[datetime] = None
	ended_at: Optional[datetime] = None
	duration_minutes: Optional[int] = None
	
	# Outcome and follow-up
	outcome: Optional[CommunicationOutcome] = None
	outcome_notes: Optional[str] = None
	follow_up_required: bool = False
	follow_up_date: Optional[datetime] = None
	follow_up_notes: Optional[str] = None
	
	# Attachments and references
	attachments: List[Dict[str, Any]] = Field(default_factory=list)
	external_id: Optional[str] = None  # ID from external system (email, calendar, etc.)
	external_source: Optional[str] = None  # Source system name
	
	# Metadata
	tags: List[str] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_by: str
	version: int = 1


class CommunicationTemplate(BaseModel):
	"""Communication template for common messages"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	
	# Template details
	name: str
	description: Optional[str] = None
	communication_type: CommunicationType
	
	# Template content
	subject_template: str
	content_template: str
	
	# Template variables
	variables: List[str] = Field(default_factory=list)  # List of variable names
	
	# Usage tracking
	usage_count: int = 0
	last_used_at: Optional[datetime] = None
	
	# Categorization
	category: Optional[str] = None
	tags: List[str] = Field(default_factory=list)
	
	# Status
	is_active: bool = True
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_by: str


class CommunicationAnalytics(BaseModel):
	"""Communication analytics data"""
	total_communications: int = 0
	communications_by_type: Dict[str, int] = Field(default_factory=dict)
	communications_by_direction: Dict[str, int] = Field(default_factory=dict)
	communications_by_outcome: Dict[str, int] = Field(default_factory=dict)
	
	# Timing analytics
	avg_response_time_hours: Optional[float] = None
	total_duration_minutes: int = 0
	avg_duration_minutes: Optional[float] = None
	
	# Trend data
	communications_by_date: Dict[str, int] = Field(default_factory=dict)
	response_rate_percentage: Optional[float] = None
	
	# Entity-specific analytics
	most_active_contacts: List[Dict[str, Any]] = Field(default_factory=list)
	most_active_accounts: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Follow-up analytics
	pending_follow_ups: int = 0
	overdue_follow_ups: int = 0


class CommunicationManager:
	"""
	Advanced communication history management system
	
	Provides comprehensive tracking of all customer communications including
	emails, calls, meetings, and other interactions with intelligent analytics
	and follow-up management.
	"""
	
	def __init__(self, db_manager: DatabaseManager):
		"""
		Initialize communication manager
		
		Args:
			db_manager: Database manager instance
		"""
		self.db_manager = db_manager
		self._initialized = False
	
	async def initialize(self):
		"""Initialize the communication manager"""
		if self._initialized:
			return
		
		logger.info("🔧 Initializing Communication Manager...")
		
		# Ensure database connection
		if not self.db_manager._initialized:
			await self.db_manager.initialize()
		
		self._initialized = True
		logger.info("✅ Communication Manager initialized successfully")
	
	async def create_communication(
		self,
		communication_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> Communication:
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
			
			# Add required fields
			communication_data.update({
				'tenant_id': tenant_id,
				'created_by': created_by,
				'updated_by': created_by
			})
			
			# Create communication object
			communication = Communication(**communication_data)
			
			# Validate related entities exist
			await self._validate_related_entities(communication, tenant_id)
			
			# Store in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_communications (
						id, tenant_id, contact_id, account_id, lead_id, opportunity_id,
						communication_type, direction, status, priority,
						subject, content, summary,
						from_address, to_addresses, cc_addresses, bcc_addresses, participants,
						scheduled_at, started_at, ended_at, duration_minutes,
						outcome, outcome_notes, follow_up_required, follow_up_date, follow_up_notes,
						attachments, external_id, external_source,
						tags, metadata,
						created_at, updated_at, created_by, updated_by, version
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
						$11, $12, $13, $14, $15, $16, $17, $18,
						$19, $20, $21, $22, $23, $24, $25, $26, $27,
						$28, $29, $30, $31, $32, $33, $34, $35, $36, $37
					)
				""", 
					communication.id, communication.tenant_id, communication.contact_id, 
					communication.account_id, communication.lead_id, communication.opportunity_id,
					communication.communication_type.value, communication.direction.value,
					communication.status.value, communication.priority.value,
					communication.subject, communication.content, communication.summary,
					communication.from_address, communication.to_addresses, communication.cc_addresses,
					communication.bcc_addresses, communication.participants,
					communication.scheduled_at, communication.started_at, communication.ended_at,
					communication.duration_minutes,
					communication.outcome.value if communication.outcome else None,
					communication.outcome_notes, communication.follow_up_required,
					communication.follow_up_date, communication.follow_up_notes,
					communication.attachments, communication.external_id, communication.external_source,
					communication.tags, communication.metadata,
					communication.created_at, communication.updated_at,
					communication.created_by, communication.updated_by, communication.version
				)
			
			logger.info(f"✅ Communication created successfully: {communication.id}")
			return communication
			
		except Exception as e:
			logger.error(f"Failed to create communication: {str(e)}", exc_info=True)
			raise
	
	async def get_communication(
		self,
		communication_id: str,
		tenant_id: str
	) -> Optional[Communication]:
		"""
		Get communication by ID
		
		Args:
			communication_id: Communication identifier
			tenant_id: Tenant identifier
			
		Returns:
			Communication record if found
		"""
		try:
			async with self.db_manager.get_connection() as conn:
				row = await conn.fetchrow("""
					SELECT * FROM crm_communications 
					WHERE id = $1 AND tenant_id = $2
				""", communication_id, tenant_id)
				
				if not row:
					return None
				
				return Communication(**dict(row))
				
		except Exception as e:
			logger.error(f"Failed to get communication: {str(e)}", exc_info=True)
			raise
	
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
			Dict containing communications and pagination info
		"""
		try:
			# Build query conditions
			conditions = ["tenant_id = $1"]
			params = [tenant_id]
			param_count = 1
			
			if contact_id:
				param_count += 1
				conditions.append(f"contact_id = ${param_count}")
				params.append(contact_id)
			
			if account_id:
				param_count += 1
				conditions.append(f"account_id = ${param_count}")
				params.append(account_id)
			
			if communication_type:
				param_count += 1
				conditions.append(f"communication_type = ${param_count}")
				params.append(communication_type.value)
			
			if direction:
				param_count += 1
				conditions.append(f"direction = ${param_count}")
				params.append(direction.value)
			
			if status:
				param_count += 1
				conditions.append(f"status = ${param_count}")
				params.append(status.value)
			
			if start_date:
				param_count += 1
				conditions.append(f"created_at >= ${param_count}")
				params.append(start_date)
			
			if end_date:
				param_count += 1
				conditions.append(f"created_at <= ${param_count}")
				params.append(end_date)
			
			if tags:
				param_count += 1
				conditions.append(f"tags && ${param_count}")
				params.append(tags)
			
			where_clause = " WHERE " + " AND ".join(conditions)
			
			async with self.db_manager.get_connection() as conn:
				# Get total count
				count_query = f"SELECT COUNT(*) FROM crm_communications{where_clause}"
				total = await conn.fetchval(count_query, *params)
				
				# Get communications
				query = f"""
					SELECT * FROM crm_communications
					{where_clause}
					ORDER BY created_at DESC
					LIMIT {limit} OFFSET {offset}
				"""
				
				rows = await conn.fetch(query, *params)
				communications = [Communication(**dict(row)) for row in rows]
			
			return {
				"communications": communications,
				"total": total,
				"limit": limit,
				"offset": offset
			}
			
		except Exception as e:
			logger.error(f"Failed to list communications: {str(e)}", exc_info=True)
			raise
	
	async def update_communication(
		self,
		communication_id: str,
		update_data: Dict[str, Any],
		tenant_id: str,
		updated_by: str
	) -> Communication:
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
			
			# Get existing communication
			communication = await self.get_communication(communication_id, tenant_id)
			if not communication:
				raise ValueError(f"Communication not found: {communication_id}")
			
			# Update fields
			update_data['updated_by'] = updated_by
			update_data['updated_at'] = datetime.utcnow()
			update_data['version'] = communication.version + 1
			
			# Apply updates
			for field, value in update_data.items():
				if hasattr(communication, field):
					setattr(communication, field, value)
			
			# Validate related entities if changed
			if any(field in update_data for field in ['contact_id', 'account_id', 'lead_id', 'opportunity_id']):
				await self._validate_related_entities(communication, tenant_id)
			
			# Update in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					UPDATE crm_communications SET
						contact_id = $3, account_id = $4, lead_id = $5, opportunity_id = $6,
						communication_type = $7, direction = $8, status = $9, priority = $10,
						subject = $11, content = $12, summary = $13,
						from_address = $14, to_addresses = $15, cc_addresses = $16, 
						bcc_addresses = $17, participants = $18,
						scheduled_at = $19, started_at = $20, ended_at = $21, duration_minutes = $22,
						outcome = $23, outcome_notes = $24, follow_up_required = $25, 
						follow_up_date = $26, follow_up_notes = $27,
						attachments = $28, external_id = $29, external_source = $30,
						tags = $31, metadata = $32,
						updated_at = $33, updated_by = $34, version = $35
					WHERE id = $1 AND tenant_id = $2
				""",
					communication.id, communication.tenant_id, communication.contact_id,
					communication.account_id, communication.lead_id, communication.opportunity_id,
					communication.communication_type.value, communication.direction.value,
					communication.status.value, communication.priority.value,
					communication.subject, communication.content, communication.summary,
					communication.from_address, communication.to_addresses, communication.cc_addresses,
					communication.bcc_addresses, communication.participants,
					communication.scheduled_at, communication.started_at, communication.ended_at,
					communication.duration_minutes,
					communication.outcome.value if communication.outcome else None,
					communication.outcome_notes, communication.follow_up_required,
					communication.follow_up_date, communication.follow_up_notes,
					communication.attachments, communication.external_id, communication.external_source,
					communication.tags, communication.metadata,
					communication.updated_at, communication.updated_by, communication.version
				)
			
			logger.info(f"✅ Communication updated successfully")
			return communication
			
		except Exception as e:
			logger.error(f"Failed to update communication: {str(e)}", exc_info=True)
			raise
	
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
			
			async with self.db_manager.get_connection() as conn:
				result = await conn.execute("""
					DELETE FROM crm_communications 
					WHERE id = $1 AND tenant_id = $2
				""", communication_id, tenant_id)
				
				deleted = result.split()[-1] == '1'
				
				if deleted:
					logger.info(f"✅ Communication deleted successfully")
				else:
					logger.warning(f"Communication not found for deletion: {communication_id}")
				
				return deleted
				
		except Exception as e:
			logger.error(f"Failed to delete communication: {str(e)}", exc_info=True)
			raise
	
	async def get_communication_analytics(
		self,
		tenant_id: str,
		contact_id: Optional[str] = None,
		account_id: Optional[str] = None,
		start_date: Optional[datetime] = None,
		end_date: Optional[datetime] = None
	) -> CommunicationAnalytics:
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
			
			# Build base conditions
			conditions = ["tenant_id = $1"]
			params = [tenant_id]
			param_count = 1
			
			if contact_id:
				param_count += 1
				conditions.append(f"contact_id = ${param_count}")
				params.append(contact_id)
			
			if account_id:
				param_count += 1
				conditions.append(f"account_id = ${param_count}")
				params.append(account_id)
			
			if start_date:
				param_count += 1
				conditions.append(f"created_at >= ${param_count}")
				params.append(start_date)
			
			if end_date:
				param_count += 1
				conditions.append(f"created_at <= ${param_count}")
				params.append(end_date)
			
			where_clause = " WHERE " + " AND ".join(conditions)
			
			async with self.db_manager.get_connection() as conn:
				# Get basic counts
				total_communications = await conn.fetchval(
					f"SELECT COUNT(*) FROM crm_communications{where_clause}",
					*params
				)
				
				# Get communications by type
				type_stats = await conn.fetch(f"""
					SELECT communication_type, COUNT(*) as count
					FROM crm_communications{where_clause}
					GROUP BY communication_type
				""", *params)
				
				# Get communications by direction
				direction_stats = await conn.fetch(f"""
					SELECT direction, COUNT(*) as count
					FROM crm_communications{where_clause}
					GROUP BY direction
				""", *params)
				
				# Get communications by outcome
				outcome_stats = await conn.fetch(f"""
					SELECT outcome, COUNT(*) as count
					FROM crm_communications{where_clause}
					AND outcome IS NOT NULL
					GROUP BY outcome
				""", *params)
				
				# Get timing analytics
				timing_stats = await conn.fetchrow(f"""
					SELECT 
						AVG(EXTRACT(EPOCH FROM (ended_at - started_at))/60) as avg_duration_minutes,
						SUM(duration_minutes) as total_duration_minutes
					FROM crm_communications{where_clause}
					AND duration_minutes IS NOT NULL
				""", *params)
				
				# Get communications by date (last 30 days)
				date_stats = await conn.fetch(f"""
					SELECT 
						DATE(created_at) as comm_date,
						COUNT(*) as count
					FROM crm_communications{where_clause}
					AND created_at >= NOW() - INTERVAL '30 days'
					GROUP BY DATE(created_at)
					ORDER BY comm_date
				""", *params)
				
				# Get follow-up analytics
				follow_up_stats = await conn.fetchrow(f"""
					SELECT 
						COUNT(*) FILTER (WHERE follow_up_required = true AND follow_up_date > NOW()) as pending_follow_ups,
						COUNT(*) FILTER (WHERE follow_up_required = true AND follow_up_date <= NOW()) as overdue_follow_ups
					FROM crm_communications{where_clause}
				""", *params)
			
			# Build analytics object
			analytics = CommunicationAnalytics(
				total_communications=total_communications,
				communications_by_type={row['communication_type']: row['count'] for row in type_stats},
				communications_by_direction={row['direction']: row['count'] for row in direction_stats},
				communications_by_outcome={row['outcome']: row['count'] for row in outcome_stats if row['outcome']},
				avg_duration_minutes=float(timing_stats['avg_duration_minutes']) if timing_stats['avg_duration_minutes'] else None,
				total_duration_minutes=int(timing_stats['total_duration_minutes']) if timing_stats['total_duration_minutes'] else 0,
				communications_by_date={str(row['comm_date']): row['count'] for row in date_stats},
				pending_follow_ups=follow_up_stats['pending_follow_ups'],
				overdue_follow_ups=follow_up_stats['overdue_follow_ups']
			)
			
			logger.info(f"✅ Generated analytics for {total_communications} communications")
			return analytics
			
		except Exception as e:
			logger.error(f"Failed to generate communication analytics: {str(e)}", exc_info=True)
			raise
	
	async def get_pending_follow_ups(
		self,
		tenant_id: str,
		user_id: Optional[str] = None,
		overdue_only: bool = False
	) -> List[Communication]:
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
			conditions = ["tenant_id = $1", "follow_up_required = true"]
			params = [tenant_id]
			param_count = 1
			
			if user_id:
				param_count += 1
				conditions.append(f"created_by = ${param_count}")
				params.append(user_id)
			
			if overdue_only:
				conditions.append("follow_up_date <= NOW()")
			else:
				conditions.append("follow_up_date IS NOT NULL")
			
			where_clause = " WHERE " + " AND ".join(conditions)
			
			async with self.db_manager.get_connection() as conn:
				rows = await conn.fetch(f"""
					SELECT * FROM crm_communications
					{where_clause}
					ORDER BY follow_up_date ASC NULLS LAST
				""", *params)
				
				return [Communication(**dict(row)) for row in rows]
				
		except Exception as e:
			logger.error(f"Failed to get pending follow-ups: {str(e)}", exc_info=True)
			raise
	
	async def _validate_related_entities(
		self,
		communication: Communication,
		tenant_id: str
	):
		"""
		Validate that related entities exist
		
		Args:
			communication: Communication to validate
			tenant_id: Tenant identifier
		"""
		async with self.db_manager.get_connection() as conn:
			# Validate contact
			if communication.contact_id:
				exists = await conn.fetchval(
					"SELECT EXISTS(SELECT 1 FROM crm_contacts WHERE id = $1 AND tenant_id = $2)",
					communication.contact_id, tenant_id
				)
				if not exists:
					raise ValueError(f"Contact not found: {communication.contact_id}")
			
			# Validate account
			if communication.account_id:
				exists = await conn.fetchval(
					"SELECT EXISTS(SELECT 1 FROM crm_accounts WHERE id = $1 AND tenant_id = $2)",
					communication.account_id, tenant_id
				)
				if not exists:
					raise ValueError(f"Account not found: {communication.account_id}")
			
			# Validate lead (if leads table exists)
			if communication.lead_id:
				try:
					exists = await conn.fetchval(
						"SELECT EXISTS(SELECT 1 FROM crm_leads WHERE id = $1 AND tenant_id = $2)",
						communication.lead_id, tenant_id
					)
					if not exists:
						raise ValueError(f"Lead not found: {communication.lead_id}")
				except:
					# Leads table might not exist yet
					pass
			
			# Validate opportunity (if opportunities table exists)
			if communication.opportunity_id:
				try:
					exists = await conn.fetchval(
						"SELECT EXISTS(SELECT 1 FROM crm_opportunities WHERE id = $1 AND tenant_id = $2)",
						communication.opportunity_id, tenant_id
					)
					if not exists:
						raise ValueError(f"Opportunity not found: {communication.opportunity_id}")
				except:
					# Opportunities table might not exist yet
					pass