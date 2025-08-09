"""
APG Customer Relationship Management - Contact Segmentation Module

Advanced contact segmentation system for creating dynamic contact segments
based on sophisticated criteria for targeted marketing and sales campaigns.

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

from .models import CRMContact, ContactType, LeadSource
from .database import DatabaseManager


logger = logging.getLogger(__name__)


class SegmentType(str, Enum):
	"""Types of contact segments"""
	STATIC = "static"  # Fixed list of contacts
	DYNAMIC = "dynamic"  # Rule-based segment that updates automatically
	SMART = "smart"  # AI-powered intelligent segment


class SegmentStatus(str, Enum):
	"""Status of contact segments"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	ARCHIVED = "archived"


class CriteriaOperator(str, Enum):
	"""Operators for segment criteria"""
	EQUALS = "equals"
	NOT_EQUALS = "not_equals"
	CONTAINS = "contains"
	NOT_CONTAINS = "not_contains"
	STARTS_WITH = "starts_with"
	ENDS_WITH = "ends_with"
	GREATER_THAN = "greater_than"
	LESS_THAN = "less_than"
	GREATER_EQUAL = "greater_equal"
	LESS_EQUAL = "less_equal"
	IN = "in"
	NOT_IN = "not_in"
	IS_NULL = "is_null"
	IS_NOT_NULL = "is_not_null"
	BETWEEN = "between"
	NOT_BETWEEN = "not_between"
	REGEX = "regex"


class LogicalOperator(str, Enum):
	"""Logical operators for combining criteria"""
	AND = "and"
	OR = "or"
	NOT = "not"


class SegmentCriteria(BaseModel):
	"""Individual segment criterion"""
	id: str = Field(default_factory=uuid7str)
	field: str = Field(..., description="Field name to filter on")
	operator: CriteriaOperator = Field(..., description="Comparison operator")
	value: Union[str, int, float, bool, List[Any], None] = Field(None, description="Value to compare against")
	values: Optional[List[Any]] = Field(None, description="Multiple values for IN/NOT_IN operators")
	case_sensitive: bool = Field(False, description="Case sensitive comparison for string fields")


class SegmentRule(BaseModel):
	"""Segment rule combining multiple criteria"""
	id: str = Field(default_factory=uuid7str)
	criteria: List[SegmentCriteria] = Field(default_factory=list)
	logical_operator: LogicalOperator = Field(LogicalOperator.AND, description="How to combine criteria")
	
	# Nested rules for complex logic
	nested_rules: List['SegmentRule'] = Field(default_factory=list)
	nested_logical_operator: LogicalOperator = Field(LogicalOperator.AND, description="How to combine nested rules")


class ContactSegment(BaseModel):
	"""Contact segment model"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	
	# Basic information
	name: str = Field(..., min_length=1, max_length=200)
	description: Optional[str] = Field(None, max_length=2000)
	segment_type: SegmentType = SegmentType.DYNAMIC
	status: SegmentStatus = SegmentStatus.ACTIVE
	
	# Segmentation rules (for dynamic/smart segments)
	rules: List[SegmentRule] = Field(default_factory=list)
	
	# Static contact list (for static segments)
	contact_ids: List[str] = Field(default_factory=list)
	
	# Metadata and settings
	auto_refresh: bool = Field(True, description="Automatically refresh dynamic segments")
	refresh_frequency_hours: int = Field(24, description="How often to refresh in hours")
	last_refreshed_at: Optional[datetime] = None
	
	# Performance tracking
	contact_count: int = Field(0, description="Current number of contacts in segment")
	estimated_count: Optional[int] = Field(None, description="Estimated contact count for performance")
	
	# Usage tracking
	usage_count: int = Field(0, description="Number of times segment has been used")
	last_used_at: Optional[datetime] = None
	
	# Categorization
	category: Optional[str] = Field(None, description="Segment category")
	tags: List[str] = Field(default_factory=list)
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_by: str
	version: int = 1


class SegmentAnalytics(BaseModel):
	"""Analytics data for contact segments"""
	segment_id: str
	segment_name: str
	
	# Contact metrics
	total_contacts: int = 0
	active_contacts: int = 0
	inactive_contacts: int = 0
	
	# Demographic breakdown
	contact_types: Dict[str, int] = Field(default_factory=dict)
	lead_sources: Dict[str, int] = Field(default_factory=dict)
	locations: Dict[str, int] = Field(default_factory=dict)
	industries: Dict[str, int] = Field(default_factory=dict)
	
	# Engagement metrics
	avg_engagement_score: Optional[float] = None
	high_value_contacts: int = 0
	recent_interactions: int = 0
	
	# Growth metrics
	contacts_added_last_30_days: int = 0
	contacts_removed_last_30_days: int = 0
	growth_rate_percentage: Optional[float] = None
	
	# Performance tracking
	last_analysis_at: datetime = Field(default_factory=datetime.utcnow)
	analysis_duration_ms: Optional[int] = None


class ContactSegmentationManager:
	"""
	Advanced contact segmentation management system
	
	Provides sophisticated contact segmentation capabilities with dynamic
	rule-based segments, analytics, and automated refresh functionality.
	"""
	
	def __init__(self, db_manager: DatabaseManager):
		"""
		Initialize segmentation manager
		
		Args:
			db_manager: Database manager instance
		"""
		self.db_manager = db_manager
		self._initialized = False
	
	async def initialize(self):
		"""Initialize the segmentation manager"""
		if self._initialized:
			return
		
		logger.info("🔧 Initializing Contact Segmentation Manager...")
		
		# Ensure database connection
		if not self.db_manager._initialized:
			await self.db_manager.initialize()
		
		self._initialized = True
		logger.info("✅ Contact Segmentation Manager initialized successfully")
	
	async def create_segment(
		self,
		segment_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> ContactSegment:
		"""
		Create a new contact segment
		
		Args:
			segment_data: Segment configuration data
			tenant_id: Tenant identifier
			created_by: User creating the segment
			
		Returns:
			Created segment
		"""
		try:
			logger.info(f"📊 Creating contact segment: {segment_data.get('name')}")
			
			# Add required fields
			segment_data.update({
				'tenant_id': tenant_id,
				'created_by': created_by,
				'updated_by': created_by
			})
			
			# Create segment object
			segment = ContactSegment(**segment_data)
			
			# For dynamic segments, calculate initial contact count
			if segment.segment_type == SegmentType.DYNAMIC:
				segment.contact_count = await self._calculate_segment_size(segment, tenant_id)
				segment.last_refreshed_at = datetime.utcnow()
			elif segment.segment_type == SegmentType.STATIC:
				segment.contact_count = len(segment.contact_ids)
			
			# Store in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_contact_segments (
						id, tenant_id, name, description, segment_type, status,
						rules, contact_ids, auto_refresh, refresh_frequency_hours,
						last_refreshed_at, contact_count, estimated_count,
						usage_count, last_used_at, category, tags, metadata,
						created_at, updated_at, created_by, updated_by, version
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
						$11, $12, $13, $14, $15, $16, $17, $18,
						$19, $20, $21, $22, $23
					)
				""", 
					segment.id, segment.tenant_id, segment.name, segment.description,
					segment.segment_type.value, segment.status.value,
					[rule.model_dump() for rule in segment.rules], segment.contact_ids,
					segment.auto_refresh, segment.refresh_frequency_hours,
					segment.last_refreshed_at, segment.contact_count, segment.estimated_count,
					segment.usage_count, segment.last_used_at, segment.category,
					segment.tags, segment.metadata,
					segment.created_at, segment.updated_at,
					segment.created_by, segment.updated_by, segment.version
				)
			
			logger.info(f"✅ Contact segment created successfully: {segment.id}")
			return segment
			
		except Exception as e:
			logger.error(f"Failed to create contact segment: {str(e)}", exc_info=True)
			raise
	
	async def get_segment(
		self,
		segment_id: str,
		tenant_id: str
	) -> Optional[ContactSegment]:
		"""
		Get segment by ID
		
		Args:
			segment_id: Segment identifier
			tenant_id: Tenant identifier
			
		Returns:
			Segment if found
		"""
		try:
			async with self.db_manager.get_connection() as conn:
				row = await conn.fetchrow("""
					SELECT * FROM crm_contact_segments 
					WHERE id = $1 AND tenant_id = $2
				""", segment_id, tenant_id)
				
				if not row:
					return None
				
				# Convert row to dict and handle nested objects
				segment_dict = dict(row)
				
				# Parse rules from JSON
				if segment_dict['rules']:
					segment_dict['rules'] = [SegmentRule(**rule) for rule in segment_dict['rules']]
				
				return ContactSegment(**segment_dict)
				
		except Exception as e:
			logger.error(f"Failed to get contact segment: {str(e)}", exc_info=True)
			raise
	
	async def list_segments(
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
		List segments with filtering
		
		Args:
			tenant_id: Tenant identifier
			segment_type: Filter by segment type
			status: Filter by status
			category: Filter by category
			tags: Filter by tags
			limit: Maximum results
			offset: Results offset
			
		Returns:
			Dict containing segments and pagination info
		"""
		try:
			# Build query conditions
			conditions = ["tenant_id = $1"]
			params = [tenant_id]
			param_count = 1
			
			if segment_type:
				param_count += 1
				conditions.append(f"segment_type = ${param_count}")
				params.append(segment_type.value)
			
			if status:
				param_count += 1
				conditions.append(f"status = ${param_count}")
				params.append(status.value)
			
			if category:
				param_count += 1
				conditions.append(f"category = ${param_count}")
				params.append(category)
			
			if tags:
				param_count += 1
				conditions.append(f"tags && ${param_count}")
				params.append(tags)
			
			where_clause = " WHERE " + " AND ".join(conditions)
			
			async with self.db_manager.get_connection() as conn:
				# Get total count
				count_query = f"SELECT COUNT(*) FROM crm_contact_segments{where_clause}"
				total = await conn.fetchval(count_query, *params)
				
				# Get segments
				query = f"""
					SELECT * FROM crm_contact_segments
					{where_clause}
					ORDER BY updated_at DESC
					LIMIT {limit} OFFSET {offset}
				"""
				
				rows = await conn.fetch(query, *params)
				segments = []
				
				for row in rows:
					segment_dict = dict(row)
					# Parse rules from JSON
					if segment_dict['rules']:
						segment_dict['rules'] = [SegmentRule(**rule) for rule in segment_dict['rules']]
					segments.append(ContactSegment(**segment_dict))
			
			return {
				"segments": segments,
				"total": total,
				"limit": limit,
				"offset": offset
			}
			
		except Exception as e:
			logger.error(f"Failed to list contact segments: {str(e)}", exc_info=True)
			raise
	
	async def update_segment(
		self,
		segment_id: str,
		update_data: Dict[str, Any],
		tenant_id: str,
		updated_by: str
	) -> ContactSegment:
		"""
		Update segment
		
		Args:
			segment_id: Segment identifier
			update_data: Fields to update
			tenant_id: Tenant identifier
			updated_by: User making the update
			
		Returns:
			Updated segment
		"""
		try:
			logger.info(f"📝 Updating contact segment: {segment_id}")
			
			# Get existing segment
			segment = await self.get_segment(segment_id, tenant_id)
			if not segment:
				raise ValueError(f"Segment not found: {segment_id}")
			
			# Update fields
			update_data['updated_by'] = updated_by
			update_data['updated_at'] = datetime.utcnow()
			update_data['version'] = segment.version + 1
			
			# Apply updates
			for field, value in update_data.items():
				if hasattr(segment, field):
					if field == 'rules' and value:
						# Parse rules if provided
						setattr(segment, field, [SegmentRule(**rule) for rule in value])
					else:
						setattr(segment, field, value)
			
			# Recalculate segment size if rules changed
			if 'rules' in update_data and segment.segment_type == SegmentType.DYNAMIC:
				segment.contact_count = await self._calculate_segment_size(segment, tenant_id)
				segment.last_refreshed_at = datetime.utcnow()
			elif 'contact_ids' in update_data and segment.segment_type == SegmentType.STATIC:
				segment.contact_count = len(segment.contact_ids)
			
			# Update in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					UPDATE crm_contact_segments SET
						name = $3, description = $4, segment_type = $5, status = $6,
						rules = $7, contact_ids = $8, auto_refresh = $9, 
						refresh_frequency_hours = $10, last_refreshed_at = $11, 
						contact_count = $12, estimated_count = $13, usage_count = $14,
						last_used_at = $15, category = $16, tags = $17, metadata = $18,
						updated_at = $19, updated_by = $20, version = $21
					WHERE id = $1 AND tenant_id = $2
				""",
					segment.id, segment.tenant_id, segment.name, segment.description,
					segment.segment_type.value, segment.status.value,
					[rule.model_dump() for rule in segment.rules], segment.contact_ids,
					segment.auto_refresh, segment.refresh_frequency_hours,
					segment.last_refreshed_at, segment.contact_count, segment.estimated_count,
					segment.usage_count, segment.last_used_at, segment.category,
					segment.tags, segment.metadata,
					segment.updated_at, segment.updated_by, segment.version
				)
			
			logger.info(f"✅ Contact segment updated successfully")
			return segment
			
		except Exception as e:
			logger.error(f"Failed to update contact segment: {str(e)}", exc_info=True)
			raise
	
	async def delete_segment(
		self,
		segment_id: str,
		tenant_id: str
	) -> bool:
		"""
		Delete segment
		
		Args:
			segment_id: Segment identifier
			tenant_id: Tenant identifier
			
		Returns:
			True if deleted successfully
		"""
		try:
			logger.info(f"🗑️ Deleting contact segment: {segment_id}")
			
			async with self.db_manager.get_connection() as conn:
				result = await conn.execute("""
					DELETE FROM crm_contact_segments 
					WHERE id = $1 AND tenant_id = $2
				""", segment_id, tenant_id)
				
				deleted = result.split()[-1] == '1'
				
				if deleted:
					logger.info(f"✅ Contact segment deleted successfully")
				else:
					logger.warning(f"Contact segment not found for deletion: {segment_id}")
				
				return deleted
				
		except Exception as e:
			logger.error(f"Failed to delete contact segment: {str(e)}", exc_info=True)
			raise
	
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
			Dict containing contacts and pagination info
		"""
		try:
			logger.info(f"👥 Getting contacts for segment: {segment_id}")
			
			# Get segment
			segment = await self.get_segment(segment_id, tenant_id)
			if not segment:
				raise ValueError(f"Segment not found: {segment_id}")
			
			# Update usage tracking
			segment.usage_count += 1
			segment.last_used_at = datetime.utcnow()
			await self._update_segment_usage(segment_id, tenant_id, segment.usage_count, segment.last_used_at)
			
			async with self.db_manager.get_connection() as conn:
				if segment.segment_type == SegmentType.STATIC:
					# For static segments, use the contact_ids list
					if not segment.contact_ids:
						return {"contacts": [], "total": 0, "limit": limit, "offset": offset}
					
					# Get contacts by IDs
					contacts_query = """
						SELECT c.* FROM crm_contacts c
						WHERE c.id = ANY($1) AND c.tenant_id = $2
						AND c.status = 'active'
						ORDER BY c.updated_at DESC
						LIMIT $3 OFFSET $4
					"""
					
					contacts = await conn.fetch(contacts_query, segment.contact_ids, tenant_id, limit, offset)
					total = len(segment.contact_ids)
					
				else:
					# For dynamic segments, execute the rules
					contacts_query, query_params = self._build_segment_query(segment, tenant_id, limit, offset)
					contacts = await conn.fetch(contacts_query, *query_params)
					
					# Get total count
					count_query, count_params = self._build_segment_query(segment, tenant_id, count_only=True)
					total = await conn.fetchval(count_query, *count_params)
			
			# Convert to contact objects
			contact_list = []
			for row in contacts:
				contact_dict = dict(row)
				# Handle enum conversions
				if contact_dict.get('contact_type'):
					contact_dict['contact_type'] = ContactType(contact_dict['contact_type'])
				if contact_dict.get('lead_source'):
					contact_dict['lead_source'] = LeadSource(contact_dict['lead_source'])
				contact_list.append(CRMContact(**contact_dict))
			
			return {
				"contacts": contact_list,
				"total": total,
				"limit": limit,
				"offset": offset
			}
			
		except Exception as e:
			logger.error(f"Failed to get segment contacts: {str(e)}", exc_info=True)
			raise
	
	async def refresh_segment(
		self,
		segment_id: str,
		tenant_id: str
	) -> ContactSegment:
		"""
		Refresh a dynamic segment's contact count
		
		Args:
			segment_id: Segment identifier
			tenant_id: Tenant identifier
			
		Returns:
			Updated segment
		"""
		try:
			logger.info(f"🔄 Refreshing segment: {segment_id}")
			
			# Get segment
			segment = await self.get_segment(segment_id, tenant_id)
			if not segment:
				raise ValueError(f"Segment not found: {segment_id}")
			
			if segment.segment_type != SegmentType.DYNAMIC:
				logger.warning(f"Cannot refresh non-dynamic segment: {segment_id}")
				return segment
			
			# Recalculate contact count
			new_count = await self._calculate_segment_size(segment, tenant_id)
			
			# Update segment
			update_data = {
				'contact_count': new_count,
				'last_refreshed_at': datetime.utcnow()
			}
			
			return await self.update_segment(
				segment_id=segment_id,
				update_data=update_data,
				tenant_id=tenant_id,
				updated_by="system"
			)
			
		except Exception as e:
			logger.error(f"Failed to refresh segment: {str(e)}", exc_info=True)
			raise
	
	async def get_segment_analytics(
		self,
		segment_id: str,
		tenant_id: str
	) -> SegmentAnalytics:
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
			
			start_time = datetime.utcnow()
			
			# Get segment
			segment = await self.get_segment(segment_id, tenant_id)
			if not segment:
				raise ValueError(f"Segment not found: {segment_id}")
			
			async with self.db_manager.get_connection() as conn:
				# Get contacts in segment for analysis
				if segment.segment_type == SegmentType.STATIC:
					if not segment.contact_ids:
						return SegmentAnalytics(
							segment_id=segment_id,
							segment_name=segment.name
						)
					
					analytics_query = """
						SELECT 
							COUNT(*) as total_contacts,
							COUNT(*) FILTER (WHERE status = 'active') as active_contacts,
							COUNT(*) FILTER (WHERE status != 'active') as inactive_contacts,
							contact_type,
							lead_source
						FROM crm_contacts
						WHERE id = ANY($1) AND tenant_id = $2
						GROUP BY contact_type, lead_source
					"""
					
					analytics_rows = await conn.fetch(analytics_query, segment.contact_ids, tenant_id)
					
				else:
					# For dynamic segments, use the rules
					analytics_query, query_params = self._build_segment_analytics_query(segment, tenant_id)
					analytics_rows = await conn.fetch(analytics_query, *query_params)
			
			# Process analytics data
			analytics = SegmentAnalytics(
				segment_id=segment_id,
				segment_name=segment.name
			)
			
			for row in analytics_rows:
				analytics.total_contacts += row.get('total_contacts', 0)
				analytics.active_contacts += row.get('active_contacts', 0)
				analytics.inactive_contacts += row.get('inactive_contacts', 0)
				
				# Build breakdowns
				if row.get('contact_type'):
					contact_type = row['contact_type']
					analytics.contact_types[contact_type] = analytics.contact_types.get(contact_type, 0) + row.get('total_contacts', 0)
				
				if row.get('lead_source'):
					lead_source = row['lead_source']
					analytics.lead_sources[lead_source] = analytics.lead_sources.get(lead_source, 0) + row.get('total_contacts', 0)
			
			# Calculate duration
			end_time = datetime.utcnow()
			analytics.analysis_duration_ms = int((end_time - start_time).total_seconds() * 1000)
			analytics.last_analysis_at = end_time
			
			logger.info(f"✅ Generated analytics for {analytics.total_contacts} contacts")
			return analytics
			
		except Exception as e:
			logger.error(f"Failed to generate segment analytics: {str(e)}", exc_info=True)
			raise
	
	async def _calculate_segment_size(
		self,
		segment: ContactSegment,
		tenant_id: str
	) -> int:
		"""Calculate the number of contacts in a segment"""
		try:
			if segment.segment_type == SegmentType.STATIC:
				return len(segment.contact_ids)
			
			if not segment.rules:
				return 0
			
			# Build and execute count query
			query, params = self._build_segment_query(segment, tenant_id, count_only=True)
			
			async with self.db_manager.get_connection() as conn:
				count = await conn.fetchval(query, *params)
				return count or 0
				
		except Exception as e:
			logger.error(f"Failed to calculate segment size: {str(e)}", exc_info=True)
			return 0
	
	def _build_segment_query(
		self,
		segment: ContactSegment,
		tenant_id: str,
		limit: Optional[int] = None,
		offset: Optional[int] = None,
		count_only: bool = False
	) -> Tuple[str, List[Any]]:
		"""Build SQL query for segment rules"""
		
		base_select = "SELECT COUNT(*)" if count_only else "SELECT c.*"
		
		query = f"""
			{base_select}
			FROM crm_contacts c
			WHERE c.tenant_id = $1 AND c.status = 'active'
		"""
		
		params = [tenant_id]
		param_count = 1
		
		# Build conditions from rules
		if segment.rules:
			conditions = []
			for rule in segment.rules:
				rule_condition, rule_params = self._build_rule_condition(rule, param_count)
				if rule_condition:
					conditions.append(rule_condition)
					params.extend(rule_params)
					param_count += len(rule_params)
			
			if conditions:
				# Combine rule conditions with AND (can be made configurable)
				query += " AND (" + " AND ".join(conditions) + ")"
		
		if not count_only:
			query += " ORDER BY c.updated_at DESC"
			if limit:
				query += f" LIMIT {limit}"
			if offset:
				query += f" OFFSET {offset}"
		
		return query, params
	
	def _build_rule_condition(
		self,
		rule: SegmentRule,
		start_param_count: int
	) -> Tuple[str, List[Any]]:
		"""Build SQL condition for a single rule"""
		
		if not rule.criteria:
			return "", []
		
		conditions = []
		params = []
		param_count = start_param_count
		
		for criteria in rule.criteria:
			condition, criteria_params = self._build_criteria_condition(criteria, param_count)
			if condition:
				conditions.append(condition)
				params.extend(criteria_params)
				param_count += len(criteria_params)
		
		if not conditions:
			return "", []
		
		# Combine criteria with the rule's logical operator
		operator = " AND " if rule.logical_operator == LogicalOperator.AND else " OR "
		combined_condition = f"({operator.join(conditions)})"
		
		# Handle nested rules (simplified for now)
		if rule.nested_rules:
			nested_conditions = []
			for nested_rule in rule.nested_rules:
				nested_condition, nested_params = self._build_rule_condition(nested_rule, param_count)
				if nested_condition:
					nested_conditions.append(nested_condition)
					params.extend(nested_params)
					param_count += len(nested_params)
			
			if nested_conditions:
				nested_operator = " AND " if rule.nested_logical_operator == LogicalOperator.AND else " OR "
				nested_combined = f"({nested_operator.join(nested_conditions)})"
				combined_condition = f"({combined_condition} AND {nested_combined})"
		
		return combined_condition, params
	
	def _build_criteria_condition(
		self,
		criteria: SegmentCriteria,
		param_count: int
	) -> Tuple[str, List[Any]]:
		"""Build SQL condition for a single criterion"""
		
		field = f"c.{criteria.field}"
		operator = criteria.operator
		value = criteria.value
		values = criteria.values or []
		
		params = []
		
		if operator == CriteriaOperator.EQUALS:
			param_count += 1
			condition = f"{field} = ${param_count}"
			params.append(value)
			
		elif operator == CriteriaOperator.NOT_EQUALS:
			param_count += 1
			condition = f"{field} != ${param_count}"
			params.append(value)
			
		elif operator == CriteriaOperator.CONTAINS:
			param_count += 1
			if criteria.case_sensitive:
				condition = f"{field} LIKE ${param_count}"
			else:
				condition = f"LOWER({field}) LIKE LOWER(${param_count})"
			params.append(f"%{value}%")
			
		elif operator == CriteriaOperator.NOT_CONTAINS:
			param_count += 1
			if criteria.case_sensitive:
				condition = f"{field} NOT LIKE ${param_count}"
			else:
				condition = f"LOWER({field}) NOT LIKE LOWER(${param_count})"
			params.append(f"%{value}%")
			
		elif operator == CriteriaOperator.STARTS_WITH:
			param_count += 1
			if criteria.case_sensitive:
				condition = f"{field} LIKE ${param_count}"
			else:
				condition = f"LOWER({field}) LIKE LOWER(${param_count})"
			params.append(f"{value}%")
			
		elif operator == CriteriaOperator.ENDS_WITH:
			param_count += 1
			if criteria.case_sensitive:
				condition = f"{field} LIKE ${param_count}"
			else:
				condition = f"LOWER({field}) LIKE LOWER(${param_count})"
			params.append(f"%{value}")
			
		elif operator == CriteriaOperator.GREATER_THAN:
			param_count += 1
			condition = f"{field} > ${param_count}"
			params.append(value)
			
		elif operator == CriteriaOperator.LESS_THAN:
			param_count += 1
			condition = f"{field} < ${param_count}"
			params.append(value)
			
		elif operator == CriteriaOperator.GREATER_EQUAL:
			param_count += 1
			condition = f"{field} >= ${param_count}"
			params.append(value)
			
		elif operator == CriteriaOperator.LESS_EQUAL:
			param_count += 1
			condition = f"{field} <= ${param_count}"
			params.append(value)
			
		elif operator == CriteriaOperator.IN:
			if values:
				placeholders = []
				for val in values:
					param_count += 1
					placeholders.append(f"${param_count}")
					params.append(val)
				condition = f"{field} IN ({', '.join(placeholders)})"
			else:
				condition = "FALSE"  # No values provided
				
		elif operator == CriteriaOperator.NOT_IN:
			if values:
				placeholders = []
				for val in values:
					param_count += 1
					placeholders.append(f"${param_count}")
					params.append(val)
				condition = f"{field} NOT IN ({', '.join(placeholders)})"
			else:
				condition = "TRUE"  # No values to exclude
				
		elif operator == CriteriaOperator.IS_NULL:
			condition = f"{field} IS NULL"
			
		elif operator == CriteriaOperator.IS_NOT_NULL:
			condition = f"{field} IS NOT NULL"
			
		elif operator == CriteriaOperator.BETWEEN:
			if isinstance(value, list) and len(value) == 2:
				param_count += 2
				condition = f"{field} BETWEEN ${param_count-1} AND ${param_count}"
				params.extend(value)
			else:
				condition = "TRUE"  # Invalid between values
				
		elif operator == CriteriaOperator.NOT_BETWEEN:
			if isinstance(value, list) and len(value) == 2:
				param_count += 2
				condition = f"{field} NOT BETWEEN ${param_count-1} AND ${param_count}"
				params.extend(value)
			else:
				condition = "TRUE"  # Invalid between values
				
		elif operator == CriteriaOperator.REGEX:
			param_count += 1
			condition = f"{field} ~ ${param_count}"
			params.append(value)
			
		else:
			logger.warning(f"Unknown operator: {operator}")
			condition = "TRUE"
		
		return condition, params
	
	def _build_segment_analytics_query(
		self,
		segment: ContactSegment,
		tenant_id: str
	) -> Tuple[str, List[Any]]:
		"""Build analytics query for segment"""
		
		query = """
			SELECT 
				COUNT(*) as total_contacts,
				COUNT(*) FILTER (WHERE c.status = 'active') as active_contacts,
				COUNT(*) FILTER (WHERE c.status != 'active') as inactive_contacts,
				c.contact_type,
				c.lead_source
			FROM crm_contacts c
			WHERE c.tenant_id = $1 AND c.status = 'active'
		"""
		
		params = [tenant_id]
		param_count = 1
		
		# Add segment rules
		if segment.rules:
			conditions = []
			for rule in segment.rules:
				rule_condition, rule_params = self._build_rule_condition(rule, param_count)
				if rule_condition:
					conditions.append(rule_condition)
					params.extend(rule_params)
					param_count += len(rule_params)
			
			if conditions:
				query += " AND (" + " AND ".join(conditions) + ")"
		
		query += " GROUP BY c.contact_type, c.lead_source"
		
		return query, params
	
	async def _update_segment_usage(
		self,
		segment_id: str,
		tenant_id: str,
		usage_count: int,
		last_used_at: datetime
	):
		"""Update segment usage statistics"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					UPDATE crm_contact_segments 
					SET usage_count = $3, last_used_at = $4, updated_at = NOW()
					WHERE id = $1 AND tenant_id = $2
				""", segment_id, tenant_id, usage_count, last_used_at)
		except Exception as e:
			logger.error(f"Failed to update segment usage: {str(e)}", exc_info=True)
			# Don't raise as this is not critical