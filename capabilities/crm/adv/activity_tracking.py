"""
APG Customer Relationship Management - Activity Tracking Module

Advanced contact activity tracking system for comprehensive interaction history,
task management, and engagement analytics.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ValidationError

from .models import CRMActivity, ActivityType, ActivityStatus, Priority
from .database import DatabaseManager


logger = logging.getLogger(__name__)


class ActivityCategory(str, Enum):
	"""Categories for activities"""
	COMMUNICATION = "communication"
	MEETING = "meeting"
	TASK = "task"
	FOLLOW_UP = "follow_up"
	SALES = "sales"
	SUPPORT = "support"
	MARKETING = "marketing"
	ADMINISTRATIVE = "administrative"


class EngagementLevel(str, Enum):
	"""Levels of engagement in activities"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


class ActivityOutcome(str, Enum):
	"""Possible outcomes of activities"""
	SUCCESSFUL = "successful"
	PARTIAL = "partial"
	UNSUCCESSFUL = "unsuccessful"
	RESCHEDULED = "rescheduled"
	CANCELLED = "cancelled"
	NO_RESPONSE = "no_response"


class ContactActivityTracker:
	"""
	Comprehensive contact activity tracking system with advanced analytics
	and intelligent task management capabilities.
	"""
	
	def __init__(self, database_manager: DatabaseManager):
		"""
		Initialize activity tracker
		
		Args:
			database_manager: Database manager instance
		"""
		self.db_manager = database_manager
		self.activity_rules = self._load_activity_rules()
	
	def _load_activity_rules(self) -> Dict[str, Any]:
		"""Load activity scoring and automation rules"""
		return {
			"engagement_scoring": {
				"email_reply": 2,
				"phone_call_answered": 5,
				"meeting_attended": 8,
				"proposal_viewed": 6,
				"contract_signed": 10
			},
			"follow_up_rules": {
				"no_response_days": 3,
				"meeting_follow_up_hours": 24,
				"proposal_follow_up_days": 7
			},
			"priority_escalation": {
				"overdue_days": 2,
				"high_value_threshold": 50000
			}
		}
	
	# ================================
	# Core Activity Operations
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
			
			# Set default values
			if "id" not in activity_data:
				activity_data["id"] = uuid7str()
			
			if "status" not in activity_data:
				activity_data["status"] = ActivityStatus.PLANNED
			
			if "priority" not in activity_data:
				activity_data["priority"] = Priority.MEDIUM
			
			# Validate activity
			activity = CRMActivity(**activity_data)
			
			# Verify contact exists
			if activity.contact_id:
				contact = await self.db_manager.get_contact(activity.contact_id, tenant_id)
				if not contact:
					raise ValueError(f"Contact not found: {activity.contact_id}")
			
			# Save to database
			saved_activity = await self._save_activity(activity)
			
			# Schedule automatic follow-ups if applicable
			asyncio.create_task(
				self._schedule_automatic_follow_ups(saved_activity)
			)
			
			logger.info(f"Created activity: {saved_activity.id}")
			return saved_activity
			
		except ValidationError as e:
			raise ValueError(f"Invalid activity data: {str(e)}")
		except Exception as e:
			logger.error(f"Failed to create activity: {str(e)}", exc_info=True)
			raise Exception(f"Activity creation failed: {str(e)}")
	
	async def get_activity(
		self,
		activity_id: str,
		tenant_id: str
	) -> Optional[CRMActivity]:
		"""
		Get activity by ID
		
		Args:
			activity_id: Activity identifier
			tenant_id: Tenant identifier
			
		Returns:
			Activity object or None
		"""
		try:
			async with self.db_manager.get_connection() as conn:
				row = await conn.fetchrow("""
					SELECT * FROM crm_activities 
					WHERE id = $1 AND tenant_id = $2
				""", activity_id, tenant_id)
				
				if row:
					return self._row_to_activity(row)
				return None
				
		except Exception as e:
			logger.error(f"Failed to get activity: {str(e)}", exc_info=True)
			raise Exception(f"Get activity failed: {str(e)}")
	
	async def get_contact_activities(
		self,
		contact_id: str,
		tenant_id: str,
		activity_type: Optional[ActivityType] = None,
		status: Optional[ActivityStatus] = None,
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
			async with self.db_manager.get_connection() as conn:
				# Build query
				query = "SELECT * FROM crm_activities WHERE tenant_id = $1 AND contact_id = $2"
				params = [tenant_id, contact_id]
				param_counter = 3
				
				if activity_type:
					query += f" AND activity_type = ${param_counter}"
					params.append(activity_type.value)
					param_counter += 1
				
				if status:
					query += f" AND status = ${param_counter}"
					params.append(status.value)
					param_counter += 1
				
				if start_date:
					query += f" AND created_at >= ${param_counter}"
					params.append(start_date)
					param_counter += 1
				
				if end_date:
					query += f" AND created_at <= ${param_counter}"
					params.append(end_date)
					param_counter += 1
				
				# Get total count
				count_query = query.replace("SELECT *", "SELECT COUNT(*)")
				total_count = await conn.fetchval(count_query, *params)
				
				# Add ordering and pagination
				query += f" ORDER BY created_at DESC LIMIT ${param_counter} OFFSET ${param_counter + 1}"
				params.extend([limit, offset])
				
				rows = await conn.fetch(query, *params)
				
				activities = [self._row_to_activity(row) for row in rows]
				
				return {
					"items": [activity.model_dump() for activity in activities],
					"total": total_count,
					"limit": limit,
					"offset": offset,
					"has_more": offset + len(activities) < total_count
				}
				
		except Exception as e:
			logger.error(f"Failed to get contact activities: {str(e)}", exc_info=True)
			raise Exception(f"Get contact activities failed: {str(e)}")
	
	async def update_activity(
		self,
		activity_id: str,
		update_data: Dict[str, Any],
		tenant_id: str,
		updated_by: str
	) -> CRMActivity:
		"""
		Update an existing activity
		
		Args:
			activity_id: Activity identifier
			update_data: Fields to update
			tenant_id: Tenant identifier
			updated_by: User updating the activity
			
		Returns:
			Updated activity object
		"""
		try:
			# Get existing activity
			existing = await self.get_activity(activity_id, tenant_id)
			if not existing:
				raise ValueError(f"Activity not found: {activity_id}")
			
			# Update fields
			update_data.update({
				"updated_by": updated_by,
				"updated_at": datetime.utcnow(),
				"version": existing.version + 1
			})
			
			# Apply updates
			existing_dict = existing.model_dump()
			existing_dict.update(update_data)
			
			# Validate updated activity
			updated_activity = CRMActivity(**existing_dict)
			
			# Save updated activity
			saved_activity = await self._save_activity(updated_activity)
			
			logger.info(f"Updated activity: {activity_id}")
			return saved_activity
			
		except Exception as e:
			logger.error(f"Failed to update activity: {str(e)}", exc_info=True)
			raise Exception(f"Activity update failed: {str(e)}")
	
	async def complete_activity(
		self,
		activity_id: str,
		outcome: ActivityOutcome,
		outcome_notes: Optional[str],
		tenant_id: str,
		completed_by: str
	) -> CRMActivity:
		"""
		Mark activity as completed
		
		Args:
			activity_id: Activity identifier
			outcome: Activity outcome
			outcome_notes: Notes about the outcome
			tenant_id: Tenant identifier
			completed_by: User completing the activity
			
		Returns:
			Completed activity object
		"""
		try:
			# Update activity status
			update_data = {
				"status": ActivityStatus.COMPLETED,
				"completed_at": datetime.utcnow(),
				"outcome": outcome.value,
				"outcome_summary": outcome_notes
			}
			
			completed_activity = await self.update_activity(
				activity_id, update_data, tenant_id, completed_by
			)
			
			# Update engagement scoring
			await self._update_engagement_score(completed_activity)
			
			# Generate follow-up activities if needed
			await self._generate_follow_up_activities(completed_activity, completed_by)
			
			logger.info(f"Completed activity: {activity_id} with outcome: {outcome.value}")
			return completed_activity
			
		except Exception as e:
			logger.error(f"Failed to complete activity: {str(e)}", exc_info=True)
			raise Exception(f"Activity completion failed: {str(e)}")
	
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
			async with self.db_manager.get_connection() as conn:
				result = await conn.execute("""
					DELETE FROM crm_activities 
					WHERE id = $1 AND tenant_id = $2
				""", activity_id, tenant_id)
				
				deleted = result.split()[-1] == '1'
				
				if deleted:
					logger.info(f"Deleted activity: {activity_id}")
				
				return deleted
				
		except Exception as e:
			logger.error(f"Failed to delete activity: {str(e)}", exc_info=True)
			raise Exception(f"Activity deletion failed: {str(e)}")
	
	# ================================
	# Activity Analytics
	# ================================
	
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
			if not start_date:
				start_date = datetime.utcnow() - timedelta(days=30)
			if not end_date:
				end_date = datetime.utcnow()
			
			analytics = {
				"period": {
					"start_date": start_date.isoformat(),
					"end_date": end_date.isoformat()
				},
				"total_activities": 0,
				"activity_types": {},
				"activity_status": {},
				"activity_outcomes": {},
				"engagement_trends": [],
				"productivity_metrics": {},
				"overdue_activities": 0
			}
			
			async with self.db_manager.get_connection() as conn:
				# Base query conditions
				base_conditions = "tenant_id = $1 AND created_at BETWEEN $2 AND $3"
				base_params = [tenant_id, start_date, end_date]
				
				if contact_id:
					base_conditions += " AND contact_id = $4"
					base_params.append(contact_id)
				
				# Get total activities
				total_count = await conn.fetchval(f"""
					SELECT COUNT(*) FROM crm_activities 
					WHERE {base_conditions}
				""", *base_params)
				analytics["total_activities"] = total_count
				
				# Get activity type breakdown
				type_breakdown = await conn.fetch(f"""
					SELECT activity_type, COUNT(*) as count
					FROM crm_activities 
					WHERE {base_conditions}
					GROUP BY activity_type
					ORDER BY count DESC
				""", *base_params)
				
				analytics["activity_types"] = {
					row['activity_type']: row['count'] 
					for row in type_breakdown
				}
				
				# Get status breakdown
				status_breakdown = await conn.fetch(f"""
					SELECT status, COUNT(*) as count
					FROM crm_activities 
					WHERE {base_conditions}
					GROUP BY status
				""", *base_params)
				
				analytics["activity_status"] = {
					row['status']: row['count'] 
					for row in status_breakdown
				}
				
				# Get outcome breakdown (for completed activities)
				outcome_breakdown = await conn.fetch(f"""
					SELECT outcome, COUNT(*) as count
					FROM crm_activities 
					WHERE {base_conditions} AND outcome IS NOT NULL
					GROUP BY outcome
				""", *base_params)
				
				analytics["activity_outcomes"] = {
					row['outcome']: row['count'] 
					for row in outcome_breakdown
				}
				
				# Get overdue activities
				overdue_count = await conn.fetchval(f"""
					SELECT COUNT(*) FROM crm_activities 
					WHERE {base_conditions} 
					AND status IN ('planned', 'in_progress') 
					AND due_date < NOW()
				""", *base_params)
				analytics["overdue_activities"] = overdue_count
				
				# Get engagement trends (weekly)
				engagement_trends = await conn.fetch(f"""
					SELECT 
						DATE_TRUNC('week', created_at) as week,
						COUNT(*) as activity_count,
						COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_count,
						AVG(CASE WHEN engagement_score IS NOT NULL THEN engagement_score END) as avg_engagement
					FROM crm_activities 
					WHERE {base_conditions}
					GROUP BY week
					ORDER BY week
				""", *base_params)
				
				analytics["engagement_trends"] = [
					{
						"week": row['week'].isoformat(),
						"activity_count": row['activity_count'],
						"completed_count": row['completed_count'],
						"completion_rate": row['completed_count'] / row['activity_count'] if row['activity_count'] > 0 else 0,
						"avg_engagement": float(row['avg_engagement']) if row['avg_engagement'] else 0
					}
					for row in engagement_trends
				]
				
				# Calculate productivity metrics
				completed_activities = await conn.fetchval(f"""
					SELECT COUNT(*) FROM crm_activities 
					WHERE {base_conditions} AND status = 'completed'
				""", *base_params)
				
				avg_completion_time = await conn.fetchval(f"""
					SELECT AVG(EXTRACT(EPOCH FROM (completed_at - created_at))/3600) 
					FROM crm_activities 
					WHERE {base_conditions} AND completed_at IS NOT NULL
				""", *base_params)
				
				analytics["productivity_metrics"] = {
					"completion_rate": completed_activities / total_count if total_count > 0 else 0,
					"avg_completion_time_hours": float(avg_completion_time) if avg_completion_time else 0,
					"activities_per_day": total_count / max(1, (end_date - start_date).days)
				}
			
			return analytics
			
		except Exception as e:
			logger.error(f"Activity analytics failed: {str(e)}", exc_info=True)
			raise Exception(f"Activity analytics failed: {str(e)}")
	
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
			start_date = datetime.utcnow() - timedelta(days=days)
			
			async with self.db_manager.get_connection() as conn:
				activities = await conn.fetch("""
					SELECT 
						id, activity_type, subject, status, outcome,
						engagement_score, created_at, completed_at, due_date
					FROM crm_activities 
					WHERE tenant_id = $1 AND contact_id = $2 
					AND created_at >= $3
					ORDER BY created_at DESC
				""", tenant_id, contact_id, start_date)
				
				timeline = []
				for activity in activities:
					timeline.append({
						"id": activity['id'],
						"activity_type": activity['activity_type'],
						"subject": activity['subject'],
						"status": activity['status'],
						"outcome": activity['outcome'],
						"engagement_score": float(activity['engagement_score']) if activity['engagement_score'] else 0,
						"created_at": activity['created_at'].isoformat(),
						"completed_at": activity['completed_at'].isoformat() if activity['completed_at'] else None,
						"due_date": activity['due_date'].isoformat() if activity['due_date'] else None,
						"is_overdue": activity['due_date'] and activity['due_date'] < datetime.utcnow() and activity['status'] != 'completed'
					})
				
				return timeline
				
		except Exception as e:
			logger.error(f"Engagement timeline failed: {str(e)}", exc_info=True)
			raise Exception(f"Engagement timeline failed: {str(e)}")
	
	# ================================
	# Task Management
	# ================================
	
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
			async with self.db_manager.get_connection() as conn:
				query = """
					SELECT a.*, c.first_name, c.last_name, c.email, c.company
					FROM crm_activities a
					LEFT JOIN crm_contacts c ON a.contact_id = c.id
					WHERE a.tenant_id = $1 
					AND a.status IN ('planned', 'in_progress')
					AND a.due_date < NOW()
				"""
				params = [tenant_id]
				param_counter = 2
				
				if assigned_to:
					query += f" AND a.assigned_to_id = ${param_counter}"
					params.append(assigned_to)
					param_counter += 1
				
				if days_overdue:
					query += f" AND a.due_date < NOW() - INTERVAL '{days_overdue} days'"
				
				query += " ORDER BY a.due_date ASC"
				
				rows = await conn.fetch(query, *params)
				
				overdue_activities = []
				for row in rows:
					days_overdue = (datetime.utcnow().date() - row['due_date'].date()).days
					
					overdue_activities.append({
						"id": row['id'],
						"activity_type": row['activity_type'],
						"subject": row['subject'],
						"description": row['description'],
						"priority": row['priority'],
						"due_date": row['due_date'].isoformat(),
						"days_overdue": days_overdue,
						"contact": {
							"id": row['contact_id'],
							"name": f"{row['first_name']} {row['last_name']}" if row['first_name'] else None,
							"email": row['email'],
							"company": row['company']
						} if row['contact_id'] else None,
						"assigned_to_id": row['assigned_to_id'],
						"owner_id": row['owner_id']
					})
				
				return overdue_activities
				
		except Exception as e:
			logger.error(f"Get overdue activities failed: {str(e)}", exc_info=True)
			raise Exception(f"Get overdue activities failed: {str(e)}")
	
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
			end_date = datetime.utcnow() + timedelta(days=days_ahead)
			
			async with self.db_manager.get_connection() as conn:
				query = """
					SELECT a.*, c.first_name, c.last_name, c.email, c.company
					FROM crm_activities a
					LEFT JOIN crm_contacts c ON a.contact_id = c.id
					WHERE a.tenant_id = $1 
					AND a.status IN ('planned', 'in_progress')
					AND a.due_date BETWEEN NOW() AND $2
				"""
				params = [tenant_id, end_date]
				param_counter = 3
				
				if assigned_to:
					query += f" AND a.assigned_to_id = ${param_counter}"
					params.append(assigned_to)
				
				query += " ORDER BY a.due_date ASC"
				
				rows = await conn.fetch(query, *params)
				
				upcoming_activities = []
				for row in rows:
					days_until = (row['due_date'].date() - datetime.utcnow().date()).days
					
					upcoming_activities.append({
						"id": row['id'],
						"activity_type": row['activity_type'],
						"subject": row['subject'],
						"description": row['description'],
						"priority": row['priority'],
						"due_date": row['due_date'].isoformat(),
						"days_until": days_until,
						"contact": {
							"id": row['contact_id'],
							"name": f"{row['first_name']} {row['last_name']}" if row['first_name'] else None,
							"email": row['email'],
							"company": row['company']
						} if row['contact_id'] else None,
						"assigned_to_id": row['assigned_to_id'],
						"owner_id": row['owner_id']
					})
				
				return upcoming_activities
				
		except Exception as e:
			logger.error(f"Get upcoming activities failed: {str(e)}", exc_info=True)
			raise Exception(f"Get upcoming activities failed: {str(e)}")
	
	# ================================
	# Helper Methods
	# ================================
	
	async def _save_activity(self, activity: CRMActivity) -> CRMActivity:
		"""Save activity to database"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_activities (
						id, tenant_id, activity_type, subject, description, status,
						priority, contact_id, account_id, opportunity_id, lead_id,
						due_date, start_time, end_time, duration_minutes,
						location, meeting_link, engagement_score, outcome,
						outcome_summary, notes, tags, metadata,
						owner_id, assigned_to_id, created_at, updated_at,
						created_by, updated_by, version, completed_at
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
						$15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26,
						$27, $28, $29, $30, $31
					)
					ON CONFLICT (id) DO UPDATE SET
						activity_type = EXCLUDED.activity_type,
						subject = EXCLUDED.subject,
						description = EXCLUDED.description,
						status = EXCLUDED.status,
						priority = EXCLUDED.priority,
						contact_id = EXCLUDED.contact_id,
						account_id = EXCLUDED.account_id,
						opportunity_id = EXCLUDED.opportunity_id,
						lead_id = EXCLUDED.lead_id,
						due_date = EXCLUDED.due_date,
						start_time = EXCLUDED.start_time,
						end_time = EXCLUDED.end_time,
						duration_minutes = EXCLUDED.duration_minutes,
						location = EXCLUDED.location,
						meeting_link = EXCLUDED.meeting_link,
						engagement_score = EXCLUDED.engagement_score,
						outcome = EXCLUDED.outcome,
						outcome_summary = EXCLUDED.outcome_summary,
						notes = EXCLUDED.notes,
						tags = EXCLUDED.tags,
						metadata = EXCLUDED.metadata,
						owner_id = EXCLUDED.owner_id,
						assigned_to_id = EXCLUDED.assigned_to_id,
						updated_at = EXCLUDED.updated_at,
						updated_by = EXCLUDED.updated_by,
						version = EXCLUDED.version,
						completed_at = EXCLUDED.completed_at
				""",
					activity.id, activity.tenant_id, activity.activity_type.value,
					activity.subject, activity.description, activity.status.value,
					activity.priority.value, activity.contact_id, activity.account_id,
					activity.opportunity_id, activity.lead_id, activity.due_date,
					activity.start_time, activity.end_time, activity.duration_minutes,
					activity.location, activity.meeting_link, activity.engagement_score,
					activity.outcome, activity.outcome_summary, activity.notes,
					activity.tags, activity.metadata, activity.owner_id,
					activity.assigned_to_id, activity.created_at, activity.updated_at,
					activity.created_by, activity.updated_by, activity.version,
					activity.completed_at
				)
			
			return activity
			
		except Exception as e:
			logger.error(f"Failed to save activity: {str(e)}", exc_info=True)
			raise Exception(f"Save activity failed: {str(e)}")
	
	def _row_to_activity(self, row) -> CRMActivity:
		"""Convert database row to CRMActivity object"""
		return CRMActivity(
			id=row['id'],
			tenant_id=row['tenant_id'],
			activity_type=ActivityType(row['activity_type']),
			subject=row['subject'],
			description=row['description'],
			status=ActivityStatus(row['status']),
			priority=Priority(row['priority']),
			contact_id=row['contact_id'],
			account_id=row['account_id'],
			opportunity_id=row['opportunity_id'],
			lead_id=row['lead_id'],
			due_date=row['due_date'],
			start_time=row['start_time'],
			end_time=row['end_time'],
			duration_minutes=row['duration_minutes'],
			location=row['location'],
			meeting_link=row['meeting_link'],
			engagement_score=row['engagement_score'],
			outcome=row['outcome'],
			outcome_summary=row['outcome_summary'],
			notes=row['notes'],
			tags=row['tags'] or [],
			metadata=row['metadata'] or {},
			owner_id=row['owner_id'],
			assigned_to_id=row['assigned_to_id'],
			created_at=row['created_at'],
			updated_at=row['updated_at'],
			created_by=row['created_by'],
			updated_by=row['updated_by'],
			version=row['version'],
			completed_at=row['completed_at']
		)
	
	async def _update_engagement_score(self, activity: CRMActivity):
		"""Update engagement score based on activity outcome"""
		try:
			if not activity.contact_id or not activity.outcome:
				return
			
			# Calculate engagement score based on activity type and outcome
			base_score = self.activity_rules["engagement_scoring"].get(
				activity.activity_type.value, 1
			)
			
			# Adjust score based on outcome
			outcome_multipliers = {
				"successful": 1.0,
				"partial": 0.7,
				"unsuccessful": 0.3,
				"no_response": 0.1,
				"cancelled": 0.0
			}
			
			multiplier = outcome_multipliers.get(activity.outcome, 0.5)
			engagement_score = base_score * multiplier
			
			# Update activity engagement score
			await self.update_activity(
				activity.id,
				{"engagement_score": engagement_score},
				activity.tenant_id,
				"system"
			)
			
		except Exception as e:
			logger.error(f"Failed to update engagement score: {str(e)}", exc_info=True)
	
	async def _schedule_automatic_follow_ups(self, activity: CRMActivity):
		"""Schedule automatic follow-up activities"""
		try:
			if not activity.contact_id:
				return
			
			follow_up_rules = self.activity_rules["follow_up_rules"]
			
			# Schedule follow-ups based on activity type
			if activity.activity_type == ActivityType.EMAIL:
				# Schedule follow-up if no response in 3 days
				follow_up_date = datetime.utcnow() + timedelta(
					days=follow_up_rules["no_response_days"]
				)
				
				await self.create_activity({
					"activity_type": ActivityType.FOLLOW_UP,
					"subject": f"Follow-up: {activity.subject}",
					"description": f"Follow-up on email: {activity.subject}",
					"contact_id": activity.contact_id,
					"due_date": follow_up_date,
					"priority": Priority.MEDIUM,
					"status": ActivityStatus.PLANNED,
					"owner_id": activity.owner_id,
					"assigned_to_id": activity.assigned_to_id
				}, activity.tenant_id, "system")
			
		except Exception as e:
			logger.error(f"Failed to schedule follow-ups: {str(e)}", exc_info=True)
	
	async def _generate_follow_up_activities(self, completed_activity: CRMActivity, created_by: str):
		"""Generate follow-up activities based on completed activity outcome"""
		try:
			if not completed_activity.contact_id or not completed_activity.outcome:
				return
			
			# Generate follow-ups based on outcome
			if completed_activity.outcome == "successful":
				if completed_activity.activity_type == ActivityType.CALL:
					# Schedule follow-up email after successful call
					await self.create_activity({
						"activity_type": ActivityType.EMAIL,
						"subject": f"Follow-up email after call",
						"description": f"Send follow-up email after successful call: {completed_activity.subject}",
						"contact_id": completed_activity.contact_id,
						"due_date": datetime.utcnow() + timedelta(hours=24),
						"priority": Priority.MEDIUM,
						"status": ActivityStatus.PLANNED,
						"owner_id": completed_activity.owner_id,
						"assigned_to_id": completed_activity.assigned_to_id
					}, completed_activity.tenant_id, created_by)
			
			elif completed_activity.outcome == "no_response":
				# Schedule another attempt
				await self.create_activity({
					"activity_type": completed_activity.activity_type,
					"subject": f"Retry: {completed_activity.subject}",
					"description": f"Retry attempt - no response to: {completed_activity.subject}",
					"contact_id": completed_activity.contact_id,
					"due_date": datetime.utcnow() + timedelta(days=3),
					"priority": Priority.HIGH,
					"status": ActivityStatus.PLANNED,
					"owner_id": completed_activity.owner_id,
					"assigned_to_id": completed_activity.assigned_to_id
				}, completed_activity.tenant_id, created_by)
			
		except Exception as e:
			logger.error(f"Failed to generate follow-up activities: {str(e)}", exc_info=True)