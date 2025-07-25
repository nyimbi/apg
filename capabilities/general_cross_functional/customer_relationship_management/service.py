"""
Customer Relationship Management Service Layer

Comprehensive business logic implementation for CRM with AI integration,
background processing, automation workflows, and advanced analytics.
"""

import asyncio
import json
import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from uuid_extensions import uuid7str

from sqlalchemy import and_, or_, func, desc, asc, text
from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from flask import current_app
from celery import Celery

from .models import (
	GCCRMAccount, GCCRMCustomer, GCCRMContact, GCCRMLead, GCCRMOpportunity,
	GCCRMSalesStage, GCCRMActivity, GCCRMTask, GCCRMAppointment, GCCRMCampaign,
	GCCRMCampaignMember, GCCRMMarketingList, GCCRMEmailTemplate, GCCRMCase,
	GCCRMCaseComment, GCCRMProduct, GCCRMPriceList, GCCRMQuote, GCCRMQuoteLine,
	GCCRMTerritory, GCCRMTeam, GCCRMForecast, GCCRMDashboardWidget, GCCRMReport,
	GCCRMIntegrationLog, GCCRMAuditLog, GCCRMLeadSource, GCCRMCustomerSegment,
	GCCRMCustomerScore, GCCRMSocialProfile, GCCRMCommunication, GCCRMWorkflowDefinition,
	GCCRMWorkflowExecution, GCCRMNotification, GCCRMKnowledgeBase, GCCRMCustomField,
	GCCRMCustomFieldValue, GCCRMDocumentAttachment, GCCRMEventLog, GCCRMSystemConfiguration,
	GCCRMWebhookEndpoint, GCCRMWebhookDelivery, LeadStatus, LeadRating, OpportunityStage,
	ActivityType, ActivityStatus, CaseStatus, CasePriority, CampaignStatus
)

# Configure logging
logger = logging.getLogger(__name__)

# AI/ML Integration imports (placeholder for actual implementations)
try:
	import openai
	import pandas as pd
	import numpy as np
	from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score, mean_squared_error
	AI_ENABLED = True
except ImportError:
	AI_ENABLED = False
	logger.warning("AI/ML libraries not available. AI features will be disabled.")

# Celery setup for background processing
celery = Celery('crm_service')

class CRMServiceError(Exception):
	"""Base exception for CRM service errors"""
	pass

class ValidationError(CRMServiceError):
	"""Validation error in CRM operations"""
	pass

class PermissionError(CRMServiceError):
	"""Permission denied for CRM operation"""
	pass

class NotFoundError(CRMServiceError):
	"""Entity not found error"""
	pass

class AIModelError(CRMServiceError):
	"""AI model related error"""
	pass

# Core CRM Service Classes

class LeadManagementService:
	"""Advanced lead management with AI-powered scoring and automation"""
	
	def __init__(self, db_session: Session, tenant_id: str, user_id: str):
		self.db = db_session
		self.tenant_id = tenant_id
		self.user_id = user_id
	
	def create_lead(self, lead_data: Dict[str, Any]) -> GCCRMLead:
		"""Create a new lead with AI scoring and automatic routing"""
		try:
			# Validate lead data
			self._validate_lead_data(lead_data)
			
			# Create lead instance
			lead = GCCRMLead(
				id=uuid7str(),
				tenant_id=self.tenant_id,
				**{k: v for k, v in lead_data.items() if k != 'id'}
			)
			
			# AI-powered lead scoring
			if AI_ENABLED:
				lead.lead_score = self._calculate_lead_score(lead_data)
				lead.ai_insights = self._generate_lead_insights(lead_data)
			
			# Automatic lead routing
			lead.lead_owner_id = self._route_lead_automatically(lead)
			
			# Save to database
			self.db.add(lead)
			self.db.flush()
			
			# Log event
			self._log_lead_event(lead.id, 'lead_created', {'score': lead.lead_score})
			
			# Trigger workflows
			self._trigger_lead_workflows(lead)
			
			# Schedule follow-up tasks
			self._schedule_lead_follow_up(lead)
			
			self.db.commit()
			
			logger.info(f"Lead created: {lead.id} with score: {lead.lead_score}")
			return lead
			
		except Exception as e:
			self.db.rollback()
			logger.error(f"Error creating lead: {str(e)}")
			raise CRMServiceError(f"Failed to create lead: {str(e)}")
	
	def update_lead(self, lead_id: str, update_data: Dict[str, Any]) -> GCCRMLead:
		"""Update lead with re-scoring and progression tracking"""
		lead = self._get_lead_or_404(lead_id)
		
		try:
			# Store previous values for audit
			previous_values = {
				'lead_status': lead.lead_status,
				'lead_score': lead.lead_score,
				'lead_rating': lead.lead_rating
			}
			
			# Update lead fields
			for key, value in update_data.items():
				if hasattr(lead, key) and key != 'id':
					setattr(lead, key, value)
			
			# Re-calculate AI score if significant changes
			if AI_ENABLED and self._significant_lead_change(update_data):
				lead.lead_score = self._calculate_lead_score(lead.__dict__)
				lead.ai_insights = self._generate_lead_insights(lead.__dict__)
			
			# Track progression
			if previous_values['lead_status'] != lead.lead_status:
				self._track_lead_progression(lead, previous_values['lead_status'])
			
			self.db.commit()
			
			# Log changes
			self._log_lead_event(lead.id, 'lead_updated', {
				'previous': previous_values,
				'current': {k: getattr(lead, k) for k in previous_values.keys()}
			})
			
			return lead
			
		except Exception as e:
			self.db.rollback()
			logger.error(f"Error updating lead {lead_id}: {str(e)}")
			raise CRMServiceError(f"Failed to update lead: {str(e)}")
	
	def convert_lead_to_opportunity(self, lead_id: str, opportunity_data: Dict[str, Any]) -> Tuple[GCCRMOpportunity, GCCRMContact]:
		"""Convert qualified lead to opportunity and contact"""
		lead = self._get_lead_or_404(lead_id)
		
		if lead.lead_status != LeadStatus.QUALIFIED:
			raise ValidationError("Only qualified leads can be converted")
		
		try:
			# Create contact from lead
			contact = GCCRMContact(
				id=uuid7str(),
				tenant_id=self.tenant_id,
				first_name=lead.first_name,
				last_name=lead.last_name,
				email=lead.email,
				phone=lead.phone,
				company=lead.company,
				job_title=lead.job_title,
				contact_owner_id=lead.lead_owner_id,
				lead_source=lead.lead_source,
				converted_from_lead_id=lead.id
			)
			
			# Create opportunity
			opportunity = GCCRMOpportunity(
				id=uuid7str(),
				tenant_id=self.tenant_id,
				opportunity_name=opportunity_data.get('name', f"{lead.company} - {lead.first_name} {lead.last_name}"),
				amount=opportunity_data.get('amount', Decimal('0')),
				close_date=opportunity_data.get('close_date'),
				stage=OpportunityStage.PROSPECTING,
				probability=self._calculate_initial_opportunity_probability(lead),
				opportunity_owner_id=lead.lead_owner_id,
				primary_contact_id=contact.id,
				lead_source=lead.lead_source,
				converted_from_lead_id=lead.id
			)
			
			# Update lead status
			lead.lead_status = LeadStatus.CONVERTED
			lead.converted_date = datetime.utcnow()
			lead.converted_to_opportunity_id = opportunity.id
			lead.converted_to_contact_id = contact.id
			
			# Save all entities
			self.db.add_all([contact, opportunity])
			self.db.commit()
			
			# Log conversion
			self._log_lead_event(lead.id, 'lead_converted', {
				'opportunity_id': opportunity.id,
				'contact_id': contact.id
			})
			
			# Trigger conversion workflows
			self._trigger_conversion_workflows(lead, opportunity, contact)
			
			logger.info(f"Lead {lead_id} converted to opportunity {opportunity.id}")
			return opportunity, contact
			
		except Exception as e:
			self.db.rollback()
			logger.error(f"Error converting lead {lead_id}: {str(e)}")
			raise CRMServiceError(f"Failed to convert lead: {str(e)}")
	
	def get_lead_analytics(self, date_from: date, date_to: date) -> Dict[str, Any]:
		"""Get comprehensive lead analytics and performance metrics"""
		try:
			# Basic metrics
			total_leads = self.db.query(GCCRMLead).filter(
				GCCRMLead.tenant_id == self.tenant_id,
				GCCRMLead.created_on.between(date_from, date_to)
			).count()
			
			qualified_leads = self.db.query(GCCRMLead).filter(
				GCCRMLead.tenant_id == self.tenant_id,
				GCCRMLead.lead_status == LeadStatus.QUALIFIED,
				GCCRMLead.created_on.between(date_from, date_to)
			).count()
			
			converted_leads = self.db.query(GCCRMLead).filter(
				GCCRMLead.tenant_id == self.tenant_id,
				GCCRMLead.lead_status == LeadStatus.CONVERTED,
				GCCRMLead.converted_date.between(date_from, date_to)
			).count()
			
			# Lead source analysis
			source_analysis = self.db.query(
				GCCRMLead.lead_source,
				func.count(GCCRMLead.id).label('count'),
				func.avg(GCCRMLead.lead_score).label('avg_score')
			).filter(
				GCCRMLead.tenant_id == self.tenant_id,
				GCCRMLead.created_on.between(date_from, date_to)
			).group_by(GCCRMLead.lead_source).all()
			
			# Score distribution
			score_distribution = self.db.query(
				func.count(GCCRMLead.id).label('count')
			).filter(
				GCCRMLead.tenant_id == self.tenant_id,
				GCCRMLead.created_on.between(date_from, date_to)
			).group_by(
				func.floor(GCCRMLead.lead_score / 10) * 10
			).all()
			
			# Conversion funnel
			conversion_funnel = {
				'new': self.db.query(GCCRMLead).filter(
					GCCRMLead.tenant_id == self.tenant_id,
					GCCRMLead.lead_status == LeadStatus.NEW,
					GCCRMLead.created_on.between(date_from, date_to)
				).count(),
				'contacted': self.db.query(GCCRMLead).filter(
					GCCRMLead.tenant_id == self.tenant_id,
					GCCRMLead.lead_status == LeadStatus.CONTACTED,
					GCCRMLead.created_on.between(date_from, date_to)
				).count(),
				'qualified': qualified_leads,
				'converted': converted_leads
			}
			
			# Calculate rates
			qualification_rate = (qualified_leads / total_leads * 100) if total_leads > 0 else 0
			conversion_rate = (converted_leads / qualified_leads * 100) if qualified_leads > 0 else 0
			
			analytics = {
				'summary': {
					'total_leads': total_leads,
					'qualified_leads': qualified_leads,
					'converted_leads': converted_leads,
					'qualification_rate': round(qualification_rate, 2),
					'conversion_rate': round(conversion_rate, 2)
				},
				'source_analysis': [
					{
						'source': row.lead_source,
						'count': row.count,
						'avg_score': float(row.avg_score) if row.avg_score else 0
					}
					for row in source_analysis
				],
				'conversion_funnel': conversion_funnel,
				'score_distribution': [{'range': f"{i*10}-{(i+1)*10}", 'count': count} 
									  for i, (count,) in enumerate(score_distribution)]
			}
			
			return analytics
			
		except Exception as e:
			logger.error(f"Error generating lead analytics: {str(e)}")
			raise CRMServiceError(f"Failed to generate analytics: {str(e)}")
	
	def bulk_lead_operations(self, operation: str, lead_ids: List[str], operation_data: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Perform bulk operations on multiple leads"""
		try:
			results = {'success': [], 'failed': []}
			
			for lead_id in lead_ids:
				try:
					if operation == 'update':
						self.update_lead(lead_id, operation_data or {})
					elif operation == 'delete':
						self._soft_delete_lead(lead_id)
					elif operation == 'assign':
						self._assign_lead(lead_id, operation_data.get('owner_id'))
					elif operation == 'qualify':
						self._qualify_lead(lead_id)
					elif operation == 'disqualify':
						self._disqualify_lead(lead_id, operation_data.get('reason'))
					
					results['success'].append(lead_id)
					
				except Exception as e:
					results['failed'].append({'lead_id': lead_id, 'error': str(e)})
			
			# Log bulk operation
			self._log_bulk_operation('leads', operation, results)
			
			return results
			
		except Exception as e:
			logger.error(f"Error in bulk lead operation: {str(e)}")
			raise CRMServiceError(f"Bulk operation failed: {str(e)}")
	
	# Private helper methods
	def _validate_lead_data(self, data: Dict[str, Any]) -> None:
		"""Validate lead data before creation/update"""
		required_fields = ['first_name', 'last_name', 'email']
		
		for field in required_fields:
			if not data.get(field):
				raise ValidationError(f"Required field missing: {field}")
		
		# Email validation
		email = data.get('email')
		if email and '@' not in email:
			raise ValidationError("Invalid email format")
		
		# Check for duplicate email in tenant
		existing_lead = self.db.query(GCCRMLead).filter(
			GCCRMLead.tenant_id == self.tenant_id,
			GCCRMLead.email == email,
			GCCRMLead.is_active == True
		).first()
		
		if existing_lead:
			raise ValidationError(f"Lead with email {email} already exists")
	
	def _calculate_lead_score(self, lead_data: Dict[str, Any]) -> int:
		"""AI-powered lead scoring using machine learning"""
		if not AI_ENABLED:
			return 50  # Default score
		
		try:
			# Feature extraction for ML model
			features = self._extract_lead_features(lead_data)
			
			# Load or train ML model
			model = self._get_lead_scoring_model()
			
			# Predict score
			score = model.predict([features])[0]
			
			# Normalize to 0-100 range
			return max(0, min(100, int(score)))
			
		except Exception as e:
			logger.warning(f"AI scoring failed, using heuristic: {str(e)}")
			return self._heuristic_lead_score(lead_data)
	
	def _heuristic_lead_score(self, lead_data: Dict[str, Any]) -> int:
		"""Fallback heuristic lead scoring"""
		score = 50  # Base score
		
		# Company size indicators
		if lead_data.get('company_size', '').lower() in ['large', 'enterprise']:
			score += 20
		elif lead_data.get('company_size', '').lower() in ['medium', 'mid-size']:
			score += 10
		
		# Job title importance
		title = lead_data.get('job_title', '').lower()
		if any(keyword in title for keyword in ['ceo', 'cto', 'director', 'vp', 'president']):
			score += 15
		elif any(keyword in title for keyword in ['manager', 'lead', 'head']):
			score += 10
		
		# Lead source quality
		source = lead_data.get('lead_source')
		if source == 'referral':
			score += 15
		elif source == 'website':
			score += 10
		elif source == 'trade_show':
			score += 8
		
		# Engagement indicators
		if lead_data.get('website_visits', 0) > 5:
			score += 10
		if lead_data.get('email_opens', 0) > 3:
			score += 5
		
		return max(0, min(100, score))
	
	def _generate_lead_insights(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate AI insights for lead"""
		if not AI_ENABLED:
			return {}
		
		insights = {
			'buying_signals': [],
			'next_best_action': '',
			'risk_factors': [],
			'similar_won_deals': []
		}
		
		# Analyze buying signals
		if lead_data.get('website_visits', 0) > 10:
			insights['buying_signals'].append('High website engagement')
		
		if lead_data.get('requested_demo', False):
			insights['buying_signals'].append('Requested product demo')
		
		# Recommend next action
		score = lead_data.get('lead_score', 50)
		if score > 80:
			insights['next_best_action'] = 'Schedule immediate call'
		elif score > 60:
			insights['next_best_action'] = 'Send personalized proposal'
		else:
			insights['next_best_action'] = 'Continue nurturing campaign'
		
		return insights
	
	def _route_lead_automatically(self, lead: GCCRMLead) -> str:
		"""Automatically route lead to appropriate sales rep"""
		# Round-robin assignment logic
		available_reps = self.db.query(GCCRMTeam).filter(
			GCCRMTeam.tenant_id == self.tenant_id,
			GCCRMTeam.is_active == True
		).all()
		
		if not available_reps:
			return self.user_id
		
		# Simple round-robin for now
		# In production, consider factors like territory, workload, expertise
		rep_index = hash(lead.email) % len(available_reps)
		return available_reps[rep_index].id
	
	def _get_lead_or_404(self, lead_id: str) -> GCCRMLead:
		"""Get lead or raise NotFoundError"""
		lead = self.db.query(GCCRMLead).filter(
			GCCRMLead.id == lead_id,
			GCCRMLead.tenant_id == self.tenant_id
		).first()
		
		if not lead:
			raise NotFoundError(f"Lead {lead_id} not found")
		
		return lead
	
	def _log_lead_event(self, lead_id: str, event_type: str, event_data: Dict[str, Any]) -> None:
		"""Log lead-related events for audit and analytics"""
		event_log = GCCRMEventLog(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			entity_type='lead',
			entity_id=lead_id,
			event_type=event_type,
			event_data=event_data,
			user_id=self.user_id,
			event_timestamp=datetime.utcnow()
		)
		self.db.add(event_log)
	
	def _trigger_lead_workflows(self, lead: GCCRMLead) -> None:
		"""Trigger automated workflows for new lead"""
		# Schedule background task for workflow execution
		execute_lead_workflows.delay(lead.id, self.tenant_id)
	
	def _schedule_lead_follow_up(self, lead: GCCRMLead) -> None:
		"""Schedule automatic follow-up tasks"""
		# Create follow-up task
		follow_up_task = GCCRMTask(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			task_name=f"Follow up with {lead.first_name} {lead.last_name}",
			description=f"Initial follow-up for lead from {lead.lead_source}",
			assigned_to_id=lead.lead_owner_id,
			due_date=datetime.utcnow() + timedelta(hours=24),
			activity_status=ActivityStatus.PENDING,
			related_lead_id=lead.id,
			task_type='follow_up',
			priority='high' if lead.lead_score > 70 else 'medium'
		)
		self.db.add(follow_up_task)


class OpportunityManagementService:
	"""Advanced opportunity management with predictive analytics"""
	
	def __init__(self, db_session: Session, tenant_id: str, user_id: str):
		self.db = db_session
		self.tenant_id = tenant_id
		self.user_id = user_id
	
	def create_opportunity(self, opportunity_data: Dict[str, Any]) -> GCCRMOpportunity:
		"""Create new opportunity with AI-powered insights"""
		try:
			# Validate opportunity data
			self._validate_opportunity_data(opportunity_data)
			
			# Create opportunity
			opportunity = GCCRMOpportunity(
				id=uuid7str(),
				tenant_id=self.tenant_id,
				**{k: v for k, v in opportunity_data.items() if k != 'id'}
			)
			
			# AI-powered probability calculation
			if AI_ENABLED:
				opportunity.probability = self._calculate_win_probability(opportunity_data)
				opportunity.ai_insights = self._generate_opportunity_insights(opportunity_data)
			
			# Set expected revenue
			opportunity.expected_revenue = (opportunity.amount or Decimal('0')) * (opportunity.probability / 100)
			
			self.db.add(opportunity)
			self.db.flush()
			
			# Log creation
			self._log_opportunity_event(opportunity.id, 'opportunity_created', {
				'amount': float(opportunity.amount),
				'probability': opportunity.probability
			})
			
			# Create initial activities
			self._create_initial_opportunity_activities(opportunity)
			
			self.db.commit()
			
			logger.info(f"Opportunity created: {opportunity.id}")
			return opportunity
			
		except Exception as e:
			self.db.rollback()
			logger.error(f"Error creating opportunity: {str(e)}")
			raise CRMServiceError(f"Failed to create opportunity: {str(e)}")
	
	def update_opportunity_stage(self, opportunity_id: str, new_stage: OpportunityStage, stage_data: Dict[str, Any] = None) -> GCCRMOpportunity:
		"""Update opportunity stage with progression tracking"""
		opportunity = self._get_opportunity_or_404(opportunity_id)
		
		try:
			previous_stage = opportunity.stage
			
			# Update stage
			opportunity.stage = new_stage
			
			# Recalculate probability based on stage
			stage_probabilities = {
				OpportunityStage.PROSPECTING: 10,
				OpportunityStage.QUALIFICATION: 25,
				OpportunityStage.NEEDS_ANALYSIS: 40,
				OpportunityStage.VALUE_PROPOSITION: 55,
				OpportunityStage.ID_DECISION_MAKERS: 70,
				OpportunityStage.PERCEPTION_ANALYSIS: 80,
				OpportunityStage.PROPOSAL_QUOTE: 85,
				OpportunityStage.NEGOTIATION_REVIEW: 90,
				OpportunityStage.CLOSED_WON: 100,
				OpportunityStage.CLOSED_LOST: 0
			}
			
			opportunity.probability = stage_probabilities.get(new_stage, opportunity.probability)
			opportunity.expected_revenue = (opportunity.amount or Decimal('0')) * (opportunity.probability / 100)
			
			# Handle stage-specific logic
			if new_stage == OpportunityStage.CLOSED_WON:
				opportunity.close_date = date.today()
				opportunity.actual_close_date = datetime.utcnow()
				self._handle_won_opportunity(opportunity)
			
			elif new_stage == OpportunityStage.CLOSED_LOST:
				opportunity.close_date = date.today()
				opportunity.actual_close_date = datetime.utcnow()
				opportunity.lost_reason = stage_data.get('lost_reason') if stage_data else None
				self._handle_lost_opportunity(opportunity)
			
			self.db.commit()
			
			# Log stage change
			self._log_opportunity_event(opportunity.id, 'stage_changed', {
				'previous_stage': previous_stage.value if previous_stage else None,
				'new_stage': new_stage.value,
				'probability': opportunity.probability
			})
			
			# Trigger stage-specific workflows
			self._trigger_stage_workflows(opportunity, previous_stage, new_stage)
			
			return opportunity
			
		except Exception as e:
			self.db.rollback()
			logger.error(f"Error updating opportunity stage: {str(e)}")
			raise CRMServiceError(f"Failed to update stage: {str(e)}")
	
	def get_sales_pipeline_analytics(self) -> Dict[str, Any]:
		"""Get comprehensive sales pipeline analytics"""
		try:
			# Pipeline by stage
			pipeline_by_stage = self.db.query(
				GCCRMOpportunity.stage,
				func.count(GCCRMOpportunity.id).label('count'),
				func.sum(GCCRMOpportunity.amount).label('total_value'),
				func.sum(GCCRMOpportunity.expected_revenue).label('expected_revenue')
			).filter(
				GCCRMOpportunity.tenant_id == self.tenant_id,
				GCCRMOpportunity.is_active == True,
				GCCRMOpportunity.stage.notin_([OpportunityStage.CLOSED_WON, OpportunityStage.CLOSED_LOST])
			).group_by(GCCRMOpportunity.stage).all()
			
			# Win rate analysis
			total_closed = self.db.query(GCCRMOpportunity).filter(
				GCCRMOpportunity.tenant_id == self.tenant_id,
				GCCRMOpportunity.stage.in_([OpportunityStage.CLOSED_WON, OpportunityStage.CLOSED_LOST])
			).count()
			
			won_count = self.db.query(GCCRMOpportunity).filter(
				GCCRMOpportunity.tenant_id == self.tenant_id,
				GCCRMOpportunity.stage == OpportunityStage.CLOSED_WON
			).count()
			
			win_rate = (won_count / total_closed * 100) if total_closed > 0 else 0
			
			# Sales velocity
			avg_sales_cycle = self.db.query(
				func.avg(
					func.extract('days', GCCRMOpportunity.actual_close_date - GCCRMOpportunity.created_on)
				)
			).filter(
				GCCRMOpportunity.tenant_id == self.tenant_id,
				GCCRMOpportunity.stage == OpportunityStage.CLOSED_WON,
				GCCRMOpportunity.actual_close_date.isnot(None)
			).scalar()
			
			# Top opportunities
			top_opportunities = self.db.query(GCCRMOpportunity).filter(
				GCCRMOpportunity.tenant_id == self.tenant_id,
				GCCRMOpportunity.is_active == True,
				GCCRMOpportunity.stage.notin_([OpportunityStage.CLOSED_WON, OpportunityStage.CLOSED_LOST])
			).order_by(desc(GCCRMOpportunity.expected_revenue)).limit(10).all()
			
			analytics = {
				'pipeline_overview': {
					'total_opportunities': sum(row.count for row in pipeline_by_stage),
					'total_pipeline_value': float(sum(row.total_value or 0 for row in pipeline_by_stage)),
					'total_expected_revenue': float(sum(row.expected_revenue or 0 for row in pipeline_by_stage)),
					'win_rate': round(win_rate, 2),
					'avg_sales_cycle_days': int(avg_sales_cycle) if avg_sales_cycle else 0
				},
				'pipeline_by_stage': [
					{
						'stage': row.stage.value if row.stage else 'unknown',
						'count': row.count,
						'total_value': float(row.total_value or 0),
						'expected_revenue': float(row.expected_revenue or 0)
					}
					for row in pipeline_by_stage
				],
				'top_opportunities': [
					{
						'id': opp.id,
						'name': opp.opportunity_name,
						'amount': float(opp.amount or 0),
						'expected_revenue': float(opp.expected_revenue or 0),
						'probability': opp.probability,
						'stage': opp.stage.value if opp.stage else 'unknown',
						'close_date': opp.close_date.isoformat() if opp.close_date else None
					}
					for opp in top_opportunities
				]
			}
			
			return analytics
			
		except Exception as e:
			logger.error(f"Error generating pipeline analytics: {str(e)}")
			raise CRMServiceError(f"Failed to generate analytics: {str(e)}")
	
	def forecast_revenue(self, forecast_period: str = 'quarter') -> Dict[str, Any]:
		"""Generate AI-powered revenue forecast"""
		if not AI_ENABLED:
			return self._simple_revenue_forecast(forecast_period)
		
		try:
			# Get historical data
			historical_data = self._get_historical_revenue_data()
			
			# Prepare features for ML model
			features = self._prepare_forecast_features(historical_data)
			
			# Load or train forecasting model
			model = self._get_forecasting_model()
			
			# Generate predictions
			predictions = model.predict(features)
			
			# Calculate confidence intervals
			confidence_intervals = self._calculate_confidence_intervals(predictions)
			
			forecast = {
				'period': forecast_period,
				'predicted_revenue': float(predictions[-1]),
				'confidence_low': float(confidence_intervals['low']),
				'confidence_high': float(confidence_intervals['high']),
				'accuracy_score': self._get_model_accuracy(),
				'contributing_factors': self._identify_forecast_factors(features)
			}
			
			return forecast
			
		except Exception as e:
			logger.error(f"Error generating revenue forecast: {str(e)}")
			return self._simple_revenue_forecast(forecast_period)
	
	# Private helper methods
	def _validate_opportunity_data(self, data: Dict[str, Any]) -> None:
		"""Validate opportunity data"""
		required_fields = ['opportunity_name', 'amount', 'close_date']
		
		for field in required_fields:
			if not data.get(field):
				raise ValidationError(f"Required field missing: {field}")
		
		# Amount validation
		amount = data.get('amount')
		if amount and amount <= 0:
			raise ValidationError("Opportunity amount must be positive")
	
	def _calculate_win_probability(self, opportunity_data: Dict[str, Any]) -> int:
		"""AI-powered win probability calculation"""
		if not AI_ENABLED:
			return 50  # Default probability
		
		try:
			# Extract features for ML model
			features = self._extract_opportunity_features(opportunity_data)
			
			# Load ML model
			model = self._get_win_probability_model()
			
			# Predict probability
			probability = model.predict_proba([features])[0][1] * 100
			
			return max(0, min(100, int(probability)))
			
		except Exception as e:
			logger.warning(f"AI probability calculation failed: {str(e)}")
			return self._heuristic_win_probability(opportunity_data)
	
	def _get_opportunity_or_404(self, opportunity_id: str) -> GCCRMOpportunity:
		"""Get opportunity or raise NotFoundError"""
		opportunity = self.db.query(GCCRMOpportunity).filter(
			GCCRMOpportunity.id == opportunity_id,
			GCCRMOpportunity.tenant_id == self.tenant_id
		).first()
		
		if not opportunity:
			raise NotFoundError(f"Opportunity {opportunity_id} not found")
		
		return opportunity


class CustomerManagementService:
	"""Comprehensive customer management with 360° view and analytics"""
	
	def __init__(self, db_session: Session, tenant_id: str, user_id: str):
		self.db = db_session
		self.tenant_id = tenant_id
		self.user_id = user_id
	
	def get_customer_360_view(self, customer_id: str) -> Dict[str, Any]:
		"""Get comprehensive 360° customer view with all interactions"""
		customer = self._get_customer_or_404(customer_id)
		
		try:
			# Basic customer info
			customer_info = {
				'id': customer.id,
				'name': f"{customer.first_name} {customer.last_name}",
				'email': customer.email,
				'phone': customer.phone,
				'company': customer.company,
				'customer_since': customer.customer_since.isoformat() if customer.customer_since else None,
				'customer_status': customer.customer_status,
				'customer_value': float(customer.customer_value or 0),
				'last_contact_date': customer.last_contact_date.isoformat() if customer.last_contact_date else None
			}
			
			# Related contacts
			contacts = self.db.query(GCCRMContact).filter(
				GCCRMContact.customer_id == customer_id,
				GCCRMContact.tenant_id == self.tenant_id,
				GCCRMContact.is_active == True
			).all()
			
			# Opportunities
			opportunities = self.db.query(GCCRMOpportunity).filter(
				GCCRMOpportunity.customer_id == customer_id,
				GCCRMOpportunity.tenant_id == self.tenant_id,
				GCCRMOpportunity.is_active == True
			).order_by(desc(GCCRMOpportunity.created_on)).all()
			
			# Cases/Support tickets
			cases = self.db.query(GCCRMCase).filter(
				GCCRMCase.customer_id == customer_id,
				GCCRMCase.tenant_id == self.tenant_id,
				GCCRMCase.is_active == True
			).order_by(desc(GCCRMCase.created_on)).all()
			
			# Recent activities
			activities = self.db.query(GCCRMActivity).filter(
				GCCRMActivity.customer_id == customer_id,
				GCCRMActivity.tenant_id == self.tenant_id
			).order_by(desc(GCCRMActivity.created_on)).limit(20).all()
			
			# Communications
			communications = self.db.query(GCCRMCommunication).filter(
				GCCRMCommunication.customer_id == customer_id,
				GCCRMCommunication.tenant_id == self.tenant_id
			).order_by(desc(GCCRMCommunication.sent_date)).limit(10).all()
			
			# Customer scores
			scores = self.db.query(GCCRMCustomerScore).filter(
				GCCRMCustomerScore.customer_id == customer_id,
				GCCRMCustomerScore.tenant_id == self.tenant_id,
				GCCRMCustomerScore.is_current == True
			).all()
			
			# Document attachments
			documents = self.db.query(GCCRMDocumentAttachment).filter(
				GCCRMDocumentAttachment.entity_type == 'customer',
				GCCRMDocumentAttachment.entity_id == customer_id,
				GCCRMDocumentAttachment.tenant_id == self.tenant_id
			).order_by(desc(GCCRMDocumentAttachment.upload_date)).all()
			
			# Build 360° view
			customer_360 = {
				'customer_info': customer_info,
				'contacts': [
					{
						'id': contact.id,
						'name': f"{contact.first_name} {contact.last_name}",
						'email': contact.email,
						'phone': contact.phone,
						'job_title': contact.job_title,
						'is_primary': contact.is_primary_contact
					}
					for contact in contacts
				],
				'opportunities': [
					{
						'id': opp.id,
						'name': opp.opportunity_name,
						'amount': float(opp.amount or 0),
						'stage': opp.stage.value if opp.stage else None,
						'probability': opp.probability,
						'close_date': opp.close_date.isoformat() if opp.close_date else None,
						'created_on': opp.created_on.isoformat()
					}
					for opp in opportunities
				],
				'cases': [
					{
						'id': case.id,
						'subject': case.subject,
						'status': case.status.value if case.status else None,
						'priority': case.priority.value if case.priority else None,
						'created_on': case.created_on.isoformat()
					}
					for case in cases
				],
				'recent_activities': [
					{
						'id': activity.id,
						'subject': activity.subject,
						'activity_type': activity.activity_type.value if activity.activity_type else None,
						'due_date': activity.due_date.isoformat() if activity.due_date else None,
						'status': activity.activity_status.value if activity.activity_status else None
					}
					for activity in activities
				],
				'communications': [
					{
						'id': comm.id,
						'communication_type': comm.communication_type,
						'subject': comm.subject,
						'sent_date': comm.sent_date.isoformat() if comm.sent_date else None,
						'sentiment_score': float(comm.sentiment_score or 0)
					}
					for comm in communications
				],
				'scores': {
					score.score_type: {
						'value': float(score.score_value),
						'percentile': score.score_percentile,
						'calculated_date': score.calculated_date.isoformat() if score.calculated_date else None
					}
					for score in scores
				},
				'documents': [
					{
						'id': doc.id,
						'file_name': doc.file_name,
						'document_type': doc.document_type,
						'upload_date': doc.upload_date.isoformat()
					}
					for doc in documents
				]
			}
			
			# Add AI insights if available
			if AI_ENABLED:
				customer_360['ai_insights'] = self._generate_customer_insights(customer, customer_360)
			
			return customer_360
			
		except Exception as e:
			logger.error(f"Error generating customer 360 view: {str(e)}")
			raise CRMServiceError(f"Failed to generate customer view: {str(e)}")
	
	def calculate_customer_lifetime_value(self, customer_id: str) -> Dict[str, Any]:
		"""Calculate customer lifetime value using historical data and ML"""
		customer = self._get_customer_or_404(customer_id)
		
		try:
			# Get historical purchase data
			purchases = self.db.query(GCCRMOpportunity).filter(
				GCCRMOpportunity.customer_id == customer_id,
				GCCRMOpportunity.tenant_id == self.tenant_id,
				GCCRMOpportunity.stage == OpportunityStage.CLOSED_WON
			).order_by(GCCRMOpportunity.actual_close_date).all()
			
			if not purchases:
				return {
					'customer_id': customer_id,
					'clv': 0.0,
					'method': 'no_data',
					'message': 'No purchase history available'
				}
			
			# Calculate using different methods
			historical_clv = self._calculate_historical_clv(purchases)
			predictive_clv = self._calculate_predictive_clv(customer, purchases) if AI_ENABLED else None
			
			# Calculate metrics
			total_revenue = sum(float(p.amount or 0) for p in purchases)
			avg_order_value = total_revenue / len(purchases)
			purchase_frequency = len(purchases) / max(1, self._get_customer_tenure_months(customer))
			
			clv_data = {
				'customer_id': customer_id,
				'historical_clv': historical_clv,
				'predictive_clv': predictive_clv,
				'recommended_clv': predictive_clv if predictive_clv else historical_clv,
				'metrics': {
					'total_revenue': total_revenue,
					'avg_order_value': avg_order_value,
					'purchase_frequency': purchase_frequency,
					'total_purchases': len(purchases),
					'customer_tenure_months': self._get_customer_tenure_months(customer)
				},
				'calculation_date': datetime.utcnow().isoformat()
			}
			
			# Update customer record with CLV
			customer.customer_value = clv_data['recommended_clv']
			self.db.commit()
			
			return clv_data
			
		except Exception as e:
			logger.error(f"Error calculating CLV for customer {customer_id}: {str(e)}")
			raise CRMServiceError(f"Failed to calculate CLV: {str(e)}")
	
	def segment_customers(self, criteria: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Segment customers using AI-powered clustering or rule-based criteria"""
		try:
			if AI_ENABLED and not criteria:
				return self._ai_customer_segmentation()
			else:
				return self._rule_based_segmentation(criteria or {})
		
		except Exception as e:
			logger.error(f"Error segmenting customers: {str(e)}")
			raise CRMServiceError(f"Customer segmentation failed: {str(e)}")
	
	def predict_churn_risk(self, customer_id: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
		"""Predict customer churn risk using ML models"""
		if not AI_ENABLED:
			return {'error': 'AI models not available'}
		
		try:
			if customer_id:
				# Single customer churn prediction
				return self._predict_single_customer_churn(customer_id)
			else:
				# Batch churn prediction for all customers
				return self._predict_batch_customer_churn()
		
		except Exception as e:
			logger.error(f"Error predicting churn risk: {str(e)}")
			raise CRMServiceError(f"Churn prediction failed: {str(e)}")
	
	# Private helper methods
	def _get_customer_or_404(self, customer_id: str) -> GCCRMCustomer:
		"""Get customer or raise NotFoundError"""
		customer = self.db.query(GCCRMCustomer).filter(
			GCCRMCustomer.id == customer_id,
			GCCRMCustomer.tenant_id == self.tenant_id
		).first()
		
		if not customer:
			raise NotFoundError(f"Customer {customer_id} not found")
		
		return customer


# Background Tasks with Celery

@celery.task(bind=True, max_retries=3)
def execute_lead_workflows(self, lead_id: str, tenant_id: str):
	"""Execute automated workflows for lead"""
	try:
		# Implementation for workflow execution
		logger.info(f"Executing workflows for lead {lead_id}")
		
		# Simulate workflow execution
		# In real implementation, this would:
		# 1. Query active workflows for lead events
		# 2. Execute workflow steps
		# 3. Send notifications
		# 4. Update lead data
		# 5. Log workflow execution
		
		return {'status': 'success', 'lead_id': lead_id}
		
	except Exception as e:
		logger.error(f"Workflow execution failed for lead {lead_id}: {str(e)}")
		if self.request.retries < self.max_retries:
			self.retry(countdown=60 * (self.request.retries + 1))
		raise

@celery.task(bind=True)
def calculate_customer_scores(self, tenant_id: str):
	"""Background task to calculate/update customer scores"""
	try:
		logger.info(f"Calculating customer scores for tenant {tenant_id}")
		
		# Implementation for batch score calculation
		# This would typically:
		# 1. Get all active customers for tenant
		# 2. Calculate various scores (health, churn risk, LTV, etc.)
		# 3. Update GCCRMCustomerScore records
		# 4. Send alerts for significant score changes
		
		return {'status': 'success', 'tenant_id': tenant_id}
		
	except Exception as e:
		logger.error(f"Customer score calculation failed: {str(e)}")
		raise

@celery.task(bind=True)
def data_enrichment_job(self, tenant_id: str):
	"""Background task for data enrichment from external sources"""
	try:
		logger.info(f"Starting data enrichment for tenant {tenant_id}")
		
		# Implementation for data enrichment
		# This would:
		# 1. Identify records needing enrichment
		# 2. Call external APIs (LinkedIn, Clearbit, etc.)
		# 3. Update customer/contact records
		# 4. Log enrichment results
		
		return {'status': 'success', 'tenant_id': tenant_id}
		
	except Exception as e:
		logger.error(f"Data enrichment failed: {str(e)}")
		raise

@celery.task(bind=True)
def generate_ai_insights(self, entity_type: str, entity_id: str, tenant_id: str):
	"""Generate AI insights for CRM entities"""
	try:
		if not AI_ENABLED:
			return {'status': 'skipped', 'reason': 'AI not available'}
		
		logger.info(f"Generating AI insights for {entity_type} {entity_id}")
		
		# Implementation for AI insight generation
		# This would:
		# 1. Gather entity data and context
		# 2. Apply ML models for predictions
		# 3. Generate natural language insights
		# 4. Store insights in database
		# 5. Trigger notifications for important insights
		
		return {'status': 'success', 'entity_type': entity_type, 'entity_id': entity_id}
		
	except Exception as e:
		logger.error(f"AI insight generation failed: {str(e)}")
		raise


# Main CRM Service Factory

class CRMService:
	"""Main CRM service factory providing access to all sub-services"""
	
	def __init__(self, db_session: Session, tenant_id: str, user_id: str):
		self.db = db_session
		self.tenant_id = tenant_id
		self.user_id = user_id
		
		# Initialize sub-services
		self.leads = LeadManagementService(db_session, tenant_id, user_id)
		self.opportunities = OpportunityManagementService(db_session, tenant_id, user_id)
		self.customers = CustomerManagementService(db_session, tenant_id, user_id)
	
	def get_dashboard_data(self) -> Dict[str, Any]:
		"""Get comprehensive dashboard data for CRM"""
		try:
			# Get key metrics
			today = date.today()
			last_30_days = today - timedelta(days=30)
			
			# Lead metrics
			total_leads = self.db.query(GCCRMLead).filter(
				GCCRMLead.tenant_id == self.tenant_id,
				GCCRMLead.created_on >= last_30_days,
				GCCRMLead.is_active == True
			).count()
			
			# Opportunity metrics
			pipeline_value = self.db.query(func.sum(GCCRMOpportunity.expected_revenue)).filter(
				GCCRMOpportunity.tenant_id == self.tenant_id,
				GCCRMOpportunity.is_active == True,
				GCCRMOpportunity.stage.notin_([OpportunityStage.CLOSED_WON, OpportunityStage.CLOSED_LOST])
			).scalar() or 0
			
			# Customer metrics
			total_customers = self.db.query(GCCRMCustomer).filter(
				GCCRMCustomer.tenant_id == self.tenant_id,
				GCCRMCustomer.is_active == True
			).count()
			
			# Activity metrics
			pending_activities = self.db.query(GCCRMActivity).filter(
				GCCRMActivity.tenant_id == self.tenant_id,
				GCCRMActivity.activity_status == ActivityStatus.PENDING,
				GCCRMActivity.due_date >= today
			).count()
			
			# Recent activities
			recent_activities = self.db.query(GCCRMActivity).filter(
				GCCRMActivity.tenant_id == self.tenant_id
			).order_by(desc(GCCRMActivity.created_on)).limit(10).all()
			
			dashboard_data = {
				'key_metrics': {
					'total_leads_30d': total_leads,
					'pipeline_value': float(pipeline_value),
					'total_customers': total_customers,
					'pending_activities': pending_activities
				},
				'recent_activities': [
					{
						'id': activity.id,
						'subject': activity.subject,
						'type': activity.activity_type.value if activity.activity_type else None,
						'due_date': activity.due_date.isoformat() if activity.due_date else None,
						'status': activity.activity_status.value if activity.activity_status else None
					}
					for activity in recent_activities
				],
				'generated_at': datetime.utcnow().isoformat()
			}
			
			return dashboard_data
			
		except Exception as e:
			logger.error(f"Error generating dashboard data: {str(e)}")
			raise CRMServiceError(f"Failed to generate dashboard: {str(e)}")
	
	def health_check(self) -> Dict[str, Any]:
		"""Perform health check of CRM service"""
		try:
			# Test database connectivity
			self.db.execute(text("SELECT 1"))
			
			# Test AI services if available
			ai_status = 'available' if AI_ENABLED else 'disabled'
			
			# Test celery connectivity
			celery_status = 'unknown'  # Would need proper celery inspection
			
			return {
				'status': 'healthy',
				'database': 'connected',
				'ai_services': ai_status,
				'background_processing': celery_status,
				'timestamp': datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			logger.error(f"Health check failed: {str(e)}")
			return {
				'status': 'unhealthy',
				'error': str(e),
				'timestamp': datetime.utcnow().isoformat()
			}


# Utility functions for AI/ML integration

def _extract_lead_features(lead_data: Dict[str, Any]) -> List[float]:
	"""Extract features for ML lead scoring model"""
	features = []
	
	# Categorical features encoded as numbers
	source_encoding = {
		'website': 1, 'social_media': 2, 'email_campaign': 3,
		'trade_show': 4, 'referral': 5, 'cold_call': 6,
		'advertisement': 7, 'partner': 8, 'other': 0
	}
	features.append(source_encoding.get(lead_data.get('lead_source'), 0))
	
	# Behavioral features
	features.append(lead_data.get('website_visits', 0))
	features.append(lead_data.get('email_opens', 0))
	features.append(lead_data.get('document_downloads', 0))
	
	# Demographic features
	company_size_encoding = {'small': 1, 'medium': 2, 'large': 3, 'enterprise': 4}
	features.append(company_size_encoding.get(lead_data.get('company_size', '').lower(), 0))
	
	# Title seniority
	title = lead_data.get('job_title', '').lower()
	seniority = 0
	if any(keyword in title for keyword in ['ceo', 'cto', 'president']):
		seniority = 4
	elif any(keyword in title for keyword in ['vp', 'director']):
		seniority = 3
	elif any(keyword in title for keyword in ['manager', 'lead']):
		seniority = 2
	elif any(keyword in title for keyword in ['senior', 'sr']):
		seniority = 1
	features.append(seniority)
	
	# Budget indicators
	features.append(1 if lead_data.get('has_budget', False) else 0)
	features.append(1 if lead_data.get('decision_maker', False) else 0)
	
	return features

def _get_lead_scoring_model():
	"""Get or train lead scoring ML model"""
	# In production, this would load a pre-trained model
	# For now, return a simple model
	from sklearn.ensemble import RandomForestClassifier
	
	# This would be loaded from storage in production
	model = RandomForestClassifier(n_estimators=100, random_state=42)
	
	# Sample training (in production, use real historical data)
	# X = [[...], [...]]  # Historical lead features
	# y = [1, 0, 1, ...]  # 1 for converted, 0 for not converted
	# model.fit(X, y)
	
	return model


# Service initialization function
def create_crm_service(db_session: Session, tenant_id: str, user_id: str) -> CRMService:
	"""Factory function to create CRM service instance"""
	return CRMService(db_session, tenant_id, user_id)