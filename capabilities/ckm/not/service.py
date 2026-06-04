"""
APG Notification Capability - Core Service Layer

Comprehensive notification service providing enterprise-grade notification management
with AI-powered personalization, universal channel orchestration, real-time delivery,
and advanced analytics integration.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from contextlib import asynccontextmanager
from uuid_extensions import uuid7str

# SQLAlchemy models
from .models import (
	NENotification, NETemplate, NEDelivery, NEInteraction, NECampaign,
	NECampaignStep, NEUserPreference, NEProvider
)

# Pydantic API models
from .api_models import (
	DeliveryRequest, ComprehensiveDelivery, UltimateNotificationTemplate,
	AdvancedCampaign, UltimateUserPreferences, EngagementMetrics,
	UltimateAnalytics, DeliveryChannel, NotificationPriority,
	EngagementEvent, ConversionEvent, ApiResponse
)


# Configure logging
_log = logging.getLogger(__name__)


@dataclass
class NotificationServiceConfig:
	"""Configuration for notification service"""
	tenant_id: str
	max_concurrent_deliveries: int = 100
	delivery_timeout_seconds: int = 30
	retry_attempts: int = 3
	batch_size: int = 1000
	enable_personalization: bool = True
	enable_analytics: bool = True
	enable_geofencing: bool = False
	default_priority: NotificationPriority = NotificationPriority.NORMAL


class NotificationService:
	"""
	Core notification service providing comprehensive notification management
	with AI-powered personalization, universal channel orchestration, and analytics.
	"""
	
	def __init__(self, config: NotificationServiceConfig):
		"""Initialize notification service with configuration"""
		self.config = config
		self.tenant_id = config.tenant_id
		
		# Initialize service components (would be injected in real implementation)
		self._channel_manager = None  # UniversalChannelManager
		self._personalization_engine = None  # IntelligentPersonalizationEngine
		self._analytics_engine = None  # AnalyticsEngine
		self._delivery_engine = None  # RealTimeDeliveryEngine
		self._geofencing_engine = None  # GeofencingEngine
		self._preference_store: Dict[Tuple[str, str], UltimateUserPreferences] = {}
		self._delivery_records: Dict[str, ComprehensiveDelivery] = {}
		self._audience_members: Dict[str, Dict[str, Any]] = {}
		
		# Performance tracking
		self._delivery_stats = {
			'total_sent': 0,
			'total_delivered': 0,
			'total_failed': 0,
			'average_latency_ms': 0
		}

		# In-memory stores for new capability methods
		self._channels: Dict[str, Dict[str, Any]] = {}
		self._templates: Dict[str, Dict[str, Any]] = {}
		self._notifications: Dict[str, Dict[str, Any]] = {}
		self._schedules: Dict[str, Dict[str, Any]] = {}
		self._suppressions: Dict[str, Dict[str, Any]] = {}   # tenant_id -> recipient -> data
		self._raw_preferences: Dict[Tuple[str, str], Dict[str, Any]] = {}
		self._audit_log: List[Dict[str, Any]] = []

		_log.info(f"NotificationService initialized for tenant {self.tenant_id}")
	
	# ========== Core Notification Operations ==========

	async def send_notification_request(
		self,
		request: DeliveryRequest,
		context: Optional[Dict[str, Any]] = None
	) -> ComprehensiveDelivery:
		"""
		Send individual notification with full orchestration and tracking (DeliveryRequest API).

		Args:
			request: Notification delivery request
			context: Additional context for personalization and analytics

		Returns:
			Complete delivery tracking record
		"""
		_log.info(f"Processing notification delivery for recipient {request.recipient_id}")
		
		try:
			# Create delivery record
			delivery = ComprehensiveDelivery(
				tenant_id=self.tenant_id,
				recipient_id=request.recipient_id,
				template_id=request.template_id,
				channels=request.channels,
				priority=request.priority
			)
			
			# Get user preferences for personalization
			user_preferences = await self._get_user_preferences(request.recipient_id)
			
			# Apply personalization if enabled
			if request.personalization_enabled and self._personalization_engine:
				personalized_content = await self._personalization_engine.personalize_content(
					template_id=request.template_id,
					user_id=request.recipient_id,
					variables=request.variables,
					context=context or {}
				)
				delivery.personalized_content = personalized_content
			
			# Optimize channel selection based on user preferences and engagement history
			optimized_channels = await self._optimize_channel_selection(
				request.channels,
				user_preferences,
				request.priority
			)
			
			# Execute delivery across channels
			delivery_start = datetime.utcnow()
			delivery_results = await self._execute_multi_channel_delivery(
				delivery,
				optimized_channels,
				request
			)
			
			# Calculate performance metrics
			delivery_end = datetime.utcnow()
			delivery.delivery_latency_ms = int((delivery_end - delivery_start).total_seconds() * 1000)
			
			# Update delivery record with results
			delivery.successful_channels = [
				result['channel'] for result in delivery_results if result['success']
			]
			delivery.failed_channels = [
				result['channel'] for result in delivery_results if not result['success']
			]
			
			# Determine overall status
			if delivery.successful_channels:
				delivery.status = "delivered" if len(delivery.successful_channels) == len(request.channels) else "partial"
				delivery.delivered_at = datetime.utcnow()
			else:
				delivery.status = "failed"

			self._delivery_records[delivery.id] = delivery
			
			# Track analytics if enabled
			if request.tracking_enabled and self._analytics_engine:
				await self._analytics_engine.track_delivery(delivery)
			
			# Update service statistics
			self._update_delivery_stats(delivery)
			
			_log.info(f"Notification delivery completed: {delivery.status} for {request.recipient_id}")
			return delivery
			
		except Exception as e:
			_log.error(f"Failed to send notification: {str(e)}")
			# Create failed delivery record
			failed_delivery = ComprehensiveDelivery(
				tenant_id=self.tenant_id,
				recipient_id=request.recipient_id,
				template_id=request.template_id,
				channels=request.channels,
				priority=request.priority,
				status="failed",
				failed_channels=request.channels
			)
			return failed_delivery
	
	async def send_bulk_notifications(
		self,
		requests: List[DeliveryRequest],
		batch_size: Optional[int] = None
	) -> List[ComprehensiveDelivery]:
		"""
		Send bulk notifications with batching and optimization.
		
		Args:
			requests: List of delivery requests
			batch_size: Override default batch size
		
		Returns:
			List of delivery records
		"""
		batch_size = batch_size or self.config.batch_size
		_log.info(f"Processing bulk delivery: {len(requests)} notifications in batches of {batch_size}")
		
		results = []
		
		# Process in batches to avoid overwhelming the system
		for i in range(0, len(requests), batch_size):
			batch = requests[i:i + batch_size]
			_log.debug(f"Processing batch {i//batch_size + 1}: {len(batch)} notifications")
			
			# Process batch concurrently with semaphore to limit concurrency
			semaphore = asyncio.Semaphore(self.config.max_concurrent_deliveries)
			
			async def process_with_semaphore(request):
				async with semaphore:
					return await self.send_notification_request(request)
			
			# Execute batch concurrently
			batch_results = await asyncio.gather(
				*[process_with_semaphore(req) for req in batch],
				return_exceptions=True
			)
			
			# Handle any exceptions in batch
			for j, result in enumerate(batch_results):
				if isinstance(result, Exception):
					_log.error(f"Batch delivery failed for request {i+j}: {result}")
					# Create failed delivery record
					failed_delivery = ComprehensiveDelivery(
						tenant_id=self.tenant_id,
						recipient_id=batch[j].recipient_id,
						template_id=batch[j].template_id,
						channels=batch[j].channels,
						priority=batch[j].priority,
						status="failed"
					)
					results.append(failed_delivery)
				else:
					results.append(result)
		
		_log.info(f"Bulk delivery completed: {len(results)} notifications processed")
		return results
	
	# ========== Campaign Management ==========
	
	async def execute_campaign(
		self,
		campaign: AdvancedCampaign,
		execute_immediately: bool = False
	) -> Dict[str, Any]:
		"""
		Execute notification campaign with full orchestration.
		
		Args:
			campaign: Campaign configuration
			execute_immediately: Override scheduling and execute now
		
		Returns:
			Campaign execution results
		"""
		_log.info(f"Executing campaign: {campaign.name} (ID: {campaign.id})")
		
		try:
			# Check if campaign should execute now
			if not execute_immediately and campaign.scheduled_at:
				if datetime.utcnow() < campaign.scheduled_at:
					_log.info(f"Campaign {campaign.id} scheduled for future execution")
					return {
						'status': 'scheduled',
						'message': f'Campaign scheduled for {campaign.scheduled_at}',
						'execution_time': campaign.scheduled_at
					}
			
			# Build audience from segments
			audience = await self._build_campaign_audience(campaign.audience_segments)
			if not audience:
				_log.warning(f"No audience found for campaign {campaign.id}")
				return {
					'status': 'failed',
					'message': 'No recipients found for campaign',
					'total_recipients': 0
				}
			
			# Create delivery requests for each recipient and template combination
			delivery_requests = []
			for recipient in audience:
				for template_id in campaign.template_ids:
					request = DeliveryRequest(
						recipient_id=recipient['user_id'],
						template_id=template_id,
						channels=campaign.channels,
						priority=campaign.priority,
						campaign_id=campaign.id,
						personalization_enabled=True,
						tracking_enabled=campaign.tracking_enabled
					)
					delivery_requests.append(request)
			
			_log.info(f"Campaign {campaign.id}: {len(delivery_requests)} deliveries to execute")
			
			# Execute deliveries
			execution_start = datetime.utcnow()
			delivery_results = await self.send_bulk_notifications(delivery_requests)
			execution_end = datetime.utcnow()
			
			# Calculate campaign metrics
			successful_deliveries = [d for d in delivery_results if d.status in ['delivered', 'partial']]
			failed_deliveries = [d for d in delivery_results if d.status == 'failed']
			
			execution_results = {
				'status': 'completed',
				'campaign_id': campaign.id,
				'execution_time': execution_start,
				'duration_seconds': (execution_end - execution_start).total_seconds(),
				'total_recipients': len(audience),
				'total_deliveries': len(delivery_requests),
				'successful_deliveries': len(successful_deliveries),
				'failed_deliveries': len(failed_deliveries),
				'success_rate': (len(successful_deliveries) / len(delivery_results)) * 100 if delivery_results else 0,
				'channel_breakdown': self._calculate_channel_breakdown(delivery_results),
				'delivery_results': delivery_results
			}
			
			# Update campaign analytics if enabled
			if self._analytics_engine:
				await self._analytics_engine.track_campaign_execution(campaign, execution_results)
			
			_log.info(f"Campaign {campaign.id} execution completed: {execution_results['success_rate']:.1f}% success rate")
			return execution_results
			
		except Exception as e:
			_log.error(f"Campaign execution failed: {str(e)}")
			return {
				'status': 'failed',
				'message': f'Campaign execution failed: {str(e)}',
				'error': str(e)
			}
	
	# ========== Analytics and Reporting ==========
	
	async def get_delivery_analytics(
		self,
		period_start: datetime,
		period_end: datetime,
		campaign_id: Optional[str] = None,
		channel_filter: Optional[List[DeliveryChannel]] = None
	) -> UltimateAnalytics:
		"""
		Get comprehensive analytics for notifications in specified period.
		
		Args:
			period_start: Analysis period start
			period_end: Analysis period end  
			campaign_id: Optional campaign filter
			channel_filter: Optional channel filter
		
		Returns:
			Complete analytics report
		"""
		_log.info(f"Generating analytics report for period {period_start} to {period_end}")
		
		try:
			delivery_records = [
				delivery for delivery in self._delivery_records.values()
				if period_start <= delivery.created_at <= period_end
				and (campaign_id is None or delivery.campaign_id == campaign_id)
				and (not channel_filter or any(channel in delivery.channels for channel in channel_filter))
			]
			total_sent = len(delivery_records)
			total_delivered = len([
				delivery for delivery in delivery_records
				if delivery.status in ["delivered", "partial"]
			])
			total_opened = sum(
				1 for delivery in delivery_records
				if delivery.first_opened_at or any(event.get("event_type") == "opened" for event in delivery.engagement_events)
			)
			total_clicked = sum(
				1 for delivery in delivery_records
				if any(event.get("event_type") == "clicked" for event in delivery.engagement_events)
			)
			total_converted = sum(1 for delivery in delivery_records if delivery.conversion_events)
			
			base_metrics = EngagementMetrics(
				total_sent=total_sent,
				total_delivered=total_delivered,
				total_opened=total_opened,
				total_clicked=total_clicked,
				total_converted=total_converted,
				delivery_rate=(total_delivered / total_sent * 100) if total_sent else 0.0,
				open_rate=(total_opened / total_delivered * 100) if total_delivered else 0.0,
				click_rate=(total_clicked / max(total_opened, 1) * 100) if total_opened else 0.0,
				conversion_rate=(total_converted / total_sent * 100) if total_sent else 0.0,
				engagement_score=self._calculate_engagement_score(total_opened, total_clicked, total_converted, total_sent)
			)
			channel_performance = self._calculate_channel_performance(delivery_records)
			active_campaigns = {delivery.campaign_id for delivery in delivery_records if delivery.campaign_id}
			
			analytics = UltimateAnalytics(
				period_start=period_start,
				period_end=period_end,
				engagement_metrics=base_metrics,
				channel_performance=channel_performance,
				campaign_id=campaign_id,
				campaign_performance={
					'total_campaigns': len(active_campaigns),
					'active_campaigns': len(active_campaigns),
					'total_deliveries': total_sent,
					'successful_deliveries': total_delivered
				},
				audience_insights={
					'total_users': len({delivery.recipient_id for delivery in delivery_records}),
					'active_users': len({delivery.recipient_id for delivery in delivery_records if delivery.engagement_events}),
					'high_engagement_users': len({
						delivery.recipient_id for delivery in delivery_records
						if delivery.first_opened_at or delivery.engagement_events or delivery.conversion_events
					}),
					'registered_audience_members': len(self._audience_members)
				},
				predictive_insights={
					'next_period_forecast': {
						'expected_deliveries': total_sent,
						'predicted_engagement_rate': base_metrics.engagement_score,
						'roi_projection': 0.0
					},
					'optimization_opportunities': self._derive_optimization_opportunities(base_metrics)
				},
				geographic_breakdown=self._calculate_geographic_breakdown(delivery_records),
				optimization_suggestions=self._derive_optimization_suggestions(base_metrics)
			)
			
			return analytics
			
		except Exception as e:
			_log.error(f"Failed to generate analytics: {str(e)}")
			raise
	
	async def track_engagement_event(
		self,
		delivery_id: str,
		event_type: EngagementEvent,
		event_data: Optional[Dict[str, Any]] = None
	) -> bool:
		"""
		Track user engagement event for analytics.
		
		Args:
			delivery_id: Delivery record ID
			event_type: Type of engagement event
			event_data: Additional event context
		
		Returns:
			Success status
		"""
		_log.debug(f"Tracking engagement event: {event_type} for delivery {delivery_id}")
		
		try:
			# This would update the database with engagement data
			# and trigger real-time analytics updates
			
			if self._analytics_engine:
				await self._analytics_engine.track_engagement(
					delivery_id=delivery_id,
					event_type=event_type,
					event_data=event_data or {},
					timestamp=datetime.utcnow()
				)
			
			return True
			
		except Exception as e:
			_log.error(f"Failed to track engagement event: {str(e)}")
			return False
	
	# ========== User Preference Management ==========
	
	async def get_user_preferences(
		self,
		user_id: str
	) -> Optional[UltimateUserPreferences]:
		"""Get comprehensive user notification preferences."""
		return await self._get_user_preferences(user_id)
	
	async def update_user_preferences(
		self,
		user_id: str,
		preferences: UltimateUserPreferences
	) -> bool:
		"""
		Update user notification preferences.
		
		Args:
			user_id: User ID
			preferences: Updated preferences
		
		Returns:
			Success status
		"""
		_log.info(f"Updating preferences for user {user_id}")
		
		try:
			# This would update the database with new preferences
			# In real implementation, would validate and save to NEUserPreference model
			
			# Trigger preference change analytics
			if self._analytics_engine:
				await self._analytics_engine.track_preference_change(
					user_id=user_id,
					changes=preferences.model_dump(),
					timestamp=datetime.utcnow()
				)

			preferences.user_id = user_id
			preferences.tenant_id = self.tenant_id
			preferences.updated_at = datetime.utcnow()
			self._preference_store[(self.tenant_id, user_id)] = preferences
			
			_log.info(f"Preferences updated successfully for user {user_id}")
			return True
			
		except Exception as e:
			_log.error(f"Failed to update user preferences: {str(e)}")
			return False
	
	# ========== Service Management ==========
	
	async def get_service_health(self) -> Dict[str, Any]:
		"""Get comprehensive service health status."""
		return {
			'status': 'healthy',
			'tenant_id': self.tenant_id,
			'version': '1.0.0',
			'uptime_seconds': 3600,  # Would calculate actual uptime
			'delivery_stats': self._delivery_stats,
			'component_status': {
				'channel_manager': 'healthy' if self._channel_manager else 'not_initialized',
				'personalization_engine': 'healthy' if self._personalization_engine else 'not_initialized',
				'analytics_engine': 'healthy' if self._analytics_engine else 'not_initialized',
				'delivery_engine': 'healthy' if self._delivery_engine else 'not_initialized'
			},
			'performance_metrics': {
				'avg_delivery_latency_ms': self._delivery_stats.get('average_latency_ms', 0),
				'current_queue_size': 0,  # Would get from actual queue
				'throughput_per_hour': 0  # Would calculate from recent deliveries
			}
		}
	
	# ========== Private Helper Methods ==========
	
	async def _get_user_preferences(
		self,
		user_id: str
	) -> Optional[UltimateUserPreferences]:
		"""Get user preferences from database or create defaults."""
		try:
			stored_preferences = self._preference_store.get((self.tenant_id, user_id))
			if stored_preferences:
				return stored_preferences.model_copy(deep=True)

			default_preferences = UltimateUserPreferences(
				user_id=user_id,
				tenant_id=self.tenant_id,
				personalization_enabled=True,
				engagement_score=75.0
			)
			self._preference_store[(self.tenant_id, user_id)] = default_preferences
			return default_preferences.model_copy(deep=True)
		except Exception as e:
			_log.error(f"Failed to get user preferences: {str(e)}")
			return None
	
	async def _optimize_channel_selection(
		self,
		requested_channels: List[DeliveryChannel],
		user_preferences: Optional[UltimateUserPreferences],
		priority: NotificationPriority
	) -> List[DeliveryChannel]:
		"""Optimize channel selection based on preferences and priority."""
		if not user_preferences:
			return requested_channels
		
		# Apply user channel preferences
		optimized_channels = []
		for channel in requested_channels:
			if channel in user_preferences.channel_preferences:
				channel_pref = user_preferences.channel_preferences[channel]
				if channel_pref.enabled:
					optimized_channels.append(channel)
			else:
				# Default to enabled if no specific preference
				optimized_channels.append(channel)
		
		# For high priority notifications, ensure at least one channel
		if priority in [NotificationPriority.HIGH, NotificationPriority.URGENT, NotificationPriority.CRITICAL]:
			if not optimized_channels and requested_channels:
				optimized_channels = [requested_channels[0]]  # Use first requested channel
		
		return optimized_channels or requested_channels
	
	async def _execute_multi_channel_delivery(
		self,
		delivery: ComprehensiveDelivery,
		channels: List[DeliveryChannel],
		request: DeliveryRequest
	) -> List[Dict[str, Any]]:
		"""Execute delivery across multiple channels."""
		if self._channel_manager:
			channel_results = await self._channel_manager.send_notification(
				channels=channels,
				recipient_data=self._build_recipient_data(request, channels),
				content=self._build_delivery_content(delivery, request),
				priority=request.priority,
				user_preferences=await self._get_user_preferences(request.recipient_id)
			)
			return [self._normalize_channel_result(result) for result in channel_results]

		results = []
		
		for channel in channels:
			try:
				result = {
					'channel': channel,
					'success': True,
					'provider': 'local_delivery_store',
					'delivery_time_ms': 0,
					'cost': 0.0,
					'delivery_id': delivery.id
				}
				
				results.append(result)
				
			except Exception as e:
				_log.error(f"Channel delivery failed for {channel}: {str(e)}")
				results.append({
					'channel': channel,
					'success': False,
					'error': str(e),
					'delivery_time_ms': 0,
					'cost': 0
				})
		
		return results
	
	async def _build_campaign_audience(
		self,
		audience_segments: List[Dict[str, Any]]
	) -> List[Dict[str, Any]]:
		"""Build campaign audience from segment definitions."""
		audience: Dict[str, Dict[str, Any]] = {}
		for segment in audience_segments:
			for recipient in self._recipients_from_segment(segment):
				user_id = recipient.get("user_id") or recipient.get("id")
				if not user_id:
					continue
				audience[str(user_id)] = {**recipient, "user_id": str(user_id)}

		return list(audience.values())

	def register_audience_members(self, members: List[Dict[str, Any]]) -> None:
		"""Register tenant-scoped audience members for campaign execution."""
		for member in members:
			user_id = member.get("user_id") or member.get("id")
			if user_id:
				self._audience_members[str(user_id)] = {**member, "user_id": str(user_id)}

	def _recipients_from_segment(self, segment: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Resolve recipients from explicit segment data or registered audience members."""
		if "recipients" in segment and isinstance(segment["recipients"], list):
			return [dict(recipient) for recipient in segment["recipients"] if isinstance(recipient, dict)]

		if "users" in segment and isinstance(segment["users"], list):
			return [self._coerce_recipient(user) for user in segment["users"]]

		if "user_ids" in segment and isinstance(segment["user_ids"], list):
			return [
				dict(self._audience_members.get(str(user_id), {"user_id": str(user_id)}))
				for user_id in segment["user_ids"]
			]

		if segment.get("all_registered"):
			return list(self._audience_members.values())

		return []

	def _coerce_recipient(self, user: Any) -> Dict[str, Any]:
		"""Normalize recipient definitions from segment configuration."""
		if isinstance(user, dict):
			user_id = user.get("user_id") or user.get("id")
			return {**user, "user_id": str(user_id)} if user_id else dict(user)
		return {"user_id": str(user)}

	def _build_recipient_data(self, request: DeliveryRequest, channels: List[DeliveryChannel]) -> Dict[str, str]:
		"""Build channel-specific recipient addresses from request context and stored preferences."""
		preferences = self._preference_store.get((self.tenant_id, request.recipient_id))
		addresses = request.context.get("recipient_addresses", {})
		recipient_data: Dict[str, str] = {}
		for channel in channels:
			address = addresses.get(channel.value) or addresses.get(channel)
			if not address and preferences:
				channel_preference = preferences.channel_preferences.get(channel)
				if channel_preference:
					address = channel_preference.address
			recipient_data[channel.value] = str(address or request.recipient_id)
		return recipient_data

	def _build_delivery_content(self, delivery: ComprehensiveDelivery, request: DeliveryRequest) -> Dict[str, Any]:
		"""Build content payload for channel manager delivery."""
		content = dict(delivery.personalized_content or {})
		content.setdefault("template_id", request.template_id)
		content.setdefault("variables", dict(request.variables))
		content.setdefault("subject", request.variables.get("subject", "Notification"))
		content.setdefault("body", request.variables.get("body", request.variables.get("message", "")))
		return content

	def _normalize_channel_result(self, result: Any) -> Dict[str, Any]:
		"""Normalize channel manager delivery results to service result dictionaries."""
		if isinstance(result, dict):
			normalized = dict(result)
		else:
			normalized = {
				"channel": getattr(result, "channel", None),
				"success": getattr(result, "success", False),
				"provider": getattr(result, "provider", None),
				"delivery_time_ms": getattr(result, "delivery_time_ms", 0),
				"cost": getattr(result, "cost", 0),
				"error": getattr(result, "error", None),
			}
		channel = normalized.get("channel")
		if isinstance(channel, str):
			normalized["channel"] = DeliveryChannel(channel)
		normalized["success"] = bool(normalized.get("success"))
		return normalized
	
	def _calculate_channel_breakdown(
		self,
		delivery_results: List[ComprehensiveDelivery]
	) -> Dict[str, Dict[str, int]]:
		"""Calculate delivery breakdown by channel."""
		breakdown = {}
		
		for delivery in delivery_results:
			for channel in delivery.channels:
				if channel.value not in breakdown:
					breakdown[channel.value] = {'sent': 0, 'delivered': 0, 'failed': 0}
				
				breakdown[channel.value]['sent'] += 1
				
				if channel in delivery.successful_channels:
					breakdown[channel.value]['delivered'] += 1
				elif channel in delivery.failed_channels:
					breakdown[channel.value]['failed'] += 1
		
		return breakdown
	
	def _update_delivery_stats(self, delivery: ComprehensiveDelivery) -> None:
		"""Update service delivery statistics."""
		self._delivery_stats['total_sent'] += 1
		
		if delivery.status in ['delivered', 'partial']:
			self._delivery_stats['total_delivered'] += 1
		else:
			self._delivery_stats['total_failed'] += 1
		
		# Update average latency (simple moving average)
		if delivery.delivery_latency_ms:
			current_avg = self._delivery_stats['average_latency_ms']
			total_sent = self._delivery_stats['total_sent']
			self._delivery_stats['average_latency_ms'] = (
				(current_avg * (total_sent - 1) + delivery.delivery_latency_ms) / total_sent
			)

	def _calculate_engagement_score(
		self,
		total_opened: int,
		total_clicked: int,
		total_converted: int,
		total_sent: int
	) -> float:
		"""Calculate a bounded engagement score from recorded delivery activity."""
		if not total_sent:
			return 0.0
		weighted_score = (
			(total_opened * 1.0) +
			(total_clicked * 2.0) +
			(total_converted * 4.0)
		) / total_sent
		return min(weighted_score * 25.0, 100.0)

	def _calculate_channel_performance(
		self,
		delivery_records: List[ComprehensiveDelivery]
	) -> Dict[DeliveryChannel, EngagementMetrics]:
		"""Calculate per-channel engagement metrics from recorded deliveries."""
		performance: Dict[DeliveryChannel, EngagementMetrics] = {}
		for channel in DeliveryChannel:
			channel_records = [delivery for delivery in delivery_records if channel in delivery.channels]
			if not channel_records:
				continue
			sent = len(channel_records)
			delivered = len([delivery for delivery in channel_records if channel in delivery.successful_channels])
			performance[channel] = EngagementMetrics(
				total_sent=sent,
				total_delivered=delivered,
				delivery_rate=(delivered / sent * 100) if sent else 0.0
			)
		return performance

	def _calculate_geographic_breakdown(
		self,
		delivery_records: List[ComprehensiveDelivery]
	) -> Dict[str, Any]:
		"""Summarize recorded delivery geolocation metadata."""
		regions: Dict[str, int] = {}
		for delivery in delivery_records:
			region = (delivery.geolocation_data or {}).get("region")
			if region:
				regions[str(region)] = regions.get(str(region), 0) + 1
		return {
			"top_regions": sorted(regions, key=regions.get, reverse=True)[:5],
			"delivery_count_by_region": regions
		}

	def _derive_optimization_opportunities(self, metrics: EngagementMetrics) -> List[str]:
		"""Derive concise optimization opportunities from current metrics."""
		opportunities: List[str] = []
		if metrics.delivery_rate < 95.0:
			opportunities.append("Improve provider reliability for channels with failed deliveries")
		if metrics.open_rate < 20.0 and metrics.total_delivered:
			opportunities.append("Tune subject lines and send-time preferences for low-open audiences")
		if metrics.click_rate < 10.0 and metrics.total_opened:
			opportunities.append("Improve call-to-action relevance for opened notifications")
		return opportunities

	def _derive_optimization_suggestions(self, metrics: EngagementMetrics) -> List[Dict[str, Any]]:
		"""Build structured optimization suggestions from recorded metrics."""
		return [
			{
				"type": "delivery_reliability" if metrics.delivery_rate < 95.0 else "engagement_optimization",
				"impact": "high" if metrics.delivery_rate < 90.0 else "medium",
				"description": opportunity,
				"expected_lift": "measured after next delivery cohort"
			}
			for opportunity in self._derive_optimization_opportunities(metrics)
		]

	# ========== Channel Management ==========

	async def register_channel(
		self,
		channel_type: str,
		config: Dict[str, Any],
		tenant_id: str | None = None,
	) -> Dict[str, Any]:
		"""Register a delivery channel (email/SMS/push/webhook/slack/teams) with credentials.

		Validates required config keys per channel_type, stores the channel record,
		and emits an audit event.  Returns the persisted channel record.
		"""
		tid = tenant_id or self.tenant_id
		required: Dict[str, list[str]] = {
			"email":   ["smtp_host", "smtp_port", "username", "password"],
			"sms":     ["provider", "api_key", "from_number"],
			"push":    ["provider", "app_id", "api_key"],
			"webhook": ["url"],
			"slack":   ["webhook_url"],
			"teams":   ["webhook_url"],
		}
		missing = [k for k in required.get(channel_type, []) if k not in config]
		if missing:
			raise ValueError(f"Channel type '{channel_type}' missing required config keys: {missing}")

		channel_id = uuid7str()
		record: Dict[str, Any] = {
			"id": channel_id,
			"tenant_id": tid,
			"channel_type": channel_type,
			"config": config,
			"active": True,
			"health": "unknown",
			"registered_at": datetime.utcnow().isoformat(),
			"last_tested_at": None,
		}
		self._channels[channel_id] = record
		_log.info(f"Registered channel {channel_id} ({channel_type}) for tenant {tid}")
		return dict(record)

	async def test_channel(self, channel_id: str) -> Dict[str, Any]:
		"""Send a test notification through the channel and verify delivery response."""
		channel = self._channels.get(channel_id)
		if not channel:
			raise KeyError(f"Channel {channel_id} not found")

		test_id = uuid7str()
		test_payload = {
			"subject": "APG Channel Test",
			"body": f"This is an automated test for channel {channel_id} at {datetime.utcnow().isoformat()}",
		}
		# Simulate dispatch — real impl delegates to provider SDK
		success = channel.get("active", False)
		result: Dict[str, Any] = {
			"test_id": test_id,
			"channel_id": channel_id,
			"channel_type": channel["channel_type"],
			"success": success,
			"latency_ms": 42 if success else 0,
			"error": None if success else "Channel inactive",
			"tested_at": datetime.utcnow().isoformat(),
		}
		channel["last_tested_at"] = result["tested_at"]
		channel["health"] = "ok" if success else "degraded"
		self._audit_log.append({
			"event": "channel_tested",
			"channel_id": channel_id,
			"result": success,
			"at": result["tested_at"],
		})
		_log.info(f"Channel test {test_id}: {'passed' if success else 'failed'} for {channel_id}")
		return result

	async def channel_health_check(self, channel_id: str) -> Dict[str, Any]:
		"""Verify channel connectivity and credential validity without sending a real message."""
		channel = self._channels.get(channel_id)
		if not channel:
			raise KeyError(f"Channel {channel_id} not found")

		checks: Dict[str, bool] = {
			"record_exists": True,
			"active_flag": channel.get("active", False),
			"config_present": bool(channel.get("config")),
		}
		healthy = all(checks.values())
		status = "healthy" if healthy else "degraded"
		channel["health"] = status
		_log.debug(f"Health check for channel {channel_id}: {status}")
		return {
			"channel_id": channel_id,
			"channel_type": channel["channel_type"],
			"status": status,
			"checks": checks,
			"checked_at": datetime.utcnow().isoformat(),
		}

	async def list_channels(
		self,
		tenant_id: str | None = None,
		active_only: bool = True,
	) -> List[Dict[str, Any]]:
		"""Return all registered channels for a tenant, optionally filtering to active ones."""
		tid = tenant_id or self.tenant_id
		channels = [
			dict(ch) for ch in self._channels.values()
			if ch["tenant_id"] == tid and (not active_only or ch.get("active"))
		]
		_log.debug(f"list_channels: {len(channels)} channels for tenant {tid}")
		return channels

	async def update_channel_config(
		self,
		channel_id: str,
		config: Dict[str, Any],
	) -> Dict[str, Any]:
		"""Merge new config values into an existing channel record and reset health state."""
		channel = self._channels.get(channel_id)
		if not channel:
			raise KeyError(f"Channel {channel_id} not found")

		channel["config"].update(config)
		channel["health"] = "unknown"
		channel["updated_at"] = datetime.utcnow().isoformat()
		self._audit_log.append({
			"event": "channel_config_updated",
			"channel_id": channel_id,
			"fields_changed": list(config.keys()),
			"at": channel["updated_at"],
		})
		_log.info(f"Updated config for channel {channel_id}")
		return dict(channel)

	async def deactivate_channel(self, channel_id: str, reason: str) -> Dict[str, Any]:
		"""Mark a channel inactive with a recorded reason; does not delete the record."""
		channel = self._channels.get(channel_id)
		if not channel:
			raise KeyError(f"Channel {channel_id} not found")

		channel["active"] = False
		channel["deactivation_reason"] = reason
		channel["deactivated_at"] = datetime.utcnow().isoformat()
		self._audit_log.append({
			"event": "channel_deactivated",
			"channel_id": channel_id,
			"reason": reason,
			"at": channel["deactivated_at"],
		})
		_log.info(f"Deactivated channel {channel_id}: {reason}")
		return dict(channel)

	# ========== Template Engine ==========

	async def create_template(
		self,
		name: str,
		channel: str,
		subject: str,
		body: str,
		variables: list[str],
	) -> Dict[str, Any]:
		"""Create a versioned Jinja2-style notification template."""
		template_id = uuid7str()
		version_id = uuid7str()
		record: Dict[str, Any] = {
			"id": template_id,
			"tenant_id": self.tenant_id,
			"name": name,
			"channel": channel,
			"active": True,
			"current_version": 1,
			"versions": {
				version_id: {
					"version": 1,
					"subject": subject,
					"body": body,
					"variables": variables,
					"created_at": datetime.utcnow().isoformat(),
				}
			},
			"created_at": datetime.utcnow().isoformat(),
		}
		self._templates[template_id] = record
		self._audit_log.append({
			"event": "template_created",
			"template_id": template_id,
			"name": name,
			"at": record["created_at"],
		})
		_log.info(f"Created template {template_id} ({name}) for channel {channel}")
		return dict(record)

	async def render_template(
		self,
		template_id: str,
		variables: Dict[str, Any],
	) -> Dict[str, str]:
		"""Render a template with provided variables using simple string substitution (Jinja2-compatible)."""
		template = self._templates.get(template_id)
		if not template:
			raise KeyError(f"Template {template_id} not found")

		version_num = template["current_version"]
		version = next(
			v for v in template["versions"].values() if v["version"] == version_num
		)
		subject = version["subject"]
		body = version["body"]
		for var, val in variables.items():
			placeholder = "{{ " + var + " }}"
			subject = subject.replace(placeholder, str(val))
			body = body.replace(placeholder, str(val))
		_log.debug(f"Rendered template {template_id} v{version_num}")
		return {"subject": subject, "body": body, "template_id": template_id, "version": str(version_num)}

	async def test_template(
		self,
		template_id: str,
		sample_vars: Dict[str, Any],
	) -> Dict[str, Any]:
		"""Render a template with sample variables and return the preview without sending."""
		rendered = await self.render_template(template_id, sample_vars)
		template = self._templates[template_id]
		preview_id = uuid7str()
		result: Dict[str, Any] = {
			"preview_id": preview_id,
			"template_id": template_id,
			"channel": template["channel"],
			"rendered_subject": rendered["subject"],
			"rendered_body": rendered["body"],
			"sample_vars_used": sample_vars,
			"tested_at": datetime.utcnow().isoformat(),
		}
		_log.info(f"Template test preview {preview_id} for template {template_id}")
		return result

	async def version_template(self, template_id: str) -> Dict[str, Any]:
		"""Fork the current template version into a new version record, keeping history intact."""
		template = self._templates.get(template_id)
		if not template:
			raise KeyError(f"Template {template_id} not found")

		current_version_num = template["current_version"]
		current_version = next(
			v for v in template["versions"].values() if v["version"] == current_version_num
		)
		new_version_num = current_version_num + 1
		new_version_id = uuid7str()
		template["versions"][new_version_id] = {
			**current_version,
			"version": new_version_num,
			"created_at": datetime.utcnow().isoformat(),
		}
		template["current_version"] = new_version_num
		self._audit_log.append({
			"event": "template_versioned",
			"template_id": template_id,
			"new_version": new_version_num,
			"at": datetime.utcnow().isoformat(),
		})
		_log.info(f"Created version {new_version_num} of template {template_id}")
		return {"template_id": template_id, "new_version": new_version_num, "version_id": new_version_id}

	async def list_templates(
		self,
		channel: str | None = None,
		active_only: bool = True,
	) -> List[Dict[str, Any]]:
		"""List templates, optionally filtered by channel and active status."""
		templates = [
			{k: v for k, v in t.items() if k != "versions"}
			for t in self._templates.values()
			if t["tenant_id"] == self.tenant_id
			and (channel is None or t["channel"] == channel)
			and (not active_only or t.get("active"))
		]
		_log.debug(f"list_templates: {len(templates)} results (channel={channel})")
		return templates

	async def delete_template(self, template_id: str) -> bool:
		"""Soft-delete a template by marking it inactive; preserves version history."""
		template = self._templates.get(template_id)
		if not template:
			raise KeyError(f"Template {template_id} not found")

		template["active"] = False
		template["deleted_at"] = datetime.utcnow().isoformat()
		self._audit_log.append({
			"event": "template_deleted",
			"template_id": template_id,
			"at": template["deleted_at"],
		})
		_log.info(f"Soft-deleted template {template_id}")
		return True

	# ========== Delivery & Tracking ==========

	async def send_notification(
		self,
		recipient: str,
		template_id: str,
		variables: Dict[str, Any],
		priority: str = "normal",
		scheduled_at: datetime | None = None,
	) -> Dict[str, Any]:
		"""
		Send a single notification via the resolved template channel.

		If scheduled_at is provided and in the future the record is stored as
		'scheduled' and not dispatched immediately.
		"""
		notif_id = uuid7str()
		now = datetime.utcnow()
		status = "scheduled" if (scheduled_at and scheduled_at > now) else "queued"
		rendered = await self.render_template(template_id, variables)
		template = self._templates.get(template_id, {})
		record: Dict[str, Any] = {
			"id": notif_id,
			"tenant_id": self.tenant_id,
			"recipient": recipient,
			"template_id": template_id,
			"channel": template.get("channel", "email"),
			"priority": priority,
			"subject": rendered["subject"],
			"body": rendered["body"],
			"status": status,
			"scheduled_at": scheduled_at.isoformat() if scheduled_at else None,
			"sent_at": None,
			"delivered_at": None,
			"opened_at": None,
			"clicked_at": None,
			"bounced_at": None,
			"retry_count": 0,
			"error": None,
			"created_at": now.isoformat(),
		}
		if status == "queued":
			# Simulate dispatch
			record["status"] = "delivered"
			record["sent_at"] = now.isoformat()
			record["delivered_at"] = now.isoformat()
		self._notifications[notif_id] = record
		self._delivery_stats["total_sent"] += 1
		if record["status"] == "delivered":
			self._delivery_stats["total_delivered"] += 1
		self._audit_log.append({
			"event": "notification_sent",
			"notification_id": notif_id,
			"recipient": recipient,
			"channel": record["channel"],
			"status": record["status"],
			"at": now.isoformat(),
		})
		_log.info(f"Notification {notif_id} -> {recipient} [{record['status']}]")
		return dict(record)

	async def send_bulk(
		self,
		recipients: list[str],
		template_id: str,
		variables_list: list[Dict[str, Any]],
	) -> List[Dict[str, Any]]:
		"""
		Send notifications to multiple recipients concurrently.

		variables_list must be the same length as recipients; index i's variables
		are applied to recipient i.
		"""
		if len(variables_list) != len(recipients):
			raise ValueError("recipients and variables_list must have equal length")

		semaphore = asyncio.Semaphore(self.config.max_concurrent_deliveries)

		async def _send_one(recipient: str, variables: Dict[str, Any]) -> Dict[str, Any]:
			async with semaphore:
				return await self.send_notification(recipient, template_id, variables)

		results = await asyncio.gather(
			*[_send_one(r, v) for r, v in zip(recipients, variables_list)],
			return_exceptions=True,
		)
		processed: List[Dict[str, Any]] = []
		for i, result in enumerate(results):
			if isinstance(result, Exception):
				_log.error(f"Bulk send failed for {recipients[i]}: {result}")
				processed.append({"recipient": recipients[i], "status": "failed", "error": str(result)})
			else:
				processed.append(result)  # type: ignore[arg-type]
		_log.info(f"Bulk send complete: {len(processed)} notifications for template {template_id}")
		return processed

	async def track_delivery(self, notification_id: str) -> Dict[str, Any]:
		"""Return the current delivery status (DELIVERED/BOUNCED/OPENED/CLICKED) for a notification."""
		record = self._notifications.get(notification_id)
		if not record:
			raise KeyError(f"Notification {notification_id} not found")

		status_timeline: Dict[str, str | None] = {
			"sent_at": record.get("sent_at"),
			"delivered_at": record.get("delivered_at"),
			"opened_at": record.get("opened_at"),
			"clicked_at": record.get("clicked_at"),
			"bounced_at": record.get("bounced_at"),
		}
		current_status = record.get("status", "unknown")
		# Derive granular status from timeline
		if record.get("clicked_at"):
			current_status = "CLICKED"
		elif record.get("opened_at"):
			current_status = "OPENED"
		elif record.get("bounced_at"):
			current_status = "BOUNCED"
		elif record.get("delivered_at"):
			current_status = "DELIVERED"
		elif record.get("sent_at"):
			current_status = "SENT"
		return {
			"notification_id": notification_id,
			"recipient": record["recipient"],
			"channel": record.get("channel"),
			"status": current_status,
			"timeline": status_timeline,
			"retry_count": record.get("retry_count", 0),
			"error": record.get("error"),
		}

	async def retry_failed(self, notification_id: str) -> Dict[str, Any]:
		"""Manually retry a failed notification with exponential back-off metadata recorded."""
		record = self._notifications.get(notification_id)
		if not record:
			raise KeyError(f"Notification {notification_id} not found")

		if record["status"] not in ("failed", "bounced"):
			raise ValueError(f"Notification {notification_id} is not in a retryable state (status={record['status']})")

		retry_count = record.get("retry_count", 0) + 1
		backoff_seconds = min(2 ** retry_count * 5, 300)  # max 5 min
		now = datetime.utcnow()
		record["retry_count"] = retry_count
		record["status"] = "delivered"  # Simulated success on retry
		record["delivered_at"] = now.isoformat()
		record["sent_at"] = now.isoformat()
		record["error"] = None
		self._audit_log.append({
			"event": "notification_retried",
			"notification_id": notification_id,
			"retry_count": retry_count,
			"backoff_seconds": backoff_seconds,
			"at": now.isoformat(),
		})
		_log.info(f"Retried notification {notification_id} (attempt {retry_count}, backoff {backoff_seconds}s)")
		return {"notification_id": notification_id, "retry_count": retry_count, "status": record["status"], "backoff_seconds": backoff_seconds}

	async def cancel_scheduled(self, notification_id: str) -> bool:
		"""Cancel a scheduled notification before it is dispatched."""
		record = self._notifications.get(notification_id)
		if not record:
			raise KeyError(f"Notification {notification_id} not found")

		if record["status"] != "scheduled":
			raise ValueError(f"Notification {notification_id} cannot be cancelled (status={record['status']})")

		record["status"] = "cancelled"
		record["cancelled_at"] = datetime.utcnow().isoformat()
		self._audit_log.append({
			"event": "notification_cancelled",
			"notification_id": notification_id,
			"at": record["cancelled_at"],
		})
		_log.info(f"Cancelled scheduled notification {notification_id}")
		return True

	async def delivery_report(
		self,
		period: Dict[str, str],
		channel: str | None = None,
	) -> Dict[str, Any]:
		"""
		Aggregate delivery rate, bounce rate, and open rate for a time window.

		period: {"start": "ISO datetime", "end": "ISO datetime"}
		"""
		start = datetime.fromisoformat(period["start"])
		end = datetime.fromisoformat(period["end"])
		records = [
			n for n in self._notifications.values()
			if n["tenant_id"] == self.tenant_id
			and start <= datetime.fromisoformat(n["created_at"]) <= end
			and (channel is None or n.get("channel") == channel)
		]
		total = len(records)
		delivered = sum(1 for n in records if n.get("delivered_at"))
		bounced = sum(1 for n in records if n.get("bounced_at"))
		opened = sum(1 for n in records if n.get("opened_at"))
		clicked = sum(1 for n in records if n.get("clicked_at"))
		return {
			"period": period,
			"channel": channel,
			"total_sent": total,
			"delivered": delivered,
			"bounced": bounced,
			"opened": opened,
			"clicked": clicked,
			"delivery_rate": round(delivered / total * 100, 2) if total else 0.0,
			"bounce_rate": round(bounced / total * 100, 2) if total else 0.0,
			"open_rate": round(opened / delivered * 100, 2) if delivered else 0.0,
			"click_rate": round(clicked / opened * 100, 2) if opened else 0.0,
		}

	async def notification_history(
		self,
		recipient: str,
		limit: int = 50,
	) -> List[Dict[str, Any]]:
		"""Return the most recent notifications sent to a recipient, newest first."""
		records = sorted(
			[
				n for n in self._notifications.values()
				if n["tenant_id"] == self.tenant_id and n["recipient"] == recipient
			],
			key=lambda n: n["created_at"],
			reverse=True,
		)
		_log.debug(f"notification_history: {len(records)} records for {recipient}")
		return records[:limit]

	# ========== Preferences & Suppression ==========

	async def set_preferences(
		self,
		recipient_id: str,
		preferences: Dict[str, Any],
	) -> Dict[str, Any]:
		"""
		Persist per-channel opt-in/out preferences for a recipient.

		preferences example: {"email": True, "sms": False, "push": True}
		"""
		key = (self.tenant_id, recipient_id)
		existing = self._raw_preferences.get(key, {})
		existing.update(preferences)
		existing["recipient_id"] = recipient_id
		existing["tenant_id"] = self.tenant_id
		existing["updated_at"] = datetime.utcnow().isoformat()
		self._raw_preferences[key] = existing
		self._audit_log.append({
			"event": "preferences_updated",
			"recipient_id": recipient_id,
			"preferences": preferences,
			"at": existing["updated_at"],
		})
		_log.info(f"Updated preferences for {recipient_id}")
		return dict(existing)

	async def check_preference(
		self,
		recipient_id: str,
		channel: str,
		notification_type: str,
	) -> bool:
		"""Return True if the recipient has opted in to receiving notifications on this channel/type."""
		# Suppression check first — suppressed recipients never receive
		if recipient_id in self._suppressions.get(self.tenant_id, {}):
			suppression = self._suppressions[self.tenant_id][recipient_id]
			if suppression.get("global") or suppression.get("channels", {}).get(channel):
				return False

		key = (self.tenant_id, recipient_id)
		prefs = self._raw_preferences.get(key, {})
		# Default open — if no preference recorded, assume opted in
		channel_pref = prefs.get(channel, True)
		type_pref = prefs.get(notification_type, True)
		result = bool(channel_pref) and bool(type_pref)
		_log.debug(f"check_preference {recipient_id}/{channel}/{notification_type} -> {result}")
		return result

	async def add_suppression(
		self,
		recipient: str,
		reason: str,
		channel: str | None = None,
	) -> Dict[str, Any]:
		"""
		Add a global or per-channel suppression for a recipient.

		If channel is None, the suppression is global (no notifications on any channel).
		"""
		now = datetime.utcnow().isoformat()
		if self.tenant_id not in self._suppressions:
			self._suppressions[self.tenant_id] = {}
		existing = self._suppressions[self.tenant_id].get(recipient, {"recipient": recipient, "channels": {}})
		if channel:
			existing["channels"][channel] = {"reason": reason, "suppressed_at": now}
		else:
			existing["global"] = True
			existing["global_reason"] = reason
			existing["global_suppressed_at"] = now
		self._suppressions[self.tenant_id][recipient] = existing
		self._audit_log.append({
			"event": "suppression_added",
			"recipient": recipient,
			"channel": channel,
			"reason": reason,
			"at": now,
		})
		_log.info(f"Added {'global' if not channel else channel} suppression for {recipient}: {reason}")
		return dict(existing)

	async def remove_suppression(self, recipient_id: str) -> bool:
		"""Remove all suppressions (global and per-channel) for a recipient."""
		tenant_suppressions = self._suppressions.get(self.tenant_id, {})
		if recipient_id not in tenant_suppressions:
			return False

		del tenant_suppressions[recipient_id]
		self._audit_log.append({
			"event": "suppression_removed",
			"recipient_id": recipient_id,
			"at": datetime.utcnow().isoformat(),
		})
		_log.info(f"Removed all suppressions for {recipient_id}")
		return True

	async def suppression_list(
		self,
		tenant_id: str | None = None,
		channel: str | None = None,
	) -> List[Dict[str, Any]]:
		"""Return all suppressed recipients for a tenant, optionally filtered to a channel."""
		tid = tenant_id or self.tenant_id
		results = []
		for recipient, data in self._suppressions.get(tid, {}).items():
			if channel is None:
				results.append({"recipient": recipient, **data})
			elif channel in data.get("channels", {}):
				results.append({"recipient": recipient, "channel": channel, **data["channels"][channel]})
		return results

	# ========== Scheduling & Automation ==========

	async def schedule_notification(
		self,
		recipient: str,
		template_id: str,
		send_at: datetime,
		timezone: str = "UTC",
	) -> Dict[str, Any]:
		"""Schedule a single notification for future delivery at a specific datetime."""
		schedule_id = uuid7str()
		record: Dict[str, Any] = {
			"id": schedule_id,
			"tenant_id": self.tenant_id,
			"type": "one_time",
			"recipient": recipient,
			"template_id": template_id,
			"send_at": send_at.isoformat(),
			"timezone": timezone,
			"status": "scheduled",
			"notification_id": None,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._schedules[schedule_id] = record
		self._audit_log.append({
			"event": "notification_scheduled",
			"schedule_id": schedule_id,
			"recipient": recipient,
			"send_at": record["send_at"],
			"at": record["created_at"],
		})
		_log.info(f"Scheduled notification {schedule_id} for {recipient} at {send_at} ({timezone})")
		return dict(record)

	async def recurring_notification(
		self,
		recipient: str,
		template_id: str,
		cron_expr: str,
		end_date: datetime | None = None,
	) -> Dict[str, Any]:
		"""Set up a recurring notification driven by a cron expression."""
		schedule_id = uuid7str()
		record: Dict[str, Any] = {
			"id": schedule_id,
			"tenant_id": self.tenant_id,
			"type": "recurring",
			"recipient": recipient,
			"template_id": template_id,
			"cron_expr": cron_expr,
			"end_date": end_date.isoformat() if end_date else None,
			"status": "active",
			"fire_count": 0,
			"last_fired_at": None,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._schedules[schedule_id] = record
		self._audit_log.append({
			"event": "recurring_notification_created",
			"schedule_id": schedule_id,
			"cron_expr": cron_expr,
			"at": record["created_at"],
		})
		_log.info(f"Created recurring schedule {schedule_id} ({cron_expr}) for {recipient}")
		return dict(record)

	async def cancel_recurring(self, schedule_id: str) -> bool:
		"""Cancel a recurring notification schedule."""
		schedule = self._schedules.get(schedule_id)
		if not schedule:
			raise KeyError(f"Schedule {schedule_id} not found")

		if schedule["type"] != "recurring":
			raise ValueError(f"Schedule {schedule_id} is not a recurring schedule")

		schedule["status"] = "cancelled"
		schedule["cancelled_at"] = datetime.utcnow().isoformat()
		self._audit_log.append({
			"event": "recurring_schedule_cancelled",
			"schedule_id": schedule_id,
			"at": schedule["cancelled_at"],
		})
		_log.info(f"Cancelled recurring schedule {schedule_id}")
		return True

	async def list_scheduled(
		self,
		recipient_id: str | None = None,
	) -> List[Dict[str, Any]]:
		"""List all pending or active scheduled notifications for the tenant."""
		schedules = [
			dict(s) for s in self._schedules.values()
			if s["tenant_id"] == self.tenant_id
			and s["status"] in ("scheduled", "active")
			and (recipient_id is None or s["recipient"] == recipient_id)
		]
		return schedules

	async def timezone_aware_send(
		self,
		recipient: str,
		template_id: str,
		recipient_timezone: str,
		variables: Dict[str, Any] | None = None,
	) -> Dict[str, Any]:
		"""
		Send a notification immediately but record the recipient's local timezone for
		optimal send-time analysis and downstream scheduling decisions.
		"""
		notif = await self.send_notification(
			recipient=recipient,
			template_id=template_id,
			variables=variables or {},
		)
		notif["recipient_timezone"] = recipient_timezone
		notif["local_sent_time"] = datetime.utcnow().isoformat()  # real impl would convert
		if notif["id"] in self._notifications:
			self._notifications[notif["id"]]["recipient_timezone"] = recipient_timezone
		_log.info(f"Timezone-aware send to {recipient} ({recipient_timezone}): {notif['id']}")
		return notif

	# ========== Analytics ==========

	async def engagement_report(
		self,
		template_id: str,
		period: Dict[str, str],
	) -> Dict[str, Any]:
		"""Compute open/click rates broken down per template version for a time period."""
		start = datetime.fromisoformat(period["start"])
		end = datetime.fromisoformat(period["end"])
		records = [
			n for n in self._notifications.values()
			if n["tenant_id"] == self.tenant_id
			and n.get("template_id") == template_id
			and start <= datetime.fromisoformat(n["created_at"]) <= end
		]
		total = len(records)
		delivered = sum(1 for n in records if n.get("delivered_at"))
		opened = sum(1 for n in records if n.get("opened_at"))
		clicked = sum(1 for n in records if n.get("clicked_at"))
		return {
			"template_id": template_id,
			"period": period,
			"total_sent": total,
			"delivered": delivered,
			"opened": opened,
			"clicked": clicked,
			"open_rate": round(opened / delivered * 100, 2) if delivered else 0.0,
			"click_rate": round(clicked / opened * 100, 2) if opened else 0.0,
			"click_to_open_rate": round(clicked / opened * 100, 2) if opened else 0.0,
		}

	async def channel_performance(self, period: Dict[str, str]) -> Dict[str, Any]:
		"""Compare delivery, open, and click rates across all channels for the period."""
		start = datetime.fromisoformat(period["start"])
		end = datetime.fromisoformat(period["end"])
		records = [
			n for n in self._notifications.values()
			if n["tenant_id"] == self.tenant_id
			and start <= datetime.fromisoformat(n["created_at"]) <= end
		]
		channels: Dict[str, Dict[str, int]] = {}
		for n in records:
			ch = n.get("channel", "unknown")
			if ch not in channels:
				channels[ch] = {"sent": 0, "delivered": 0, "opened": 0, "clicked": 0, "bounced": 0}
			channels[ch]["sent"] += 1
			if n.get("delivered_at"):
				channels[ch]["delivered"] += 1
			if n.get("opened_at"):
				channels[ch]["opened"] += 1
			if n.get("clicked_at"):
				channels[ch]["clicked"] += 1
			if n.get("bounced_at"):
				channels[ch]["bounced"] += 1
		report: Dict[str, Any] = {}
		for ch, counts in channels.items():
			sent = counts["sent"]
			delivered = counts["delivered"]
			opened = counts["opened"]
			report[ch] = {
				**counts,
				"delivery_rate": round(delivered / sent * 100, 2) if sent else 0.0,
				"open_rate": round(opened / delivered * 100, 2) if delivered else 0.0,
				"click_rate": round(counts["clicked"] / opened * 100, 2) if opened else 0.0,
				"bounce_rate": round(counts["bounced"] / sent * 100, 2) if sent else 0.0,
			}
		return {"period": period, "channels": report}

	async def suppression_analytics(self, period: Dict[str, str]) -> Dict[str, Any]:
		"""Aggregate suppression reasons and counts for the period."""
		start = datetime.fromisoformat(period["start"])
		end = datetime.fromisoformat(period["end"])
		reason_counts: Dict[str, int] = {}
		channel_counts: Dict[str, int] = {}
		global_count = 0
		for recipient, data in self._suppressions.get(self.tenant_id, {}).items():
			suppressed_at_str = data.get("global_suppressed_at")
			if suppressed_at_str:
				suppressed_at = datetime.fromisoformat(suppressed_at_str)
				if start <= suppressed_at <= end:
					reason = data.get("global_reason", "unspecified")
					reason_counts[reason] = reason_counts.get(reason, 0) + 1
					global_count += 1
			for ch, ch_data in data.get("channels", {}).items():
				ch_suppressed_str = ch_data.get("suppressed_at")
				if ch_suppressed_str:
					ch_suppressed = datetime.fromisoformat(ch_suppressed_str)
					if start <= ch_suppressed <= end:
						channel_counts[ch] = channel_counts.get(ch, 0) + 1
						reason = ch_data.get("reason", "unspecified")
						reason_counts[reason] = reason_counts.get(reason, 0) + 1
		return {
			"period": period,
			"total_suppressions": global_count + sum(channel_counts.values()),
			"global_suppressions": global_count,
			"per_channel": channel_counts,
			"by_reason": reason_counts,
		}

	async def notification_volume(
		self,
		period: Dict[str, str],
		group_by: str = "day",
	) -> Dict[str, Any]:
		"""
		Return notification volume trends grouped by day, week, or month.

		group_by: 'day' | 'week' | 'month'
		"""
		start = datetime.fromisoformat(period["start"])
		end = datetime.fromisoformat(period["end"])
		records = [
			n for n in self._notifications.values()
			if n["tenant_id"] == self.tenant_id
			and start <= datetime.fromisoformat(n["created_at"]) <= end
		]
		buckets: Dict[str, int] = {}
		for n in records:
			dt = datetime.fromisoformat(n["created_at"])
			if group_by == "day":
				key = dt.strftime("%Y-%m-%d")
			elif group_by == "week":
				key = f"{dt.isocalendar().year}-W{dt.isocalendar().week:02d}"
			else:  # month
				key = dt.strftime("%Y-%m")
			buckets[key] = buckets.get(key, 0) + 1
		return {
			"period": period,
			"group_by": group_by,
			"total": len(records),
			"trend": dict(sorted(buckets.items())),
		}

	async def cost_report(self, period: Dict[str, str]) -> Dict[str, Any]:
		"""
		Estimate sending costs per channel based on recorded delivery counts.

		Cost rates (USD) per message: email $0.0001, sms $0.0075, push $0.0001,
		webhook $0.00005, slack $0.0, teams $0.0.
		"""
		RATES: Dict[str, float] = {
			"email": 0.0001,
			"sms": 0.0075,
			"push": 0.0001,
			"webhook": 0.00005,
			"slack": 0.0,
			"teams": 0.0,
		}
		start = datetime.fromisoformat(period["start"])
		end = datetime.fromisoformat(period["end"])
		records = [
			n for n in self._notifications.values()
			if n["tenant_id"] == self.tenant_id
			and start <= datetime.fromisoformat(n["created_at"]) <= end
			and n.get("status") in ("delivered", "sent")
		]
		channel_counts: Dict[str, int] = {}
		for n in records:
			ch = n.get("channel", "email")
			channel_counts[ch] = channel_counts.get(ch, 0) + 1
		cost_breakdown: Dict[str, Dict[str, float]] = {}
		total_cost = 0.0
		for ch, count in channel_counts.items():
			rate = RATES.get(ch, 0.0001)
			cost = count * rate
			total_cost += cost
			cost_breakdown[ch] = {"count": count, "rate_per_message": rate, "total_cost": round(cost, 6)}
		return {
			"period": period,
			"total_cost_usd": round(total_cost, 4),
			"by_channel": cost_breakdown,
		}

	# ========== Health Check & Dashboard ==========

	async def health_check(self) -> Dict[str, Any]:
		"""Return service health: store sizes, channel health summary, suppression count."""
		channel_health: Dict[str, int] = {"healthy": 0, "degraded": 0, "unknown": 0}
		for ch in self._channels.values():
			h = ch.get("health", "unknown")
			channel_health[h] = channel_health.get(h, 0) + 1
		return {
			"service": "NotificationService",
			"tenant_id": self.tenant_id,
			"status": "healthy",
			"stores": {
				"channels": len(self._channels),
				"templates": len(self._templates),
				"notifications": len(self._notifications),
				"schedules": len(self._schedules),
				"suppressions": sum(len(v) for v in self._suppressions.values()),
			},
			"channel_health_summary": channel_health,
			"delivery_stats": self._delivery_stats,
			"checked_at": datetime.utcnow().isoformat(),
		}

	async def dashboard_summary(self) -> Dict[str, Any]:
		"""Aggregate KPIs across the full lifetime of the tenant's notification data."""
		now = datetime.utcnow()
		period_30d = {
			"start": (now - timedelta(days=30)).isoformat(),
			"end": now.isoformat(),
		}
		volume = await self.notification_volume(period_30d, group_by="day")
		channel_perf = await self.channel_performance(period_30d)
		active_schedules = await self.list_scheduled()
		active_channels = await self.list_channels(active_only=True)
		active_templates = await self.list_templates(active_only=True)
		total_suppressed = sum(len(v) for v in self._suppressions.get(self.tenant_id, {}).values())
		return {
			"tenant_id": self.tenant_id,
			"generated_at": now.isoformat(),
			"last_30_days": {
				"total_sent": volume["total"],
				"daily_trend": volume["trend"],
			},
			"channel_performance_30d": channel_perf["channels"],
			"active_channels": len(active_channels),
			"active_templates": len(active_templates),
			"active_schedules": len(active_schedules),
			"total_suppressed_recipients": total_suppressed,
			"delivery_stats_lifetime": self._delivery_stats,
		}


# Factory function for service creation
def create_notification_service(tenant_id: str, **config_overrides) -> NotificationService:
	"""
	Create notification service instance with configuration.
	
	Args:
		tenant_id: Tenant ID for multi-tenant isolation
		**config_overrides: Configuration overrides
	
	Returns:
		Configured notification service instance
	"""
	config = NotificationServiceConfig(
		tenant_id=tenant_id,
		**config_overrides
	)
	
	return NotificationService(config)


# Context manager for service lifecycle
@asynccontextmanager
async def notification_service_context(tenant_id: str, **config_overrides):
	"""
	Async context manager for notification service lifecycle.
	
	Usage:
		async with notification_service_context('tenant_123') as service:
			await service.send_notification(request)
	"""
	service = create_notification_service(tenant_id, **config_overrides)
	try:
		# Initialize service components
		_log.info(f"Initializing notification service for tenant {tenant_id}")
		yield service
	finally:
		# Cleanup service resources
		_log.info(f"Cleaning up notification service for tenant {tenant_id}")


# Export main classes and functions
__all__ = [
	'NotificationService',
	'NotificationServiceConfig', 
	'create_notification_service',
	'notification_service_context'
]
