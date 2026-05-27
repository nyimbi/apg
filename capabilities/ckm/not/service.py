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
		
		_log.info(f"NotificationService initialized for tenant {self.tenant_id}")
	
	# ========== Core Notification Operations ==========
	
	async def send_notification(
		self,
		request: DeliveryRequest,
		context: Optional[Dict[str, Any]] = None
	) -> ComprehensiveDelivery:
		"""
		Send individual notification with full orchestration and tracking.
		
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
					return await self.send_notification(request)
			
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
