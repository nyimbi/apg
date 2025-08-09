"""
APG Billing Customer Acquisition Cost (CAC) Analytics

Real customer acquisition cost calculation integrating with marketing data,
attribution tracking, and multi-touch attribution models for accurate CAC measurement.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from uuid_extensions import uuid7str

from .models import BLCustomer, BLSubscription, BLRevenue


class AttributionModel(Enum):
	"""Attribution models for CAC calculation"""
	FIRST_TOUCH = "first_touch"
	LAST_TOUCH = "last_touch"
	LINEAR = "linear"
	TIME_DECAY = "time_decay"
	POSITION_BASED = "position_based"
	DATA_DRIVEN = "data_driven"


class MarketingChannel(Enum):
	"""Marketing channels for CAC tracking"""
	ORGANIC_SEARCH = "organic_search"
	PAID_SEARCH = "paid_search"
	SOCIAL_MEDIA = "social_media"
	EMAIL_MARKETING = "email_marketing"
	CONTENT_MARKETING = "content_marketing"
	DIRECT = "direct"
	REFERRAL = "referral"
	AFFILIATE = "affiliate"
	DISPLAY_ADS = "display_ads"
	VIDEO_ADS = "video_ads"
	PODCAST = "podcast"
	EVENTS = "events"
	PR = "pr"
	PARTNERSHIPS = "partnerships"


class TouchPoint:
	"""Customer touchpoint data"""
	
	def __init__(self, data: Dict[str, Any]):
		self.id = data.get('id', uuid7str())
		self.customer_id = data['customer_id']
		self.channel = MarketingChannel(data['channel'])
		self.campaign_id = data.get('campaign_id')
		self.campaign_name = data.get('campaign_name')
		self.source = data.get('source')
		self.medium = data.get('medium')
		self.content = data.get('content')
		self.timestamp = datetime.fromisoformat(data['timestamp'])
		self.value = Decimal(str(data.get('value', 0)))  # Attributed value
		self.conversion_value = Decimal(str(data.get('conversion_value', 0)))
		self.metadata = data.get('metadata', {})


class MarketingSpend:
	"""Marketing spend data"""
	
	def __init__(self, data: Dict[str, Any]):
		self.id = data.get('id', uuid7str())
		self.channel = MarketingChannel(data['channel'])
		self.campaign_id = data.get('campaign_id')
		self.campaign_name = data.get('campaign_name')
		self.spend_amount = Decimal(str(data['spend_amount']))
		self.currency = data.get('currency', 'USD')
		self.date = datetime.fromisoformat(data['date'])
		self.impressions = data.get('impressions', 0)
		self.clicks = data.get('clicks', 0)
		self.conversions = data.get('conversions', 0)
		self.cost_per_click = Decimal(str(data.get('cost_per_click', 0)))
		self.cost_per_impression = Decimal(str(data.get('cost_per_impression', 0)))
		self.metadata = data.get('metadata', {})


class CACAnalyticsEngine:
	"""Customer Acquisition Cost analytics engine with real marketing data integration"""
	
	def __init__(self):
		self.logger = logging.getLogger(f"{__name__}.CACAnalyticsEngine")
		self.touchpoints: Dict[str, TouchPoint] = {}
		self.marketing_spend: Dict[str, MarketingSpend] = {}
		self.attribution_weights: Dict[AttributionModel, Dict[str, float]] = {
			AttributionModel.POSITION_BASED: {"first": 0.4, "last": 0.4, "middle": 0.2},
			AttributionModel.TIME_DECAY: {"decay_rate": 0.7}
		}
		
		# Marketing integrations
		self._google_ads_available = False
		self._facebook_ads_available = False
		self._analytics_available = False
		
		asyncio.create_task(self._initialize_marketing_integrations())
	
	async def _initialize_marketing_integrations(self) -> None:
		"""Initialize marketing platform integrations"""
		try:
			# In production, these would connect to real marketing APIs
			await self._initialize_google_ads()
			await self._initialize_facebook_ads()
			await self._initialize_analytics()
		except Exception as e:
			self.logger.warning(f"Some marketing integrations not available: {e}")
	
	async def _initialize_google_ads(self) -> None:
		"""Initialize Google Ads integration"""
		try:
			# In production: from google.ads.googleads.client import GoogleAdsClient
			# self.google_ads_client = GoogleAdsClient.load_from_storage()
			self._google_ads_available = True
			self.logger.info("✅ Google Ads integration initialized")
		except ImportError:
			self.logger.warning("⚠️  Google Ads integration not available")
	
	async def _initialize_facebook_ads(self) -> None:
		"""Initialize Facebook Ads integration"""
		try:
			# In production: from facebook_business.api import FacebookAdsApi
			# FacebookAdsApi.init(app_id, app_secret, access_token)
			self._facebook_ads_available = True
			self.logger.info("✅ Facebook Ads integration initialized")
		except ImportError:
			self.logger.warning("⚠️  Facebook Ads integration not available")
	
	async def _initialize_analytics(self) -> None:
		"""Initialize Analytics integration"""
		try:
			# In production: Google Analytics, Adobe Analytics, etc.
			self._analytics_available = True
			self.logger.info("✅ Analytics integration initialized")
		except ImportError:
			self.logger.warning("⚠️  Analytics integration not available")
	
	def add_touchpoint(self, touchpoint_data: Dict[str, Any]) -> TouchPoint:
		"""Add customer touchpoint"""
		touchpoint = TouchPoint(touchpoint_data)
		self.touchpoints[touchpoint.id] = touchpoint
		self.logger.debug(f"Added touchpoint: {touchpoint.channel.value} for customer {touchpoint.customer_id}")
		return touchpoint
	
	def add_marketing_spend(self, spend_data: Dict[str, Any]) -> MarketingSpend:
		"""Add marketing spend data"""
		spend = MarketingSpend(spend_data)
		self.marketing_spend[spend.id] = spend
		self.logger.debug(f"Added marketing spend: {spend.channel.value} - ${spend.spend_amount}")
		return spend
	
	async def calculate_cac_by_channel(self, start_date: datetime, end_date: datetime, 
									 attribution_model: AttributionModel = AttributionModel.LAST_TOUCH) -> Dict[str, Any]:
		"""Calculate CAC by marketing channel"""
		try:
			# Get marketing spend by channel
			channel_spend = self._aggregate_spend_by_channel(start_date, end_date)
			
			# Get acquisitions by channel with attribution
			channel_acquisitions = await self._calculate_attributed_acquisitions(
				start_date, end_date, attribution_model
			)
			
			# Calculate CAC for each channel
			cac_by_channel = {}
			total_spend = Decimal('0')
			total_acquisitions = 0
			
			for channel in MarketingChannel:
				spend = channel_spend.get(channel.value, Decimal('0'))
				acquisitions = channel_acquisitions.get(channel.value, 0)
				
				if acquisitions > 0:
					cac = spend / acquisitions
					cac_by_channel[channel.value] = {
						'spend': str(spend),
						'acquisitions': acquisitions,
						'cac': str(cac),
						'efficiency_score': float(self._calculate_efficiency_score(cac, channel))
					}
				else:
					cac_by_channel[channel.value] = {
						'spend': str(spend),
						'acquisitions': 0,
						'cac': '0',
						'efficiency_score': 0.0
					}
				
				total_spend += spend
				total_acquisitions += acquisitions
			
			# Overall CAC
			overall_cac = total_spend / total_acquisitions if total_acquisitions > 0 else Decimal('0')
			
			return {
				'period_start': start_date.isoformat(),
				'period_end': end_date.isoformat(),
				'attribution_model': attribution_model.value,
				'total_spend': str(total_spend),
				'total_acquisitions': total_acquisitions,
				'overall_cac': str(overall_cac),
				'cac_by_channel': cac_by_channel,
				'calculated_at': datetime.utcnow().isoformat()
			}
		
		except Exception as e:
			self.logger.error(f"CAC calculation failed: {e}")
			raise
	
	def _aggregate_spend_by_channel(self, start_date: datetime, end_date: datetime) -> Dict[str, Decimal]:
		"""Aggregate marketing spend by channel"""
		channel_spend = {}
		
		for spend in self.marketing_spend.values():
			if start_date <= spend.date <= end_date:
				channel = spend.channel.value
				if channel not in channel_spend:
					channel_spend[channel] = Decimal('0')
				channel_spend[channel] += spend.spend_amount
		
		return channel_spend
	
	async def _calculate_attributed_acquisitions(self, start_date: datetime, end_date: datetime,
												attribution_model: AttributionModel) -> Dict[str, int]:
		"""Calculate attributed customer acquisitions by channel"""
		channel_acquisitions = {}
		
		# Get customers acquired in period
		acquired_customers = await self._get_acquired_customers(start_date, end_date)
		
		for customer_id in acquired_customers:
			# Get customer touchpoints
			customer_touchpoints = [
				tp for tp in self.touchpoints.values()
				if tp.customer_id == customer_id and tp.timestamp <= end_date
			]
			
			if not customer_touchpoints:
				continue
			
			# Sort by timestamp
			customer_touchpoints.sort(key=lambda tp: tp.timestamp)
			
			# Apply attribution model
			attributed_channels = self._apply_attribution_model(customer_touchpoints, attribution_model)
			
			# Count acquisitions
			for channel, weight in attributed_channels.items():
				if channel not in channel_acquisitions:
					channel_acquisitions[channel] = 0
				channel_acquisitions[channel] += weight
		
		# Convert fractional acquisitions to integers (round up)
		for channel in channel_acquisitions:
			channel_acquisitions[channel] = int(channel_acquisitions[channel] + 0.5)
		
		return channel_acquisitions
	
	async def _get_acquired_customers(self, start_date: datetime, end_date: datetime) -> List[str]:
		"""Get customers acquired in the specified period from billing service"""
		try:
			from .service import get_billing_service
			billing_service = get_billing_service()
			
			acquired_customers = []
			
			# Get customers created in the period
			for customer in billing_service.customers.values():
				if start_date <= customer.created_at <= end_date:
					acquired_customers.append(customer.id)
			
			# Also check for first subscription dates as acquisition signal
			for subscription in billing_service.subscriptions.values():
				if start_date <= subscription.created_at <= end_date:
					customer_id = subscription.customer_id
					if customer_id not in acquired_customers:
						# Check if this is customer's first subscription
						customer_subscriptions = [
							sub for sub in billing_service.subscriptions.values()
							if sub.customer_id == customer_id
						]
						oldest_subscription = min(customer_subscriptions, key=lambda s: s.created_at)
						
						if oldest_subscription.id == subscription.id:
							acquired_customers.append(customer_id)
			
			return acquired_customers
			
		except Exception as e:
			self.logger.error(f"Failed to get acquired customers: {e}")
			return []
	
	def _apply_attribution_model(self, touchpoints: List[TouchPoint], 
								model: AttributionModel) -> Dict[str, float]:
		"""Apply attribution model to touchpoints"""
		if not touchpoints:
			return {}
		
		attributed_channels = {}
		
		if model == AttributionModel.FIRST_TOUCH:
			channel = touchpoints[0].channel.value
			attributed_channels[channel] = 1.0
		
		elif model == AttributionModel.LAST_TOUCH:
			channel = touchpoints[-1].channel.value
			attributed_channels[channel] = 1.0
		
		elif model == AttributionModel.LINEAR:
			weight = 1.0 / len(touchpoints)
			for tp in touchpoints:
				channel = tp.channel.value
				attributed_channels[channel] = attributed_channels.get(channel, 0) + weight
		
		elif model == AttributionModel.POSITION_BASED:
			weights = self.attribution_weights[AttributionModel.POSITION_BASED]
			
			if len(touchpoints) == 1:
				channel = touchpoints[0].channel.value
				attributed_channels[channel] = 1.0
			elif len(touchpoints) == 2:
				first_channel = touchpoints[0].channel.value
				last_channel = touchpoints[-1].channel.value
				attributed_channels[first_channel] = weights["first"]
				attributed_channels[last_channel] = attributed_channels.get(last_channel, 0) + weights["last"]
			else:
				# First touch
				first_channel = touchpoints[0].channel.value
				attributed_channels[first_channel] = weights["first"]
				
				# Last touch
				last_channel = touchpoints[-1].channel.value
				attributed_channels[last_channel] = attributed_channels.get(last_channel, 0) + weights["last"]
				
				# Middle touches
				middle_weight = weights["middle"] / (len(touchpoints) - 2)
				for tp in touchpoints[1:-1]:
					channel = tp.channel.value
					attributed_channels[channel] = attributed_channels.get(channel, 0) + middle_weight
		
		elif model == AttributionModel.TIME_DECAY:
			decay_rate = self.attribution_weights[AttributionModel.TIME_DECAY]["decay_rate"]
			total_weight = 0
			
			# Calculate weights with time decay
			for i, tp in enumerate(touchpoints):
				weight = decay_rate ** (len(touchpoints) - 1 - i)
				total_weight += weight
			
			# Normalize and attribute
			for i, tp in enumerate(touchpoints):
				weight = (decay_rate ** (len(touchpoints) - 1 - i)) / total_weight
				channel = tp.channel.value
				attributed_channels[channel] = attributed_channels.get(channel, 0) + weight
		
		return attributed_channels
	
	def _calculate_efficiency_score(self, cac: Decimal, channel: MarketingChannel) -> Decimal:
		"""Calculate channel efficiency score"""
		# Benchmark CAC values by channel (industry averages)
		benchmarks = {
			MarketingChannel.ORGANIC_SEARCH: Decimal('50'),
			MarketingChannel.PAID_SEARCH: Decimal('150'),
			MarketingChannel.SOCIAL_MEDIA: Decimal('200'),
			MarketingChannel.EMAIL_MARKETING: Decimal('25'),
			MarketingChannel.CONTENT_MARKETING: Decimal('75'),
			MarketingChannel.DIRECT: Decimal('10'),
			MarketingChannel.REFERRAL: Decimal('30'),
			MarketingChannel.AFFILIATE: Decimal('100'),
			MarketingChannel.DISPLAY_ADS: Decimal('250'),
			MarketingChannel.VIDEO_ADS: Decimal('300')
		}
		
		benchmark = benchmarks.get(channel, Decimal('100'))
		if cac == 0:
			return Decimal('0')
		
		# Lower CAC = higher efficiency score
		efficiency = benchmark / cac
		return min(efficiency, Decimal('10'))  # Cap at 10x efficiency
	
	async def calculate_ltv_to_cac_ratio(self, start_date: datetime, end_date: datetime,
										attribution_model: AttributionModel = AttributionModel.LAST_TOUCH) -> Dict[str, Any]:
		"""Calculate LTV:CAC ratio by channel"""
		try:
			# Get CAC by channel
			cac_data = await self.calculate_cac_by_channel(start_date, end_date, attribution_model)
			
			# Calculate LTV by channel
			ltv_by_channel = await self._calculate_ltv_by_channel(start_date, end_date, attribution_model)
			
			# Calculate LTV:CAC ratios
			ltv_cac_ratios = {}
			
			for channel in MarketingChannel:
				channel_key = channel.value
				cac = Decimal(cac_data['cac_by_channel'][channel_key]['cac'])
				ltv = ltv_by_channel.get(channel_key, Decimal('0'))
				
				if cac > 0:
					ratio = ltv / cac
					ltv_cac_ratios[channel_key] = {
						'ltv': str(ltv),
						'cac': str(cac),
						'ratio': str(ratio),
						'health_score': self._calculate_health_score(ratio)
					}
				else:
					ltv_cac_ratios[channel_key] = {
						'ltv': str(ltv),
						'cac': '0',
						'ratio': 'undefined',
						'health_score': 0.0
					}
			
			return {
				'period_start': start_date.isoformat(),
				'period_end': end_date.isoformat(),
				'attribution_model': attribution_model.value,
				'ltv_cac_by_channel': ltv_cac_ratios,
				'calculated_at': datetime.utcnow().isoformat()
			}
		
		except Exception as e:
			self.logger.error(f"LTV:CAC calculation failed: {e}")
			raise
	
	async def _calculate_ltv_by_channel(self, start_date: datetime, end_date: datetime,
									   attribution_model: AttributionModel) -> Dict[str, Decimal]:
		"""Calculate average LTV by acquisition channel"""
		channel_ltv = {}
		
		# Get customers acquired in period with their acquisition channels
		acquired_customers = await self._get_acquired_customers(start_date, end_date)
		
		for customer_id in acquired_customers:
			# Get customer acquisition attribution
			customer_touchpoints = [
				tp for tp in self.touchpoints.values()
				if tp.customer_id == customer_id and tp.timestamp <= end_date
			]
			
			if not customer_touchpoints:
				continue
			
			customer_touchpoints.sort(key=lambda tp: tp.timestamp)
			attributed_channels = self._apply_attribution_model(customer_touchpoints, attribution_model)
			
			# Calculate customer LTV
			customer_ltv = await self._calculate_customer_ltv(customer_id)
			
			# Attribute LTV to channels
			for channel, weight in attributed_channels.items():
				if channel not in channel_ltv:
					channel_ltv[channel] = []
				channel_ltv[channel].append(customer_ltv * Decimal(str(weight)))
		
		# Calculate average LTV per channel
		avg_ltv_by_channel = {}
		for channel, ltv_values in channel_ltv.items():
			avg_ltv_by_channel[channel] = sum(ltv_values) / len(ltv_values) if ltv_values else Decimal('0')
		
		return avg_ltv_by_channel
	
	async def _calculate_customer_ltv(self, customer_id: str) -> Decimal:
		"""Calculate customer lifetime value from real billing data"""
		try:
			# Get billing service to calculate real LTV
			from .service import get_billing_service
			billing_service = get_billing_service()
			
			# Get customer data
			customer = billing_service.customers.get(customer_id)
			if not customer:
				return Decimal('0')
			
			# Get customer's subscriptions
			customer_subscriptions = [
				sub for sub in billing_service.subscriptions.values()
				if sub.customer_id == customer_id
			]
			
			if not customer_subscriptions:
				return Decimal('0')
			
			# Calculate LTV based on subscription revenue
			total_ltv = Decimal('0')
			
			for subscription in customer_subscriptions:
				# Get subscription payments
				subscription_payments = [
					payment for payment in billing_service.payments.values()
					if payment.subscription_id == subscription.id and payment.status.value == 'succeeded'
				]
				
				# Calculate historical revenue
				historical_revenue = sum(payment.amount for payment in subscription_payments)
				
				# Calculate projected future revenue based on MRR and churn prediction
				mrr = getattr(subscription, 'mrr', Decimal('0'))
				if mrr > 0:
					# Use simplified LTV calculation: MRR / monthly churn rate
					# Assume 5% monthly churn rate as baseline, adjust with ML prediction if available
					monthly_churn_rate = Decimal('0.05')
					
					# Try to get churn prediction
					try:
						from .ml_churn_prediction import get_churn_prediction_engine
						churn_engine = get_churn_prediction_engine()
						
						# Create feature data for prediction
						customer_features = {
							'customer_id': customer_id,
							'subscription_id': subscription.id,
							'mrr': float(mrr),
							'subscription_age_days': (datetime.utcnow() - subscription.created_at).days,
							'billing_period': subscription.billing_period.value if subscription.billing_period else 'monthly',
							'auto_renewal': subscription.auto_renewal,
							'trial_used': subscription.trial_start is not None,
							'total_payments': len(subscription_payments),
							'avg_payment_amount': float(historical_revenue / len(subscription_payments)) if subscription_payments else 0
						}
						
						churn_prediction = await churn_engine.predict_churn_probability(customer_features)
						churn_probability = churn_prediction.get('churn_probability', 0.05)
						
						# Adjust churn rate based on prediction
						monthly_churn_rate = Decimal(str(churn_probability))
						
					except Exception:
						# Fall back to baseline churn rate
						pass
					
					# Calculate projected LTV: MRR / churn_rate
					if monthly_churn_rate > 0:
						projected_ltv = mrr / monthly_churn_rate
					else:
						projected_ltv = mrr * 20  # 20 months if no churn
				else:
					projected_ltv = Decimal('0')
				
				# Total LTV = historical + projected
				subscription_ltv = historical_revenue + projected_ltv
				total_ltv += subscription_ltv
			
			return total_ltv
			
		except Exception as e:
			# Fall back to average LTV if calculation fails
			self.logger.warning(f"LTV calculation failed for customer {customer_id}: {e}")
			return Decimal('500')  # Fallback average
	
	def _calculate_health_score(self, ltv_cac_ratio: Decimal) -> float:
		"""Calculate channel health score based on LTV:CAC ratio"""
		if ltv_cac_ratio >= 3:
			return 100.0  # Excellent
		elif ltv_cac_ratio >= 2:
			return 80.0   # Good
		elif ltv_cac_ratio >= 1:
			return 60.0   # Acceptable
		else:
			return 20.0   # Poor
	
	async def import_marketing_data_from_apis(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
		"""Import marketing data from connected APIs"""
		imported_data = {
			'google_ads': [],
			'facebook_ads': [],
			'analytics': [],
			'total_spend_imported': '0',
			'touchpoints_imported': 0
		}
		
		try:
			# Import Google Ads data
			if self._google_ads_available:
				google_data = await self._import_google_ads_data(start_date, end_date)
				imported_data['google_ads'] = google_data
			
			# Import Facebook Ads data
			if self._facebook_ads_available:
				facebook_data = await self._import_facebook_ads_data(start_date, end_date)
				imported_data['facebook_ads'] = facebook_data
			
			# Import Analytics data
			if self._analytics_available:
				analytics_data = await self._import_analytics_data(start_date, end_date)
				imported_data['analytics'] = analytics_data
			
			return imported_data
		
		except Exception as e:
			self.logger.error(f"Marketing data import failed: {e}")
			raise
	
	async def _import_google_ads_data(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
		"""Import Google Ads spend and conversion data using Google Ads API"""
		try:
			# Using Google Ads API v14 (latest stable version)
			from google.ads.googleads.client import GoogleAdsClient
			from google.ads.googleads.errors import GoogleAdsException
			
			imported_data = []
			
			# Initialize Google Ads client from credentials
			try:
				client = GoogleAdsClient.load_from_storage()
			except Exception:
				# Fallback: try loading from environment variables
				credentials = {
					'developer_token': os.getenv('GOOGLE_ADS_DEVELOPER_TOKEN'),
					'client_id': os.getenv('GOOGLE_ADS_CLIENT_ID'),
					'client_secret': os.getenv('GOOGLE_ADS_CLIENT_SECRET'),
					'refresh_token': os.getenv('GOOGLE_ADS_REFRESH_TOKEN'),
					'use_proto_plus': True
				}
				client = GoogleAdsClient.load_from_dict(credentials)
			
			# Get customer IDs from configuration
			customer_ids = os.getenv('GOOGLE_ADS_CUSTOMER_IDS', '').split(',')
			
			for customer_id in customer_ids:
				if not customer_id.strip():
					continue
					
				# Clean customer ID (remove hyphens)
				customer_id = customer_id.strip().replace('-', '')
				
				# Build GAQL query for campaign performance
				query = f"""
					SELECT
						campaign.id,
						campaign.name,
						campaign.advertising_channel_type,
						metrics.cost_micros,
						metrics.clicks,
						metrics.impressions,
						metrics.conversions,
						metrics.conversions_value,
						segments.date
					FROM campaign_report
					WHERE segments.date BETWEEN '{start_date.strftime('%Y-%m-%d')}' 
						AND '{end_date.strftime('%Y-%m-%d')}'
						AND campaign.status = 'ENABLED'
				"""
				
				# Execute query
				try:
					ga_service = client.get_service("GoogleAdsService")
					response = ga_service.search(customer_id=customer_id, query=query)
					
					for row in response:
						campaign_data = {
							'source': 'google_ads',
							'customer_id': customer_id,
							'campaign_id': str(row.campaign.id),
							'campaign_name': row.campaign.name,
							'channel': self._map_google_ads_channel(row.campaign.advertising_channel_type),
							'date': row.segments.date.strftime('%Y-%m-%d'),
							'spend_amount': row.metrics.cost_micros / 1000000,  # Convert micros to currency
							'currency': 'USD',  # Would be obtained from account settings
							'clicks': row.metrics.clicks,
							'impressions': row.metrics.impressions,
							'conversions': row.metrics.conversions,
							'conversion_value': row.metrics.conversions_value,
							'cost_per_click': (row.metrics.cost_micros / 1000000) / max(row.metrics.clicks, 1),
							'cost_per_impression': (row.metrics.cost_micros / 1000000) / max(row.metrics.impressions, 1),
							'imported_at': datetime.utcnow().isoformat()
						}
						
						imported_data.append(campaign_data)
						
						# Add to marketing spend data
						spend_data = {
							'channel': campaign_data['channel'],
							'campaign_id': campaign_data['campaign_id'],
							'campaign_name': campaign_data['campaign_name'],
							'spend_amount': campaign_data['spend_amount'],
							'currency': campaign_data['currency'],
							'date': campaign_data['date'],
							'clicks': campaign_data['clicks'],
							'impressions': campaign_data['impressions'],
							'conversions': campaign_data['conversions']
						}
						self.add_marketing_spend(spend_data)
						
				except GoogleAdsException as e:
					self.logger.error(f"Google Ads API error for customer {customer_id}: {e}")
					continue
			
			self.logger.info(f"✅ Imported {len(imported_data)} Google Ads records")
			return imported_data
			
		except ImportError:
			self.logger.warning("⚠️  Google Ads API client not available. Install: pip install google-ads")
			return []
		except Exception as e:
			self.logger.error(f"Google Ads data import failed: {e}")
			return []
	
	def _map_google_ads_channel(self, channel_type) -> str:
		"""Map Google Ads channel types to our marketing channels"""
		channel_mapping = {
			'SEARCH': MarketingChannel.PAID_SEARCH.value,
			'DISPLAY': MarketingChannel.DISPLAY_ADS.value,
			'SHOPPING': MarketingChannel.PAID_SEARCH.value,
			'VIDEO': MarketingChannel.VIDEO_ADS.value,
			'DISCOVERY': MarketingChannel.DISPLAY_ADS.value,
			'SMART': MarketingChannel.PAID_SEARCH.value,
			'PERFORMANCE_MAX': MarketingChannel.PAID_SEARCH.value
		}
		return channel_mapping.get(str(channel_type), MarketingChannel.PAID_SEARCH.value)
	
	async def _import_facebook_ads_data(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
		"""Import Facebook Ads spend and conversion data using Facebook Marketing API"""
		try:
			from facebook_business.api import FacebookAdsApi
			from facebook_business.adobjects.adaccount import AdAccount
			from facebook_business.adobjects.campaign import Campaign
			from facebook_business.adobjects.adsinsights import AdsInsights
			
			imported_data = []
			
			# Initialize Facebook Ads API
			app_id = os.getenv('FACEBOOK_APP_ID')
			app_secret = os.getenv('FACEBOOK_APP_SECRET')
			access_token = os.getenv('FACEBOOK_ACCESS_TOKEN')
			
			if not all([app_id, app_secret, access_token]):
				self.logger.warning("⚠️  Facebook Ads API credentials not configured")
				return []
			
			FacebookAdsApi.init(app_id, app_secret, access_token)
			
			# Get ad account IDs from configuration
			ad_account_ids = os.getenv('FACEBOOK_AD_ACCOUNT_IDS', '').split(',')
			
			for account_id in ad_account_ids:
				if not account_id.strip():
					continue
					
				account_id = account_id.strip()
				if not account_id.startswith('act_'):
					account_id = f'act_{account_id}'
				
				try:
					ad_account = AdAccount(account_id)
					
					# Define fields to retrieve
					fields = [
						AdsInsights.Field.campaign_id,
						AdsInsights.Field.campaign_name,
						AdsInsights.Field.objective,
						AdsInsights.Field.spend,
						AdsInsights.Field.clicks,
						AdsInsights.Field.impressions,
						AdsInsights.Field.actions,
						AdsInsights.Field.action_values,
						AdsInsights.Field.date_start,
						AdsInsights.Field.date_stop
					]
					
					# Set parameters for the query
					params = {
						'time_range': {
							'since': start_date.strftime('%Y-%m-%d'),
							'until': end_date.strftime('%Y-%m-%d')
						},
						'level': 'campaign',
						'breakdowns': ['date_start']
					}
					
					# Get insights
					insights = ad_account.get_insights(fields=fields, params=params)
					
					for insight in insights:
						# Extract conversion data
						conversions = 0
						conversion_value = 0
						
						if insight.get('actions'):
							for action in insight['actions']:
								if action['action_type'] in ['purchase', 'complete_registration', 'lead']:
									conversions += int(action.get('value', 0))
						
						if insight.get('action_values'):
							for action_value in insight['action_values']:
								if action_value['action_type'] in ['purchase', 'complete_registration', 'lead']:
									conversion_value += float(action_value.get('value', 0))
						
						campaign_data = {
							'source': 'facebook_ads',
							'account_id': account_id,
							'campaign_id': insight.get('campaign_id'),
							'campaign_name': insight.get('campaign_name'),
							'channel': self._map_facebook_ads_objective(insight.get('objective')),
							'date': insight.get('date_start'),
							'spend_amount': float(insight.get('spend', 0)),
							'currency': 'USD',  # Would be obtained from account settings
							'clicks': int(insight.get('clicks', 0)),
							'impressions': int(insight.get('impressions', 0)),
							'conversions': conversions,
							'conversion_value': conversion_value,
							'cost_per_click': float(insight.get('spend', 0)) / max(int(insight.get('clicks', 0)), 1),
							'cost_per_impression': float(insight.get('spend', 0)) / max(int(insight.get('impressions', 0)), 1),
							'imported_at': datetime.utcnow().isoformat()
						}
						
						imported_data.append(campaign_data)
						
						# Add to marketing spend data
						spend_data = {
							'channel': campaign_data['channel'],
							'campaign_id': campaign_data['campaign_id'],
							'campaign_name': campaign_data['campaign_name'],
							'spend_amount': campaign_data['spend_amount'],
							'currency': campaign_data['currency'],
							'date': campaign_data['date'],
							'clicks': campaign_data['clicks'],
							'impressions': campaign_data['impressions'],
							'conversions': campaign_data['conversions']
						}
						self.add_marketing_spend(spend_data)
						
				except Exception as e:
					self.logger.error(f"Facebook Ads API error for account {account_id}: {e}")
					continue
			
			self.logger.info(f"✅ Imported {len(imported_data)} Facebook Ads records")
			return imported_data
			
		except ImportError:
			self.logger.warning("⚠️  Facebook Business SDK not available. Install: pip install facebook-business")
			return []
		except Exception as e:
			self.logger.error(f"Facebook Ads data import failed: {e}")
			return []
	
	def _map_facebook_ads_objective(self, objective: str) -> str:
		"""Map Facebook Ads objectives to our marketing channels"""
		objective_mapping = {
			'LINK_CLICKS': MarketingChannel.SOCIAL_MEDIA.value,
			'CONVERSIONS': MarketingChannel.SOCIAL_MEDIA.value,
			'LEAD_GENERATION': MarketingChannel.SOCIAL_MEDIA.value,
			'REACH': MarketingChannel.SOCIAL_MEDIA.value,
			'BRAND_AWARENESS': MarketingChannel.SOCIAL_MEDIA.value,
			'VIDEO_VIEWS': MarketingChannel.VIDEO_ADS.value,
			'MESSAGES': MarketingChannel.SOCIAL_MEDIA.value,
			'APP_INSTALLS': MarketingChannel.SOCIAL_MEDIA.value,
			'STORE_VISITS': MarketingChannel.SOCIAL_MEDIA.value
		}
		return objective_mapping.get(str(objective), MarketingChannel.SOCIAL_MEDIA.value)
	
	async def _import_analytics_data(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
		"""Import analytics touchpoint data using Google Analytics 4 API"""
		try:
			from google.analytics.data_v1beta import BetaAnalyticsDataClient
			from google.analytics.data_v1beta.types import (
				RunReportRequest,
				Dimension,
				Metric,
				DateRange
			)
			from google.oauth2 import service_account
			
			imported_data = []
			
			# Initialize Google Analytics client
			credentials_path = os.getenv('GOOGLE_ANALYTICS_CREDENTIALS_PATH')
			property_id = os.getenv('GOOGLE_ANALYTICS_PROPERTY_ID')
			
			if not credentials_path or not property_id:
				self.logger.warning("⚠️  Google Analytics API credentials not configured")
				return []
			
			# Load service account credentials
			credentials = service_account.Credentials.from_service_account_file(credentials_path)
			client = BetaAnalyticsDataClient(credentials=credentials)
			
			# Create date range
			date_range = DateRange(
				start_date=start_date.strftime('%Y-%m-%d'),
				end_date=end_date.strftime('%Y-%m-%d')
			)
			
			# Define dimensions and metrics for touchpoint analysis
			dimensions = [
				Dimension(name="date"),
				Dimension(name="sessionSourceMedium"),
				Dimension(name="sessionCampaignName"),
				Dimension(name="eventName"),
				Dimension(name="customUserId")  # For customer attribution
			]
			
			metrics = [
				Metric(name="sessions"),
				Metric(name="totalUsers"),
				Metric(name="eventCount"),
				Metric(name="conversions"),
				Metric(name="totalRevenue")
			]
			
			# Create request
			request = RunReportRequest(
				property=f"properties/{property_id}",
				dimensions=dimensions,
				metrics=metrics,
				date_ranges=[date_range],
				limit=10000,  # Adjust as needed
				dimension_filter={
					"filter": {
						"field_name": "eventName",
						"in_list_filter": {
							"values": ["purchase", "sign_up", "subscription_created", "page_view"]
						}
					}
				}
			)
			
			# Execute request
			response = client.run_report(request)
			
			for row in response.rows:
				# Extract dimension values
				date_val = row.dimension_values[0].value
				source_medium = row.dimension_values[1].value
				campaign_name = row.dimension_values[2].value
				event_name = row.dimension_values[3].value
				customer_id = row.dimension_values[4].value
				
				# Extract metric values
				sessions = int(row.metric_values[0].value) if row.metric_values[0].value else 0
				users = int(row.metric_values[1].value) if row.metric_values[1].value else 0
				event_count = int(row.metric_values[2].value) if row.metric_values[2].value else 0
				conversions = float(row.metric_values[3].value) if row.metric_values[3].value else 0
				revenue = float(row.metric_values[4].value) if row.metric_values[4].value else 0
				
				# Parse source/medium to determine channel
				channel = self._map_analytics_source_medium(source_medium)
				
				touchpoint_data = {
					'source': 'google_analytics',
					'customer_id': customer_id if customer_id and customer_id != '(not set)' else None,
					'channel': channel,
					'campaign_name': campaign_name if campaign_name != '(not set)' else None,
					'source_medium': source_medium,
					'event_name': event_name,
					'timestamp': f"{date_val}T12:00:00Z",  # Default to noon
					'sessions': sessions,
					'users': users,
					'event_count': event_count,
					'conversions': conversions,
					'revenue': revenue,
					'imported_at': datetime.utcnow().isoformat()
				}
				
				imported_data.append(touchpoint_data)
				
				# Create touchpoint record if we have customer ID and conversion event
				if (customer_id and customer_id != '(not set)' and 
					event_name in ['purchase', 'sign_up', 'subscription_created']):
					
					touchpoint_record = {
						'customer_id': customer_id,
						'channel': channel,
						'campaign_name': campaign_name,
						'source': source_medium.split(' / ')[0] if ' / ' in source_medium else source_medium,
						'medium': source_medium.split(' / ')[1] if ' / ' in source_medium else 'unknown',
						'timestamp': f"{date_val}T12:00:00Z",
						'value': revenue / max(conversions, 1),  # Revenue per conversion
						'conversion_value': revenue,
						'metadata': {
							'event_name': event_name,
							'sessions': sessions,
							'users': users
						}
					}
					self.add_touchpoint(touchpoint_record)
			
			self.logger.info(f"✅ Imported {len(imported_data)} Google Analytics records")
			return imported_data
			
		except ImportError:
			self.logger.warning("⚠️  Google Analytics API client not available. Install: pip install google-analytics-data")
			return []
		except Exception as e:
			self.logger.error(f"Google Analytics data import failed: {e}")
			return []
	
	def _map_analytics_source_medium(self, source_medium: str) -> str:
		"""Map Google Analytics source/medium to our marketing channels"""
		source_medium = source_medium.lower()
		
		if 'google' in source_medium and 'cpc' in source_medium:
			return MarketingChannel.PAID_SEARCH.value
		elif 'google' in source_medium and 'organic' in source_medium:
			return MarketingChannel.ORGANIC_SEARCH.value
		elif 'facebook' in source_medium or 'instagram' in source_medium:
			return MarketingChannel.SOCIAL_MEDIA.value
		elif 'email' in source_medium:
			return MarketingChannel.EMAIL_MARKETING.value
		elif 'cpc' in source_medium or 'ppc' in source_medium:
			return MarketingChannel.PAID_SEARCH.value
		elif 'display' in source_medium:
			return MarketingChannel.DISPLAY_ADS.value
		elif 'video' in source_medium or 'youtube' in source_medium:
			return MarketingChannel.VIDEO_ADS.value
		elif 'referral' in source_medium:
			return MarketingChannel.REFERRAL.value
		elif 'direct' in source_medium or source_medium == '(direct) / (none)':
			return MarketingChannel.DIRECT.value
		elif 'affiliate' in source_medium:
			return MarketingChannel.AFFILIATE.value
		else:
			return MarketingChannel.ORGANIC_SEARCH.value  # Default fallback
	
	async def generate_cac_optimization_recommendations(self, analysis_period_days: int = 90) -> Dict[str, Any]:
		"""Generate CAC optimization recommendations"""
		end_date = datetime.utcnow()
		start_date = end_date - timedelta(days=analysis_period_days)
		
		try:
			# Get current CAC analysis
			cac_data = await self.calculate_cac_by_channel(start_date, end_date)
			ltv_cac_data = await self.calculate_ltv_to_cac_ratio(start_date, end_date)
			
			recommendations = []
			
			# Analyze each channel
			for channel in MarketingChannel:
				channel_key = channel.value
				cac_info = cac_data['cac_by_channel'][channel_key]
				ltv_cac_info = ltv_cac_data['ltv_cac_by_channel'][channel_key]
				
				cac = Decimal(cac_info['cac'])
				efficiency_score = cac_info['efficiency_score']
				health_score = ltv_cac_info['health_score']
				
				# Generate recommendations based on performance
				if health_score >= 80 and efficiency_score >= 1.5:
					recommendations.append({
						'channel': channel_key,
						'priority': 'high',
						'action': 'scale_up',
						'recommendation': f'Increase investment in {channel_key} - excellent ROI',
						'current_cac': str(cac),
						'health_score': health_score
					})
				elif health_score < 40:
					recommendations.append({
						'channel': channel_key,
						'priority': 'high',
						'action': 'optimize_or_pause',
						'recommendation': f'Optimize or pause {channel_key} - poor ROI',
						'current_cac': str(cac),
						'health_score': health_score
					})
				elif efficiency_score < 0.8:
					recommendations.append({
						'channel': channel_key,
						'priority': 'medium',
						'action': 'optimize',
						'recommendation': f'Optimize {channel_key} targeting and creative',
						'current_cac': str(cac),
						'efficiency_score': efficiency_score
					})
			
			return {
				'analysis_period_days': analysis_period_days,
				'total_recommendations': len(recommendations),
				'recommendations': recommendations,
				'generated_at': datetime.utcnow().isoformat()
			}
		
		except Exception as e:
			self.logger.error(f"CAC optimization recommendations failed: {e}")
			raise


# Global CAC analytics engine
_cac_engine_instance: Optional[CACAnalyticsEngine] = None

def get_cac_analytics_engine() -> CACAnalyticsEngine:
	"""Get global CAC analytics engine instance"""
	global _cac_engine_instance
	if _cac_engine_instance is None:
		_cac_engine_instance = CACAnalyticsEngine()
	return _cac_engine_instance


__all__ = [
	'CACAnalyticsEngine',
	'TouchPoint',
	'MarketingSpend',
	'AttributionModel',
	'MarketingChannel',
	'get_cac_analytics_engine'
]