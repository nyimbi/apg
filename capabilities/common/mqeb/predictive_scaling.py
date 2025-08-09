#!/usr/bin/env python3
"""
APG Message Queue Event Bus (MQEB) - Predictive Scaling Service
AI-powered predictive auto-scaling with self-healing capabilities

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

from .models import BrokerNode, TopicConfiguration
from .service import MQEBService


class ScalingAction(str, Enum):
	"""Types of scaling actions"""
	SCALE_UP = "scale_up"
	SCALE_DOWN = "scale_down"
	SCALE_OUT = "scale_out"  # Add more brokers
	SCALE_IN = "scale_in"   # Remove brokers
	REBALANCE = "rebalance"
	NO_ACTION = "no_action"


class ResourceType(str, Enum):
	"""Types of resources to scale"""
	BROKER_NODES = "broker_nodes"
	TOPIC_PARTITIONS = "topic_partitions"
	CONNECTION_POOLS = "connection_pools"
	MEMORY_BUFFERS = "memory_buffers"
	DISK_STORAGE = "disk_storage"


@dataclass
class ScalingMetrics:
	"""Resource utilization metrics for scaling decisions"""
	timestamp: datetime
	cpu_usage: float
	memory_usage: float
	disk_usage: float
	network_io_mbps: float
	active_connections: int
	messages_per_second: float
	queue_depth: int
	error_rate: float
	response_time_p99: float


@dataclass
class ScalingRecommendation:
	"""Scaling recommendation from predictive engine"""
	action: ScalingAction
	resource_type: ResourceType
	current_value: int
	recommended_value: int
	confidence: float
	reasoning: str
	estimated_impact: Dict[str, float]
	urgency: str  # low, medium, high, critical
	estimated_cost_change: float


@dataclass
class ScalingEvent:
	"""Record of a scaling event"""
	event_id: str
	timestamp: datetime
	action: ScalingAction
	resource_type: ResourceType
	old_value: int
	new_value: int
	triggered_by: str
	success: bool
	duration_seconds: float
	impact_metrics: Dict[str, float]


class ResourcePredictor:
	"""Predicts resource usage using machine learning"""
	
	def __init__(self):
		self.cpu_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
		self.memory_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
		self.throughput_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
		self.scaler = StandardScaler()
		
		self.historical_metrics = deque(maxlen=10080)  # 1 week of minutes
		self.is_trained = False
		self.last_training = None
		
	async def record_metrics(self, metrics: ScalingMetrics) -> None:
		"""Record metrics for training and prediction"""
		self.historical_metrics.append(metrics)
		
		# Retrain models periodically
		if len(self.historical_metrics) > 100 and (
			self.last_training is None or 
			(datetime.utcnow() - self.last_training).total_seconds() > 3600
		):
			await self._retrain_models()
	
	async def predict_resource_usage(self, horizon_minutes: int = 60) -> Dict[str, List[float]]:
		"""Predict resource usage for the next N minutes"""
		if not self.is_trained or len(self.historical_metrics) < 50:
			return await self._default_predictions(horizon_minutes)
		
		try:
			predictions = {
				'cpu_usage': [],
				'memory_usage': [],
				'messages_per_second': [],
				'timestamps': []
			}
			
			current_time = datetime.utcnow()
			
			# Use latest metrics as starting point
			latest_metrics = self.historical_metrics[-1]
			
			for i in range(horizon_minutes):
				future_time = current_time + timedelta(minutes=i)
				features = self._extract_time_features(future_time, latest_metrics)
				
				# Scale features
				features_scaled = self.scaler.transform([features])
				
				# Make predictions
				cpu_pred = max(0, min(100, self.cpu_predictor.predict(features_scaled)[0]))
				memory_pred = max(0, min(100, self.memory_predictor.predict(features_scaled)[0]))
				throughput_pred = max(0, self.throughput_predictor.predict(features_scaled)[0])
				
				predictions['cpu_usage'].append(cpu_pred)
				predictions['memory_usage'].append(memory_pred)
				predictions['messages_per_second'].append(throughput_pred)
				predictions['timestamps'].append(future_time)
			
			return predictions
			
		except Exception as e:
			logging.error(f"Resource prediction failed: {e}")
			return await self._default_predictions(horizon_minutes)
	
	async def _retrain_models(self) -> None:
		"""Retrain prediction models"""
		try:
			if len(self.historical_metrics) < 50:
				return
			
			# Prepare training data
			X = []
			y_cpu = []
			y_memory = []
			y_throughput = []
			
			metrics_list = list(self.historical_metrics)
			
			for i in range(10, len(metrics_list)):  # Need history for features
				current_metrics = metrics_list[i]
				
				features = self._extract_time_features(current_metrics.timestamp, current_metrics)
				X.append(features)
				y_cpu.append(current_metrics.cpu_usage)
				y_memory.append(current_metrics.memory_usage)
				y_throughput.append(current_metrics.messages_per_second)
			
			if len(X) < 20:
				return
			
			# Scale features
			X_scaled = self.scaler.fit_transform(X)
			
			# Train models
			self.cpu_predictor.fit(X_scaled, y_cpu)
			self.memory_predictor.fit(X_scaled, y_memory)
			self.throughput_predictor.fit(X_scaled, y_throughput)
			
			self.is_trained = True
			self.last_training = datetime.utcnow()
			
			logging.info(f"Resource prediction models retrained on {len(X)} samples")
			
		except Exception as e:
			logging.error(f"Model retraining failed: {e}")
	
	def _extract_time_features(self, timestamp: datetime, metrics: ScalingMetrics) -> List[float]:
		"""Extract features for prediction models"""
		# Time-based features
		hour = timestamp.hour
		day_of_week = timestamp.weekday()
		minute_of_day = hour * 60 + timestamp.minute
		is_weekend = 1 if day_of_week >= 5 else 0
		
		# Cyclical time features
		hour_sin = np.sin(2 * np.pi * hour / 24)
		hour_cos = np.cos(2 * np.pi * hour / 24)
		dow_sin = np.sin(2 * np.pi * day_of_week / 7)
		dow_cos = np.cos(2 * np.pi * day_of_week / 7)
		
		# Historical trend features (using recent metrics)
		recent_metrics = list(self.historical_metrics)[-10:]
		if len(recent_metrics) >= 5:
			cpu_trend = recent_metrics[-1].cpu_usage - recent_metrics[-5].cpu_usage
			memory_trend = recent_metrics[-1].memory_usage - recent_metrics[-5].memory_usage
			throughput_trend = recent_metrics[-1].messages_per_second - recent_metrics[-5].messages_per_second
		else:
			cpu_trend = memory_trend = throughput_trend = 0
		
		return [
			hour, day_of_week, minute_of_day, is_weekend,
			hour_sin, hour_cos, dow_sin, dow_cos,
			metrics.cpu_usage, metrics.memory_usage, metrics.messages_per_second,
			metrics.active_connections, metrics.queue_depth, metrics.error_rate,
			cpu_trend, memory_trend, throughput_trend
		]
	
	async def _default_predictions(self, horizon_minutes: int) -> Dict[str, List[float]]:
		"""Default predictions when models aren't trained"""
		if len(self.historical_metrics) > 0:
			latest = self.historical_metrics[-1]
			base_cpu = latest.cpu_usage
			base_memory = latest.memory_usage
			base_throughput = latest.messages_per_second
		else:
			base_cpu = base_memory = 50.0
			base_throughput = 1000.0
		
		return {
			'cpu_usage': [base_cpu] * horizon_minutes,
			'memory_usage': [base_memory] * horizon_minutes,
			'messages_per_second': [base_throughput] * horizon_minutes,
			'timestamps': [datetime.utcnow() + timedelta(minutes=i) for i in range(horizon_minutes)]
		}


class AutoScaler:
	"""Implements automatic scaling decisions and actions"""
	
	def __init__(self):
		self.scaling_policies = self._initialize_scaling_policies()
		self.scaling_history = deque(maxlen=1000)
		self.last_scaling_action = None
		self.cooldown_period_seconds = 300  # 5 minutes
		
	def _initialize_scaling_policies(self) -> Dict[str, Dict]:
		"""Initialize default scaling policies"""
		return {
			'cpu_scale_up': {
				'metric': 'cpu_usage',
				'threshold': 80.0,
				'duration_minutes': 5,
				'action': ScalingAction.SCALE_UP,
				'resource_type': ResourceType.BROKER_NODES,
				'scale_factor': 1.5
			},
			'cpu_scale_down': {
				'metric': 'cpu_usage',
				'threshold': 30.0,
				'duration_minutes': 15,
				'action': ScalingAction.SCALE_DOWN,
				'resource_type': ResourceType.BROKER_NODES,
				'scale_factor': 0.8
			},
			'memory_scale_up': {
				'metric': 'memory_usage',
				'threshold': 85.0,
				'duration_minutes': 3,
				'action': ScalingAction.SCALE_UP,
				'resource_type': ResourceType.MEMORY_BUFFERS,
				'scale_factor': 1.3
			},
			'throughput_scale_out': {
				'metric': 'messages_per_second',
				'threshold': 50000.0,
				'duration_minutes': 2,
				'action': ScalingAction.SCALE_OUT,
				'resource_type': ResourceType.BROKER_NODES,
				'scale_factor': 1.2
			},
			'error_rate_emergency': {
				'metric': 'error_rate',
				'threshold': 0.05,  # 5% error rate
				'duration_minutes': 1,
				'action': ScalingAction.SCALE_OUT,
				'resource_type': ResourceType.BROKER_NODES,
				'scale_factor': 2.0
			}
		}
	
	async def evaluate_scaling_need(self, current_metrics: ScalingMetrics,
									predicted_metrics: Dict[str, List[float]]) -> List[ScalingRecommendation]:
		"""Evaluate if scaling is needed based on current and predicted metrics"""
		recommendations = []
		
		# Check current metrics against thresholds
		current_recommendations = await self._evaluate_current_metrics(current_metrics)
		recommendations.extend(current_recommendations)
		
		# Check predicted metrics for proactive scaling
		predicted_recommendations = await self._evaluate_predicted_metrics(predicted_metrics)
		recommendations.extend(predicted_recommendations)
		
		# Remove duplicate recommendations and apply prioritization
		recommendations = await self._prioritize_recommendations(recommendations)
		
		return recommendations
	
	async def _evaluate_current_metrics(self, metrics: ScalingMetrics) -> List[ScalingRecommendation]:
		"""Evaluate current metrics against scaling policies"""
		recommendations = []
		
		metric_values = {
			'cpu_usage': metrics.cpu_usage,
			'memory_usage': metrics.memory_usage,
			'messages_per_second': metrics.messages_per_second,
			'error_rate': metrics.error_rate,
			'response_time_p99': metrics.response_time_p99
		}
		
		for policy_name, policy in self.scaling_policies.items():
			metric_name = policy['metric']
			if metric_name not in metric_values:
				continue
			
			current_value = metric_values[metric_name]
			threshold = policy['threshold']
			
			# Check if threshold is breached
			breached = False
			if policy['action'] in [ScalingAction.SCALE_UP, ScalingAction.SCALE_OUT]:
				breached = current_value > threshold
			elif policy['action'] in [ScalingAction.SCALE_DOWN, ScalingAction.SCALE_IN]:
				breached = current_value < threshold
			
			if breached and await self._check_duration_requirement(policy, metrics):
				recommendation = await self._create_recommendation(policy, current_value, "current_metrics")
				recommendations.append(recommendation)
		
		return recommendations
	
	async def _evaluate_predicted_metrics(self, predicted_metrics: Dict[str, List[float]]) -> List[ScalingRecommendation]:
		"""Evaluate predicted metrics for proactive scaling"""
		recommendations = []
		
		for metric_name, predictions in predicted_metrics.items():
			if len(predictions) < 10:
				continue
			
			# Look for sustained threshold breaches in predictions
			next_10_minutes = predictions[:10]
			
			# Check for scale-up conditions
			if metric_name in ['cpu_usage', 'memory_usage']:
				if statistics.mean(next_10_minutes) > 75 and max(next_10_minutes) > 90:
					recommendation = ScalingRecommendation(
						action=ScalingAction.SCALE_UP,
						resource_type=ResourceType.BROKER_NODES,
						current_value=1,
						recommended_value=2,
						confidence=0.8,
						reasoning=f"Predicted {metric_name} will exceed 90% within 10 minutes",
						estimated_impact={'latency_reduction': 0.3, 'throughput_increase': 0.5},
						urgency="high",
						estimated_cost_change=25.0
					)
					recommendations.append(recommendation)
			
			elif metric_name == 'messages_per_second':
				peak_predicted = max(next_10_minutes)
				if peak_predicted > 60000:  # High throughput predicted
					recommendation = ScalingRecommendation(
						action=ScalingAction.SCALE_OUT,
						resource_type=ResourceType.TOPIC_PARTITIONS,
						current_value=10,
						recommended_value=15,
						confidence=0.9,
						reasoning=f"Predicted throughput spike to {peak_predicted:.0f} msg/sec",
						estimated_impact={'capacity_increase': 0.5, 'latency_reduction': 0.2},
						urgency="medium",
						estimated_cost_change=15.0
					)
					recommendations.append(recommendation)
		
		return recommendations
	
	async def _check_duration_requirement(self, policy: Dict, metrics: ScalingMetrics) -> bool:
		"""Check if threshold breach has lasted for required duration"""
		# Simplified: in production would check historical data
		return True
	
	async def _create_recommendation(self, policy: Dict, current_value: float, source: str) -> ScalingRecommendation:
		"""Create scaling recommendation from policy"""
		scale_factor = policy.get('scale_factor', 1.2)
		
		if policy['action'] in [ScalingAction.SCALE_UP, ScalingAction.SCALE_OUT]:
			new_value = int(current_value * scale_factor)
			urgency = "high" if current_value > policy['threshold'] * 1.2 else "medium"
		else:
			new_value = int(current_value * scale_factor)
			urgency = "low"
		
		return ScalingRecommendation(
			action=policy['action'],
			resource_type=policy['resource_type'],
			current_value=int(current_value),
			recommended_value=new_value,
			confidence=0.8,
			reasoning=f"{policy['metric']} threshold breach detected ({source})",
			estimated_impact=self._estimate_scaling_impact(policy['action']),
			urgency=urgency,
			estimated_cost_change=self._estimate_cost_change(policy['action'], scale_factor)
		)
	
	def _estimate_scaling_impact(self, action: ScalingAction) -> Dict[str, float]:
		"""Estimate the impact of scaling action"""
		impact_maps = {
			ScalingAction.SCALE_UP: {
				'cpu_reduction': 0.3,
				'memory_reduction': 0.2,
				'latency_reduction': 0.25,
				'throughput_increase': 0.4
			},
			ScalingAction.SCALE_OUT: {
				'capacity_increase': 0.8,
				'latency_reduction': 0.4,
				'throughput_increase': 0.6,
				'availability_increase': 0.1
			},
			ScalingAction.SCALE_DOWN: {
				'cost_reduction': 0.2,
				'resource_efficiency': 0.15
			},
			ScalingAction.SCALE_IN: {
				'cost_reduction': 0.4,
				'operational_simplification': 0.3
			}
		}
		
		return impact_maps.get(action, {'unknown_impact': 0.1})
	
	def _estimate_cost_change(self, action: ScalingAction, scale_factor: float) -> float:
		"""Estimate cost change percentage"""
		base_costs = {
			ScalingAction.SCALE_UP: 20.0,
			ScalingAction.SCALE_OUT: 50.0,
			ScalingAction.SCALE_DOWN: -15.0,
			ScalingAction.SCALE_IN: -40.0
		}
		
		base_cost = base_costs.get(action, 0.0)
		return base_cost * (scale_factor - 1.0)
	
	async def _prioritize_recommendations(self, recommendations: List[ScalingRecommendation]) -> List[ScalingRecommendation]:
		"""Prioritize and deduplicate scaling recommendations"""
		if not recommendations:
			return recommendations
		
		# Remove duplicates
		seen_actions = set()
		unique_recommendations = []
		
		for rec in recommendations:
			action_key = (rec.action, rec.resource_type)
			if action_key not in seen_actions:
				seen_actions.add(action_key)
				unique_recommendations.append(rec)
		
		# Sort by urgency and confidence
		urgency_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
		
		unique_recommendations.sort(
			key=lambda r: (urgency_order.get(r.urgency, 0), r.confidence),
			reverse=True
		)
		
		return unique_recommendations[:5]  # Limit to top 5 recommendations
	
	async def check_cooldown(self) -> bool:
		"""Check if we're still in cooldown period from last scaling action"""
		if self.last_scaling_action is None:
			return False
		
		time_since_last = (datetime.utcnow() - self.last_scaling_action).total_seconds()
		return time_since_last < self.cooldown_period_seconds
	
	async def execute_scaling_action(self, recommendation: ScalingRecommendation) -> ScalingEvent:
		"""Execute a scaling action (simulation for now)"""
		event_id = f"scale_{int(datetime.utcnow().timestamp())}"
		start_time = datetime.utcnow()
		
		try:
			# Simulate scaling action
			await asyncio.sleep(0.1)  # Simulate execution time
			
			# Record successful scaling event
			event = ScalingEvent(
				event_id=event_id,
				timestamp=start_time,
				action=recommendation.action,
				resource_type=recommendation.resource_type,
				old_value=recommendation.current_value,
				new_value=recommendation.recommended_value,
				triggered_by="predictive_scaling",
				success=True,
				duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
				impact_metrics=recommendation.estimated_impact
			)
			
			self.scaling_history.append(event)
			self.last_scaling_action = datetime.utcnow()
			
			logging.info(f"Scaling action executed: {recommendation.action} {recommendation.resource_type}")
			
			return event
			
		except Exception as e:
			# Record failed scaling event
			event = ScalingEvent(
				event_id=event_id,
				timestamp=start_time,
				action=recommendation.action,
				resource_type=recommendation.resource_type,
				old_value=recommendation.current_value,
				new_value=recommendation.current_value,  # No change due to failure
				triggered_by="predictive_scaling",
				success=False,
				duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
				impact_metrics={}
			)
			
			self.scaling_history.append(event)
			logging.error(f"Scaling action failed: {e}")
			
			return event


class PredictiveScalingService:
	"""Main predictive scaling service"""
	
	def __init__(self, mqeb_service: MQEBService):
		self.service = mqeb_service
		self.resource_predictor = ResourcePredictor()
		self.auto_scaler = AutoScaler()
		
		# Service state
		self.enabled = True
		self.monitoring_interval_seconds = 60
		self._background_tasks: Set[asyncio.Task] = set()
		
		# Metrics
		self.scaling_metrics_history = deque(maxlen=1440)  # 24 hours
		
		self.logger = logging.getLogger('mqeb.predictive_scaling')
	
	async def initialize(self) -> None:
		"""Initialize predictive scaling service"""
		self.logger.info("Initializing predictive scaling service...")
		
		# Start background monitoring
		await self._start_background_tasks()
		
		self.logger.info("Predictive scaling service initialized")
	
	async def shutdown(self) -> None:
		"""Shutdown predictive scaling service"""
		self.enabled = False
		
		# Cancel background tasks
		for task in self._background_tasks:
			task.cancel()
		
		await asyncio.gather(*self._background_tasks, return_exceptions=True)
		self.logger.info("Predictive scaling service shut down")
	
	async def _start_background_tasks(self) -> None:
		"""Start background tasks"""
		
		# Metrics collection task
		task = asyncio.create_task(self._metrics_collection_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# Scaling evaluation task
		task = asyncio.create_task(self._scaling_evaluation_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# Self-healing task
		task = asyncio.create_task(self._self_healing_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
	
	async def _collect_current_metrics(self) -> ScalingMetrics:
		"""Collect current system metrics"""
		# Get metrics from broker nodes
		total_cpu = total_memory = total_connections = 0
		total_messages_per_sec = total_queue_depth = 0
		node_count = 0
		
		for node in self.service.broker_nodes.values():
			total_cpu += node.cpu_usage
			total_memory += node.memory_usage
			total_connections += node.active_connections
			total_messages_per_sec += node.messages_per_second
			node_count += 1
		
		# Calculate queue depths
		for queue in self.service.message_queues.values():
			total_queue_depth += len(queue)
		
		# Calculate error rate (simplified)
		total_messages = self.service.metrics.get('messages_published', 1)
		failed_messages = self.service.metrics.get('messages_failed', 0)
		error_rate = failed_messages / max(1, total_messages)
		
		return ScalingMetrics(
			timestamp=datetime.utcnow(),
			cpu_usage=total_cpu / max(1, node_count),
			memory_usage=total_memory / max(1, node_count),
			disk_usage=35.0,  # Simplified
			network_io_mbps=100.0,  # Simplified
			active_connections=total_connections,
			messages_per_second=total_messages_per_sec,
			queue_depth=total_queue_depth,
			error_rate=error_rate,
			response_time_p99=5.0  # Simplified
		)
	
	async def _metrics_collection_loop(self) -> None:
		"""Background task to collect metrics"""
		while self.enabled:
			try:
				metrics = await self._collect_current_metrics()
				self.scaling_metrics_history.append(metrics)
				
				# Feed metrics to predictor
				await self.resource_predictor.record_metrics(metrics)
				
				await asyncio.sleep(self.monitoring_interval_seconds)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Metrics collection error: {e}")
				await asyncio.sleep(self.monitoring_interval_seconds)
	
	async def _scaling_evaluation_loop(self) -> None:
		"""Background task to evaluate scaling needs"""
		while self.enabled:
			try:
				await asyncio.sleep(120)  # Evaluate every 2 minutes
				
				if not self.scaling_metrics_history:
					continue
				
				# Check if we're in cooldown
				if await self.auto_scaler.check_cooldown():
					self.logger.debug("Scaling in cooldown period, skipping evaluation")
					continue
				
				# Get current metrics
				current_metrics = self.scaling_metrics_history[-1]
				
				# Get predictions
				predictions = await self.resource_predictor.predict_resource_usage(horizon_minutes=30)
				
				# Evaluate scaling needs
				recommendations = await self.auto_scaler.evaluate_scaling_need(current_metrics, predictions)
				
				if recommendations:
					self.logger.info(f"Generated {len(recommendations)} scaling recommendations")
					
					# Execute the highest priority recommendation
					top_recommendation = recommendations[0]
					if top_recommendation.urgency in ['high', 'critical']:
						scaling_event = await self.auto_scaler.execute_scaling_action(top_recommendation)
						self.logger.info(f"Executed scaling action: {scaling_event.action} - Success: {scaling_event.success}")
					else:
						self.logger.info(f"Low priority scaling recommendation: {top_recommendation.action} (deferred)")
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Scaling evaluation error: {e}")
	
	async def _self_healing_loop(self) -> None:
		"""Background task for self-healing operations"""
		while self.enabled:
			try:
				await asyncio.sleep(300)  # Check every 5 minutes
				
				# Check for system health issues
				if self.scaling_metrics_history:
					latest_metrics = self.scaling_metrics_history[-1]
					
					# Emergency scaling for critical issues
					if latest_metrics.error_rate > 0.1:  # 10% error rate
						emergency_recommendation = ScalingRecommendation(
							action=ScalingAction.SCALE_OUT,
							resource_type=ResourceType.BROKER_NODES,
							current_value=len(self.service.broker_nodes),
							recommended_value=len(self.service.broker_nodes) + 2,
							confidence=1.0,
							reasoning="Emergency scaling due to high error rate",
							estimated_impact={'error_reduction': 0.7},
							urgency="critical",
							estimated_cost_change=100.0
						)
						
						await self.auto_scaler.execute_scaling_action(emergency_recommendation)
						self.logger.warning("Emergency scaling executed due to high error rate")
					
					# Self-healing for resource exhaustion
					if latest_metrics.cpu_usage > 95 or latest_metrics.memory_usage > 95:
						healing_recommendation = ScalingRecommendation(
							action=ScalingAction.SCALE_UP,
							resource_type=ResourceType.BROKER_NODES,
							current_value=1,
							recommended_value=2,
							confidence=0.9,
							reasoning="Self-healing for resource exhaustion",
							estimated_impact={'resource_relief': 0.5},
							urgency="high",
							estimated_cost_change=50.0
						)
						
						await self.auto_scaler.execute_scaling_action(healing_recommendation)
						self.logger.warning("Self-healing scaling executed due to resource exhaustion")
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Self-healing error: {e}")
	
	async def get_scaling_status(self) -> Dict[str, Any]:
		"""Get current scaling status and metrics"""
		current_metrics = self.scaling_metrics_history[-1] if self.scaling_metrics_history else None
		predictions = await self.resource_predictor.predict_resource_usage(horizon_minutes=60)
		
		return {
			'enabled': self.enabled,
			'current_metrics': {
				'cpu_usage': current_metrics.cpu_usage if current_metrics else 0,
				'memory_usage': current_metrics.memory_usage if current_metrics else 0,
				'messages_per_second': current_metrics.messages_per_second if current_metrics else 0,
				'error_rate': current_metrics.error_rate if current_metrics else 0
			} if current_metrics else {},
			'predictions': {
				'next_hour_peak_cpu': max(predictions['cpu_usage']) if predictions['cpu_usage'] else 0,
				'next_hour_peak_memory': max(predictions['memory_usage']) if predictions['memory_usage'] else 0,
				'predicted_max_throughput': max(predictions['messages_per_second']) if predictions['messages_per_second'] else 0
			},
			'predictor_status': {
				'is_trained': self.resource_predictor.is_trained,
				'training_data_points': len(self.resource_predictor.historical_metrics),
				'last_training': self.resource_predictor.last_training.isoformat() if self.resource_predictor.last_training else None
			},
			'recent_scaling_events': [
				{
					'timestamp': event.timestamp.isoformat(),
					'action': event.action.value,
					'resource_type': event.resource_type.value,
					'success': event.success,
					'reasoning': f"{event.action.value} {event.resource_type.value}"
				}
				for event in list(self.auto_scaler.scaling_history)[-5:]
			],
			'cooldown_remaining_seconds': max(0, 
				self.auto_scaler.cooldown_period_seconds - 
				(datetime.utcnow() - self.auto_scaler.last_scaling_action).total_seconds()
			) if self.auto_scaler.last_scaling_action else 0
		}
	
	async def manual_scaling_recommendation(self) -> List[ScalingRecommendation]:
		"""Get manual scaling recommendations on demand"""
		if not self.scaling_metrics_history:
			return []
		
		current_metrics = self.scaling_metrics_history[-1]
		predictions = await self.resource_predictor.predict_resource_usage(horizon_minutes=60)
		
		recommendations = await self.auto_scaler.evaluate_scaling_need(current_metrics, predictions)
		return recommendations


# Factory function
async def create_predictive_scaling_service(mqeb_service: MQEBService) -> PredictiveScalingService:
	"""Create and initialize predictive scaling service"""
	service = PredictiveScalingService(mqeb_service)
	await service.initialize()
	return service


# Export components
__all__ = [
	'PredictiveScalingService', 'ResourcePredictor', 'AutoScaler',
	'ScalingMetrics', 'ScalingRecommendation', 'ScalingEvent',
	'ScalingAction', 'ResourceType',
	'create_predictive_scaling_service'
]