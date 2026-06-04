#!/usr/bin/env python3
"""
APG Monitoring - Alert Engine Foundation
Intelligent alerting system with correlation, deduplication, and escalation management

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict
from uuid6 import uuid7
def uuid7str() -> str: return str(uuid7())

from .models import (
	MonitoringAlert, MonitoringRule, MonitoringMetric, AlertSeverity, 
	AlertStatus, AlertConditionType, MonitoringScope
)


class AlertCorrelationStrategy(str, Enum):
	"""Alert correlation strategies"""
	TIME_BASED = "time_based"
	CAUSE_EFFECT = "cause_effect"
	SIMILARITY = "similarity"
	DEPENDENCY = "dependency"
	ML_BASED = "ml_based"


class EscalationAction(str, Enum):
	"""Escalation action types"""
	NOTIFY = "notify"
	CREATE_TICKET = "create_ticket"
	AUTO_REMEDIATE = "auto_remediate"
	ESCALATE_TEAM = "escalate_team"
	EXECUTIVE_ALERT = "executive_alert"


@dataclass
class AlertCorrelation:
	"""Alert correlation information"""
	correlation_id: str
	primary_alert_id: str
	related_alert_ids: List[str]
	correlation_strategy: AlertCorrelationStrategy
	confidence_score: float
	created_at: datetime
	correlation_reason: str
	
	def add_related_alert(self, alert_id: str) -> None:
		"""Add related alert to correlation group"""
		if alert_id not in self.related_alert_ids:
			self.related_alert_ids.append(alert_id)
	
	def get_total_alerts(self) -> int:
		"""Get total number of alerts in correlation"""
		return 1 + len(self.related_alert_ids)


@dataclass
class EscalationPolicy:
	"""Alert escalation policy configuration"""
	policy_id: str
	name: str
	tenant_id: str
	severity_levels: List[AlertSeverity]
	escalation_steps: List[Dict[str, Any]]
	max_escalation_time_minutes: int
	auto_resolve: bool = True
	created_by: str = "system"
	created_at: datetime = field(default_factory=datetime.utcnow)


class AlertEvaluator:
	"""Evaluates alert conditions against metrics"""
	
	def __init__(self):
		self.expression_cache: Dict[str, callable] = {}
		self.evaluation_stats = {
			'total_evaluations': 0,
			'successful_evaluations': 0,
			'failed_evaluations': 0,
			'avg_evaluation_time_ms': 0.0
		}
	
	async def evaluate_rule(self, rule: MonitoringRule, metrics: List[MonitoringMetric]) -> Optional[MonitoringAlert]:
		"""Evaluate alert rule against metrics and return alert if triggered"""
		start_time = time.time()
		
		try:
			self.evaluation_stats['total_evaluations'] += 1
			
			# Filter metrics for this rule
			relevant_metrics = self._filter_metrics_for_rule(rule, metrics)
			if not relevant_metrics:
				return None
			
			# Evaluate condition based on type
			triggered = False
			trigger_value = None
			
			if rule.condition_type == AlertConditionType.THRESHOLD:
				triggered, trigger_value = await self._evaluate_threshold_condition(rule, relevant_metrics)
			elif rule.condition_type == AlertConditionType.ANOMALY:
				triggered, trigger_value = await self._evaluate_anomaly_condition(rule, relevant_metrics)
			elif rule.condition_type == AlertConditionType.RATE:
				triggered, trigger_value = await self._evaluate_rate_condition(rule, relevant_metrics)
			elif rule.condition_type == AlertConditionType.ABSENCE:
				triggered, trigger_value = await self._evaluate_absence_condition(rule, relevant_metrics)
			elif rule.condition_type == AlertConditionType.COMPOSITE:
				triggered, trigger_value = await self._evaluate_composite_condition(rule, relevant_metrics)
			
			if triggered:
				# Create alert
				alert = MonitoringAlert(
					tenant_id=rule.tenant_id,
					rule_id=rule.rule_id,
					name=rule.name,
					description=f"Alert triggered for rule: {rule.name}",
					severity=rule.severity,
					message=self._format_alert_message(rule, trigger_value),
					summary=rule.alert_summary or f"Alert: {rule.name}",
					runbook_url=rule.runbook_url,
					source_metric=rule.metric_name,
					source_value=trigger_value,
					threshold_value=rule.threshold_value,
					correlation_key=rule.correlation_key,
					escalation_interval_minutes=rule.escalation_interval_minutes,
					max_escalation_level=rule.max_escalation_level,
					labels=rule.metric_labels.copy(),
					annotations={
						'rule_condition': rule.condition,
						'evaluation_time': datetime.utcnow().isoformat(),
						'trigger_value': str(trigger_value)
					}
				)
				
				# Update rule statistics
				rule.trigger_count += 1
				rule.last_triggered = datetime.utcnow()
				
				self.evaluation_stats['successful_evaluations'] += 1
				return alert
			
			self.evaluation_stats['successful_evaluations'] += 1
			return None
			
		except Exception as e:
			self.evaluation_stats['failed_evaluations'] += 1
			print(f"Error evaluating rule {rule.rule_id}: {e}")
			return None
		
		finally:
			# Update evaluation time statistics
			evaluation_time = (time.time() - start_time) * 1000
			current_avg = self.evaluation_stats['avg_evaluation_time_ms']
			self.evaluation_stats['avg_evaluation_time_ms'] = (current_avg * 0.9) + (evaluation_time * 0.1)
	
	def _filter_metrics_for_rule(self, rule: MonitoringRule, metrics: List[MonitoringMetric]) -> List[MonitoringMetric]:
		"""Filter metrics that match the rule criteria"""
		filtered = []
		
		for metric in metrics:
			# Check metric name
			if metric.name != rule.metric_name:
				continue
			
			# Check tenant isolation
			if metric.tenant_id != rule.tenant_id:
				continue
			
			# Check label filters
			labels_match = True
			for key, value in rule.metric_labels.items():
				if key not in metric.labels or metric.labels[key] != value:
					labels_match = False
					break
			
			if labels_match:
				# Check if metric is within evaluation window
				window_start = datetime.utcnow() - timedelta(minutes=rule.evaluation_window_minutes)
				if metric.timestamp >= window_start:
					filtered.append(metric)
		
		return filtered
	
	async def _evaluate_threshold_condition(self, rule: MonitoringRule, metrics: List[MonitoringMetric]) -> tuple[bool, Optional[float]]:
		"""Evaluate threshold-based condition"""
		if not metrics or rule.threshold_value is None:
			return False, None
		
		# Get latest metric value
		latest_metric = max(metrics, key=lambda m: m.timestamp)
		value = latest_metric.value
		operator = rule.threshold_operator
		threshold = rule.threshold_value
		
		# Evaluate based on operator
		if operator == "gt":
			triggered = value > threshold
		elif operator == "gte":
			triggered = value >= threshold
		elif operator == "lt":
			triggered = value < threshold
		elif operator == "lte":
			triggered = value <= threshold
		elif operator == "eq":
			triggered = value == threshold
		elif operator == "ne":
			triggered = value != threshold
		else:
			print(f"Unknown threshold operator: {operator}")
			return False, None
		
		return triggered, value if triggered else None
	
	async def _evaluate_anomaly_condition(self, rule: MonitoringRule, metrics: List[MonitoringMetric]) -> tuple[bool, Optional[float]]:
		"""Evaluate anomaly detection condition"""
		if not metrics or len(metrics) < 10:  # Need minimum data points
			return False, None
		
		# Simple anomaly detection using statistical analysis
		values = [m.value for m in metrics]
		
		# Calculate mean and standard deviation
		mean_value = sum(values) / len(values)
		variance = sum((x - mean_value) ** 2 for x in values) / len(values)
		std_dev = variance ** 0.5
		
		# Check if latest value is anomalous (beyond 2 standard deviations)
		latest_value = values[-1]
		z_score = abs(latest_value - mean_value) / std_dev if std_dev > 0 else 0
		
		# Consider anomalous if z-score > sensitivity threshold (scaled to 0-1)
		anomaly_threshold = 2.0 * (1.0 - rule.anomaly_sensitivity)  # Higher sensitivity = lower threshold
		triggered = z_score > anomaly_threshold
		
		return triggered, latest_value if triggered else None
	
	async def _evaluate_rate_condition(self, rule: MonitoringRule, metrics: List[MonitoringMetric]) -> tuple[bool, Optional[float]]:
		"""Evaluate rate of change condition"""
		if len(metrics) < 2:
			return False, None
		
		# Sort metrics by timestamp
		sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
		
		# Calculate rate of change
		first_value = sorted_metrics[0].value
		last_value = sorted_metrics[-1].value
		time_diff = (sorted_metrics[-1].timestamp - sorted_metrics[0].timestamp).total_seconds()
		
		if time_diff <= 0:
			return False, None
		
		rate = (last_value - first_value) / time_diff
		
		# Parse rate condition from rule condition string
		# Expected format: "rate > 10" or "rate < -5"
		condition_match = re.search(r'rate\s*([><=!]+)\s*([-+]?\d+\.?\d*)', rule.condition)
		if not condition_match:
			return False, None
		
		operator = condition_match.group(1)
		threshold = float(condition_match.group(2))
		
		if operator in ['>', 'gt']:
			triggered = rate > threshold
		elif operator in ['<', 'lt']:
			triggered = rate < threshold
		elif operator in ['>=', 'gte']:
			triggered = rate >= threshold
		elif operator in ['<=', 'lte']:
			triggered = rate <= threshold
		else:
			return False, None
		
		return triggered, rate if triggered else None
	
	async def _evaluate_absence_condition(self, rule: MonitoringRule, metrics: List[MonitoringMetric]) -> tuple[bool, Optional[float]]:
		"""Evaluate metric absence condition"""
		# Check if we haven't received metrics in the expected window
		expected_interval = rule.evaluation_window_minutes * 60  # Convert to seconds
		
		if not metrics:
			# No metrics received - this could be an absence
			return True, 0.0
		
		# Check if latest metric is too old
		latest_metric = max(metrics, key=lambda m: m.timestamp)
		age_seconds = (datetime.utcnow() - latest_metric.timestamp).total_seconds()
		
		triggered = age_seconds > expected_interval
		return triggered, age_seconds if triggered else None
	
	async def _evaluate_composite_condition(self, rule: MonitoringRule, metrics: List[MonitoringMetric]) -> tuple[bool, Optional[float]]:
		"""Evaluate composite condition with multiple criteria"""
		# Parse composite condition - simple implementation
		# Expected format: "value > 80 AND rate > 5" or "value < 10 OR absence"
		
		condition = rule.condition.lower()
		
		# Split on AND/OR operators
		if ' and ' in condition:
			parts = condition.split(' and ')
			operator = 'and'
		elif ' or ' in condition:
			parts = condition.split(' or ')
			operator = 'or'
		else:
			# Single condition - treat as threshold
			return await self._evaluate_threshold_condition(rule, metrics)
		
		results = []
		trigger_value = None
		
		for part in parts:
			part = part.strip()
			
			if 'value' in part:
				# Parse value condition
				match = re.search(r'value\s*([><=!]+)\s*([-+]?\d+\.?\d*)', part)
				if match and metrics:
					op = match.group(1)
					threshold = float(match.group(2))
					latest_value = max(metrics, key=lambda m: m.timestamp).value
					
					if op in ['>', 'gt']:
						result = latest_value > threshold
					elif op in ['<', 'lt']:
						result = latest_value < threshold
					elif op in ['>=', 'gte']:
						result = latest_value >= threshold
					elif op in ['<=', 'lte']:
						result = latest_value <= threshold
					else:
						result = False
					
					results.append(result)
					if result:
						trigger_value = latest_value
			
			elif 'rate' in part:
				# Parse rate condition
				rate_result, rate_value = await self._evaluate_rate_condition(rule, metrics)
				results.append(rate_result)
				if rate_result:
					trigger_value = rate_value
			
			elif 'absence' in part:
				# Parse absence condition
				absence_result, absence_value = await self._evaluate_absence_condition(rule, metrics)
				results.append(absence_result)
				if absence_result:
					trigger_value = absence_value
		
		# Combine results based on operator
		if operator == 'and':
			final_result = all(results)
		else:  # or
			final_result = any(results)
		
		return final_result, trigger_value if final_result else None
	
	def _format_alert_message(self, rule: MonitoringRule, trigger_value: Optional[float]) -> str:
		"""Format alert message with variable substitution"""
		message = rule.alert_message
		
		# Simple variable substitution
		if trigger_value is not None:
			message = message.replace('{value}', str(trigger_value))
			if rule.threshold_value is not None:
				message = message.replace('{threshold}', str(rule.threshold_value))
		
		message = message.replace('{metric_name}', rule.metric_name)
		message = message.replace('{rule_name}', rule.name)
		message = message.replace('{timestamp}', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'))
		
		return message


class AlertCorrelationEngine:
	"""Intelligent alert correlation and deduplication"""
	
	def __init__(self, config: dict = None):
		self.config = config or {}
		self.correlations: Dict[str, AlertCorrelation] = {}
		self.correlation_window_minutes = self.config.get('correlation_window_minutes', 5)
		self.similarity_threshold = self.config.get('similarity_threshold', 0.8)
		
		# Correlation strategies
		self.strategies: Dict[AlertCorrelationStrategy, callable] = {
			AlertCorrelationStrategy.TIME_BASED: self._correlate_by_time,
			AlertCorrelationStrategy.SIMILARITY: self._correlate_by_similarity,
			AlertCorrelationStrategy.DEPENDENCY: self._correlate_by_dependency,
			AlertCorrelationStrategy.CAUSE_EFFECT: self._correlate_by_cause_effect
		}
	
	async def correlate_alert(self, alert: MonitoringAlert, existing_alerts: List[MonitoringAlert]) -> Optional[str]:
		"""Correlate new alert with existing alerts"""
		
		# Try each correlation strategy
		for strategy in self.strategies:
			correlation_id = await self.strategies[strategy](alert, existing_alerts)
			if correlation_id:
				return correlation_id
		
		return None
	
	async def _correlate_by_time(self, alert: MonitoringAlert, existing_alerts: List[MonitoringAlert]) -> Optional[str]:
		"""Correlate alerts that occurred within time window"""
		window_start = alert.created_at - timedelta(minutes=self.correlation_window_minutes)
		
		for existing_alert in existing_alerts:
			if (existing_alert.created_at >= window_start and 
			    existing_alert.tenant_id == alert.tenant_id and
			    existing_alert.severity == alert.severity):
				
				# Check if already part of correlation
				correlation_id = self._find_correlation_for_alert(existing_alert.alert_id)
				if correlation_id:
					correlation = self.correlations[correlation_id]
					correlation.add_related_alert(alert.alert_id)
					return correlation_id
				else:
					# Create new correlation
					correlation_id = uuid7str()
					correlation = AlertCorrelation(
						correlation_id=correlation_id,
						primary_alert_id=existing_alert.alert_id,
						related_alert_ids=[alert.alert_id],
						correlation_strategy=AlertCorrelationStrategy.TIME_BASED,
						confidence_score=0.7,
						created_at=datetime.utcnow(),
						correlation_reason=f"Alerts occurred within {self.correlation_window_minutes} minutes"
					)
					self.correlations[correlation_id] = correlation
					return correlation_id
		
		return None
	
	async def _correlate_by_similarity(self, alert: MonitoringAlert, existing_alerts: List[MonitoringAlert]) -> Optional[str]:
		"""Correlate alerts with similar characteristics"""
		for existing_alert in existing_alerts:
			similarity_score = self._calculate_alert_similarity(alert, existing_alert)
			
			if similarity_score >= self.similarity_threshold:
				correlation_id = self._find_correlation_for_alert(existing_alert.alert_id)
				if correlation_id:
					correlation = self.correlations[correlation_id]
					correlation.add_related_alert(alert.alert_id)
					return correlation_id
				else:
					# Create new correlation
					correlation_id = uuid7str()
					correlation = AlertCorrelation(
						correlation_id=correlation_id,
						primary_alert_id=existing_alert.alert_id,
						related_alert_ids=[alert.alert_id],
						correlation_strategy=AlertCorrelationStrategy.SIMILARITY,
						confidence_score=similarity_score,
						created_at=datetime.utcnow(),
						correlation_reason=f"Alerts are {similarity_score:.1%} similar"
					)
					self.correlations[correlation_id] = correlation
					return correlation_id
		
		return None
	
	async def _correlate_by_dependency(self, alert: MonitoringAlert, existing_alerts: List[MonitoringAlert]) -> Optional[str]:
		"""Correlate alerts based on service dependencies"""
		# This would integrate with service dependency mapping
		# For now, implement basic dependency correlation
		
		for existing_alert in existing_alerts:
			if self._are_services_dependent(alert, existing_alert):
				correlation_id = self._find_correlation_for_alert(existing_alert.alert_id)
				if correlation_id:
					correlation = self.correlations[correlation_id]
					correlation.add_related_alert(alert.alert_id)
					return correlation_id
				else:
					correlation_id = uuid7str()
					correlation = AlertCorrelation(
						correlation_id=correlation_id,
						primary_alert_id=existing_alert.alert_id,
						related_alert_ids=[alert.alert_id],
						correlation_strategy=AlertCorrelationStrategy.DEPENDENCY,
						confidence_score=0.9,
						created_at=datetime.utcnow(),
						correlation_reason="Services have dependency relationship"
					)
					self.correlations[correlation_id] = correlation
					return correlation_id
		
		return None
	
	async def _correlate_by_cause_effect(self, alert: MonitoringAlert, existing_alerts: List[MonitoringAlert]) -> Optional[str]:
		"""Correlate alerts based on cause-effect relationships"""
		# Implement cause-effect correlation logic
		# This would use ML models or rule-based patterns
		
		for existing_alert in existing_alerts:
			if self._is_cause_effect_relationship(existing_alert, alert):
				correlation_id = self._find_correlation_for_alert(existing_alert.alert_id)
				if correlation_id:
					correlation = self.correlations[correlation_id]
					correlation.add_related_alert(alert.alert_id)
					return correlation_id
				else:
					correlation_id = uuid7str()
					correlation = AlertCorrelation(
						correlation_id=correlation_id,
						primary_alert_id=existing_alert.alert_id,
						related_alert_ids=[alert.alert_id],
						correlation_strategy=AlertCorrelationStrategy.CAUSE_EFFECT,
						confidence_score=0.85,
						created_at=datetime.utcnow(),
						correlation_reason="Cause-effect relationship detected"
					)
					self.correlations[correlation_id] = correlation
					return correlation_id
		
		return None
	
	def _calculate_alert_similarity(self, alert1: MonitoringAlert, alert2: MonitoringAlert) -> float:
		"""Calculate similarity score between two alerts"""
		if alert1.tenant_id != alert2.tenant_id:
			return 0.0
		
		score = 0.0
		
		# Severity match
		if alert1.severity == alert2.severity:
			score += 0.3
		
		# Source metric match
		if alert1.source_metric == alert2.source_metric:
			score += 0.4
		
		# Label similarity
		common_labels = set(alert1.labels.items()) & set(alert2.labels.items())
		total_labels = len(set(alert1.labels.items()) | set(alert2.labels.items()))
		if total_labels > 0:
			label_similarity = len(common_labels) / total_labels
			score += 0.3 * label_similarity
		
		return score
	
	def _are_services_dependent(self, alert1: MonitoringAlert, alert2: MonitoringAlert) -> bool:
		"""Check if services in alerts have dependency relationship"""
		# Placeholder for service dependency logic
		# In real implementation, this would query service dependency graph
		
		service1 = alert1.labels.get('service', 'unknown')
		service2 = alert2.labels.get('service', 'unknown')
		
		# Simple heuristic - services with similar names might be related
		return service1 != 'unknown' and service2 != 'unknown' and service1 in service2 or service2 in service1
	
	def _is_cause_effect_relationship(self, cause_alert: MonitoringAlert, effect_alert: MonitoringAlert) -> bool:
		"""Check if there's a cause-effect relationship between alerts"""
		# Placeholder for cause-effect analysis
		# In real implementation, this would use ML models or predefined patterns
		
		# Time-based cause-effect (cause happens before effect)
		if cause_alert.created_at >= effect_alert.created_at:
			return False
		
		# Simple pattern matching
		cause_metric = cause_alert.source_metric or ""
		effect_metric = effect_alert.source_metric or ""
		
		# Example patterns: disk space -> application errors, memory -> performance
		patterns = [
			("disk", "error"),
			("memory", "latency"),
			("cpu", "response_time"),
			("network", "timeout")
		]
		
		for cause_pattern, effect_pattern in patterns:
			if cause_pattern in cause_metric.lower() and effect_pattern in effect_metric.lower():
				return True
		
		return False
	
	def _find_correlation_for_alert(self, alert_id: str) -> Optional[str]:
		"""Find existing correlation for alert"""
		for correlation_id, correlation in self.correlations.items():
			if (correlation.primary_alert_id == alert_id or 
			    alert_id in correlation.related_alert_ids):
				return correlation_id
		return None
	
	def get_correlation_stats(self) -> dict:
		"""Get correlation statistics"""
		total_correlations = len(self.correlations)
		total_alerts_in_correlations = sum(
			correlation.get_total_alerts() for correlation in self.correlations.values()
		)
		
		strategy_distribution = defaultdict(int)
		for correlation in self.correlations.values():
			strategy_distribution[correlation.correlation_strategy.value] += 1
		
		return {
			'total_correlations': total_correlations,
			'total_alerts_in_correlations': total_alerts_in_correlations,
			'strategy_distribution': dict(strategy_distribution),
			'avg_alerts_per_correlation': total_alerts_in_correlations / max(total_correlations, 1)
		}


class AlertEngine:
	"""
	Comprehensive alert engine with rule evaluation, correlation, and escalation
	Provides intelligent alerting with reduced noise and automated management
	"""
	
	def __init__(self, config: dict = None):
		self.config = config or {}
		self.running = False
		
		# Core components
		self.evaluator = AlertEvaluator()
		self.correlator = AlertCorrelationEngine(config.get('correlation', {}))
		
		# Rule and alert storage
		self.rules: Dict[str, MonitoringRule] = {}
		self.active_alerts: Dict[str, MonitoringAlert] = {}
		self.escalation_policies: Dict[str, EscalationPolicy] = {}
		
		# Processing queues and background tasks
		self.evaluation_queue = asyncio.Queue()
		self.alert_queue = asyncio.Queue()
		self.background_tasks: List[asyncio.Task] = []
		
		# Performance tracking
		self.stats = {
			'total_rules': 0,
			'active_alerts': 0,
			'alerts_created': 0,
			'alerts_resolved': 0,
			'suppressed_alerts': 0,
			'correlated_alerts': 0,
			'avg_evaluation_time_ms': 0.0,
			'escalations_triggered': 0
		}
		
		print("[AlertEngine] Alert engine initialized")
	
	async def initialize(self) -> None:
		"""Initialize the alert engine"""
		assert not self.running, "Alert engine is already running"
		
		# Start background processors
		self.background_tasks = [
			asyncio.create_task(self._rule_evaluation_loop()),
			asyncio.create_task(self._alert_processing_loop()),
			asyncio.create_task(self._escalation_manager_loop()),
			asyncio.create_task(self._stats_update_loop())
		]
		
		self.running = True
		print("[AlertEngine] Alert engine started successfully")
	
	async def shutdown(self) -> None:
		"""Shutdown the alert engine"""
		if not self.running:
			return
		
		self.running = False
		
		# Cancel background tasks
		for task in self.background_tasks:
			task.cancel()
		
		await asyncio.gather(*self.background_tasks, return_exceptions=True)
		print("[AlertEngine] Alert engine shutdown complete")
	
	async def add_rule(self, rule: MonitoringRule) -> str:
		"""Add alert rule to engine"""
		assert rule.enabled, "Cannot add disabled rule"
		
		self.rules[rule.rule_id] = rule
		self.stats['total_rules'] = len(self.rules)
		
		print(f"[AlertEngine] Added rule: {rule.name} ({rule.rule_id})")
		return rule.rule_id
	
	async def remove_rule(self, rule_id: str) -> bool:
		"""Remove alert rule from engine"""
		if rule_id in self.rules:
			rule = self.rules.pop(rule_id)
			self.stats['total_rules'] = len(self.rules)
			print(f"[AlertEngine] Removed rule: {rule.name} ({rule_id})")
			return True
		return False
	
	async def evaluate_rules_for_metrics(self, metrics: List[MonitoringMetric]) -> List[MonitoringAlert]:
		"""Evaluate all rules against metrics and return triggered alerts"""
		triggered_alerts = []
		
		for rule in self.rules.values():
			if not rule.enabled or not rule.is_due_for_evaluation():
				continue
			
			alert = await self.evaluator.evaluate_rule(rule, metrics)
			if alert:
				# Check for suppression
				if await self._should_suppress_alert(alert):
					self.stats['suppressed_alerts'] += 1
					continue
				
				# Add to processing queue for correlation and escalation
				await self.alert_queue.put(alert)
				triggered_alerts.append(alert)
		
		return triggered_alerts
	
	async def get_active_alerts(self, tenant_id: str = None, severity: AlertSeverity = None) -> List[MonitoringAlert]:
		"""Get active alerts with optional filtering"""
		alerts = list(self.active_alerts.values())
		
		if tenant_id:
			alerts = [a for a in alerts if a.tenant_id == tenant_id]
		
		if severity:
			alerts = [a for a in alerts if a.severity == severity]
		
		# Only return active alerts
		return [a for a in alerts if a.is_active()]
	
	async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
		"""Acknowledge an alert"""
		if alert_id in self.active_alerts:
			alert = self.active_alerts[alert_id]
			alert.status = AlertStatus.ACKNOWLEDGED
			alert.acknowledged_at = datetime.utcnow()
			alert.annotations['acknowledged_by'] = acknowledged_by
			
			print(f"[AlertEngine] Alert acknowledged: {alert_id}")
			return True
		
		return False
	
	async def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
		"""Resolve an alert"""
		if alert_id in self.active_alerts:
			alert = self.active_alerts[alert_id]
			alert.status = AlertStatus.RESOLVED
			alert.resolved_at = datetime.utcnow()
			alert.annotations['resolved_by'] = resolved_by
			
			self.stats['alerts_resolved'] += 1
			print(f"[AlertEngine] Alert resolved: {alert_id}")
			return True
		
		return False
	
	async def get_engine_stats(self) -> dict:
		"""Get comprehensive engine statistics"""
		correlation_stats = self.correlator.get_correlation_stats()
		evaluation_stats = self.evaluator.evaluation_stats
		
		return {
			**self.stats,
			'correlation_stats': correlation_stats,
			'evaluation_stats': evaluation_stats,
			'queue_sizes': {
				'evaluation_queue': self.evaluation_queue.qsize(),
				'alert_queue': self.alert_queue.qsize()
			},
			'running': self.running,
			'timestamp': datetime.utcnow().isoformat()
		}
	
	# Private implementation methods
	async def _rule_evaluation_loop(self) -> None:
		"""Background loop for rule evaluation"""
		try:
			while self.running:
				await asyncio.sleep(10)  # Evaluate every 10 seconds
				
				# This would be triggered by incoming metrics in real implementation
				# For now, we simulate evaluation requests
				
		except asyncio.CancelledError:
			pass
		except Exception as e:
			print(f"[AlertEngine] Error in rule evaluation loop: {e}")
	
	async def _alert_processing_loop(self) -> None:
		"""Background loop for alert processing and correlation"""
		try:
			while self.running:
				try:
					# Wait for new alert
					alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
					
					# Process alert
					await self._process_new_alert(alert)
					
				except asyncio.TimeoutError:
					continue
					
		except asyncio.CancelledError:
			pass
		except Exception as e:
			print(f"[AlertEngine] Error in alert processing loop: {e}")
	
	async def _escalation_manager_loop(self) -> None:
		"""Background loop for alert escalation management"""
		try:
			while self.running:
				await asyncio.sleep(30)  # Check escalations every 30 seconds
				
				# Check active alerts for escalation
				for alert in self.active_alerts.values():
					if alert.should_escalate():
						await self._escalate_alert(alert)
				
		except asyncio.CancelledError:
			pass
		except Exception as e:
			print(f"[AlertEngine] Error in escalation manager loop: {e}")
	
	async def _stats_update_loop(self) -> None:
		"""Background loop for statistics updates"""
		try:
			while self.running:
				await asyncio.sleep(60)  # Update stats every minute
				
				# Update active alert count
				self.stats['active_alerts'] = len([
					a for a in self.active_alerts.values() if a.is_active()
				])
				
				# Clean up resolved alerts older than 24 hours
				await self._cleanup_resolved_alerts()
				
		except asyncio.CancelledError:
			pass
		except Exception as e:
			print(f"[AlertEngine] Error in stats update loop: {e}")
	
	async def _process_new_alert(self, alert: MonitoringAlert) -> None:
		"""Process new alert with correlation and storage"""
		try:
			# Attempt correlation with existing alerts
			existing_alerts = list(self.active_alerts.values())
			correlation_id = await self.correlator.correlate_alert(alert, existing_alerts)
			
			if correlation_id:
				alert.correlation_key = correlation_id
				self.stats['correlated_alerts'] += 1
			
			# Store alert
			self.active_alerts[alert.alert_id] = alert
			self.stats['alerts_created'] += 1
			
			print(f"[AlertEngine] New alert created: {alert.name} ({alert.alert_id})")
			
		except Exception as e:
			print(f"[AlertEngine] Error processing new alert: {e}")
	
	async def _should_suppress_alert(self, alert: MonitoringAlert) -> bool:
		"""Check if alert should be suppressed"""
		if not alert.parent_alert_id:
			return False
		
		# Check if parent alert exists and is still active
		parent_alert = self.active_alerts.get(alert.parent_alert_id)
		return parent_alert is not None and parent_alert.is_active()
	
	async def _escalate_alert(self, alert: MonitoringAlert) -> None:
		"""Escalate alert to next level"""
		try:
			if not alert.can_escalate():
				return
			
			alert.escalation_level += 1
			alert.updated_at = datetime.utcnow()
			
			self.stats['escalations_triggered'] += 1
			
			print(f"[AlertEngine] Alert escalated to level {alert.escalation_level}: {alert.alert_id}")
			
			# Here we would integrate with notification system
			
		except Exception as e:
			print(f"[AlertEngine] Error escalating alert {alert.alert_id}: {e}")
	
	async def _cleanup_resolved_alerts(self) -> None:
		"""Clean up old resolved alerts"""
		cutoff_time = datetime.utcnow() - timedelta(hours=24)
		
		alerts_to_remove = [
			alert_id for alert_id, alert in self.active_alerts.items()
			if (alert.status == AlertStatus.RESOLVED and 
			    alert.resolved_at and alert.resolved_at < cutoff_time)
		]
		
		for alert_id in alerts_to_remove:
			del self.active_alerts[alert_id]


# Factory function
def create_alert_engine(config: dict = None) -> AlertEngine:
	"""Create and configure alert engine"""
	return AlertEngine(config)