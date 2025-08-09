#!/usr/bin/env python3
"""
APG Cache Management (CACH) - Intelligent Cache Warming
Smart cold start elimination with historical pattern analysis

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import math

from .models import CacheEntry, CacheAccessPattern, CacheTier
from .predictive_engine import PredictionResult, ContentRelationship


class WarmingStrategy(str, Enum):
	"""Cache warming strategies"""
	HISTORICAL_PATTERN = "historical_pattern"
	BUSINESS_LOGIC = "business_logic"
	PREDICTIVE_MODEL = "predictive_model"
	USER_BEHAVIOR = "user_behavior"
	CONTENT_RELATIONSHIP = "content_relationship"
	TEMPORAL_SCHEDULE = "temporal_schedule"
	ADAPTIVE_LEARNING = "adaptive_learning"


class WarmingPriority(str, Enum):
	"""Warming task priorities"""
	CRITICAL = "critical"     # System critical data
	HIGH = "high"            # Frequently accessed data
	MEDIUM = "medium"        # Moderately accessed data  
	LOW = "low"              # Rarely accessed but beneficial
	BACKGROUND = "background" # Best-effort warming


@dataclass
class WarmingTask:
	"""Cache warming task definition"""
	task_id: str
	key: str
	strategy: WarmingStrategy
	priority: WarmingPriority
	scheduled_time: datetime
	data_source: str  # Where to fetch the data from
	fetch_function: Optional[Callable] = None
	
	# Prediction metadata
	confidence_score: float = 0.0
	expected_hit_rate_improvement: float = 0.0
	estimated_fetch_cost: float = 0.0
	
	# Execution metadata
	created_at: datetime = field(default_factory=datetime.utcnow)
	attempted_at: Optional[datetime] = None
	completed_at: Optional[datetime] = None
	success: bool = False
	error_message: Optional[str] = None
	
	# Performance tracking
	fetch_duration_ms: float = 0.0
	cache_size_bytes: int = 0
	actual_hit_count: int = 0


@dataclass
class WarmingPattern:
	"""Historical warming pattern"""
	pattern_id: str
	key_pattern: str
	temporal_pattern: Dict[int, float]  # Hour -> warming effectiveness
	success_rate: float
	average_hit_rate_improvement: float
	last_updated: datetime
	usage_count: int


class IntelligentWarmingEngine:
	"""
	Revolutionary intelligent cache warming system
	Revolutionary Differentiator #3: Intelligent Cache Warming
	"""
	
	def __init__(self, config: Dict[str, Any] = None):
		self.config = config or {}
		self.logger = logging.getLogger('cach.warming_engine')
		
		# Warming state
		self.warming_tasks: Dict[str, WarmingTask] = {}
		self.completed_tasks: deque = deque(maxlen=1000)
		self.warming_patterns: Dict[str, WarmingPattern] = {}
		self.data_sources: Dict[str, Callable] = {}
		
		# Warming queue management
		self.priority_queues = {
			WarmingPriority.CRITICAL: deque(),
			WarmingPriority.HIGH: deque(), 
			WarmingPriority.MEDIUM: deque(),
			WarmingPriority.LOW: deque(),
			WarmingPriority.BACKGROUND: deque()
		}
		
		# Configuration
		self.max_concurrent_warming = self.config.get('max_concurrent_warming', 10)
		self.warming_batch_size = self.config.get('warming_batch_size', 50)
		self.historical_analysis_days = self.config.get('historical_analysis_days', 30)
		self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.6)
		
		# Performance tracking
		self.warming_effectiveness_history: deque = deque(maxlen=1000)
		self.cold_start_elimination_rate = 0.0
		
		# Background task management
		self._warming_worker_tasks: Set[asyncio.Task] = set()
		self._running = False
	
	async def initialize(self) -> None:
		"""Initialize intelligent warming engine"""
		self.logger.info("Initializing intelligent cache warming engine...")
		
		# Load historical warming patterns
		await self._load_warming_patterns()
		
		# Initialize data source connectors
		await self._initialize_data_sources()
		
		# Start background warming workers
		await self._start_warming_workers()
		
		self._running = True
		self.logger.info("Intelligent warming engine initialized")
	
	async def shutdown(self) -> None:
		"""Shutdown warming engine"""
		self.logger.info("Shutting down intelligent warming engine...")
		
		self._running = False
		
		# Cancel background workers
		for task in self._warming_worker_tasks:
			task.cancel()
		
		await asyncio.gather(*self._warming_worker_tasks, return_exceptions=True)
		
		# Save warming patterns
		await self._save_warming_patterns()
		
		self.logger.info("Intelligent warming engine shut down")
	
	async def analyze_cold_start_opportunities(self, cache_entries: Dict[str, CacheEntry],
											   access_history: List[Dict[str, Any]] = None) -> List[WarmingTask]:
		"""
		Analyze cache for cold start elimination opportunities
		Revolutionary Differentiator #3: Smart Cold Start Elimination
		"""
		
		opportunities = []
		access_history = access_history or []
		
		# Historical pattern analysis
		historical_tasks = await self._analyze_historical_patterns(cache_entries, access_history)
		opportunities.extend(historical_tasks)
		
		# Business logic analysis
		business_logic_tasks = await self._analyze_business_logic_warming(cache_entries)
		opportunities.extend(business_logic_tasks)
		
		# Predictive model analysis
		predictive_tasks = await self._analyze_predictive_warming(cache_entries)
		opportunities.extend(predictive_tasks)
		
		# Content relationship analysis
		relationship_tasks = await self._analyze_relationship_warming(cache_entries)
		opportunities.extend(relationship_tasks)
		
		# Temporal schedule analysis
		temporal_tasks = await self._analyze_temporal_warming()
		opportunities.extend(temporal_tasks)
		
		# Filter and prioritize tasks
		filtered_opportunities = await self._filter_and_prioritize_tasks(opportunities)
		
		self.logger.info(f"Identified {len(filtered_opportunities)} warming opportunities")
		return filtered_opportunities
	
	async def schedule_warming_task(self, task: WarmingTask) -> str:
		"""Schedule a warming task for execution"""
		
		# Validate task
		if not await self._validate_warming_task(task):
			raise ValueError(f"Invalid warming task: {task.task_id}")
		
		# Store task
		self.warming_tasks[task.task_id] = task
		
		# Add to appropriate priority queue
		self.priority_queues[task.priority].append(task.task_id)
		
		self.logger.debug(f"Scheduled warming task: {task.key} (priority: {task.priority.value})")
		return task.task_id
	
	async def execute_warming_batch(self, max_tasks: int = None) -> Dict[str, Any]:
		"""Execute a batch of warming tasks"""
		
		max_tasks = max_tasks or self.warming_batch_size
		executed_tasks = []
		results = {
			'executed': 0,
			'succeeded': 0,
			'failed': 0,
			'tasks': []
		}
		
		# Execute tasks from priority queues
		tasks_executed = 0
		for priority in WarmingPriority:
			if tasks_executed >= max_tasks:
				break
			
			queue = self.priority_queues[priority]
			while queue and tasks_executed < max_tasks:
				task_id = queue.popleft()
				
				if task_id in self.warming_tasks:
					task = self.warming_tasks[task_id]
					result = await self._execute_warming_task(task)
					executed_tasks.append(task)
					results['tasks'].append(result)
					tasks_executed += 1
		
		# Update results
		results['executed'] = len(executed_tasks)
		results['succeeded'] = sum(1 for task in executed_tasks if task.success)
		results['failed'] = sum(1 for task in executed_tasks if not task.success)
		
		# Move completed tasks to history
		for task in executed_tasks:
			if task.task_id in self.warming_tasks:
				del self.warming_tasks[task.task_id]
			self.completed_tasks.append(task)
		
		# Update warming effectiveness
		await self._update_warming_effectiveness(executed_tasks)
		
		self.logger.info(f"Executed warming batch: {results['succeeded']}/{results['executed']} succeeded")
		return results
	
	async def proactive_warming_cycle(self, cache_entries: Dict[str, CacheEntry],
									  predictions: List[PredictionResult] = None) -> Dict[str, Any]:
		"""
		Execute proactive warming cycle based on predictions
		Proactive content loading with usage pattern analysis
		"""
		
		predictions = predictions or []
		cycle_results = {
			'opportunities_identified': 0,
			'tasks_scheduled': 0,
			'tasks_executed': 0,
			'cold_starts_eliminated': 0,
			'effectiveness_score': 0.0
		}
		
		# Identify warming opportunities
		opportunities = await self.analyze_cold_start_opportunities(cache_entries)
		cycle_results['opportunities_identified'] = len(opportunities)
		
		# Enhance opportunities with prediction data
		enhanced_opportunities = await self._enhance_with_predictions(opportunities, predictions)
		
		# Schedule high-confidence tasks
		scheduled_count = 0
		for opportunity in enhanced_opportunities:
			if opportunity.confidence_score >= self.min_confidence_threshold:
				await self.schedule_warming_task(opportunity)
				scheduled_count += 1
		
		cycle_results['tasks_scheduled'] = scheduled_count
		
		# Execute immediate high-priority tasks
		if scheduled_count > 0:
			execution_results = await self.execute_warming_batch(max_tasks=min(scheduled_count, 20))
			cycle_results['tasks_executed'] = execution_results['succeeded']
		
		# Calculate effectiveness
		cycle_results['effectiveness_score'] = await self._calculate_warming_effectiveness()
		
		self.logger.info(f"Proactive warming cycle completed: {cycle_results}")
		return cycle_results
	
	async def register_data_source(self, source_name: str, fetch_function: Callable) -> None:
		"""Register a data source for cache warming"""
		
		self.data_sources[source_name] = fetch_function
		self.logger.info(f"Registered data source: {source_name}")
	
	async def get_warming_statistics(self) -> Dict[str, Any]:
		"""Get comprehensive warming statistics"""
		
		total_completed = len(self.completed_tasks)
		successful_tasks = sum(1 for task in self.completed_tasks if task.success)
		
		return {
			'total_patterns': len(self.warming_patterns),
			'active_tasks': len(self.warming_tasks),
			'completed_tasks': total_completed,
			'success_rate': successful_tasks / max(total_completed, 1),
			'cold_start_elimination_rate': self.cold_start_elimination_rate,
			'average_effectiveness': sum(self.warming_effectiveness_history) / max(len(self.warming_effectiveness_history), 1),
			'queue_sizes': {
				priority.value: len(queue) 
				for priority, queue in self.priority_queues.items()
			},
			'data_sources_registered': len(self.data_sources)
		}
	
	# Private implementation methods
	
	async def _analyze_historical_patterns(self, cache_entries: Dict[str, CacheEntry],
											access_history: List[Dict[str, Any]]) -> List[WarmingTask]:
		"""Analyze historical access patterns for warming opportunities"""
		
		tasks = []
		
		# Analyze access patterns by hour of day
		hourly_access = defaultdict(list)
		for access_record in access_history:
			access_time = access_record.get('timestamp')
			if isinstance(access_time, datetime):
				hour = access_time.hour
				key = access_record.get('key')
				if key:
					hourly_access[hour].append(key)
		
		# Find patterns that consistently occur at specific times
		current_hour = datetime.utcnow().hour
		next_hour = (current_hour + 1) % 24
		
		# Keys frequently accessed in the next hour
		if next_hour in hourly_access:
			frequent_keys = defaultdict(int)
			for key in hourly_access[next_hour]:
				frequent_keys[key] += 1
			
			# Create warming tasks for frequently accessed keys
			for key, frequency in frequent_keys.items():
				if frequency >= 3 and key not in cache_entries:  # Threshold: 3+ accesses
					confidence = min(frequency / 10.0, 0.9)  # Scale confidence
					
					task = WarmingTask(
						task_id=f"historical_{key}_{next_hour}",
						key=key,
						strategy=WarmingStrategy.HISTORICAL_PATTERN,
						priority=WarmingPriority.HIGH if confidence > 0.7 else WarmingPriority.MEDIUM,
						scheduled_time=datetime.utcnow() + timedelta(minutes=30),
						data_source="default",
						confidence_score=confidence,
						expected_hit_rate_improvement=confidence * 20.0
					)
					tasks.append(task)
		
		return tasks
	
	async def _analyze_business_logic_warming(self, cache_entries: Dict[str, CacheEntry]) -> List[WarmingTask]:
		"""Analyze business logic for warming opportunities"""
		
		tasks = []
		
		# Common business logic patterns
		business_patterns = [
			{
				'name': 'user_profile_completion',
				'pattern': 'user:*:profile',
				'related_keys': ['user:*:preferences', 'user:*:settings'],
				'confidence': 0.85
			},
			{
				'name': 'product_catalog_browsing',
				'pattern': 'product:*:details',
				'related_keys': ['product:*:reviews', 'product:*:recommendations'],
				'confidence': 0.75
			},
			{
				'name': 'api_endpoint_dependencies',
				'pattern': 'api:*:response',
				'related_keys': ['api:*:metadata', 'api:*:cache'],
				'confidence': 0.8
			}
		]
		
		# Check if any cached keys match patterns and warm related keys
		for entry_key in cache_entries:
			for pattern_config in business_patterns:
				if self._key_matches_pattern(entry_key, pattern_config['pattern']):
					# Warm related keys
					for related_pattern in pattern_config['related_keys']:
						related_key = self._generate_related_key(entry_key, related_pattern)
						
						if related_key and related_key not in cache_entries:
							task = WarmingTask(
								task_id=f"business_{pattern_config['name']}_{related_key}",
								key=related_key,
								strategy=WarmingStrategy.BUSINESS_LOGIC,
								priority=WarmingPriority.MEDIUM,
								scheduled_time=datetime.utcnow(),
								data_source="default",
								confidence_score=pattern_config['confidence'],
								expected_hit_rate_improvement=pattern_config['confidence'] * 15.0
							)
							tasks.append(task)
		
		return tasks
	
	async def _analyze_predictive_warming(self, cache_entries: Dict[str, CacheEntry]) -> List[WarmingTask]:
		"""Analyze predictive model outputs for warming opportunities"""
		
		tasks = []
		
		# This would integrate with the predictive engine
		# For now, simulate predictive warming based on access patterns
		
		high_frequency_keys = [
			key for key, entry in cache_entries.items()
			if entry.access_frequency > 10
		]
		
		# Predict related keys that might be accessed soon
		for key in high_frequency_keys:
			predicted_keys = self._predict_related_keys(key)
			
			for predicted_key, confidence in predicted_keys:
				if predicted_key not in cache_entries and confidence > 0.6:
					task = WarmingTask(
						task_id=f"predictive_{predicted_key}",
						key=predicted_key,
						strategy=WarmingStrategy.PREDICTIVE_MODEL,
						priority=WarmingPriority.HIGH if confidence > 0.8 else WarmingPriority.MEDIUM,
						scheduled_time=datetime.utcnow() + timedelta(minutes=5),
						data_source="default",
						confidence_score=confidence,
						expected_hit_rate_improvement=confidence * 25.0
					)
					tasks.append(task)
		
		return tasks
	
	async def _analyze_relationship_warming(self, cache_entries: Dict[str, CacheEntry]) -> List[WarmingTask]:
		"""Analyze content relationships for warming opportunities"""
		
		tasks = []
		
		# Find keys with strong access correlations
		recently_accessed = [
			key for key, entry in cache_entries.items()
			if (entry.last_accessed and 
				(datetime.utcnow() - entry.last_accessed).total_seconds() < 3600)
		]
		
		# For each recently accessed key, find related keys to warm
		for key in recently_accessed:
			related_keys = self._find_relationship_keys(key, cache_entries)
			
			for related_key, strength in related_keys:
				if related_key not in cache_entries and strength > 0.5:
					task = WarmingTask(
						task_id=f"relationship_{related_key}",
						key=related_key,
						strategy=WarmingStrategy.CONTENT_RELATIONSHIP,
						priority=WarmingPriority.MEDIUM,
						scheduled_time=datetime.utcnow() + timedelta(minutes=10),
						data_source="default",
						confidence_score=strength,
						expected_hit_rate_improvement=strength * 18.0
					)
					tasks.append(task)
		
		return tasks
	
	async def _analyze_temporal_warming(self) -> List[WarmingTask]:
		"""Analyze temporal schedules for warming opportunities"""
		
		tasks = []
		current_time = datetime.utcnow()
		
		# Define temporal warming schedules
		schedules = [
			{
				'name': 'morning_rush',
				'time_range': (8, 10),
				'keys': ['dashboard:*', 'reports:*', 'notifications:*'],
				'confidence': 0.9
			},
			{
				'name': 'lunch_break',
				'time_range': (12, 13),
				'keys': ['social:*', 'news:*', 'weather:*'],
				'confidence': 0.7
			},
			{
				'name': 'end_of_day',
				'time_range': (17, 18),
				'keys': ['analytics:*', 'reports:*', 'backup:*'],
				'confidence': 0.8
			}
		]
		
		current_hour = current_time.hour
		
		# Check if we're approaching any scheduled warming times
		for schedule in schedules:
			start_hour, end_hour = schedule['time_range']
			
			# If we're 30 minutes before the scheduled time
			if (start_hour - 1) <= current_hour < start_hour or (current_hour == 23 and start_hour == 0):
				for key_pattern in schedule['keys']:
					# Generate specific keys from patterns
					specific_keys = self._generate_keys_from_pattern(key_pattern, limit=5)
					
					for specific_key in specific_keys:
						task = WarmingTask(
							task_id=f"temporal_{schedule['name']}_{specific_key}",
							key=specific_key,
							strategy=WarmingStrategy.TEMPORAL_SCHEDULE,
							priority=WarmingPriority.HIGH,
							scheduled_time=current_time.replace(hour=start_hour, minute=0, second=0),
							data_source="default",
							confidence_score=schedule['confidence'],
							expected_hit_rate_improvement=schedule['confidence'] * 30.0
						)
						tasks.append(task)
		
		return tasks
	
	async def _filter_and_prioritize_tasks(self, tasks: List[WarmingTask]) -> List[WarmingTask]:
		"""Filter and prioritize warming tasks"""
		
		# Remove duplicate tasks
		unique_tasks = {}
		for task in tasks:
			if task.key not in unique_tasks or task.confidence_score > unique_tasks[task.key].confidence_score:
				unique_tasks[task.key] = task
		
		filtered_tasks = list(unique_tasks.values())
		
		# Filter by confidence threshold
		high_confidence_tasks = [
			task for task in filtered_tasks
			if task.confidence_score >= self.min_confidence_threshold
		]
		
		# Sort by priority and confidence
		prioritized_tasks = sorted(high_confidence_tasks, key=lambda t: (
			t.priority == WarmingPriority.CRITICAL,
			t.priority == WarmingPriority.HIGH,
			t.confidence_score,
			t.expected_hit_rate_improvement
		), reverse=True)
		
		# Limit total tasks to prevent overwhelming the system
		return prioritized_tasks[:100]
	
	async def _validate_warming_task(self, task: WarmingTask) -> bool:
		"""Validate a warming task"""
		
		# Check required fields
		if not task.task_id or not task.key:
			return False
		
		# Check data source availability
		if task.data_source not in self.data_sources:
			return False
		
		# Check confidence threshold
		if task.confidence_score < 0.1:  # Minimum confidence
			return False
		
		# Check scheduling time is reasonable
		if task.scheduled_time < datetime.utcnow() - timedelta(hours=1):
			return False
		
		return True
	
	async def _execute_warming_task(self, task: WarmingTask) -> Dict[str, Any]:
		"""Execute a single warming task"""
		
		start_time = datetime.utcnow()
		task.attempted_at = start_time
		
		try:
			# Fetch data from source
			if task.data_source in self.data_sources:
				fetch_function = self.data_sources[task.data_source]
				data = await fetch_function(task.key)
				
				if data is not None:
					# Successfully fetched data
					task.success = True
					task.completed_at = datetime.utcnow()
					task.fetch_duration_ms = (task.completed_at - start_time).total_seconds() * 1000
					
					# Estimate cache size (simplified)
					if isinstance(data, (str, bytes)):
						task.cache_size_bytes = len(data)
					else:
						task.cache_size_bytes = len(str(data))
					
					return {
						'task_id': task.task_id,
						'key': task.key,
						'success': True,
						'fetch_duration_ms': task.fetch_duration_ms,
						'cache_size_bytes': task.cache_size_bytes
					}
				else:
					task.error_message = "No data returned from source"
			else:
				task.error_message = f"Data source not found: {task.data_source}"
		
		except Exception as e:
			task.error_message = str(e)
		
		# Task failed
		task.success = False
		task.completed_at = datetime.utcnow()
		task.fetch_duration_ms = (task.completed_at - start_time).total_seconds() * 1000
		
		return {
			'task_id': task.task_id,
			'key': task.key,
			'success': False,
			'error': task.error_message
		}
	
	async def _enhance_with_predictions(self, opportunities: List[WarmingTask],
									   predictions: List[PredictionResult]) -> List[WarmingTask]:
		"""Enhance warming opportunities with prediction data"""
		
		# Create prediction lookup
		prediction_map = {pred.target_key: pred for pred in predictions}
		
		enhanced_opportunities = []
		for opportunity in opportunities:
			if opportunity.key in prediction_map:
				prediction = prediction_map[opportunity.key]
				
				# Boost confidence based on prediction
				original_confidence = opportunity.confidence_score
				prediction_boost = prediction.confidence_score * 0.3
				opportunity.confidence_score = min(original_confidence + prediction_boost, 1.0)
				
				# Adjust expected improvement
				opportunity.expected_hit_rate_improvement *= (1.0 + prediction_boost)
				
				# Upgrade priority if prediction is very confident
				if prediction.confidence_score > 0.9 and opportunity.priority != WarmingPriority.CRITICAL:
					if opportunity.priority == WarmingPriority.MEDIUM:
						opportunity.priority = WarmingPriority.HIGH
					elif opportunity.priority == WarmingPriority.LOW:
						opportunity.priority = WarmingPriority.MEDIUM
			
			enhanced_opportunities.append(opportunity)
		
		return enhanced_opportunities
	
	# Utility helper methods
	
	def _key_matches_pattern(self, key: str, pattern: str) -> bool:
		"""Check if key matches a pattern"""
		import fnmatch
		return fnmatch.fnmatch(key, pattern)
	
	def _generate_related_key(self, base_key: str, related_pattern: str) -> Optional[str]:
		"""Generate a related key from base key and pattern"""
		
		# Simple pattern replacement (would be more sophisticated)
		if ':*:' in related_pattern:
			parts = base_key.split(':')
			if len(parts) >= 3:
				pattern_parts = related_pattern.split(':')
				if len(pattern_parts) == 3 and pattern_parts[1] == '*':
					return f"{pattern_parts[0]}:{parts[1]}:{pattern_parts[2]}"
		
		return None
	
	def _predict_related_keys(self, key: str) -> List[Tuple[str, float]]:
		"""Predict related keys (simplified implementation)"""
		
		related_keys = []
		
		# Simple prediction logic based on key patterns
		if 'user:' in key:
			user_id = key.split(':')[1] if ':' in key else 'unknown'
			related_keys.extend([
				(f"user:{user_id}:preferences", 0.8),
				(f"user:{user_id}:settings", 0.7),
				(f"user:{user_id}:history", 0.6)
			])
		
		elif 'product:' in key:
			product_id = key.split(':')[1] if ':' in key else 'unknown'
			related_keys.extend([
				(f"product:{product_id}:reviews", 0.75),
				(f"product:{product_id}:recommendations", 0.7),
				(f"product:{product_id}:inventory", 0.65)
			])
		
		return related_keys
	
	def _find_relationship_keys(self, key: str, cache_entries: Dict[str, CacheEntry]) -> List[Tuple[str, float]]:
		"""Find related keys based on access patterns"""
		
		related_keys = []
		
		# Find keys accessed around the same time
		if key not in cache_entries:
			return related_keys
		
		base_entry = cache_entries[key]
		if not base_entry.last_accessed:
			return related_keys
		
		# Find entries accessed within 5 minutes
		time_window = timedelta(minutes=5)
		
		for other_key, other_entry in cache_entries.items():
			if other_key != key and other_entry.last_accessed:
				time_diff = abs((other_entry.last_accessed - base_entry.last_accessed).total_seconds())
				if time_diff <= time_window.total_seconds():
					strength = 1.0 - (time_diff / time_window.total_seconds())
					related_keys.append((other_key, strength))
		
		return related_keys
	
	def _generate_keys_from_pattern(self, pattern: str, limit: int = 10) -> List[str]:
		"""Generate specific keys from a pattern"""
		
		keys = []
		
		# Simple key generation (would be more sophisticated)
		if pattern == 'dashboard:*':
			keys = ['dashboard:main', 'dashboard:analytics', 'dashboard:reports']
		elif pattern == 'reports:*':
			keys = ['reports:daily', 'reports:weekly', 'reports:monthly']
		elif pattern == 'notifications:*':
			keys = ['notifications:system', 'notifications:user', 'notifications:alerts']
		elif pattern == 'social:*':
			keys = ['social:feed', 'social:messages', 'social:friends']
		elif pattern == 'news:*':
			keys = ['news:headlines', 'news:tech', 'news:business']
		elif pattern == 'weather:*':
			keys = ['weather:current', 'weather:forecast', 'weather:alerts']
		elif pattern == 'analytics:*':
			keys = ['analytics:traffic', 'analytics:performance', 'analytics:users']
		elif pattern == 'backup:*':
			keys = ['backup:database', 'backup:files', 'backup:config']
		
		return keys[:limit]
	
	async def _calculate_warming_effectiveness(self) -> float:
		"""Calculate overall warming effectiveness"""
		
		if not self.completed_tasks:
			return 0.0
		
		# Calculate success rate
		successful_tasks = sum(1 for task in self.completed_tasks if task.success)
		success_rate = successful_tasks / len(self.completed_tasks)
		
		# Calculate average hit rate improvement
		improvements = [
			task.expected_hit_rate_improvement for task in self.completed_tasks
			if task.success and task.expected_hit_rate_improvement > 0
		]
		
		avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0
		
		# Combined effectiveness score
		effectiveness = (success_rate * 0.7) + (min(avg_improvement / 100.0, 1.0) * 0.3)
		
		return effectiveness
	
	async def _update_warming_effectiveness(self, executed_tasks: List[WarmingTask]) -> None:
		"""Update warming effectiveness metrics"""
		
		if executed_tasks:
			effectiveness = await self._calculate_warming_effectiveness()
			self.warming_effectiveness_history.append(effectiveness)
			
			# Update cold start elimination rate
			successful_count = sum(1 for task in executed_tasks if task.success)
			if successful_count > 0:
				self.cold_start_elimination_rate = (self.cold_start_elimination_rate * 0.9) + (successful_count / len(executed_tasks) * 0.1)
	
	async def _start_warming_workers(self) -> None:
		"""Start background warming worker tasks"""
		
		for i in range(min(self.max_concurrent_warming, 5)):
			task = asyncio.create_task(self._warming_worker(f"worker_{i}"))
			self._warming_worker_tasks.add(task)
			task.add_done_callback(self._warming_worker_tasks.discard)
	
	async def _warming_worker(self, worker_name: str) -> None:
		"""Background warming worker"""
		
		while self._running:
			try:
				# Check for tasks to execute
				has_tasks = any(len(queue) > 0 for queue in self.priority_queues.values())
				
				if has_tasks:
					await self.execute_warming_batch(max_tasks=5)
				else:
					await asyncio.sleep(30)  # Check every 30 seconds
			
			except Exception as e:
				self.logger.error(f"Error in warming worker {worker_name}: {e}")
				await asyncio.sleep(60)
	
	async def _load_warming_patterns(self) -> None:
		"""Load historical warming patterns"""
		# Placeholder - would load from persistent storage
		pass
	
	async def _save_warming_patterns(self) -> None:
		"""Save warming patterns to persistent storage"""
		# Placeholder - would save to persistent storage
		pass
	
	async def _initialize_data_sources(self) -> None:
		"""Initialize default data sources"""
		
		# Register default data source
		async def default_fetch_function(key: str) -> Optional[Any]:
			"""Default data fetch function"""
			# Placeholder implementation
			# Would integrate with actual data sources (databases, APIs, etc.)
			await asyncio.sleep(0.1)  # Simulate fetch time
			return f"data_for_{key}"
		
		await self.register_data_source("default", default_fetch_function)


# Export main components
__all__ = [
	'IntelligentWarmingEngine',
	'WarmingStrategy',
	'WarmingPriority', 
	'WarmingTask',
	'WarmingPattern'
]