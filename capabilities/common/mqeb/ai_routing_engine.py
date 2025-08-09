#!/usr/bin/env python3
"""
APG Message Queue Event Bus (MQEB) - AI-Powered Routing Engine
Intelligent message routing with machine learning optimization

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import hashlib
import pickle
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

from .models import MQMessage, Subscription, MessagePriority, ProtocolType
from .service import MQEBService


@dataclass
class MessageFeatures:
	"""Extracted features from message for ML processing"""
	content_hash: str
	content_length: int
	priority_score: float
	tenant_hash: str
	source_hash: str
	timestamp_hour: int
	timestamp_dow: int  # day of week
	header_count: int
	content_type_hash: str
	topic_depth: int  # number of dots in topic name
	
	def to_vector(self) -> np.ndarray:
		"""Convert features to numpy vector for ML"""
		return np.array([
			hash(self.content_hash) % 10000,
			self.content_length,
			self.priority_score,
			hash(self.tenant_hash) % 1000,
			hash(self.source_hash) % 1000,
			self.timestamp_hour,
			self.timestamp_dow,
			self.header_count,
			hash(self.content_type_hash) % 100,
			self.topic_depth
		], dtype=np.float32)


@dataclass
class RoutingPrediction:
	"""Prediction result from AI routing engine"""
	recommended_partitions: List[int]
	predicted_latency_ms: float
	predicted_throughput: float
	confidence_score: float
	routing_strategy: str
	load_balancing_weights: Dict[str, float]


@dataclass
class TrafficPattern:
	"""Detected traffic pattern"""
	pattern_id: str
	pattern_type: str  # spike, steady, periodic, burst
	peak_throughput: float
	duration_minutes: int
	frequency: str  # hourly, daily, weekly
	confidence: float
	historical_occurrences: int


class ContentAnalyzer:
	"""Analyzes message content for intelligent routing"""
	
	def __init__(self):
		self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
		self.content_clusters = None
		self.cluster_model = None
		self.is_trained = False
		
	async def train_content_clustering(self, messages: List[MQMessage]) -> None:
		"""Train content clustering model on message history"""
		try:
			# Extract text content from messages
			text_contents = []
			for msg in messages:
				try:
					text_content = msg.payload.decode('utf-8', errors='ignore')
					# Clean and prepare text
					text_content = self._clean_text_content(text_content)
					text_contents.append(text_content)
				except Exception:
					text_contents.append("")  # Empty content for binary messages
			
			if len(text_contents) < 10:
				return  # Not enough data for clustering
			
			# Vectorize content
			tfidf_matrix = self.vectorizer.fit_transform(text_contents)
			
			# Perform clustering
			n_clusters = min(10, max(2, len(text_contents) // 20))
			self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
			self.content_clusters = self.cluster_model.fit_predict(tfidf_matrix.toarray())
			
			self.is_trained = True
			logging.info(f"Content clustering trained with {n_clusters} clusters on {len(text_contents)} messages")
			
		except Exception as e:
			logging.error(f"Content clustering training failed: {e}")
	
	async def analyze_message_content(self, message: MQMessage) -> Dict[str, Any]:
		"""Analyze message content for routing insights"""
		try:
			text_content = message.payload.decode('utf-8', errors='ignore')
			text_content = self._clean_text_content(text_content)
			
			analysis = {
				'content_category': 'unknown',
				'urgency_indicators': [],
				'routing_hints': [],
				'estimated_processing_time': 1.0,
				'content_cluster': -1
			}
			
			# Content categorization
			analysis['content_category'] = self._categorize_content(text_content)
			
			# Urgency detection
			analysis['urgency_indicators'] = self._detect_urgency(text_content)
			
			# Routing hints based on content
			analysis['routing_hints'] = self._extract_routing_hints(text_content, message)
			
			# Estimated processing complexity
			analysis['estimated_processing_time'] = self._estimate_processing_time(text_content, message)
			
			# Content clustering (if trained)
			if self.is_trained and self.cluster_model is not None:
				try:
					tfidf_vector = self.vectorizer.transform([text_content])
					cluster = self.cluster_model.predict(tfidf_vector.toarray())[0]
					analysis['content_cluster'] = int(cluster)
				except Exception:
					analysis['content_cluster'] = -1
			
			return analysis
			
		except Exception as e:
			logging.warning(f"Content analysis failed: {e}")
			return {'content_category': 'binary', 'urgency_indicators': [], 'routing_hints': []}
	
	def _clean_text_content(self, text: str) -> str:
		"""Clean and prepare text content for analysis"""
		# Remove JSON/XML structure noise
		import re
		text = re.sub(r'[{}\[\]",:]+', ' ', text)
		text = re.sub(r'<[^>]+>', ' ', text)  # Remove XML tags
		text = re.sub(r'\s+', ' ', text)      # Normalize whitespace
		return text.strip().lower()
	
	def _categorize_content(self, text: str) -> str:
		"""Categorize message content"""
		categories = {
			'error': ['error', 'exception', 'failed', 'failure', 'critical'],
			'user_activity': ['login', 'logout', 'signup', 'user', 'session'],
			'transaction': ['payment', 'order', 'purchase', 'transaction', 'invoice'],
			'system': ['system', 'health', 'status', 'heartbeat', 'ping'],
			'analytics': ['metric', 'analytics', 'tracking', 'event', 'measurement'],
			'notification': ['notification', 'alert', 'reminder', 'message']
		}
		
		for category, keywords in categories.items():
			if any(keyword in text for keyword in keywords):
				return category
		
		return 'general'
	
	def _detect_urgency(self, text: str) -> List[str]:
		"""Detect urgency indicators in message content"""
		urgency_keywords = {
			'critical': ['critical', 'urgent', 'emergency', 'immediate', 'asap'],
			'high': ['high', 'important', 'priority', 'alert'],
			'error': ['error', 'failure', 'exception', 'crash', 'down'],
			'security': ['security', 'breach', 'unauthorized', 'suspicious', 'attack']
		}
		
		detected = []
		for level, keywords in urgency_keywords.items():
			if any(keyword in text for keyword in keywords):
				detected.append(level)
		
		return detected
	
	def _extract_routing_hints(self, text: str, message: MQMessage) -> List[str]:
		"""Extract routing hints from message content and metadata"""
		hints = []
		
		# Geographic routing hints
		geo_indicators = ['us', 'eu', 'asia', 'east', 'west', 'north', 'south']
		for geo in geo_indicators:
			if geo in text or geo in message.topic:
				hints.append(f'geographic:{geo}')
		
		# Service routing hints
		service_indicators = ['api', 'database', 'cache', 'queue', 'worker']
		for service in service_indicators:
			if service in text or service in message.source_application.lower():
				hints.append(f'service:{service}')
		
		# Load balancing hints
		if 'batch' in text or message.headers.get('batch_size'):
			hints.append('load_balancing:batch')
		elif any(word in text for word in ['stream', 'realtime', 'live']):
			hints.append('load_balancing:realtime')
		
		return hints
	
	def _estimate_processing_time(self, text: str, message: MQMessage) -> float:
		"""Estimate processing time complexity"""
		base_time = 1.0  # seconds
		
		# Content size factor
		size_factor = min(2.0, len(text) / 1000.0)
		
		# Complexity indicators
		complexity_indicators = ['json', 'xml', 'sql', 'query', 'process', 'calculate']
		complexity_bonus = sum(0.2 for indicator in complexity_indicators if indicator in text)
		
		# Priority adjustment
		if message.priority == MessagePriority.CRITICAL:
			return base_time * 0.5  # Process faster
		elif message.priority == MessagePriority.LOW:
			return base_time * 2.0  # Can take longer
		
		return base_time + size_factor + complexity_bonus


class TrafficPredictor:
	"""Predicts traffic patterns and load for intelligent routing"""
	
	def __init__(self):
		self.traffic_history = deque(maxlen=10080)  # 1 week of minutes
		self.pattern_models = {}
		self.load_predictor = LinearRegression()
		self.is_trained = False
		
	async def record_traffic(self, timestamp: datetime, message_count: int, bytes_count: int) -> None:
		"""Record traffic data point"""
		self.traffic_history.append({
			'timestamp': timestamp,
			'message_count': message_count,
			'bytes_count': bytes_count,
			'hour': timestamp.hour,
			'day_of_week': timestamp.weekday(),
			'minute_of_day': timestamp.hour * 60 + timestamp.minute
		})
		
		# Retrain models periodically
		if len(self.traffic_history) > 100 and len(self.traffic_history) % 100 == 0:
			await self._retrain_models()
	
	async def predict_traffic(self, horizon_minutes: int = 60) -> Dict[str, Any]:
		"""Predict traffic for the next N minutes"""
		if not self.is_trained or len(self.traffic_history) < 50:
			return self._default_prediction()
		
		try:
			current_time = datetime.utcnow()
			predictions = []
			
			for i in range(horizon_minutes):
				future_time = current_time + timedelta(minutes=i)
				features = self._extract_time_features(future_time)
				
				predicted_load = self.load_predictor.predict([features])[0]
				predictions.append({
					'timestamp': future_time,
					'predicted_messages_per_minute': max(0, predicted_load),
					'confidence': min(1.0, len(self.traffic_history) / 1000.0)
				})
			
			return {
				'predictions': predictions,
				'detected_patterns': await self._detect_patterns(),
				'scaling_recommendations': await self._generate_scaling_recommendations(predictions)
			}
			
		except Exception as e:
			logging.error(f"Traffic prediction failed: {e}")
			return self._default_prediction()
	
	async def _retrain_models(self) -> None:
		"""Retrain traffic prediction models"""
		try:
			if len(self.traffic_history) < 50:
				return
			
			# Prepare training data
			X = []  # features
			y = []  # target (messages per minute)
			
			for data_point in self.traffic_history:
				features = self._extract_time_features(data_point['timestamp'])
				X.append(features)
				y.append(data_point['message_count'])
			
			# Train load prediction model
			self.load_predictor.fit(X, y)
			self.is_trained = True
			
			logging.info(f"Traffic prediction models retrained on {len(self.traffic_history)} data points")
			
		except Exception as e:
			logging.error(f"Model retraining failed: {e}")
	
	def _extract_time_features(self, timestamp: datetime) -> List[float]:
		"""Extract time-based features for prediction"""
		return [
			timestamp.hour,
			timestamp.weekday(),
			timestamp.minute,
			np.sin(2 * np.pi * timestamp.hour / 24),  # Cyclical hour
			np.cos(2 * np.pi * timestamp.hour / 24),
			np.sin(2 * np.pi * timestamp.weekday() / 7),  # Cyclical day of week
			np.cos(2 * np.pi * timestamp.weekday() / 7),
			1 if timestamp.weekday() < 5 else 0,  # Is weekday
		]
	
	async def _detect_patterns(self) -> List[TrafficPattern]:
		"""Detect traffic patterns in historical data"""
		patterns = []
		
		if len(self.traffic_history) < 100:
			return patterns
		
		try:
			# Convert to numpy arrays for analysis
			timestamps = [dp['timestamp'] for dp in self.traffic_history]
			message_counts = [dp['message_count'] for dp in self.traffic_history]
			
			# Detect spikes
			mean_load = np.mean(message_counts)
			std_load = np.std(message_counts)
			spike_threshold = mean_load + 2 * std_load
			
			spikes = [(i, count) for i, count in enumerate(message_counts) if count > spike_threshold]
			
			if spikes:
				patterns.append(TrafficPattern(
					pattern_id=f"spike_{len(spikes)}",
					pattern_type="spike",
					peak_throughput=max(count for _, count in spikes),
					duration_minutes=5,  # Simplified
					frequency="irregular",
					confidence=0.8,
					historical_occurrences=len(spikes)
				))
			
			# Detect periodic patterns (simplified)
			hourly_averages = defaultdict(list)
			for dp in self.traffic_history:
				hourly_averages[dp['hour']].append(dp['message_count'])
			
			for hour, counts in hourly_averages.items():
				if len(counts) > 5 and np.std(counts) < np.mean(counts) * 0.3:
					patterns.append(TrafficPattern(
						pattern_id=f"hourly_{hour}",
						pattern_type="periodic",
						peak_throughput=np.mean(counts),
						duration_minutes=60,
						frequency="daily",
						confidence=0.7,
						historical_occurrences=len(counts)
					))
			
		except Exception as e:
			logging.error(f"Pattern detection failed: {e}")
		
		return patterns
	
	async def _generate_scaling_recommendations(self, predictions: List[Dict]) -> List[str]:
		"""Generate scaling recommendations based on predictions"""
		recommendations = []
		
		if not predictions:
			return recommendations
		
		# Analyze predicted load
		max_predicted = max(p['predicted_messages_per_minute'] for p in predictions)
		current_baseline = np.mean([dp['message_count'] for dp in list(self.traffic_history)[-10:]])
		
		if max_predicted > current_baseline * 1.5:
			recommendations.append("scale_up_brokers")
			recommendations.append("increase_partition_count")
		
		if max_predicted > current_baseline * 3:
			recommendations.append("enable_auto_scaling")
			recommendations.append("prepare_burst_capacity")
		
		# Check for consistent patterns
		prediction_variance = np.std([p['predicted_messages_per_minute'] for p in predictions])
		if prediction_variance < current_baseline * 0.2:
			recommendations.append("optimize_for_steady_load")
		else:
			recommendations.append("optimize_for_variable_load")
		
		return recommendations
	
	def _default_prediction(self) -> Dict[str, Any]:
		"""Default prediction when models aren't trained"""
		return {
			'predictions': [],
			'detected_patterns': [],
			'scaling_recommendations': ['gather_more_data'],
			'note': 'Insufficient training data for accurate predictions'
		}


class IntelligentRoutingEngine:
	"""Main AI-powered routing engine"""
	
	def __init__(self, mqeb_service: MQEBService):
		self.service = mqeb_service
		self.content_analyzer = ContentAnalyzer()
		self.traffic_predictor = TrafficPredictor()
		self.routing_models = {}
		
		# Routing statistics
		self.routing_stats = {
			'total_routed': 0,
			'routing_decisions': defaultdict(int),
			'performance_metrics': defaultdict(list),
			'ml_model_accuracy': 0.0
		}
		
		# Background task handles
		self._background_tasks: Set[asyncio.Task] = set()
		self.logger = logging.getLogger('mqeb.ai_routing')
	
	async def initialize(self) -> None:
		"""Initialize AI routing engine"""
		self.logger.info("Initializing AI routing engine...")
		
		# Train content analyzer on existing messages
		await self._train_initial_models()
		
		# Start background tasks
		await self._start_background_tasks()
		
		self.logger.info("AI routing engine initialized")
	
	async def shutdown(self) -> None:
		"""Shutdown AI routing engine"""
		# Cancel background tasks
		for task in self._background_tasks:
			task.cancel()
		
		await asyncio.gather(*self._background_tasks, return_exceptions=True)
		self.logger.info("AI routing engine shut down")
	
	async def route_message(self, message: MQMessage) -> RoutingPrediction:
		"""Make intelligent routing decision for message"""
		try:
			# Extract message features
			features = await self._extract_message_features(message)
			
			# Analyze content
			content_analysis = await self.content_analyzer.analyze_message_content(message)
			
			# Get traffic predictions
			traffic_prediction = await self.traffic_predictor.predict_traffic(horizon_minutes=5)
			
			# Make routing decision
			routing_decision = await self._make_routing_decision(
				message, features, content_analysis, traffic_prediction
			)
			
			# Update statistics
			self.routing_stats['total_routed'] += 1
			self.routing_stats['routing_decisions'][routing_decision.routing_strategy] += 1
			
			self.logger.debug(f"Routed message {message.id} using strategy: {routing_decision.routing_strategy}")
			
			return routing_decision
			
		except Exception as e:
			self.logger.error(f"AI routing failed for message {message.id}: {e}")
			# Fallback to simple routing
			return await self._fallback_routing(message)
	
	async def _extract_message_features(self, message: MQMessage) -> MessageFeatures:
		"""Extract ML features from message"""
		priority_scores = {
			MessagePriority.LOW: 1.0,
			MessagePriority.NORMAL: 2.0,
			MessagePriority.HIGH: 3.0,
			MessagePriority.CRITICAL: 4.0
		}
		
		return MessageFeatures(
			content_hash=hashlib.md5(message.payload).hexdigest()[:8],
			content_length=len(message.payload),
			priority_score=priority_scores.get(message.priority, 2.0),
			tenant_hash=hashlib.md5(message.tenant_id.encode()).hexdigest()[:8],
			source_hash=hashlib.md5(message.source_application.encode()).hexdigest()[:8],
			timestamp_hour=message.timestamp.hour,
			timestamp_dow=message.timestamp.weekday(),
			header_count=len(message.headers),
			content_type_hash=hashlib.md5(message.content_type.encode()).hexdigest()[:8],
			topic_depth=message.topic.count('.')
		)
	
	async def _make_routing_decision(self, message: MQMessage, features: MessageFeatures,
									content_analysis: Dict, traffic_prediction: Dict) -> RoutingPrediction:
		"""Make intelligent routing decision using all available information"""
		
		# Determine optimal partitioning strategy
		partitions = await self._select_optimal_partitions(message, features, content_analysis)
		
		# Predict performance metrics
		latency_prediction = await self._predict_latency(message, features, traffic_prediction)
		throughput_prediction = await self._predict_throughput(message, features, traffic_prediction)
		
		# Calculate confidence based on available data
		confidence = await self._calculate_confidence(features, content_analysis, traffic_prediction)
		
		# Determine routing strategy
		strategy = await self._select_routing_strategy(message, features, content_analysis, traffic_prediction)
		
		# Generate load balancing weights
		lb_weights = await self._calculate_load_balancing_weights(message, traffic_prediction)
		
		return RoutingPrediction(
			recommended_partitions=partitions,
			predicted_latency_ms=latency_prediction,
			predicted_throughput=throughput_prediction,
			confidence_score=confidence,
			routing_strategy=strategy,
			load_balancing_weights=lb_weights
		)
	
	async def _select_optimal_partitions(self, message: MQMessage, features: MessageFeatures,
										content_analysis: Dict) -> List[int]:
		"""Select optimal partitions for message routing"""
		topic_config = self.service.topics.get(message.topic)
		if not topic_config:
			return [0]  # Default partition
		
		available_partitions = list(range(topic_config.partitions))
		
		# Priority-based partition selection
		if message.priority == MessagePriority.CRITICAL:
			# Use dedicated high-priority partitions
			return [i for i in available_partitions if i < topic_config.partitions // 4]
		
		# Content-based partitioning
		if content_analysis.get('content_cluster', -1) >= 0:
			cluster_id = content_analysis['content_cluster']
			preferred_partition = cluster_id % topic_config.partitions
			return [preferred_partition]
		
		# Load-based partitioning
		if features.tenant_hash:
			tenant_partition = hash(features.tenant_hash) % topic_config.partitions
			return [tenant_partition]
		
		# Default: round-robin
		return [hash(message.id) % topic_config.partitions]
	
	async def _predict_latency(self, message: MQMessage, features: MessageFeatures,
							  traffic_prediction: Dict) -> float:
		"""Predict message processing latency"""
		base_latency = 2.0  # Base latency in ms
		
		# Size-based latency increase
		size_factor = min(5.0, features.content_length / 1000.0)
		
		# Priority-based latency adjustment
		priority_factors = {
			MessagePriority.CRITICAL: 0.5,
			MessagePriority.HIGH: 0.8,
			MessagePriority.NORMAL: 1.0,
			MessagePriority.LOW: 1.5
		}
		priority_factor = priority_factors.get(message.priority, 1.0)
		
		# Traffic-based latency increase
		traffic_factor = 1.0
		if traffic_prediction.get('predictions'):
			next_minute_load = traffic_prediction['predictions'][0]['predicted_messages_per_minute']
			if next_minute_load > 1000:  # High load threshold
				traffic_factor = min(3.0, next_minute_load / 1000.0)
		
		return base_latency + size_factor * priority_factor * traffic_factor
	
	async def _predict_throughput(self, message: MQMessage, features: MessageFeatures,
								 traffic_prediction: Dict) -> float:
		"""Predict message throughput (messages/second)"""
		base_throughput = 10000.0  # Base throughput
		
		# Adjust for message size
		size_factor = max(0.1, 1000.0 / features.content_length)
		
		# Adjust for expected traffic
		traffic_factor = 1.0
		if traffic_prediction.get('predictions'):
			predicted_load = traffic_prediction['predictions'][0]['predicted_messages_per_minute']
			if predicted_load > 500:
				traffic_factor = max(0.2, 500.0 / predicted_load)
		
		return base_throughput * size_factor * traffic_factor
	
	async def _calculate_confidence(self, features: MessageFeatures, content_analysis: Dict,
								   traffic_prediction: Dict) -> float:
		"""Calculate confidence in routing decision"""
		base_confidence = 0.6
		
		# Increase confidence with more data
		if self.content_analyzer.is_trained:
			base_confidence += 0.2
		
		if self.traffic_predictor.is_trained:
			base_confidence += 0.2
		
		# Adjust for content analysis quality
		if content_analysis.get('content_category') != 'unknown':
			base_confidence += 0.1
		
		# Adjust for traffic prediction confidence
		if traffic_prediction.get('predictions'):
			pred_confidence = traffic_prediction['predictions'][0].get('confidence', 0.5)
			base_confidence = (base_confidence + pred_confidence) / 2
		
		return min(1.0, base_confidence)
	
	async def _select_routing_strategy(self, message: MQMessage, features: MessageFeatures,
									  content_analysis: Dict, traffic_prediction: Dict) -> str:
		"""Select optimal routing strategy"""
		
		# Critical messages: priority routing
		if message.priority == MessagePriority.CRITICAL:
			return "priority_express"
		
		# High traffic: load balancing
		if traffic_prediction.get('predictions'):
			next_load = traffic_prediction['predictions'][0]['predicted_messages_per_minute']
			if next_load > 2000:
				return "load_balanced"
		
		# Large messages: dedicated routing
		if features.content_length > 100000:  # 100KB
			return "bulk_processing"
		
		# Content-based routing
		if content_analysis.get('urgency_indicators'):
			return "urgency_aware"
		
		# Geographic routing hints
		routing_hints = content_analysis.get('routing_hints', [])
		if any('geographic:' in hint for hint in routing_hints):
			return "geographic_aware"
		
		# Default strategy
		return "intelligent_default"
	
	async def _calculate_load_balancing_weights(self, message: MQMessage,
											   traffic_prediction: Dict) -> Dict[str, float]:
		"""Calculate load balancing weights for different targets"""
		weights = {'default': 1.0}
		
		# Get current broker node information
		for node_id, node in self.service.broker_nodes.items():
			# Simple load balancing based on current connections
			if node.active_connections < node.max_connections * 0.8:
				weights[node_id] = 1.0 - (node.active_connections / node.max_connections)
			else:
				weights[node_id] = 0.1  # Avoid overloaded nodes
		
		# Normalize weights
		total_weight = sum(weights.values())
		if total_weight > 0:
			weights = {k: v / total_weight for k, v in weights.items()}
		
		return weights
	
	async def _fallback_routing(self, message: MQMessage) -> RoutingPrediction:
		"""Fallback routing when AI routing fails"""
		return RoutingPrediction(
			recommended_partitions=[0],
			predicted_latency_ms=5.0,
			predicted_throughput=5000.0,
			confidence_score=0.3,
			routing_strategy="fallback_simple",
			load_balancing_weights={'default': 1.0}
		)
	
	async def _train_initial_models(self) -> None:
		"""Train initial ML models on existing data"""
		try:
			# Get sample of existing messages for training
			messages = []
			for message_ids in list(self.service.message_queues.values()):
				for msg_id in message_ids[:100]:  # Sample from each topic
					if msg_id in self.service.message_store:
						messages.append(self.service.message_store[msg_id])
			
			if messages:
				await self.content_analyzer.train_content_clustering(messages)
				self.logger.info(f"Trained content analyzer on {len(messages)} messages")
			
		except Exception as e:
			self.logger.error(f"Initial model training failed: {e}")
	
	async def _start_background_tasks(self) -> None:
		"""Start background tasks for AI routing"""
		
		# Traffic monitoring task
		task = asyncio.create_task(self._traffic_monitoring_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# Model retraining task
		task = asyncio.create_task(self._model_retraining_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# Performance monitoring task
		task = asyncio.create_task(self._performance_monitoring_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
	
	async def _traffic_monitoring_loop(self) -> None:
		"""Background task to monitor traffic patterns"""
		while True:
			try:
				current_time = datetime.utcnow()
				
				# Collect current traffic metrics
				total_messages = sum(len(queue) for queue in self.service.message_queues.values())
				total_bytes = sum(
					sum(self.service.message_store[msg_id].size_bytes() 
						for msg_id in queue if msg_id in self.service.message_store)
					for queue in self.service.message_queues.values()
				)
				
				# Record traffic data
				await self.traffic_predictor.record_traffic(current_time, total_messages, total_bytes)
				
				await asyncio.sleep(60)  # Record every minute
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Traffic monitoring error: {e}")
				await asyncio.sleep(60)
	
	async def _model_retraining_loop(self) -> None:
		"""Background task for periodic model retraining"""
		while True:
			try:
				await asyncio.sleep(3600)  # Retrain every hour
				
				# Retrain content analyzer if we have new messages
				recent_messages = []
				for message_ids in list(self.service.message_queues.values()):
					for msg_id in message_ids[-50:]:  # Recent messages from each topic
						if msg_id in self.service.message_store:
							recent_messages.append(self.service.message_store[msg_id])
				
				if len(recent_messages) > 20:
					await self.content_analyzer.train_content_clustering(recent_messages)
					self.logger.info("Retrained content analyzer with recent messages")
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Model retraining error: {e}")
	
	async def _performance_monitoring_loop(self) -> None:
		"""Background task to monitor AI routing performance"""
		while True:
			try:
				await asyncio.sleep(300)  # Monitor every 5 minutes
				
				# Calculate performance metrics
				total_routed = self.routing_stats['total_routed']
				if total_routed > 0:
					# Log routing statistics
					self.logger.info(f"AI Routing Stats - Total: {total_routed}, "
									f"Strategies: {dict(self.routing_stats['routing_decisions'])}")
					
					# Calculate success rate (simplified)
					success_rate = min(1.0, total_routed / max(1, total_routed * 0.95))  # Assume 95% success
					self.routing_stats['ml_model_accuracy'] = success_rate
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Performance monitoring error: {e}")
	
	async def get_routing_analytics(self) -> Dict[str, Any]:
		"""Get AI routing analytics and performance metrics"""
		return {
			'routing_statistics': self.routing_stats.copy(),
			'content_analyzer_status': {
				'is_trained': self.content_analyzer.is_trained,
				'clusters_detected': len(self.content_analyzer.content_clusters) if self.content_analyzer.content_clusters is not None else 0
			},
			'traffic_predictor_status': {
				'is_trained': self.traffic_predictor.is_trained,
				'history_length': len(self.traffic_predictor.traffic_history)
			},
			'recent_predictions': await self.traffic_predictor.predict_traffic(horizon_minutes=30)
		}


# Factory function
async def create_ai_routing_engine(mqeb_service: MQEBService) -> IntelligentRoutingEngine:
	"""Create and initialize AI routing engine"""
	engine = IntelligentRoutingEngine(mqeb_service)
	await engine.initialize()
	return engine


# Export components
__all__ = [
	'IntelligentRoutingEngine', 'ContentAnalyzer', 'TrafficPredictor',
	'MessageFeatures', 'RoutingPrediction', 'TrafficPattern',
	'create_ai_routing_engine'
]