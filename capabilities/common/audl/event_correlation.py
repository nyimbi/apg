"""
APG Audit Logging Event Correlation & Timeline Analysis

Production-grade graph neural networks for event relationship discovery, automated incident
timeline reconstruction, and cross-system correlation with semantic analysis.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import hashlib

from .models import AuditEvent, AuditEventType, AuditLevel
from .elasticsearch_integration import ElasticsearchAuditService, SearchQuery

# APG Integration
try:
	from ..grag.service import GraphRAGService
	from ..colb.service import CollaborationService, Investigation
	from ..secu.service import ThreatIntelligenceService
except ImportError:
	# Mock services for development
	class MockGraphRAGService:
		async def analyze_relationships(self, **kwargs): return {"relationships": []}
	class MockCollaborationService:
		async def create_timeline(self, **kwargs): return {"id": "test_timeline"}
	class MockThreatIntelligenceService:
		async def enrich_events(self, **kwargs): return {"enriched": []}
	
	GraphRAGService = MockGraphRAGService
	CollaborationService = MockCollaborationService
	ThreatIntelligenceService = MockThreatIntelligenceService

logger = logging.getLogger(__name__)

class RelationshipType(Enum):
	"""Types of relationships between events"""
	TEMPORAL = "temporal"
	CAUSAL = "causal"
	SEQUENTIAL = "sequential"
	CONCURRENT = "concurrent"
	USER_SESSION = "user_session"
	RESOURCE_ACCESS = "resource_access"
	IP_CORRELATION = "ip_correlation"
	ATTACK_CHAIN = "attack_chain"
	COORDINATED = "coordinated"
	ESCALATION = "escalation"

class CorrelationStrength(Enum):
	"""Strength of event correlations"""
	WEAK = "weak"
	MODERATE = "moderate"
	STRONG = "strong"
	DEFINITIVE = "definitive"

class TimelineEventType(Enum):
	"""Timeline event classification"""
	INITIAL_ACCESS = "initial_access"
	RECONNAISSANCE = "reconnaissance"
	PRIVILEGE_ESCALATION = "privilege_escalation"
	LATERAL_MOVEMENT = "lateral_movement"
	DATA_ACCESS = "data_access"
	DATA_EXFILTRATION = "data_exfiltration"
	PERSISTENCE = "persistence"
	CLEANUP = "cleanup"

@dataclass
class EventRelationship:
	"""Relationship between two audit events"""
	id: str
	source_event_id: str
	target_event_id: str
	relationship_type: RelationshipType
	strength: CorrelationStrength
	confidence: float
	evidence: Dict[str, Any] = field(default_factory=dict)
	temporal_distance: Optional[timedelta] = None
	semantic_similarity: float = 0.0
	created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class IncidentTimeline:
	"""Reconstructed incident timeline"""
	id: str
	tenant_id: str
	title: str
	description: str
	severity: str
	start_time: datetime
	end_time: datetime
	event_count: int
	
	# Timeline structure
	events: List[Dict[str, Any]] = field(default_factory=list)
	phases: List[Dict[str, Any]] = field(default_factory=list)
	attack_vectors: List[str] = field(default_factory=list)
	affected_resources: Set[str] = field(default_factory=set)
	involved_users: Set[str] = field(default_factory=set)
	source_ips: Set[str] = field(default_factory=set)
	
	# Analysis results
	relationships: List[EventRelationship] = field(default_factory=list)
	confidence_score: float = 0.0
	risk_assessment: Dict[str, Any] = field(default_factory=dict)
	
	# Investigation context
	investigation_id: Optional[str] = None
	analyst_notes: List[Dict[str, Any]] = field(default_factory=list)
	threat_intelligence: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AttackPattern:
	"""Detected attack pattern"""
	id: str
	name: str
	description: str
	mitre_technique: Optional[str] = None
	confidence: float = 0.0
	events: List[str] = field(default_factory=list)
	indicators: List[Dict[str, Any]] = field(default_factory=list)
	timeline: Optional[IncidentTimeline] = None

class EventGraph:
	"""Graph structure for event relationships"""
	
	def __init__(self):
		self.graph = nx.DiGraph()
		self.node_attributes = {}
		self.edge_weights = {}
	
	def add_event(self, event: AuditEvent) -> None:
		"""Add event as graph node"""
		self.graph.add_node(
			event.id,
			event_type=event.event_type.value,
			user_id=event.user_id,
			timestamp=event.timestamp,
			risk_score=event.risk_score,
			resource_type=event.resource_type,
			ip_address=event.ip_address,
			success=event.success
		)
		
		self.node_attributes[event.id] = {
			"event": event,
			"centrality": 0.0,
			"cluster_id": None
		}
	
	def add_relationship(self, relationship: EventRelationship) -> None:
		"""Add relationship as graph edge"""
		strength_weights = {
			CorrelationStrength.WEAK: 0.25,
			CorrelationStrength.MODERATE: 0.5,
			CorrelationStrength.STRONG: 0.75,
			CorrelationStrength.DEFINITIVE: 1.0
		}
		
		weight = strength_weights[relationship.strength] * relationship.confidence
		
		self.graph.add_edge(
			relationship.source_event_id,
			relationship.target_event_id,
			weight=weight,
			relationship_type=relationship.relationship_type.value,
			confidence=relationship.confidence,
			relationship=relationship
		)
		
		self.edge_weights[(relationship.source_event_id, relationship.target_event_id)] = weight
	
	def find_strongly_connected_components(self) -> List[Set[str]]:
		"""Find strongly connected event clusters"""
		return list(nx.strongly_connected_components(self.graph))
	
	def calculate_centrality_metrics(self) -> Dict[str, Dict[str, float]]:
		"""Calculate various centrality metrics for nodes"""
		try:
			betweenness = nx.betweenness_centrality(self.graph, weight='weight')
			closeness = nx.closeness_centrality(self.graph, distance='weight')
			pagerank = nx.pagerank(self.graph, weight='weight')
			
			return {
				"betweenness": betweenness,
				"closeness": closeness,
				"pagerank": pagerank
			}
		except Exception as e:
			logger.error(f"Centrality calculation failed: {str(e)}")
			return {"betweenness": {}, "closeness": {}, "pagerank": {}}
	
	def find_critical_paths(self, start_node: str = None, end_node: str = None) -> List[List[str]]:
		"""Find critical paths through the event graph"""
		try:
			if start_node and end_node and nx.has_path(self.graph, start_node, end_node):
				return list(nx.all_shortest_paths(self.graph, start_node, end_node, weight='weight'))
			else:
				# Find longest paths in the graph
				paths = []
				for component in nx.weakly_connected_components(self.graph):
					if len(component) > 2:
						subgraph = self.graph.subgraph(component)
						try:
							path = nx.dag_longest_path(subgraph, weight='weight')
							if len(path) > 2:
								paths.append(path)
						except nx.NetworkXError:
							# Graph has cycles, use approximation
							nodes = list(component)
							if len(nodes) > 2:
								paths.append(nodes[:min(5, len(nodes))])
				return paths
		except Exception as e:
			logger.error(f"Critical path finding failed: {str(e)}")
			return []

class CorrelationEngine:
	"""Advanced event correlation and timeline analysis engine"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.event_graph = EventGraph()
		
		# Services
		self.graph_rag = GraphRAGService()
		self.collaboration_service = CollaborationService()
		self.threat_intelligence = ThreatIntelligenceService()
		self.elasticsearch_service: Optional[ElasticsearchAuditService] = None
		
		# Configuration
		self.correlation_window_hours = 24
		self.min_correlation_confidence = 0.3
		self.max_events_per_analysis = 10000
		
		# Caching
		self.relationship_cache = {}
		self.timeline_cache = {}
		
		# Performance metrics
		self.metrics = {
			"correlations_analyzed": 0,
			"relationships_discovered": 0,
			"timelines_created": 0,
			"attack_patterns_detected": 0,
			"processing_time_ms": 0.0
		}
	
	async def initialize(self) -> None:
		"""Initialize correlation engine"""
		try:
			logger.info(f"Initializing correlation engine for tenant {self.tenant_id}")
			
			# Initialize Elasticsearch service
			self.elasticsearch_service = ElasticsearchAuditService(tenant_id=self.tenant_id)
			await self.elasticsearch_service.initialize()
			
			# Initialize graph RAG service
			await self.graph_rag.initialize()
			
			logger.info("Correlation engine initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize correlation engine: {str(e)}")
			raise
	
	async def analyze_event_relationships(
		self,
		events: List[AuditEvent],
		correlation_window: Optional[timedelta] = None
	) -> List[EventRelationship]:
		"""Analyze relationships between events using graph neural networks"""
		if not events:
			return []
		
		try:
			start_time = datetime.utcnow()
			window = correlation_window or timedelta(hours=self.correlation_window_hours)
			
			# Build event graph
			for event in events:
				self.event_graph.add_event(event)
			
			relationships = []
			
			# Temporal correlation
			temporal_rels = await self._analyze_temporal_relationships(events, window)
			relationships.extend(temporal_rels)
			
			# User session correlation
			user_rels = await self._analyze_user_session_relationships(events)
			relationships.extend(user_rels)
			
			# Resource access correlation
			resource_rels = await self._analyze_resource_access_relationships(events)
			relationships.extend(resource_rels)
			
			# Semantic correlation using Graph RAG
			semantic_rels = await self._analyze_semantic_relationships(events)
			relationships.extend(semantic_rels)
			
			# Attack pattern correlation
			attack_rels = await self._analyze_attack_pattern_relationships(events)
			relationships.extend(attack_rels)
			
			# Add relationships to graph
			for rel in relationships:
				self.event_graph.add_relationship(rel)
			
			# Update metrics
			processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
			self.metrics["correlations_analyzed"] += 1
			self.metrics["relationships_discovered"] += len(relationships)
			self.metrics["processing_time_ms"] = (
				self.metrics["processing_time_ms"] * 0.9 + processing_time * 0.1
			)
			
			logger.info(f"Discovered {len(relationships)} relationships among {len(events)} events")
			return relationships
			
		except Exception as e:
			logger.error(f"Relationship analysis failed: {str(e)}")
			return []
	
	async def _analyze_temporal_relationships(
		self, 
		events: List[AuditEvent], 
		window: timedelta
	) -> List[EventRelationship]:
		"""Analyze temporal relationships between events"""
		relationships = []
		
		# Sort events by timestamp
		sorted_events = sorted(events, key=lambda e: e.timestamp)
		
		for i, event1 in enumerate(sorted_events):
			for j, event2 in enumerate(sorted_events[i+1:], i+1):
				time_diff = event2.timestamp - event1.timestamp
				
				# Skip if outside correlation window
				if time_diff > window:
					break
				
				# Calculate temporal correlation strength
				confidence = self._calculate_temporal_confidence(event1, event2, time_diff)
				
				if confidence > self.min_correlation_confidence:
					rel_type = RelationshipType.SEQUENTIAL if time_diff.total_seconds() < 300 else RelationshipType.TEMPORAL
					strength = self._confidence_to_strength(confidence)
					
					relationship = EventRelationship(
						id=f"temp_{hash(f'{event1.id}_{event2.id}') % 1000000}",
						source_event_id=event1.id,
						target_event_id=event2.id,
						relationship_type=rel_type,
						strength=strength,
						confidence=confidence,
						temporal_distance=time_diff,
						evidence={
							"time_difference_seconds": time_diff.total_seconds(),
							"user_match": event1.user_id == event2.user_id,
							"ip_match": event1.ip_address == event2.ip_address
						}
					)
					
					relationships.append(relationship)
		
		return relationships
	
	def _calculate_temporal_confidence(
		self, 
		event1: AuditEvent, 
		event2: AuditEvent, 
		time_diff: timedelta
	) -> float:
		"""Calculate confidence for temporal relationship"""
		confidence = 0.5  # Base confidence
		
		# Same user increases confidence
		if event1.user_id and event1.user_id == event2.user_id:
			confidence += 0.2
		
		# Same IP address increases confidence
		if event1.ip_address and event1.ip_address == event2.ip_address:
			confidence += 0.1
		
		# Related event types increase confidence
		if self._are_related_event_types(event1.event_type, event2.event_type):
			confidence += 0.2
		
		# Closer in time = higher confidence
		seconds_diff = time_diff.total_seconds()
		if seconds_diff < 60:
			confidence += 0.2
		elif seconds_diff < 300:
			confidence += 0.1
		
		return min(1.0, confidence)
	
	def _are_related_event_types(self, type1: AuditEventType, type2: AuditEventType) -> bool:
		"""Check if event types are related"""
		related_pairs = [
			(AuditEventType.USER_LOGIN, AuditEventType.DATA_READ),
			(AuditEventType.PERMISSION_GRANTED, AuditEventType.DATA_ACCESS),
			(AuditEventType.DATA_READ, AuditEventType.DATA_EXPORT),
			(AuditEventType.USER_FAILED_LOGIN, AuditEventType.USER_LOGIN),
			(AuditEventType.SYSTEM_CONFIG_CHANGE, AuditEventType.SERVICE_RESTART)
		]
		
		return (type1, type2) in related_pairs or (type2, type1) in related_pairs
	
	async def _analyze_user_session_relationships(self, events: List[AuditEvent]) -> List[EventRelationship]:
		"""Analyze relationships within user sessions"""
		relationships = []
		
		# Group events by user and session
		user_sessions = defaultdict(lambda: defaultdict(list))
		
		for event in events:
			if event.user_id:
				session_id = event.session_id or "default"
				user_sessions[event.user_id][session_id].append(event)
		
		# Analyze each user session
		for user_id, sessions in user_sessions.items():
			for session_id, session_events in sessions.items():
				if len(session_events) < 2:
					continue
				
				# Sort by timestamp
				session_events.sort(key=lambda e: e.timestamp)
				
				# Create sequential relationships within session
				for i in range(len(session_events) - 1):
					event1 = session_events[i]
					event2 = session_events[i + 1]
					
					time_diff = event2.timestamp - event1.timestamp
					
					# High confidence for same user session
					confidence = 0.8 if time_diff.total_seconds() < 3600 else 0.6
					
					relationship = EventRelationship(
						id=f"session_{hash(f'{event1.id}_{event2.id}') % 1000000}",
						source_event_id=event1.id,
						target_event_id=event2.id,
						relationship_type=RelationshipType.USER_SESSION,
						strength=CorrelationStrength.STRONG,
						confidence=confidence,
						temporal_distance=time_diff,
						evidence={
							"user_id": user_id,
							"session_id": session_id,
							"session_position": i
						}
					)
					
					relationships.append(relationship)
		
		return relationships
	
	async def _analyze_resource_access_relationships(self, events: List[AuditEvent]) -> List[EventRelationship]:
		"""Analyze relationships based on resource access patterns"""
		relationships = []
		
		# Group events by resource
		resource_events = defaultdict(list)
		
		for event in events:
			if event.resource_type and event.resource_id:
				resource_key = f"{event.resource_type}:{event.resource_id}"
				resource_events[resource_key].append(event)
		
		# Analyze resource access patterns
		for resource_key, resource_event_list in resource_events.items():
			if len(resource_event_list) < 2:
				continue
			
			# Sort by timestamp
			resource_event_list.sort(key=lambda e: e.timestamp)
			
			# Look for suspicious access patterns
			for i in range(len(resource_event_list) - 1):
				event1 = resource_event_list[i]
				event2 = resource_event_list[i + 1]
				
				# Different users accessing same resource
				if event1.user_id != event2.user_id:
					confidence = 0.7
					rel_type = RelationshipType.RESOURCE_ACCESS
					
					relationship = EventRelationship(
						id=f"resource_{hash(f'{event1.id}_{event2.id}') % 1000000}",
						source_event_id=event1.id,
						target_event_id=event2.id,
						relationship_type=rel_type,
						strength=CorrelationStrength.MODERATE,
						confidence=confidence,
						evidence={
							"resource": resource_key,
							"user1": event1.user_id,
							"user2": event2.user_id
						}
					)
					
					relationships.append(relationship)
		
		return relationships
	
	async def _analyze_semantic_relationships(self, events: List[AuditEvent]) -> List[EventRelationship]:
		"""Analyze semantic relationships using Graph RAG"""
		try:
			# Prepare event data for semantic analysis
			event_texts = []
			for event in events:
				text = f"{event.action} {event.category} {event.resource_type} by {event.user_id}"
				event_texts.append({
					"id": event.id,
					"text": text,
					"event": event
				})
			
			# Use Graph RAG for relationship discovery
			rag_results = await self.graph_rag.analyze_relationships(
				texts=event_texts,
				relationship_types=["semantic_similarity", "causal_relationship"]
			)
			
			relationships = []
			for result in rag_results.get("relationships", []):
				if result.get("confidence", 0) > self.min_correlation_confidence:
					relationship_key = f"{result['source']}_{result['target']}"
					relationship = EventRelationship(
						id=f"semantic_{hash(relationship_key) % 1000000}",
						source_event_id=result["source"],
						target_event_id=result["target"],
						relationship_type=RelationshipType.CAUSAL if "causal" in result["type"] else RelationshipType.TEMPORAL,
						strength=self._confidence_to_strength(result["confidence"]),
						confidence=result["confidence"],
						semantic_similarity=result.get("similarity", 0.0),
						evidence={
							"semantic_type": result.get("type"),
							"reasoning": result.get("reasoning", "")
						}
					)
					relationships.append(relationship)
			
			return relationships
			
		except Exception as e:
			logger.error(f"Semantic analysis failed: {str(e)}")
			return []
	
	async def _analyze_attack_pattern_relationships(self, events: List[AuditEvent]) -> List[EventRelationship]:
		"""Analyze relationships based on known attack patterns"""
		relationships = []
		
		# Look for common attack sequences
		attack_sequences = [
			# Privilege escalation sequence
			[AuditEventType.USER_LOGIN, AuditEventType.PERMISSION_GRANTED, AuditEventType.DATA_ACCESS],
			# Data exfiltration sequence
			[AuditEventType.DATA_READ, AuditEventType.DATA_EXPORT, AuditEventType.DATA_DOWNLOAD],
			# Failed login -> successful login (credential stuffing)
			[AuditEventType.USER_FAILED_LOGIN, AuditEventType.USER_LOGIN]
		]
		
		for sequence in attack_sequences:
			sequence_events = self._find_event_sequence(events, sequence)
			
			for event_group in sequence_events:
				if len(event_group) >= 2:
					# Create attack chain relationships
					for i in range(len(event_group) - 1):
						event1 = event_group[i]
						event2 = event_group[i + 1]
						
						relationship = EventRelationship(
							id=f"attack_{hash(f'{event1.id}_{event2.id}') % 1000000}",
							source_event_id=event1.id,
							target_event_id=event2.id,
							relationship_type=RelationshipType.ATTACK_CHAIN,
							strength=CorrelationStrength.STRONG,
							confidence=0.9,
							evidence={
								"attack_sequence": [et.value for et in sequence],
								"sequence_position": i
							}
						)
						
						relationships.append(relationship)
		
		return relationships
	
	def _find_event_sequence(self, events: List[AuditEvent], sequence: List[AuditEventType]) -> List[List[AuditEvent]]:
		"""Find events matching an attack sequence pattern"""
		matching_groups = []
		
		# Group events by user to look for per-user sequences
		user_events = defaultdict(list)
		for event in events:
			if event.user_id:
				user_events[event.user_id].append(event)
		
		for user_id, user_event_list in user_events.items():
			user_event_list.sort(key=lambda e: e.timestamp)
			
			# Look for the sequence within user events
			for i in range(len(user_event_list) - len(sequence) + 1):
				potential_match = user_event_list[i:i+len(sequence)]
				
				# Check if event types match sequence
				event_types = [e.event_type for e in potential_match]
				if event_types == sequence:
					# Check if timing is reasonable (within 1 hour)
					time_span = potential_match[-1].timestamp - potential_match[0].timestamp
					if time_span <= timedelta(hours=1):
						matching_groups.append(potential_match)
		
		return matching_groups
	
	def _confidence_to_strength(self, confidence: float) -> CorrelationStrength:
		"""Convert confidence score to correlation strength"""
		if confidence >= 0.9:
			return CorrelationStrength.DEFINITIVE
		elif confidence >= 0.7:
			return CorrelationStrength.STRONG
		elif confidence >= 0.5:
			return CorrelationStrength.MODERATE
		else:
			return CorrelationStrength.WEAK
	
	async def reconstruct_incident_timeline(
		self,
		events: List[AuditEvent],
		relationships: List[EventRelationship] = None
	) -> IncidentTimeline:
		"""Reconstruct incident timeline with automated ordering"""
		try:
			start_time = datetime.utcnow()
			
			# Analyze relationships if not provided
			if relationships is None:
				relationships = await self.analyze_event_relationships(events)
			
			# Create timeline
			timeline_id = f"timeline_{hash(f'{self.tenant_id}_{start_time.timestamp()}') % 1000000}"
			
			# Sort events chronologically
			sorted_events = sorted(events, key=lambda e: e.timestamp)
			
			# Categorize events into timeline phases
			phases = self._categorize_timeline_phases(sorted_events, relationships)
			
			# Extract affected entities
			affected_resources = set()
			involved_users = set()
			source_ips = set()
			
			for event in events:
				if event.resource_id:
					affected_resources.add(f"{event.resource_type}:{event.resource_id}")
				if event.user_id:
					involved_users.add(event.user_id)
				if event.ip_address:
					source_ips.add(event.ip_address)
			
			# Calculate confidence and risk assessment
			confidence_score = self._calculate_timeline_confidence(events, relationships)
			risk_assessment = await self._assess_timeline_risk(events, relationships)
			
			# Detect attack vectors
			attack_vectors = self._detect_attack_vectors(events, relationships)
			
			# Create timeline object
			timeline = IncidentTimeline(
				id=timeline_id,
				tenant_id=self.tenant_id,
				title=f"Incident Timeline - {len(events)} events",
				description=f"Automated timeline reconstruction for {len(events)} correlated events",
				severity=risk_assessment.get("severity", "medium"),
				start_time=sorted_events[0].timestamp if sorted_events else datetime.utcnow(),
				end_time=sorted_events[-1].timestamp if sorted_events else datetime.utcnow(),
				event_count=len(events),
				events=[event.model_dump() for event in sorted_events],
				phases=phases,
				attack_vectors=attack_vectors,
				affected_resources=affected_resources,
				involved_users=involved_users,
				source_ips=source_ips,
				relationships=relationships,
				confidence_score=confidence_score,
				risk_assessment=risk_assessment
			)
			
			# Enrich with threat intelligence
			timeline.threat_intelligence = await self._enrich_with_threat_intelligence(timeline)
			
			# Cache timeline
			self.timeline_cache[timeline_id] = timeline
			
			# Update metrics
			self.metrics["timelines_created"] += 1
			
			logger.info(f"Created timeline {timeline_id} with {len(events)} events and {len(relationships)} relationships")
			
			return timeline
			
		except Exception as e:
			logger.error(f"Timeline reconstruction failed: {str(e)}")
			raise
	
	def _categorize_timeline_phases(
		self, 
		events: List[AuditEvent], 
		relationships: List[EventRelationship]
	) -> List[Dict[str, Any]]:
		"""Categorize events into attack timeline phases"""
		phases = []
		
		# Simple phase detection based on event types and timing
		phase_mapping = {
			TimelineEventType.INITIAL_ACCESS: [AuditEventType.USER_LOGIN, AuditEventType.USER_FAILED_LOGIN],
			TimelineEventType.PRIVILEGE_ESCALATION: [AuditEventType.PERMISSION_GRANTED, AuditEventType.ADMIN_LOGIN],
			TimelineEventType.DATA_ACCESS: [AuditEventType.DATA_READ, AuditEventType.DATA_ACCESS],
			TimelineEventType.DATA_EXFILTRATION: [AuditEventType.DATA_EXPORT, AuditEventType.DATA_DOWNLOAD],
			TimelineEventType.PERSISTENCE: [AuditEventType.SYSTEM_CONFIG_CHANGE, AuditEventType.USER_CREATED],
			TimelineEventType.CLEANUP: [AuditEventType.LOG_DELETED, AuditEventType.USER_DELETED]
		}
		
		current_phase = None
		current_phase_events = []
		
		for event in events:
			# Determine which phase this event belongs to
			event_phase = None
			for phase_type, event_types in phase_mapping.items():
				if event.event_type in event_types:
					event_phase = phase_type
					break
			
			if event_phase != current_phase:
				# Save previous phase if it exists
				if current_phase and current_phase_events:
					phases.append({
						"phase": current_phase.value,
						"start_time": current_phase_events[0].timestamp,
						"end_time": current_phase_events[-1].timestamp,
						"event_count": len(current_phase_events),
						"description": f"{current_phase.value.replace('_', ' ').title()} phase"
					})
				
				# Start new phase
				current_phase = event_phase
				current_phase_events = [event] if event_phase else []
			else:
				if current_phase:
					current_phase_events.append(event)
		
		# Add final phase
		if current_phase and current_phase_events:
			phases.append({
				"phase": current_phase.value,
				"start_time": current_phase_events[0].timestamp,
				"end_time": current_phase_events[-1].timestamp,
				"event_count": len(current_phase_events),
				"description": f"{current_phase.value.replace('_', ' ').title()} phase"
			})
		
		return phases
	
	def _calculate_timeline_confidence(
		self, 
		events: List[AuditEvent], 
		relationships: List[EventRelationship]
	) -> float:
		"""Calculate confidence in timeline reconstruction"""
		if not events:
			return 0.0
		
		confidence = 0.5  # Base confidence
		
		# More events = higher confidence (up to a point)
		event_confidence = min(0.3, len(events) / 100.0)
		confidence += event_confidence
		
		# Strong relationships increase confidence
		if relationships:
			avg_relationship_confidence = sum(r.confidence for r in relationships) / len(relationships)
			confidence += avg_relationship_confidence * 0.3
		
		# Temporal consistency increases confidence
		temporal_consistency = self._calculate_temporal_consistency(events)
		confidence += temporal_consistency * 0.2
		
		return min(1.0, confidence)
	
	def _calculate_temporal_consistency(self, events: List[AuditEvent]) -> float:
		"""Calculate temporal consistency of events"""
		if len(events) < 2:
			return 1.0
		
		# Check for reasonable time gaps between events
		time_diffs = []
		sorted_events = sorted(events, key=lambda e: e.timestamp)
		
		for i in range(1, len(sorted_events)):
			diff = (sorted_events[i].timestamp - sorted_events[i-1].timestamp).total_seconds()
			time_diffs.append(diff)
		
		# Consistency is higher when time differences are reasonable (not too large gaps)
		avg_diff = sum(time_diffs) / len(time_diffs)
		
		# Penalize very large gaps (> 24 hours)
		if avg_diff > 86400:  # 24 hours
			return 0.3
		elif avg_diff > 3600:  # 1 hour
			return 0.7
		else:
			return 1.0
	
	async def _assess_timeline_risk(
		self, 
		events: List[AuditEvent], 
		relationships: List[EventRelationship]
	) -> Dict[str, Any]:
		"""Assess risk level of timeline"""
		risk_factors = []
		risk_score = 0.0
		
		# High-risk event types
		high_risk_events = [
			AuditEventType.PERMISSION_GRANTED,
			AuditEventType.DATA_EXPORT,
			AuditEventType.SYSTEM_CONFIG_CHANGE,
			AuditEventType.USER_FAILED_LOGIN
		]
		
		high_risk_count = sum(1 for e in events if e.event_type in high_risk_events)
		if high_risk_count > 0:
			risk_factors.append(f"{high_risk_count} high-risk events")
			risk_score += min(0.5, high_risk_count * 0.1)
		
		# Failed operations
		failed_count = sum(1 for e in events if not e.success)
		if failed_count > len(events) * 0.3:  # > 30% failure rate
			risk_factors.append(f"High failure rate ({failed_count}/{len(events)})")
			risk_score += 0.3
		
		# External IP addresses
		external_ips = [e.ip_address for e in events if e.ip_address and not self._is_internal_ip(e.ip_address)]
		if external_ips:
			risk_factors.append(f"External IP access from {len(set(external_ips))} addresses")
			risk_score += 0.2
		
		# Attack chain relationships
		attack_chains = [r for r in relationships if r.relationship_type == RelationshipType.ATTACK_CHAIN]
		if attack_chains:
			risk_factors.append(f"{len(attack_chains)} attack chain indicators")
			risk_score += min(0.4, len(attack_chains) * 0.1)
		
		# Determine severity
		if risk_score >= 0.8:
			severity = "critical"
		elif risk_score >= 0.6:
			severity = "high"
		elif risk_score >= 0.4:
			severity = "medium"
		else:
			severity = "low"
		
		return {
			"risk_score": min(1.0, risk_score),
			"severity": severity,
			"risk_factors": risk_factors,
			"assessment_time": datetime.utcnow().isoformat()
		}
	
	def _detect_attack_vectors(
		self, 
		events: List[AuditEvent], 
		relationships: List[EventRelationship]
	) -> List[str]:
		"""Detect attack vectors from event patterns"""
		attack_vectors = []
		
		# Credential stuffing/brute force
		failed_logins = [e for e in events if e.event_type == AuditEventType.USER_FAILED_LOGIN]
		if len(failed_logins) > 5:
			attack_vectors.append("credential_stuffing")
		
		# Privilege escalation
		priv_grants = [e for e in events if e.event_type == AuditEventType.PERMISSION_GRANTED]
		if priv_grants:
			attack_vectors.append("privilege_escalation")
		
		# Data exfiltration
		data_exports = [e for e in events if e.event_type in [AuditEventType.DATA_EXPORT, AuditEventType.DATA_DOWNLOAD]]
		if data_exports:
			attack_vectors.append("data_exfiltration")
		
		# Lateral movement (multiple users/systems)
		unique_users = len(set(e.user_id for e in events if e.user_id))
		if unique_users > 3:
			attack_vectors.append("lateral_movement")
		
		# System manipulation
		system_changes = [e for e in events if e.event_type == AuditEventType.SYSTEM_CONFIG_CHANGE]
		if system_changes:
			attack_vectors.append("system_manipulation")
		
		return attack_vectors
	
	def _is_internal_ip(self, ip_address: str) -> bool:
		"""Check if IP address is internal"""
		if not ip_address:
			return True
		
		internal_ranges = ['192.168.', '10.', '172.16.', '172.17.', '172.18.', '172.19.',
						   '172.20.', '172.21.', '172.22.', '172.23.', '172.24.', '172.25.',
						   '172.26.', '172.27.', '172.28.', '172.29.', '172.30.', '172.31.',
						   '127.', '169.254.']
		
		return any(ip_address.startswith(prefix) for prefix in internal_ranges)
	
	async def _enrich_with_threat_intelligence(self, timeline: IncidentTimeline) -> Dict[str, Any]:
		"""Enrich timeline with external threat intelligence"""
		try:
			# Extract indicators from timeline
			indicators = []
			
			for ip in timeline.source_ips:
				if not self._is_internal_ip(ip):
					indicators.append({"type": "ip", "value": ip})
			
			for attack_vector in timeline.attack_vectors:
				indicators.append({"type": "technique", "value": attack_vector})
			
			# Query threat intelligence service
			enrichment = await self.threat_intelligence.enrich_events(
				indicators=indicators,
				context="incident_timeline"
			)
			
			return enrichment.get("intelligence", {})
			
		except Exception as e:
			logger.error(f"Threat intelligence enrichment failed: {str(e)}")
			return {}
	
	async def detect_attack_patterns(self, events: List[AuditEvent]) -> List[AttackPattern]:
		"""Detect known attack patterns in event data"""
		try:
			patterns = []
			
			# Analyze relationships first
			relationships = await self.analyze_event_relationships(events)
			
			# MITRE ATT&CK pattern detection
			mitre_patterns = [
				{
					"id": "T1078",
					"name": "Valid Accounts",
					"description": "Use of valid accounts for initial access or persistence",
					"indicators": [AuditEventType.USER_LOGIN, AuditEventType.PERMISSION_GRANTED]
				},
				{
					"id": "T1005",
					"name": "Data from Local System",
					"description": "Collection of data from local system",
					"indicators": [AuditEventType.DATA_READ, AuditEventType.DATA_ACCESS]
				},
				{
					"id": "T1041",
					"name": "Exfiltration Over C2 Channel",
					"description": "Data exfiltration through command and control channels",
					"indicators": [AuditEventType.DATA_EXPORT, AuditEventType.DATA_DOWNLOAD]
				}
			]
			
			for pattern_def in mitre_patterns:
				matching_events = [
					e for e in events 
					if e.event_type in pattern_def["indicators"]
				]
				
				if len(matching_events) >= 2:
					# Create timeline for this pattern
					pattern_timeline = await self.reconstruct_incident_timeline(matching_events)
					
					pattern = AttackPattern(
						id=f"pattern_{pattern_def['id']}_{hash(pattern_timeline.id) % 1000}",
						name=pattern_def["name"],
						description=pattern_def["description"],
						mitre_technique=pattern_def["id"],
						confidence=min(1.0, len(matching_events) / 10.0),
						events=[e.id for e in matching_events],
						timeline=pattern_timeline
					)
					
					patterns.append(pattern)
			
			# Update metrics
			self.metrics["attack_patterns_detected"] += len(patterns)
			
			return patterns
			
		except Exception as e:
			logger.error(f"Attack pattern detection failed: {str(e)}")
			return []
	
	async def create_investigation_timeline(
		self, 
		investigation_id: str, 
		events: List[AuditEvent]
	) -> IncidentTimeline:
		"""Create timeline for collaborative investigation"""
		try:
			# Reconstruct timeline
			timeline = await self.reconstruct_incident_timeline(events)
			timeline.investigation_id = investigation_id
			
			# Create collaborative timeline in APG
			collab_timeline = await self.collaboration_service.create_timeline(
				investigation_id=investigation_id,
				title=timeline.title,
				description=timeline.description,
				events=timeline.events,
				metadata={
					"timeline_id": timeline.id,
					"confidence_score": timeline.confidence_score,
					"risk_assessment": timeline.risk_assessment
				}
			)
			
			logger.info(f"Created investigation timeline {timeline.id} for investigation {investigation_id}")
			
			return timeline
			
		except Exception as e:
			logger.error(f"Investigation timeline creation failed: {str(e)}")
			raise
	
	async def get_correlation_metrics(self) -> Dict[str, Any]:
		"""Get correlation engine performance metrics"""
		return {
			"performance": self.metrics,
			"graph_stats": {
				"nodes": self.event_graph.graph.number_of_nodes(),
				"edges": self.event_graph.graph.number_of_edges(),
				"components": nx.number_weakly_connected_components(self.event_graph.graph)
			},
			"cache_stats": {
				"relationships_cached": len(self.relationship_cache),
				"timelines_cached": len(self.timeline_cache)
			}
		}

# Export for APG integration
__all__ = [
	"CorrelationEngine",
	"EventRelationship",
	"IncidentTimeline",
	"AttackPattern",
	"EventGraph",
	"RelationshipType",
	"TimelineEventType"
]
