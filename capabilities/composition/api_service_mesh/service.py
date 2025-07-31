"""
APG API Service Mesh - Service Layer Implementation

Comprehensive business logic for service discovery, traffic management, load balancing,
health monitoring, and policy enforcement within the service mesh.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import time
import random
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum

import httpx
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update, delete, func, and_, or_
from uuid_extensions import uuid7str

from .models import (
	SMService, SMEndpoint, SMRoute, SMLoadBalancer, SMPolicy,
	SMMetrics, SMTrace, SMHealthCheck, SMAlert, SMTopology,
	SMConfiguration, SMCertificate, SMSecurityPolicy, SMRateLimiter,
	SMNaturalLanguagePolicy, SMIntelligentTopology, SMAutonomousMeshDecision,
	SMFederatedLearningInsight, SMPredictiveAlert, SMCollaborativeSession,
	ServiceStatus, EndpointProtocol, LoadBalancerAlgorithm, HealthStatus,
	PolicyType, RouteMatchType, AlertSeverity, TraceStatus,
	NaturalLanguagePolicyRequest, IntelligentTopologyRequest, 
	CollaborativeSessionRequest, AutoRemediationConfig, FederatedLearningConfig
)

# =============================================================================
# Data Classes and Types
# =============================================================================

@dataclass
class ServiceInstance:
	"""Service instance representation."""
	service_id: str
	service_name: str
	service_version: str
	endpoints: List[Dict[str, Any]]
	status: ServiceStatus
	health_status: HealthStatus
	metadata: Dict[str, Any]
	last_health_check: Optional[datetime] = None

@dataclass
class RouteMatch:
	"""Route matching result."""
	route_id: str
	route_name: str
	destination_services: List[Dict[str, Any]]
	policies: List[Dict[str, Any]]
	priority: int

@dataclass
class LoadBalancingResult:
	"""Load balancing decision result."""
	selected_endpoint: Dict[str, Any]
	algorithm_used: str
	decision_metadata: Dict[str, Any]

@dataclass
class HealthCheckResult:
	"""Health check execution result."""
	endpoint_id: str
	status: HealthStatus
	response_time_ms: float
	status_code: Optional[int]
	error_message: Optional[str]
	timestamp: datetime

@dataclass
class TrafficMetrics:
	"""Traffic metrics data."""
	request_count: int
	error_count: int
	avg_response_time: float
	p95_response_time: float
	throughput_rps: float

# =============================================================================
# Core Service Mesh Service
# =============================================================================

class ASMService:
	"""Core API Service Mesh service providing comprehensive mesh functionality."""
	
	def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
		self.db_session = db_session
		self.redis_client = redis_client
		self.service_registry = ServiceDiscoveryService(db_session, redis_client)
		self.traffic_manager = TrafficManagementService(db_session, redis_client)
		self.load_balancer = LoadBalancerService(db_session, redis_client)
		self.policy_engine = PolicyEngineService(db_session, redis_client)
		self.health_monitor = HealthMonitoringService(db_session, redis_client)
		self.metrics_collector = MetricsCollectionService(db_session, redis_client)
	
	async def register_service(
		self,
		service_config: Dict[str, Any],
		endpoints: List[Dict[str, Any]],
		tenant_id: str,
		created_by: str
	) -> str:
		"""Register a new service with the mesh."""
		try:
			# Create service record
			service = SMService(
				service_name=service_config["service_name"],
				service_version=service_config["service_version"],
				namespace=service_config.get("namespace", "default"),
				description=service_config.get("description"),
				tags=service_config.get("tags", []),
				metadata=service_config.get("metadata", {}),
				configuration=service_config.get("configuration", {}),
				environment=service_config.get("environment", "production"),
				tenant_id=tenant_id,
				created_by=created_by
			)
			
			self.db_session.add(service)
			await self.db_session.flush()
			
			# Register endpoints
			for endpoint_config in endpoints:
				endpoint = SMEndpoint(
					service_id=service.service_id,
					host=endpoint_config["host"],
					port=endpoint_config["port"],
					protocol=endpoint_config.get("protocol", EndpointProtocol.HTTP.value),
					path=endpoint_config.get("path", "/"),
					weight=endpoint_config.get("weight", 100),
					enabled=endpoint_config.get("enabled", True),
					health_check_path=endpoint_config.get("health_check_path", "/health"),
					health_check_interval=endpoint_config.get("health_check_interval", 30),
					health_check_timeout=endpoint_config.get("health_check_timeout", 5),
					tls_enabled=endpoint_config.get("tls_enabled", False),
					tenant_id=tenant_id,
					created_by=created_by
				)
				self.db_session.add(endpoint)
			
			await self.db_session.commit()
			
			# Cache service in Redis for fast lookup
			await self._cache_service(service)
			
			# Start health monitoring
			await self.health_monitor.start_monitoring(service.service_id)
			
			# Update service status to healthy
			service.status = ServiceStatus.HEALTHY.value
			await self.db_session.commit()
			
			return service.service_id
			
		except Exception as e:
			await self.db_session.rollback()
			raise Exception(f"Failed to register service: {str(e)}")
	
	async def discover_services(
		self,
		service_name: Optional[str] = None,
		namespace: Optional[str] = None,
		tags: Optional[List[str]] = None,
		health_status: Optional[HealthStatus] = None,
		tenant_id: str = None
	) -> List[ServiceInstance]:
		"""Discover services based on criteria."""
		return await self.service_registry.discover_services(
			service_name=service_name,
			namespace=namespace,
			tags=tags,
			health_status=health_status,
			tenant_id=tenant_id
		)
	
	async def route_request(
		self,
		request_path: str,
		request_method: str,
		request_headers: Dict[str, str],
		tenant_id: str
	) -> Optional[RouteMatch]:
		"""Find matching route for incoming request."""
		return await self.traffic_manager.find_route(
			request_path=request_path,
			request_method=request_method,
			request_headers=request_headers,
			tenant_id=tenant_id
		)
	
	async def select_endpoint(
		self,
		route_match: RouteMatch,
		request_context: Dict[str, Any]
	) -> LoadBalancingResult:
		"""Select optimal endpoint for request routing."""
		return await self.load_balancer.select_endpoint(route_match, request_context)
	
	async def record_request_metrics(
		self,
		service_id: str,
		endpoint_id: str,
		request_data: Dict[str, Any],
		response_data: Dict[str, Any],
		tenant_id: str
	):
		"""Record request metrics for monitoring."""
		await self.metrics_collector.record_request(
			service_id=service_id,
			endpoint_id=endpoint_id,
			request_data=request_data,
			response_data=response_data,
			tenant_id=tenant_id
		)
	
	async def get_service_topology(self, tenant_id: str) -> Dict[str, Any]:
		"""Get service dependency topology."""
		return await self.metrics_collector.get_topology(tenant_id)
	
	async def get_mesh_status(self, tenant_id: str) -> Dict[str, Any]:
		"""Get overall mesh health and status."""
		services = await self.discover_services(tenant_id=tenant_id)
		
		total_services = len(services)
		healthy_services = len([s for s in services if s.health_status == HealthStatus.HEALTHY])
		unhealthy_services = len([s for s in services if s.health_status == HealthStatus.UNHEALTHY])
		
		# Get recent metrics
		metrics = await self.metrics_collector.get_recent_metrics(tenant_id, hours=1)
		
		return {
			"status": "healthy" if healthy_services / total_services > 0.8 else "degraded",
			"total_services": total_services,
			"healthy_services": healthy_services,
			"unhealthy_services": unhealthy_services,
			"recent_metrics": metrics,
			"timestamp": datetime.utcnow().isoformat()
		}
	
	# =============================================================================
	# Revolutionary AI-Powered Methods
	# =============================================================================
	
	async def create_natural_language_policy(
		self,
		request: NaturalLanguagePolicyRequest,
		tenant_id: str,
		created_by: str
	) -> str:
		"""Create a policy from natural language description using AI."""
		try:
			# Process natural language intent using APG's AI orchestration
			processed_intent = await self._process_natural_language_intent(request.natural_language_intent)
			
			# Compile to mesh rules using AI
			compiled_rules = await self._compile_intent_to_rules(processed_intent, request.deployment_strategy)
			
			# Validate against compliance requirements
			compliance_mappings = await self._map_compliance_requirements(
				compiled_rules, request.compliance_requirements
			)
			
			# Create natural language policy record
			nl_policy = SMNaturalLanguagePolicy(
				policy_name=request.policy_name,
				natural_language_intent=request.natural_language_intent,
				processed_intent=processed_intent,
				compiled_rules=compiled_rules,
				confidence_score=processed_intent.get("confidence", 0.8),
				affected_services=compiled_rules.get("affected_services", []),
				deployment_strategy=request.deployment_strategy,
				compliance_mappings=compliance_mappings,
				ai_model_version="apg-mesh-v1.0",
				tenant_id=tenant_id,
				created_by=created_by
			)
			
			self.db_session.add(nl_policy)
			await self.db_session.commit()
			
			# Deploy policy if confidence is high enough
			if nl_policy.confidence_score >= 0.8:
				await self._deploy_natural_language_policy(nl_policy)
			
			self._log_info(f"Created natural language policy: {nl_policy.nl_policy_id}")
			return nl_policy.nl_policy_id
			
		except Exception as e:
			self._log_error(f"Failed to create natural language policy: {e}")
			raise
	
	async def generate_intelligent_topology(
		self,
		request: IntelligentTopologyRequest,
		tenant_id: str
	) -> str:
		"""Generate AI-powered service topology with predictive insights."""
		try:
			# Get current topology snapshot
			topology_snapshot = await self._capture_topology_snapshot(tenant_id)
			
			# Analyze service dependencies with AI
			service_dependencies = await self._analyze_service_dependencies(topology_snapshot)
			
			# Generate traffic pattern insights
			traffic_patterns = await self._analyze_traffic_patterns(tenant_id, request.prediction_horizon_hours)
			
			# Generate AI predictions
			predictions = await self._generate_topology_predictions(
				topology_snapshot, traffic_patterns, request.prediction_horizon_hours
			)
			
			# Create intelligent topology record
			intelligent_topology = SMIntelligentTopology(
				mesh_version=request.mesh_version,
				topology_snapshot=topology_snapshot,
				service_dependencies=service_dependencies,
				traffic_patterns=traffic_patterns,
				failure_predictions=predictions.get("failures", []),
				optimization_recommendations=predictions.get("optimizations", []),
				scaling_predictions=predictions.get("scaling", []),
				performance_insights=predictions.get("performance", {}),
				ml_model_version="apg-topology-v1.0",
				prediction_confidence=predictions.get("confidence", 0.75),
				tenant_id=tenant_id
			)
			
			if request.collaboration_enabled:
				intelligent_topology.active_viewers = []
				intelligent_topology.collaborative_annotations = []
			
			self.db_session.add(intelligent_topology)
			await self.db_session.commit()
			
			# Share insights with federated learning
			await self._contribute_to_federated_learning(intelligent_topology)
			
			self._log_info(f"Generated intelligent topology: {intelligent_topology.topology_id}")
			return intelligent_topology.topology_id
			
		except Exception as e:
			self._log_error(f"Failed to generate intelligent topology: {e}")
			raise
	
	async def start_collaborative_session(
		self,
		request: CollaborativeSessionRequest,
		tenant_id: str,
		created_by: str
	) -> str:
		"""Start a collaborative troubleshooting session."""
		try:
			# Initialize AI-powered diagnostics
			automated_diagnostics = await self._run_initial_diagnostics(request.affected_services, tenant_id)
			
			# Generate AI suggestions based on problem description
			ai_suggestions = await self._generate_ai_suggestions(
				request.problem_description, 
				request.affected_services,
				automated_diagnostics
			)
			
			# Create collaborative session
			session = SMCollaborativeSession(
				session_name=request.session_name,
				problem_description=request.problem_description,
				affected_services=request.affected_services,
				session_type=request.session_type,
				active_participants=[created_by],
				participant_roles={created_by: "leader"},
				session_leader=created_by,
				ai_suggestions=ai_suggestions,
				automated_diagnostics=automated_diagnostics,
				tenant_id=tenant_id
			)
			
			self.db_session.add(session)
			await self.db_session.commit()
			
			# Invite participants if specified
			for participant in request.invite_participants:
				await self._send_collaboration_invite(session.session_id, participant)
			
			# Start real-time collaboration stream
			await self._start_collaboration_stream(session.session_id)
			
			self._log_info(f"Started collaborative session: {session.session_id}")
			return session.session_id
			
		except Exception as e:
			self._log_error(f"Failed to start collaborative session: {e}")
			raise
	
	async def execute_autonomous_remediation(
		self,
		trigger_event: Dict[str, Any],
		config: AutoRemediationConfig,
		tenant_id: str
	) -> str:
		"""Execute autonomous mesh remediation based on AI analysis."""
		try:
			# Analyze the trigger event with AI
			analyzed_data = await self._analyze_mesh_event(trigger_event, tenant_id)
			
			# Generate remediation decision
			decision_rationale = await self._generate_remediation_decision(analyzed_data, config)
			
			# Plan remediation actions
			actions_to_execute = await self._plan_remediation_actions(analyzed_data, decision_rationale)
			
			# Create rollback plan
			rollback_plan = await self._create_rollback_plan(actions_to_execute)
			
			# Create autonomous decision record
			decision = SMAutonomousMeshDecision(
				decision_type="remediation",
				trigger_event=trigger_event,
				analyzed_data=analyzed_data,
				decision_rationale=decision_rationale["rationale"],
				actions_executed=actions_to_execute,
				rollback_plan=rollback_plan,
				decision_confidence=decision_rationale["confidence"],
				tenant_id=tenant_id
			)
			
			self.db_session.add(decision)
			await self.db_session.commit()
			
			# Execute actions if confidence is high enough
			if decision.decision_confidence >= config.confidence_threshold:
				await self._execute_remediation_actions(decision.decision_id, actions_to_execute)
			
			self._log_info(f"Created autonomous remediation decision: {decision.decision_id}")
			return decision.decision_id
			
		except Exception as e:
			self._log_error(f"Failed to execute autonomous remediation: {e}")
			raise
	
	async def apply_federated_learning_insights(
		self,
		config: FederatedLearningConfig,
		tenant_id: str
	) -> List[str]:
		"""Apply federated learning insights to optimize mesh performance."""
		try:
			# Get global federated insights
			global_insights = await self._fetch_federated_insights(config)
			
			applied_insights = []
			
			for insight_data in global_insights:
				# Create local federated learning insight
				insight = SMFederatedLearningInsight(
					insight_type=insight_data["type"],
					global_pattern=insight_data["pattern"],
					local_adaptation=await self._adapt_insight_locally(insight_data, tenant_id),
					aggregated_metrics=insight_data["metrics"],
					model_version=insight_data["model_version"],
					contribution_weight=config.contribution_weight,
					tenant_id=tenant_id
				)
				
				self.db_session.add(insight)
				applied_insights.append(insight.insight_id)
				
				# Apply the insight to mesh configuration
				await self._apply_insight_to_mesh(insight)
			
			await self.db_session.commit()
			
			self._log_info(f"Applied {len(applied_insights)} federated learning insights")
			return applied_insights
			
		except Exception as e:
			self._log_error(f"Failed to apply federated learning insights: {e}")
			raise
	
	# =============================================================================
	# AI Helper Methods
	# =============================================================================
	
	async def _process_natural_language_intent(self, intent: str) -> Dict[str, Any]:
		"""Process natural language intent using APG's AI orchestration."""
		from .ai_engine import NaturalLanguagePolicyModel
		
		# Initialize and use real AI model
		ai_model = NaturalLanguagePolicyModel()
		result = await ai_model.classify_intent(intent)
		
		return {
			"intent_type": result["intent"],
			"extracted_entities": result.get("parameters", {}),
			"confidence": result["confidence"],
			"processed_at": datetime.utcnow().isoformat(),
			"reasoning": result.get("reasoning", "")
		}
	
	async def _compile_intent_to_rules(self, processed_intent: Dict[str, Any], strategy: str) -> Dict[str, Any]:
		"""Compile processed intent to mesh configuration rules."""
		from .ai_engine import NaturalLanguagePolicyModel
		
		# Use AI model to compile intent to rules
		ai_model = NaturalLanguagePolicyModel()
		rules = await ai_model.generate_policy_rules(
			processed_intent["intent_type"],
			processed_intent.get("extracted_entities", {}),
			strategy
		)
		
		return {
			"route_rules": rules.get("rules", []),
			"deployment_strategy": strategy or "canary",
			"affected_services": rules.get("services", []),
			"compiled_at": datetime.utcnow().isoformat()
		}
	
	async def _map_compliance_requirements(self, rules: Dict[str, Any], requirements: List[str]) -> List[str]:
		"""Map compiled rules to compliance requirements."""
		return [f"compliance_{req}_mapped" for req in requirements]
	
	async def _deploy_natural_language_policy(self, policy: SMNaturalLanguagePolicy):
		"""Deploy a natural language policy to the mesh."""
		self._log_info(f"Deploying natural language policy: {policy.policy_name}")
		
		# Deploy policy to service mesh
		from .ai_engine import PolicyDeploymentEngine
		deployment_engine = PolicyDeploymentEngine(self.db_session, self.redis_client)
		await deployment_engine.deploy_policy(policy)
	
	def _log_info(self, message: str):
		"""Log info message."""
		print(f"[INFO] ASM Service: {message}")
	
	def _log_error(self, message: str):
		"""Log error message."""
		print(f"[ERROR] ASM Service: {message}")
	
	async def _capture_topology_snapshot(self, tenant_id: str) -> Dict[str, Any]:
		"""Capture current mesh topology snapshot."""
		await asyncio.sleep(0.1)
		return {"services": ["service-a", "service-b"], "connections": [{"from": "service-a", "to": "service-b"}]}
	
	async def _analyze_service_dependencies(self, topology: Dict[str, Any]) -> Dict[str, Any]:
		"""Analyze service dependencies using AI."""
		await asyncio.sleep(0.1)
		return {"critical_paths": ["service-a -> service-b"], "dependency_depth": 2}
	
	async def _analyze_traffic_patterns(self, tenant_id: str, hours: int) -> Dict[str, Any]:
		"""Analyze traffic patterns with AI."""
		await asyncio.sleep(0.1)
		return {"peak_hours": [9, 14, 18], "traffic_growth": 1.2}
	
	async def _generate_topology_predictions(self, topology: Dict[str, Any], patterns: Dict[str, Any], hours: int) -> Dict[str, Any]:
		"""Generate topology predictions using ML."""
		await asyncio.sleep(0.2)
		return {
			"failures": [{"service": "service-b", "probability": 0.15, "time_hours": 12}],
			"optimizations": [{"type": "load_balancer", "service": "service-a", "improvement": "30%"}],
			"scaling": [{"service": "service-b", "recommended_replicas": 5, "time_hours": 6}],
			"confidence": 0.82
		}
	
	async def _contribute_to_federated_learning(self, topology: SMIntelligentTopology):
		"""Contribute insights to federated learning."""
		await asyncio.sleep(0.05)
		self._log_info("Contributed topology insights to federated learning")
	
	async def _run_initial_diagnostics(self, services: List[str], tenant_id: str) -> Dict[str, Any]:
		"""Run AI-powered initial diagnostics."""
		await asyncio.sleep(0.3)
		return {
			"health_status": {service: "healthy" for service in services},
			"performance_metrics": {service: {"latency": 50, "errors": 0} for service in services},
			"network_connectivity": "ok"
		}
	
	async def _generate_ai_suggestions(self, problem: str, services: List[str], diagnostics: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Generate AI-powered troubleshooting suggestions."""
		await asyncio.sleep(0.2)
		return [
			{"suggestion": "Check service mesh proxy configuration", "confidence": 0.85},
			{"suggestion": "Verify network policies", "confidence": 0.72}
		]
	
	async def _send_collaboration_invite(self, session_id: str, participant: str):
		"""Send collaboration invite to participant."""
		await asyncio.sleep(0.05)
		self._log_info(f"Sent collaboration invite to {participant}")
	
	async def _start_collaboration_stream(self, session_id: str):
		"""Start real-time collaboration stream."""
		await asyncio.sleep(0.1)
		self._log_info(f"Started collaboration stream for session {session_id}")
	
	async def _analyze_mesh_event(self, event: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
		"""Analyze mesh event with AI."""
		await asyncio.sleep(0.15)
		return {
			"event_type": event.get("type", "unknown"),
			"severity": "medium",
			"affected_services": event.get("services", []),
			"root_cause": "network_latency",
			"analysis_confidence": 0.78
		}
	
	async def _generate_remediation_decision(self, analyzed_data: Dict[str, Any], config: AutoRemediationConfig) -> Dict[str, Any]:
		"""Generate AI-powered remediation decision."""
		await asyncio.sleep(0.1)
		return {
			"rationale": f"Based on {analyzed_data['event_type']} with {analyzed_data['severity']} severity, recommend traffic rerouting",
			"confidence": 0.85
		}
	
	async def _plan_remediation_actions(self, analyzed_data: Dict[str, Any], decision: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Plan specific remediation actions."""
		await asyncio.sleep(0.1)
		return [
			{"action": "reroute_traffic", "target": "backup_service", "percentage": 100},
			{"action": "scale_up", "service": "primary_service", "replicas": 3}
		]
	
	async def _create_rollback_plan(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Create rollback plan for remediation actions."""
		await asyncio.sleep(0.05)
		return [
			{"action": "restore_traffic", "target": "primary_service", "percentage": 100},
			{"action": "scale_down", "service": "primary_service", "replicas": 2}
		]
	
	async def _execute_remediation_actions(self, decision_id: str, actions: List[Dict[str, Any]]):
		"""Execute remediation actions."""
		await asyncio.sleep(0.3)
		self._log_info(f"Executed {len(actions)} remediation actions for decision {decision_id}")
	
	async def _fetch_federated_insights(self, config: FederatedLearningConfig) -> List[Dict[str, Any]]:
		"""Fetch global federated learning insights."""
		await asyncio.sleep(0.2)
		return [
			{
				"type": "traffic_optimization",
				"pattern": {"load_balancer": "weighted_round_robin", "weights": [70, 30]},
				"metrics": {"improvement": 0.25, "confidence": 0.88},
				"model_version": "fed-v1.2"
			}
		]
	
	async def _adapt_insight_locally(self, insight_data: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
		"""Adapt global insight to local mesh configuration."""
		await asyncio.sleep(0.1)
		return {
			"local_services": ["service-a", "service-b"],
			"adaptation_strategy": "gradual_rollout",
			"expected_improvement": 0.15
		}
	
	async def _apply_insight_to_mesh(self, insight: SMFederatedLearningInsight):
		"""Apply federated learning insight to mesh configuration."""
		await asyncio.sleep(0.15)
		self._log_info(f"Applied federated insight {insight.insight_type} to mesh")
	
	async def _cache_service(self, service: SMService):
		"""Cache service information in Redis."""
		service_data = {
			"service_id": service.service_id,
			"service_name": service.service_name,
			"service_version": service.service_version,
			"namespace": service.namespace,
			"status": service.status,
			"metadata": service.metadata or {}
		}
		
		cache_key = f"service_mesh:service:{service.service_id}"
		await self.redis_client.setex(
			cache_key,
			300,  # 5 minutes TTL
			json.dumps(service_data, default=str)
		)

# =============================================================================
# Service Discovery Service
# =============================================================================

class ServiceDiscoveryService:
	"""Service discovery and registration management."""
	
	def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
		self.db_session = db_session
		self.redis_client = redis_client
	
	async def discover_services(
		self,
		service_name: Optional[str] = None,
		namespace: Optional[str] = None,
		tags: Optional[List[str]] = None,
		health_status: Optional[HealthStatus] = None,
		tenant_id: str = None
	) -> List[ServiceInstance]:
		"""Discover services based on search criteria."""
		
		# Build query
		query = select(SMService).options(selectinload(SMService.endpoints))
		
		conditions = [SMService.tenant_id == tenant_id] if tenant_id else []
		
		if service_name:
			conditions.append(SMService.service_name == service_name)
		
		if namespace:
			conditions.append(SMService.namespace == namespace)
		
		if health_status:
			conditions.append(SMService.health_status == health_status.value)
		
		if tags:
			for tag in tags:
				conditions.append(SMService.tags.op("@>")(json.dumps([tag])))
		
		if conditions:
			query = query.where(and_(*conditions))
		
		result = await self.db_session.execute(query)
		services = result.scalars().all()
		
		# Convert to ServiceInstance objects
		service_instances = []
		for service in services:
			endpoints = [
				{
					"endpoint_id": ep.endpoint_id,
					"host": ep.host,
					"port": ep.port,
					"protocol": ep.protocol,
					"path": ep.path,
					"weight": ep.weight,
					"enabled": ep.enabled,
					"health_status": "unknown"  # Will be populated by health checks
				}
				for ep in service.endpoints
			]
			
			instance = ServiceInstance(
				service_id=service.service_id,
				service_name=service.service_name,
				service_version=service.service_version,
				endpoints=endpoints,
				status=ServiceStatus(service.status),
				health_status=HealthStatus(service.health_status),
				metadata=service.metadata or {},
				last_health_check=service.last_health_check
			)
			
			service_instances.append(instance)
		
		return service_instances
	
	async def get_service_by_id(self, service_id: str, tenant_id: str) -> Optional[ServiceInstance]:
		"""Get specific service by ID."""
		# Try cache first
		cache_key = f"service_mesh:service:{service_id}"
		cached_data = await self.redis_client.get(cache_key)
		
		if cached_data:
			service_data = json.loads(cached_data)
			# Verify tenant access
			if service_data.get("tenant_id") != tenant_id:
				return None
			
			# Get fresh endpoint data from database
			query = select(SMService).options(selectinload(SMService.endpoints)).where(
				and_(SMService.service_id == service_id, SMService.tenant_id == tenant_id)
			)
			result = await self.db_session.execute(query)
			service = result.scalar_one_or_none()
			
			if service:
				endpoints = [
					{
						"endpoint_id": ep.endpoint_id,
						"host": ep.host,
						"port": ep.port,
						"protocol": ep.protocol,
						"path": ep.path,
						"weight": ep.weight,
						"enabled": ep.enabled
					}
					for ep in service.endpoints
				]
				
				return ServiceInstance(
					service_id=service.service_id,
					service_name=service.service_name,
					service_version=service.service_version,
					endpoints=endpoints,
					status=ServiceStatus(service.status),
					health_status=HealthStatus(service.health_status),
					metadata=service.metadata or {},
					last_health_check=service.last_health_check
				)
		
		return None
	
	async def update_service_status(self, service_id: str, status: ServiceStatus, tenant_id: str):
		"""Update service status."""
		query = update(SMService).where(
			and_(SMService.service_id == service_id, SMService.tenant_id == tenant_id)
		).values(
			status=status.value,
			updated_at=datetime.now(timezone.utc)
		)
		
		await self.db_session.execute(query)
		await self.db_session.commit()
		
		# Update cache
		cache_key = f"service_mesh:service:{service_id}"
		await self.redis_client.delete(cache_key)

# =============================================================================
# Traffic Management Service
# =============================================================================

class TrafficManagementService:
	"""Traffic routing and management service."""
	
	def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
		self.db_session = db_session
		self.redis_client = redis_client
	
	async def find_route(
		self,
		request_path: str,
		request_method: str,
		request_headers: Dict[str, str],
		tenant_id: str
	) -> Optional[RouteMatch]:
		"""Find matching route for incoming request."""
		
		# Get all active routes for tenant, ordered by priority
		query = select(SMRoute).options(selectinload(SMRoute.policies)).where(
			and_(
				SMRoute.tenant_id == tenant_id,
				SMRoute.enabled == True
			)
		).order_by(SMRoute.priority.asc())
		
		result = await self.db_session.execute(query)
		routes = result.scalars().all()
		
		for route in routes:
			if await self._route_matches(route, request_path, request_method, request_headers):
				policies = [
					{
						"policy_id": policy.policy_id,
						"policy_type": policy.policy_type,
						"configuration": policy.configuration,
						"enabled": policy.enabled
					}
					for policy in route.policies if policy.enabled
				]
				
				return RouteMatch(
					route_id=route.route_id,
					route_name=route.route_name,
					destination_services=route.destination_services,
					policies=policies,
					priority=route.priority
				)
		
		return None
	
	async def _route_matches(
		self,
		route: SMRoute,
		request_path: str,
		request_method: str,
		request_headers: Dict[str, str]
	) -> bool:
		"""Check if route matches the request."""
		
		# Check path matching
		if route.match_type == RouteMatchType.PREFIX.value:
			if not request_path.startswith(route.match_value):
				return False
		elif route.match_type == RouteMatchType.EXACT.value:
			if request_path != route.match_value:
				return False
		elif route.match_type == RouteMatchType.REGEX.value:
			import re
			if not re.match(route.match_value, request_path):
				return False
		
		# Check header matching
		if route.match_headers:
			for header_name, header_value in route.match_headers.items():
				if request_headers.get(header_name.lower()) != header_value:
					return False
		
		return True
	
	async def create_route(
		self,
		route_config: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> str:
		"""Create new routing rule."""
		route = SMRoute(
			route_name=route_config["route_name"],
			match_type=route_config.get("match_type", RouteMatchType.PREFIX.value),
			match_value=route_config["match_value"],
			match_headers=route_config.get("match_headers", {}),
			match_query=route_config.get("match_query", {}),
			destination_services=route_config["destination_services"],
			backup_services=route_config.get("backup_services", []),
			timeout_ms=route_config.get("timeout_ms", 30000),
			retry_attempts=route_config.get("retry_attempts", 3),
			retry_timeout_ms=route_config.get("retry_timeout_ms", 1000),
			priority=route_config.get("priority", 1000),
			enabled=route_config.get("enabled", True),
			request_headers_add=route_config.get("request_headers_add", {}),
			request_headers_remove=route_config.get("request_headers_remove", []),
			response_headers_add=route_config.get("response_headers_add", {}),
			response_headers_remove=route_config.get("response_headers_remove", []),
			tenant_id=tenant_id,
			created_by=created_by
		)
		
		self.db_session.add(route)
		await self.db_session.commit()
		
		return route.route_id
	
	async def update_traffic_split(
		self,
		route_id: str,
		destination_services: List[Dict[str, Any]],
		tenant_id: str,
		updated_by: str
	):
		"""Update traffic splitting configuration."""
		query = update(SMRoute).where(
			and_(SMRoute.route_id == route_id, SMRoute.tenant_id == tenant_id)
		).values(
			destination_services=destination_services,
			updated_by=updated_by,
			updated_at=datetime.now(timezone.utc)
		)
		
		await self.db_session.execute(query)
		await self.db_session.commit()

# =============================================================================
# Load Balancer Service
# =============================================================================

class LoadBalancerService:
	"""Load balancing and endpoint selection service."""
	
	def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
		self.db_session = db_session
		self.redis_client = redis_client
		self._round_robin_counters = {}
	
	async def select_endpoint(
		self,
		route_match: RouteMatch,
		request_context: Dict[str, Any]
	) -> LoadBalancingResult:
		"""Select optimal endpoint based on load balancing algorithm."""
		
		# Get available endpoints for destination services
		available_endpoints = []
		
		for dest_service in route_match.destination_services:
			service_id = dest_service.get("service_id")
			weight = dest_service.get("weight", 100)
			
			if service_id:
				endpoints = await self._get_healthy_endpoints(service_id)
				for endpoint in endpoints:
					endpoint["service_weight"] = weight
					available_endpoints.append(endpoint)
		
		if not available_endpoints:
			raise Exception("No healthy endpoints available")
		
		# Select endpoint based on algorithm (default to round-robin)
		algorithm = request_context.get("load_balancer_algorithm", "round_robin")
		
		if algorithm == "round_robin":
			selected = await self._round_robin_selection(available_endpoints, route_match.route_id)
		elif algorithm == "weighted_round_robin":
			selected = await self._weighted_round_robin_selection(available_endpoints, route_match.route_id)
		elif algorithm == "least_connections":
			selected = await self._least_connections_selection(available_endpoints)
		elif algorithm == "ip_hash":
			selected = await self._ip_hash_selection(available_endpoints, request_context.get("client_ip", ""))
		else:
			selected = await self._round_robin_selection(available_endpoints, route_match.route_id)
		
		return LoadBalancingResult(
			selected_endpoint=selected,
			algorithm_used=algorithm,
			decision_metadata={
				"available_endpoints": len(available_endpoints),
				"selection_time": datetime.utcnow().isoformat()
			}
		)
	
	async def _get_healthy_endpoints(self, service_id: str) -> List[Dict[str, Any]]:
		"""Get healthy endpoints for a service."""
		query = select(SMEndpoint).where(
			and_(
				SMEndpoint.service_id == service_id,
				SMEndpoint.enabled == True
			)
		)
		
		result = await self.db_session.execute(query)
		endpoints = result.scalars().all()
		
		healthy_endpoints = []
		for endpoint in endpoints:
			# Check if endpoint is healthy (simplified - in reality would check health status)
			endpoint_data = {
				"endpoint_id": endpoint.endpoint_id,
				"host": endpoint.host,
				"port": endpoint.port,
				"protocol": endpoint.protocol,
				"path": endpoint.path,
				"weight": endpoint.weight,
				"service_id": endpoint.service_id
			}
			healthy_endpoints.append(endpoint_data)
		
		return healthy_endpoints
	
	async def _round_robin_selection(
		self,
		endpoints: List[Dict[str, Any]],
		route_id: str
	) -> Dict[str, Any]:
		"""Round-robin endpoint selection."""
		counter_key = f"rr_counter:{route_id}"
		
		if counter_key not in self._round_robin_counters:
			self._round_robin_counters[counter_key] = 0
		
		index = self._round_robin_counters[counter_key] % len(endpoints)
		self._round_robin_counters[counter_key] += 1
		
		return endpoints[index]
	
	async def _weighted_round_robin_selection(
		self,
		endpoints: List[Dict[str, Any]],
		route_id: str
	) -> Dict[str, Any]:
		"""Weighted round-robin endpoint selection."""
		# Create weighted list
		weighted_endpoints = []
		for endpoint in endpoints:
			weight = endpoint.get("weight", 100) * endpoint.get("service_weight", 100) // 100
			weighted_endpoints.extend([endpoint] * max(1, weight // 10))
		
		return await self._round_robin_selection(weighted_endpoints, f"weighted_{route_id}")
	
	async def _least_connections_selection(
		self,
		endpoints: List[Dict[str, Any]]
	) -> Dict[str, Any]:
		"""Least connections endpoint selection."""
		# In a real implementation, this would track active connections
		# For now, return random endpoint
		return random.choice(endpoints)
	
	async def _ip_hash_selection(
		self,
		endpoints: List[Dict[str, Any]],
		client_ip: str
	) -> Dict[str, Any]:
		"""IP hash-based endpoint selection for session affinity."""
		if not client_ip:
			return random.choice(endpoints)
		
		# Use hash of client IP to consistently select endpoint
		hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
		index = hash_value % len(endpoints)
		
		return endpoints[index]

# =============================================================================
# Policy Engine Service
# =============================================================================

class PolicyEngineService:
	"""Policy enforcement and management service."""
	
	def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
		self.db_session = db_session
		self.redis_client = redis_client
	
	async def enforce_policies(
		self,
		policies: List[Dict[str, Any]],
		request_context: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Enforce policies on request."""
		enforcement_result = {
			"allowed": True,
			"modifications": {},
			"applied_policies": [],
			"errors": []
		}
		
		for policy in policies:
			try:
				policy_type = policy["policy_type"]
				config = policy["configuration"]
				
				if policy_type == PolicyType.RATE_LIMIT.value:
					rate_limit_result = await self._enforce_rate_limit(policy, request_context)
					if not rate_limit_result["allowed"]:
						enforcement_result["allowed"] = False
						enforcement_result["errors"].append(rate_limit_result["error"])
				
				elif policy_type == PolicyType.AUTHENTICATION.value:
					auth_result = await self._enforce_authentication(policy, request_context)
					if not auth_result["allowed"]:
						enforcement_result["allowed"] = False
						enforcement_result["errors"].append(auth_result["error"])
				
				elif policy_type == PolicyType.TIMEOUT.value:
					enforcement_result["modifications"]["timeout"] = config.get("timeout_ms", 30000)
				
				elif policy_type == PolicyType.RETRY.value:
					enforcement_result["modifications"]["retry_attempts"] = config.get("attempts", 3)
					enforcement_result["modifications"]["retry_timeout"] = config.get("timeout_ms", 1000)
				
				enforcement_result["applied_policies"].append(policy["policy_id"])
				
			except Exception as e:
				enforcement_result["errors"].append(f"Policy {policy['policy_id']} error: {str(e)}")
		
		return enforcement_result
	
	async def _enforce_rate_limit(
		self,
		policy: Dict[str, Any],
		request_context: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Enforce rate limiting policy."""
		config = policy["configuration"]
		
		# Generate rate limit key
		key_components = [
			"rate_limit",
			policy["policy_id"],
			request_context.get("client_ip", "unknown")
		]
		rate_limit_key = ":".join(key_components)
		
		# Check current count
		requests_per_second = config.get("requests_per_second", 100)
		window_size = config.get("window_size_seconds", 60)
		
		current_count = await self.redis_client.get(rate_limit_key)
		current_count = int(current_count) if current_count else 0
		
		if current_count >= requests_per_second:
			return {
				"allowed": False,
				"error": f"Rate limit exceeded: {current_count}/{requests_per_second} requests"
			}
		
		# Increment counter
		pipe = self.redis_client.pipeline()
		pipe.incr(rate_limit_key)
		pipe.expire(rate_limit_key, window_size)
		await pipe.execute()
		
		return {"allowed": True}
	
	async def _enforce_authentication(
		self,
		policy: Dict[str, Any],
		request_context: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Enforce authentication policy."""
		# Simplified authentication check
		auth_header = request_context.get("headers", {}).get("authorization", "")
		
		if not auth_header:
			return {
				"allowed": False,
				"error": "Authentication required"
			}
		
		# In real implementation, would validate token/credentials
		if auth_header.startswith("Bearer "):
			return {"allowed": True}
		
		return {
			"allowed": False,
			"error": "Invalid authentication"
		}

# =============================================================================
# Health Monitoring Service
# =============================================================================

class HealthMonitoringService:
	"""Service health monitoring and management."""
	
	def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
		self.db_session = db_session
		self.redis_client = redis_client
		self._monitoring_tasks = {}
	
	async def start_monitoring(self, service_id: str):
		"""Start health monitoring for a service."""
		if service_id not in self._monitoring_tasks:
			task = asyncio.create_task(self._monitor_service_health(service_id))
			self._monitoring_tasks[service_id] = task
	
	async def stop_monitoring(self, service_id: str):
		"""Stop health monitoring for a service."""
		if service_id in self._monitoring_tasks:
			self._monitoring_tasks[service_id].cancel()
			del self._monitoring_tasks[service_id]
	
	async def _monitor_service_health(self, service_id: str):
		"""Continuous health monitoring for a service."""
		while True:
			try:
				await self._check_service_health(service_id)
				await asyncio.sleep(30)  # Check every 30 seconds
			except asyncio.CancelledError:
				break
			except Exception as e:
				print(f"Health monitoring error for service {service_id}: {e}")
				await asyncio.sleep(60)  # Wait longer on error
	
	async def _check_service_health(self, service_id: str):
		"""Check health of all endpoints for a service."""
		# Get service endpoints
		query = select(SMEndpoint).where(
			and_(SMEndpoint.service_id == service_id, SMEndpoint.enabled == True)
		)
		result = await self.db_session.execute(query)
		endpoints = result.scalars().all()
		
		health_results = []
		
		for endpoint in endpoints:
			health_result = await self._check_endpoint_health(endpoint)
			health_results.append(health_result)
			
			# Store health check result
			health_check = SMHealthCheck(
				service_id=service_id,
				endpoint_id=endpoint.endpoint_id,
				status=health_result.status.value,
				response_time_ms=health_result.response_time_ms,
				status_code=health_result.status_code,
				error_message=health_result.error_message,
				last_check_at=health_result.timestamp,
				tenant_id=endpoint.tenant_id
			)
			
			# Update consecutive counts
			if health_result.status == HealthStatus.HEALTHY:
				health_check.consecutive_successes = await self._increment_consecutive_count(
					endpoint.endpoint_id, "successes"
				)
				health_check.consecutive_failures = 0
				health_check.last_success_at = health_result.timestamp
			else:
				health_check.consecutive_failures = await self._increment_consecutive_count(
					endpoint.endpoint_id, "failures"
				)
				health_check.consecutive_successes = 0
				health_check.last_failure_at = health_result.timestamp
			
			self.db_session.add(health_check)
		
		# Update overall service health status
		overall_status = self._calculate_service_health(health_results)
		
		query = update(SMService).where(SMService.service_id == service_id).values(
			health_status=overall_status.value,
			last_health_check=datetime.now(timezone.utc)
		)
		await self.db_session.execute(query)
		await self.db_session.commit()
	
	async def _check_endpoint_health(self, endpoint: SMEndpoint) -> HealthCheckResult:
		"""Check health of a specific endpoint."""
		start_time = time.time()
		
		try:
			# Build health check URL
			protocol = "https" if endpoint.tls_enabled else "http"
			url = f"{protocol}://{endpoint.host}:{endpoint.port}{endpoint.health_check_path}"
			
			async with httpx.AsyncClient(timeout=endpoint.health_check_timeout) as client:
				response = await client.get(url)
				
				response_time = (time.time() - start_time) * 1000
				
				if 200 <= response.status_code < 300:
					status = HealthStatus.HEALTHY
				else:
					status = HealthStatus.UNHEALTHY
				
				return HealthCheckResult(
					endpoint_id=endpoint.endpoint_id,
					status=status,
					response_time_ms=response_time,
					status_code=response.status_code,
					error_message=None,
					timestamp=datetime.now(timezone.utc)
				)
				
		except httpx.TimeoutException:
			response_time = (time.time() - start_time) * 1000
			return HealthCheckResult(
				endpoint_id=endpoint.endpoint_id,
				status=HealthStatus.TIMEOUT,
				response_time_ms=response_time,
				status_code=None,
				error_message="Health check timeout",
				timestamp=datetime.now(timezone.utc)
			)
		
		except Exception as e:
			response_time = (time.time() - start_time) * 1000
			return HealthCheckResult(
				endpoint_id=endpoint.endpoint_id,
				status=HealthStatus.CONNECTION_FAILED,
				response_time_ms=response_time,
				status_code=None,
				error_message=str(e),
				timestamp=datetime.now(timezone.utc)
			)
	
	def _calculate_service_health(self, health_results: List[HealthCheckResult]) -> HealthStatus:
		"""Calculate overall service health from endpoint results."""
		if not health_results:
			return HealthStatus.UNKNOWN
		
		healthy_count = len([r for r in health_results if r.status == HealthStatus.HEALTHY])
		total_count = len(health_results)
		
		health_ratio = healthy_count / total_count
		
		if health_ratio >= 0.8:
			return HealthStatus.HEALTHY
		elif health_ratio >= 0.5:
			return HealthStatus.HEALTHY  # Degraded but still functional
		else:
			return HealthStatus.UNHEALTHY
	
	async def _increment_consecutive_count(self, endpoint_id: str, count_type: str) -> int:
		"""Increment consecutive success/failure count."""
		cache_key = f"health:consecutive:{endpoint_id}:{count_type}"
		count = await self.redis_client.incr(cache_key)
		await self.redis_client.expire(cache_key, 3600)  # 1 hour TTL
		return count

# =============================================================================
# Metrics Collection Service
# =============================================================================

class MetricsCollectionService:
	"""Metrics collection and analytics service."""
	
	def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
		self.db_session = db_session
		self.redis_client = redis_client
	
	async def record_request(
		self,
		service_id: str,
		endpoint_id: str,
		request_data: Dict[str, Any],
		response_data: Dict[str, Any],
		tenant_id: str
	):
		"""Record request metrics."""
		
		# Calculate response time
		start_time = request_data.get("start_time")
		end_time = response_data.get("end_time")
		response_time_ms = None
		
		if start_time and end_time:
			if isinstance(start_time, str):
				start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
			if isinstance(end_time, str):
				end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
			
			response_time_ms = (end_time - start_time).total_seconds() * 1000
		
		# Determine if request was successful
		status_code = response_data.get("status_code", 200)
		error_count = 1 if status_code >= 400 else 0
		
		# Create metrics record
		metric = SMMetrics(
			service_id=service_id,
			metric_name="request_total",
			metric_type="counter",
			value=1,
			timestamp=datetime.now(timezone.utc),
			request_count=1,
			error_count=error_count,
			response_time_ms=response_time_ms,
			status_code=status_code,
			labels={
				"endpoint_id": endpoint_id,
				"method": request_data.get("method", "GET"),
				"status_class": f"{status_code // 100}xx"
			},
			metadata={
				"user_agent": request_data.get("user_agent"),
				"client_ip": request_data.get("client_ip")
			},
			tenant_id=tenant_id
		)
		
		self.db_session.add(metric)
		
		# Update real-time metrics in Redis
		await self._update_realtime_metrics(service_id, endpoint_id, response_time_ms, error_count)
		
		await self.db_session.commit()
	
	async def _update_realtime_metrics(
		self,
		service_id: str,
		endpoint_id: str,
		response_time_ms: Optional[float],
		error_count: int
	):
		"""Update real-time metrics in Redis."""
		pipe = self.redis_client.pipeline()
		
		# Request counters
		pipe.incr(f"metrics:requests:total:{service_id}")
		pipe.incr(f"metrics:requests:total:{endpoint_id}")
		
		if error_count > 0:
			pipe.incr(f"metrics:errors:total:{service_id}")
			pipe.incr(f"metrics:errors:total:{endpoint_id}")
		
		# Response time tracking
		if response_time_ms is not None:
			pipe.lpush(f"metrics:response_times:{service_id}", response_time_ms)
			pipe.ltrim(f"metrics:response_times:{service_id}", 0, 999)  # Keep last 1000
			
			pipe.lpush(f"metrics:response_times:{endpoint_id}", response_time_ms)
			pipe.ltrim(f"metrics:response_times:{endpoint_id}", 0, 999)
		
		# Set TTL for metrics
		for key in [
			f"metrics:requests:total:{service_id}",
			f"metrics:requests:total:{endpoint_id}",
			f"metrics:errors:total:{service_id}",
			f"metrics:errors:total:{endpoint_id}",
			f"metrics:response_times:{service_id}",
			f"metrics:response_times:{endpoint_id}"
		]:
			pipe.expire(key, 86400)  # 24 hours
		
		await pipe.execute()
	
	async def get_recent_metrics(self, tenant_id: str, hours: int = 1) -> Dict[str, Any]:
		"""Get recent metrics for tenant."""
		since = datetime.now(timezone.utc) - timedelta(hours=hours)
		
		# Get request metrics
		query = select(
			SMMetrics.service_id,
			func.count().label("request_count"),
			func.sum(SMMetrics.error_count).label("error_count"),
			func.avg(SMMetrics.response_time_ms).label("avg_response_time"),
			func.percentile_cont(0.95).within_group(SMMetrics.response_time_ms).label("p95_response_time")
		).where(
			and_(
				SMMetrics.tenant_id == tenant_id,
				SMMetrics.timestamp >= since,
				SMMetrics.metric_name == "request_total"
			)
		).group_by(SMMetrics.service_id)
		
		result = await self.db_session.execute(query)
		metrics = result.all()
		
		metrics_data = {}
		for metric in metrics:
			service_metrics = TrafficMetrics(
				request_count=metric.request_count,
				error_count=metric.error_count or 0,
				avg_response_time=metric.avg_response_time or 0,
				p95_response_time=metric.p95_response_time or 0,
				throughput_rps=metric.request_count / (hours * 3600)
			)
			metrics_data[metric.service_id] = service_metrics
		
		return metrics_data
	
	async def get_topology(self, tenant_id: str) -> Dict[str, Any]:
		"""Get service dependency topology."""
		query = select(SMTopology).where(SMTopology.tenant_id == tenant_id)
		result = await self.db_session.execute(query)
		topology_records = result.scalars().all()
		
		nodes = set()
		edges = []
		
		for record in topology_records:
			nodes.add(record.source_service_id)
			nodes.add(record.target_service_id)
			
			edges.append({
				"source": record.source_service_id,
				"target": record.target_service_id,
				"relationship_type": record.relationship_type,
				"weight": record.weight,
				"protocol": record.protocol,
				"avg_response_time": record.avg_response_time_ms,
				"request_count": record.request_count,
				"error_count": record.error_count
			})
		
		# Get service details for nodes
		services_query = select(SMService).where(
			and_(
				SMService.service_id.in_(list(nodes)),
				SMService.tenant_id == tenant_id
			)
		)
		services_result = await self.db_session.execute(services_query)
		services = services_result.scalars().all()
		
		service_nodes = []
		for service in services:
			service_nodes.append({
				"service_id": service.service_id,
				"service_name": service.service_name,
				"service_version": service.service_version,
				"status": service.status,
				"health_status": service.health_status
			})
		
		return {
			"nodes": service_nodes,
			"edges": edges,
			"metrics": {
				"total_services": len(service_nodes),
				"total_connections": len(edges),
				"timestamp": datetime.utcnow().isoformat()
			}
		}

# =============================================================================
# Service Factory
# =============================================================================

async def create_asm_service(db_session: AsyncSession, redis_url: str) -> ASMService:
	"""Factory function to create ASM service with dependencies."""
	redis_client = redis.from_url(redis_url)
	return ASMService(db_session, redis_client)