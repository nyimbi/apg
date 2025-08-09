"""
APG API Service Mesh - Service Mesh Federation Engine

Multi-cluster service discovery, cross-region traffic routing,
and federated mesh management for distributed deployments.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import socket

import httpx
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_
from uuid_extensions import uuid7str

from .models import SMService, SMEndpoint, SMRoute, SMMetrics, SMTopology
from .tls_certificate_manager import TLSCertificateManager, CertificateBundle


class FederationState(str, Enum):
	"""Federation connection states."""
	CONNECTED = "connected"
	DISCONNECTED = "disconnected"
	CONNECTING = "connecting"
	ERROR = "error"
	SYNCHRONIZING = "synchronizing"


class ClusterRole(str, Enum):
	"""Cluster roles in federation."""
	PRIMARY = "primary"      # Primary cluster
	SECONDARY = "secondary"  # Secondary cluster
	OBSERVER = "observer"    # Read-only observer


@dataclass
class ClusterInfo:
	"""Information about a federated cluster."""
	cluster_id: str
	cluster_name: str
	region: str
	zone: str
	endpoint: str
	role: ClusterRole
	version: str
	capabilities: List[str]
	metadata: Dict[str, Any] = field(default_factory=dict)
	last_seen: Optional[datetime] = None
	state: FederationState = FederationState.DISCONNECTED


@dataclass
class FederatedService:
	"""Service distributed across clusters."""
	service_name: str
	namespace: str
	clusters: Dict[str, Dict[str, Any]]  # cluster_id -> service info
	total_endpoints: int
	healthy_endpoints: int
	traffic_weights: Dict[str, float]  # cluster_id -> weight
	failover_policy: Dict[str, Any]
	load_balancing_strategy: str


@dataclass
class CrossClusterRoute:
	"""Route spanning multiple clusters."""
	route_id: str
	source_cluster: str
	target_clusters: List[str]
	service_name: str
	namespace: str
	traffic_split: Dict[str, float]  # cluster_id -> percentage
	latency_requirements: Optional[Dict[str, float]] = None
	failover_order: Optional[List[str]] = None


class ServiceMeshFederation:
	"""Multi-cluster service mesh federation manager."""
	
	def __init__(
		self,
		cluster_id: str,
		cluster_name: str,
		region: str,
		zone: str,
		db_session: AsyncSession,
		redis_client: redis.Redis,
		cert_manager: TLSCertificateManager,
		federation_endpoint: str = "https://federation.service-mesh.local",
		role: ClusterRole = ClusterRole.SECONDARY
	):
		self.cluster_id = cluster_id
		self.cluster_name = cluster_name
		self.region = region
		self.zone = zone
		self.db_session = db_session
		self.redis_client = redis_client
		self.cert_manager = cert_manager
		self.federation_endpoint = federation_endpoint
		self.role = role
		
		# Federation state
		self.connected_clusters: Dict[str, ClusterInfo] = {}
		self.federated_services: Dict[str, FederatedService] = {}
		self.cross_cluster_routes: Dict[str, CrossClusterRoute] = {}
		
		# Network clients for cluster communication
		self.http_clients: Dict[str, httpx.AsyncClient] = {}
		
		# Synchronization state
		self._sync_in_progress = False
		self._last_sync_time: Optional[datetime] = None
		
		# Discovery cache
		self._service_cache: Dict[str, Dict[str, Any]] = {}
		self._cache_ttl = 300  # 5 minutes
	
	async def initialize(self) -> None:
		"""Initialize federation manager."""
		await self._setup_federation_certificate()
		await self._start_federation_services()
		await self._discover_existing_clusters()
		
		# Start background tasks
		asyncio.create_task(self._federation_heartbeat_loop())
		asyncio.create_task(self._service_synchronization_loop())
		asyncio.create_task(self._health_monitoring_loop())
	
	async def _setup_federation_certificate(self) -> None:
		"""Setup certificates for federation communication."""
		# Generate federation certificate
		federation_cert = await self.cert_manager.generate_service_certificate(
			service_name=f"federation-{self.cluster_id}",
			namespace="mesh-system",
			sans=[
				f"federation.{self.cluster_name}",
				f"federation.{self.region}",
				"federation.mesh.local"
			],
			validity_days=365
		)
		
		# Store federation certificate
		await self.redis_client.set(
			f"federation:cert:{self.cluster_id}",
			json.dumps({
				"certificate": federation_cert.certificate,
				"private_key": federation_cert.private_key,
				"ca_certificate": federation_cert.ca_certificate,
				"fingerprint": federation_cert.fingerprint
			})
		)
	
	async def register_cluster(
		self,
		target_cluster_endpoint: str,
		shared_secret: str
	) -> bool:
		"""Register this cluster with a federation."""
		try:
			# Prepare registration payload
			registration_data = {
				"cluster_id": self.cluster_id,
				"cluster_name": self.cluster_name,
				"region": self.region,
				"zone": self.zone,
				"endpoint": self.federation_endpoint,
				"role": self.role.value,
				"version": "1.0.0",
				"capabilities": [
					"service_discovery",
					"traffic_routing",
					"health_monitoring",
					"mtls_support",
					"load_balancing"
				],
				"timestamp": datetime.now(timezone.utc).isoformat(),
				"shared_secret_hash": hashlib.sha256(shared_secret.encode()).hexdigest()
			}
			
			# Get federation certificate
			cert_bundle = await self.cert_manager.get_certificate_bundle(
				f"federation-{self.cluster_id}"
			)
			
			# Send registration request
			async with httpx.AsyncClient(
				verify=False,  # Custom CA
				cert=(cert_bundle.certificate, cert_bundle.private_key) if cert_bundle else None
			) as client:
				response = await client.post(
					f"{target_cluster_endpoint}/api/v1/federation/register",
					json=registration_data,
					timeout=30.0
				)
				
				if response.status_code == 200:
					federation_info = response.json()
					await self._process_federation_response(federation_info)
					return True
				else:
					print(f"Federation registration failed: {response.status_code} - {response.text}")
					return False
		
		except Exception as e:
			print(f"Error registering cluster: {e}")
			return False
	
	async def _process_federation_response(self, federation_info: Dict[str, Any]) -> None:
		"""Process federation registration response."""
		# Add connected clusters
		for cluster_data in federation_info.get("clusters", []):
			cluster_info = ClusterInfo(
				cluster_id=cluster_data["cluster_id"],
				cluster_name=cluster_data["cluster_name"],
				region=cluster_data["region"],
				zone=cluster_data["zone"],
				endpoint=cluster_data["endpoint"],
				role=ClusterRole(cluster_data["role"]),
				version=cluster_data["version"],
				capabilities=cluster_data["capabilities"],
				metadata=cluster_data.get("metadata", {}),
				last_seen=datetime.now(timezone.utc),
				state=FederationState.CONNECTED
			)
			
			self.connected_clusters[cluster_info.cluster_id] = cluster_info
			
			# Setup HTTP client for this cluster
			await self._setup_cluster_client(cluster_info)
		
		# Store federation topology
		await self.redis_client.set(
			f"federation:topology:{self.cluster_id}",
			json.dumps({
				"clusters": [
					{
						"cluster_id": info.cluster_id,
						"cluster_name": info.cluster_name,
						"region": info.region,
						"zone": info.zone,
						"endpoint": info.endpoint,
						"role": info.role.value,
						"state": info.state.value
					}
					for info in self.connected_clusters.values()
				],
				"updated_at": datetime.now(timezone.utc).isoformat()
			}),
			ex=3600  # 1 hour TTL
		)
	
	async def _setup_cluster_client(self, cluster_info: ClusterInfo) -> None:
		"""Setup HTTP client for cluster communication."""
		# Get federation certificate
		cert_bundle = await self.cert_manager.get_certificate_bundle(
			f"federation-{self.cluster_id}"
		)
		
		# Create HTTP client with mTLS
		client = httpx.AsyncClient(
			base_url=cluster_info.endpoint,
			verify=False,  # Custom CA
			cert=(cert_bundle.certificate, cert_bundle.private_key) if cert_bundle else None,
			timeout=httpx.Timeout(10.0),
			headers={
				"X-Cluster-ID": self.cluster_id,
				"X-Cluster-Name": self.cluster_name,
				"X-Region": self.region
			}
		)
		
		self.http_clients[cluster_info.cluster_id] = client
	
	async def discover_federated_services(
		self,
		service_name: Optional[str] = None,
		namespace: Optional[str] = None
	) -> List[FederatedService]:
		"""Discover services across federated clusters."""
		federated_services = []
		
		# Query each connected cluster
		for cluster_id, cluster_info in self.connected_clusters.items():
			if cluster_info.state != FederationState.CONNECTED:
				continue
			
			try:
				client = self.http_clients.get(cluster_id)
				if not client:
					continue
				
				# Build query parameters
				params = {}
				if service_name:
					params["service_name"] = service_name
				if namespace:
					params["namespace"] = namespace
				
				# Query cluster services
				response = await client.get(
					"/api/v1/services",
					params=params
				)
				
				if response.status_code == 200:
					services_data = response.json()
					
					# Process services from this cluster
					for service_data in services_data.get("services", []):
						await self._process_federated_service(
							cluster_id,
							cluster_info,
							service_data
						)
			
			except Exception as e:
				print(f"Error discovering services from cluster {cluster_id}: {e}")
				# Mark cluster as having errors
				cluster_info.state = FederationState.ERROR
		
		# Build federated service list
		for service_key, service in self.federated_services.items():
			federated_services.append(service)
		
		return federated_services
	
	async def _process_federated_service(
		self,
		cluster_id: str,
		cluster_info: ClusterInfo,
		service_data: Dict[str, Any]
	) -> None:
		"""Process service discovered from a federated cluster."""
		service_name = service_data["name"]
		namespace = service_data["namespace"]
		service_key = f"{namespace}/{service_name}"
		
		# Get or create federated service
		if service_key not in self.federated_services:
			self.federated_services[service_key] = FederatedService(
				service_name=service_name,
				namespace=namespace,
				clusters={},
				total_endpoints=0,
				healthy_endpoints=0,
				traffic_weights={},
				failover_policy={},
				load_balancing_strategy="round_robin"
			)
		
		federated_service = self.federated_services[service_key]
		
		# Add cluster information
		federated_service.clusters[cluster_id] = {
			"cluster_name": cluster_info.cluster_name,
			"region": cluster_info.region,
			"zone": cluster_info.zone,
			"endpoints": service_data.get("endpoints", []),
			"health_status": service_data.get("health_status", "unknown"),
			"version": service_data.get("version", "unknown"),
			"metadata": service_data.get("metadata", {})
		}
		
		# Update endpoint counts
		cluster_endpoints = len(service_data.get("endpoints", []))
		healthy_endpoints = len([
			ep for ep in service_data.get("endpoints", [])
			if ep.get("health_status") == "healthy"
		])
		
		federated_service.total_endpoints += cluster_endpoints
		federated_service.healthy_endpoints += healthy_endpoints
		
		# Set default traffic weight (can be customized later)
		if cluster_id not in federated_service.traffic_weights:
			federated_service.traffic_weights[cluster_id] = 1.0
	
	async def create_cross_cluster_route(
		self,
		service_name: str,
		namespace: str,
		traffic_split: Dict[str, float],
		latency_requirements: Optional[Dict[str, float]] = None,
		failover_order: Optional[List[str]] = None
	) -> str:
		"""Create route spanning multiple clusters."""
		route_id = uuid7str()
		
		# Validate clusters exist
		for cluster_id in traffic_split.keys():
			if cluster_id not in self.connected_clusters:
				raise ValueError(f"Unknown cluster: {cluster_id}")
		
		# Validate traffic split sums to 100%
		total_traffic = sum(traffic_split.values())
		if not (0.99 <= total_traffic <= 1.01):  # Allow for floating point precision
			raise ValueError(f"Traffic split must sum to 1.0, got {total_traffic}")
		
		# Create cross-cluster route
		cross_route = CrossClusterRoute(
			route_id=route_id,
			source_cluster=self.cluster_id,
			target_clusters=list(traffic_split.keys()),
			service_name=service_name,
			namespace=namespace,
			traffic_split=traffic_split,
			latency_requirements=latency_requirements,
			failover_order=failover_order or list(traffic_split.keys())
		)
		
		self.cross_cluster_routes[route_id] = cross_route
		
		# Store in Redis for persistence
		await self.redis_client.set(
			f"federation:route:{route_id}",
			json.dumps({
				"route_id": route_id,
				"source_cluster": self.cluster_id,
				"target_clusters": cross_route.target_clusters,
				"service_name": service_name,
				"namespace": namespace,
				"traffic_split": traffic_split,
				"latency_requirements": latency_requirements,
				"failover_order": failover_order,
				"created_at": datetime.now(timezone.utc).isoformat()
			}),
			ex=86400  # 24 hours TTL
		)
		
		# Distribute route to target clusters
		await self._distribute_route_configuration(cross_route)
		
		return route_id
	
	async def _distribute_route_configuration(self, cross_route: CrossClusterRoute) -> None:
		"""Distribute route configuration to target clusters."""
		route_config = {
			"route_id": cross_route.route_id,
			"source_cluster": cross_route.source_cluster,
			"service_name": cross_route.service_name,
			"namespace": cross_route.namespace,
			"traffic_split": cross_route.traffic_split,
			"latency_requirements": cross_route.latency_requirements,
			"failover_order": cross_route.failover_order
		}
		
		# Send configuration to each target cluster
		for cluster_id in cross_route.target_clusters:
			try:
				client = self.http_clients.get(cluster_id)
				if not client:
					continue
				
				response = await client.post(
					"/api/v1/federation/routes",
					json=route_config
				)
				
				if response.status_code != 200:
					print(f"Failed to distribute route to cluster {cluster_id}: {response.text}")
			
			except Exception as e:
				print(f"Error distributing route to cluster {cluster_id}: {e}")
	
	async def resolve_federated_service(
		self,
		service_name: str,
		namespace: str,
		client_region: Optional[str] = None,
		latency_preference: bool = True
	) -> List[Dict[str, Any]]:
		"""Resolve service endpoints across federated clusters."""
		service_key = f"{namespace}/{service_name}"
		federated_service = self.federated_services.get(service_key)
		
		if not federated_service:
			# Try to discover the service
			await self.discover_federated_services(service_name, namespace)
			federated_service = self.federated_services.get(service_key)
		
		if not federated_service:
			return []
		
		endpoints = []
		
		# Collect endpoints from all clusters
		for cluster_id, cluster_data in federated_service.clusters.items():
			cluster_info = self.connected_clusters.get(cluster_id)
			if not cluster_info or cluster_info.state != FederationState.CONNECTED:
				continue
			
			for endpoint in cluster_data.get("endpoints", []):
				if endpoint.get("health_status") == "healthy":
					endpoint_data = {
						"cluster_id": cluster_id,
						"cluster_name": cluster_info.cluster_name,
						"region": cluster_info.region,
						"zone": cluster_info.zone,
						"endpoint": endpoint,
						"weight": federated_service.traffic_weights.get(cluster_id, 1.0)
					}
					endpoints.append(endpoint_data)
		
		# Apply region preference if specified
		if client_region and latency_preference:
			# Sort by region preference (same region first)
			endpoints.sort(key=lambda ep: (
				0 if ep["region"] == client_region else 1,
				ep["cluster_name"]
			))
		
		return endpoints
	
	async def get_federation_metrics(self) -> Dict[str, Any]:
		"""Get federation metrics and status."""
		metrics = {
			"cluster_info": {
				"cluster_id": self.cluster_id,
				"cluster_name": self.cluster_name,
				"region": self.region,
				"zone": self.zone,
				"role": self.role.value
			},
			"connected_clusters": len(self.connected_clusters),
			"cluster_states": {},
			"federated_services": len(self.federated_services),
			"cross_cluster_routes": len(self.cross_cluster_routes),
			"last_sync_time": self._last_sync_time.isoformat() if self._last_sync_time else None,
			"sync_in_progress": self._sync_in_progress
		}
		
		# Cluster state breakdown
		for state in FederationState:
			count = sum(1 for cluster in self.connected_clusters.values() 
					   if cluster.state == state)
			metrics["cluster_states"][state.value] = count
		
		# Service distribution metrics
		service_metrics = {
			"total_services": len(self.federated_services),
			"multi_cluster_services": 0,
			"single_cluster_services": 0,
			"services_by_region": defaultdict(int)
		}
		
		for service in self.federated_services.values():
			if len(service.clusters) > 1:
				service_metrics["multi_cluster_services"] += 1
			else:
				service_metrics["single_cluster_services"] += 1
			
			for cluster_id in service.clusters:
				cluster_info = self.connected_clusters.get(cluster_id)
				if cluster_info:
					service_metrics["services_by_region"][cluster_info.region] += 1
		
		metrics["service_distribution"] = dict(service_metrics["services_by_region"])
		metrics.update(service_metrics)
		
		return metrics
	
	async def _federation_heartbeat_loop(self) -> None:
		"""Send heartbeats to federated clusters."""
		while True:
			try:
				await asyncio.sleep(30)  # Heartbeat every 30 seconds
				
				heartbeat_data = {
					"cluster_id": self.cluster_id,
					"cluster_name": self.cluster_name,
					"timestamp": datetime.now(timezone.utc).isoformat(),
					"services_count": len(self.federated_services),
					"health_status": "healthy"
				}
				
				# Send heartbeat to all connected clusters
				for cluster_id, cluster_info in self.connected_clusters.items():
					try:
						client = self.http_clients.get(cluster_id)
						if not client:
							continue
						
						response = await client.post(
							"/api/v1/federation/heartbeat",
							json=heartbeat_data,
							timeout=10.0
						)
						
						if response.status_code == 200:
							cluster_info.last_seen = datetime.now(timezone.utc)
							cluster_info.state = FederationState.CONNECTED
						else:
							cluster_info.state = FederationState.ERROR
					
					except Exception as e:
						print(f"Heartbeat failed for cluster {cluster_id}: {e}")
						cluster_info.state = FederationState.ERROR
			
			except Exception as e:
				print(f"Error in federation heartbeat loop: {e}")
	
	async def _service_synchronization_loop(self) -> None:
		"""Synchronize services across federated clusters."""
		while True:
			try:
				await asyncio.sleep(60)  # Sync every minute
				
				if not self._sync_in_progress:
					self._sync_in_progress = True
					await self.discover_federated_services()
					self._last_sync_time = datetime.now(timezone.utc)
					self._sync_in_progress = False
			
			except Exception as e:
				print(f"Error in service synchronization loop: {e}")
				self._sync_in_progress = False
	
	async def _health_monitoring_loop(self) -> None:
		"""Monitor health of federated clusters."""
		while True:
			try:
				await asyncio.sleep(120)  # Health check every 2 minutes
				
				# Check for stale clusters
				stale_threshold = datetime.now(timezone.utc) - timedelta(minutes=5)
				
				for cluster_id, cluster_info in self.connected_clusters.items():
					if cluster_info.last_seen and cluster_info.last_seen < stale_threshold:
						if cluster_info.state == FederationState.CONNECTED:
							cluster_info.state = FederationState.DISCONNECTED
							print(f"Cluster {cluster_id} marked as disconnected (stale)")
			
			except Exception as e:
				print(f"Error in health monitoring loop: {e}")
	
	async def _discover_existing_clusters(self) -> None:
		"""Discover existing federated clusters from storage."""
		try:
			# Load federation topology from Redis
			topology_data = await self.redis_client.get(f"federation:topology:{self.cluster_id}")
			if topology_data:
				topology = json.loads(topology_data)
				
				for cluster_data in topology.get("clusters", []):
					cluster_info = ClusterInfo(
						cluster_id=cluster_data["cluster_id"],
						cluster_name=cluster_data["cluster_name"],
						region=cluster_data["region"],
						zone=cluster_data["zone"],
						endpoint=cluster_data["endpoint"],
						role=ClusterRole(cluster_data["role"]),
						version=cluster_data.get("version", "unknown"),
						capabilities=cluster_data.get("capabilities", []),
						state=FederationState(cluster_data.get("state", "disconnected"))
					)
					
					self.connected_clusters[cluster_info.cluster_id] = cluster_info
					await self._setup_cluster_client(cluster_info)
		
		except Exception as e:
			print(f"Error discovering existing clusters: {e}")
	
	async def _start_federation_services(self) -> None:
		"""Start federation-specific services."""
		# This would typically start additional services like:
		# - Federation API server
		# - Service mesh proxy with federation routing
		# - Certificate rotation service
		# - Federation metrics collector
		pass


# =============================================================================
# Utilities
# =============================================================================

from collections import defaultdict