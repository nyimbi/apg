"""
APG API Service Mesh - gRPC Protocol Support Engine

Complete gRPC protocol support with health checking, load balancing,
streaming support, and service mesh integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, AsyncIterator, Callable
from dataclasses import dataclass
from enum import Enum
import socket

import grpc
from grpc import aio
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection
import grpc_status
from google.rpc import status_pb2, code_pb2

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from uuid_extensions import uuid7str

from .models import SMService, SMEndpoint, SMHealthCheck, SMMetrics, SMRoute
from .advanced_circuit_breaker import AdvancedCircuitBreaker, CircuitConfig
from .tls_certificate_manager import TLSCertificateManager


class GRPCServiceStatus(str, Enum):
	"""gRPC service health status."""
	SERVING = "SERVING"
	NOT_SERVING = "NOT_SERVING"
	UNKNOWN = "UNKNOWN"


class GRPCMethodType(str, Enum):
	"""gRPC method types."""
	UNARY = "unary"
	CLIENT_STREAMING = "client_streaming"
	SERVER_STREAMING = "server_streaming"
	BIDIRECTIONAL_STREAMING = "bidirectional_streaming"


@dataclass
class GRPCServiceInfo:
	"""gRPC service information."""
	service_name: str
	methods: Dict[str, GRPCMethodType]
	health_status: GRPCServiceStatus
	endpoint: str
	port: int
	reflection_enabled: bool
	tls_enabled: bool
	metadata: Dict[str, Any]


@dataclass
class GRPCCallMetrics:
	"""Metrics for gRPC calls."""
	service_name: str
	method_name: str
	request_count: int = 0
	error_count: int = 0
	total_duration_ms: float = 0.0
	min_duration_ms: float = float('inf')
	max_duration_ms: float = 0.0
	streaming_messages: int = 0
	last_call_time: Optional[datetime] = None


class GRPCHealthChecker:
	"""gRPC health checking implementation."""
	
	def __init__(self, db_session: AsyncSession):
		self.db_session = db_session
		self.health_clients: Dict[str, health_pb2_grpc.HealthStub] = {}
		self.channels: Dict[str, aio.Channel] = {}
	
	async def check_service_health(
		self,
		endpoint: str,
		port: int,
		service_name: str = "",
		timeout: float = 5.0,
		use_tls: bool = False,
		cert_bundle: Optional[Any] = None
	) -> GRPCServiceStatus:
		"""Check health of a gRPC service."""
		try:
			channel_key = f"{endpoint}:{port}"
			
			# Create channel if not exists
			if channel_key not in self.channels:
				if use_tls and cert_bundle:
					credentials = grpc.ssl_channel_credentials(
						root_certificates=cert_bundle.ca_certificate.encode(),
						private_key=cert_bundle.private_key.encode(),
						certificate_chain=cert_bundle.certificate.encode()
					)
					channel = aio.secure_channel(f"{endpoint}:{port}", credentials)
				else:
					channel = aio.insecure_channel(f"{endpoint}:{port}")
				
				self.channels[channel_key] = channel
				self.health_clients[channel_key] = health_pb2_grpc.HealthStub(channel)
			
			# Perform health check
			health_client = self.health_clients[channel_key]
			request = health_pb2.HealthCheckRequest(service=service_name)
			
			response = await asyncio.wait_for(
				health_client.Check(request),
				timeout=timeout
			)
			
			status_map = {
				health_pb2.HealthCheckResponse.SERVING: GRPCServiceStatus.SERVING,
				health_pb2.HealthCheckResponse.NOT_SERVING: GRPCServiceStatus.NOT_SERVING,
				health_pb2.HealthCheckResponse.UNKNOWN: GRPCServiceStatus.UNKNOWN
			}
			
			return status_map.get(response.status, GRPCServiceStatus.UNKNOWN)
		
		except asyncio.TimeoutError:
			return GRPCServiceStatus.NOT_SERVING
		except grpc.RpcError as e:
			if e.code() == grpc.StatusCode.UNIMPLEMENTED:
				# Service doesn't implement health checking, assume healthy
				return GRPCServiceStatus.SERVING
			return GRPCServiceStatus.NOT_SERVING
		except Exception as e:
			print(f"gRPC health check error: {e}")
			return GRPCServiceStatus.UNKNOWN
	
	async def watch_service_health(
		self,
		endpoint: str,
		port: int,
		service_name: str = "",
		callback: Optional[Callable[[GRPCServiceStatus], None]] = None
	) -> None:
		"""Watch health status changes for a service."""
		try:
			channel_key = f"{endpoint}:{port}"
			if channel_key not in self.health_clients:
				await self.check_service_health(endpoint, port, service_name)
			
			health_client = self.health_clients[channel_key]
			request = health_pb2.HealthCheckRequest(service=service_name)
			
			async for response in health_client.Watch(request):
				status_map = {
					health_pb2.HealthCheckResponse.SERVING: GRPCServiceStatus.SERVING,
					health_pb2.HealthCheckResponse.NOT_SERVING: GRPCServiceStatus.NOT_SERVING,
					health_pb2.HealthCheckResponse.UNKNOWN: GRPCServiceStatus.UNKNOWN
				}
				
				status = status_map.get(response.status, GRPCServiceStatus.UNKNOWN)
				
				if callback:
					callback(status)
				
				# Update database
				await self._update_health_status(endpoint, port, service_name, status)
		
		except Exception as e:
			print(f"gRPC health watch error: {e}")
	
	async def _update_health_status(
		self,
		endpoint: str,
		port: int,
		service_name: str,
		status: GRPCServiceStatus
	) -> None:
		"""Update health status in database."""
		try:
			# Find the service endpoint
			result = await self.db_session.execute(
				select(SMEndpoint).where(
					and_(
						SMEndpoint.host == endpoint,
						SMEndpoint.port == port,
						SMEndpoint.protocol == "grpc"
					)
				)
			)
			endpoint_record = result.scalars().first()
			
			if endpoint_record:
				# Update health check
				health_check = SMHealthCheck(
					id=uuid7str(),
					service_id=endpoint_record.service_id,
					endpoint_id=endpoint_record.id,
					health_status="healthy" if status == GRPCServiceStatus.SERVING else "unhealthy",
					response_time_ms=0,  # Health check response time
					status_code=200 if status == GRPCServiceStatus.SERVING else 503,
					metadata={
						"grpc_status": status.value,
						"service_name": service_name
					},
					checked_at=datetime.now(timezone.utc)
				)
				
				self.db_session.add(health_check)
				await self.db_session.commit()
		
		except Exception as e:
			print(f"Error updating gRPC health status: {e}")
	
	async def close_connections(self) -> None:
		"""Close all gRPC channels."""
		for channel in self.channels.values():
			await channel.close()
		
		self.channels.clear()
		self.health_clients.clear()


class GRPCServiceDiscovery:
	"""gRPC service discovery using reflection."""
	
	def __init__(self):
		self.reflection_clients: Dict[str, Any] = {}
		self.service_cache: Dict[str, GRPCServiceInfo] = {}
	
	async def discover_services(
		self,
		endpoint: str,
		port: int,
		use_tls: bool = False,
		cert_bundle: Optional[Any] = None
	) -> List[GRPCServiceInfo]:
		"""Discover gRPC services using reflection."""
		try:
			channel_key = f"{endpoint}:{port}"
			
			# Create channel
			if use_tls and cert_bundle:
				credentials = grpc.ssl_channel_credentials(
					root_certificates=cert_bundle.ca_certificate.encode(),
					private_key=cert_bundle.private_key.encode(),
					certificate_chain=cert_bundle.certificate.encode()
				)
				channel = aio.secure_channel(f"{endpoint}:{port}", credentials)
			else:
				channel = aio.insecure_channel(f"{endpoint}:{port}")
			
			# Use reflection to discover services
			reflection_client = reflection.ReflectionClient(channel)
			
			services = []
			service_names = await reflection_client.list_services()
			
			for service_name in service_names:
				if service_name.startswith("grpc."):
					continue  # Skip internal gRPC services
				
				# Get service descriptor
				service_descriptor = await reflection_client.get_service_descriptor(service_name)
				
				# Extract method information
				methods = {}
				for method in service_descriptor.methods:
					method_type = self._determine_method_type(method)
					methods[method.name] = method_type
				
				# Check health status
				health_checker = GRPCHealthChecker(None)  # Temporary instance
				health_status = await health_checker.check_service_health(
					endpoint, port, service_name, use_tls=use_tls, cert_bundle=cert_bundle
				)
				
				service_info = GRPCServiceInfo(
					service_name=service_name,
					methods=methods,
					health_status=health_status,
					endpoint=endpoint,
					port=port,
					reflection_enabled=True,
					tls_enabled=use_tls,
					metadata={
						"package": service_descriptor.package if hasattr(service_descriptor, 'package') else "",
						"method_count": len(methods)
					}
				)
				
				services.append(service_info)
				self.service_cache[f"{endpoint}:{port}:{service_name}"] = service_info
			
			await channel.close()
			return services
		
		except Exception as e:
			print(f"gRPC service discovery error: {e}")
			return []
	
	def _determine_method_type(self, method_descriptor: Any) -> GRPCMethodType:
		"""Determine gRPC method type from descriptor."""
		if hasattr(method_descriptor, 'client_streaming') and hasattr(method_descriptor, 'server_streaming'):
			if method_descriptor.client_streaming and method_descriptor.server_streaming:
				return GRPCMethodType.BIDIRECTIONAL_STREAMING
			elif method_descriptor.client_streaming:
				return GRPCMethodType.CLIENT_STREAMING
			elif method_descriptor.server_streaming:
				return GRPCMethodType.SERVER_STREAMING
		
		return GRPCMethodType.UNARY


class GRPCLoadBalancer:
	"""Load balancer for gRPC services."""
	
	def __init__(self, circuit_breaker_manager: Any):
		self.circuit_breaker_manager = circuit_breaker_manager
		self.service_endpoints: Dict[str, List[Dict[str, Any]]] = {}
		self.endpoint_metrics: Dict[str, GRPCCallMetrics] = {}
	
	async def add_service_endpoint(
		self,
		service_name: str,
		endpoint: str,
		port: int,
		weight: float = 1.0,
		metadata: Optional[Dict[str, Any]] = None
	) -> None:
		"""Add endpoint for gRPC service load balancing."""
		if service_name not in self.service_endpoints:
			self.service_endpoints[service_name] = []
		
		endpoint_info = {
			"endpoint": endpoint,
			"port": port,
			"weight": weight,
			"metadata": metadata or {},
			"added_at": datetime.now(timezone.utc)
		}
		
		self.service_endpoints[service_name].append(endpoint_info)
		
		# Initialize metrics
		endpoint_key = f"{endpoint}:{port}"
		if endpoint_key not in self.endpoint_metrics:
			self.endpoint_metrics[endpoint_key] = GRPCCallMetrics(
				service_name=service_name,
				method_name="*"
			)
	
	async def get_endpoint(
		self,
		service_name: str,
		method_name: Optional[str] = None,
		load_balancing_strategy: str = "round_robin"
	) -> Optional[Dict[str, Any]]:
		"""Get endpoint for gRPC service call."""
		endpoints = self.service_endpoints.get(service_name, [])
		if not endpoints:
			return None
		
		# Filter healthy endpoints
		healthy_endpoints = []
		for endpoint_info in endpoints:
			endpoint_key = f"{endpoint_info['endpoint']}:{endpoint_info['port']}"
			circuit_breaker = self.circuit_breaker_manager.get_circuit_breaker(
				f"grpc:{service_name}:{endpoint_key}"
			)
			
			# Check if circuit breaker allows calls
			if circuit_breaker.state.value != "open":
				healthy_endpoints.append(endpoint_info)
		
		if not healthy_endpoints:
			return None
		
		# Apply load balancing strategy
		if load_balancing_strategy == "round_robin":
			# Simple round-robin implementation
			import random
			return random.choice(healthy_endpoints)
		
		elif load_balancing_strategy == "weighted":
			# Weighted random selection
			total_weight = sum(ep["weight"] for ep in healthy_endpoints)
			if total_weight == 0:
				return healthy_endpoints[0]
			
			import random
			r = random.uniform(0, total_weight)
			current_weight = 0
			
			for endpoint_info in healthy_endpoints:
				current_weight += endpoint_info["weight"]
				if r <= current_weight:
					return endpoint_info
			
			return healthy_endpoints[-1]
		
		elif load_balancing_strategy == "least_connections":
			# Select endpoint with least active connections
			# This would require connection tracking
			return min(healthy_endpoints, key=lambda ep: ep.get("active_connections", 0))
		
		else:
			return healthy_endpoints[0]
	
	async def record_call_metrics(
		self,
		service_name: str,
		method_name: str,
		endpoint: str,
		port: int,
		duration_ms: float,
		success: bool,
		streaming_messages: int = 0
	) -> None:
		"""Record gRPC call metrics."""
		endpoint_key = f"{endpoint}:{port}"
		
		if endpoint_key not in self.endpoint_metrics:
			self.endpoint_metrics[endpoint_key] = GRPCCallMetrics(
				service_name=service_name,
				method_name=method_name
			)
		
		metrics = self.endpoint_metrics[endpoint_key]
		metrics.request_count += 1
		
		if not success:
			metrics.error_count += 1
		
		metrics.total_duration_ms += duration_ms
		metrics.min_duration_ms = min(metrics.min_duration_ms, duration_ms)
		metrics.max_duration_ms = max(metrics.max_duration_ms, duration_ms)
		metrics.streaming_messages += streaming_messages
		metrics.last_call_time = datetime.now(timezone.utc)


class GRPCInterceptor:
	"""gRPC interceptor for service mesh integration."""
	
	def __init__(
		self,
		load_balancer: GRPCLoadBalancer,
		circuit_breaker_manager: Any,
		cert_manager: Optional[TLSCertificateManager] = None
	):
		self.load_balancer = load_balancer
		self.circuit_breaker_manager = circuit_breaker_manager
		self.cert_manager = cert_manager
	
	async def intercept_unary_unary(
		self,
		continuation: Callable,
		client_call_details: grpc.ClientCallDetails,
		request: Any
	) -> Any:
		"""Intercept unary-unary gRPC calls."""
		service_name, method_name = self._parse_method(client_call_details.method)
		
		# Get endpoint from load balancer
		endpoint_info = await self.load_balancer.get_endpoint(service_name, method_name)
		if not endpoint_info:
			raise grpc.RpcError(f"No healthy endpoints for service: {service_name}")
		
		# Update call details with selected endpoint
		endpoint = endpoint_info["endpoint"]
		port = endpoint_info["port"]
		
		# Create new call details with selected endpoint
		new_call_details = client_call_details._replace(
			target=f"{endpoint}:{port}"
		)
		
		# Get circuit breaker
		circuit_breaker = self.circuit_breaker_manager.get_circuit_breaker(
			f"grpc:{service_name}:{endpoint}:{port}"
		)
		
		start_time = time.time()
		success = False
		
		try:
			# Execute call through circuit breaker
			response = await circuit_breaker.call(
				continuation,
				new_call_details,
				request
			)
			success = True
			return response
		
		except Exception as e:
			success = False
			raise e
		
		finally:
			# Record metrics
			duration_ms = (time.time() - start_time) * 1000
			await self.load_balancer.record_call_metrics(
				service_name=service_name,
				method_name=method_name,
				endpoint=endpoint,
				port=port,
				duration_ms=duration_ms,
				success=success
			)
	
	async def intercept_stream(
		self,
		continuation: Callable,
		client_call_details: grpc.ClientCallDetails,
		request_iterator: AsyncIterator
	) -> AsyncIterator:
		"""Intercept streaming gRPC calls."""
		service_name, method_name = self._parse_method(client_call_details.method)
		
		# Get endpoint from load balancer
		endpoint_info = await self.load_balancer.get_endpoint(service_name, method_name)
		if not endpoint_info:
			raise grpc.RpcError(f"No healthy endpoints for service: {service_name}")
		
		endpoint = endpoint_info["endpoint"]
		port = endpoint_info["port"]
		
		# Update call details
		new_call_details = client_call_details._replace(
			target=f"{endpoint}:{port}"
		)
		
		start_time = time.time()
		message_count = 0
		success = False
		
		try:
			# Execute streaming call
			response_iterator = continuation(new_call_details, request_iterator)
			
			async for response in response_iterator:
				message_count += 1
				yield response
			
			success = True
		
		except Exception as e:
			success = False
			raise e
		
		finally:
			# Record streaming metrics
			duration_ms = (time.time() - start_time) * 1000
			await self.load_balancer.record_call_metrics(
				service_name=service_name,
				method_name=method_name,
				endpoint=endpoint,
				port=port,
				duration_ms=duration_ms,
				success=success,
				streaming_messages=message_count
			)
	
	def _parse_method(self, method: str) -> Tuple[str, str]:
		"""Parse gRPC method path."""
		# Method format: /package.Service/Method
		parts = method.strip('/').split('/')
		if len(parts) != 2:
			return "unknown", "unknown"
		
		service_path, method_name = parts
		service_name = service_path.split('.')[-1]  # Get last part as service name
		
		return service_name, method_name


class GRPCServiceMeshProxy:
	"""gRPC service mesh proxy with full protocol support."""
	
	def __init__(
		self,
		db_session: AsyncSession,
		cert_manager: TLSCertificateManager,
		circuit_breaker_manager: Any,
		listen_port: int = 50051
	):
		self.db_session = db_session
		self.cert_manager = cert_manager
		self.circuit_breaker_manager = circuit_breaker_manager
		self.listen_port = listen_port
		
		# Components
		self.health_checker = GRPCHealthChecker(db_session)
		self.service_discovery = GRPCServiceDiscovery()
		self.load_balancer = GRPCLoadBalancer(circuit_breaker_manager)
		self.interceptor = GRPCInterceptor(
			self.load_balancer,
			circuit_breaker_manager,
			cert_manager
		)
		
		# gRPC server
		self.server: Optional[aio.Server] = None
		self.registered_services: Dict[str, Any] = {}
	
	async def start_proxy(self) -> None:
		"""Start gRPC service mesh proxy."""
		# Create gRPC server
		self.server = aio.server()
		
		# Add health service
		health_servicer = self._create_health_servicer()
		health_pb2_grpc.add_HealthServicer_to_server(health_servicer, self.server)
		
		# Enable reflection
		service_names = [
			health_pb2.DESCRIPTOR.services_by_name['Health'].full_name,
		]
		reflection.enable_server_reflection(service_names, self.server)
		
		# Setup TLS if available
		server_credentials = None
		cert_bundle = await self.cert_manager.get_certificate_bundle("grpc-proxy")
		if cert_bundle:
			server_credentials = grpc.ssl_server_credentials([
				(cert_bundle.private_key.encode(), cert_bundle.certificate.encode())
			])
		
		# Start server
		if server_credentials:
			self.server.add_secure_port(f"[::]:{self.listen_port}", server_credentials)
		else:
			self.server.add_insecure_port(f"[::]:{self.listen_port}")
		
		await self.server.start()
		print(f"gRPC service mesh proxy started on port {self.listen_port}")
		
		# Start background tasks
		asyncio.create_task(self._service_discovery_loop())
		asyncio.create_task(self._health_monitoring_loop())
	
	async def stop_proxy(self) -> None:
		"""Stop gRPC service mesh proxy."""
		if self.server:
			await self.server.stop(grace=5.0)
		
		await self.health_checker.close_connections()
	
	def _create_health_servicer(self) -> Any:
		"""Create health check servicer."""
		class HealthServicer(health_pb2_grpc.HealthServicer):
			def __init__(self, proxy):
				self.proxy = proxy
			
			async def Check(self, request, context):
				# Check if requested service is available
				service = request.service
				if not service:
					# Check overall proxy health
					status = health_pb2.HealthCheckResponse.SERVING
				else:
					# Check specific service health
					endpoints = self.proxy.load_balancer.service_endpoints.get(service, [])
					status = health_pb2.HealthCheckResponse.SERVING if endpoints else health_pb2.HealthCheckResponse.NOT_SERVING
				
				return health_pb2.HealthCheckResponse(status=status)
			
			async def Watch(self, request, context):
				# Implement health status watching
				service = request.service
				while True:
					# Check service health periodically
					await asyncio.sleep(10)
					
					endpoints = self.proxy.load_balancer.service_endpoints.get(service, [])
					status = health_pb2.HealthCheckResponse.SERVING if endpoints else health_pb2.HealthCheckResponse.NOT_SERVING
					
					yield health_pb2.HealthCheckResponse(status=status)
		
		return HealthServicer(self)
	
	async def register_service(
		self,
		service_name: str,
		endpoints: List[Dict[str, Any]]
	) -> None:
		"""Register gRPC service with endpoints."""
		for endpoint_info in endpoints:
			await self.load_balancer.add_service_endpoint(
				service_name=service_name,
				endpoint=endpoint_info["endpoint"],
				port=endpoint_info["port"],
				weight=endpoint_info.get("weight", 1.0),
				metadata=endpoint_info.get("metadata", {})
			)
	
	async def _service_discovery_loop(self) -> None:
		"""Background service discovery loop."""
		while True:
			try:
				await asyncio.sleep(60)  # Discover services every minute
				
				# Query database for gRPC services
				result = await self.db_session.execute(
					select(SMEndpoint).where(SMEndpoint.protocol == "grpc")
				)
				
				for endpoint in result.scalars():
					# Discover services on this endpoint
					services = await self.service_discovery.discover_services(
						endpoint.host,
						endpoint.port,
						use_tls=endpoint.tls_enabled
					)
					
					# Register discovered services
					for service_info in services:
						await self.register_service(
							service_info.service_name,
							[{
								"endpoint": service_info.endpoint,
								"port": service_info.port,
								"weight": 1.0,
								"metadata": service_info.metadata
							}]
						)
			
			except Exception as e:
				print(f"Error in gRPC service discovery loop: {e}")
	
	async def _health_monitoring_loop(self) -> None:
		"""Background health monitoring loop."""
		while True:
			try:
				await asyncio.sleep(30)  # Health check every 30 seconds
				
				# Check health of all registered services
				for service_name, endpoints in self.load_balancer.service_endpoints.items():
					for endpoint_info in endpoints:
						status = await self.health_checker.check_service_health(
							endpoint_info["endpoint"],
							endpoint_info["port"],
							service_name
						)
						
						# Update endpoint health status based on result
						endpoint_info["health_status"] = status.value
						endpoint_info["last_health_check"] = datetime.now(timezone.utc)
			
			except Exception as e:
				print(f"Error in gRPC health monitoring loop: {e}")
	
	async def get_metrics(self) -> Dict[str, Any]:
		"""Get gRPC proxy metrics."""
		metrics = {
			"proxy_status": "running" if self.server else "stopped",
			"listen_port": self.listen_port,
			"registered_services": len(self.load_balancer.service_endpoints),
			"total_endpoints": sum(
				len(endpoints) for endpoints in self.load_balancer.service_endpoints.values()
			),
			"services": {},
			"endpoint_metrics": {}
		}
		
		# Service metrics
		for service_name, endpoints in self.load_balancer.service_endpoints.items():
			healthy_endpoints = [ep for ep in endpoints if ep.get("health_status") == "SERVING"]
			metrics["services"][service_name] = {
				"total_endpoints": len(endpoints),
				"healthy_endpoints": len(healthy_endpoints),
				"health_percentage": len(healthy_endpoints) / len(endpoints) * 100 if endpoints else 0
			}
		
		# Endpoint call metrics
		for endpoint_key, call_metrics in self.load_balancer.endpoint_metrics.items():
			avg_duration = (
				call_metrics.total_duration_ms / call_metrics.request_count
				if call_metrics.request_count > 0 else 0
			)
			error_rate = (
				call_metrics.error_count / call_metrics.request_count * 100
				if call_metrics.request_count > 0 else 0
			)
			
			metrics["endpoint_metrics"][endpoint_key] = {
				"request_count": call_metrics.request_count,
				"error_count": call_metrics.error_count,
				"error_rate_percentage": error_rate,
				"avg_duration_ms": avg_duration,
				"min_duration_ms": call_metrics.min_duration_ms if call_metrics.min_duration_ms != float('inf') else 0,
				"max_duration_ms": call_metrics.max_duration_ms,
				"streaming_messages": call_metrics.streaming_messages,
				"last_call_time": call_metrics.last_call_time.isoformat() if call_metrics.last_call_time else None
			}
		
		return metrics