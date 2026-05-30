#!/usr/bin/env python3
"""
APG Intelligent Gateway - Adapter-Backed Service Layer

Runtime service implementation for APG platform integration. Generated
applications should use gateway_runtime.ApigService for dependency-light
guardrail decisions before binding this adapter-backed runtime.

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, AsyncContextManager
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from models import (
    AgGatewayConfig, AgApiRoute, AgPolicy, AgUpstreamService,
    AgTrafficMetrics, AgSecurityEvent, AgWasmModule, AgHttpRequest, 
    AgHttpResponse, EnvironmentType, PolicyType, ThreatLevel
)

from apg_clients import (
    APGAuthRBACClient, APGMonitoringClient, APGConfigurationClient,
    APGAIOrchestrationClient, APGMessageQueueClient, APGAuditComplianceClient,
    APGServiceConfig, AuthResult
)

from edge_engine_production import (
    ProductionEdgeEngine, EdgeProcessingResult
)

from wasm_runtime import (
    ProductionWASMRuntime, WASMExecutionContext, WASMExecutionResult
)

from ollama_client import (
    ProductionOllamaClient, OllamaConfig, GenerationRequest
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class APGServiceConnections:
    """Container for APG service connections."""
    auth_rbac: Optional[APGAuthRBACClient] = None
    monitoring: Optional[APGMonitoringClient] = None
    configuration: Optional[APGConfigurationClient] = None
    ai_orchestration: Optional[APGAIOrchestrationClient] = None
    message_queue: Optional[APGMessageQueueClient] = None
    audit_compliance: Optional[APGAuditComplianceClient] = None


@dataclass
class ServiceMetrics:
    """Service performance metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    cache_hits: int = 0
    security_blocks: int = 0
    wasm_executions: int = 0
    policy_generations: int = 0
    uptime_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ProductionAPGIntelligentGatewayService:
    """
    Adapter-backed APG Intelligent Gateway Service.

    This class coordinates APG service clients, WebAssembly runtime surfaces,
    optional AI features, and observability adapters for runtime deployments.
    
    Features:
    - Real APG platform service integrations
    - Production WebAssembly runtime
    - AI-powered policy generation with Ollama
    - Edge computing with intelligent caching
    - Multi-layer security analysis
    - Comprehensive metrics and monitoring
    - Circuit breaker patterns for resilience
    """
    
    def __init__(self, tenant_id: str, user_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize APG Intelligent Gateway Service.
        
        Args:
            tenant_id: APG tenant identifier
            user_id: User identifier for audit trails
            config: Optional service configuration
        """
        assert isinstance(tenant_id, str) and tenant_id, "tenant_id must be non-empty string"
        assert isinstance(user_id, str) and user_id, "user_id must be non-empty string"
        
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.config = config or {}
        self.initialized = False
        
        # Service state
        self.gateway_configs: Dict[str, AgGatewayConfig] = {}
        self.policies: Dict[str, AgPolicy] = {}
        self.traffic_metrics: Dict[str, AgTrafficMetrics] = {}
        self.security_events: List[AgSecurityEvent] = []
        self.wasm_modules: Dict[str, AgWasmModule] = {}
        
        # APG service connections
        self.apg_services = APGServiceConnections()
        
        # Core components
        self.edge_engine: Optional[ProductionEdgeEngine] = None
        self.wasm_runtime: Optional[ProductionWASMRuntime] = None
        self.ollama_client: Optional[ProductionOllamaClient] = None
        
        # Service metrics
        self.metrics = ServiceMetrics()
        
        # Service configuration
        self.service_config = {
            'apg_base_url': self.config.get('apg_base_url', 'http://localhost:8000'),
            'apg_api_key': self.config.get('apg_api_key', 'demo-api-key'),
            'redis_url': self.config.get('redis_url', 'redis://localhost:6379'),
            'ollama_url': self.config.get('ollama_url', 'http://localhost:11434'),
            'edge_location': self.config.get('edge_location', 'default'),
            'enable_wasm': self.config.get('enable_wasm', True),
            'enable_ai': self.config.get('enable_ai', True),
            'max_wasm_modules': self.config.get('max_wasm_modules', 50),
            'circuit_breaker_threshold': self.config.get('circuit_breaker_threshold', 5),
            'request_timeout': self.config.get('request_timeout', 30)
        }
        
        logger.info(f"APIG Service initialized for tenant {tenant_id} by user {user_id}")
    
    async def initialize(self) -> None:
        """
        Initialize all service components and APG integrations.
        
        Raises:
            RuntimeError: If initialization fails
        """
        if self.initialized:
            logger.warning("Service already initialized")
            return
        
        start_time = time.perf_counter()
        
        try:
            await self._log_info("Initializing APG Intelligent Gateway Service...")
            
            # Initialize APG platform connections
            await self._initialize_apg_connections()
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Initialize monitoring and metrics
            await self._initialize_monitoring()
            
            # Load initial configuration
            await self._load_initial_configuration()
            
            self.initialized = True
            initialization_time = (time.perf_counter() - start_time) * 1000
            
            await self._log_info(f"APIG Service initialized successfully in {initialization_time:.2f}ms")
            
            # Send initialization event
            await self._audit_log('service_initialized', {
                'initialization_time_ms': initialization_time,
                'apg_integrations': 6,
                'components_loaded': ['edge_engine', 'wasm_runtime', 'ollama_client']
            })
            
        except Exception as e:
            await self._log_error(f"Service initialization failed: {str(e)}")
            raise RuntimeError(f"APIG Service initialization failed: {str(e)}")
    
    async def _initialize_apg_connections(self) -> None:
        """Initialize connections to all APG platform services."""
        apg_config = APGServiceConfig(
            base_url=self.service_config['apg_base_url'],
            api_key=self.service_config['apg_api_key'],
            timeout=self.service_config['request_timeout'],
            circuit_breaker_threshold=self.service_config['circuit_breaker_threshold']
        )
        
        # Initialize all APG service clients
        self.apg_services.auth_rbac = APGAuthRBACClient(apg_config, self.tenant_id)
        await self.apg_services.auth_rbac.initialize()
        
        self.apg_services.monitoring = APGMonitoringClient(apg_config, self.tenant_id)
        await self.apg_services.monitoring.initialize()
        
        self.apg_services.configuration = APGConfigurationClient(apg_config, self.tenant_id)
        await self.apg_services.configuration.initialize()
        
        self.apg_services.ai_orchestration = APGAIOrchestrationClient(apg_config, self.tenant_id)
        await self.apg_services.ai_orchestration.initialize()
        
        self.apg_services.message_queue = APGMessageQueueClient(apg_config, self.tenant_id)
        await self.apg_services.message_queue.initialize()
        
        self.apg_services.audit_compliance = APGAuditComplianceClient(apg_config, self.tenant_id)
        await self.apg_services.audit_compliance.initialize()
        
        await self._log_info("✓ APG platform integrations established")
    
    async def _initialize_core_components(self) -> None:
        """Initialize core gateway components."""
        # Initialize edge engine
        self.edge_engine = ProductionEdgeEngine(
            self.tenant_id,
            self.service_config['edge_location']
        )
        await self.edge_engine.initialize()
        
        # Initialize WASM runtime if enabled
        if self.service_config['enable_wasm']:
            self.wasm_runtime = ProductionWASMRuntime(
                self.tenant_id,
                self.service_config['max_wasm_modules']
            )
            await self.wasm_runtime.initialize()
        
        # Initialize Ollama client if AI enabled
        if self.service_config['enable_ai']:
            ollama_config = OllamaConfig(
                base_url=self.service_config['ollama_url']
            )
            self.ollama_client = ProductionOllamaClient(ollama_config, self.tenant_id)
            await self.ollama_client.initialize()
        
        await self._log_info("✓ Core components initialized")
    
    async def _initialize_monitoring(self) -> None:
        """Initialize monitoring and metrics collection."""
        # Create base monitoring alerts
        if self.apg_services.monitoring:
            await self.apg_services.monitoring.create_alert(
                "apig_high_error_rate",
                "error_rate > 0.05",
                "medium"
            )
            
            await self.apg_services.monitoring.create_alert(
                "apig_high_response_time",
                "avg_response_time > 1000",
                "high"
            )
        
        await self._log_info("✓ Monitoring systems initialized")
    
    async def _load_initial_configuration(self) -> None:
        """Load initial configuration from APG configuration service."""
        try:
            if self.apg_services.configuration:
                config = await self.apg_services.configuration.get_configuration('apig_defaults')
                if config:
                    # Apply configuration overrides
                    self.service_config.update(config)
            
            await self._log_info("✓ Initial configuration loaded")
            
        except Exception as e:
            await self._log_warning(f"Could not load initial configuration: {str(e)}")
    
    async def create_gateway(self, gateway_config: AgGatewayConfig) -> AgGatewayConfig:
        """
        Create new gateway configuration with APG integration.
        
        Args:
            gateway_config: Gateway configuration
            
        Returns:
            AgGatewayConfig: Created gateway configuration
        """
        assert isinstance(gateway_config, AgGatewayConfig), "gateway_config must be AgGatewayConfig instance"
        assert self.initialized, "Service must be initialized"
        
        start_time = time.perf_counter()
        
        try:
            # Store configuration
            self.gateway_configs[gateway_config.id] = gateway_config
            
            # Initialize traffic metrics
            self.traffic_metrics[gateway_config.id] = AgTrafficMetrics(
                gateway_id=gateway_config.id,
                tenant_id=self.tenant_id
            )
            
            creation_time = (time.perf_counter() - start_time) * 1000
            await self._log_info(f"Gateway {gateway_config.name} created in {creation_time:.2f}ms")
            
            # Audit log
            await self._audit_log('gateway_created', {
                'gateway_id': gateway_config.id,
                'gateway_name': gateway_config.name,
                'environment': gateway_config.environment.value,
                'creation_time_ms': creation_time
            })
            
            return gateway_config
            
        except Exception as e:
            await self._log_error(f"Gateway creation failed: {str(e)}")
            raise RuntimeError(f"Failed to create gateway: {str(e)}")
    
    async def process_request(
        self, 
        request: AgHttpRequest, 
        gateway_id: Optional[str] = None
    ) -> EdgeProcessingResult:
        """
        Process HTTP request through the intelligent gateway.
        
        Args:
            request: HTTP request to process
            gateway_id: Optional gateway identifier
            
        Returns:
            EdgeProcessingResult: Complete processing result
        """
        assert isinstance(request, AgHttpRequest), "request must be AgHttpRequest instance"
        assert self.initialized, "Service must be initialized"
        
        start_time = time.perf_counter()
        self.metrics.total_requests += 1
        
        try:
            # Find applicable gateway
            gateway = await self._find_gateway_for_request(request, gateway_id)
            if not gateway:
                raise ValueError("No applicable gateway found for request")
            
            # Authentication and authorization
            auth_result = await self._authenticate_request(request)
            if not auth_result.authenticated:
                return await self._create_auth_error_response(request, "Authentication failed")
            
            # Find matching route
            route = await self._find_matching_route(request, gateway)
            if not route:
                return await self._create_not_found_response(request, "No matching route")
            
            # Process through edge engine
            processing_result = await self.edge_engine.process_request(
                request, 
                route.upstream_services
            )
            
            # Update metrics
            await self._update_request_metrics(processing_result, gateway.id)
            
            self.metrics.successful_requests += 1
            total_time = (time.perf_counter() - start_time) * 1000
            self.metrics.total_response_time += total_time
            
            return processing_result
            
        except Exception as e:
            self.metrics.failed_requests += 1
            total_time = (time.perf_counter() - start_time) * 1000
            
            await self._log_error(f"Request processing failed: {str(e)}")
            
            # Create error response
            error_response = AgHttpResponse(
                request_id=request.id,
                status_code=500,
                headers={'X-Error': 'processing_failed'},
                body=json.dumps({'error': 'Internal processing error'}).encode()
            )
            
            return EdgeProcessingResult(
                response=error_response,
                cache_hit=False,
                processing_time_ms=total_time,
                metadata={'error': str(e)}
            )
    
    async def create_policy_from_natural_language(
        self, 
        description: str, 
        target_routes: Optional[List[str]] = None
    ) -> AgPolicy:
        """
        Create gateway policy from natural language description using AI.
        
        Args:
            description: Natural language policy description
            target_routes: Optional target routes for policy
            
        Returns:
            AgPolicy: Generated policy configuration
        """
        assert isinstance(description, str) and description.strip(), "description must be non-empty string"
        assert self.initialized, "Service must be initialized"
        
        start_time = time.perf_counter()
        
        try:
            await self._log_info(f"Generating policy from natural language: '{description[:50]}...'")
            
            # Create generation request for Ollama
            generation_request = GenerationRequest(
                model="llama3.2:latest",
                prompt=await self._create_policy_generation_prompt(description, target_routes),
                system="You are an expert API gateway policy generator. Generate technical policy configurations from natural language descriptions.",
                options={
                    'temperature': 0.1,  # Low temperature for consistent results
                    'top_p': 0.9,
                    'max_tokens': 1000
                }
            )
            
            # Generate policy using AI
            ai_response = await self.ollama_client.generate(generation_request)
            
            # Parse AI response into policy configuration
            policy_config = await self._parse_ai_policy_response(ai_response.response)
            
            # Create policy object
            policy = AgPolicy(
                name=policy_config.get('name', 'AI Generated Policy'),
                type=PolicyType(policy_config.get('type', 'security')),
                configuration=policy_config.get('configuration', {}),
                conditions=policy_config.get('conditions', []),
                natural_language_description=description,
                created_by=self.user_id,
                tenant_id=self.tenant_id,
                priority=policy_config.get('priority', 1000)
            )
            
            # Store policy
            self.policies[policy.id] = policy
            self.metrics.policy_generations += 1
            
            generation_time = (time.perf_counter() - start_time) * 1000
            await self._log_info(f"Policy generated successfully in {generation_time:.2f}ms: {policy.name}")
            
            return policy
            
        except Exception as e:
            await self._log_error(f"AI policy generation failed: {str(e)}")
            raise RuntimeError(f"Policy generation failed: {str(e)}")
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status and health information."""
        try:
            uptime = (datetime.now(timezone.utc) - self.metrics.uptime_start).total_seconds()
            
            # Get component statuses
            apg_status = await self._get_apg_services_status()
            component_status = await self._get_component_status()
            
            # Calculate performance metrics
            avg_response_time = (
                self.metrics.total_response_time / self.metrics.successful_requests
                if self.metrics.successful_requests > 0 else 0.0
            )
            
            success_rate = (
                self.metrics.successful_requests / self.metrics.total_requests
                if self.metrics.total_requests > 0 else 0.0
            )
            
            return {
                'service': {
                    'status': 'healthy' if self.initialized else 'initializing',
                    'initialized': self.initialized,
                    'tenant_id': self.tenant_id,
                    'uptime_seconds': uptime,
                    'version': '1.0.0'
                },
                'performance': {
                    'total_requests': self.metrics.total_requests,
                    'successful_requests': self.metrics.successful_requests,
                    'failed_requests': self.metrics.failed_requests,
                    'success_rate': success_rate,
                    'average_response_time_ms': avg_response_time,
                    'cache_hits': self.metrics.cache_hits,
                    'security_blocks': self.metrics.security_blocks,
                    'wasm_executions': self.metrics.wasm_executions,
                    'policy_generations': self.metrics.policy_generations
                },
                'resources': {
                    'gateways_configured': len(self.gateway_configs),
                    'policies_active': len(self.policies),
                    'wasm_modules_loaded': len(self.wasm_modules),
                    'security_events': len(self.security_events)
                },
                'apg_integrations': apg_status,
                'components': component_status
            }
            
        except Exception as e:
            await self._log_error(f"Failed to get service status: {str(e)}")
            return {
                'service': {'status': 'error', 'error': str(e)},
                'initialized': self.initialized
            }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the service and cleanup all resources."""
        try:
            await self._log_info("Shutting down APIG Service...")
            
            # Cleanup components
            if self.edge_engine:
                await self.edge_engine.cleanup()
            
            if self.wasm_runtime:
                await self.wasm_runtime.cleanup()
            
            if self.ollama_client:
                await self.ollama_client.close()
            
            # Close APG service connections
            await self._close_apg_connections()
            
            self.initialized = False
            
            await self._log_info("APIG Service shutdown completed")
            
        except Exception as e:
            await self._log_error(f"Service shutdown error: {str(e)}")
    
    # Private implementation methods
    
    async def _find_gateway_for_request(
        self, 
        request: AgHttpRequest, 
        gateway_id: Optional[str]
    ) -> Optional[AgGatewayConfig]:
        """Find appropriate gateway for request."""
        if gateway_id and gateway_id in self.gateway_configs:
            return self.gateway_configs[gateway_id]
        
        # Find first available gateway (simple selection)
        for gateway in self.gateway_configs.values():
            if gateway.tenant_id == self.tenant_id:
                return gateway
        
        return None
    
    async def _authenticate_request(self, request: AgHttpRequest) -> AuthResult:
        """Authenticate request using APG auth service."""
        try:
            if self.apg_services.auth_rbac:
                # Extract token from Authorization header
                auth_header = request.headers.get('authorization', '')
                if auth_header.startswith('Bearer '):
                    token = auth_header[7:]
                    return await self.apg_services.auth_rbac.authenticate_token(token)
            
            # Default to authenticated for demo
            return AuthResult(
                authenticated=True,
                user_id='anonymous',
                tenant_id=self.tenant_id,
                roles=['user']
            )
            
        except Exception as e:
            await self._log_error(f"Authentication error: {str(e)}")
            return AuthResult(authenticated=False)
    
    async def _find_matching_route(
        self, 
        request: AgHttpRequest, 
        gateway: AgGatewayConfig
    ) -> Optional[AgApiRoute]:
        """Find matching route for request."""
        for route in gateway.routes:
            if self._route_matches_request(route, request):
                return route
        return None
    
    def _route_matches_request(self, route: AgApiRoute, request: AgHttpRequest) -> bool:
        """Check if route matches request."""
        # Simple path matching (would use more sophisticated matching in production)
        if route.method != request.method:
            return False
        
        # Exact match or prefix match with wildcard
        if route.path == request.path:
            return True
        
        if route.path.endswith('/*') and request.path.startswith(route.path[:-1]):
            return True
        
        return False
    
    async def _create_auth_error_response(
        self, 
        request: AgHttpRequest, 
        message: str
    ) -> EdgeProcessingResult:
        """Create authentication error response."""
        response = AgHttpResponse(
            request_id=request.id,
            status_code=401,
            headers={'WWW-Authenticate': 'Bearer'},
            body=json.dumps({'error': message}).encode()
        )
        
        return EdgeProcessingResult(
            response=response,
            cache_hit=False,
            processing_time_ms=0.1,
            metadata={'auth_error': True}
        )
    
    async def _create_not_found_response(
        self, 
        request: AgHttpRequest, 
        message: str
    ) -> EdgeProcessingResult:
        """Create not found error response."""
        response = AgHttpResponse(
            request_id=request.id,
            status_code=404,
            headers={'X-Error': 'not_found'},
            body=json.dumps({'error': message}).encode()
        )
        
        return EdgeProcessingResult(
            response=response,
            cache_hit=False,
            processing_time_ms=0.1,
            metadata={'not_found': True}
        )
    
    async def _update_request_metrics(
        self, 
        result: EdgeProcessingResult, 
        gateway_id: str
    ) -> None:
        """Update request metrics."""
        if result.cache_hit:
            self.metrics.cache_hits += 1
        
        if 'security_block' in result.metadata:
            self.metrics.security_blocks += 1
        
        # Update gateway-specific metrics
        if gateway_id in self.traffic_metrics:
            metrics = self.traffic_metrics[gateway_id]
            metrics.request_count += 1
            metrics.response_time_p50 = result.processing_time_ms  # Simplified
    
    async def _create_policy_generation_prompt(
        self, 
        description: str, 
        target_routes: Optional[List[str]]
    ) -> str:
        """Create prompt for AI policy generation."""
        routes_text = ""
        if target_routes:
            routes_text = f"\nTarget routes: {', '.join(target_routes)}"
        
        return f"""Generate an API gateway policy configuration from this natural language description:
"{description}"{routes_text}

Return a JSON configuration with these fields:
- name: Policy name
- type: Policy type (rate_limiting, authentication, authorization, security, caching)
- configuration: Policy-specific configuration parameters
- conditions: List of conditions when policy applies
- priority: Execution priority (1-10000, lower = higher priority)

Focus on practical, implementable configurations. Be specific with parameters."""
    
    async def _parse_ai_policy_response(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI response into policy configuration."""
        try:
            # Try to extract JSON from AI response
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback to simple parsing
                return {
                    'name': 'AI Generated Policy',
                    'type': 'security',
                    'configuration': {},
                    'conditions': [],
                    'priority': 1000
                }
        except Exception as e:
            await self._log_warning(f"Could not parse AI response: {str(e)}")
            return {
                'name': 'AI Generated Policy',
                'type': 'security',
                'configuration': {},
                'conditions': [],
                'priority': 1000
            }
    
    async def _get_apg_services_status(self) -> Dict[str, str]:
        """Get status of APG service connections."""
        status = {}
        
        if self.apg_services.auth_rbac:
            status['auth_rbac'] = self.apg_services.auth_rbac.status.value
        
        if self.apg_services.monitoring:
            status['monitoring'] = self.apg_services.monitoring.status.value
        
        if self.apg_services.configuration:
            status['configuration'] = self.apg_services.configuration.status.value
        
        if self.apg_services.ai_orchestration:
            status['ai_orchestration'] = self.apg_services.ai_orchestration.status.value
        
        if self.apg_services.message_queue:
            status['message_queue'] = self.apg_services.message_queue.status.value
        
        if self.apg_services.audit_compliance:
            status['audit_compliance'] = self.apg_services.audit_compliance.status.value
        
        return status
    
    async def _get_component_status(self) -> Dict[str, str]:
        """Get status of core components."""
        status = {}
        
        if self.edge_engine:
            status['edge_engine'] = 'initialized' if self.edge_engine.initialized else 'not_initialized'
        
        if self.wasm_runtime:
            status['wasm_runtime'] = 'initialized' if self.wasm_runtime.initialized else 'not_initialized'
        
        if self.ollama_client:
            status['ollama_client'] = 'connected'  # Assume connected if client exists
        
        return status
    
    async def _close_apg_connections(self) -> None:
        """Close all APG service connections."""
        try:
            if self.apg_services.auth_rbac:
                await self.apg_services.auth_rbac.close()
            
            if self.apg_services.monitoring:
                await self.apg_services.monitoring.close()
            
            if self.apg_services.configuration:
                await self.apg_services.configuration.close()
            
            if self.apg_services.ai_orchestration:
                await self.apg_services.ai_orchestration.close()
            
            if self.apg_services.message_queue:
                await self.apg_services.message_queue.close()
            
            if self.apg_services.audit_compliance:
                await self.apg_services.audit_compliance.close()
                
        except Exception as e:
            await self._log_error(f"Error closing APG connections: {str(e)}")
    
    # Logging methods
    
    async def _log_info(self, message: str) -> None:
        """Log info message."""
        timestamp = datetime.now(timezone.utc).isoformat()
        logger.info(f"[{timestamp}] APIG [{self.tenant_id}:{self.user_id}] {message}")
    
    async def _log_debug(self, message: str) -> None:
        """Log debug message."""
        timestamp = datetime.now(timezone.utc).isoformat()
        logger.debug(f"[{timestamp}] APIG [{self.tenant_id}:{self.user_id}] {message}")
    
    async def _log_warning(self, message: str) -> None:
        """Log warning message."""
        timestamp = datetime.now(timezone.utc).isoformat()
        logger.warning(f"[{timestamp}] APIG [{self.tenant_id}:{self.user_id}] {message}")
    
    async def _log_error(self, message: str) -> None:
        """Log error message."""
        timestamp = datetime.now(timezone.utc).isoformat()
        logger.error(f"[{timestamp}] APIG [{self.tenant_id}:{self.user_id}] {message}")
    
    async def _audit_log(self, event_type: str, details: Dict[str, Any]) -> None:
        """Send audit log to APG compliance system."""
        try:
            if self.apg_services.audit_compliance:
                await self.apg_services.audit_compliance.log_audit_event(
                    event_type=event_type,
                    user_id=self.user_id,
                    resource='apig_service',
                    action=event_type,
                    details=details
                )
        except Exception as e:
            await self._log_debug(f"Audit logging error: {str(e)}")


# Backward compatibility alias
APGIntelligentGatewayService = ProductionAPGIntelligentGatewayService

# Export main class
__all__ = ['ProductionAPGIntelligentGatewayService', 'APGIntelligentGatewayService']
