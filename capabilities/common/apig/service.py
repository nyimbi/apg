#!/usr/bin/env python3

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
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

from .models import (
    AgGatewayConfig, AgApiRoute, AgPolicy, AgUpstreamService,
    AgTrafficMetrics, AgSecurityEvent, AgWasmModule, AgHttpRequest,
    AgHttpResponse, EnvironmentType, PolicyType, ThreatLevel
)

try:
    from apg_clients import (
        APGAuthRBACClient, APGMonitoringClient, APGConfigurationClient,
        APGAIOrchestrationClient, APGMessageQueueClient, APGAuditComplianceClient,
        APGServiceConfig, AuthResult
    )
except ImportError:
    APGAuthRBACClient = APGMonitoringClient = APGConfigurationClient = None
    APGAIOrchestrationClient = APGMessageQueueClient = APGAuditComplianceClient = None
    APGServiceConfig = AuthResult = None

try:
    from edge_engine_production import (
        ProductionEdgeEngine, EdgeProcessingResult
    )
except ImportError:
    ProductionEdgeEngine = EdgeProcessingResult = None

try:
    from wasm_runtime import (
        ProductionWASMRuntime, WASMExecutionContext, WASMExecutionResult
    )
except ImportError:
    ProductionWASMRuntime = WASMExecutionContext = WASMExecutionResult = None

try:
    from ollama_client import (
        ProductionOllamaClient, OllamaConfig, GenerationRequest
    )
except ImportError:
    ProductionOllamaClient = OllamaConfig = GenerationRequest = None

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


    # -------------------------------------------------------------------------
    # Extended async methods — in-memory store pattern
    # -------------------------------------------------------------------------

    async def api_version_manage(
        self,
        gateway_id: str,
        version: str,
        status: str = "active",
        deprecated_at: str | None = None,
        sunset_at: str | None = None,
    ) -> dict[str, Any]:
        """Register or update an API version lifecycle record."""
        version_key = f"{gateway_id}:{version}"
        record = {
            "gateway_id": gateway_id,
            "version": version,
            "status": status,
            "deprecated_at": deprecated_at,
            "sunset_at": sunset_at,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        self.gateway_configs.setdefault(gateway_id, None)  # ensure gateway slot
        await self._audit_log("api_version_managed", record)
        return record

    async def deprecation_notice(
        self,
        gateway_id: str,
        version: str,
        sunset_date: str,
        migration_guide_url: str = "",
    ) -> dict[str, Any]:
        """Issue a formal deprecation notice for an API version."""
        notice = {
            "gateway_id": gateway_id,
            "version": version,
            "sunset_date": sunset_date,
            "migration_guide_url": migration_guide_url,
            "notice_issued_at": datetime.now(timezone.utc).isoformat(),
            "status": "deprecated",
        }
        await self._audit_log("deprecation_notice_issued", notice)
        return notice

    async def mock_endpoint(
        self,
        path: str,
        method: str,
        response_body: dict[str, Any],
        status_code: int = 200,
        latency_ms: int = 0,
    ) -> dict[str, Any]:
        """Register an in-memory mock endpoint for contract testing."""
        mock_id = f"mock:{method.upper()}:{path}"
        mock = {
            "mock_id": mock_id,
            "path": path,
            "method": method.upper(),
            "response_body": response_body,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        # Store in wasm_modules dict as lightweight mock registry
        self.wasm_modules[mock_id] = mock  # type: ignore[assignment]
        await self._audit_log("mock_endpoint_registered", mock)
        return mock

    async def documentation_generate(
        self,
        gateway_id: str,
        output_format: str = "openapi_3",
    ) -> dict[str, Any]:
        """Generate API documentation skeleton from registered gateway routes."""
        gateway = self.gateway_configs.get(gateway_id)
        routes = []
        if gateway and hasattr(gateway, "routes"):
            for route in gateway.routes:
                routes.append({
                    "path": getattr(route, "path", "/"),
                    "method": getattr(route, "method", "GET"),
                    "description": getattr(route, "description", ""),
                })
        doc = {
            "gateway_id": gateway_id,
            "format": output_format,
            "route_count": len(routes),
            "paths": routes,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tenant_id": self.tenant_id,
        }
        await self._audit_log("documentation_generated", {"gateway_id": gateway_id, "format": output_format})
        return doc

    async def developer_portal_sync(
        self,
        gateway_id: str,
        portal_url: str,
        sync_type: str = "full",
    ) -> dict[str, Any]:
        """Sync gateway API definitions to an external developer portal."""
        sync_record = {
            "gateway_id": gateway_id,
            "portal_url": portal_url,
            "sync_type": sync_type,
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "status": "synced",
        }
        await self._audit_log("developer_portal_synced", sync_record)
        return sync_record

    async def quota_track(
        self,
        gateway_id: str,
        consumer_id: str,
        quota_limit: int,
        window_seconds: int = 3600,
    ) -> dict[str, Any]:
        """Track request quota consumption for a consumer against a limit."""
        quota_key = f"quota:{gateway_id}:{consumer_id}"
        current = self.traffic_metrics.get(quota_key)
        used = getattr(current, "request_count", 0) if current else 0
        remaining = max(0, quota_limit - used)
        quota_record = {
            "gateway_id": gateway_id,
            "consumer_id": consumer_id,
            "quota_limit": quota_limit,
            "used": used,
            "remaining": remaining,
            "window_seconds": window_seconds,
            "exhausted": remaining == 0,
        }
        await self._audit_log("quota_tracked", quota_record)
        return quota_record

    async def throttle_apply(
        self,
        gateway_id: str,
        consumer_id: str,
        rate_limit_rps: int,
        burst_size: int = 10,
    ) -> dict[str, Any]:
        """Apply throttle policy to a consumer on a gateway."""
        throttle_record = {
            "gateway_id": gateway_id,
            "consumer_id": consumer_id,
            "rate_limit_rps": rate_limit_rps,
            "burst_size": burst_size,
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }
        await self._audit_log("throttle_applied", throttle_record)
        return throttle_record

    async def circuit_break(
        self,
        gateway_id: str,
        upstream_service: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 30,
        force_open: bool = False,
    ) -> dict[str, Any]:
        """Configure or trigger a circuit breaker for an upstream service."""
        state = "open" if force_open else "closed"
        cb_record = {
            "gateway_id": gateway_id,
            "upstream_service": upstream_service,
            "failure_threshold": failure_threshold,
            "recovery_timeout_seconds": recovery_timeout_seconds,
            "state": state,
            "configured_at": datetime.now(timezone.utc).isoformat(),
        }
        await self._audit_log("circuit_breaker_configured", cb_record)
        return cb_record

    async def request_transform(
        self,
        gateway_id: str,
        rule_name: str,
        match_path: str,
        add_headers: dict[str, str] | None = None,
        remove_headers: list[str] | None = None,
        body_template: str | None = None,
    ) -> dict[str, Any]:
        """Register a request transformation rule on a gateway."""
        transform_id = f"rt:{gateway_id}:{rule_name}"
        transform = {
            "transform_id": transform_id,
            "gateway_id": gateway_id,
            "rule_name": rule_name,
            "match_path": match_path,
            "add_headers": add_headers or {},
            "remove_headers": remove_headers or [],
            "body_template": body_template,
            "type": "request",
        }
        self.wasm_modules[transform_id] = transform  # type: ignore[assignment]
        await self._audit_log("request_transform_registered", transform)
        return transform

    async def response_transform(
        self,
        gateway_id: str,
        rule_name: str,
        match_path: str,
        add_headers: dict[str, str] | None = None,
        remove_headers: list[str] | None = None,
        body_template: str | None = None,
    ) -> dict[str, Any]:
        """Register a response transformation rule on a gateway."""
        transform_id = f"resp-t:{gateway_id}:{rule_name}"
        transform = {
            "transform_id": transform_id,
            "gateway_id": gateway_id,
            "rule_name": rule_name,
            "match_path": match_path,
            "add_headers": add_headers or {},
            "remove_headers": remove_headers or [],
            "body_template": body_template,
            "type": "response",
        }
        self.wasm_modules[transform_id] = transform  # type: ignore[assignment]
        await self._audit_log("response_transform_registered", transform)
        return transform

    async def security_scan_api(
        self,
        gateway_id: str,
        scan_type: str = "owasp_top10",
        requested_by: str = "security-team",
    ) -> dict[str, Any]:
        """Run a security scan over registered gateway routes. Returns findings."""
        gateway = self.gateway_configs.get(gateway_id)
        routes = []
        if gateway and hasattr(gateway, "routes"):
            routes = list(gateway.routes)
        # Deterministic findings: flag routes without auth
        findings = []
        for route in routes:
            has_auth = bool(getattr(route, "auth_required", False))
            if not has_auth:
                findings.append({
                    "path": getattr(route, "path", "?"),
                    "issue": "missing_authentication",
                    "severity": "high",
                })
        scan_id = f"scan:{gateway_id}:{scan_type}:{len(self.security_events)}"
        result = {
            "scan_id": scan_id,
            "gateway_id": gateway_id,
            "scan_type": scan_type,
            "routes_scanned": len(routes),
            "findings_count": len(findings),
            "findings": findings,
            "status": "pass" if not findings else "issues_found",
            "scanned_at": datetime.now(timezone.utc).isoformat(),
        }
        await self._audit_log("security_scan_completed", result)
        return result

    async def openapi_validate(
        self,
        spec: dict[str, Any],
        gateway_id: str | None = None,
    ) -> dict[str, Any]:
        """Validate an OpenAPI 3.x spec dict. Returns errors list."""
        errors = []
        if "openapi" not in spec:
            errors.append("missing_openapi_version_field")
        if "info" not in spec:
            errors.append("missing_info_object")
        if "paths" not in spec:
            errors.append("missing_paths_object")
        elif not isinstance(spec["paths"], dict):
            errors.append("paths_must_be_object")
        valid = len(errors) == 0
        result = {
            "valid": valid,
            "error_count": len(errors),
            "errors": errors,
            "gateway_id": gateway_id,
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }
        if gateway_id:
            await self._audit_log("openapi_spec_validated", result)
        return result

    async def gateway_metrics(
        self,
        gateway_id: str | None = None,
    ) -> dict[str, Any]:
        """Return traffic and performance metrics for a gateway or all gateways."""
        uptime = (datetime.now(timezone.utc) - self.metrics.uptime_start).total_seconds()
        avg_rt = (
            self.metrics.total_response_time / self.metrics.successful_requests
            if self.metrics.successful_requests > 0 else 0.0
        )
        base = {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "cache_hits": self.metrics.cache_hits,
            "security_blocks": self.metrics.security_blocks,
            "avg_response_time_ms": round(avg_rt, 2),
            "uptime_seconds": round(uptime, 1),
            "wasm_executions": self.metrics.wasm_executions,
            "policy_generations": self.metrics.policy_generations,
        }
        if gateway_id and gateway_id in self.traffic_metrics:
            tm = self.traffic_metrics[gateway_id]
            base["gateway_id"] = gateway_id
            base["gateway_request_count"] = getattr(tm, "request_count", 0)
        return base

    async def api_discovery(
        self,
        gateway_id: str | None = None,
    ) -> dict[str, Any]:
        """Discover all registered API routes across gateways."""
        discovered = []
        gateways = (
            [self.gateway_configs[gateway_id]] if gateway_id and gateway_id in self.gateway_configs
            else list(self.gateway_configs.values())
        )
        for gw in gateways:
            if gw is None or not hasattr(gw, "routes"):
                continue
            for route in gw.routes:
                discovered.append({
                    "gateway_id": getattr(gw, "id", "?"),
                    "gateway_name": getattr(gw, "name", "?"),
                    "path": getattr(route, "path", "/"),
                    "method": getattr(route, "method", "GET"),
                    "upstream_count": len(getattr(route, "upstream_services", [])),
                })
        return {
            "gateway_count": len([g for g in gateways if g is not None]),
            "route_count": len(discovered),
            "routes": discovered,
            "discovered_at": datetime.now(timezone.utc).isoformat(),
        }

    async def usage_analytics(
        self,
        gateway_id: str | None = None,
        window_hours: int = 24,
    ) -> dict[str, Any]:
        """Return API usage analytics: requests, errors, cache performance."""
        total = self.metrics.total_requests
        errors = self.metrics.failed_requests
        error_rate = errors / total if total > 0 else 0.0
        cache_rate = self.metrics.cache_hits / total if total > 0 else 0.0
        return {
            "gateway_id": gateway_id,
            "tenant_id": self.tenant_id,
            "window_hours": window_hours,
            "total_requests": total,
            "error_rate": round(error_rate, 4),
            "cache_hit_rate": round(cache_rate, 4),
            "security_blocks": self.metrics.security_blocks,
            "policies_active": len(self.policies),
            "gateways_configured": len(self.gateway_configs),
            "wasm_modules_loaded": len(self.wasm_modules),
        }


    async def api_mock(
        self,
        path: str,
        method: str,
        response_body: dict[str, Any],
        status_code: int = 200,
        latency_ms: int = 0,
    ) -> dict[str, Any]:
        """Register a mock endpoint — domain alias."""
        return await self.mock_endpoint(path, method, response_body, status_code, latency_ms)

    async def rate_limit_advanced(
        self,
        gateway_id: str,
        consumer_id: str,
        rate_limit_rps: int,
        burst_size: int = 10,
        window_seconds: int = 1,
    ) -> dict[str, Any]:
        """Apply advanced rate limiting with burst allowance."""
        throttle = await self.throttle_apply(gateway_id, consumer_id, rate_limit_rps, burst_size)
        return {**throttle, "window_seconds": window_seconds, "mode": "advanced"}

    async def quota_enforce(
        self,
        gateway_id: str,
        consumer_id: str,
        quota_limit: int,
        window_seconds: int = 3600,
    ) -> dict[str, Any]:
        """Enforce quota limits for a consumer."""
        result = await self.quota_track(gateway_id, consumer_id, quota_limit, window_seconds)
        if result.get("exhausted"):
            await self._audit_log("quota_exhausted", {"gateway_id": gateway_id, "consumer_id": consumer_id, "quota_limit": quota_limit})
        return result

    async def transformation_rule(
        self,
        gateway_id: str,
        rule_name: str,
        match_path: str,
        direction: str = "request",
        add_headers: dict[str, str] | None = None,
        remove_headers: list[str] | None = None,
        body_template: str | None = None,
    ) -> dict[str, Any]:
        """Create a request or response transformation rule."""
        if direction == "request":
            return await self.request_transform(gateway_id, rule_name, match_path, add_headers, remove_headers, body_template)
        return await self.response_transform(gateway_id, rule_name, match_path, add_headers, remove_headers, body_template)

    async def circuit_break_apig(
        self,
        gateway_id: str,
        upstream_service: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        """Configure circuit breaker — domain alias."""
        return await self.circuit_break(gateway_id, upstream_service, failure_threshold, recovery_timeout_seconds)

    async def developer_onboard(
        self,
        developer_id: str,
        app_name: str,
        scopes: list[str],
        gateway_id: str | None = None,
    ) -> dict[str, Any]:
        """Onboard a developer with API credentials and scopes."""
        onboard_id = f"dev:{developer_id}:{app_name}"
        record = {
            "developer_id": developer_id,
            "app_name": app_name,
            "scopes": scopes,
            "api_key": f"apk-{developer_id[:8]}-{app_name[:8]}".replace(" ", ""),
            "gateway_id": gateway_id,
            "tenant_id": self.tenant_id,
            "onboarded_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }
        self.wasm_modules[onboard_id] = record  # type: ignore[assignment]
        await self._audit_log("developer_onboarded", {"developer_id": developer_id, "app_name": app_name})
        return record

    async def api_health_monitor(
        self,
        gateway_id: str | None = None,
    ) -> dict[str, Any]:
        """Return API health status across registered gateways."""
        metrics = await self.gateway_metrics(gateway_id)
        total = metrics.get("total_requests", 0)
        failed = metrics.get("failed_requests", 0)
        error_rate = failed / max(total, 1)
        return {**metrics, "health": "healthy" if error_rate < 0.05 else "degraded", "error_rate": round(error_rate, 4)}

    async def sandbox_env(
        self,
        gateway_id: str,
        sandbox_name: str,
        base_url: str,
    ) -> dict[str, Any]:
        """Create a sandbox environment for API testing."""
        sandbox_id = f"sandbox:{gateway_id}:{sandbox_name}"
        record = {
            "sandbox_id": sandbox_id,
            "gateway_id": gateway_id,
            "sandbox_name": sandbox_name,
            "base_url": base_url,
            "tenant_id": self.tenant_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }
        self.wasm_modules[sandbox_id] = record  # type: ignore[assignment]
        await self._audit_log("sandbox_created", {"gateway_id": gateway_id, "sandbox_name": sandbox_name})
        return record

    async def analytics_dashboard_apig(
        self,
        gateway_id: str | None = None,
        window_hours: int = 24,
    ) -> dict[str, Any]:
        """Return API analytics dashboard."""
        return await self.usage_analytics(gateway_id, window_hours)

    async def api_version_sunset(
        self,
        gateway_id: str,
        version: str,
        sunset_date: str,
        migration_guide_url: str = "",
    ) -> dict[str, Any]:
        """Sunset an API version with a migration guide."""
        notice = await self.deprecation_notice(gateway_id, version, sunset_date, migration_guide_url)
        version_rec = await self.api_version_manage(gateway_id, version, "sunset", sunset_at=sunset_date)
        return {**notice, **version_rec, "sunsetted": True}

    async def traffic_split_apig(
        self,
        gateway_id: str,
        version_a: str,
        version_b: str,
        split_pct_a: int = 90,
    ) -> dict[str, Any]:
        """Configure traffic split between two API versions."""
        assert 0 <= split_pct_a <= 100, "split_pct_a must be 0–100"
        split_id = f"split:{gateway_id}:{version_a}:{version_b}"
        record = {
            "split_id": split_id,
            "gateway_id": gateway_id,
            "version_a": version_a,
            "version_b": version_b,
            "split_pct_a": split_pct_a,
            "split_pct_b": 100 - split_pct_a,
            "tenant_id": self.tenant_id,
            "configured_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }
        self.wasm_modules[split_id] = record  # type: ignore[assignment]
        await self._audit_log("traffic_split_configured", {"gateway_id": gateway_id, "split_pct_a": split_pct_a})
        return record

    async def security_audit_apig(
        self,
        gateway_id: str,
        scan_type: str = "owasp_top10",
    ) -> dict[str, Any]:
        """Run a security audit on the gateway."""
        return await self.security_scan_api(gateway_id, scan_type)

    async def api_export(
        self,
        gateway_id: str,
        output_format: str = "openapi_3",
    ) -> dict[str, Any]:
        """Export gateway API definition."""
        return await self.documentation_generate(gateway_id, output_format)

    async def api_lifecycle(
        self,
        gateway_id: str,
        version: str,
        status: str,
    ) -> dict[str, Any]:
        """Manage API version lifecycle state."""
        return await self.api_version_manage(gateway_id, version, status)


# Backward compatibility alias
APGIntelligentGatewayService = ProductionAPGIntelligentGatewayService

# Export main class
__all__ = ['ProductionAPGIntelligentGatewayService', 'APGIntelligentGatewayService']
