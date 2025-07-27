# Architecture Documentation

This document provides a comprehensive overview of the Integration API Management capability architecture, design patterns, and implementation details.

## Architecture Overview

The Integration API Management capability is built using a microservices architecture with clean separation of concerns, high availability, and enterprise-scale performance requirements.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APG Platform Integration                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  Service Discovery  │  Workflow Engine  │  Event Bus  │  Policy Manager    │
├─────────────────────────────────────────────────────────────────────────────┤
│                            API Gateway Layer                                │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              │
│  │ Load Balancer   │ │ Circuit Breaker │ │ Rate Limiter    │              │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘              │
├─────────────────────────────────────────────────────────────────────────────┤
│                            Middleware Stack                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              │
│  │ Authentication  │ │ Transformation  │ │ Validation      │              │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘              │
├─────────────────────────────────────────────────────────────────────────────┤
│                            Service Layer                                    │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              │
│  │ API Lifecycle   │ │ Consumer Mgmt   │ │ Analytics       │              │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘              │
├─────────────────────────────────────────────────────────────────────────────┤
│                            Data Layer                                       │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              │
│  │ PostgreSQL      │ │ Redis Cache     │ │ Time Series DB  │              │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. API Gateway

The API Gateway serves as the entry point for all API requests, providing high-performance routing, load balancing, and policy enforcement.

#### Key Features

- **High Throughput**: 100,000+ requests per second
- **Sub-millisecond Routing**: Ultra-low latency request routing
- **Load Balancing**: Multiple strategies (round-robin, weighted, least-connections, IP hash)
- **Circuit Breaking**: Automatic failure detection and recovery
- **Health Monitoring**: Real-time upstream health checks

#### Implementation

```python
class APIGateway:
    """High-performance API gateway implementation."""
    
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.router = GatewayRouter()
        self.load_balancer = LoadBalancer()
        self.circuit_breaker = CircuitBreaker()
        self.middleware_stack = MiddlewareStack()
    
    async def handle_request(self, request: GatewayRequest) -> GatewayResponse:
        """Process incoming request through middleware stack."""
        try:
            # Apply pre-processing middleware
            await self.middleware_stack.process_request(request)
            
            # Route to upstream service
            response = await self.router.route_request(request)
            
            # Apply post-processing middleware
            await self.middleware_stack.process_response(response)
            
            return response
            
        except Exception as e:
            return self._handle_error(e, request)
```

### 2. Service Layer

The service layer implements core business logic for API management, consumer onboarding, and analytics collection.

#### API Lifecycle Service

Manages the complete API lifecycle from registration to retirement.

```python
class APILifecycleService:
    """Manages API lifecycle operations."""
    
    async def register_api(self, config: APIConfig, tenant_id: str, created_by: str) -> str:
        """Register a new API."""
        # Validate configuration
        await self._validate_api_config(config, tenant_id)
        
        # Create API record
        api = AMAPI(
            api_id=self._generate_api_id(),
            tenant_id=tenant_id,
            created_by=created_by,
            status=APIStatus.DRAFT,
            **config.dict()
        )
        
        # Store in database
        self.session.add(api)
        await self.session.commit()
        
        # Trigger registration event
        await self.event_bus.publish(EventType.API_REGISTERED, {
            "api_id": api.api_id,
            "tenant_id": tenant_id,
            "timestamp": datetime.utcnow()
        })
        
        return api.api_id
```

#### Consumer Management Service

Handles consumer onboarding, API key management, and access control.

```python
class ConsumerManagementService:
    """Manages API consumers and access control."""
    
    async def generate_api_key(self, config: APIKeyConfig, tenant_id: str, created_by: str) -> Tuple[str, str]:
        """Generate a new API key for a consumer."""
        # Generate secure API key
        api_key = self._generate_secure_key()
        key_hash = self._hash_api_key(api_key)
        
        # Create API key record
        key_record = AMAPIKey(
            key_id=self._generate_key_id(),
            consumer_id=config.consumer_id,
            key_name=config.key_name,
            key_hash=key_hash,
            key_prefix=api_key[:8],
            scopes=config.scopes,
            allowed_apis=config.allowed_apis,
            active=True,
            created_by=created_by
        )
        
        # Store in database
        self.session.add(key_record)
        await self.session.commit()
        
        return key_record.key_id, api_key
```

### 3. Policy Management

Implements a flexible policy engine for request/response transformation, validation, and access control.

#### Policy Engine Architecture

```python
class PolicyEngine:
    """Executes policies in order of priority."""
    
    def __init__(self):
        self.policies: List[Policy] = []
    
    async def execute_policies(self, context: RequestContext) -> PolicyResult:
        """Execute all applicable policies."""
        results = []
        
        for policy in sorted(self.policies, key=lambda p: p.execution_order):
            if await policy.applies_to(context):
                result = await policy.execute(context)
                results.append(result)
                
                if result.should_terminate:
                    break
        
        return PolicyResult.aggregate(results)

class RateLimitingPolicy(Policy):
    """Rate limiting policy implementation."""
    
    async def execute(self, context: RequestContext) -> PolicyResult:
        """Apply rate limiting logic."""
        key = self._generate_rate_limit_key(context)
        current_count = await self.redis.get(key) or 0
        
        if current_count >= self.config.requests_per_minute:
            return PolicyResult(
                action=PolicyAction.REJECT,
                status_code=429,
                message="Rate limit exceeded"
            )
        
        # Increment counter
        await self.redis.incr(key)
        await self.redis.expire(key, 60)  # 1 minute window
        
        return PolicyResult(action=PolicyAction.CONTINUE)
```

### 4. Analytics and Monitoring

Comprehensive analytics collection and real-time monitoring system.

#### Metrics Collection

```python
class MetricsCollector:
    """Collects and aggregates metrics."""
    
    async def record_request_metric(self, request: GatewayRequest, response: GatewayResponse, duration_ms: float):
        """Record request metrics."""
        metrics = [
            # Request count metric
            Metric(
                name="api_requests_total",
                metric_type=MetricType.COUNTER,
                value=1,
                labels={
                    "api_id": request.api_id,
                    "method": request.method,
                    "status": str(response.status_code),
                    "tenant_id": request.tenant_id
                }
            ),
            
            # Response time metric
            Metric(
                name="api_response_time_ms",
                metric_type=MetricType.HISTOGRAM,
                value=duration_ms,
                labels={
                    "api_id": request.api_id,
                    "method": request.method
                }
            )
        ]
        
        await self._store_metrics(metrics)
```

## Data Architecture

### Database Design

The system uses PostgreSQL as the primary database with a multi-tenant architecture.

#### Core Tables

```sql
-- APIs table
CREATE TABLE am_apis (
    api_id VARCHAR(50) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    api_name VARCHAR(100) NOT NULL,
    api_title VARCHAR(200) NOT NULL,
    version VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    base_path VARCHAR(200) NOT NULL,
    upstream_url VARCHAR(500) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(tenant_id, api_name, version),
    INDEX idx_apis_tenant_status (tenant_id, status),
    INDEX idx_apis_base_path (base_path)
);

-- Consumers table
CREATE TABLE am_consumers (
    consumer_id VARCHAR(50) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    consumer_name VARCHAR(100) NOT NULL,
    organization VARCHAR(200),
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(tenant_id, consumer_name),
    INDEX idx_consumers_tenant_status (tenant_id, status)
);

-- API Keys table
CREATE TABLE am_api_keys (
    key_id VARCHAR(50) PRIMARY KEY,
    consumer_id VARCHAR(50) REFERENCES am_consumers(consumer_id),
    key_hash VARCHAR(128) NOT NULL,
    key_prefix VARCHAR(20) NOT NULL,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    
    UNIQUE(key_hash),
    INDEX idx_api_keys_consumer (consumer_id),
    INDEX idx_api_keys_active (active, expires_at)
);
```

#### Data Partitioning

For high-volume analytics data, we use time-based partitioning:

```sql
-- Usage records with monthly partitioning
CREATE TABLE am_usage_records (
    record_id VARCHAR(50),
    request_id VARCHAR(50) NOT NULL,
    consumer_id VARCHAR(50),
    api_id VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    response_status INTEGER,
    response_time_ms INTEGER,
    tenant_id VARCHAR(50) NOT NULL,
    
    PRIMARY KEY (record_id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions
CREATE TABLE am_usage_records_2025_01 
    PARTITION OF am_usage_records 
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
```

### Caching Strategy

Redis is used for high-performance caching and session management.

#### Cache Patterns

```python
class CacheManager:
    """Manages caching strategies."""
    
    async def get_api_config(self, api_id: str, tenant_id: str) -> Optional[APIConfig]:
        """Get API configuration with caching."""
        cache_key = f"api_config:{tenant_id}:{api_id}"
        
        # Try cache first
        cached_data = await self.redis.get(cache_key)
        if cached_data:
            return APIConfig.parse_raw(cached_data)
        
        # Load from database
        api_config = await self._load_api_config_from_db(api_id, tenant_id)
        if api_config:
            # Cache for 5 minutes
            await self.redis.setex(cache_key, 300, api_config.json())
        
        return api_config
    
    async def increment_rate_limit_counter(self, key: str, window_seconds: int) -> int:
        """Increment rate limit counter with sliding window."""
        lua_script = """
        local key = KEYS[1]
        local window = tonumber(ARGV[1])
        local now = tonumber(ARGV[2])
        
        -- Remove expired entries
        redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
        
        -- Count current requests
        local current = redis.call('ZCARD', key)
        
        -- Add current request
        redis.call('ZADD', key, now, now)
        redis.call('EXPIRE', key, window)
        
        return current + 1
        """
        
        return await self.redis.eval(lua_script, [key], [window_seconds, time.time()])
```

## Security Architecture

### Multi-Tenant Security

The system implements comprehensive multi-tenant security with data isolation at multiple layers.

#### Data Isolation

```python
class TenantContext:
    """Provides tenant context for all operations."""
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
    
    def apply_tenant_filter(self, query: Query) -> Query:
        """Apply tenant filtering to database queries."""
        return query.filter(self.model.tenant_id == self.tenant_id)

class TenantAwareService:
    """Base class for tenant-aware services."""
    
    def __init__(self, tenant_context: TenantContext):
        self.tenant_context = tenant_context
    
    async def get_apis(self) -> List[AMAPI]:
        """Get APIs for current tenant only."""
        query = self.session.query(AMAPI)
        query = self.tenant_context.apply_tenant_filter(query)
        return await query.all()
```

#### Authentication and Authorization

```python
class AuthenticationManager:
    """Handles authentication and authorization."""
    
    async def authenticate_api_key(self, api_key: str, tenant_id: str) -> Optional[AuthContext]:
        """Authenticate API key and return context."""
        key_hash = self._hash_api_key(api_key)
        
        # Look up API key
        key_record = await self.session.query(AMAPIKey).filter_by(
            key_hash=key_hash,
            active=True
        ).first()
        
        if not key_record or key_record.consumer.tenant_id != tenant_id:
            return None
        
        # Check expiration
        if key_record.expires_at and key_record.expires_at < datetime.utcnow():
            return None
        
        return AuthContext(
            consumer_id=key_record.consumer_id,
            tenant_id=tenant_id,
            scopes=key_record.scopes,
            allowed_apis=key_record.allowed_apis
        )
```

## Performance Architecture

### High-Performance Gateway

The gateway is designed for extreme performance with careful attention to bottlenecks.

#### Async Request Processing

```python
class HighPerformanceRouter:
    """High-performance request router."""
    
    def __init__(self):
        self.route_cache = LRUCache(maxsize=10000)
        self.connection_pool = aiohttp.TCPConnector(
            limit=1000,
            limit_per_host=100,
            keepalive_timeout=30
        )
    
    async def route_request(self, request: GatewayRequest) -> GatewayResponse:
        """Route request with optimized performance."""
        # Fast path: check route cache
        route_key = f"{request.tenant_id}:{request.path}"
        route_config = self.route_cache.get(route_key)
        
        if not route_config:
            route_config = await self._resolve_route(request)
            self.route_cache[route_key] = route_config
        
        # Select upstream server
        upstream = await self.load_balancer.select_server(
            route_config.upstream_servers,
            route_config.load_balancing_strategy
        )
        
        # Forward request
        return await self._forward_request(request, upstream)
```

#### Connection Pooling

```python
class ConnectionPoolManager:
    """Manages connection pools for upstream services."""
    
    def __init__(self):
        self.pools: Dict[str, aiohttp.TCPConnector] = {}
    
    def get_connector(self, upstream_url: str) -> aiohttp.TCPConnector:
        """Get or create connection pool for upstream."""
        if upstream_url not in self.pools:
            self.pools[upstream_url] = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=50,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
        
        return self.pools[upstream_url]
```

### Scaling Strategies

#### Horizontal Scaling

```yaml
# Kubernetes HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-management-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-management
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
```

#### Database Scaling

```python
class DatabaseSharding:
    """Database sharding for multi-tenant scaling."""
    
    def __init__(self):
        self.shards = {
            'shard_1': 'postgresql://user:pass@db1:5432/api_mgmt',
            'shard_2': 'postgresql://user:pass@db2:5432/api_mgmt',
            'shard_3': 'postgresql://user:pass@db3:5432/api_mgmt'
        }
    
    def get_shard_for_tenant(self, tenant_id: str) -> str:
        """Determine which shard to use for a tenant."""
        shard_key = hash(tenant_id) % len(self.shards)
        return list(self.shards.keys())[shard_key]
    
    def get_connection(self, tenant_id: str) -> AsyncSession:
        """Get database connection for tenant's shard."""
        shard = self.get_shard_for_tenant(tenant_id)
        return self.shard_connections[shard]
```

## Monitoring Architecture

### Observability Stack

The system implements comprehensive observability with metrics, logs, and traces.

#### Metrics Architecture

```python
class MetricsArchitecture:
    """Defines metrics collection architecture."""
    
    def __init__(self):
        self.collectors = {
            'business': BusinessMetricsCollector(),
            'technical': TechnicalMetricsCollector(),
            'security': SecurityMetricsCollector()
        }
    
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all collectors."""
        metrics = {}
        
        for name, collector in self.collectors.items():
            try:
                metrics[name] = await collector.collect()
            except Exception as e:
                logger.error(f"Failed to collect {name} metrics: {e}")
                metrics[name] = {"error": str(e)}
        
        return metrics

class BusinessMetricsCollector:
    """Collects business-focused metrics."""
    
    async def collect(self) -> Dict[str, Any]:
        """Collect business metrics."""
        return {
            'total_apis': await self._count_active_apis(),
            'total_consumers': await self._count_active_consumers(),
            'api_calls_today': await self._count_api_calls_today(),
            'revenue_impact': await self._calculate_revenue_impact()
        }
```

#### Distributed Tracing

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

class TracingSetup:
    """Sets up distributed tracing."""
    
    def __init__(self):
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)
        
        jaeger_exporter = JaegerExporter(
            agent_host_name="jaeger-agent",
            agent_port=6831,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

@trace_request
async def handle_api_request(request: GatewayRequest) -> GatewayResponse:
    """Handle API request with tracing."""
    with tracer.start_as_current_span("api_request") as span:
        span.set_attribute("api.id", request.api_id)
        span.set_attribute("tenant.id", request.tenant_id)
        span.set_attribute("method", request.method)
        
        try:
            response = await process_request(request)
            span.set_attribute("response.status", response.status_code)
            return response
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            raise
```

## Deployment Architecture

### Container Architecture

```dockerfile
# Multi-stage Dockerfile for optimized builds
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim as runtime

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy dependencies
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application
WORKDIR /app
COPY . .
RUN chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080 8081
CMD ["python", "runner.py"]
```

### Kubernetes Architecture

```yaml
# Complete Kubernetes deployment
apiVersion: v1
kind: Namespace
metadata:
  name: integration-api-management

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-management
  namespace: integration-api-management
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: api-management
  template:
    metadata:
      labels:
        app: api-management
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - api-management
              topologyKey: kubernetes.io/hostname
      containers:
      - name: api-management
        image: datacraft/integration-api-management:1.0.0
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 8081
          name: gateway
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: connection-string
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Design Patterns

### Repository Pattern

```python
class APIRepository:
    """Repository for API data access."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def find_by_id(self, api_id: str, tenant_id: str) -> Optional[AMAPI]:
        """Find API by ID within tenant."""
        return await self.session.query(AMAPI).filter_by(
            api_id=api_id,
            tenant_id=tenant_id
        ).first()
    
    async def find_active_apis(self, tenant_id: str) -> List[AMAPI]:
        """Find all active APIs for tenant."""
        return await self.session.query(AMAPI).filter_by(
            tenant_id=tenant_id,
            status=APIStatus.ACTIVE
        ).all()
    
    async def save(self, api: AMAPI) -> None:
        """Save API to database."""
        self.session.add(api)
        await self.session.commit()
```

### Command Pattern

```python
class Command:
    """Base command interface."""
    
    async def execute(self) -> Any:
        raise NotImplementedError

class RegisterAPICommand(Command):
    """Command to register a new API."""
    
    def __init__(self, config: APIConfig, tenant_id: str, created_by: str):
        self.config = config
        self.tenant_id = tenant_id
        self.created_by = created_by
    
    async def execute(self) -> str:
        """Execute API registration."""
        # Validation
        await self._validate()
        
        # Create API
        api = self._create_api()
        
        # Save to database
        await self._save_api(api)
        
        # Publish event
        await self._publish_event(api)
        
        return api.api_id
```

### Observer Pattern

```python
class EventPublisher:
    """Publishes events to observers."""
    
    def __init__(self):
        self.observers: List[EventObserver] = []
    
    def subscribe(self, observer: EventObserver) -> None:
        """Subscribe to events."""
        self.observers.append(observer)
    
    async def publish(self, event: Event) -> None:
        """Publish event to all observers."""
        for observer in self.observers:
            try:
                await observer.handle_event(event)
            except Exception as e:
                logger.error(f"Observer {observer} failed to handle event: {e}")

class ServiceDiscoveryObserver(EventObserver):
    """Updates service discovery when APIs change."""
    
    async def handle_event(self, event: Event) -> None:
        """Handle API events."""
        if event.type == EventType.API_REGISTERED:
            await self._register_api_in_discovery(event.data)
        elif event.type == EventType.API_ACTIVATED:
            await self._activate_api_in_discovery(event.data)
```

## Error Handling

### Error Hierarchy

```python
class APIManagementError(Exception):
    """Base exception for API management errors."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(message)

class ValidationError(APIManagementError):
    """Validation error."""
    pass

class AuthenticationError(APIManagementError):
    """Authentication error."""
    pass

class RateLimitError(APIManagementError):
    """Rate limit exceeded error."""
    pass

class UpstreamError(APIManagementError):
    """Upstream service error."""
    pass
```

### Error Handler

```python
class ErrorHandler:
    """Centralized error handling."""
    
    async def handle_error(self, error: Exception, request: GatewayRequest) -> GatewayResponse:
        """Handle errors and return appropriate response."""
        if isinstance(error, ValidationError):
            return self._create_error_response(400, error)
        elif isinstance(error, AuthenticationError):
            return self._create_error_response(401, error)
        elif isinstance(error, RateLimitError):
            return self._create_error_response(429, error)
        elif isinstance(error, UpstreamError):
            return self._create_error_response(502, error)
        else:
            logger.exception("Unhandled error", extra={"request_id": request.request_id})
            return self._create_error_response(500, APIManagementError("Internal server error"))
    
    def _create_error_response(self, status_code: int, error: APIManagementError) -> GatewayResponse:
        """Create standardized error response."""
        return GatewayResponse(
            status_code=status_code,
            headers={"Content-Type": "application/json"},
            body=json.dumps({
                "success": False,
                "error": {
                    "code": error.error_code,
                    "message": error.message,
                    "details": error.details
                }
            }).encode()
        )
```

This architecture documentation provides a comprehensive view of the Integration API Management capability's design and implementation, serving as a reference for developers, architects, and operations teams.