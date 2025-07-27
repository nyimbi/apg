# APG Event Streaming Bus

Enterprise-grade event streaming platform providing real-time event-driven communication, stream processing, and message orchestration for the APG ecosystem.

## Overview

The APG Event Streaming Bus is a foundational capability that enables:

- **Real-time Event Streaming** - High-throughput event publishing and consumption with priority handling
- **Event Sourcing & CQRS** - Immutable event logs with optimistic concurrency control and aggregate reconstruction
- **Stream Processing** - Real-time filtering, mapping, aggregation, windowing, and complex event processing
- **Enhanced Schema Registry** - Schema evolution, validation, and compatibility management
- **Consumer Management** - Advanced consumer group operations with lag monitoring and rebalancing
- **Cross-Capability Integration** - Event-driven communication between APG capabilities
- **Multi-tenant Isolation** - Secure separation of event data across tenants
- **Enterprise Monitoring** - Comprehensive processing history, audit trails, and real-time metrics

## Key Features

### 🚀 High Performance
- **1M+ events/second** throughput per node
- **<10ms latency** for 95th percentile event processing
- **Horizontal scaling** to 100+ nodes
- **Linear performance scaling** with cluster size

### 🔒 Enterprise Security
- **Multi-tenant isolation** with strict data separation
- **OAuth 2.0/JWT authentication** for API access
- **Role-based access control** for streams and subscriptions
- **End-to-end encryption** (TLS 1.3 in transit, AES-256 at rest)

### 🔄 Event Processing
- **Exactly-once delivery** semantics where needed
- **At-least-once** and **at-most-once** delivery modes
- **Priority-based routing** with HIGH, NORMAL, LOW, and CRITICAL priorities
- **Dead letter queues** for failed message handling
- **Automatic retry** with exponential backoff and configurable retry policies
- **Event compression** with GZIP, SNAPPY, LZ4, and ZSTD support
- **Multiple serialization formats** (JSON, Avro, Protobuf)
- **Schema validation** with evolution support

### 📊 Real-time Analytics
- **Stream aggregation** with tumbling, hopping, and session windows
- **Complex event processing** for pattern detection and correlation
- **Event correlation** across time windows with causation tracking
- **Real-time dashboards** with enhanced enterprise metrics
- **Stream processor management** with start/stop/metrics monitoring
- **Consumer lag monitoring** with automatic rebalancing
- **Processing history tracking** with detailed audit trails
- **Performance metrics** with duration tracking and error analysis

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database schema
python -m alembic upgrade head

# Start the service
python -m uvicorn api:app --host 0.0.0.0 --port 8080
```

### 2. Create Your First Stream

```python
from event_streaming_bus import StreamManagementService, StreamConfig
from event_streaming_bus.models import CompressionType, SerializationFormat

# Initialize service
stream_service = StreamManagementService()

# Create stream configuration
stream_config = StreamConfig(
    stream_name="user_events",
    topic_name="apg-user-events",
    source_capability="user_management",
    partitions=6,
    replication_factor=3,
    compression_type=CompressionType.SNAPPY,
    default_serialization=SerializationFormat.JSON,
    retention_time_ms=604800000  # 7 days
)

# Create the stream
stream_id = await stream_service.create_stream(
    stream_config=stream_config,
    tenant_id="your_tenant",
    user_id="your_user"
)
```

### 3. Publish Events

```python
from event_streaming_bus import EventPublishingService, EventConfig
from event_streaming_bus.models import EventPriority, CompressionType, SerializationFormat

# Initialize publishing service
publishing_service = EventPublishingService()

# Create event configuration with enterprise features
event_config = EventConfig(
    event_type="user.created",
    source_capability="user_management",
    aggregate_id="user_123",
    aggregate_type="User",
    priority=EventPriority.HIGH,
    compression_type=CompressionType.SNAPPY,
    serialization_format=SerializationFormat.JSON,
    schema_id="sch_user_created_v1",
    max_retries=5
)

# Publish event with enhanced payload
event_id = await publishing_service.publish_event(
    event_config=event_config,
    payload={
        "user_name": "john.doe",
        "email": "john.doe@company.com",
        "department": "Engineering",
        "created_at": "2025-01-26T10:30:00Z",
        "verification_status": "pending"
    },
    stream_id=stream_id,
    tenant_id="your_tenant",
    user_id="your_user"
)
```

### 4. Subscribe to Events

```python
from event_streaming_bus import ConsumerManagementService, SubscriptionConfig
from event_streaming_bus.models import DeliveryMode

# Initialize consumer management service
consumer_service = ConsumerManagementService()

# Create subscription configuration with advanced features
subscription_config = SubscriptionConfig(
    subscription_name="user_notifications",
    stream_id=stream_id,
    consumer_group_id="notification_service",
    consumer_name="notification_consumer",
    event_type_patterns=["user.*", "profile.updated"],
    delivery_mode=DeliveryMode.EXACTLY_ONCE,
    batch_size=50,
    max_wait_time_ms=1000,
    webhook_url="https://your-service.com/webhook",
    webhook_timeout_ms=5000,
    dead_letter_enabled=True,
    retry_policy={
        "max_retries": 3,
        "retry_delay_ms": 1000,
        "exponential_backoff": True
    }
)

# Create subscription
subscription_id = await consumer_service.create_subscription(
    config=subscription_config,
    tenant_id="your_tenant",
    created_by="your_user"
)
```

## API Reference

### REST API

The Event Streaming Bus provides a comprehensive REST API:

#### Event Publishing
```
POST   /api/v1/events                    # Publish single event
POST   /api/v1/events/batch              # Publish event batch  
GET    /api/v1/events/{event_id}         # Get event by ID
POST   /api/v1/events/query              # Query events with filters
```

#### Stream Management
```
GET    /api/v1/streams                   # List streams
POST   /api/v1/streams                   # Create stream
GET    /api/v1/streams/{id}              # Get stream details
GET    /api/v1/streams/{id}/events       # Get stream events
GET    /api/v1/streams/{id}/metrics      # Get stream metrics
```

#### Event Sourcing & CQRS
```
POST   /api/v1/event-sourcing/append     # Append event to aggregate
POST   /api/v1/event-sourcing/reconstruct # Reconstruct aggregate state
GET    /api/v1/event-sourcing/aggregate/{id}/events # Get aggregate events
```

#### Stream Processing
```
POST   /api/v1/stream-processors         # Create stream processor
GET    /api/v1/stream-processors         # List stream processors
POST   /api/v1/stream-processors/{id}/start # Start processor
POST   /api/v1/stream-processors/{id}/stop  # Stop processor
GET    /api/v1/stream-processors/{id}/metrics # Get processor metrics
```

#### Consumer Management
```
POST   /api/v1/consumer-groups           # Create consumer group
GET    /api/v1/consumer-groups           # List consumer groups
GET    /api/v1/consumer-groups/{id}/lag  # Get consumer lag
POST   /api/v1/consumer-groups/{id}/rebalance # Trigger rebalance

POST   /api/v1/subscriptions             # Create subscription
GET    /api/v1/subscriptions             # List subscriptions
GET    /api/v1/subscriptions/{id}/status # Get subscription status
DELETE /api/v1/subscriptions/{id}        # Cancel subscription
```

#### Enhanced Schema Registry
```
POST   /api/v1/schemas                   # Register schema (legacy)
POST   /api/v1/schemas/enhanced          # Register enhanced schema
GET    /api/v1/schemas                   # List schemas
GET    /api/v1/schemas/{id}              # Get schema details
POST   /api/v1/schemas/{id}/validate     # Validate event against schema
GET    /api/v1/schemas/evolution/{type}  # Get schema evolution history
```

### WebSocket API

Real-time streaming via WebSocket:

```
/ws/events/{stream_name}                 # Real-time event stream
/ws/subscriptions/{subscription_id}      # Subscription updates  
/ws/monitoring                           # Real-time metrics
```

### Python SDK

```python
from event_streaming_bus import (
    EventStreamingService,
    EventPublishingService, 
    EventConsumptionService,
    StreamProcessingService,
    EventSourcingService,
    SchemaRegistryService,
    StreamManagementService,
    ConsumerManagementService,
    APGEventStreamingIntegration
)

# Core services
streaming = EventStreamingService()
publishing = EventPublishingService()
consumption = EventConsumptionService()
processing = StreamProcessingService()

# Enterprise services
sourcing = EventSourcingService()
schema_registry = SchemaRegistryService()
stream_management = StreamManagementService()
consumer_management = ConsumerManagementService()

# APG platform integration
integration = APGEventStreamingIntegration(
    event_streaming_service=streaming,
    publishing_service=publishing,
    consumption_service=consumption,
    sourcing_service=sourcing,
    schema_registry_service=schema_registry,
    stream_management_service=stream_management,
    consumer_management_service=consumer_management
)
```

## Architecture

### Core Components

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       APG Event Streaming Bus Enterprise                        │
├──────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Web UI     │  │  REST API    │  │  WebSocket   │  │   GraphQL    │     │
│  │ Dashboard    │  │   Layer      │  │   Gateway    │  │   Gateway    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
├──────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Event      │  │   Stream     │  │  Enhanced    │  │   Consumer   │     │
│  │ Publishing   │  │ Processing   │  │   Schema     │  │ Management   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
├──────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │    Event     │  │   Stream     │  │  Processing  │  │    Audit     │     │
│  │  Sourcing    │  │ Assignment   │  │   History    │  │   Trails     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
├──────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Apache     │  │    Redis     │  │ PostgreSQL   │  │ Monitoring   │     │
│  │   Kafka      │  │   Streams    │  │  Database    │  │ & Metrics    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Event Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Producer   │───▶│   Schema    │───▶│   Kafka     │───▶│  Stream     │
│ Application │    │ Validation  │    │   Broker    │    │ Processor   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Event    │    │   Priority  │    │   Event     │    │  Consumer   │
│  Sourcing   │    │  Routing    │    │ Assignment  │    │   Groups    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Processing  │    │   Audit     │    │   Dead      │    │  Consumer   │
│  History    │    │   Logs      │    │  Letter     │    │ Application │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## Configuration

### Environment Variables

```bash
# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/apg_esb
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Kafka Configuration  
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_SECURITY_PROTOCOL=PLAINTEXT
KAFKA_ACKS=all
KAFKA_RETRIES=3

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=100

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
API_WORKERS=4
API_MAX_CONNECTIONS=1000

# Security Configuration
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Monitoring Configuration
METRICS_ENABLED=true
METRICS_PORT=9090
TRACING_ENABLED=true
JAEGER_ENDPOINT=http://localhost:14268/api/traces
```

### Stream Configuration

```yaml
# streams.yaml
streams:
  user_events:
    partitions: 6
    replication_factor: 3
    retention_time_ms: 604800000  # 7 days
    compression_type: snappy
    cleanup_policy: delete
    
  order_events:
    partitions: 12
    replication_factor: 3
    retention_time_ms: 2592000000  # 30 days
    compression_type: lz4
    cleanup_policy: compact
```

## Monitoring and Operations

### Metrics

The Event Streaming Bus exposes comprehensive metrics:

- **Throughput**: Events per second by stream
- **Latency**: End-to-end processing latency percentiles
- **Consumer Lag**: Backlog size per consumer group
- **Error Rates**: Failed events and retry counts
- **Resource Usage**: CPU, memory, and disk utilization

### Health Checks

```bash
# Service health
curl http://localhost:8080/health

# Detailed status
curl http://localhost:8080/api/v1/status

# Kafka cluster health
curl http://localhost:8080/api/v1/kafka/health

# Database health
curl http://localhost:8080/api/v1/database/health
```

### Logging

Structured logging with configurable levels:

```json
{
  "timestamp": "2025-01-26T10:30:00.000Z",
  "level": "INFO",
  "logger": "event_streaming_bus.publishing",
  "message": "Event published successfully",
  "event_id": "evt_123",
  "stream_id": "user_events",
  "tenant_id": "tenant_001",
  "duration_ms": 5.2
}
```

## Development

### Prerequisites

- Python 3.11+
- Apache Kafka 3.0+
- Redis 7.0+
- PostgreSQL 15+

### Setup Development Environment

```bash
# Clone repository
git clone <repository-url>
cd event_streaming_bus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Run tests
python run_tests.py --all
```

### Running Tests

```bash
# Quick tests (unit only)
python run_tests.py

# All tests
python run_tests.py --all

# Specific test categories
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --performance

# Generate coverage report
python run_tests.py --report

# Run linting
python run_tests.py --lint
```

### Code Quality

The project maintains high code quality standards:

- **Test Coverage**: >95% required for enterprise features
- **Type Hints**: Full typing coverage with strict mypy configuration
- **Linting**: Ruff for Python linting and formatting
- **Documentation**: Comprehensive docstrings, examples, and API documentation
- **Security**: Regular security audits and dependency scanning
- **Performance**: Continuous performance benchmarking and optimization

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t apg-event-streaming-bus .

# Run container
docker run -d \
  --name esb \
  -p 8080:8080 \
  -e DATABASE_URL=postgresql://... \
  -e KAFKA_BOOTSTRAP_SERVERS=kafka:9092 \
  -e REDIS_URL=redis://redis:6379 \
  apg-event-streaming-bus
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: event-streaming-bus
spec:
  replicas: 3
  selector:
    matchLabels:
      app: event-streaming-bus
  template:
    metadata:
      labels:
        app: event-streaming-bus
    spec:
      containers:
      - name: esb
        image: apg-event-streaming-bus:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: esb-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

### Production Considerations

- **Resource Planning**: Plan for 2-4 CPU cores and 4-8GB RAM per instance
- **Storage**: Use SSD storage for Kafka brokers and database
- **Networking**: Configure proper network segmentation and security groups
- **Monitoring**: Set up comprehensive monitoring and alerting
- **Backup**: Implement regular backups of PostgreSQL and Kafka topics
- **Security**: Use TLS for all communications and proper authentication

## Troubleshooting

### Common Issues

**High Consumer Lag**
```bash
# Check consumer group status
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --describe --group your-consumer-group

# Scale up consumers
kubectl scale deployment consumer --replicas=6
```

**Memory Issues**
```bash
# Check memory usage
curl http://localhost:8080/api/v1/metrics | grep memory

# Tune JVM settings for Kafka
export KAFKA_HEAP_OPTS="-Xmx2G -Xms2G"
```

**Network Connectivity**
```bash
# Test Kafka connectivity
kafka-topics.sh --bootstrap-server localhost:9092 --list

# Test Redis connectivity  
redis-cli ping

# Test database connectivity
psql $DATABASE_URL -c "SELECT 1"
```

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
export KAFKA_LOG_LEVEL=DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

© 2025 Datacraft. All rights reserved.

## Support

- **Documentation**: [API Reference](docs/api.md)
- **Examples**: [examples/](examples/)
- **Issues**: [GitHub Issues](https://github.com/datacraft/apg-event-streaming-bus/issues)
- **Email**: support@datacraft.co.ke