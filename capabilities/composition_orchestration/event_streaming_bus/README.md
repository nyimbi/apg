# APG Event Streaming Bus

Enterprise-grade event streaming platform providing real-time event-driven communication, stream processing, and message orchestration for the APG ecosystem.

## Overview

The APG Event Streaming Bus is a foundational capability that enables:

- **Real-time Event Streaming** - High-throughput event publishing and consumption
- **Event Sourcing** - Immutable event logs for audit trails and state reconstruction  
- **Stream Processing** - Real-time aggregation, windowing, and complex event processing
- **Cross-Capability Integration** - Event-driven communication between APG capabilities
- **Multi-tenant Isolation** - Secure separation of event data across tenants

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
- **Dead letter queues** for failed message handling
- **Automatic retry** with exponential backoff

### 📊 Real-time Analytics
- **Stream aggregation** with windowing functions
- **Complex event processing** for pattern detection
- **Event correlation** across time windows
- **Real-time dashboards** and metrics

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
from event_streaming_bus import EventStreamingService, StreamConfig

# Initialize service
streaming_service = EventStreamingService()

# Create stream configuration
stream_config = StreamConfig(
    stream_name="user_events",
    topic_name="apg-user-events",
    source_capability="user_management",
    partitions=6,
    replication_factor=3
)

# Create the stream
stream_id = await streaming_service.create_stream(
    config=stream_config,
    tenant_id="your_tenant",
    created_by="your_user"
)
```

### 3. Publish Events

```python
from event_streaming_bus import EventPublishingService, EventConfig

# Initialize publishing service
publishing_service = EventPublishingService()

# Create event configuration
event_config = EventConfig(
    event_type="user.created",
    source_capability="user_management",
    aggregate_id="user_123",
    aggregate_type="User"
)

# Publish event
event_id = await publishing_service.publish_event(
    event_config=event_config,
    payload={
        "user_name": "john.doe",
        "email": "john.doe@company.com",
        "department": "Engineering"
    },
    stream_id=stream_id,
    tenant_id="your_tenant",
    user_id="your_user"
)
```

### 4. Subscribe to Events

```python
from event_streaming_bus import EventConsumptionService, SubscriptionConfig

# Initialize consumption service
consumption_service = EventConsumptionService()

# Create subscription configuration
subscription_config = SubscriptionConfig(
    subscription_name="user_notifications",
    stream_id=stream_id,
    consumer_group_id="notification_service",
    consumer_name="notification_consumer",
    event_type_patterns=["user.*"],
    webhook_url="https://your-service.com/webhook"
)

# Create subscription
subscription_id = await consumption_service.create_subscription(
    config=subscription_config,
    tenant_id="your_tenant",
    created_by="your_user"
)
```

## API Reference

### REST API

The Event Streaming Bus provides a comprehensive REST API:

```
POST   /api/v1/events                    # Publish single event
POST   /api/v1/events/batch              # Publish event batch  
GET    /api/v1/events/{event_id}         # Get event by ID
POST   /api/v1/events/query              # Query events with filters

GET    /api/v1/streams                   # List streams
POST   /api/v1/streams                   # Create stream
GET    /api/v1/streams/{id}              # Get stream details
GET    /api/v1/streams/{id}/events       # Get stream events
GET    /api/v1/streams/{id}/metrics      # Get stream metrics

POST   /api/v1/subscriptions             # Create subscription
GET    /api/v1/subscriptions             # List subscriptions
GET    /api/v1/subscriptions/{id}/status # Get subscription status
DELETE /api/v1/subscriptions/{id}        # Cancel subscription

POST   /api/v1/schemas                   # Register schema
GET    /api/v1/schemas                   # List schemas
GET    /api/v1/schemas/{id}              # Get schema details
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
    APGEventStreamingIntegration
)

# Core services
streaming = EventStreamingService()
publishing = EventPublishingService()
consumption = EventConsumptionService()
processing = StreamProcessingService()

# APG platform integration
integration = APGEventStreamingIntegration(
    event_streaming_service=streaming,
    publishing_service=publishing,
    consumption_service=consumption
)
```

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    APG Event Streaming Bus                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Web UI    │  │  REST API   │  │  WebSocket  │         │
│  │ Dashboard   │  │   Layer     │  │   Gateway   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Event     │  │   Stream    │  │   Schema    │         │
│  │ Publishing  │  │ Processing  │  │  Registry   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Apache    │  │    Redis    │  │ PostgreSQL  │         │
│  │   Kafka     │  │   Streams   │  │  Database   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Event Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Producer   │───▶│   Kafka     │───▶│  Consumer   │
│ Application │    │   Broker    │    │ Application │
└─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Schema    │    │   Stream    │    │   Event     │
│ Validation  │    │ Processing  │    │  Handlers   │
└─────────────┘    └─────────────┘    └─────────────┘
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

- **Test Coverage**: >80% required
- **Type Hints**: Full typing coverage
- **Linting**: Ruff for Python linting and formatting
- **Documentation**: Comprehensive docstrings and examples

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