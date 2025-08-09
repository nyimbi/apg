# APG System Health Management - Architecture Guide

Technical architecture documentation for the APG System Health Management capability.

## 🏗️ Architecture Overview

APG HLTH is built as a highly scalable, cloud-native system with a modular architecture designed for enterprise-grade performance, security, and reliability.

### Core Principles

1. **Microservices Architecture**: Loosely coupled, independently deployable services
2. **Event-Driven Design**: Asynchronous communication via events and messages
3. **Cloud-Native**: Designed for containerized, cloud deployments
4. **API-First**: RESTful APIs with comprehensive OpenAPI specifications
5. **Security by Design**: Built-in security controls and compliance features
6. **Horizontal Scalability**: Linear scaling with load increases
7. **Multi-Tenancy**: Secure isolation and resource sharing
8. **Observability**: Built-in monitoring, logging, and tracing

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           APG Platform                              │
├─────────────────┬───────────────────┬───────────────────┬───────────┤
│   Composition   │   Authentication  │   Notifications   │  Audit    │
│    Engine       │      (AUTH)       │      (NTFY)       │  (AUDT)   │
└─────────────────┼───────────────────┼───────────────────┼───────────┘
                  │                   │                   │
         ┌────────▼───────────────────▼───────────────────▼───────┐
         │                APG HLTH Core                           │
         ├─────────────────────────────────────────────────────────┤
         │  ┌─────────────────┐  ┌─────────────────────────────┐   │
         │  │  API Gateway    │  │     Health Assessment       │   │
         │  │                 │  │        Engine              │   │
         │  │ • Rate Limiting │  │ • Multi-dimensional        │   │
         │  │ • Auth & AuthZ  │  │ • Contextual Scoring       │   │
         │  │ • Request       │  │ • Baseline Intelligence    │   │
         │  │   Validation    │  │ • Business Impact          │   │
         │  └─────────────────┘  └─────────────────────────────┘   │
         │                                                         │
         │  ┌─────────────────┐  ┌─────────────────────────────┐   │
         │  │   ML & Analytics│  │    Autonomous Remediation   │   │
         │  │     Engine      │  │         Engine             │   │
         │  │                 │  │ • Action Library           │   │
         │  │ • Prediction    │  │ • Safety Checks           │   │
         │  │ • Anomaly Det.  │  │ • Verification            │   │
         │  │ • Optimization  │  │ • Rollback               │   │
         │  └─────────────────┘  └─────────────────────────────┘   │
         │                                                         │
         │  ┌─────────────────┐  ┌─────────────────────────────┐   │
         │  │ Alert & Notif.  │  │   Enterprise Features      │   │
         │  │    Engine       │  │                            │   │
         │  │                 │  │ • Multi-Tenancy           │   │
         │  │ • Correlation   │  │ • Compliance              │   │
         │  │ • Prioritization│  │ • RBAC & Security         │   │
         │  │ • Fatigue Redn. │  │ • Audit & Governance      │   │
         │  └─────────────────┘  └─────────────────────────────┘   │
         └─────────────────────────────────────────────────────────┘
                                       │
         ┌─────────────────────────────▼─────────────────────────────┐
         │                  Data & Storage Layer                     │
         ├─────────────────┬───────────────────┬─────────────────────┤
         │   PostgreSQL    │      Redis        │    Object Storage   │
         │                 │                   │                     │
         │ • Health Data   │ • Caching        │ • Reports           │
         │ • Components    │ • Session Mgmt   │ • Backups          │
         │ • Configurations│ • Message Queue  │ • ML Models        │
         │ • Audit Logs    │ • Rate Limiting  │ • Large Objects    │
         └─────────────────┴───────────────────┴─────────────────────┘
```

---

## 🔧 Core Components

### 1. API Gateway

The API Gateway serves as the single entry point for all client requests, providing:

#### Features
- **Request Routing**: Route requests to appropriate backend services
- **Authentication & Authorization**: JWT token validation and RBAC enforcement
- **Rate Limiting**: Per-tenant and per-user rate limiting
- **Request Validation**: Input validation and sanitization
- **Response Transformation**: Data format conversion and filtering
- **Circuit Breaking**: Fault tolerance for downstream services
- **Monitoring & Metrics**: Request/response logging and metrics collection

#### Technology Stack
- **Framework**: Python FastAPI with async/await
- **Authentication**: JWT tokens with APG platform integration
- **Rate Limiting**: Redis-based token bucket algorithm
- **Circuit Breaking**: Hystrix-style circuit breaker pattern
- **Monitoring**: OpenTelemetry with Prometheus metrics

#### Configuration
```yaml
api_gateway:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  
  rate_limiting:
    default_rpm: 1000
    burst_factor: 2
    storage: "redis://localhost:6379/1"
    
  circuit_breaker:
    failure_threshold: 10
    timeout_seconds: 60
    half_open_max_calls: 3
    
  cors:
    allowed_origins: ["*"]
    allowed_methods: ["GET", "POST", "PUT", "DELETE"]
    allowed_headers: ["*"]
```

### 2. Health Assessment Engine

The core engine responsible for processing health metrics and calculating health scores.

#### Components

**Metric Processor**
- Ingests health metrics from various sources
- Validates metric format and quality
- Applies business rules and transformations
- Stores processed metrics in time-series database

**Baseline Manager**
- Automatically establishes baselines for new components
- Updates baselines based on historical data
- Detects significant baseline shifts
- Manages seasonal and cyclical patterns

**Health Scorer**
- Calculates multi-dimensional health scores
- Applies contextual business weights
- Aggregates component scores to system-level scores
- Generates health status classifications

**Assessment Engine**
- Performs comprehensive health assessments
- Analyzes cross-component dependencies
- Identifies health correlations and patterns
- Generates actionable insights and recommendations

#### Assessment Methods
1. **Threshold-Based**: Simple threshold checking
2. **Statistical**: Z-score and percentile analysis
3. **Trend-Based**: Time-series trend analysis
4. **Anomaly-Based**: ML-powered anomaly detection
5. **Baseline-Based**: Deviation from established baselines
6. **Contextual**: Business-impact weighted scoring
7. **Dependency-Based**: Cross-component impact analysis
8. **Predictive**: Future health score predictions
9. **Composite**: Multi-metric aggregation
10. **Business-Impact**: Criticality-aware scoring
11. **ML-Enhanced**: Advanced ML model evaluation

#### Data Flow
```
Metric Input → Validation → Transformation → Storage
     ↓
Assessment Triggers → Health Calculation → Score Updates
     ↓
Alert Evaluation → Notification → Dashboard Updates
```

### 3. ML & Analytics Engine

Advanced machine learning and analytics capabilities for predictive insights.

#### Machine Learning Models

**Health Prediction Models**
- **Random Forest**: Ensemble model for robust predictions
- **Gradient Boosting**: High-accuracy sequential learning
- **LSTM Neural Networks**: Time-series pattern recognition
- **Isolation Forest**: Unsupervised anomaly detection
- **ARIMA**: Statistical time-series forecasting

**Model Management**
- **Model Versioning**: Track model versions and performance
- **A/B Testing**: Compare model performance
- **Auto-Retraining**: Automatic model updates with new data
- **Model Monitoring**: Track model drift and performance degradation
- **Feature Engineering**: Automated feature selection and engineering

**Training Pipeline**
```yaml
ml_pipeline:
  data_collection:
    window: "30 days"
    min_samples: 1000
    validation_split: 0.2
    
  feature_engineering:
    lag_features: [1, 7, 24]  # hours
    rolling_windows: [6, 24, 168]  # hours
    statistical_features: ["mean", "std", "percentiles"]
    
  model_training:
    algorithms: ["random_forest", "gradient_boosting", "lstm"]
    cross_validation: 5
    hyperparameter_tuning: "bayesian"
    
  model_evaluation:
    metrics: ["mse", "mae", "mape", "r2_score"]
    validation_window: "7 days"
    min_accuracy: 0.85
```

#### Analytics Capabilities
- **Trend Analysis**: Statistical trend detection and forecasting
- **Correlation Analysis**: Multi-variate correlation discovery
- **Pattern Recognition**: Seasonal and cyclical pattern identification
- **Root Cause Analysis**: Automated root cause identification
- **Impact Analysis**: Business impact quantification
- **Optimization Analysis**: Resource optimization recommendations

### 4. Autonomous Remediation Engine

Self-healing capabilities with safety verification and rollback.

#### Remediation Framework

**Action Library**
- **Service Management**: Restart, scale, configure services
- **Resource Optimization**: CPU, memory, storage adjustments
- **Performance Tuning**: Query optimization, cache management
- **Infrastructure**: Load balancer, DNS, network adjustments
- **Security**: Automated security updates and patches

**Safety System**
- **Pre-Execution Checks**: Validate prerequisites and safety conditions
- **Risk Assessment**: Evaluate remediation risk levels
- **Approval Workflows**: Route high-risk actions for approval
- **Execution Monitoring**: Real-time monitoring during execution
- **Success Verification**: Post-execution validation
- **Automatic Rollback**: Revert changes if verification fails

**Execution Engine**
```python
class RemediationEngine:
    async def execute_remediation(self, action: RemediationAction) -> ExecutionResult:
        # 1. Safety assessment
        safety_result = await self.assess_safety(action)
        if not safety_result.safe:
            return ExecutionResult.unsafe(safety_result.reason)
        
        # 2. Pre-execution snapshot
        snapshot = await self.create_snapshot(action.component_id)
        
        # 3. Execute action
        execution_result = await self.execute_action(action)
        
        # 4. Verify success
        verification_result = await self.verify_success(action, snapshot)
        
        # 5. Rollback if needed
        if not verification_result.success:
            await self.rollback(snapshot)
            return ExecutionResult.failed_with_rollback()
        
        return ExecutionResult.success(execution_result)
```

#### Remediation Actions

**Service Management Actions**
```yaml
restart_service:
  type: "service_management"
  risk_level: "medium"
  prerequisites: ["service_exists", "not_in_maintenance"]
  verification:
    - health_improved: true
    - error_rate_decreased: 0.1
    - response_time_stable: true
  rollback_plan: "restore_previous_version"
  max_execution_time: 300
```

**Resource Scaling Actions**
```yaml
scale_resources:
  type: "resource_optimization"
  risk_level: "low"
  parameters:
    - max_scale_factor: 2.0
    - min_scale_factor: 0.5
  verification:
    - resource_utilization_improved: true
    - performance_stable: true
    - cost_within_limits: true
  rollback_plan: "restore_previous_allocation"
```

### 5. Alert & Notification Engine

Intelligent alerting with correlation, prioritization, and fatigue reduction.

#### Alert Processing Pipeline

**Alert Generation**
1. **Rule Evaluation**: Continuous evaluation of alert rules
2. **Context Enrichment**: Add component and business context
3. **Correlation Analysis**: Group related alerts
4. **Priority Calculation**: Calculate alert priority scores
5. **Notification Routing**: Route to appropriate channels

**Correlation Engine**
```python
class AlertCorrelationEngine:
    async def correlate_alerts(self, new_alert: Alert) -> CorrelationResult:
        # Find temporally related alerts
        temporal_alerts = await self.find_temporal_correlations(
            new_alert, window_minutes=15
        )
        
        # Find dependency-based correlations
        dependency_alerts = await self.find_dependency_correlations(new_alert)
        
        # Find pattern-based correlations
        pattern_alerts = await self.find_pattern_correlations(new_alert)
        
        # Calculate correlation strength
        correlation_strength = self.calculate_correlation_strength(
            temporal_alerts, dependency_alerts, pattern_alerts
        )
        
        return CorrelationResult(
            correlated_alerts=[temporal_alerts, dependency_alerts, pattern_alerts],
            correlation_strength=correlation_strength,
            root_cause_probability=self.calculate_root_cause_probability(new_alert)
        )
```

**False Positive Reduction**
- **ML-Based Classification**: Machine learning models to identify false positives
- **Historical Analysis**: Learn from past alert resolutions
- **Context Awareness**: Consider business context and timing
- **Feedback Loop**: Incorporate user feedback to improve accuracy
- **Dynamic Thresholds**: Automatically adjust thresholds based on patterns

#### Notification Channels

**Channel Types**
- **Email**: SMTP with HTML templates and attachments
- **Slack**: Rich message formatting with interactive buttons
- **Microsoft Teams**: Adaptive cards with action buttons
- **Webhook**: HTTP callbacks with customizable payloads
- **SMS**: Text messaging for critical alerts
- **Push Notifications**: Mobile app notifications
- **PagerDuty**: Integration with on-call management
- **ServiceNow**: Automatic ticket creation

**Channel Configuration**
```yaml
notification_channels:
  slack:
    type: "slack"
    webhook_url: "${SLACK_WEBHOOK_URL}"
    channel: "#ops-alerts"
    template: "rich_card"
    rate_limiting:
      max_per_hour: 50
      burst: 10
    
  email:
    type: "smtp"
    server: "smtp.company.com"
    port: 587
    username: "${SMTP_USERNAME}"
    password: "${SMTP_PASSWORD}"
    template: "detailed_html"
    
  webhook:
    type: "webhook"
    url: "https://api.company.com/alerts"
    headers:
      Authorization: "Bearer ${WEBHOOK_TOKEN}"
    retry_policy:
      max_attempts: 3
      backoff: "exponential"
```

### 6. Enterprise Features Engine

Multi-tenancy, compliance, and enterprise-grade capabilities.

#### Multi-Tenant Architecture

**Tenant Isolation Levels**
1. **Shared**: Single database with tenant ID partitioning
2. **Hybrid**: Separate schemas within shared database
3. **Isolated**: Separate databases per tenant
4. **Dedicated**: Dedicated infrastructure per tenant

**Data Isolation**
```python
class TenantIsolationManager:
    def __init__(self, isolation_level: IsolationLevel):
        self.isolation_level = isolation_level
        self.encryption_keys = {}
        self.database_connections = {}
    
    async def get_tenant_data(self, tenant_id: str, query: str) -> QueryResult:
        if self.isolation_level == IsolationLevel.DEDICATED:
            conn = self.get_dedicated_connection(tenant_id)
        else:
            conn = self.get_shared_connection()
            query = self.add_tenant_filter(query, tenant_id)
        
        # Apply encryption if required
        if self.requires_encryption(tenant_id):
            query = self.add_decryption(query, tenant_id)
        
        return await conn.execute(query)
```

**Resource Quotas**
```yaml
tenant_quotas:
  basic:
    components: 50
    users: 5
    api_calls_per_hour: 1000
    data_retention_days: 30
    ml_predictions_per_day: 100
    
  enterprise:
    components: 5000
    users: 100
    api_calls_per_hour: 20000
    data_retention_days: 365
    ml_predictions_per_day: 10000
    custom_models: true
    dedicated_support: true
```

#### Compliance Framework

**Supported Frameworks**
- **SOC 2 Type II**: Security, availability, confidentiality, processing integrity, privacy
- **HIPAA**: Healthcare data protection and privacy
- **ISO 27001**: Information security management system
- **PCI DSS**: Payment card industry data security
- **GDPR**: European Union data protection regulation
- **FedRAMP**: Federal risk and authorization management program
- **NIST**: Cybersecurity framework implementation

**Compliance Engine**
```python
class ComplianceEngine:
    def __init__(self):
        self.frameworks = {
            'soc2': SOC2ComplianceChecker(),
            'hipaa': HIPAAComplianceChecker(),
            'iso27001': ISO27001ComplianceChecker(),
            'pci_dss': PCIDSSComplianceChecker(),
            'gdpr': GDPRComplianceChecker()
        }
    
    async def assess_compliance(self, tenant_id: str, framework: str) -> ComplianceReport:
        checker = self.frameworks[framework]
        
        # Collect compliance evidence
        evidence = await self.collect_evidence(tenant_id, framework)
        
        # Assess controls
        control_results = await checker.assess_controls(evidence)
        
        # Generate report
        return ComplianceReport(
            framework=framework,
            tenant_id=tenant_id,
            assessment_date=datetime.utcnow(),
            overall_score=self.calculate_overall_score(control_results),
            control_results=control_results,
            recommendations=self.generate_recommendations(control_results)
        )
```

---

## 💾 Data Architecture

### Database Design

#### PostgreSQL Schema

**Core Tables**
```sql
-- System Components
CREATE TABLE system_components (
    component_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    component_type component_type_enum NOT NULL,
    environment VARCHAR(50),
    business_criticality criticality_enum,
    dependencies TEXT[],
    metadata JSONB,
    tags JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Health Metrics (Time Series)
CREATE TABLE health_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(255) NOT NULL,
    component_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    dimension health_dimension_enum NOT NULL,
    unit VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    tags JSONB,
    business_context JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (timestamp);

-- Health Baselines
CREATE TABLE health_baselines (
    baseline_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(255) NOT NULL,
    component_id VARCHAR(255) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    dimension health_dimension_enum NOT NULL,
    baseline_value DOUBLE PRECISION NOT NULL,
    confidence_level DOUBLE PRECISION NOT NULL,
    lower_bound DOUBLE PRECISION,
    upper_bound DOUBLE PRECISION,
    sample_size INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Health Alerts
CREATE TABLE health_alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(255) NOT NULL,
    component_id VARCHAR(255) NOT NULL,
    rule_id VARCHAR(255),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    severity severity_enum NOT NULL,
    status alert_status_enum DEFAULT 'active',
    metric_info JSONB,
    correlation_info JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by VARCHAR(255),
    resolution_notes TEXT
);

-- Enterprise Tenants
CREATE TABLE enterprise_tenants (
    tenant_id VARCHAR(255) PRIMARY KEY,
    tenant_name VARCHAR(255) NOT NULL,
    tier tenant_tier_enum NOT NULL,
    compliance_frameworks compliance_framework_enum[],
    custom_branding JSONB,
    sla_requirements JSONB,
    resource_quotas JSONB,
    isolation_config JSONB,
    audit_requirements JSONB,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Indexes for Performance**
```sql
-- Health metrics indexes
CREATE INDEX idx_health_metrics_tenant_component_time 
ON health_metrics (tenant_id, component_id, timestamp DESC);

CREATE INDEX idx_health_metrics_name_time 
ON health_metrics (name, timestamp DESC);

CREATE INDEX idx_health_metrics_dimension 
ON health_metrics USING BTREE (dimension);

-- Component indexes
CREATE INDEX idx_system_components_tenant 
ON system_components (tenant_id);

CREATE INDEX idx_system_components_type 
ON system_components (component_type);

-- Alert indexes
CREATE INDEX idx_health_alerts_tenant_status 
ON health_alerts (tenant_id, status);

CREATE INDEX idx_health_alerts_component_time 
ON health_alerts (component_id, created_at DESC);
```

**Partitioning Strategy**
```sql
-- Partition health_metrics by month for performance
CREATE TABLE health_metrics_2024_01 PARTITION OF health_metrics
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE health_metrics_2024_02 PARTITION OF health_metrics
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Automated partition management
CREATE OR REPLACE FUNCTION create_monthly_partition(table_name text, start_date date)
RETURNS void AS $$
DECLARE
    partition_name text;
    end_date date;
BEGIN
    end_date := start_date + interval '1 month';
    partition_name := table_name || '_' || to_char(start_date, 'YYYY_MM');
    
    EXECUTE format('CREATE TABLE %I PARTITION OF %I
                    FOR VALUES FROM (%L) TO (%L)',
                   partition_name, table_name, start_date, end_date);
END;
$$ LANGUAGE plpgsql;
```

#### Redis Data Structures

**Caching Strategy**
```yaml
redis_usage:
  session_cache:
    key_pattern: "session:{session_id}"
    ttl: 3600  # 1 hour
    data: "user session data"
    
  health_scores:
    key_pattern: "health:{tenant_id}:{component_id}"
    ttl: 300   # 5 minutes
    data: "current health score and status"
    
  alert_suppression:
    key_pattern: "suppress:{alert_rule_id}"
    ttl: 1800  # 30 minutes
    data: "suppression status and reason"
    
  rate_limiting:
    key_pattern: "rate:{tenant_id}:{endpoint}"
    ttl: 3600  # sliding window
    data: "request count and timestamps"
    
  ml_predictions:
    key_pattern: "prediction:{component_id}:{window}"
    ttl: 900   # 15 minutes
    data: "cached prediction results"
```

**Message Queues**
```yaml
message_queues:
  metric_processing:
    queue: "metrics:processing"
    pattern: "work queue"
    durability: true
    max_length: 10000
    
  alert_notifications:
    queue: "alerts:notifications"
    pattern: "fanout"
    durability: true
    priority_levels: 4
    
  remediation_actions:
    queue: "remediation:actions"
    pattern: "work queue"
    durability: true
    max_retries: 3
    
  ml_training:
    queue: "ml:training"
    pattern: "delayed queue"
    durability: true
    delay_seconds: 3600
```

#### Object Storage

**Storage Organization**
```
s3://hlth-data-bucket/
├── reports/
│   ├── {tenant_id}/
│   │   ├── executive/
│   │   ├── operational/
│   │   └── compliance/
├── ml-models/
│   ├── {model_type}/
│   │   ├── {version}/
│   │   └── metadata.json
├── backups/
│   ├── database/
│   │   ├── daily/
│   │   └── weekly/
│   └── configurations/
└── exports/
    └── {tenant_id}/
        ├── metrics/
        └── alerts/
```

---

## 🚀 Deployment Architecture

### Container Architecture

#### Microservices Containers

**API Gateway Service**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements/api-gateway.txt .
RUN pip install -r api-gateway.txt

# Copy source code
COPY src/api_gateway/ .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Health Assessment Service**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements/health-assessment.txt .
RUN pip install -r health-assessment.txt

# Copy source code
COPY src/health_assessment/ .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python health_check.py || exit 1

CMD ["python", "main.py"]
```

**ML Engine Service**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install ML dependencies
COPY requirements/ml-engine.txt .
RUN pip install -r ml-engine.txt

# Copy ML models
COPY models/ ./models/

# Copy source code
COPY src/ml_engine/ .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python health_check.py || exit 1

CMD ["python", "main.py"]
```

### Kubernetes Deployment

#### Namespace and Configuration
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: hlth
  labels:
    name: hlth
    managed-by: apg

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hlth-config
  namespace: hlth
data:
  database_url: "postgresql://hlth:${DB_PASSWORD}@postgres:5432/hlth"
  redis_url: "redis://redis:6379/0"
  ml_models_path: "/app/models"
  log_level: "INFO"
```

#### Database Deployment
```yaml
# postgres-deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: hlth
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: "hlth"
        - name: POSTGRES_USER
          value: "hlth"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: hlth-secrets
              key: db-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

#### Redis Deployment
```yaml
# redis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: hlth
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-storage
          mountPath: /data
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc
```

#### Application Deployments
```yaml
# api-gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hlth-api-gateway
  namespace: hlth
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hlth-api-gateway
  template:
    metadata:
      labels:
        app: hlth-api-gateway
    spec:
      containers:
      - name: api-gateway
        image: datacraft/hlth-api-gateway:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: hlth-config
              key: database_url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: hlth-config
              key: redis_url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
# health-assessment-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hlth-health-assessment
  namespace: hlth
spec:
  replicas: 2
  selector:
    matchLabels:
      app: hlth-health-assessment
  template:
    metadata:
      labels:
        app: hlth-health-assessment
    spec:
      containers:
      - name: health-assessment
        image: datacraft/hlth-health-assessment:latest
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: hlth-config
              key: database_url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: hlth-config
              key: redis_url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

#### Services and Ingress
```yaml
# services.yaml
apiVersion: v1
kind: Service
metadata:
  name: hlth-api-gateway
  namespace: hlth
spec:
  selector:
    app: hlth-api-gateway
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hlth-ingress
  namespace: hlth
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - hlth.your-domain.com
    secretName: hlth-tls
  rules:
  - host: hlth.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: hlth-api-gateway
            port:
              number: 80
```

### High Availability Setup

#### Multi-Region Deployment
```yaml
# Multi-region deployment configuration
regions:
  primary:
    region: "us-west-2"
    availability_zones: ["us-west-2a", "us-west-2b", "us-west-2c"]
    kubernetes_cluster: "hlth-primary"
    database: 
      type: "RDS PostgreSQL"
      instance_class: "db.r6g.xlarge"
      multi_az: true
      backup_retention: 30
    
  secondary:
    region: "us-east-1" 
    availability_zones: ["us-east-1a", "us-east-1b", "us-east-1c"]
    kubernetes_cluster: "hlth-secondary"
    database:
      type: "RDS PostgreSQL Read Replica"
      source_region: "us-west-2"
    
  disaster_recovery:
    region: "eu-west-1"
    kubernetes_cluster: "hlth-dr"
    database:
      type: "RDS Cross-Region Backup"
      automated_backup: true
```

#### Load Balancing and Traffic Management
```yaml
# Global load balancer configuration
global_load_balancer:
  type: "AWS Application Load Balancer"
  health_checks:
    interval: 30
    timeout: 5
    healthy_threshold: 2
    unhealthy_threshold: 3
  
  traffic_routing:
    primary_weight: 100
    secondary_weight: 0  # Failover only
    
  failover_conditions:
    - primary_health_check_failure: true
    - response_time_threshold: 2000ms
    - error_rate_threshold: 5%
```

---

## 📊 Monitoring & Observability

### Application Metrics

#### Custom Metrics
```python
from prometheus_client import Counter, Histogram, Gauge

# Business metrics
health_assessments_total = Counter(
    'hlth_health_assessments_total',
    'Total number of health assessments performed',
    ['tenant_id', 'component_type']
)

health_score_gauge = Gauge(
    'hlth_health_score',
    'Current health score for components',
    ['tenant_id', 'component_id', 'dimension']
)

metric_processing_duration = Histogram(
    'hlth_metric_processing_duration_seconds',
    'Time spent processing health metrics',
    ['tenant_id', 'metric_type']
)

# System metrics
active_alerts_gauge = Gauge(
    'hlth_active_alerts',
    'Number of active alerts',
    ['tenant_id', 'severity']
)

remediation_actions_total = Counter(
    'hlth_remediation_actions_total',
    'Total remediation actions executed',
    ['tenant_id', 'action_type', 'status']
)

ml_prediction_accuracy = Gauge(
    'hlth_ml_prediction_accuracy',
    'ML model prediction accuracy',
    ['model_type', 'tenant_id']
)
```

#### Dashboards
```yaml
# Grafana dashboard configuration
dashboards:
  system_overview:
    title: "HLTH System Overview"
    panels:
      - title: "Request Rate"
        type: "graph"
        targets:
          - expr: 'rate(hlth_http_requests_total[5m])'
            legendFormat: "{{method}} {{endpoint}}"
      
      - title: "Health Assessments"
        type: "stat"
        targets:
          - expr: 'increase(hlth_health_assessments_total[1h])'
            legendFormat: "Assessments/Hour"
      
      - title: "Active Alerts by Severity"
        type: "pie"
        targets:
          - expr: 'hlth_active_alerts'
            legendFormat: "{{severity}}"
  
  ml_performance:
    title: "ML Engine Performance"
    panels:
      - title: "Prediction Accuracy"
        type: "graph"
        targets:
          - expr: 'hlth_ml_prediction_accuracy'
            legendFormat: "{{model_type}}"
      
      - title: "Model Training Duration"
        type: "graph"
        targets:
          - expr: 'hlth_ml_training_duration_seconds'
            legendFormat: "{{model_type}}"
```

### Logging Strategy

#### Structured Logging
```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage examples
logger.info(
    "health_metric_processed",
    tenant_id="acme-corp",
    component_id="web-server-01",
    metric_name="cpu_utilization",
    metric_value=75.5,
    health_score=82.3,
    processing_time_ms=45
)

logger.warning(
    "alert_generated",
    alert_id="alert-123",
    tenant_id="acme-corp",
    component_id="database-01",
    severity="high",
    alert_type="threshold_breach"
)
```

#### Log Aggregation
```yaml
# ELK Stack configuration
elasticsearch:
  cluster_name: "hlth-logs"
  indices:
    - name: "hlth-application-logs"
      settings:
        number_of_shards: 3
        number_of_replicas: 1
      mappings:
        properties:
          timestamp:
            type: date
          level:
            type: keyword
          logger:
            type: keyword
          tenant_id:
            type: keyword
          component_id:
            type: keyword
          message:
            type: text

logstash:
  pipelines:
    - name: "hlth-logs"
      config: |
        input {
          beats {
            port => 5044
          }
        }
        filter {
          if [fields][service] == "hlth" {
            json {
              source => "message"
            }
            date {
              match => [ "timestamp", "ISO8601" ]
            }
          }
        }
        output {
          elasticsearch {
            hosts => ["elasticsearch:9200"]
            index => "hlth-application-logs-%{+YYYY.MM.dd}"
          }
        }
```

### Distributed Tracing

#### OpenTelemetry Configuration
```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Add Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument frameworks
FastAPIInstrumentor.instrument()
SQLAlchemyInstrumentor.instrument()
RedisInstrumentor.instrument()

# Custom tracing
async def process_health_metric(metric: HealthMetric) -> dict:
    with tracer.start_as_current_span("process_health_metric") as span:
        span.set_attribute("tenant_id", metric.tenant_id)
        span.set_attribute("component_id", metric.component_id)
        span.set_attribute("metric_name", metric.name)
        
        # Process metric
        with tracer.start_as_current_span("validate_metric"):
            validation_result = await validate_metric(metric)
        
        with tracer.start_as_current_span("calculate_health_score"):
            health_score = await calculate_health_score(metric)
        
        with tracer.start_as_current_span("store_metric"):
            await store_metric(metric, health_score)
        
        span.set_attribute("health_score", health_score)
        return {"status": "success", "health_score": health_score}
```

---

## 🔒 Security Architecture

### Authentication & Authorization

#### JWT Token Validation
```python
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

class TokenManager:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    async def create_access_token(
        self, 
        data: dict, 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    async def verify_token(self, token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            return None
```

#### Role-Based Access Control
```python
from enum import Enum
from typing import List, Set

class Permission(Enum):
    READ_METRICS = "read:metrics"
    WRITE_METRICS = "write:metrics"
    READ_COMPONENTS = "read:components"
    WRITE_COMPONENTS = "write:components"
    READ_ALERTS = "read:alerts"
    MANAGE_ALERTS = "manage:alerts"
    EXECUTE_REMEDIATION = "execute:remediation"
    MANAGE_TENANTS = "manage:tenants"
    VIEW_REPORTS = "view:reports"
    ADMIN_ACCESS = "admin:access"

class Role(Enum):
    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

ROLE_PERMISSIONS = {
    Role.VIEWER: {
        Permission.READ_METRICS,
        Permission.READ_COMPONENTS,
        Permission.READ_ALERTS,
        Permission.VIEW_REPORTS
    },
    Role.OPERATOR: {
        Permission.READ_METRICS,
        Permission.WRITE_METRICS,
        Permission.READ_COMPONENTS,
        Permission.WRITE_COMPONENTS,
        Permission.READ_ALERTS,
        Permission.MANAGE_ALERTS,
        Permission.EXECUTE_REMEDIATION,
        Permission.VIEW_REPORTS
    },
    Role.ADMIN: {
        Permission.READ_METRICS,
        Permission.WRITE_METRICS,
        Permission.READ_COMPONENTS,
        Permission.WRITE_COMPONENTS,
        Permission.READ_ALERTS,
        Permission.MANAGE_ALERTS,
        Permission.EXECUTE_REMEDIATION,
        Permission.VIEW_REPORTS,
        Permission.MANAGE_TENANTS
    },
    Role.SUPER_ADMIN: {perm for perm in Permission}
}

class RBACManager:
    def check_permission(
        self, 
        user_roles: List[Role], 
        required_permission: Permission
    ) -> bool:
        for role in user_roles:
            if required_permission in ROLE_PERMISSIONS.get(role, set()):
                return True
        return False
```

### Data Encryption

#### Encryption at Rest
```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class EncryptionManager:
    def __init__(self, password: bytes, salt: bytes = None):
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.cipher_suite = Fernet(key)
        self.salt = salt
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_dict(self, data: dict) -> dict:
        """Encrypt sensitive fields in dictionary"""
        sensitive_fields = ['password', 'api_key', 'secret', 'token']
        encrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in encrypted_data:
                encrypted_data[field] = self.encrypt(str(encrypted_data[field]))
        
        return encrypted_data
```

#### Encryption in Transit
```yaml
# TLS Configuration
tls_config:
  version: "1.3"
  cipher_suites:
    - "TLS_AES_256_GCM_SHA384"
    - "TLS_CHACHA20_POLY1305_SHA256"
    - "TLS_AES_128_GCM_SHA256"
  
  certificate:
    type: "Let's Encrypt"
    auto_renewal: true
    domains:
      - "hlth.your-domain.com"
      - "api.hlth.your-domain.com"
  
  hsts:
    enabled: true
    max_age: 31536000
    include_subdomains: true
```

### Security Monitoring

#### Security Event Detection
```python
import asyncio
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class SecurityEvent:
    event_type: str
    severity: str
    source_ip: str
    user_id: str
    tenant_id: str
    description: str
    timestamp: datetime
    metadata: Dict

class SecurityMonitor:
    def __init__(self):
        self.failed_logins: Dict[str, List[datetime]] = {}
        self.suspicious_ips: set = set()
        self.rate_limiters: Dict[str, List[datetime]] = {}
    
    async def monitor_authentication(self, event: SecurityEvent):
        """Monitor authentication events for suspicious activity"""
        if event.event_type == "login_failed":
            await self._track_failed_login(event)
        elif event.event_type == "login_success":
            await self._check_unusual_login(event)
    
    async def _track_failed_login(self, event: SecurityEvent):
        """Track failed login attempts"""
        key = f"{event.source_ip}:{event.user_id}"
        now = datetime.utcnow()
        
        if key not in self.failed_logins:
            self.failed_logins[key] = []
        
        # Remove old attempts (>1 hour)
        self.failed_logins[key] = [
            attempt for attempt in self.failed_logins[key]
            if now - attempt < timedelta(hours=1)
        ]
        
        self.failed_logins[key].append(now)
        
        # Check for brute force attack
        if len(self.failed_logins[key]) >= 5:
            await self._trigger_security_alert(
                "brute_force_detected",
                f"Multiple failed login attempts from {event.source_ip}",
                event
            )
    
    async def _trigger_security_alert(
        self, 
        alert_type: str, 
        message: str, 
        event: SecurityEvent
    ):
        """Trigger security alert"""
        security_alert = {
            "type": alert_type,
            "severity": "high",
            "message": message,
            "source_event": event,
            "timestamp": datetime.utcnow(),
            "recommended_actions": self._get_recommended_actions(alert_type)
        }
        
        # Send to security team
        await self._notify_security_team(security_alert)
        
        # Take automated action if configured
        await self._execute_security_action(alert_type, event)
    
    def _get_recommended_actions(self, alert_type: str) -> List[str]:
        actions = {
            "brute_force_detected": [
                "Block source IP address",
                "Force password reset for affected account",
                "Enable additional MFA requirements"
            ],
            "unusual_access_pattern": [
                "Verify user identity",
                "Review access logs",
                "Consider temporary account suspension"
            ]
        }
        return actions.get(alert_type, ["Review security logs"])
```

---

## 🔄 Performance Optimization

### Caching Strategy

#### Multi-Level Caching
```python
from typing import Optional, Any
import redis
from functools import wraps
import json
import hashlib

class CacheManager:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.local_cache = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "local_hits": 0,
            "redis_hits": 0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with fallback hierarchy"""
        # Level 1: Local cache
        if key in self.local_cache:
            self.cache_stats["hits"] += 1
            self.cache_stats["local_hits"] += 1
            return self.local_cache[key]
        
        # Level 2: Redis cache
        redis_value = await self.redis.get(key)
        if redis_value:
            value = json.loads(redis_value)
            # Populate local cache
            self.local_cache[key] = value
            self.cache_stats["hits"] += 1
            self.cache_stats["redis_hits"] += 1
            return value
        
        self.cache_stats["misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in both cache levels"""
        serialized_value = json.dumps(value)
        
        # Set in Redis
        await self.redis.setex(key, ttl, serialized_value)
        
        # Set in local cache
        self.local_cache[key] = value
        
        # Limit local cache size
        if len(self.local_cache) > 1000:
            # Remove oldest entries (simple LRU simulation)
            keys_to_remove = list(self.local_cache.keys())[:100]
            for k in keys_to_remove:
                del self.local_cache[k]

def cache_result(ttl: int = 300, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
            cache_key = f"{key_prefix}:{hashlib.md5(key_data.encode()).hexdigest()}"
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

# Usage example
@cache_result(ttl=600, key_prefix="health_score")
async def calculate_health_score(component_id: str, tenant_id: str) -> float:
    # Expensive calculation
    return await expensive_health_calculation(component_id, tenant_id)
```

### Database Optimization

#### Query Optimization
```sql
-- Optimized queries with proper indexing

-- 1. Get recent health metrics for component
EXPLAIN (ANALYZE, BUFFERS) 
SELECT metric_id, name, value, timestamp
FROM health_metrics 
WHERE component_id = $1 
  AND tenant_id = $2 
  AND timestamp >= $3 
ORDER BY timestamp DESC 
LIMIT 100;

-- Index: idx_health_metrics_component_time covers this query perfectly

-- 2. Get health score trends
WITH metric_averages AS (
  SELECT 
    date_trunc('hour', timestamp) as hour,
    AVG(value) as avg_value
  FROM health_metrics 
  WHERE component_id = $1 
    AND name = $2 
    AND timestamp >= $3
  GROUP BY date_trunc('hour', timestamp)
)
SELECT 
  hour,
  avg_value,
  LAG(avg_value) OVER (ORDER BY hour) as prev_value,
  (avg_value - LAG(avg_value) OVER (ORDER BY hour)) / 
    NULLIF(LAG(avg_value) OVER (ORDER BY hour), 0) * 100 as pct_change
FROM metric_averages
ORDER BY hour;

-- 3. Component health summary with alerting
SELECT 
  c.component_id,
  c.name,
  c.component_type,
  COALESCE(hs.current_health_score, 0) as health_score,
  COUNT(ha.alert_id) as active_alerts,
  MAX(ha.severity) as max_severity
FROM system_components c
LEFT JOIN LATERAL (
  SELECT AVG(value) as current_health_score
  FROM health_metrics hm
  WHERE hm.component_id = c.component_id 
    AND hm.timestamp >= NOW() - INTERVAL '1 hour'
) hs ON true
LEFT JOIN health_alerts ha ON ha.component_id = c.component_id 
  AND ha.status = 'active'
WHERE c.tenant_id = $1
GROUP BY c.component_id, c.name, c.component_type, hs.current_health_score
ORDER BY health_score ASC, active_alerts DESC;
```

#### Connection Pooling
```python
import asyncpg
from asyncpg import pool
from contextlib import asynccontextmanager

class DatabaseManager:
    def __init__(self):
        self.pool: Optional[pool.Pool] = None
    
    async def initialize(
        self,
        database_url: str,
        min_size: int = 10,
        max_size: int = 20,
        command_timeout: int = 60
    ):
        """Initialize connection pool"""
        self.pool = await asyncpg.create_pool(
            database_url,
            min_size=min_size,
            max_size=max_size,
            command_timeout=command_timeout,
            server_settings={
                'application_name': 'hlth-service',
                'timezone': 'UTC'
            }
        )
    
    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool"""
        async with self.pool.acquire() as connection:
            yield connection
    
    async def execute_query(self, query: str, *args) -> list:
        """Execute query and return results"""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)
    
    async def execute_transaction(self, queries: list) -> bool:
        """Execute multiple queries in transaction"""
        async with self.get_connection() as conn:
            async with conn.transaction():
                try:
                    for query, args in queries:
                        await conn.execute(query, *args)
                    return True
                except Exception as e:
                    # Transaction will be rolled back automatically
                    logger.error("Transaction failed", error=str(e))
                    return False
```

### Async Processing

#### Message Queue Processing
```python
import asyncio
import aioredis
from typing import Callable, Dict, Any
import json
from dataclasses import dataclass

@dataclass
class QueueMessage:
    id: str
    type: str
    payload: Dict[Any, Any]
    retry_count: int = 0
    max_retries: int = 3

class AsyncMessageProcessor:
    def __init__(self, redis_url: str):
        self.redis = None
        self.redis_url = redis_url
        self.handlers: Dict[str, Callable] = {}
        self.running = False
    
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis = await aioredis.from_url(self.redis_url)
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register message handler"""
        self.handlers[message_type] = handler
    
    async def enqueue(self, queue_name: str, message: QueueMessage):
        """Add message to queue"""
        serialized_message = json.dumps({
            "id": message.id,
            "type": message.type,
            "payload": message.payload,
            "retry_count": message.retry_count,
            "max_retries": message.max_retries
        })
        await self.redis.lpush(queue_name, serialized_message)
    
    async def process_queue(self, queue_name: str, concurrency: int = 10):
        """Process messages from queue with concurrency"""
        semaphore = asyncio.Semaphore(concurrency)
        
        while self.running:
            try:
                # Get message (blocking with timeout)
                result = await self.redis.brpop(queue_name, timeout=5)
                if not result:
                    continue
                
                queue, message_data = result
                message_dict = json.loads(message_data)
                message = QueueMessage(**message_dict)
                
                # Process message with concurrency control
                task = asyncio.create_task(
                    self._process_message_with_semaphore(semaphore, message)
                )
                
                # Don't await here to maintain concurrency
                asyncio.create_task(self._handle_task_completion(task, queue_name, message))
                
            except Exception as e:
                logger.error("Queue processing error", error=str(e))
                await asyncio.sleep(1)
    
    async def _process_message_with_semaphore(
        self, 
        semaphore: asyncio.Semaphore, 
        message: QueueMessage
    ):
        async with semaphore:
            return await self._process_message(message)
    
    async def _process_message(self, message: QueueMessage) -> bool:
        """Process individual message"""
        try:
            handler = self.handlers.get(message.type)
            if not handler:
                logger.warning("No handler for message type", type=message.type)
                return False
            
            # Execute handler
            await handler(message.payload)
            return True
            
        except Exception as e:
            logger.error(
                "Message processing failed",
                message_id=message.id,
                type=message.type,
                error=str(e)
            )
            return False
    
    async def _handle_task_completion(
        self, 
        task: asyncio.Task, 
        queue_name: str, 
        message: QueueMessage
    ):
        """Handle completed processing task"""
        try:
            success = await task
            if not success:
                await self._handle_message_retry(queue_name, message)
        except Exception as e:
            logger.error("Task completion error", error=str(e))
            await self._handle_message_retry(queue_name, message)
    
    async def _handle_message_retry(self, queue_name: str, message: QueueMessage):
        """Handle message retry logic"""
        if message.retry_count < message.max_retries:
            message.retry_count += 1
            # Add delay before retry
            await asyncio.sleep(2 ** message.retry_count)  # Exponential backoff
            await self.enqueue(f"{queue_name}:retry", message)
        else:
            # Move to dead letter queue
            await self.enqueue(f"{queue_name}:dead", message)
            logger.error("Message moved to dead letter queue", message_id=message.id)

# Usage
async def main():
    processor = AsyncMessageProcessor("redis://localhost:6379")
    await processor.initialize()
    
    # Register handlers
    processor.register_handler("process_metric", process_health_metric_handler)
    processor.register_handler("generate_alert", generate_alert_handler)
    processor.register_handler("execute_remediation", execute_remediation_handler)
    
    # Start processing queues
    processor.running = True
    
    tasks = [
        asyncio.create_task(processor.process_queue("health_metrics", concurrency=20)),
        asyncio.create_task(processor.process_queue("alerts", concurrency=10)),
        asyncio.create_task(processor.process_queue("remediation", concurrency=5))
    ]
    
    await asyncio.gather(*tasks)
```

---

## 📦 API Design Patterns

### RESTful API Design

#### Resource-Based URLs
```python
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List
from datetime import datetime

router = APIRouter(prefix="/api/v1/hlth")

# Health Metrics Resource
@router.post("/metrics")
async def create_health_metric(
    metric: HealthMetricRequest,
    current_user: User = Depends(get_current_user)
):
    """Process a new health metric"""
    # Validate tenant access
    if not await validate_tenant_access(current_user, metric.tenant_id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    result = await health_service.process_health_metric(metric.to_model())
    return HealthMetricResponse(**result)

@router.get("/metrics/{component_id}/history")
async def get_metric_history(
    component_id: str,
    tenant_id: str = Query(..., description="Tenant identifier"),
    metric_name: Optional[str] = Query(None, description="Specific metric name"),
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum records"),
    current_user: User = Depends(get_current_user)
):
    """Get metric history for a component"""
    # Validate access
    if not await validate_component_access(current_user, tenant_id, component_id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    history = await health_service.get_metric_history(
        component_id=component_id,
        tenant_id=tenant_id,
        metric_name=metric_name,
        start_time=start_time,
        end_time=end_time,
        limit=limit
    )
    
    return MetricHistoryResponse(**history)

# Components Resource
@router.post("/components")
async def register_component(
    component: ComponentRegistrationRequest,
    current_user: User = Depends(get_current_user)
):
    """Register a new system component"""
    if not await validate_tenant_access(current_user, component.tenant_id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    result = await health_service.register_system_component(component.to_model())
    return ComponentRegistrationResponse(**result)

@router.get("/components")
async def list_components(
    tenant_id: str = Query(..., description="Tenant identifier"),
    component_type: Optional[ComponentType] = Query(None),
    environment: Optional[str] = Query(None),
    health_status: Optional[HealthStatus] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user)
):
    """List system components with optional filtering"""
    if not await validate_tenant_access(current_user, tenant_id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    components = await health_service.list_components(
        tenant_id=tenant_id,
        component_type=component_type,
        environment=environment,
        health_status=health_status,
        limit=limit,
        offset=offset
    )
    
    return ComponentListResponse(
        components=[ComponentSummary.from_model(c) for c in components],
        total=len(components),
        limit=limit,
        offset=offset
    )
```

#### Error Handling
```python
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback

class APIError(Exception):
    def __init__(self, code: str, message: str, details: str = None, status_code: int = 400):
        self.code = code
        self.message = message
        self.details = details
        self.status_code = status_code
        super().__init__(message)

class ErrorResponse:
    def __init__(self, code: str, message: str, details: str = None, request_id: str = None):
        self.error = {
            "code": code,
            "message": message,
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id
        }

@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            code=exc.code,
            message=exc.message,
            details=exc.details,
            request_id=request.headers.get("X-Request-ID")
        ).__dict__
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            code="VALIDATION_ERROR",
            message="Request validation failed",
            details=str(exc.errors()),
            request_id=request.headers.get("X-Request-ID")
        ).__dict__
    )

# Usage in endpoints
async def process_health_metric(metric: HealthMetricRequest):
    try:
        # Validate metric
        if not metric.tenant_id:
            raise APIError(
                code="INVALID_TENANT",
                message="Tenant ID is required",
                status_code=400
            )
        
        if metric.value < 0:
            raise APIError(
                code="INVALID_METRIC_VALUE",
                message="Metric value cannot be negative",
                details=f"Received value: {metric.value}",
                status_code=400
            )
        
        # Process metric
        result = await health_service.process_health_metric(metric.to_model())
        return result
        
    except ValueError as e:
        raise APIError(
            code="PROCESSING_ERROR",
            message="Failed to process health metric",
            details=str(e),
            status_code=500
        )
```

### API Versioning Strategy

#### URL-Based Versioning
```python
# Version 1 API
v1_router = APIRouter(prefix="/api/v1/hlth")

@v1_router.post("/metrics")
async def create_metric_v1(metric: HealthMetricV1):
    # Legacy implementation
    pass

# Version 2 API with enhanced features
v2_router = APIRouter(prefix="/api/v2/hlth")

@v2_router.post("/metrics")
async def create_metric_v2(metric: HealthMetricV2):
    # Enhanced implementation with new fields
    pass

# Version compatibility layer
class VersionCompatibility:
    @staticmethod
    async def convert_v1_to_v2(metric_v1: HealthMetricV1) -> HealthMetricV2:
        """Convert V1 metric to V2 format"""
        return HealthMetricV2(
            tenant_id=metric_v1.tenant_id,
            component_id=metric_v1.component_id,
            name=metric_v1.name,
            value=metric_v1.value,
            dimension=metric_v1.dimension,
            # New V2 fields with defaults
            business_context=BusinessContext(
                criticality=BusinessCriticality.MEDIUM,
                impact_scope="component"
            ),
            quality_score=1.0,
            confidence_level=0.9
        )
```

---

**APG System Health Management - Architecture Guide**

*Comprehensive technical architecture for revolutionary system health monitoring*