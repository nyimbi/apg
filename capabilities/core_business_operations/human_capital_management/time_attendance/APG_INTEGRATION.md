# APG Time & Attendance - Ecosystem Integration Guide

## 🌐 Revolutionary APG Ecosystem Integration

**Capability:** `core_business_operations/human_capital_management/time_attendance`  
**Version:** 1.0.0  
**Integration Level:** Full APG Ecosystem Composability  

---

## 🎯 Integration Overview

The APG Time & Attendance capability is designed for **seamless integration** with the entire APG ecosystem, providing **10x superior composability** compared to traditional monolithic systems.

### Core Integration Principles
- **Service-First Architecture** - Every component exposes well-defined interfaces
- **Event-Driven Communication** - Real-time coordination with other capabilities
- **Multi-Tenant by Design** - Complete isolation across organizational boundaries  
- **APG-Native Discovery** - Automatic capability registration and health monitoring
- **Composable Workflows** - Plug-and-play integration with business processes

---

## 🔌 APG Capability Registry Integration

### Capability Registration

```python
# capability.py - APG Capability Registration
from apg.core.capability import APGCapability, CapabilityInfo
from apg.core.registry import capability_registry

class TimeAttendanceCapability(APGCapability):
    """APG Time & Attendance Capability"""
    
    def __init__(self):
        super().__init__(
            info=CapabilityInfo(
                id="time_attendance",
                name="Time & Attendance Management", 
                version="1.0.0",
                category="human_capital_management",
                domain="core_business_operations",
                description="Revolutionary workforce time tracking with AI-powered insights",
                author="Nyimbi Odero <nyimbi@gmail.com>",
                license="Proprietary - Datacraft © 2025",
                tags=["time-tracking", "attendance", "workforce", "ai", "fraud-detection", "remote-work"],
                capabilities=[
                    "time_tracking",
                    "attendance_monitoring", 
                    "fraud_detection",
                    "remote_work_management",
                    "ai_agent_tracking",
                    "hybrid_collaboration",
                    "compliance_reporting",
                    "predictive_analytics"
                ]
            )
        )
    
    async def initialize(self) -> bool:
        """Initialize capability and register with APG ecosystem"""
        try:
            # Register service endpoints
            await self.register_endpoints()
            
            # Initialize database connections
            await self.setup_database()
            
            # Start background services
            await self.start_services()
            
            # Register with APG discovery
            await capability_registry.register(self)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Time & Attendance capability: {e}")
            return False
    
    async def register_endpoints(self):
        """Register REST and WebSocket endpoints with APG gateway"""
        endpoints = {
            "rest_api": {
                "base_path": "/api/human_capital_management/time_attendance",
                "health_check": "/health",
                "openapi_spec": "/openapi.json",
                "authentication": "required",
                "rate_limiting": "enabled"
            },
            "websocket": {
                "base_path": "/ws/time_attendance", 
                "authentication": "required",
                "real_time_events": True
            },
            "mobile_api": {
                "base_path": "/api/mobile/human_capital_management/time_attendance",
                "optimized_payloads": True,
                "offline_support": True
            }
        }
        
        await self.apg_gateway.register_capability_endpoints(self.info.id, endpoints)
```

### Service Discovery Integration

```python
# discovery.py - APG Service Discovery
from apg.core.discovery import ServiceDiscovery
from apg.core.events import EventBus

class TimeAttendanceDiscovery:
    """Handle service discovery and inter-capability communication"""
    
    def __init__(self, service_discovery: ServiceDiscovery, event_bus: EventBus):
        self.discovery = service_discovery
        self.event_bus = event_bus
        self.dependent_services = {}
    
    async def discover_dependencies(self):
        """Discover and connect to dependent APG capabilities"""
        
        # Employee Data Management capability
        edm_service = await self.discovery.find_capability(
            "employee_data_management",
            version=">=1.0.0"
        )
        if edm_service:
            self.dependent_services["edm"] = edm_service
            await self.setup_edm_integration(edm_service)
        
        # Computer Vision capability (for biometric verification)
        cv_service = await self.discovery.find_capability(
            "computer_vision",
            version=">=1.0.0",
            capabilities=["biometric_verification", "face_recognition"]
        )
        if cv_service:
            self.dependent_services["computer_vision"] = cv_service
            await self.setup_cv_integration(cv_service)
        
        # Notification Engine capability
        notification_service = await self.discovery.find_capability(
            "notification_engine",
            version=">=1.0.0",
            capabilities=["push_notifications", "email", "sms"]
        )
        if notification_service:
            self.dependent_services["notifications"] = notification_service
            await self.setup_notification_integration(notification_service)
        
        # Workflow & BPM capability
        workflow_service = await self.discovery.find_capability(
            "workflow_bpm",
            version=">=1.0.0",
            capabilities=["approval_workflows", "process_automation"]
        )
        if workflow_service:
            self.dependent_services["workflow"] = workflow_service
            await self.setup_workflow_integration(workflow_service)
    
    async def setup_edm_integration(self, edm_service):
        """Setup integration with Employee Data Management"""
        # Subscribe to employee lifecycle events
        await self.event_bus.subscribe(
            "employee.created",
            self.handle_employee_created
        )
        await self.event_bus.subscribe(
            "employee.updated", 
            self.handle_employee_updated
        )
        await self.event_bus.subscribe(
            "employee.terminated",
            self.handle_employee_terminated
        )
    
    async def setup_cv_integration(self, cv_service):
        """Setup integration with Computer Vision capability"""
        # Register for biometric verification services
        self.cv_client = await cv_service.get_client("biometric_verification")
    
    async def setup_notification_integration(self, notification_service):
        """Setup integration with Notification Engine"""
        # Register notification templates
        templates = {
            "clock_in_success": {
                "channels": ["push", "email"],
                "template": "Successfully clocked in at {timestamp}",
                "priority": "normal"
            },
            "fraud_alert": {
                "channels": ["push", "email", "sms"],
                "template": "Potential time fraud detected for {employee_name}",
                "priority": "high"
            },
            "overtime_warning": {
                "channels": ["push"],
                "template": "Approaching overtime limit: {hours} hours worked",
                "priority": "medium"
            }
        }
        
        await notification_service.register_templates("time_attendance", templates)
    
    async def setup_workflow_integration(self, workflow_service):
        """Setup integration with Workflow & BPM"""
        # Register approval workflows
        workflows = {
            "overtime_approval": {
                "trigger": "overtime_threshold_exceeded",
                "approvers": ["manager", "hr"],
                "auto_approve_threshold": 1.0,  # hours
                "escalation_timeout": "24h"
            },
            "time_correction": {
                "trigger": "time_entry_correction_requested",
                "approvers": ["supervisor"],
                "documentation_required": True
            }
        }
        
        await workflow_service.register_workflows("time_attendance", workflows)
```

---

## 🔄 Event-Driven Integration

### Event Publishing

```python
# event_handlers.py - APG Event Publishing
from apg.core.events import EventBus, Event
from datetime import datetime

class TimeAttendanceEventPublisher:
    """Publish time & attendance events to APG ecosystem"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
    
    async def publish_clock_in_event(self, employee_id: str, tenant_id: str, 
                                   timestamp: datetime, location: dict = None):
        """Publish clock-in event for ecosystem consumption"""
        event = Event(
            event_type="time_attendance.clock_in",
            source="time_attendance_capability",
            tenant_id=tenant_id,
            timestamp=timestamp,
            data={
                "employee_id": employee_id,
                "clock_in_time": timestamp.isoformat(),
                "location": location,
                "work_mode": "office" if not location else "remote",
                "verification_status": "verified"
            },
            correlation_id=f"clockin_{employee_id}_{int(timestamp.timestamp())}"
        )
        
        await self.event_bus.publish(event)
        
        # Trigger dependent workflows
        await self.trigger_attendance_workflows(employee_id, tenant_id, "clock_in")
    
    async def publish_fraud_detection_event(self, employee_id: str, tenant_id: str,
                                          fraud_score: float, anomaly_type: str):
        """Publish fraud detection event for security monitoring"""
        event = Event(
            event_type="time_attendance.fraud_detected",
            source="time_attendance_capability", 
            tenant_id=tenant_id,
            timestamp=datetime.utcnow(),
            data={
                "employee_id": employee_id,
                "fraud_score": fraud_score,
                "anomaly_type": anomaly_type,
                "risk_level": "high" if fraud_score > 0.8 else "medium",
                "requires_investigation": fraud_score > 0.7
            },
            priority="high"
        )
        
        await self.event_bus.publish(event)
    
    async def publish_productivity_insights(self, tenant_id: str, insights: dict):
        """Publish productivity analytics for business intelligence"""
        event = Event(
            event_type="time_attendance.productivity_insights",
            source="time_attendance_capability",
            tenant_id=tenant_id,
            timestamp=datetime.utcnow(),
            data={
                "period": insights["period"],
                "metrics": insights["metrics"],
                "trends": insights["trends"],
                "recommendations": insights["recommendations"]
            }
        )
        
        await self.event_bus.publish(event)
```

### Event Consumption

```python
# event_subscribers.py - APG Event Consumption
class TimeAttendanceEventSubscriber:
    """Subscribe to APG ecosystem events"""
    
    def __init__(self, event_bus: EventBus, service: TimeAttendanceService):
        self.event_bus = event_bus
        self.service = service
    
    async def setup_subscriptions(self):
        """Subscribe to relevant APG ecosystem events"""
        
        # Employee lifecycle events
        await self.event_bus.subscribe(
            "employee_data_management.employee_created",
            self.handle_employee_created
        )
        
        await self.event_bus.subscribe(
            "employee_data_management.employee_updated",
            self.handle_employee_updated
        )
        
        # Workflow approval events
        await self.event_bus.subscribe(
            "workflow_bpm.approval_completed",
            self.handle_approval_completed
        )
        
        # Security events
        await self.event_bus.subscribe(
            "security.threat_detected",
            self.handle_security_threat
        )
        
        # System maintenance events
        await self.event_bus.subscribe(
            "system.maintenance_scheduled",
            self.handle_maintenance_scheduled
        )
    
    async def handle_employee_created(self, event: Event):
        """Handle new employee creation"""
        employee_data = event.data
        
        # Initialize time tracking profile
        await self.service.initialize_employee_profile(
            employee_id=employee_data["employee_id"],
            tenant_id=event.tenant_id,
            work_schedule=employee_data.get("work_schedule"),
            location_constraints=employee_data.get("work_locations")
        )
    
    async def handle_approval_completed(self, event: Event):
        """Handle workflow approval completion"""
        if event.data.get("workflow_type") == "overtime_approval":
            # Update overtime entry status
            await self.service.approve_overtime_entry(
                entry_id=event.data["entity_id"],
                approved_by=event.data["approved_by"],
                approval_timestamp=event.timestamp
            )
```

---

## 🌉 API Gateway Integration

### APG Gateway Configuration

```yaml
# apg-gateway-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: apg-gateway-time-attendance
  namespace: apg-core
data:
  time-attendance-routes.yaml: |
    routes:
      # REST API Routes
      - id: time-attendance-api
        uri: http://time-attendance-app:8000
        predicates:
          - Path=/api/human_capital_management/time_attendance/**
        filters:
          - RewritePath=/api/human_capital_management/time_attendance/(?<segment>.*), /$\{segment}
          - name: RateLimiter
            args:
              redis-rate-limiter.replenishRate: 100
              redis-rate-limiter.burstCapacity: 200
          - name: AuthenticationFilter
            args:
              required: true
              scopes: ["time_attendance:read", "time_attendance:write"]
      
      # Mobile API Routes  
      - id: time-attendance-mobile
        uri: http://time-attendance-app:8000
        predicates:
          - Path=/api/mobile/human_capital_management/time_attendance/**
        filters:
          - RewritePath=/api/mobile/human_capital_management/time_attendance/(?<segment>.*), /mobile/$\{segment}
          - name: MobileOptimizationFilter
          - name: OfflineSyncFilter
      
      # WebSocket Routes
      - id: time-attendance-websocket
        uri: ws://time-attendance-app:8000
        predicates:
          - Path=/ws/time_attendance/**
        filters:
          - name: WebSocketAuthFilter
    
    # Cross-cutting concerns
    global_filters:
      - name: TenantIsolationFilter
        args:
          header_name: X-Tenant-ID
          required: true
      - name: AuditLoggingFilter
        args:
          log_requests: true
          log_responses: false
          sensitive_headers: ["authorization"]
      - name: MetricsCollectionFilter
        args:
          capability: time_attendance
```

### Authentication & Authorization

```python
# auth_integration.py - APG Authentication Integration
from apg.core.auth import APGAuthProvider, Permission, Role

class TimeAttendanceAuthProvider(APGAuthProvider):
    """Time & Attendance specific authentication and authorization"""
    
    def __init__(self):
        super().__init__("time_attendance")
        self.setup_permissions()
        self.setup_roles()
    
    def setup_permissions(self):
        """Define capability-specific permissions"""
        permissions = [
            Permission("time_attendance:clock_in", "Clock in/out for self"),
            Permission("time_attendance:view_own", "View own time records"),
            Permission("time_attendance:view_team", "View team time records"),
            Permission("time_attendance:approve", "Approve time entries"),
            Permission("time_attendance:admin", "Full administrative access"),
            Permission("time_attendance:reports", "Generate reports"),
            Permission("time_attendance:fraud_investigate", "Investigate fraud alerts"),
            Permission("time_attendance:mobile_api", "Access mobile API"),
            Permission("time_attendance:websocket", "Real-time connections")
        ]
        
        for permission in permissions:
            self.register_permission(permission)
    
    def setup_roles(self):
        """Define capability-specific roles"""
        roles = [
            Role("employee", permissions=[
                "time_attendance:clock_in",
                "time_attendance:view_own",
                "time_attendance:mobile_api",
                "time_attendance:websocket"
            ]),
            Role("supervisor", permissions=[
                "time_attendance:clock_in", 
                "time_attendance:view_own",
                "time_attendance:view_team",
                "time_attendance:approve",
                "time_attendance:mobile_api",
                "time_attendance:websocket"
            ]),
            Role("hr_manager", permissions=[
                "time_attendance:view_team",
                "time_attendance:approve", 
                "time_attendance:reports",
                "time_attendance:fraud_investigate",
                "time_attendance:websocket"
            ]),
            Role("system_admin", permissions=[
                "time_attendance:admin",
                "time_attendance:reports",
                "time_attendance:fraud_investigate"
            ])
        ]
        
        for role in roles:
            self.register_role(role)
    
    async def validate_tenant_access(self, user_id: str, tenant_id: str) -> bool:
        """Validate user access to specific tenant"""
        # Integration with APG multi-tenant authorization
        return await self.apg_auth.validate_tenant_membership(user_id, tenant_id)
```

---

## 💾 Data Integration Patterns

### Data Synchronization

```python
# data_sync.py - APG Data Synchronization
from apg.core.data import DataSyncManager, SyncPolicy

class TimeAttendanceDataSync:
    """Handle data synchronization across APG capabilities"""
    
    def __init__(self, sync_manager: DataSyncManager):
        self.sync_manager = sync_manager
        self.setup_sync_policies()
    
    def setup_sync_policies(self):
        """Configure data synchronization policies"""
        
        # Employee master data sync
        employee_sync = SyncPolicy(
            source_capability="employee_data_management",
            target_capability="time_attendance",
            sync_frequency="real_time",
            sync_direction="one_way",
            conflict_resolution="source_wins",
            data_mappings={
                "employee_id": "employee_id",
                "full_name": "employee_name",
                "department_id": "department_id",
                "manager_id": "supervisor_id",
                "work_schedule": "default_schedule",
                "employment_status": "status"
            }
        )
        
        self.sync_manager.register_policy("employee_master_data", employee_sync)
        
        # Payroll data sync
        payroll_sync = SyncPolicy(
            source_capability="time_attendance",
            target_capability="payroll_management", 
            sync_frequency="daily",
            sync_direction="one_way",
            data_mappings={
                "employee_id": "employee_id",
                "pay_period": "pay_period",
                "regular_hours": "regular_hours",
                "overtime_hours": "overtime_hours",
                "total_hours": "total_hours"
            }
        )
        
        self.sync_manager.register_policy("payroll_hours", payroll_sync)
    
    async def sync_employee_data(self, employee_id: str, tenant_id: str):
        """Sync employee data from EDM capability"""
        try:
            # Fetch latest employee data
            edm_service = await self.sync_manager.get_capability("employee_data_management")
            employee_data = await edm_service.get_employee(employee_id, tenant_id)
            
            # Update local employee profile
            await self.service.update_employee_profile(
                employee_id=employee_id,
                tenant_id=tenant_id,
                employee_data=employee_data
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to sync employee data: {e}")
            return False
```

### Master Data Management

```python
# master_data.py - APG Master Data Integration
from apg.core.mdm import MasterDataManager

class TimeAttendanceMasterData:
    """Handle master data integration"""
    
    def __init__(self, mdm: MasterDataManager):
        self.mdm = mdm
        self.register_master_entities()
    
    def register_master_entities(self):
        """Register master data entities"""
        
        # Employee master data
        self.mdm.register_entity(
            "employee",
            source_capability="employee_data_management",
            consumers=["time_attendance", "payroll_management"],
            sync_strategy="real_time"
        )
        
        # Department master data
        self.mdm.register_entity(
            "department",
            source_capability="organizational_management",
            consumers=["time_attendance", "employee_data_management"],
            sync_strategy="daily"
        )
        
        # Work schedule templates
        self.mdm.register_entity(
            "work_schedule",
            source_capability="time_attendance",
            consumers=["employee_data_management", "workforce_planning"],
            sync_strategy="on_change"
        )
```

---

## 🔧 Configuration Management

### APG Configuration Integration

```python
# config_integration.py - APG Configuration Management
from apg.core.config import APGConfigManager, ConfigSchema
from pydantic import BaseModel

class TimeAttendanceConfig(BaseModel):
    """Time & Attendance configuration schema"""
    
    # Business rules
    standard_work_hours: float = 8.0
    overtime_threshold: float = 8.0
    break_duration_minutes: int = 30
    
    # Fraud detection settings
    fraud_detection_enabled: bool = True
    fraud_threshold: float = 0.7
    biometric_verification_required: bool = True
    
    # Remote work settings
    remote_work_enabled: bool = True
    location_verification_required: bool = True
    geofence_radius_meters: int = 100
    
    # Integration settings
    employee_data_sync_enabled: bool = True
    payroll_sync_enabled: bool = True
    notification_enabled: bool = True
    
    # Performance settings
    max_concurrent_sessions: int = 1000
    database_pool_size: int = 20
    cache_ttl_seconds: int = 300

class TimeAttendanceConfigManager:
    """Manage capability configuration through APG"""
    
    def __init__(self, apg_config: APGConfigManager):
        self.apg_config = apg_config
        self.register_config_schema()
    
    def register_config_schema(self):
        """Register configuration schema with APG"""
        schema = ConfigSchema(
            capability="time_attendance",
            version="1.0.0",
            schema=TimeAttendanceConfig,
            environment_overrides=True,
            tenant_specific=True
        )
        
        self.apg_config.register_schema(schema)
    
    async def get_tenant_config(self, tenant_id: str) -> TimeAttendanceConfig:
        """Get tenant-specific configuration"""
        return await self.apg_config.get_config("time_attendance", tenant_id)
    
    async def update_tenant_config(self, tenant_id: str, config: TimeAttendanceConfig):
        """Update tenant-specific configuration""" 
        await self.apg_config.update_config("time_attendance", tenant_id, config)
```

---

## 📊 Monitoring & Observability Integration

### APG Monitoring Integration

```python
# monitoring_integration.py - APG Monitoring Integration
from apg.core.monitoring import APGMonitoring, MetricDefinition, AlertRule

class TimeAttendanceMonitoring:
    """Integrate with APG monitoring infrastructure"""
    
    def __init__(self, apg_monitoring: APGMonitoring):
        self.monitoring = apg_monitoring
        self.register_metrics()
        self.register_alerts()
    
    def register_metrics(self):
        """Register capability-specific metrics"""
        metrics = [
            MetricDefinition(
                name="time_attendance_active_sessions",
                type="gauge",
                description="Number of active time tracking sessions",
                labels=["tenant_id", "work_mode"]
            ),
            MetricDefinition(
                name="time_attendance_fraud_score",
                type="histogram",
                description="Fraud detection scores",
                labels=["tenant_id", "result"]
            ),
            MetricDefinition(
                name="time_attendance_api_duration",
                type="histogram", 
                description="API request duration",
                labels=["method", "endpoint", "status_code"]
            )
        ]
        
        for metric in metrics:
            self.monitoring.register_metric("time_attendance", metric)
    
    def register_alerts(self):
        """Register capability-specific alerts"""
        alerts = [
            AlertRule(
                name="high_fraud_detection_rate",
                expression="avg_over_time(time_attendance_fraud_score[10m]) > 0.8",
                severity="warning",
                description="High fraud detection rate detected"
            ),
            AlertRule(
                name="api_high_latency",
                expression="histogram_quantile(0.95, time_attendance_api_duration) > 1",
                severity="critical",
                description="API response time is too high"
            )
        ]
        
        for alert in alerts:
            self.monitoring.register_alert("time_attendance", alert)
```

---

## 🚀 Deployment Integration

### APG Deployment Pipeline

```yaml
# .apg/deployment.yaml - APG Deployment Configuration
apiVersion: apg.datacraft.co.ke/v1
kind: CapabilityDeployment
metadata:
  name: time-attendance
  namespace: apg-capabilities
spec:
  capability:
    id: time_attendance
    version: 1.0.0
    category: human_capital_management
    domain: core_business_operations
  
  deployment:
    strategy: rolling
    replicas:
      min: 3
      max: 20
      targetCPUUtilization: 70
    
    dependencies:
      - capability: employee_data_management
        version: ">=1.0.0"
        required: true
      - capability: notification_engine
        version: ">=1.0.0"
        required: false
      - capability: computer_vision
        version: ">=1.0.0"
        required: false
    
    resources:
      requests:
        memory: 512Mi
        cpu: 250m
      limits:
        memory: 1Gi
        cpu: 500m
    
    storage:
      - name: logs
        size: 50Gi
        accessMode: ReadWriteOnce
      - name: backup
        size: 200Gi
        accessMode: ReadWriteOnce
    
    networking:
      ports:
        - name: http
          port: 8000
          protocol: TCP
        - name: websocket
          port: 8001
          protocol: TCP
        - name: metrics
          port: 9090
          protocol: TCP
      
      ingress:
        enabled: true
        tls: true
        hosts:
          - time-attendance.apg.datacraft.co.ke
          - api.time-attendance.apg.datacraft.co.ke
    
    monitoring:
      enabled: true
      prometheus: true
      grafana: true
      alerting: true
    
    security:
      rbac: true
      networkPolicies: true
      podSecurityPolicy: true
      secrets:
        - database-credentials
        - jwt-secrets
        - encryption-keys
```

---

## 🎯 Integration Testing

### APG Integration Test Suite

```python
# integration_tests.py - APG Integration Testing
import pytest
from apg.testing import APGTestFramework, CapabilityTestClient

class TestTimeAttendanceIntegration:
    """Test Time & Attendance integration with APG ecosystem"""
    
    @pytest.fixture
    async def apg_test_env(self):
        """Setup APG test environment"""
        test_env = APGTestFramework()
        await test_env.setup_capabilities([
            "time_attendance",
            "employee_data_management", 
            "notification_engine"
        ])
        return test_env
    
    @pytest.fixture
    async def ta_client(self, apg_test_env):
        """Time & Attendance test client"""
        return CapabilityTestClient("time_attendance", apg_test_env)
    
    async def test_employee_sync_integration(self, ta_client, apg_test_env):
        """Test employee data synchronization"""
        # Create employee in EDM
        employee_data = {
            "employee_id": "test_emp_001",
            "full_name": "Test Employee",
            "department_id": "dept_001"
        }
        
        edm_client = CapabilityTestClient("employee_data_management", apg_test_env)
        await edm_client.post("/employees", json=employee_data)
        
        # Verify sync to Time & Attendance
        await asyncio.sleep(1)  # Allow for sync
        
        ta_employee = await ta_client.get(f"/employees/{employee_data['employee_id']}")
        assert ta_employee["employee_name"] == employee_data["full_name"]
    
    async def test_notification_integration(self, ta_client, apg_test_env):
        """Test notification integration"""
        # Clock in
        clock_in_data = {
            "employee_id": "test_emp_001",
            "location": {"latitude": 40.7128, "longitude": -74.0060}
        }
        
        response = await ta_client.post("/clock-in", json=clock_in_data)
        assert response.status_code == 200
        
        # Verify notification was sent
        notification_client = CapabilityTestClient("notification_engine", apg_test_env)
        notifications = await notification_client.get("/notifications/recent")
        
        clock_in_notifications = [
            n for n in notifications 
            if n["template"] == "clock_in_success"
        ]
        assert len(clock_in_notifications) > 0
    
    async def test_event_bus_integration(self, apg_test_env):
        """Test event bus integration"""
        events_received = []
        
        # Subscribe to events
        async def event_handler(event):
            events_received.append(event)
        
        await apg_test_env.event_bus.subscribe(
            "time_attendance.clock_in",
            event_handler
        )
        
        # Trigger clock-in
        ta_client = CapabilityTestClient("time_attendance", apg_test_env)
        await ta_client.post("/clock-in", json={
            "employee_id": "test_emp_001"
        })
        
        # Verify event was published
        await asyncio.sleep(0.5)  # Allow for event processing
        assert len(events_received) == 1
        assert events_received[0].event_type == "time_attendance.clock_in"
```

---

## 📋 Integration Checklist

### Pre-Deployment Validation

- [ ] **Capability Registration** - Registered with APG capability registry
- [ ] **Service Discovery** - All dependent services discoverable
- [ ] **Authentication** - APG auth provider configured and tested
- [ ] **API Gateway** - Routes configured and rate limiting enabled
- [ ] **Event Bus** - Publishing and subscribing to relevant events
- [ ] **Data Sync** - Master data synchronization working
- [ ] **Configuration** - Tenant-specific config management
- [ ] **Monitoring** - Metrics and alerts registered with APG monitoring
- [ ] **Security** - RBAC, network policies, and secrets management
- [ ] **Testing** - Integration tests passing

### Post-Deployment Verification

- [ ] **Health Checks** - All endpoints responding correctly
- [ ] **Event Flow** - Events flowing correctly between capabilities
- [ ] **Data Consistency** - Master data in sync across capabilities
- [ ] **Performance** - SLA metrics being met
- [ ] **Security** - Access controls working as expected
- [ ] **Monitoring** - Metrics being collected and alerts firing correctly

---

## 🎯 Success Criteria

### Integration KPIs

- **Service Discovery**: < 100ms capability lookup time
- **Event Latency**: < 50ms end-to-end event processing
- **Data Sync**: < 5 second sync latency across capabilities
- **API Gateway**: < 10ms routing overhead
- **Authentication**: < 20ms token validation time

### Ecosystem Health

- **Capability Availability**: 99.9% uptime across integrated services
- **Event Reliability**: 99.99% event delivery success rate
- **Data Consistency**: 100% eventual consistency across capabilities
- **Security Compliance**: Zero unauthorized access incidents

---

**🌐 Your Time & Attendance capability is now fully integrated with the APG ecosystem!**

Experience the power of **true composability** with seamless inter-capability communication, shared services, and unified monitoring across your entire business application platform.