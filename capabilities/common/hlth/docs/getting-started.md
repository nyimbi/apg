# Getting Started with APG System Health Management

Welcome to APG System Health Management (HLTH) - the revolutionary system health monitoring platform that's 10x better than industry leaders like New Relic, Datadog, and Dynatrace.

## 🎯 What You'll Learn

This guide will walk you through:
- Installing and setting up APG HLTH
- Registering your first system components
- Processing health metrics
- Understanding health assessments
- Configuring predictive analytics
- Setting up autonomous remediation

## 📋 Prerequisites

### System Requirements
- Python 3.9 or higher
- 4GB RAM minimum (8GB recommended)
- 2 CPU cores minimum (4 cores recommended)
- 10GB disk space for data storage

### Dependencies
- APG Platform Core (automatically installed)
- PostgreSQL 12+ (for data persistence)
- Redis 6+ (for caching and queues)
- Optional: ML dependencies for advanced features

## 🚀 Installation

### Method 1: Using APG CLI (Recommended)

```bash
# Install APG CLI if not already installed
pip install apg-cli

# Add HLTH capability to your APG platform
apg capability add hlth

# Initialize the capability
apg capability init hlth
```

### Method 2: Direct Installation

```bash
# Install from PyPI
pip install apg-hlth

# Or install from source
git clone https://github.com/datacraft/apg-hlth.git
cd apg-hlth
pip install -e .
```

### Method 3: Docker Installation

```bash
# Pull the official image
docker pull datacraft/apg-hlth:latest

# Run with docker-compose
wget https://raw.githubusercontent.com/datacraft/apg-hlth/main/docker-compose.yml
docker-compose up -d
```

## ⚙️ Configuration

### Basic Configuration

Create a configuration file `hlth_config.yaml`:

```yaml
# hlth_config.yaml
database:
  url: "postgresql://user:pass@localhost:5432/hlth_db"
  
redis:
  url: "redis://localhost:6379/0"
  
features:
  ml_enabled: true
  auto_remediation: true
  predictive_analytics: true
  
security:
  encryption_at_rest: true
  audit_logging: true
  
compliance:
  frameworks: ["soc2", "iso27001"]
  data_retention_days: 90
  
alerting:
  channels: ["email", "slack", "webhook"]
  
performance:
  max_metrics_per_second: 10000
  batch_processing_size: 100
```

### Environment Variables

```bash
# Core settings
export HLTH_DATABASE_URL="postgresql://user:pass@localhost:5432/hlth_db"
export HLTH_REDIS_URL="redis://localhost:6379/0"
export HLTH_SECRET_KEY="your-secret-key-here"

# Feature flags
export HLTH_ML_ENABLED="true"
export HLTH_AUTO_REMEDIATION="true"
export HLTH_PREDICTIVE_ANALYTICS="true"

# Security
export HLTH_ENCRYPTION_KEY="your-encryption-key"
export HLTH_AUDIT_LOGGING="true"
```

## 🏃‍♂️ Quick Start

### Step 1: Initialize the Service

```python
import asyncio
from capabilities.common.hlth import SystemHealthService

async def main():
    # Initialize health service
    health_service = SystemHealthService()
    
    # Load configuration
    await health_service.load_config("hlth_config.yaml")
    
    # Initialize the service
    await health_service.initialize()
    
    print("✅ APG HLTH service initialized successfully!")

# Run the initialization
asyncio.run(main())
```

### Step 2: Register Your First Component

```python
from capabilities.common.hlth.models import SystemComponent, ComponentType

async def register_first_component():
    # Create a system component
    component = SystemComponent(
        component_id="web-server-01",
        tenant_id="your-org",
        name="Primary Web Server",
        component_type=ComponentType.SERVICE,
        environment="production",
        business_criticality="high",
        description="Main application web server",
        tags=["web", "frontend", "critical"]
    )
    
    # Register with health service
    result = await health_service.register_system_component(component)
    
    if result['status'] == 'success':
        print("✅ Component registered successfully!")
        print(f"   Component ID: {component.component_id}")
        print(f"   Health baseline will be established automatically")
    else:
        print(f"❌ Registration failed: {result.get('error')}")

await register_first_component()
```

### Step 3: Send Your First Health Metric

```python
from capabilities.common.hlth.models import HealthMetric, HealthDimension

async def send_first_metric():
    # Create a health metric
    metric = HealthMetric(
        tenant_id="your-org",
        component_id="web-server-01", 
        name="cpu_utilization",
        value=75.0,  # 75% CPU usage
        dimension=HealthDimension.PERFORMANCE,
        unit="percentage",
        tags={"datacenter": "us-west-1", "instance_type": "m5.large"}
    )
    
    # Process the metric
    result = await health_service.process_health_metric(metric)
    
    print("✅ Health metric processed!")
    print(f"   Health Score: {result['health_score']}")
    print(f"   Processing Time: {result['processing_time_ms']}ms")
    print(f"   Alert Triggered: {result['alert_triggered']}")

await send_first_metric()
```

### Step 4: Check Component Health

```python
async def check_component_health():
    # Assess component health
    assessment = await health_service.assess_component_health(
        component_id="web-server-01",
        tenant_id="your-org"
    )
    
    print("🔍 Component Health Assessment:")
    print(f"   Overall Health Score: {assessment['overall_health_score']}/100")
    print(f"   Health Status: {assessment['health_status']}")
    print(f"   Assessment Time: {assessment['assessment_timestamp']}")
    
    # Show dimension scores
    print("\n📊 Dimension Scores:")
    for dimension, score in assessment['dimension_scores'].items():
        print(f"   {dimension}: {score}/100")

await check_component_health()
```

## 🎛️ Advanced Setup

### Enable Predictive Analytics

```python
async def enable_predictions():
    # Enable predictive analytics for tenant
    result = await health_service.enable_predictive_analytics(
        tenant_id="your-org"
    )
    
    if result['status'] == 'enabled':
        print("🔮 Predictive analytics enabled!")
        print(f"   Models available: {result['available_models']}")
        print(f"   Training status: {result['training_status']}")
    
    # Get a health prediction
    prediction = await health_service.predict_component_health(
        component_id="web-server-01",
        tenant_id="your-org",
        prediction_window_hours=24
    )
    
    print(f"\n📈 24-hour Health Prediction:")
    print(f"   Predicted Score: {prediction['predicted_health_score']}")
    print(f"   Confidence: {prediction['confidence']:.1%}")
    print(f"   Risk Level: {prediction['risk_level']}")

await enable_predictions()
```

### Configure Autonomous Remediation

```python
async def setup_auto_remediation():
    # Configure autonomous remediation
    config = {
        'enabled_actions': [
            'restart_service',
            'scale_resources', 
            'clear_cache',
            'rotate_logs',
            'adjust_traffic'
        ],
        'safety_checks': True,
        'rollback_enabled': True,
        'approval_required': ['restart_service'],  # Require approval for restarts
        'max_actions_per_hour': 10
    }
    
    result = await health_service.configure_auto_remediation(
        tenant_id="your-org",
        config=config
    )
    
    print("🤖 Autonomous remediation configured!")
    print(f"   Enabled actions: {len(config['enabled_actions'])}")
    print(f"   Safety checks: {config['safety_checks']}")
    print(f"   Rollback enabled: {config['rollback_enabled']}")

await setup_auto_remediation()
```

### Set Up Component Discovery

```python
async def setup_auto_discovery():
    # Configure automatic component discovery
    discovery_config = {
        'kubernetes_enabled': True,
        'docker_enabled': True, 
        'cloud_providers': ['aws', 'azure', 'gcp'],
        'discovery_interval_minutes': 15,
        'auto_register': True
    }
    
    result = await health_service.configure_component_discovery(
        tenant_id="your-org",
        config=discovery_config
    )
    
    # Run immediate discovery
    discovery_result = await health_service.discover_components()
    
    print("🔍 Component discovery completed!")
    print(f"   Components discovered: {len(discovery_result.components)}")
    print(f"   Auto-registered: {discovery_result.auto_registered_count}")
    
    # Show discovered components
    for component in discovery_result.components[:5]:  # Show first 5
        print(f"   - {component.name} ({component.component_type.value})")

await setup_auto_discovery()
```

## 📊 Your First Dashboard

### Create Health Dashboard

```python
async def create_dashboard():
    # Create executive dashboard
    dashboard = await health_service.create_tenant_health_dashboard(
        tenant_id="your-org",
        dashboard_type="executive"
    )
    
    print("📊 Executive Dashboard Created!")
    print(f"   Dashboard URL: {dashboard['dashboard_url']}")
    print(f"   Components monitored: {dashboard['components_count']}")
    print(f"   Overall health: {dashboard['overall_health_score']}")
    
    # Create operational dashboard
    ops_dashboard = await health_service.create_tenant_health_dashboard(
        tenant_id="your-org", 
        dashboard_type="operational"
    )
    
    print(f"\n🔧 Operational Dashboard Created!")
    print(f"   Dashboard URL: {ops_dashboard['dashboard_url']}")
    print(f"   Real-time metrics: {ops_dashboard['realtime_enabled']}")

await create_dashboard()
```

## 🚨 Setting Up Alerts

### Configure Alert Channels

```python
async def setup_alerting():
    # Configure email alerts
    email_config = {
        'type': 'email',
        'recipients': ['ops@yourcompany.com', 'oncall@yourcompany.com'],
        'smtp_server': 'smtp.yourcompany.com',
        'smtp_port': 587,
        'username': 'alerts@yourcompany.com',
        'password': 'your-password'
    }
    
    # Configure Slack alerts
    slack_config = {
        'type': 'slack',
        'webhook_url': 'https://hooks.slack.com/services/...',
        'channel': '#ops-alerts',
        'mention_on_critical': True
    }
    
    # Configure webhook alerts  
    webhook_config = {
        'type': 'webhook',
        'url': 'https://your-webhook-endpoint.com/alerts',
        'headers': {'Authorization': 'Bearer your-token'},
        'method': 'POST'
    }
    
    # Add alert channels
    await health_service.add_alert_channel("your-org", email_config)
    await health_service.add_alert_channel("your-org", slack_config)
    await health_service.add_alert_channel("your-org", webhook_config)
    
    print("🚨 Alert channels configured!")
    print("   - Email notifications enabled")  
    print("   - Slack integration active")
    print("   - Webhook endpoint registered")

await setup_alerting()
```

### Create Custom Alert Rules

```python
from capabilities.common.hlth.models import HealthRule, HealthSeverity

async def create_alert_rules():
    # Critical CPU alert rule
    cpu_rule = HealthRule(
        rule_id="critical-cpu-rule",
        tenant_id="your-org",
        name="Critical CPU Usage",
        description="Alert when CPU usage exceeds 90% for 5 minutes",
        condition="cpu_utilization > 90",
        duration_minutes=5,
        severity=HealthSeverity.CRITICAL,
        enabled=True,
        tags={"category": "performance", "severity": "critical"}
    )
    
    # Memory leak detection rule
    memory_rule = HealthRule(
        rule_id="memory-leak-rule", 
        tenant_id="your-org",
        name="Memory Leak Detection",
        description="Detect potential memory leaks",
        condition="memory_growth_rate > 5 AND memory_utilization > 80",
        duration_minutes=10,
        severity=HealthSeverity.HIGH,
        enabled=True,
        tags={"category": "memory", "pattern": "leak"}
    )
    
    # Add rules to health service
    await health_service.add_health_rule(cpu_rule)
    await health_service.add_health_rule(memory_rule)
    
    print("📋 Custom alert rules created!")
    print("   - Critical CPU usage rule")
    print("   - Memory leak detection rule")

await create_alert_rules()
```

## 🏢 Enterprise Features

### Set Up Multi-Tenancy

```python
async def setup_enterprise_tenant():
    # Create enterprise tenant configuration
    tenant_config = {
        'tenant_id': 'enterprise-corp',
        'tenant_name': 'Enterprise Corporation',
        'tier': 'enterprise_plus',
        'compliance_frameworks': ['soc2', 'hipaa', 'iso27001'],
        'custom_branding': {
            'company_name': 'Enterprise Corp',
            'logo_url': '/static/enterprise-logo.png',
            'theme_color': '#1f2937'
        },
        'sla_requirements': {
            'availability_target': 99.99,
            'response_time_target': 100,
            'resolution_time_target': 240
        },
        'isolation_config': {
            'data_classification': 'confidential',
            'network_isolation': True,
            'storage_isolation': True,
            'encryption_at_rest': True
        }
    }
    
    # Create enterprise tenant
    result = await health_service.create_enterprise_tenant(tenant_config)
    
    print("🏢 Enterprise tenant created!")
    print(f"   Tenant ID: {result['tenant_id']}")
    print(f"   Tier: {result['tier']}")
    print(f"   Compliance: {result['compliance_frameworks']}")
    print(f"   Isolation level: {result['isolation_level']}")

await setup_enterprise_tenant()
```

### Enable Compliance Monitoring

```python
async def setup_compliance():
    tenant_id = "enterprise-corp"
    
    # Generate SOC 2 compliance report
    soc2_report = await health_service.generate_compliance_report(
        tenant_id, 'soc2', 30  # Last 30 days
    )
    
    print("📋 SOC 2 Compliance Report:")
    print(f"   Security controls: {soc2_report['report']['trust_service_criteria']['security']}")
    print(f"   Availability: {soc2_report['report']['trust_service_criteria']['availability']}")
    print(f"   Overall compliance: {soc2_report['report']['overall_compliance_percentage']:.1f}%")
    
    # Set up automated compliance monitoring
    compliance_config = {
        'frameworks': ['soc2', 'hipaa', 'iso27001'],
        'monitoring_enabled': True,
        'automated_reporting': True,
        'report_frequency': 'weekly',
        'notification_recipients': ['compliance@enterprise-corp.com']
    }
    
    await health_service.configure_compliance_monitoring(
        tenant_id, compliance_config
    )
    
    print("✅ Automated compliance monitoring enabled")

await setup_compliance()
```

## 📈 Monitoring Your Setup

### Check Service Status

```python
async def check_service_status():
    # Get service health
    status = await health_service.get_service_status()
    
    print("🔍 Service Status Check:")
    print(f"   Service healthy: {status['healthy']}")
    print(f"   Uptime: {status['uptime']}")
    print(f"   Components registered: {status['components_registered']}")
    print(f"   Metrics processed: {status['metrics_processed_total']}")
    print(f"   Active alerts: {status['active_alerts_count']}")
    print(f"   ML models trained: {status['ml_models_trained']}")
    
    # Check performance metrics
    perf = status['performance_metrics']
    print(f"\n⚡ Performance Metrics:")
    print(f"   Processing latency: {perf['avg_processing_latency_ms']}ms")
    print(f"   Throughput: {perf['metrics_per_second']}/sec")
    print(f"   Memory usage: {perf['memory_usage_mb']}MB")

await check_service_status()
```

### View Recent Activity

```python
async def view_activity():
    # Get recent alerts
    recent_alerts = await health_service.get_recent_alerts(
        tenant_id="your-org", 
        hours=24
    )
    
    print("🚨 Recent Alerts (24 hours):")
    for alert in recent_alerts[:5]:
        print(f"   [{alert.severity.value.upper()}] {alert.title}")
        print(f"      Component: {alert.component_id}")
        print(f"      Time: {alert.created_at}")
    
    # Get recent remediation actions
    remediation_history = await health_service.get_remediation_history(
        tenant_id="your-org",
        hours=24
    )
    
    print(f"\n🤖 Remediation Actions (24 hours): {len(remediation_history)}")
    for action in remediation_history[:3]:
        print(f"   {action['action_type']} on {action['component_id']}")
        print(f"      Status: {action['status']}")
        print(f"      Result: {action['result']}")

await view_activity()
```

## 🎓 Next Steps

Congratulations! You now have APG HLTH up and running. Here's what to explore next:

### 1. **Advanced Analytics**
- Set up custom ML models for your specific use cases
- Configure advanced anomaly detection
- Enable resource optimization recommendations

### 2. **Integration Development**
- Connect with your existing monitoring tools
- Set up CI/CD pipeline health checks
- Integrate with incident management systems

### 3. **Operational Excellence**
- Create custom dashboards for different teams
- Set up automated reporting
- Configure advanced alert correlation

### 4. **Scale and Optimize**
- Fine-tune performance for your workload
- Set up high availability deployment
- Configure auto-scaling policies

## 📚 Additional Resources

- **[User Manual](user-manual.md)**: Comprehensive feature documentation
- **[API Reference](api-reference.md)**: Complete API documentation
- **[Architecture Guide](architecture.md)**: Technical architecture details
- **[Best Practices](best-practices.md)**: Implementation recommendations

## 🆘 Getting Help

If you encounter any issues:

1. **Check the logs**: `tail -f /var/log/hlth/service.log`
2. **Run diagnostics**: `apg hlth diagnose`
3. **Check documentation**: [docs.datacraft.co.ke/hlth](https://docs.datacraft.co.ke/hlth)
4. **Contact support**: [hlth-support@datacraft.co.ke](mailto:hlth-support@datacraft.co.ke)

---

**Welcome to the future of system health management! 🚀**

*APG HLTH - Revolutionary Intelligence for System Health*