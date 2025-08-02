# Enterprise Connectors Guide

## Overview

The APG Central Configuration capability includes comprehensive enterprise integrations that allow seamless connectivity with popular business tools and services. These connectors enable automated configuration synchronization, event notifications, and workflow automation.

## Available Connectors

### Communication Platforms

#### Discord Integration

Sends configuration change notifications to Discord channels via webhooks.

**Features:**
- Real-time configuration change notifications
- Rich embed messages with change details
- User mention support for critical changes
- Channel-specific routing based on configuration patterns

**Setup:**

1. Create Discord webhook:
   ```
   Discord Server → Settings → Integrations → Webhooks → New Webhook
   ```

2. Configure webhook URL:
   ```python
   await service.set_config(
       "integrations.discord.webhook_url",
       "https://discord.com/api/webhooks/123456789/abcdef..."
   )
   ```

3. Configure notification settings:
   ```python
   discord_config = {
       "webhook_url": "https://discord.com/api/webhooks/...",
       "username": "APG Config Bot",
       "avatar_url": "https://example.com/bot-avatar.png",
       "mention_users": ["@admin", "@devops"],
       "critical_patterns": ["*.database.*", "*.security.*"],
       "channels": {
           "database": "https://discord.com/api/webhooks/.../db-channel",
           "security": "https://discord.com/api/webhooks/.../security-channel"
       }
   }
   await service.set_config("integrations.discord", discord_config)
   ```

**Example Notification:**
```
🔧 Configuration Changed
Key: app.database.host
Old Value: localhost
New Value: prod-db.internal
Changed By: admin@company.com
Environment: production
Time: 2025-01-01 14:30:00 UTC
```

#### Slack Integration

Similar to Discord but optimized for Slack's webhook format and features.

**Setup:**
```python
slack_config = {
    "webhook_url": "https://hooks.slack.com/services/...",
    "channel": "#config-changes",
    "username": "APG Config Bot",
    "icon_emoji": ":gear:",
    "mention_groups": ["@devops-team", "@platform-team"]
}
await service.set_config("integrations.slack", slack_config)
```

### Issue Tracking

#### Zendesk Integration

Automatically creates tickets for configuration changes and issues.

**Features:**
- Automatic ticket creation for failed configurations
- Configuration change audit tickets
- Custom fields for metadata
- Ticket assignment based on configuration ownership

**Setup:**

1. Generate Zendesk API token:
   ```
   Zendesk Admin → Channels → API → Token Access → Add API Token
   ```

2. Configure Zendesk connection:
   ```python
   zendesk_config = {
       "subdomain": "yourcompany",
       "email": "admin@yourcompany.com",
       "api_token": "your-api-token",
       "default_assignee": "platform-team@yourcompany.com",
       "ticket_templates": {
           "config_change": {
               "subject": "Configuration Change: {config_key}",
               "priority": "normal",
               "type": "incident",
               "tags": ["configuration", "automated"]
           },
           "config_error": {
               "subject": "Configuration Error: {config_key}",
               "priority": "high",
               "type": "problem",
               "tags": ["configuration", "error", "urgent"]
           }
       }
   }
   await service.set_config("integrations.zendesk", zendesk_config)
   ```

**Automated Ticket Creation:**
- Failed configuration validations
- Security policy violations
- Critical configuration changes
- Schema validation errors

#### Jira Integration

Creates and manages Jira issues for configuration management workflows.

**Setup:**
```python
jira_config = {
    "server_url": "https://yourcompany.atlassian.net",
    "username": "automation@yourcompany.com",
    "api_token": "your-jira-token",
    "default_project": "CONFIG",
    "issue_types": {
        "config_change": "Task",
        "config_error": "Bug",
        "schema_update": "Story"
    }
}
await service.set_config("integrations.jira", jira_config)
```

### Monitoring and Observability

#### New Relic Integration

Sends configuration events and metrics to New Relic for monitoring and alerting.

**Features:**
- Custom events for configuration changes
- Performance metrics for configuration operations
- Error tracking and alerting
- Dashboard integration

**Setup:**

1. Get New Relic API key:
   ```
   New Relic → API Keys → Ingest - License → Create Key
   ```

2. Configure New Relic integration:
   ```python
   newrelic_config = {
       "account_id": "1234567",
       "api_key": "your-api-key",
       "region": "US",  # or "EU"
       "event_types": {
           "config_change": "ConfigurationChanged",
           "config_error": "ConfigurationError",
           "performance": "ConfigurationPerformance"
       },
       "custom_attributes": {
           "service": "central-configuration",
           "team": "platform",
           "environment": "production"
       }
   }
   await service.set_config("integrations.newrelic", newrelic_config)
   ```

**Custom Events:**
```json
{
  "eventType": "ConfigurationChanged",
  "timestamp": 1704067200,
  "configKey": "app.database.host",
  "oldValue": "localhost",
  "newValue": "prod-db.internal",
  "userId": "admin@company.com",
  "environment": "production",
  "changeReason": "database migration"
}
```

#### Splunk Integration

Sends structured logs and events to Splunk via HTTP Event Collector (HEC).

**Setup:**

1. Configure Splunk HEC:
   ```
   Splunk → Settings → Data Inputs → HTTP Event Collector → New Token
   ```

2. Configure Splunk integration:
   ```python
   splunk_config = {
       "hec_url": "https://splunk.company.com:8088/services/collector",
       "hec_token": "your-hec-token",
       "index": "apg_config",
       "sourcetype": "apg:configuration",
       "source": "central-configuration",
       "ssl_verify": True,
       "batch_size": 100,
       "flush_interval": 30
   }
   await service.set_config("integrations.splunk", splunk_config)
   ```

**Log Formats:**
```json
{
  "time": 1704067200,
  "event": {
    "action": "config_changed",
    "config_key": "app.database.host",
    "old_value": "localhost",
    "new_value": "prod-db.internal",
    "user_id": "admin@company.com",
    "tenant_id": "company",
    "metadata": {
      "environment": "production",
      "change_reason": "database migration"
    }
  }
}
```

### DevOps and CI/CD

#### Jenkins Integration

Triggers Jenkins builds and deployments when configuration changes occur.

**Features:**
- Automatic build triggers on configuration changes
- Parameterized builds with configuration context
- Build status feedback to configuration system
- Integration with Jenkins pipelines

**Setup:**

1. Configure Jenkins API access:
   ```
   Jenkins → Manage Jenkins → Configure Global Security → API Token
   ```

2. Configure Jenkins integration:
   ```python
   jenkins_config = {
       "base_url": "https://jenkins.company.com",
       "username": "automation",
       "api_token": "your-jenkins-token",
       "default_job": "config-deployment",
       "trigger_patterns": ["app.database.*", "app.cache.*"],
       "job_parameters": {
           "ENVIRONMENT": "production",
           "NOTIFY_TEAM": "platform"
       }
   }
   await service.set_config("integrations.jenkins", jenkins_config)
   ```

**Pipeline Example:**
```groovy
pipeline {
    agent any
    parameters {
        string(name: 'CONFIG_KEY', description: 'Configuration key that changed')
        string(name: 'CONFIG_VALUE', description: 'New configuration value')
        string(name: 'USER_ID', description: 'User who made the change')
    }
    stages {
        stage('Validate Config') {
            steps {
                script {
                    // Validate configuration change
                    sh "python validate_config.py ${params.CONFIG_KEY} '${params.CONFIG_VALUE}'"
                }
            }
        }
        stage('Deploy Config') {
            steps {
                script {
                    // Deploy configuration to target systems
                    sh "ansible-playbook deploy-config.yml -e config_key=${params.CONFIG_KEY}"
                }
            }
        }
    }
}
```

#### GitHub Integration

Manages GitHub repositories, creates issues, and automates workflows.

**Setup:**
```python
github_config = {
    "token": "ghp_your-github-token",
    "organization": "yourcompany",
    "repositories": {
        "config-tracking": "yourcompany/config-tracking",
        "infrastructure": "yourcompany/infrastructure"
    },
    "auto_create_issues": True,
    "issue_labels": ["configuration", "automated"],
    "workflow_triggers": {
        "deploy": ".github/workflows/deploy-config.yml"
    }
}
await service.set_config("integrations.github", github_config)
```

#### GitLab Integration

Similar to GitHub but optimized for GitLab's API and features.

**Setup:**
```python
gitlab_config = {
    "base_url": "https://gitlab.company.com",
    "token": "your-gitlab-token",
    "project_id": 12345,
    "auto_create_mrs": True,
    "mr_target_branch": "main",
    "pipeline_triggers": {
        "config-deploy": "config-deployment"
    }
}
await service.set_config("integrations.gitlab", gitlab_config)
```

### Identity and Access Management

#### Okta Integration

Synchronizes user information and manages access control.

**Features:**
- User and group synchronization
- Single Sign-On (SSO) integration
- Role-based access control mapping
- Automated user provisioning/deprovisioning

**Setup:**

1. Create Okta API token:
   ```
   Okta Admin → Security → API → Create Token
   ```

2. Configure Okta integration:
   ```python
   okta_config = {
       "domain": "yourcompany.okta.com",
       "api_token": "your-okta-token",
       "sync_groups": True,
       "group_mappings": {
           "Platform Team": "admin",
           "Development Team": "write",
           "QA Team": "read"
       },
       "sync_interval": 3600,  # 1 hour
       "user_attributes": ["email", "firstName", "lastName", "department"]
   }
   await service.set_config("integrations.okta", okta_config)
   ```

#### Auth0 Integration

Similar to Okta but for Auth0 identity platform.

**Setup:**
```python
auth0_config = {
    "domain": "yourcompany.auth0.com",
    "client_id": "your-client-id",
    "client_secret": "your-client-secret",
    "connection": "Username-Password-Authentication",
    "role_mappings": {
        "admin": "admin",
        "developer": "write",
        "viewer": "read"
    }
}
await service.set_config("integrations.auth0", auth0_config)
```

#### Azure Active Directory Integration

Enterprise-grade integration with Microsoft Azure AD.

**Setup:**
```python
azure_ad_config = {
    "tenant_id": "your-tenant-id",
    "client_id": "your-application-id",
    "client_secret": "your-client-secret",
    "graph_api_version": "v1.0",
    "sync_groups": True,
    "group_filter": "startswith(displayName,'APG_')",
    "user_attributes": [
        "id", "displayName", "mail", "jobTitle", "department"
    ]
}
await service.set_config("integrations.azure_ad", azure_ad_config)
```

## Custom Connectors

### Creating Custom Connectors

You can create custom connectors for your specific enterprise systems:

```python
from capabilities.composition.central_configuration.integrations.enterprise_connectors import BaseConnector

class CustomSystemConnector(BaseConnector):
    """Custom connector for your enterprise system"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_url = config["api_url"]
        self.api_key = config["api_key"]
    
    async def send_notification(self, event_data: Dict[str, Any]) -> bool:
        """Send notification to custom system"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "event_type": "configuration_changed",
            "data": event_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}/webhooks/config",
                json=payload,
                headers=headers
            )
            return response.status_code == 200
    
    async def sync_data(self) -> Dict[str, Any]:
        """Sync data from custom system"""
        # Implementation for data synchronization
        pass

# Register custom connector
connector = CustomSystemConnector({
    "api_url": "https://api.yoursystem.com",
    "api_key": "your-api-key"
})

await service.register_connector("custom_system", connector)
```

## Configuration Patterns

### Event-Driven Architecture

```python
# Configure automatic notifications
notification_config = {
    "triggers": {
        "critical_changes": {
            "patterns": ["*.database.*", "*.security.*", "*.payment.*"],
            "connectors": ["discord", "zendesk", "newrelic"],
            "urgency": "high"
        },
        "feature_flags": {
            "patterns": ["features.*"],
            "connectors": ["slack", "github"],
            "urgency": "medium"
        },
        "routine_changes": {
            "patterns": ["app.logging.*", "app.cache.*"],
            "connectors": ["splunk"],
            "urgency": "low"
        }
    }
}
await service.set_config("integrations.notifications", notification_config)
```

### Approval Workflows

```python
# Configure approval workflows for sensitive changes
approval_config = {
    "required_approvals": {
        "*.database.*": {
            "approvers": ["platform-team@company.com", "dba-team@company.com"],
            "min_approvals": 2,
            "timeout": 3600  # 1 hour
        },
        "*.security.*": {
            "approvers": ["security-team@company.com"],
            "min_approvals": 1,
            "timeout": 1800  # 30 minutes
        }
    },
    "approval_methods": ["email", "slack", "jira"]
}
await service.set_config("integrations.approvals", approval_config)
```

### Multi-Environment Sync

```python
# Sync configurations across environments
sync_config = {
    "environments": {
        "staging": {
            "endpoint": "https://staging-config.company.com",
            "api_key": "staging-key",
            "sync_patterns": ["app.*", "features.*"]
        },
        "production": {
            "endpoint": "https://prod-config.company.com",
            "api_key": "prod-key",
            "sync_patterns": ["app.*"],
            "approval_required": True
        }
    },
    "sync_interval": 300,  # 5 minutes
    "conflict_resolution": "manual"
}
await service.set_config("integrations.multi_env_sync", sync_config)
```

## Best Practices

### Security

1. **Use Environment Variables for Secrets:**
   ```python
   import os
   
   # Don't store secrets in configuration
   zendesk_config = {
       "subdomain": "yourcompany",
       "email": "admin@yourcompany.com",
       "api_token": os.getenv("ZENDESK_API_TOKEN")  # From environment
   }
   ```

2. **Rotate API Keys Regularly:**
   ```python
   # Implement key rotation
   async def rotate_api_keys():
       connectors = ["zendesk", "newrelic", "github"]
       for connector in connectors:
           new_key = await generate_new_api_key(connector)
           await service.set_config(f"integrations.{connector}.api_key", new_key)
   ```

### Error Handling

```python
try:
    await service.set_config("app.database.host", "new-host")
except IntegrationError as e:
    # Handle integration failures gracefully
    logger.error(f"Integration failed: {e}")
    # Continue with local configuration change
    await service.set_config_local_only("app.database.host", "new-host")
```

### Performance

1. **Batch Notifications:**
   ```python
   # Configure batching for high-volume changes
   batch_config = {
       "batch_size": 10,
       "batch_timeout": 30,
       "max_retries": 3
   }
   await service.set_config("integrations.batching", batch_config)
   ```

2. **Async Processing:**
   ```python
   # Use async processing for non-critical notifications
   async def process_notifications_async(changes):
       tasks = []
       for change in changes:
           task = asyncio.create_task(send_notification(change))
           tasks.append(task)
       await asyncio.gather(*tasks, return_exceptions=True)
   ```

## Troubleshooting

### Common Issues

#### Authentication Failures
```bash
# Test API connectivity
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://api.service.com/test

# Check token expiration
python -c "
import jwt
token = 'YOUR_JWT_TOKEN'
decoded = jwt.decode(token, options={'verify_signature': False})
print(f'Expires: {decoded.get('exp')}')
"
```

#### Rate Limiting
```python
# Implement exponential backoff
import asyncio
from random import uniform

async def call_api_with_backoff(api_call, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await api_call()
        except RateLimitError:
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + uniform(0, 1)
                await asyncio.sleep(delay)
            else:
                raise
```

#### Webhook Failures
```python
# Implement webhook retry logic
async def send_webhook_with_retry(url, payload, max_retries=3):
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=10)
                if response.status_code == 200:
                    return True
        except Exception as e:
            logger.warning(f"Webhook attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    return False
```

### Monitoring Integration Health

```python
# Monitor integration health
async def check_integration_health():
    results = {}
    
    for connector_name in ["discord", "zendesk", "newrelic"]:
        try:
            connector = await service.get_connector(connector_name)
            health = await connector.health_check()
            results[connector_name] = {
                "status": "healthy" if health else "unhealthy",
                "last_check": datetime.utcnow().isoformat()
            }
        except Exception as e:
            results[connector_name] = {
                "status": "error",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }
    
    return results
```

## Next Steps

- Set up [Real-time Synchronization](../advanced/realtime-sync.md)
- Configure [Security and Authentication](../security/authentication.md)
- Learn about [Machine Learning Integration](../advanced/ml-models.md)
- Read [Troubleshooting Guide](../troubleshooting/common-issues.md)