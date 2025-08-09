# API Registration Guide

This guide walks you through the complete process of registering, configuring, and managing APIs in the Integration API Management platform.

## Overview

API registration is the first step in exposing your services through the APG platform. The registration process includes:

1. **API Definition**: Basic metadata and configuration
2. **Endpoint Configuration**: Define available endpoints and methods
3. **Security Settings**: Authentication and authorization requirements
4. **Policy Assignment**: Rate limiting, transformation, and validation rules
5. **Activation**: Making the API available to consumers

## Prerequisites

- Active APG tenant account
- Service deployed and accessible
- Administrative permissions for API management

## Step 1: Basic API Registration

### Using the Python SDK

```python
from integration_api_management import IntegrationAPIManagementCapability
from integration_api_management.models import APIConfig

# Initialize the capability
capability = IntegrationAPIManagementCapability()
await capability.initialize()

# Define your API configuration
api_config = APIConfig(
    api_name="user_management_api",
    api_title="User Management API",
    api_description="Comprehensive user management and authentication service",
    version="1.0.0",
    protocol_type="rest",
    base_path="/api/users/v1",
    upstream_url="http://user-service.internal:8000",
    is_public=False,
    timeout_ms=30000,
    retry_attempts=3,
    auth_type="api_key",
    category="core_business",
    tags=["users", "authentication", "management"]
)

# Register the API
api_id = await capability.api_service.register_api(
    config=api_config,
    tenant_id="your_tenant_id",
    created_by="admin@yourcompany.com"
)

print(f"API registered successfully with ID: {api_id}")
```

### Using the REST API

```bash
curl -X POST https://api-management.yourcompany.com/api/v1/apis/ \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -H "X-Tenant-ID: your_tenant_id" \
  -d '{
    "api_name": "user_management_api",
    "api_title": "User Management API",
    "api_description": "Comprehensive user management and authentication service",
    "version": "1.0.0",
    "protocol_type": "rest",
    "base_path": "/api/users/v1",
    "upstream_url": "http://user-service.internal:8000",
    "is_public": false,
    "timeout_ms": 30000,
    "retry_attempts": 3,
    "auth_type": "api_key",
    "category": "core_business",
    "tags": ["users", "authentication", "management"]
  }'
```

## Step 2: Endpoint Configuration

After registering the base API, you can define specific endpoints:

### Adding Endpoints

```python
from integration_api_management.models import EndpointConfig

# Define endpoints
endpoints = [
    EndpointConfig(
        path="/users",
        method="GET",
        summary="List users",
        description="Retrieve a paginated list of users",
        auth_required=True,
        rate_limit=1000,  # requests per minute
        cache_ttl=300,    # 5 minutes
        request_schema={
            "type": "object",
            "properties": {
                "page": {"type": "integer", "minimum": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100}
            }
        },
        response_schema={
            "type": "object",
            "properties": {
                "users": {"type": "array"},
                "total": {"type": "integer"},
                "page": {"type": "integer"}
            }
        }
    ),
    EndpointConfig(
        path="/users",
        method="POST",
        summary="Create user",
        description="Create a new user account",
        auth_required=True,
        rate_limit=100,   # Lower limit for write operations
        request_schema={
            "type": "object",
            "required": ["email", "name"],
            "properties": {
                "email": {"type": "string", "format": "email"},
                "name": {"type": "string", "minLength": 1},
                "role": {"type": "string", "enum": ["user", "admin"]}
            }
        }
    ),
    EndpointConfig(
        path="/users/{user_id}",
        method="GET",
        summary="Get user",
        description="Retrieve a specific user by ID",
        auth_required=True,
        cache_ttl=600,    # 10 minutes
        path_parameters={
            "user_id": {"type": "string", "pattern": "^[0-9a-f-]+$"}
        }
    )
]

# Add endpoints to the API
for endpoint_config in endpoints:
    endpoint_id = await capability.api_service.add_endpoint(
        api_id=api_id,
        config=endpoint_config,
        tenant_id="your_tenant_id",
        created_by="admin@yourcompany.com"
    )
    print(f"Endpoint added: {endpoint_config.method} {endpoint_config.path}")
```

## Step 3: Security Configuration

### Authentication Types

#### API Key Authentication

```python
# API key authentication is already configured in the base API
# No additional configuration needed
```

#### OAuth 2.0 Configuration

```python
oauth_config = {
    "auth_type": "oauth2",
    "oauth_config": {
        "authorization_url": "https://auth.yourcompany.com/oauth/authorize",
        "token_url": "https://auth.yourcompany.com/oauth/token",
        "scopes": {
            "read_users": "Read user information",
            "write_users": "Create and modify users",
            "admin_users": "Full user management access"
        },
        "client_id": "your_client_id",
        "audience": "user_management_api"
    }
}

# Update API with OAuth configuration
await capability.api_service.update_api_configuration(
    api_id=api_id,
    updates=oauth_config,
    tenant_id="your_tenant_id",
    updated_by="admin@yourcompany.com"
)
```

#### JWT Token Validation

```python
jwt_config = {
    "auth_type": "jwt",
    "jwt_config": {
        "issuer": "https://auth.yourcompany.com",
        "audience": "user_management_api",
        "algorithm": "RS256",
        "jwks_url": "https://auth.yourcompany.com/.well-known/jwks.json",
        "validate_expiration": True,
        "validate_signature": True,
        "required_claims": ["sub", "exp", "iat"]
    }
}

await capability.api_service.update_api_configuration(
    api_id=api_id,
    updates=jwt_config,
    tenant_id="your_tenant_id",
    updated_by="admin@yourcompany.com"
)
```

## Step 4: Policy Configuration

### Rate Limiting Policy

```python
from integration_api_management.models import PolicyConfig, PolicyType

rate_limit_policy = PolicyConfig(
    policy_name="user_api_rate_limit",
    policy_type=PolicyType.RATE_LIMITING,
    config={
        "requests_per_minute": 1000,
        "requests_per_hour": 10000,
        "requests_per_day": 100000,
        "burst_size": 100,
        "key_extraction": "consumer_id",  # Rate limit per consumer
        "violation_action": "reject",     # reject or throttle
        "response_headers": True          # Include rate limit headers
    },
    execution_order=100,
    enabled=True
)

policy_id = await capability.policy_service.create_policy(
    api_id=api_id,
    config=rate_limit_policy,
    tenant_id="your_tenant_id",
    created_by="admin@yourcompany.com"
)
```

### Request Transformation Policy

```python
transformation_policy = PolicyConfig(
    policy_name="user_api_transformation",
    policy_type=PolicyType.TRANSFORMATION,
    config={
        "request_transformations": [
            {
                "condition": "method == 'POST'",
                "transformations": [
                    {
                        "type": "add_header",
                        "header": "X-Source",
                        "value": "api_gateway"
                    },
                    {
                        "type": "lowercase_field",
                        "field": "email"
                    }
                ]
            }
        ],
        "response_transformations": [
            {
                "condition": "status_code == 200",
                "transformations": [
                    {
                        "type": "remove_field",
                        "field": "internal_id"
                    },
                    {
                        "type": "add_header",
                        "header": "X-API-Version",
                        "value": "1.0.0"
                    }
                ]
            }
        ]
    },
    execution_order=200
)

await capability.policy_service.create_policy(
    api_id=api_id,
    config=transformation_policy,
    tenant_id="your_tenant_id",
    created_by="admin@yourcompany.com"
)
```

### Validation Policy

```python
validation_policy = PolicyConfig(
    policy_name="user_api_validation",
    policy_type=PolicyType.VALIDATION,
    config={
        "validate_request": True,
        "validate_response": True,
        "strict_mode": True,
        "custom_validators": [
            {
                "name": "email_domain_check",
                "condition": "path == '/users' and method == 'POST'",
                "validation": {
                    "type": "custom",
                    "script": """
                        if 'email' in request.json:
                            domain = request.json['email'].split('@')[1]
                            if domain not in ['yourcompany.com', 'trusted-partner.com']:
                                raise ValidationError('Email domain not allowed')
                    """
                }
            }
        ]
    },
    execution_order=50  # Execute before other policies
)

await capability.policy_service.create_policy(
    api_id=api_id,
    config=validation_policy,
    tenant_id="your_tenant_id",
    created_by="admin@yourcompany.com"
)
```

## Step 5: Load Balancing Configuration

```python
load_balancing_config = {
    "load_balancing": {
        "strategy": "round_robin",  # round_robin, weighted, least_connections, ip_hash
        "health_check": {
            "enabled": True,
            "path": "/health",
            "interval_seconds": 30,
            "timeout_seconds": 5,
            "healthy_threshold": 2,
            "unhealthy_threshold": 3
        },
        "upstream_servers": [
            {
                "url": "http://user-service-1.internal:8000",
                "weight": 1,
                "max_connections": 100
            },
            {
                "url": "http://user-service-2.internal:8000",
                "weight": 1,
                "max_connections": 100
            },
            {
                "url": "http://user-service-3.internal:8000",
                "weight": 2,  # Higher weight = more traffic
                "max_connections": 200
            }
        ]
    }
}

await capability.api_service.update_api_configuration(
    api_id=api_id,
    updates=load_balancing_config,
    tenant_id="your_tenant_id",
    updated_by="admin@yourcompany.com"
)
```

## Step 6: Monitoring and Analytics

### Enable Analytics Collection

```python
analytics_config = {
    "analytics": {
        "enabled": True,
        "collect_request_body": False,    # For privacy
        "collect_response_body": False,   # For privacy
        "collect_headers": ["User-Agent", "X-Forwarded-For"],
        "sample_rate": 1.0,               # Collect 100% of requests
        "retention_days": 90,
        "custom_dimensions": [
            "user_type",
            "client_version",
            "request_source"
        ]
    }
}

await capability.api_service.update_api_configuration(
    api_id=api_id,
    updates=analytics_config,
    tenant_id="your_tenant_id",
    updated_by="admin@yourcompany.com"
)
```

### Set Up Alerts

```python
from integration_api_management.models import AlertRule, AlertSeverity

# High error rate alert
error_rate_alert = AlertRule(
    rule_id="user_api_high_error_rate",
    rule_name="User API High Error Rate",
    description="Alert when error rate exceeds 5%",
    metric_name="api_error_rate",
    condition="greater_than",
    threshold=0.05,
    severity=AlertSeverity.WARNING,
    evaluation_interval_seconds=60,
    notification_channels=["email", "slack"],
    metadata={
        "api_id": api_id,
        "runbook_url": "https://wiki.yourcompany.com/runbooks/user-api-errors"
    }
)

await capability.alert_manager.create_alert_rule(error_rate_alert)

# High latency alert
latency_alert = AlertRule(
    rule_id="user_api_high_latency",
    rule_name="User API High Latency",
    description="Alert when 95th percentile latency exceeds 2 seconds",
    metric_name="api_latency_p95",
    condition="greater_than",
    threshold=2000,  # milliseconds
    severity=AlertSeverity.CRITICAL,
    evaluation_interval_seconds=60,
    notification_channels=["email", "slack", "pagerduty"]
)

await capability.alert_manager.create_alert_rule(latency_alert)
```

## Step 7: API Activation

Once everything is configured, activate the API to make it available to consumers:

```python
# Activate the API
success = await capability.api_service.activate_api(
    api_id=api_id,
    tenant_id="your_tenant_id",
    activated_by="admin@yourcompany.com"
)

if success:
    print("API activated successfully!")
    
    # Get the API details to confirm activation
    api_details = await capability.api_service.get_api(api_id, "your_tenant_id")
    print(f"API Status: {api_details.status}")
    print(f"Gateway URL: {api_details.gateway_url}")
else:
    print("API activation failed!")
```

## Testing Your API

### Health Check

```bash
curl -X GET https://api-gateway.yourcompany.com/api/users/v1/health \
  -H "X-API-Key: YOUR_API_KEY"
```

### Test Endpoint

```bash
curl -X GET https://api-gateway.yourcompany.com/api/users/v1/users?page=1&limit=10 \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json"
```

### Check Rate Limits

```bash
# The response should include rate limit headers
curl -I https://api-gateway.yourcompany.com/api/users/v1/users \
  -H "X-API-Key: YOUR_API_KEY"

# Look for headers like:
# X-RateLimit-Limit: 1000
# X-RateLimit-Remaining: 999
# X-RateLimit-Reset: 1640995200
```

## Best Practices

### API Design

1. **Use Clear Naming**: Choose descriptive names for APIs and endpoints
2. **Follow REST Conventions**: Use standard HTTP methods and status codes
3. **Version Your APIs**: Always include version information
4. **Document Everything**: Provide comprehensive descriptions and examples

### Security

1. **Principle of Least Privilege**: Grant minimum necessary permissions
2. **Use Strong Authentication**: Prefer OAuth 2.0 or JWT over simple API keys
3. **Validate All Input**: Never trust client input
4. **Monitor for Abuse**: Set up alerts for unusual patterns

### Performance

1. **Set Appropriate Timeouts**: Balance user experience with resource usage
2. **Implement Caching**: Use cache-friendly patterns where possible
3. **Monitor Metrics**: Track performance and usage patterns
4. **Plan for Scale**: Design with growth in mind

### Operational

1. **Health Checks**: Implement comprehensive health endpoints
2. **Logging**: Log important events for debugging and auditing
3. **Gradual Rollouts**: Use deployment strategies to minimize risk
4. **Documentation**: Keep documentation up-to-date with changes

## Common Issues and Solutions

### API Not Accessible

**Problem**: API returns 404 or connection errors

**Solutions**:
- Verify API status is "active"
- Check upstream service is running and accessible
- Validate base_path configuration
- Ensure network connectivity between gateway and upstream

### Authentication Failures

**Problem**: API returns 401 Unauthorized

**Solutions**:
- Verify API key is valid and active
- Check consumer has access to the API
- Validate authentication configuration
- Review policy execution order

### High Latency

**Problem**: API responses are slow

**Solutions**:
- Check upstream service performance
- Review timeout configurations
- Analyze policy execution overhead
- Consider caching strategies

### Rate Limit Issues

**Problem**: Requests being throttled unexpectedly

**Solutions**:
- Review rate limit configurations
- Check if limits are per-consumer or global
- Verify burst size settings
- Monitor usage patterns

## Next Steps

- [Consumer Management](./consumer-onboarding.md): Learn how to onboard API consumers
- [Policy Management](./policy-management.md): Advanced policy configuration
- [Analytics](./analytics.md): Understanding API usage patterns
- [Versioning](./api-versioning.md): Managing API versions and migrations

## Support

For additional help with API registration:

- **Documentation**: [https://docs.datacraft.co.ke/apg/api-registration](https://docs.datacraft.co.ke/apg/api-registration)
- **Support Team**: [api-support@datacraft.co.ke](mailto:api-support@datacraft.co.ke)
- **Community Forum**: [https://community.datacraft.co.ke](https://community.datacraft.co.ke)