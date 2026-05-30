# REGY User Guide

REGY is the APG API and service registry. Use it to register services,
register service instances, discover healthy endpoints, govern versions,
publish eligible services to gateway adapters, and retire services with audit
evidence.

## Register A Service

```python
from capabilities.common.regy.registry_runtime import RegistryService

registry = RegistryService()
registry.register_service(
    service_id="orders",
    tenant_id="tenant-a",
    name="orders",
    owner="platform",
    service_type="rest_api",
    environment="production",
    api_version="1.0.0",
    contract_schema_ref="schemas/orders-openapi.yaml",
    health_endpoint="/health",
    routing_metadata={"path": "/orders", "strategy": "weighted"},
    production_review_recorded=True,
    trace_propagation_configured=True,
)
```

## Register An Instance

```python
registry.register_instance(
    instance_id="orders-1",
    tenant_id="tenant-a",
    service_id="orders",
    endpoint="https://orders.internal",
    region="edge-africa",
    health_probe="/health",
    weight=100,
)
```

## Discover Services

```python
result = registry.discover_services("tenant-a", service_name="orders")
assert result["total_count"] == 1
```

Cross-tenant discovery is denied unless a future adapter explicitly records
review evidence and policy allows it.

## Publish To Gateway

```python
registry.publish_to_gateway(
    publication_id="orders-public",
    tenant_id="tenant-a",
    service_id="orders",
    route_path="/orders",
)
```

Gateway publication requires a registered service, at least one healthy
instance, and routing metadata.

## Retire A Service

```python
registry.retire_service(
    tenant_id="tenant-a",
    service_id="orders",
    actor="platform",
    impact_review_recorded=True,
    gateway_unpublish_recorded=True,
)
```

Retirement is denied until impact review and gateway unpublish evidence are
recorded.
