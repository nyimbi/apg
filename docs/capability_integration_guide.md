# APG Capability Integration Guide

**How to integrate APG capabilities into your application — standalone, platform, and composed.**

---

## 1. Integration Modes

APG capabilities support three integration modes. Choose based on your deployment:

| Mode | Use when | Deps |
|---|---|---|
| **Standalone** | Single capability, no platform | `pip install apg-<cap>` only |
| **Platform** | Multiple capabilities, shared auth/audit | Full APG platform stack |
| **Composed** | Custom application of N capabilities | Selective capability packages |

---

## 2. Standalone Integration

Install a single capability and run it as an independent HTTP service:

```bash
pip install apg-intel-alerts
apg-intel-alerts --port 8080 --tenant my_org
```

Or use the Python API directly:

```python
from apg_intel_alerts import get_capability_contract, evaluate_capability_rules
from apg_intel_alerts.app import create_app
from apg_intel_alerts.service import AlertManagementService

# Zero dependencies — InMemoryStore + null adapters
svc = AlertManagementService(tenant_id="my_org")

# Execute business logic
result = svc.record_alert(
    alert_id="a1", tenant_id="my_org",
    alert_type="threat_alert", severity="critical",
    title="Suspicious login pattern",
    signal_id="sig-001", evidence_reference="OSINT-2026-001"
)

# Check governance rules before an operation
ctx = {"tenant_context_present": True, "operation": "record_alert", "policy_attached": True}
decision = evaluate_capability_rules(ctx)
# {"decision": "allow", "matched_rules": [], "actions": []}
```

---

## 3. Platform Integration (shared adapters)

When running inside the APG platform, inject real adapters to get auth, audit, and notifications:

```python
from apg_intel_alerts.service import AlertManagementService
from apg_intel_alerts.domain.adapters import get_auth_adapter, get_audit_adapter, get_notify_adapter
from apg_intel_alerts.database.store import get_store

# Wire real platform capabilities as adapters
svc = AlertManagementService(
    tenant_id="my_org",
    actor_id="analyst_001",
    auth=get_auth_adapter(),    # uses apg-common-auth if installed
    audit=get_audit_adapter(),  # uses apg-common-audl if installed
    notify=get_notify_adapter(),# uses apg-common-ntfy if installed
    db_url="postgresql+asyncpg://user:pass@localhost/apg",  # PostgreSQL
)
```

The adapter factories auto-discover installed platform capabilities:

```python
# get_auth_adapter() resolves in priority order:
# 1. Explicitly passed auth_service parameter
# 2. Installed apg-common-auth package (via importlib)
# 3. NullAuthAdapter (standalone fallback — grants all)
```

---

## 4. Database Backend

Switch from in-memory to PostgreSQL by setting `APG_DATABASE_URL`:

```bash
export APG_DATABASE_URL="postgresql+asyncpg://apg:secret@localhost/apg_prod"
apg-intel-alerts --port 8080
```

Or pass explicitly:

```python
svc = AlertManagementService(
    tenant_id="my_org",
    db_url="postgresql+asyncpg://apg:secret@localhost/apg_prod"
)
```

Run the schema migration first:

```bash
cd capabilities/intel/alerts
export APG_DATABASE_URL="postgresql+asyncpg://..."
alembic upgrade head
```

---

## 5. Consuming Capabilities via the Registry

The registry discovers all installed capabilities automatically:

```python
from capabilities.capability_contract_registry import (
    load_contract_registry,
    validate_contract_registry,
    evaluate_rules,
)

# Discover all installed + filesystem capabilities
registry = load_contract_registry()

# Inspect a capability
cap = registry["intel_alerts"]
print(cap.display_name)         # "Alert Management"
print(cap.contract["provides"]) # ["alert_authority_workflow", ...]
print(cap.contract["requires"]) # ["auth", "audl", "ntfy", ...]

# Validate all capabilities (CI/CD gate)
report = validate_contract_registry()
assert report["valid"], report["errors"]

# Evaluate governance rules
result = evaluate_rules("intel_alerts", {
    "tenant_context_present": True,
    "operation": "record_alert",
    "policy_attached": True,
})
# {"decision": "allow", "matched_rules": [], "actions": []}
```

---

## 6. Bidirectional Navigation via Manifest

```python
from capabilities.manifest import (
    get_capability,    # by capability ID
    get_by_path,       # by filesystem path
    get_by_package,    # by PyPI package name
    get_domain,        # all in a domain
    find_capabilities, # keyword search
    all_capabilities,
)

# Code → description
cap = get_capability("intel_alerts")
print(cap["display_name"])       # "Alert Management"
print(cap["provides"])           # ["alert_authority_workflow", ...]
print(cap["service_methods"])    # ["record_authority", "record_alert", ...]
print(cap["governance_rules"])   # ["tenant_context_required", ...]
print(cap["install"])            # "pip install apg-intel-alerts"

# Path → capability
cap = get_by_path("capabilities/intel/alerts")

# Package → capability
cap = get_by_package("apg-intel-alerts")

# Description → code (search)
results = find_capabilities("alert")
# Returns all capabilities whose id/name/provides/methods mention "alert"

# All in a domain
intel_caps = get_domain("intel")
for c in intel_caps:
    print(c["id"], c["service_method_count"])
```

---

## 7. Composing Multiple Capabilities

When your application needs multiple capabilities, wire them through shared adapters to get unified auth, audit, and event streams:

```python
import asyncio
from apg_intel_alerts.service import AlertManagementService
from apg_intel_osint.service import OSINTService
from apg_intel_threats.service import ThreatIntelligenceService
from apg_common_auth.service import AuthService
from apg_common_audl.service import AuditService

# Shared platform services
auth = AuthService.from_env()
audit = AuditService.from_env()

# All capabilities share the same auth/audit context
tenant = "my_org"
alerts = AlertManagementService(tenant, auth=auth, audit=audit)
osint  = OSINTService(tenant, auth=auth, audit=audit)
threats = ThreatIntelligenceService(tenant, auth=auth, audit=audit)

async def intelligence_pipeline(collection_task: dict):
    # 1. Collect OSINT
    item = await osint.collect_from_source(
        source_type="web", url=collection_task["url"], depth=2
    )
    
    # 2. Extract threat indicators
    indicators = await threats.enrich_indicator(item["id"])
    
    # 3. Fire alert if high confidence indicator found
    if any(i["confidence"] > 0.8 for i in indicators.get("indicators", [])):
        alert = await alerts.record_alert(
            alert_id=f"auto-{item['id']}",
            tenant_id=tenant,
            alert_type="threat_alert",
            severity="high",
            title=f"High-confidence IOC detected: {item['subject']}",
            signal_id=item["id"],
        )
    
    return indicators

asyncio.run(intelligence_pipeline({"url": "https://example.com/threat-feed"}))
```

---

## 8. Event-Driven Integration via Bytewax

Capabilities emit events to their Bytewax stream. Subscribe to consume:

```python
# Example: AML monitors the payments stream for suspicious activity
# capabilities/fintech/aml/integration.py

from bytewax.dataflow import Dataflow
from bytewax.inputs import KafkaInputConfig  # or RedisInputConfig

def monitor_payments():
    flow = Dataflow()
    
    # Subscribe to payments lifecycle stream
    flow.input("payments", KafkaInputConfig(
        brokers=["kafka:9092"],
        topics=["apg.fintech.lifecycle"],
        tail=True,
    ))
    
    # Filter payment_completed events
    flow.filter("payment_events", lambda item: 
        item.get("event_type") == "payment_completed"
    )
    
    # AML screening
    flow.map("aml_screen", lambda payment:
        aml_service.screen_transaction(payment["transaction_id"])
    )
    
    # Alert on suspicious transactions
    flow.filter("suspicious", lambda result: result["risk_score"] > 0.7)
    flow.output("alerts", alert_service.raise_alert)
    
    return flow
```

---

## 9. APG Source Integration (Compiled Apps)

Compose capabilities declaratively in APG and compile to a deployable Python app:

```bash
# Compile a composed application
apg compile examples/crm_platform/main.apg \
    --output /tmp/crm_app \
    --verify

# Verify it compiles and smoke-tests pass
python /tmp/crm_app/smoke_test.py

# Run the generated app
python /tmp/crm_app/app.py --host 0.0.0.0 --port 8080
```

The compiled app includes:
- `app.py` — HTTP server with all routes registered
- `apg_capabilities.py` — capability contracts and rule evaluation
- `apg_application.py` — composition graph and screen routing
- `semantic_model.json` — machine-readable application model
- `smoke_test.py` — automated runtime verification
- `Dockerfile` — container packaging

---

## 10. Testing Integration

### Contract Validation (CI/CD gate)

```python
# tests/test_capability_contracts.py
from capabilities.capability_contract_registry import validate_contract_registry

def test_all_contracts_valid():
    """All capability contracts must pass shape validation."""
    report = validate_contract_registry()
    assert report["valid"], f"Contract errors: {report['errors']}"
    assert report["contract_count"] == 259
```

### Rule Evaluation Tests

```python
# tests/test_governance_rules.py
from capabilities.capability_contract_registry import evaluate_rules

def test_unverified_kyc_denied():
    result = evaluate_rules("fintech_platform_core", {
        "kyc_status": "pending",
        "operation": "initiate_payment",
    })
    assert result["decision"] == "deny"
    assert "kyc_required_for_payment" in result["matched_rules"]

def test_verified_customer_allowed():
    result = evaluate_rules("fintech_platform_core", {
        "kyc_status": "verified",
        "kyc_tier": "standard",
        "operation": "initiate_payment",
        "transaction_amount": 10000,
        "daily_used": 0,
    })
    assert result["decision"] == "allow"
```

### Service Integration Tests

```python
# tests/test_alert_service.py
import asyncio
import pytest
from apg_intel_alerts.service import AlertManagementService

@pytest.fixture
def svc():
    return AlertManagementService(tenant_id="test_tenant")

def test_record_and_retrieve_alert(svc):
    result = svc.record_alert(
        alert_id="a1", tenant_id="test_tenant",
        alert_type="watchlist_hit", severity="high",
        title="Test alert", signal_id="s1",
    )
    assert result["alert_id"] == "a1"
    assert result["status"] == "new"
    
    summary = svc.dashboard_summary("test_tenant")
    assert summary["alerts"]["total"] >= 1

def test_rule_blocks_missing_tenant(svc):
    with pytest.raises(PermissionError, match="tenant_context_required"):
        svc.record_alert(
            alert_id="a2", tenant_id="",
            alert_type="threat_alert", severity="critical",
            title="Bad", signal_id="s2",
        )
```

### Composability Tests

```python
# tests/test_composability.py
from capabilities.manifest import get_capability, find_capabilities

def test_all_requires_are_satisfied():
    """Every capability's requires list must be a known capability ID."""
    from capabilities.capability_contract_registry import load_contract_registry
    registry = load_contract_registry()
    all_ids = set(registry.keys())
    
    for cap_id, record in registry.items():
        for req in record.contract.get("requires", []):
            assert req in all_ids, f"{cap_id} requires unknown {req}"

def test_foundation_tier_present():
    """Foundation capabilities must be present."""
    from capabilities.manifest import get_capability
    for cap_id in ["audl", "auth", "ntfy", "conf", "mten", "mqeb"]:
        cap = get_capability(cap_id)
        assert cap is not None, f"Foundation capability {cap_id} missing"
```

---

## 11. Deployment Checklist

Before deploying a composed APG application:

```bash
# 1. Validate all contracts
python -c "
from capabilities.capability_contract_registry import validate_contract_registry
r = validate_contract_registry()
print(f'{r[\"contract_count\"]} contracts: {\"OK\" if r[\"valid\"] else r[\"errors\"]}'
"

# 2. Build all packages
./scripts/build_all_packages.sh

# 3. Compile your APG source
apg compile main.apg --output ./generated --verify

# 4. Run smoke test
python ./generated/smoke_test.py

# 5. Start the app
python ./generated/app.py --host 0.0.0.0 --port 8080

# 6. Verify health
curl http://localhost:8080/health
curl http://localhost:8080/contract | python -m json.tool | head -20
```
