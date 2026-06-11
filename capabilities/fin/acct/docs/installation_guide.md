# Bank Account Management — Installation Guide

## APG Platform (Integrated Mode)

The capability is auto-discovered when the `fin` package is imported.  No separate installation step is needed within an APG deployment.

### Verify Registration

```python
from capabilities.fin import acct
print(acct.CAPABILITY_META)
```

### Register Flask Blueprint

```python
from capabilities.fin.acct.api import bp
app.register_blueprint(bp)
```

## Standalone Mode

```bash
pip install uuid6 pydantic flask
```

```python
from capabilities.fin.acct.service import BankAccountService
import asyncio

svc = BankAccountService()
asyncio.run(svc.open_account("t1", "c1", "CURR001", "KES"))
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `pydantic>=2.0` | Models and validation |
| `uuid6` | UUID v7 ID generation |
| `flask` | REST API blueprint |
| `capabilities.common.reliability` | guard_tenant_id, CircuitBreaker, BoundedCache |

## Configuration

The capability reads from `capability_contract.DEFAULT_CONFIGURATION`. Override at startup:

```python
from capabilities.fin.acct import capability_contract
capability_contract.DORMANCY_THRESHOLD_DAYS = 90
```

## Production Checklist

- [ ] Replace in-memory stores with PostgreSQL repositories (see `domain/adapters.py`)
- [ ] Wire `LoggingEventPublisher` to NATS (`apg.fin.acct.lifecycle`)
- [ ] Wire `NoOpGLAdapter` to `fin.glr` service
- [ ] Configure `CircuitBreaker` reset_timeout for your GL SLA
- [ ] Set `X-Tenant-ID` enforcement at API gateway level
- [ ] Enable HTTPS / TLS termination at load balancer
