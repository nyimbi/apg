# APG — Application Programming Generation

**259 production-grade, independently-deployable capability packages across 28 business domains.**

> Copyright (c) 2025 Datacraft · Author: Nyimbi Odero · [www.datacraft.co.ke](https://www.datacraft.co.ke)

---

## What is APG?

APG is two things:

1. **A capability library** — 259 world-class business capability packages (Finance, Fintech, Intelligence, Healthcare, Government, Transport, Telecom, and more) that can be installed independently or composed together.

2. **A DSL** — An ultra-terse language for declaring composed business systems that compiles to production Python.

---

## Quick Start — Use a Capability

```bash
# Install any capability independently
pip install apg-intel-alerts
pip install apg-fin-arc          # Accounts Receivable
pip install apg-hcm-payroll      # Payroll (7-country PAYE)
pip install apg-fintech-payments # M-Pesa/MTN MoMo/SWIFT

# Run as standalone server (InMemory store, zero config)
apg-intel-alerts --port 8080

# Run with PostgreSQL
apg-fin-arc --db-url postgresql+asyncpg://... --port 8080

# Use the Python API
from apg_intel_alerts import get_capability_contract, evaluate_capability_rules
from apg_intel_alerts.service import AlertManagementService

svc = AlertManagementService("my_org")        # zero external deps
contract = get_capability_contract("my_org")  # composability interface
```

---

## 259 Capabilities Across 28 Domains

| Domain | Capabilities | Highlights |
|---|---|---|
| **common** | 81 | auth, audl, nlpc (7,844L), moni, wflo, rag, search, ML lifecycle |
| **fintech** | 30 | payments (M-Pesa/SWIFT), KYC (3,063L), AML, lending, crypto, DeFi |
| **intel** | 22 | OSINT, SIGINT, threats, fusion, alerts, correlation, prediction |
| **fin** | 7 | AR (2,251L), GL (2,987L), AP (1,963L), payroll (3,021L) |
| **healthcare** | 9 | EMR, pharmacy, lab, patient mgmt, telemedicine |
| **government** | 10 | tax, electoral, emergency, law enforcement, citizen services |
| **transport** | 9 | fleet, routing, dispatch, cargo, tracking |
| **telecom** | 10 | billing (2,143L), provisioning, network, QoS |
| **pharma** | 9 | clinical trials, pharmacovigilance, QMS |
| **realestate** | 10 | lease (IFRS 16), property mgmt, valuation |
| **retail** | 5 | POS (2,147L), omnichannel, loyalty |
| **bia** | 8 | analytics, dashboards, data warehouse, ML |
| **ppm** | 6 | project accounting, planning, resource mgmt |
| **hcm** | 3 | payroll, time & attendance, employee data |
| **grc** | 6 | policy, risk assessment, audit, incident mgmt |
| _+ 13 more_ | 39 | energy, mining, education, pharma, loc, mob, eam, ecd, crm, scm, int, composition |

**Quality**: All 259 capabilities have 40–184 async service methods, 10–100 governance rules, streaming events via Bytewax, and standalone PyPI packaging.

---

## Composability Contract

Every capability exposes a machine-readable contract:

```python
from capabilities.capability_contract_registry import evaluate_rules, load_contract_registry

# Discover all capabilities
registry = load_contract_registry()

# Evaluate governance rules
result = evaluate_rules("intel_alerts", {
    "tenant_context_present": True,
    "operation": "record_alert",
    "policy_attached": True,
})
# → {"decision": "allow", "matched_rules": [], "actions": []}

# Navigate the manifest
from capabilities.manifest import get_capability, find_capabilities, get_domain

cap = get_capability("intel_alerts")        # by capability ID
cap = get_by_path("capabilities/intel/alerts")  # by path
cap = get_by_package("apg-intel-alerts")        # by package name
results = find_capabilities("payroll")           # keyword search
intel_caps = get_domain("intel")                 # domain listing
```

---

## APG DSL — Compose Business Systems

Declare composed applications in the APG language:

```apg
// CRM platform in 40 lines
module crm_platform version 1.0.0 { description: "Enterprise CRM"; }

table Contact { name: str; email: str; status: str; }

capability CRMCore {
    contract: {
        id: crm_platform_core,
        provides: [contact_lifecycle, opportunity_pipeline, sales_analytics],
        requires: [auth, audl, ntfy, wflo],
        rules: [
            {name: "large_deal_approval", when: "amount > 50000", action: require_review},
            {name: "cross_tenant_denied", when: "contact_tenant != actor_tenant", action: deny}
        ],
        ui: {shell: python, routes: [
            {name: "Pipeline", path: "/crm/pipeline", component: "Pipeline", permission: "crm:pipeline"}
        ]},
        theme: {name: crm_theme, tokens: {"color.primary": "#1565C0", "border.radius": "6px"}, components: {}}
    };
}

agent SalesAssistant {
    model: "openai:gpt-4.1-mini" ?? "ollama:llama3.2";
    capabilities: [opportunity_pipeline];
    memory: vector sales_memory;
}

app CRMPlatform {
    capabilities: [CRMCore];
    agents: [SalesAssistant];
    runtime: {target: python, streaming: {processor: bytewax}};
}
```

Compile and run:

```bash
apg compile examples/crm_platform/main.apg --output ./generated --verify
python ./generated/app.py --host 0.0.0.0 --port 8080
```

---

## Platform Deployment

```bash
# Run the full platform with Docker Compose
docker compose up

# Run just the intelligence capabilities
docker compose up apg-db intel-alerts intel-threats intel-osint

# Run just finance
docker compose up apg-db fin-arc fin-glr fin-apy
```

Services expose: `GET /health`, `GET /contract`, `POST /evaluate`, `GET /api/v1/...`

---

## CLI

```bash
# Capability discovery
apg capabilities search alerts
apg capabilities manifest --stats
apg capabilities manifest --domain intel
apg capabilities manifest --capability intel_alerts

# Contract validation
apg capabilities validate-contracts

# Build all packages
./scripts/build_all_packages.sh
```

---

## Documentation

| Document | Description |
|---|---|
| `docs/composability_contract.md` | Complete contract schema reference (1,384 lines) |
| `docs/capability_development_guide.md` | How to build a new capability (2,108 lines) |
| `docs/capability_integration_guide.md` | Integration patterns and testing guide |
| `capabilities/MANIFEST.md` | Full index of all 259 capabilities (13,857 lines) |
| `capabilities/COMPOSABILITY.md` | Dependency graph (1,900 edges, 0 broken) |

---

## Example Platform Programs

Fully compilable APG programs demonstrating capability composition:

| Example | Pattern | Demonstrates |
|---|---|---|
| `examples/crm_platform/` | Hub-and-Spoke | CRM + AI agent + workflows |
| `examples/accounting_platform/` | Layered | GL + AR + AP + IFRS rules |
| `examples/erp_platform/` | Full ERP | Procure-to-pay, order-to-cash, hire-to-retire |
| `examples/intelligence_platform/` | Pipeline | OSINT → Fusion → Alerts |
| `examples/fintech_platform/` | Africa-first | M-Pesa, KYC tiers, AML, CBK rules |
| `examples/healthcare_platform/` | Clinical | EMR → Pharmacy → Lab |

---

## Project Structure

```
apg/
├── capabilities/               # 259 capability packages
│   ├── MANIFEST.json           # Machine-readable index
│   ├── MANIFEST.md             # Human-readable index (13,857 lines)
│   ├── manifest.py             # Bidirectional navigation API
│   ├── COMPOSABILITY.md        # Dependency graph
│   ├── <domain>/<code>/        # Each capability directory
│   │   ├── capability_contract.py
│   │   ├── models.py
│   │   ├── service.py          # 40-184 async methods
│   │   ├── api.py
│   │   ├── app.py              # Standalone server
│   │   ├── pyproject.toml      # PyPI package
│   │   ├── domain/adapters.py  # Protocol interfaces
│   │   ├── database/store.py   # InMemory + PostgreSQL
│   │   └── alembic/            # DB migrations
├── compiler/                   # APG language compiler
├── examples/                   # 6 platform APG programs
├── docs/                       # Reference documentation
├── tests/                      # 200+ tests (0 collection errors)
├── spec/apg.g4                 # ANTLR4 grammar
├── docker-compose.yml          # Platform deployment
├── Dockerfile.capability       # Standalone capability container
└── .github/workflows/ci.yml    # CI pipeline
```

---

## License

Proprietary — © 2025 Datacraft · [nyimbi@gmail.com](mailto:nyimbi@gmail.com)

---

## Running Generated Applications

### Compile

```bash
apg compile myapp.apg --output ./out/myapp
```

Generated output in `./out/myapp/` is a **self-contained Python package** — zero dependencies, stdlib only:

```
app.py              ← main entry point
smoke_test.py       ← contract test runner
requirements.txt    ← empty by default (no pip install needed)
Dockerfile          ← container deployment
.env.example        ← environment variable reference
semantic_model.json ← APG semantic model
```

### Run

```bash
cd out/myapp
python app.py                           # starts on http://127.0.0.1:8080
python app.py --host 0.0.0.0 --port 3000
APG_HOST=0.0.0.0 APG_PORT=3000 python app.py   # env vars
docker build -t myapp . && docker run -p 8080:8080 myapp
```

### Verify

```bash
python app.py --self-test       # validates contracts, routes, component manifest
python app.py --validate        # contract validation (exit 0/1)
python app.py --describe        # application JSON description
python app.py --semantic-model  # full APG semantic model
python smoke_test.py            # importable contract test runner
```

### Core endpoints (always present)

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Runtime health + validation summary |
| `GET /component.json` | Composable component manifest |
| `GET /semantic-model.json` | Full APG semantic model |
| `GET /openapi.json` | OpenAPI 3.1 spec |
| `GET /self-test` | Smoke test results |
| `GET /applications` | Application manifest |
| `GET /capabilities` | Capability registry |
| `GET /entities` | Entity schemas |
| `GET /workflows` | Workflow definitions |
| `GET /agents` | AI agent definitions |
| `GET /ui` | Generated HTML application console |

### UI routes

```
http://127.0.0.1:8080/ui                     # application index
http://127.0.0.1:8080/ui/entities/<Name>     # entity CRUD screen
http://127.0.0.1:8080/ui/capabilities/<name> # capability console
http://127.0.0.1:8080/ui/agents/<name>       # AI agent workbench
```

### Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `APG_HOST` | `127.0.0.1` | Bind host |
| `APG_PORT` | `8080` | Bind port |
| `APG_DATA_FILE` | _(in-memory)_ | JSON persistence path |
| `APG_API_KEY` | _(none)_ | Require `Authorization: Bearer <key>` on mutations |
| `APG_DEBUG` | `0` | Set to `1` for HTTP request logging |

### Capability packages

Each compiled capability also ships as a standalone Python package:

```bash
pip install apg-fintech-gateway
apg-fintech-gateway --port 8080
# or
python -m capabilities.fintech.gateway.app --port 8080
```
