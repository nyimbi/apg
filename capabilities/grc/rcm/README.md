# Risk and Compliance Management Capability

`grc_rcm` is the APG capability packet for governed risk, control, compliance,
evidence, issue, exception, governance-decision, and AI-agent review lifecycles.
It is intentionally dependency-light at the package boundary so APG applications
can compose it immediately while production deployments attach durable stores,
workflow engines, policy libraries, document vaults, analytics, notification,
and regulatory-content adapters behind explicit integration points.

## What It Provides

- Tenant-scoped risk registration with residual score and risk-level
  classification.
- Control library registration mapped to active risks.
- Compliance obligations by framework, jurisdiction, owner, due date, and
  mapped controls.
- Control assessments with evidence-backed effective, partially effective, and
  ineffective outcomes.
- Encrypted evidence collection with retention guardrails.
- Issue opening and remediation with severity, ownership, review, and closure
  evidence.
- Governance decisions linked to risks, approvers, rationale, and review
  evidence.
- Exception management for risk acceptance, policy exceptions, control waivers,
  and deadline extensions.
- First-class RCM agent registration for Codex, Claude Code, OpenCode, and Pi
  runtime teams.
- APG UI route metadata, screen-model helpers, compact visual theme tokens, and
  Bytewax lifecycle-stream metadata.

## Package Layout

- `SPECIFICATION.md` defines the capability contract, domain model, rules,
  workflows, UI, adapters, and verification gates.
- `PLAN.md` records the implementation plan used for this coherent lifecycle
  packet.
- `capability_contract.py` exposes the executable contract, deterministic rule
  engine, UI routes, theme, and Bytewax event contract.
- `service.py` implements the dependency-light lifecycle service.
- `api.py` exposes composition helpers around the service.
- `views.py` exposes framework-neutral screen models.
- `app.py` exposes the semantic model, component manifest, and package
  self-test.
- `tests/test_package_contract.py` verifies the contract, rule engine,
  lifecycle service, guardrails, API, views, and publishable app surface.
- `semantic_model.json`, `package_manifest.json`, and `release_report.json`
  are generated package evidence consumed by APG tooling.

## Runtime Lifecycle

A typical APG application composes the capability in this order:

1. Register a risk with category, owner, likelihood, impact, and review evidence
   when the residual risk is high or critical.
2. Register controls mapped to the risk.
3. Register compliance obligations mapped to the controls.
4. Collect encrypted evidence linked to the risk, control, assessment, or issue.
5. Assess the control and attach evidence when the result is not fully
   effective.
6. Open remediation issues for gaps, assign owners, and require review for high
   or critical issues.
7. Collect remediation evidence and mark issues remediated.
8. Record governance decisions with approver, rationale, and linked risks.
9. Register approved exceptions with expiration dates where residual exposure is
   intentionally accepted.
10. Register AI agents that can review, prepare, and validate RCM work within
    explicit human-approval boundaries.

## Usage

```python
from capabilities.grc.rcm import GrcRcmService

service = GrcRcmService()

risk = service.register_risk(
	"risk-payments",
	"tenant-a",
	"Payment outage risk",
	"technology",
	"risk-owner",
	0.8,
	0.7,
	"risk-reviewer",
)
control = service.register_control(
	"control-failover",
	"tenant-a",
	"Provider failover test",
	"control-owner",
	"detective",
	[risk["id"]],
)
evidence = service.collect_evidence(
	"evidence-failover",
	"tenant-a",
	"failover-test-log",
	"control",
	control["id"],
)
assessment = service.assess_control(
	"assessment-failover",
	"tenant-a",
	control["id"],
	"assessor",
	"partially_effective",
	[evidence["id"]],
	["manual routing delay"],
)
issue = service.open_issue(
	"issue-routing",
	"tenant-a",
	"Reduce manual routing delay",
	"high",
	"issue-owner",
	"Automate provider selection",
	assessment["id"],
	"issue-reviewer",
)
print(service.dashboard_summary("tenant-a"))
```

Generated APG applications can use `api.py` helpers when they need a stable
function surface:

```python
from capabilities.grc.rcm import api

status = api.capability_status("tenant-a")
records = api.list_records("risks", "tenant-a")
```

## Guardrails

The deterministic rule engine rejects incomplete or unsafe lifecycle actions:

- tenant context is required;
- write operations must attach policy context;
- risk category, likelihood, impact, owner, and title are validated;
- high and critical risks require review evidence;
- controls must map to same-tenant risks and use supported control types;
- obligations must map to same-tenant controls;
- failed assessments require evidence;
- evidence must be encrypted and retained for at least 365 days;
- high and critical issues require review evidence;
- remediation requires a valid issue and remediation evidence;
- governance decisions for high risks require review evidence;
- exceptions require supported type, expiration, and approval;
- RCM event batches must use Bytewax metadata;
- privileged AI-agent actions require recorded human approval.

## UI And Theming

The package exposes route and screen-model metadata for dashboard, risk,
control, obligation, assessment, evidence, issue, governance, exception, agent,
and settings screens. `views.py` is framework-neutral; generated applications
can render the returned models through their chosen Python UI target.

The theme contract is `grc_rcm_control` and includes compact density, APG route
metadata, semantic status styles, and visual tokens for risk, control,
obligation, assessment, evidence, issue, governance, exception, and agent
surfaces.

## Integration Boundary

This package does not open live connections by default. Production deployments
should bind the following concerns through adapters:

- identity, authorization, and segregation-of-duties checks;
- audit vaults and immutable evidence storage;
- document management and retention engines;
- workflow and approval routing;
- notification and collaboration services;
- BI dashboards and external reporting;
- regulatory-content feeds;
- durable Bytewax topology and event sinks;
- AI-agent runtime orchestration.

## Focused Verification

Use the package checks when changing this capability:

```bash
./.venv/bin/python -m py_compile capabilities/grc/rcm/__init__.py capabilities/grc/rcm/capability_contract.py capabilities/grc/rcm/service.py capabilities/grc/rcm/api.py capabilities/grc/rcm/views.py capabilities/grc/rcm/app.py capabilities/grc/rcm/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/grc/rcm/tests/test_package_contract.py
./.venv/bin/python capabilities/grc/rcm/app.py
./.venv/bin/apg capabilities inspect grc_rcm --json
./.venv/bin/apg capabilities publish-plan capabilities/grc/rcm --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/grc/rcm --json
```

---

## World-Class Enhancements (v2.0)

- **I1.** World-Class Improvements — grc_rcm
- **I2.** Continuous Control Monitoring (CCM)
- **I3.** Quantitative Risk Scoring (FAIR-aligned)
- **I4.** Regulatory Change Intelligence Feed
- **I5.** Three-Lines-of-Defense Workflow Engine
- **I6.** Natural Language Obligation Parsing
- **I7.** Control Testing Automation Harness
- **I8.** Risk Appetite Statement as Executable Policy
- **I9.** Issue Ageing and SLA Breach Detection
- **I10.** Audit Evidence Chain of Custody
- **I11.** Predictive Risk Velocity
- **I12.** Cross-Capability Risk Propagation Graph
- **I13.** Compliance Posture Benchmarking
- **I14.** Exception Lifecycle Management with Auto-Expiry
- **I15.** Policy-as-Code Version Control

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
