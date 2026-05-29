# APG Tutorial

This tutorial builds from a minimal APG record app to a small composable ERP
slice with a capability, rule, screen, workflow, and AI agent. The goal is not
to show every grammar construct. The goal is to teach the shortest reliable
path from source text to an executable generated Python application.

## 1. Prepare The Environment

From the repository root:

```bash
uv sync
apg --help
```

If `apg` is not on your shell path, use the virtual environment entry point:

```bash
./.venv/bin/apg --help
```

## 2. Start With A Table

Create `tutorial/customer_ops.apg`:

```apg
module customer_ops version 1.0.0 {
    description: "Customer operations tutorial";
}

table Customer {
    customer_number: str;
    legal_name: str;
    email: str;
    active: bool = true;
}
```

Compile it:

```bash
apg compile tutorial/customer_ops.apg --output tutorial/output --verify
python tutorial/output/smoke_test.py
```

What this gives you:

- a Python HTTP app in `tutorial/output/app.py`
- generated record helpers for `Customer`
- `/entities/Customer` and `/entities/Customer/records` routes
- OpenAPI and component manifest routes
- a generated smoke test

Run the app:

```bash
python tutorial/output/app.py --host 127.0.0.1 --port 8080
```

Create a customer record:

```bash
curl -s \
  -H "Content-Type: application/json" \
  -d '{"record":{"customer_number":"C-1001","legal_name":"Asha Traders","email":"asha@example.com","active":true}}' \
  http://127.0.0.1:8080/entities/Customer/records
```

## 3. Add A Capability

Capabilities make behavior composable. Add this below the table:

```apg
capability CustomerCare {
    contract: {
        id: customer_care,
        provides: [case_management, customer_followup],
        requires: [audit_events],
        configuration: {sla_hours: 24, default_queue: "support"},
        rules: [
            {name: "email_required", when: "email missing", action: "deny"},
            {name: "inactive_review", when: "active == false", action: "require_review"}
        ],
        ui: {
            shell: python,
            routes: [{name: "Customer Care", path: "/customers/care", component: "CustomerCareWorkbench"}]
        },
        theme: {name: customer_theme, tokens: {accent: "#174EA6", surface: "#F8FAFC"}}
    };
}
```

Compile again:

```bash
apg compile tutorial/customer_ops.apg --output tutorial/output --verify
python tutorial/output/smoke_test.py
```

The generated app now exposes capability APIs and UI:

- `GET /capabilities`
- `GET /capabilities/CustomerCare`
- `POST /capabilities/CustomerCare/rules/evaluate`
- `POST /capabilities/CustomerCare/configuration/resolve`
- `GET /ui/capabilities/CustomerCare`

Evaluate a rule:

```bash
curl -s \
  -H "Content-Type: application/json" \
  -d '{"context":{"active":false,"email":"asha@example.com"}}' \
  http://127.0.0.1:8080/capabilities/CustomerCare/rules/evaluate
```

Expected result shape:

```json
{
  "capability": "CustomerCare",
  "decision": "require_review",
  "matched_rules": ["inactive_review"]
}
```

## 4. Add A Screen

Screens make UI composition explicit:

```apg
capability CustomerCare {
    contract: {
        id: customer_care,
        provides: [case_management, customer_followup],
        requires: [audit_events],
        configuration: {sla_hours: 24, default_queue: "support"},
        rules: [
            {name: "email_required", when: "email missing", action: "deny"},
            {name: "inactive_review", when: "active == false", action: "require_review"}
        ],
        ui: {shell: python},
        theme: {name: customer_theme, tokens: {accent: "#174EA6", surface: "#F8FAFC"}}
    };

    screens: {
        CustomerCareWorkbench: {
            route: "/customers/care",
            title: "Customer Care",
            layout: dashboard,
            contains: [CustomerSummary, OpenCases],
            composes: [FollowupQueue],
            binds: [customer.active_cases],
            actions: [assign, escalate, close],
            events: [{on: "select", do: "filter", target: FollowupQueue}],
            relationships: [
                CustomerSummary -> OpenCases,
                OpenCases -> FollowupQueue
            ]
        }
    };
}
```

The route `/customers/care` now renders a generated HTML screen. The screen also
appears in `component_manifest()` and the capability composition graph.

## 5. Add An Application Shell

An application assembles records, capabilities, routes, and runtime metadata:

```apg
app CustomerOperationsApp {
    description: "Small executable customer operations app";
    capabilities: [CustomerCare];
    routes: ["/customers/care"];
    components: {
        care_workbench: {capability: customer_followup, route: "/customers/care"}
    };
    screens: {
        Home: {route: "/", capability: CustomerCare, component: "CustomerCareWorkbench"}
    };
    theme: {name: operations_theme, tokens: {accent: "#174EA6"}};
    runtime: {target: python, deployment: local, streaming: {processor: bytewax}};
}
```

After compilation, inspect:

```bash
python - <<'PY'
import sys
sys.path.insert(0, "tutorial/output")
import app
print(app.describe_application()["application_composition_descriptions"])
print(app.component_manifest()["routes"])
PY
```

## 6. Add A Workflow

Add a simple deterministic workflow:

```apg
workflow CustomerEscalation {
    steps: str = "new -> triage -> assigned -> resolved";
    human_tasks: [triage, assigned];
    assignments: {triage: support_lead, assigned: account_manager};
    guards: {triage: "customer_number present"};
    waits: {resolved: customer_confirmed};
    retry_policy: {assigned: {attempts: 2}};
    compensation: {assigned: reopen_case};
}
```

Compile and run:

```bash
apg compile tutorial/customer_ops.apg --output tutorial/output --verify
python tutorial/output/smoke_test.py
```

Run the workflow:

```bash
curl -s \
  -H "Content-Type: application/json" \
  -d '{"payload":{"customer_number":"C-1001"}}' \
  http://127.0.0.1:8080/workflows/CustomerEscalation/run
```

Because `resolved` waits for `customer_confirmed`, the run may stop in a
`waiting` state until the event is provided in a resume payload.

## 7. Add An AI Agent

Add an agent that can later be connected to Codex, Claude Code, OpenCode, Pi,
OpenAI, Ollama, or another adapter:

```apg
agent SupportPlanner {
    role: "support planner";
    model: "openai:gpt-4.1-mini";
    runtime: codex;
    system: "Plan concise customer support follow-up.";
    capabilities: [customer_followup];
    tools: [case_search, customer_history];
    memory: vector support_memory;
    configuration: {temperature: 0.2, max_steps: 4};
    rules: [
        {name: "no_private_data_without_case", when: "case_id missing", action: "deny"}
    ];
}
```

Generated apps keep agent runtime metadata provider-neutral. If an external
runtime adapter is not configured, invocation returns structured adapter
requirements instead of failing at import time. That keeps the generated app
executable in offline tests.

## 8. Compose Multiple Agents

```apg
agent QualityReviewer {
    role: "support quality reviewer";
    model: "anthropic:claude-3-5-sonnet";
    runtime: claude_code;
    system: "Review support plans for correctness and tone.";
    capabilities: [support_quality];
}

agent_team SupportCrew {
    agents: [SupportPlanner, QualityReviewer];
    handoffs: SupportPlanner -> QualityReviewer [condition: drafted];
    capabilities: [support_response];
    configuration: {max_iterations: 2};
}
```

Semantic analysis verifies that team members and handoff endpoints reference
known agents.

## 9. Inspect The Generated Runtime

Use Python imports for fast feedback:

```bash
python - <<'PY'
import sys
sys.path.insert(0, "tutorial/output")
import app

print(app.list_entities())
print(app.list_capabilities())
print(app.list_workflows())
print(app.validate_application()["valid"])
PY
```

Use HTTP for integration feedback:

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/openapi.json
curl -s http://127.0.0.1:8080/component.json
curl -s http://127.0.0.1:8080/ui
```

## 10. Scale Up

Once the small app works, extend in this order:

1. Add tables for durable business records.
2. Add capabilities around business boundaries.
3. Add rules and configuration to each capability.
4. Add screens only where a user needs to inspect or operate the capability.
5. Add workflows for business process state.
6. Add agents where model-backed work is explicitly valuable.
7. Add an application shell to compose capabilities, routes, screens, agents,
   workflow names, runtime metadata, and deployment metadata.
8. Compile with `--verify` and run the generated smoke test.

The numbered examples under `examples/01_*` through `examples/20_*` follow this
same progression and include checked-in generated outputs for comparison.
