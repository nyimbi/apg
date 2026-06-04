  You are taking over APG capability development in:

  /Users/nyimbiodero/src/pjs/apg

  Active objective:
  Carefully and fully develop all APG capabilities. Each capability should become a world-class, executable, composable APG capability. For each capability, work methodically:
  1. Write/update SPECIFICATION.md.
  2. Write/update PLAN.md.
  3. Implement the capability end to end.
  4. Do a detailed code review and resolve emergent issues.
  5. Update docs/progress_log.md.
  6. Run focused verification.
  7. Commit and push regularly.
  8. Move to the next capability.

  Important standing constraints:
  - Preserve unrelated dirty work. As of the handoff, the worktree has only two unrelated dirty paths:
    - capabilities/common/dev_order.md
    - capabilities/fintech/terminal/terramoni_agent_app
  - Do not stage, revert, or modify those unrelated paths unless the user explicitly asks.
  - Stage only the capability packet you work on plus docs/progress_log.md.
  - Commit completed, verified slices regularly.
  - Commit messages must follow the Lore protocol in AGENTS.md.
  - Push after each coherent committed packet.
  - Keep docs/progress_log.md current.
  - Use Bytewax, not Kafka.
  - Keep AI agents provider-neutral and first-class. Support at least codex, claude_code, opencode, and pi where adding agent composition.
  - Do not invoke external Codex/Claude/OpenCode/Pi clients directly from capability runtimes; those belong behind AICR adapter contracts.
  - Battery is constrained: prefer focused package checks over full repository test suites.
  - Do not mark the global goal complete unless all capabilities are actually complete and verified.

  Current branch state:
  - Branch: main
  - main is synced with origin/main.
  - Latest commits:
    - 8a97c899 Govern workflow agent composition
    - 5edb3c69 Govern digital signing agent composition
    - 0cd4388e Govern help knowledge agent composition
    - 0b12d3e4 Govern video meeting agent composition
    - 0fa6eff7 Govern collaboration agent composition

  Recently completed packets:
  - VIDC video-agent composition and Bytewax lifecycle guardrails.
  - HELP support-knowledge agent composition and Bytewax lifecycle guardrails.
  - ESGN signing-agent composition and Bytewax lifecycle guardrails.
  - WFLO workflow-agent composition and Bytewax lifecycle guardrails.

  The next ordered capability is:
  capabilities/common/schd

  Why SCHD:
  capabilities/common/dev_order.md places Phase 7 as:
  46. wflo - Workflow Orchestration
  47. schd - Scheduling & Job Orchestration
  48. scpt - Custom Scripting Engine
  49. ncod - No-Code/Low-Code Builder

  SCHD current state:
  Files exist under capabilities/common/schd:
  - README.md
  - SPECIFICATION.md
  - PLAN.md
  - cap_spec.md
  - capability_contract.py
  - scheduling_runtime.py
  - models.py
  - service.py
  - api.py
  - views.py
  - app.py
  - semantic_model.json
  - package_manifest.json
  - release_report.json
  - test_capability_contract.py
  - tests/test_package_contract.py

  SCHD already has useful behavior:
  - Calendar policies.
  - Worker pools.
  - Job definitions.
  - Schedules.
  - Run lifecycle.
  - Retry, cancellation, dead-letter.
  - Scheduler agents in basic form.
  - Audit events.
  - UI view models.
  - Bytewax batch mutation guardrail.

  SCHD gaps to address next:
  Implement a coherent “scheduler-agent composition and Bytewax lifecycle guardrail packet”, mirroring the standards used in WFLO/ESGN/HELP:
  - Add first-class scheduler-agent metadata to capability_contract.py:
    - agents.first_class = true
    - supported_runtimes = ["codex", "claude_code", "opencode", "pi"]
    - supported_roles, privileged_roles
    - require_scope, require_owner, require_purpose, require_contribution_disclosure
    - require_human_approval_for_privileged_roles
    - adapter_contract = "aicr_provider_neutral_schd_agent_adapter"
  - Add streaming lifecycle metadata:
    - engine/processor = "bytewax"
    - lifecycle_stream = "schd.lifecycle"
    - required_operations covering calendar, worker, job, schedule, run, retry/dead-letter, scheduler_agent, audit batches
    - topics for scheduler lifecycle areas
    - broker_core_dependency_allowed = false
  - Add deterministic guardrails for:
    - missing scheduler-agent ID
    - missing readable name
    - unsupported runtime
    - unsupported role
    - missing scope
    - missing owner
    - missing purpose
    - missing contribution disclosure
    - privileged role without human approval -> require_review, not deny
    - empty lifecycle batch
    - unsupported lifecycle operation
    - non-Bytewax lifecycle batch routing
  - Extend models.py:
    - SchedulerAgent should include owner_ref, purpose, human_approval_required.
    - Add SchdLifecycleBatch or SchedulerLifecycleBatch record.
  - Extend service.py:
    - Tenant-qualified scheduler-agent storage.
    - Explicit owner_ref and purpose required; do not fallback owner to registered_by.
    - Normalize runtime/role.
    - Coerce bool strings safely so "false" does not become truthy.
    - Add validate_lifecycle_batch(...)
    - Add list_lifecycle_batches(...)
    - Update dashboard_summary with pending agent review and lifecycle batch counts.
  - Extend api.py:
    - Expose agents/streaming in capability_status.
    - Add safe bool parsing.
    - Add validate_lifecycle_batch helper.
  - Extend views.py:
    - Agent panel should expose supported_roles, privileged_roles, required controls, theme component.
    - Add lifecycle batch monitor model.
  - Extend __init__.py:
    - registration should expose agents and streaming.
    - provides should include scheduler_agent_composition and Bytewax scheduler lifecycle.
  - Replace static embedded app semantic model if present with a dynamic contract-derived model like WFLO/HELP/ESGN.
  - Regenerate semantic_model.json, package_manifest.json, and release_report.json from the live contract.
  - Update README.md, SPECIFICATION.md, PLAN.md, cap_spec.md.
  - Update focused tests:
    - contract exposes agents + streaming + lifecycle route
    - rule engine covers new agent and lifecycle batch guardrails
    - runtime positive path creates schedule/job/run/agent/lifecycle batch
    - negative path proves unsupported runtime, missing purpose/owner, non-Bytewax lifecycle, empty lifecycle
    - API/view models expose lifecycle batch and first-class agent data

  Suggested focused verification for SCHD:
  ./.venv/bin/python -m py_compile capabilities/common/schd/__init__.py capabilities/common/schd/capability_contract.py capabilities/common/schd/models.py capabilities/common/schd/
  scheduling_runtime.py capabilities/common/schd/service.py capabilities/common/schd/api.py capabilities/common/schd/views.py capabilities/common/schd/app.py capabilities/common/schd/
  test_capability_contract.py capabilities/common/schd/tests/test_package_contract.py

  ./.venv/bin/pytest -q capabilities/common/schd/test_capability_contract.py capabilities/common/schd/tests/test_package_contract.py

  ./.venv/bin/python capabilities/common/schd/app.py

  ./.venv/bin/python -m json.tool capabilities/common/schd/semantic_model.json
  ./.venv/bin/python -m json.tool capabilities/common/schd/package_manifest.json
  ./.venv/bin/python -m json.tool capabilities/common/schd/release_report.json

  ./.venv/bin/apg capabilities inspect schd --json
  ./.venv/bin/apg capabilities implementation-audit --root capabilities/common/schd --json
  ./.venv/bin/apg capabilities publish-plan capabilities/common/schd --json

  Run a runtime smoke similar to:
  calendar -> worker pool -> job -> schedule -> run -> complete run -> codex scheduler agent -> Bytewax lifecycle batch.

  Run:
  git diff --check -- capabilities/common/schd docs/progress_log.md

  Stale marker scan:
  rg -n "bool\\(payload\\.get|TODO|FIXME|placeholder|stub|baseline|kafka|broker_core|legacy_queue" capabilities/common/schd docs/progress_log.md

  Intentional acceptable hits may include:
  - broker_core_dependency_allowed: false
  - negative test names such as kafka_replay or legacy_queue
  - progress-log historical entries outside the touched packet

  After SCHD:
  - Append a detailed progress_log entry with:
    - selected capability
    - what existing behavior was preserved
    - what was added
    - verification commands and outcomes
    - review findings fixed
    - not-run list for battery reasons
  - Stage only capabilities/common/schd and docs/progress_log.md.
  - Commit with Lore protocol, push, confirm git status.
  - Leave unrelated dirty paths untouched.
  - Then proceed to the next ordered capability, likely capabilities/common/scpt.

  I checked the current state before writing this. The repo is synced to origin/main, and the only dirty paths are still the unrelated capabilities/common/dev_order.md and capabilities/
  fintech/terminal/terramoni_agent_app.