# Time and Attendance Implementation Plan

## Delivery Slice

Build one coherent lifecycle and guardrail packet that can be composed into executable APG applications immediately:

1. Replace generated contract metadata with an explicit APG capability contract.
2. Replace adapter-heavy top-level runtime imports with dependency-light services and helper APIs.
3. Add view models and application metadata for composition.
4. Document the capability contract, rules, lifecycle, UI, and integration expectations.
5. Add focused package tests and run only battery-conscious checks.

## Design Choices

- Keep the packet in Python rather than framework-specific targets.
- Keep optional FastAPI, database, biometric, payroll, and location adapters outside the top-level import path.
- Use deterministic rule evaluation for guardrails.
- Use Bytewax metadata for lifecycle events, batches, and payroll exports.
- Treat AI agents as records with supported runtime and role constraints.
- Keep visual theming compact and operational rather than decorative.

## Acceptance Criteria

- The contract passes `validate_contract_shape`.
- The lifecycle service can create policy, schedule, shift, time entry, break, timesheet, leave, exception, export, and agent records.
- The service rejects missing tenant context, unsupported values, non-Bytewax batches, unapproved payroll exports, and missing review/approval evidence.
- `app.self_test()` passes.
- `semantic_model.json`, `package_manifest.json`, and `release_report.json` reflect the live executable contract.
- Package tests pass without running the full repository suite.

## Deferred Work

- Live device registry, biometric, location, payroll, audit, notification, workflow, and persistence adapters.
- Rendered browser UI validation.
- Performance/load tests.
- End-to-end payroll integration against a durable payroll service.
