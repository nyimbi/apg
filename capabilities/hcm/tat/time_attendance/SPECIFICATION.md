# Time and Attendance Specification

## Purpose

The Time and Attendance capability lets APG applications compose attendance policy, scheduling, time capture, timesheet approval, leave, exception handling, payroll export, and attendance-agent review into larger HCM and ERP applications.

The capability must be terse enough for rapid composition, but explicit enough to make rules, UI, theming, agent roles, and integration obligations clear.

## Functional Scope

### Policy Lifecycle

- Create and manage tenant attendance policies.
- Require name, timezone, workweek, and positive overtime threshold.
- Expose policy records to schedule and timesheet workflows.

### Schedule and Shift Lifecycle

- Create employee schedules against active policies.
- Support fixed, flexible, rotating, compressed, and remote schedule types.
- Create dated shifts with start and end times.
- Keep schedule and shift records tenant scoped.

### Time Entry and Break Lifecycle

- Record time entries for supported entry types: regular, overtime, leave, holiday, training, and on-call.
- Support web, mobile, kiosk, biometric, API, and import capture methods.
- Require registered device evidence for mobile, kiosk, and biometric methods.
- Require review for geofence failures and low biometric confidence.
- Record breaks against existing time entries.

### Timesheet Lifecycle

- Submit timesheets for an employee and period.
- Require at least one valid time entry.
- Calculate total hours from submitted entries.
- Reject negative-hour submissions.
- Approve timesheets before payroll export.

### Leave Lifecycle

- Record vacation, sick, parental, unpaid, bereavement, and public holiday leave.
- Require employee, leave type, start date, end date, and reason.
- Require review for unpaid or extended leave.

### Exception Workflow

- Record missing clock-out, late arrival, early departure, overtime, geofence, biometric, and duplicate-entry exceptions.
- Require an owner for high-severity exceptions.
- Expose exceptions through UI and dashboard surfaces.

### Payroll Export

- Export only approved timesheets.
- Require period, timesheets, and approval.
- Emit export metadata using Bytewax event-stream configuration.

### AI Agent Composition

- Treat attendance agents as first-class capability citizens.
- Support Codex, Claude Code, OpenCode, and Pi runtimes.
- Support attendance reviewer, compliance reviewer, schedule reviewer, fraud reviewer, payroll export reviewer, and employee query reviewer roles.
- Limit autonomous scope to inspect, prepare, and recommend.
- Require human approval for privileged actions.

## APG Contract Requirements

The executable contract must expose:

- `configuration` and `configuration_schema`.
- `rule_engine` with deterministic rules.
- `ui` routes using the `apg_python` shell.
- `theme` with compact, 8px-radius components.
- `streaming` metadata using Bytewax.
- `provides` and `requires` for composition.
- `semantic_model()` and `component_manifest()` through `app.py`.

## Guardrails

The rule engine must reject:

- Missing tenant context.
- Write operations without policy attachment.
- Unaudited state changes.
- Incomplete policies, schedules, shifts, entries, breaks, timesheets, leave requests, exceptions, and exports.
- Unsupported schedule, entry, leave, exception, agent runtime, and agent role values.
- Tracked device methods without device evidence.
- Payroll exports containing unapproved timesheets.
- Non-Bytewax batch/export routing.

The rule engine must require review for:

- Failed geofence verification.
- Low biometric confidence.
- Unpaid or extended leave.
- Privileged AI-agent actions without human approval.

## UI Requirements

The packet exposes APG routes for dashboard, policies, schedules, shifts, time entries, timesheets, leave, exceptions, payroll exports, agents, rules, and settings. Routes are intentionally adapter-neutral and can be rendered by generated APG applications.

## Verification Requirements

Focused package verification must prove contract shape, rules, lifecycle execution, API helpers, view models, app metadata, publish plan, implementation audit, and Bytewax metadata. Full repository verification can run later when power and compute budget allow.
