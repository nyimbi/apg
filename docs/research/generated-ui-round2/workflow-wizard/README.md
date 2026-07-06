# Workflow Wizard Round-2 Research

## Commercial leader

Temporal Web UI is the best-in-class reference for durable workflow state, execution metadata, and debugging visibility. Retool Workflows is the strongest low-code adjacent reference for run history and block-level debugging, while Zapier Paths is the familiar benchmark for visible branching in business automations.

## Leader weaknesses

- Temporal is excellent for engineers, but its Web UI is more execution-inspection oriented than guided business data entry.
- Retool Workflows exposes run logs and block status, but the workflow builder and the business user completion surface are separate.
- Zapier Paths makes branching understandable, but it does not provide APG-style schema-derived record creation or inline rollback in generated CRUD flows.
- None of these leaders generate a compact, offline wizard from an APG entity workflow with duration, rollback, and template hints included by default.

## Differentiators proposed

1. Live Duration Estimate: combine prior run trace durations with schema-derived defaults to show remaining time per wizard step.
2. Rollback To Step: expose completed-step restore points directly in the wizard rather than only in a separate debug screen.
3. Save As Template: persist the workflow shape locally so repeated process variants can be reused without a server dependency.
4. Step Estimate Ledger: show every step's estimate, state, and field count before the user advances.

## Shipped verdict

APG now turns the workflow wizard from a linear form into a lightweight execution cockpit. Before, the user saw a stepper and completion record. After, the wizard shows operational time, rollback targets, local template capture, and a full estimate ledger while keeping the existing step POST flow intact.
