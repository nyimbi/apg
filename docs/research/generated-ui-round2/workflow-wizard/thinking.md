# Raw Thinking

The APG wizard already has a clear stepper and records completed runs. The main gap is that it behaves like a form sequence, not like an operational workflow. The generated app already stores workflow traces with step indexes and duration metadata, so the best differentiator is to surface that metadata while a user is still inside the wizard.

Temporal is the leader because workflow metadata and debugging are first-class. Retool adds approachable run logs for low-code workflows, and Zapier makes branch paths legible to business users. APG can beat the combination for generated CRUD workflows because it controls both the entity schema and the wizard state.

Rejected ideas:

- Adding a server-side workflow-template registry. Useful later, but it introduces persistence policy and permissions.
- Implementing true rollback mutation. The current wizard accumulates fields in hidden inputs; a visual rollback link is safe, while destructive compensation belongs in the debug surface.
- Drawing a full DAG. APG workflows in these examples are sequential steps, so the honest representation is an estimate ledger rather than a fake graph.
