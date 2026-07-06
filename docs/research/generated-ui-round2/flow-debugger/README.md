# Flow Debugger Round-2 Research

Date accessed: 2026-07-06

## Best-in-class reference

Commercial leader: Temporal Web UI. Temporal's product surface is strongest for durable workflow debugging because it exposes workflow execution metadata, event history, and operational controls around long-running executions.

Adjacent references:

- Microsoft Power Automate Desktop for familiar breakpoint, step-over, and variable-pane debugging ergonomics.
- Retool Workflows for run-history inspection and block-level failure localization.

## Leader weaknesses

- Temporal's replay power is conceptually strong, but the UI/debugging story often splits between event history, CLI/SDK replay, and code-level tooling. Operators still have to translate history into practical "what should I inspect next?" actions.
- Power Automate Desktop has approachable breakpoints and variables, but that strength is tied to desktop flow authoring and does not generalize into generated, durable application workflow pages.
- Retool's run history helps locate failed blocks, but it is less opinionated about replay checkpoints, variable diffs, and local investigation state.

## Differentiators proposed

1. Step Replay Rail: derive replay frames from the selected APG run trace, show cumulative duration and the fields touched by every step, and let users jump to the journal for the same run.
2. Breakpoint Planner: suggest breakpoints from failed, warning, or slow steps, with one-click local persistence so an operator can keep an investigation state without server mutation.
3. Variable Inspector: summarize payload, record, and event identifiers into an inspectable variable ledger, grouped by source and focused on the values most likely to explain a run.
4. Investigation Verdict: add an at-a-glance debugger verdict using trace and journal counts so completed, partial, and failed runs communicate next action immediately.

## Prioritized implementation

Ship all four in the generated debug console because they use data already present in `WORKFLOW_RUNS`, require no new dependencies, and improve the page even for generated demo apps with only a single completed run.
