# Kanban Rationale

## Decisions

- **Leader:** Jira Software for mature kanban flow reporting and WIP policies.
- **Shipped first:** inline flow intelligence because the generated board already had status columns and WIP warnings.
- **Detected swimlanes:** common business fields are enough to make generated boards more scannable without configuration.

## Rejected Alternatives

- **Historical CFD chart:** rejected for this slice because generated apps do not yet persist board snapshots.
- **Third-party kanban analytics library:** rejected because generated output must stay offline and dependency-light.
- **Admin WIP editor:** rejected because the generated default policy already surfaces bottlenecks without adding a settings flow.

## Verification Intent

The kanban page should keep existing board tests green while adding cumulative-flow, swimlane, and WIP-policy markup to regenerated outputs.
