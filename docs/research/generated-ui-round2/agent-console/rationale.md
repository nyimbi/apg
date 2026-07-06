# Rationale

## Decision

Ship an embedded agent intelligence layer in `agent_console.html.j2`: stream meter, local branch save, declared tool inspector, prompt library, and run compare. Compute the metadata in `_ui_agent_console_html` from the agent description, latest request, and latest result.

## Why this beats the benchmark

Commercial leaders excel in separate dimensions. APG improves generated internal-agent workflows by combining conversational iteration, tool visibility, prompt reuse, and compare signals in one dependency-free console.

## Rejected alternatives

- External tracing service: rejected because generated apps must stay offline and self-contained.
- Provider-specific cost math: rejected because pricing is unstable and provider-specific.
- Full branch history: rejected because local fork seed is enough for this pass without persistence policy.

## Validation target

Generated agent console HTML must still render chat input, stream checkbox, stop action, raw JSON details, and team lanes while adding streaming meter, fork action, tool inspector, prompt library, and run compare.
