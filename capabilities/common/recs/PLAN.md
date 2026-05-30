# RECS Development Plan

## Objective

Build RECS into a coherent APG recommendation lifecycle and guardrail packet
that moves from governed data to recommendations, feedback, deployment, and
experimentation with AI recommender-agent support.

## Implementation Slices

1. **Contract**
   - Expand configuration for datasets, models, ranking, experiments, feedback,
     recommender agents, deployments, governance, observability, adapters, UI,
     theme, and Bytewax streaming.
   - Expand deterministic guardrails for dataset lifecycle, interactions,
     consent, ranking, training, deployment, feedback, experiments, AI agents,
     state changes, tenant isolation, and Bytewax batch mutations.

2. **Runtime Models**
   - Add dataset, interaction event, model deployment, feedback, and AI
     recommender-agent records.
   - Extend ranking policy ownership and model approval metadata.

3. **Service Runtime**
   - Add dataset registration, interaction capture, model approval, model
     deployment, feedback capture, recommender-agent registration, state change,
     list, dashboard, and audit support.
   - Preserve existing deterministic scoring behavior.

4. **API And Views**
   - Add payload helpers for datasets, interactions, model approval/deployment,
     feedback, recommender agents, and state changes.
   - Add dataset manager, deployment center, feedback console, recommender-agent
     panel, audit trail, and analytics view models.

5. **Documentation**
   - Add README, full specification, and this implementation plan.
   - Replace stale `cap_spec.md` with a compatibility pointer.

6. **Verification**
   - Run focused `py_compile`.
   - Run RECS contract/package tests only.
   - Run generated app self-test.
   - Run APG implementation audit and publish plan for RECS.
   - Search RECS for stale generated-package claims, unsupported overclaims, and
     banned stream choices.

## Review Checklist

- The contract exposes provides, requires, rules, UI routes, theme, adapters,
  and Bytewax streaming.
- Recommendation data, model, deployment, feedback, experiment, and AI-agent
  lifecycles are executable.
- Recommendation generation cannot bypass consent, ranking policy, candidate,
  confidence, diversity, sensitive filtering, and explainability controls.
- Deployment cannot bypass model approval, target, approval, or rollback
  evidence.
- Tests prove the main executable lifecycle and representative guardrail
  failures.
- Generated package evidence is refreshed after contract changes.

