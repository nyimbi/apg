# KEYM Backlog

## Current Contract

- Keep `SPECIFICATION.md`, `PLAN.md`, `README.md`, `capability_contract.py`,
  `app.py`, `semantic_model.json`, and `release_report.json` aligned.
- Preserve first-class key-agent metadata for `codex`, `claude_code`,
  `opencode`, and `pi`.
- Preserve Bytewax as the required key lifecycle batch stream engine.
- Keep privileged key-agent roles behind human approval.

## Adapter Backlog

- Add production HSM, KMS, vault, cloud key-store, and software-HSM adapters
  behind the existing service methods.
- Add APG ENCR and SECU integrations without making the dependency-light KEYM
  runtime require live services during foundation bootstrapping.
- Add blockchain audit, AI lifecycle, security-intelligence, compliance, GRC,
  SIEM, SOAR, DLP, monitoring, and notification adapters.
- Add live Bytewax topologies once the local batch contract is stable.
- Add persistent storage and migration support after lifecycle behavior is
  stable.

## Verification Backlog

- Full repository pytest.
- Rendered browser UI review.
- Live provider integration tests.
- Bytewax topology integration test.
- Performance and load tests.
