# ENCR Backlog

## Current Contract

- Keep `SPECIFICATION.md`, `PLAN.md`, `README.md`, `capability_contract.py`,
  `app.py`, `semantic_model.json`, and `release_report.json` aligned.
- Preserve first-class crypto-agent metadata for `codex`, `claude_code`,
  `opencode`, and `pi`.
- Preserve Bytewax as the required crypto lifecycle batch stream engine.
- Keep privileged crypto-agent roles behind human approval.

## Adapter Backlog

- Add production HSM/KMS/vault adapters behind the existing service methods.
- Add APG KEYM integration without making ENCR depend on KEYM during
  foundation bootstrapping.
- Add post-quantum SDK, entropy hardware, zero-knowledge prover, and
  homomorphic compute adapters.
- Add SIEM, SOAR, DLP, GRC, ticketing, and audit-export adapters.
- Add live Bytewax topologies once the local batch contract is stable.
- Add persistent storage and migration support after the dependency-light
  lifecycle behavior is stable.

## Verification Backlog

- Full repository pytest.
- Rendered browser UI review.
- Live provider integration tests.
- Bytewax topology integration test.
- Performance and load tests.
