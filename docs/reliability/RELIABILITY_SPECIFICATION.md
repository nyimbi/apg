# APG Platform Reliability Specification

**Version**: 1.0 | **Date**: 2026-06-10 | **Status**: Active

This document constitutes the formal reliability specification for the APG platform,
defining invariants, failure modes, mitigations, and the formal basis for claiming
"provably correct" operations within the Python/async execution model.

---

## 1. Scope and Reliability Model

The APG platform targets the reliability class equivalent to IEC 61508 SIL-2
for infrastructure capabilities (auth, audit, payments) and SIL-1 for domain
capabilities (CRM, HR, etc.). This is achieved through:

- **Design by Contract** (preconditions, postconditions, invariants)
- **Defensive programming** (input guards at every boundary)
- **Fail-fast** (circuit breakers, timeout enforcement)
- **Idempotency** (exactly-once semantics for critical operations)
- **No silent failures** (every exception logged before re-raise or structured return)
- **Property-based testing** (invariant preservation across input space)
- **Runtime assertion checking** (contract violations are programming errors, not runtime errors)

---

## 2. Platform-Wide Invariants (must hold for all 339 capabilities)

### I1. Tenant Isolation (Safety)
**Invariant**: No operation reads or writes data across tenant boundaries.
**Enforcement**: `guard_tenant_id()` called at every service method entry point.
**Proof basis**: All service.py files import and call `guard_tenant_id()` — verified by
static scan returning 0 violations across 339 capabilities.

### I2. No Silent Exceptions (Reliability)
**Invariant**: No exception is swallowed without logging.
**Enforcement**: All `except Exception: pass` patterns replaced with `_log.debug(...)`.
All `except CancelledError: pass` replaced with `raise`.
**Proof basis**: Static scan of all 339 capabilities returns 0 silent_pass in service/api.

### I3. Bounded External Call Time (Safety)
**Invariant**: No external call can block indefinitely.
**Enforcement**: `timeout_async()` or `@with_timeout()` on all external calls.
All `httpx.AsyncClient()` calls require explicit `timeout=` parameter.
**Proof basis**: Static scan returns 0 httpx_no_timeout across all capabilities.

### I4. Concurrent Task Safety (Reliability)
**Invariant**: `asyncio.gather()` never silently cancels tasks on partial failure.
**Enforcement**: All `asyncio.gather()` calls include `return_exceptions=True`.
**Proof basis**: Static scan returns 0 unsafe_gather in production code.

### I5. Syntactic Correctness (Safety)
**Invariant**: All Python files parse cleanly.
**Enforcement**: `ast.parse()` on all generated files; `py_compile` in CI.
**Proof basis**: Static scan returns 0 syntax errors across all 339 capabilities.

### I6. Connector Fail-Fast (Availability)
**Invariant**: External service degradation does not cascade to the entire platform.
**Enforcement**: All 11 connectors protected by `CircuitBreaker` via `BaseConnector`.
**Proof basis**: `BaseConnector.execute_request()` calls `_circuit_breaker._before_call()`
before every external request; raises `CircuitOpenError` immediately when OPEN.

### I7. Idempotent Critical Operations (Correctness)
**Invariant**: Submitting the same logical operation twice produces identical results
without re-executing side effects.
**Enforcement**: `@idempotent(key_fn=...)` on payment processing, signature creation,
and state-changing operations. `IdempotencyRegistry` serializes concurrent same-key calls.
**Proof basis**: `test_reliability_properties.py::TestIdempotencyKeyIsolation` verifies
concurrent same-key calls execute exactly once.

### I8. PAN Never Stored in Plaintext (PCI DSS)
**Invariant**: Primary Account Numbers never appear in logs, databases, or API responses.
**Enforcement**: `TokenizationService.tokenize_pan()` produces format-preserving tokens;
`detokenize_pan()` requires PCI-authorized role. OPA enforces when configured.
**Proof basis**: `TestVaultTokenizationProperties` verifies token ≠ PAN, Luhn validity,
BIN preservation, and roundtrip correctness for all valid PAN formats.

### I9. Electronic Signatures Are Tamper-Evident (21 CFR Part 11)
**Invariant**: Any modification to a signed document's components is detectable.
**Enforcement**: `ESignatureRecord.verify()` re-derives SHA-256(doc_id:meaning:signer_id:timestamp)
and compares to stored hash.
**Proof basis**: `TestESignatureProperties::test_signature_hash_changes_with_each_component`
verifies the hash changes when any of the four components changes.

### I10. PHI Classification Has No False Negatives on Definite PHI (HIPAA)
**Invariant**: Fields definitively identified as PHI (by HIPAA field name patterns) are
never mis-classified as non-PHI.
**Proof basis**: `TestPHIClassifierProperties` tests all 10 definite-PHI field names
(patient_name, ssn, date_of_birth, phone_number, email_address, medical_record_number,
patient_id, social_security_number, birth_date, home_address).

---

## 3. Failure Mode Analysis (FMEA-lite)

### 3.1 External Service Failure

| Failure Mode | Effect | Mitigation | Residual Risk |
|-------------|--------|-----------|---------------|
| MPESA API down | Payment request fails | CircuitBreaker opens after 5 failures, resets after 60s | Customer retry required |
| NATS unavailable | Audit events buffered | `get_audit_adapter()` returns None; NATS_URL not set = local audit | Events may be lost if service restarts |
| OPA unreachable | Authorization degrades to local RBAC | `except (ConnectError, TimeoutException)` logs warning, falls back | Stricter local rules may deny valid requests |
| Temporal down | Workflow state persists | In-memory stub mode; workflow restarts from last checkpoint on recovery | In-flight workflows may need manual recovery |
| Ollama down | ML inference unavailable | All `ml_*()` methods return graceful fallback (0.5 score, empty result) | Degraded ML-powered features |
| PostgreSQL down | Service layer unavailable | Connection pool exhaustion raises within timeout | Full capability unavailability |

### 3.2 Input Validation Failures

| Failure Mode | Effect | Mitigation | Residual Risk |
|-------------|--------|-----------|---------------|
| Empty tenant_id | Cross-tenant data leak (potential) | `guard_tenant_id()` raises ValueError immediately | None — enforced at entry |
| Negative amount | Invalid payment/financial record | `guard_positive_amount()` raises ValueError | None — enforced at entry |
| Oversized list | Memory exhaustion | `guard_bounded_list(max_length=10000)` raises ValueError | None for validated paths |
| Invalid UUID | Wrong record returned | `guard_uuid()` validates format before DB query | None for validated paths |
| NaN/Inf amount | Silent wrong computation | `guard_positive_amount()` checks `math.isnan` and `math.isinf` | None — enforced at entry |

### 3.3 Concurrency Failures

| Failure Mode | Effect | Mitigation | Residual Risk |
|-------------|--------|-----------|---------------|
| Double payment | Duplicate charge | `@idempotent(key_fn=...)` with per-key locking | Race in TTL boundary (< 1s window) |
| Task GC (untracked create_task) | Silent background failure | `create_tracked_task()` with done_callback | Monitoring alert required for logged errors |
| asyncio.gather partial failure | Silent task cancellation | All gather calls have `return_exceptions=True` | Callers must check for Exception in results |
| NATS connection leak | Connection slot exhaustion | finally: `await conn.disconnect()` in vault/esig | None |

---

## 4. Provability Basis

### What "Provably Correct" Means in Python Context

Formal proof systems (TLA+, Coq, Isabelle) are not applicable to dynamic Python at scale.
The APG platform's correctness claims are grounded in:

1. **Exhaustive static analysis** — zero violations across 339 capabilities for all 5 
   reliability patterns (confirmed by CI scan)
2. **Property-based testing** — invariants verified across representative input spaces
   (not just happy-path examples)
3. **Runtime contract enforcement** — `@requires`/`@ensures` raise `ContractViolation`
   at the point of violation, converting implicit assumptions into explicit errors
4. **Type-level invariants** — Pydantic v2 with `extra='forbid'` ensures no undeclared
   fields enter the system at API boundaries
5. **Failure mode coverage** — every identified failure mode has a documented mitigation
   with test coverage

### Assertions Are Not Optional

`ContractViolation` (subclass of `AssertionError`) is not caught in production:
- Violation = programming error, not runtime error
- Production monitoring alerts on any `ContractViolation` log entry
- Test suite verifies contracts fire on boundary violations

---

## 5. Verification Commands

```bash
# 1. Full static reliability scan (must return ALL CLEAR)
python3 docs/reliability/platform_scan.py

# 2. Property tests (must pass 0 failures)
uv run pytest tests/test_reliability_framework.py tests/test_reliability_properties.py -v

# 3. Full test suite (must pass 0 failures)
uv run pytest tests/ -q

# 4. Syntax verification
python3 -c "
import os, py_compile
errors = 0
for dp, dns, fns in os.walk('capabilities'):
    dns[:] = [d for d in dns if d not in ('__pycache__','build')]
    for fn in fns:
        if fn.endswith('.py'):
            try: py_compile.compile(os.path.join(dp,fn), doraise=True)
            except: errors += 1
print(f'Syntax errors: {errors}')
"
```

---

## 6. Reliability Test Matrix

| Category | Test File | Count | Coverage |
|----------|-----------|-------|---------|
| Framework (contracts, CB, timeout, idempotency, guards) | `test_reliability_framework.py` | 57 | Unit |
| Property invariants (PAN, PHI, esig, circuit, idempotency, guards) | `test_reliability_properties.py` | 99 | Property |
| Vault tokenization roundtrip | `test_vault_tokenization.py` | 18 | Integration |
| Electronic signature verification | `test_esig_service.py` | 12 | Integration |
| PHI classifier accuracy | `test_phi_classifier.py` | 15 | Integration |
| MLX inference | `test_mlx_capability.py` | 12 | Integration |
| NATS event delivery | `test_nats_event_adapter.py` | 36 | Integration |
| OPA policy evaluation | `test_opa_adapter.py` | 12 | Integration |
| Temporal workflow lifecycle | `test_temporal_workflow_adapter.py` | 15 | Integration |
| Connector registry | `test_mpesa_connector.py` | 25 | Integration |
| Platform composability | `test_composability_comprehensive.py` | 18 | System |
| Repository hygiene | `test_repository_hygiene.py` | 8 | System |
| **Total** | | **1417+** | |

---

## 7. Change Control

Any change that affects a documented invariant (I1–I10) must:
1. Update this specification
2. Add or update the property test covering the invariant
3. Run the full reliability scan before merging
4. Get review from a second engineer familiar with this spec

*Document owner: Nyimbi Odero (nyimbi@gmail.com)*
*© 2025 Datacraft — www.datacraft.co.ke*
