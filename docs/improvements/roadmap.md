# Phased Roadmap

The ordering below is derived from the dependency graph, not from the severity table. Severity says what hurts most; dependencies say what can be built. Where they disagree, dependencies win — a critical fix that cannot be verified is not a fix.

## The ordering principle

Three facts force the sequence.

1. **CI is red** (D-11). Every phase below ships through the same pipeline. Until it is green, no fix is verifiable and no regression is caught. Phase 0 is not a warm-up; it is the precondition for having evidence at all.
2. **Four of five security defects are one defect** (authorisation at call sites, not at a boundary). Patching them individually produces four patches the next route bypasses. The boundary is one change with four consequences.
3. **Three of four scalability defects are one defect** (no data plane). Every feature in Cluster B requires the substrate to be replaced, not extended. Doing feature work before the substrate means writing it twice.

## Dependency graph

```
                          ┌─────────────────────────┐
                          │ P0  Gate restoration    │  D-11, D-19, D-18
                          │  (CI green, 3.9 floor)  │
                          └───────────┬─────────────┘
                                      │ every later phase needs a working gate
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
   ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
   │ P1  Auth boundary  │  │ P2  Data plane     │  │ P3  Bounded tele-  │
   │  D-01..D-06         │  │  D-07..D-10, D-12  │  │     metry          │
   │  F18, F19           │  │  F8..F17, D-24,25  │  │  D-14..D-16        │
   └─────────┬──────────┘  └─────────┬──────────┘  │  F26..F28          │
             │                       │             └─────────┬──────────┘
             │                       │                       │
             └───────────┬───────────┴───────────┬───────────┘
                         ▼                       ▼
              ┌────────────────────┐   ┌────────────────────┐
              │ P4  Effect type    │   │ P5  Compiler as a  │
              │     system         │   │     queryable obj  │
              │  F1..F7            │   │  F33..F39, D-21    │
              └─────────┬──────────┘   └─────────┬──────────┘
                        │                        │
                        ▼                        ▼
              ┌────────────────────┐   ┌────────────────────┐
              │ P6  Governance     │   │ P7  Agent          │
              │     plane          │   │     provenance     │
              │  F20..F25          │   │  F40..F45          │
              └─────────┬──────────┘   └─────────┬──────────┘
                        │                        │
                        └───────────┬────────────┘
                                    ▼
                         ┌────────────────────┐
                         │ P8  Offline/edge   │
                         │  F46..F48          │
                         └─────────┬──────────┘
                                   ▼
                         ┌────────────────────┐
                         │ P9  Supply chain   │
                         │  F49, F50          │
                         └────────────────────┘
```

**Edge semantics.** An edge `A → B` means B's acceptance criteria cannot be tested until A's are met. It does not mean B cannot be designed.

## Phase 0 — Restore the gate

**Blocks:** everything. **Duration driver:** small, but it is the highest-value work in the document.

| Task | Anchor | Acceptance |
| --- | --- | --- |
| Add `W191` to the lint ignore list, or switch the mandated style to spaces project-wide | `.github/workflows/ci.yml:32` | Lint job green |
| Add the missing runtime dependencies to the `dev` extra | `pyproject.toml:30-35` | Test job collects and runs |
| Either raise the floor to 3.10 or add `from __future__ import annotations` to the 7 affected modules | `pyproject.toml:8`; sites in D-19 | 3.9 job green or dropped from the matrix |
| Replace `CMD ["python", "app.py"]` with a WSGI server and a real health probe | `code_generator.py:13044`, `:13041` | Container serves concurrent requests; healthcheck does not run a self-test per interval |
| Delete `compiler/widget_rules.py` | no importers | Removes a lint failure and a liability |

**Exit criterion:** three consecutive green runs on `main`, on every matrix entry that is kept.

**Why first:** the project's own lesson — "a gate that has never run is not a gate." Five red runs mean the last five pushes had no verification. Every claim in `defects.md` about a *fix* is untestable until this phase closes.

## Phase 1 — The authorisation boundary

**Depends on:** P0. **Closes:** D-01, D-02, D-03, D-04, D-05, D-06; features 18, 19.

**The change:** deny by default in a single `before_request` hook, with each route declaring its public status explicitly. Move the tenant binding from the header to the authenticated principal. Replace `eval` with a restricted expression interpreter.

| Task | Anchor | Note |
| --- | --- | --- |
| Default-deny `before_request` with an explicit public-route registry | `code_generator.py:12586`, `:12881`, `:12785` | Retires the allowlist-at-the-dispatcher inversion |
| Delete the duplicate CSV route; keep the guarded one | `:12765` vs `:12849` | Two handlers for one path is the whole of D-01 |
| Authorise the upload route against the owning record | `:6412` | D-02 |
| Bind tenant to principal; reject a disagreeing header | `:2314`, `:2357` | D-04 |
| Refuse to start in production with a minted session secret | `:937` | D-05 |
| Restricted expression IR for computed fields | `:6262` | D-03; also unblocks feature 48 |

**Acceptance:** a probe suite that, for every route, asserts `401` without credentials and asserts that a tenant-a credential cannot read tenant-b's bytes. The suite must enumerate routes from `app.url_map` at runtime, not from a list in the test — the project's lesson "an inventory read from source is not the inventory the app serves" applies directly.

**Critical path note:** the route enumeration is the deliverable. A test that names the routes it checks reproduces D-06 one level up.

## Phase 2 — The data plane

**Depends on:** P0. **Closes:** D-07, D-08, D-09, D-10, D-12, D-24, D-25; features 8–17.

This is the largest phase and the one with the clearest asymptotic payoff.

| Task | Anchor | Asymptotic change |
| --- | --- | --- |
| Thread-local or pooled SQLite connections, WAL, busy timeout | `:6958` | Removes the correctness failure (D-07) |
| Lock or relocate id allocation | `:11082` | Removes the lost-write failure (D-08) |
| Query IR lowered to SQL; predicate pushdown | `:2747` | `O(N log N)` → `O(log N + k)` per request |
| Keyset pagination on `(sort_key, id)` | `:2808` | `O(N)` cursor scan → `O(log N)` |
| Emit declared indexes | `:6974` | Makes the pushdown's gain realisable (D-24) |
| Per-row persistence instead of full-store rewrite | `:3047` | `O(N)` → `O(1)` bytes per mutation; aggregate `O(m·N)` → `O(m)` |
| Migration ledger | `:6995` | Makes D-25 incremental and idempotent |
| Connection pool for PostgreSQL | `:11564` | Removes per-call TCP + auth (D-12) |

**Sequencing inside the phase:** correctness first (D-07, D-08), then the read path, then the write path, then PostgreSQL. The write path depends on the ledger, which depends on the index emission being correct.

**Acceptance:** a benchmark asserting page-fetch latency is flat in `N` for fixed `k`, and a mutation benchmark asserting write cost is flat in `N`. Both must be canaried — a benchmark that has never been seen to fail is not a benchmark.

**Why not earlier:** the data plane replaces the substrate under every feature in Cluster B. Building those features first means building them twice.

## Phase 3 — Bounded telemetry

**Depends on:** P0. **Closes:** D-14, D-15, D-16; features 26, 27, 28.

Small, independent of P1 and P2, and closes a remote memory-exhaustion primitive.

| Task | Anchor | Note |
| --- | --- | --- |
| Label allowlist with overflow bucketing | `:12133`, `:12170` | Caps series count |
| TTL sweep over rate buckets | `:12525` | Caps bucket count |
| Bounded executor for email and webhooks | `:962`, `:11709` | Replaces per-send threads |

**Acceptance:** a probe issuing 10⁵ distinct paths and 10⁵ distinct source addresses, asserting that both maps stay bounded. This one can be canaried cheaply: set the cap to 10 and watch it trip.

## Phase 4 — The effect type system

**Depends on:** P2 (transaction boundary requires a real store), P3 (outbox requires bounded delivery). **Delivers:** features 1–7.

This is where the platform stops being a CRUD generator. A workflow with an `irreversible` step and no compensation becomes a compile error.

| Order | Feature | Depends on |
| --- | --- | --- |
| 4.1 | 1 — effect lattice | — |
| 4.2 | 2 — compensation completeness check | 4.1 |
| 4.3 | 5 — transaction boundary | P2 |
| 4.4 | 3 — idempotency keys | 4.3 |
| 4.5 | 6 — transactional outbox | 4.3 |
| 4.6 | 4 — retry policy | 4.1, 4.4 |
| 4.7 | 7 — dead-letter queue | 4.5 |

**Acceptance:** a compile-time test suite where each `irreversible`-without-compensation case fails to compile, and each well-formed case compiles and passes an integration test that injects a mid-saga failure and asserts compensation ran.

## Phase 5 — The compiler as a queryable object

**Depends on:** P0. **Delivers:** features 33–39; closes D-21.

| Order | Feature | Note |
| --- | --- | --- |
| 5.1 | 37 — LSP on the real parser | Highest immediate user value; deletes three regex parsers |
| 5.2 | 33 — emission provenance | Precondition for 34 |
| 5.3 | 34 — round-trip verification | Catches the D-24 class systematically |
| 5.4 | 38 — capability dependency resolution | `requires` is declared by 339 of 351 capabilities and resolved by nothing |
| 5.5 | 39 — composition proof | Depends on 5.4 |
| 5.6 | 36 — incremental compilation | Depends on 33 |
| 5.7 | 35 — generator decomposition | Last, and only as a precondition for parallel work |

**Why 35 is last:** decomposing a 12,224-line method is the most invasive change available and delivers no capability. It is on the roadmap because it gates the ability to work on the generator in parallel, which becomes necessary once Phases 4–8 are concurrent. Doing it earlier spends the risk budget on refactoring.

**Acceptance for 34:** inject a dropped-construct bug (an index declaration, as in D-24) and assert the round-trip check fails. A round-trip check that has never been seen to fail is the D-24 class wearing the gate's clothes.

## Phase 6 — Governance plane

**Depends on:** P1 (boundary), P2 (provenance needs field-level history), P4 (audit chain should cover the effects). **Delivers:** features 20–25.

| Order | Feature | Note |
| --- | --- | --- |
| 6.1 | 23 — verifiable audit chain for all mutations | Extends `_journal_append` (`:1236`) from workflow events to record events |
| 6.2 | 20 — field-level provenance | Depends on 6.1 |
| 6.3 | 19 — policy as data, real engine | Depends on P1's boundary |
| 6.4 | 21 — consent / purpose limitation | Depends on 6.3 |
| 6.5 | 25 — policy conformance test generation | Depends on 6.3 |
| 6.6 | 22 — data residency routing | Depends on P2's pool |
| 6.7 | 24 — right-to-erasure with crypto-shredding | Depends on 6.2 |

**Acceptance for 6.1:** tamper with one journal entry and assert verification fails. The existing chain covers the one subsystem nothing depends on, which is why it has never been exercised adversarially.

## Phase 7 — Agent provenance

**Depends on:** P3 (bounded cost tracking), P5 (compiler provenance for prompt derivation). **Delivers:** features 40–45.

Strictly ordered: 40 → 41 → 42 → 44, with 43 and 45 parallel to 42.

**Acceptance:** an evaluation that runs twice against a pinned configuration and produces identical results; then a deliberate model change that the eval catches. The second half is the one that matters — an eval that cannot detect a regression is a benchmark.

## Phase 8 — Offline and edge

**Depends on:** P2 (server-authoritative merge needs a real store), P4 (idempotency), P1 (the restricted expression IR for edge compute). **Delivers:** features 46–48.

## Phase 9 — Supply chain

**Depends on:** P5 (deterministic emission is a compiler property), P0. **Delivers:** features 49, 50.

**Acceptance for 49:** compile the same source twice and assert byte-identical output; then sign and verify. Determinism is currently assumed rather than tested — `_load_static_assets` (`:719`) iterates a fixed tuple, which is stable, but nothing asserts the property end to end.

## Critical path

The longest dependency chain runs:

**P0 → P2 → P4 → P6 → P8**

That is: a working gate, then a real store, then effects over it, then governance over the effects, then offline over governance's merge policy. Every other phase hangs off this spine in parallel.

Two milestones gate the most downstream work:

- **M1 (end of P0):** three green CI runs. Unblocks all eight remaining phases. Nothing else can start with evidence.
- **M2 (end of P2):** read latency flat in `N`, write cost flat in `N`, no lost writes under 200-thread concurrency. Unblocks Phases 4, 6, 8 and most of Cluster B's feature work.

## What is deliberately not scheduled

**Feature 35 (generator decomposition)** is listed last in its phase because it is the only item that spends risk without delivering capability. It becomes urgent only when two or more phases run concurrently against `code_generator.py`.

**D-20 (260 duplicate stores)** is not assigned a phase. The shared store exists and 157 capabilities import it; the other ~260 copies are a migration, not a fix, and the migration's value depends on what the store's interface becomes after Phase 2. Doing it before Phase 2 means migrating 260 call sites to an interface that is about to change.

**D-13 (PostgreSQL FTS returns `[]`)** is a Phase 2 deliverable rather than its own line item: the query IR from feature 8 subsumes it, because a search is a predicate and the IR has to lower predicates to PostgreSQL anyway.

## Risk register

| Risk | Phase | Mitigation |
| --- | --- | --- |
| Phase 2 breaks every generated-app test | P2 | The 24 generated-app test modules are the regression net; run them per sub-step, not at the end |
| Auth boundary denies a legitimate internal path | P1 | Enumerate routes from `app.url_map`; canary each with a known-good credential |
| Effect lattice is too coarse to be useful | P4 | Start with 3 effects (`read`, `write`, `irreversible`); refine only when a real workflow needs the distinction |
| Provenance overhead slows emission | P5 | Provenance is a build-time map, not runtime cost; measure emission time, not request time |
| Roadmap becomes a second backlog nobody tracks | all | Each phase has an exit criterion expressed as a probe that has been seen to fail |
