# APG Functional Expansion & Architectural Audit

**Date:** 2026-10-25
**Branch:** `fix/production-readiness-blockers`
**Scope:** the `.apg` language, the compiler, the generated Flask runtime, the capability surface (351 capabilities / 33 domains), the language server, and the build pipeline.

This directory holds three artifacts:

| Document | What it answers |
| --- | --- |
| [`features.md`](features.md) | 50 capabilities the platform cannot express today, each derived from a missing primitive rather than a missing screen. |
| [`defects.md`](defects.md) | The defect catalog: security, concurrency, scalability, observability, build, structure. Every entry carries a `file:line` anchor and either an asymptotic argument or a probe result. |
| [`roadmap.md`](roadmap.md) | Phased rollout with a dependency graph, the critical path, and the milestones that gate each phase. |

## Method and evidence boundary

Three evidence classes are used, and each claim names which one it rests on.

1. **Static anchor** — a `file:line` in the current tree, read directly. This is the weakest class: it says the code exists, not that it runs.
2. **Asymptotic derivation** — a complexity class derived from the code path, with the variables named. No profiling is claimed.
3. **Runtime probe** — a throwaway script under `/tmp` that started a generated app and exercised it. Where a probe is cited, the observable result is stated. Probes were not added to the repository; the anchors they exercised are, so each result is reproducible from the tree.

What is **not** claimed: no load test, no fuzzing campaign, no third-party penetration test. The concurrency findings come from a 200-thread probe of the generated app, which is enough to show the race exists and not enough to characterise its rate in production.

## Headline findings

**The pipeline is red.** The last five GitHub Actions runs on `main` all failed. Two independent causes: the lint job rejects the project's mandated tab indentation (`W191`, not in the ignore list), and the test job cannot collect because the `dev` extra at `pyproject.toml:30-35` omits `cryptography`, `numpy`, `starlette`, and `flask_appbuilder`. Every feature below ships through this pipeline, so Phase 0 is not optional.

**Four generated-app security defects are reachable without credentials.** Unauthenticated cross-tenant CSV export (`compiler/code_generator.py:12765`), cross-tenant file download (`:12770` → `:6412`), `eval` sandbox escape through computed fields (`:6253`/`:6262`), and tenant identity taken from an unverified request header (`:2314`). The first two were confirmed with a live probe returning `200` and another tenant's bytes.

**The generated store does not survive its own concurrency model.** A single `sqlite3` connection is shared across threads with `check_same_thread=False` (`:6958`) and `NEXT_RECORD_IDS` is read-modify-write without a lock (`:11082-11083`). A 200-thread probe produced `sqlite3.DatabaseError`, `sqlite3.InterfaceError`, and `SystemError`, and lost 3–10 creates out of 200.

**Every record read is a full scan and every record write rewrites the database.** `query_records` (`:2747`) copies, filters, sorts, and slices in Python — `O(N log N)` per request regardless of page size. `_persist_record_store` (`:3047`) serialises and rewrites every entity on every mutation. The two together mean the generated app's cost model is quadratic in the workload it is meant to serve.

**The 50 features are not a feature list.** They are eight clusters that each close a missing *primitive*: an effect type system, a durable data plane, a transaction boundary, a bounded telemetry schema, an agent provenance chain, a compiler that knows what it emitted, and a governance plane that can prove its own claims. Adding a screen to any of them without the primitive reproduces the current architecture's failure mode — which is precisely the 15,617-line generator with a single 12,224-line emission method that this audit is describing.
