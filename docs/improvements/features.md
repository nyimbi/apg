# 50 Non-Incremental Capabilities

Each entry names the **primitive that is missing**, not the screen that is missing. That distinction is the whole point: APG can already generate a dashboard, a workflow wizard, a kanban board, and an agent console. What it cannot do is make any of them *correct* under concurrency, *auditable* under dispute, or *bounded* under load, because the generated runtime has no data plane to stand on.

The prior gap analysis (`docs/research/world-class-gap-analysis-2026-07/`) closed the *expressiveness* gaps — relationships, uploads, email, i18n, jobs, tenancy, xlsx, computed fields, enums, validation. Those are table stakes against Retool and Budibase and are not repeated here. What follows is the layer above: the capabilities that distinguish a generator that emits a Flask app from a generator that emits a system an enterprise will let hold money.

**Numbering is by cluster, not by priority.** Priority is assigned in `roadmap.md` from the dependency graph, not from this document.

---

## Cluster A — The effect type system (features 1–7)

The `.apg` grammar has 292 productions (`spec/apg.g4`), including `circuit_breaker_config`, `retry_policy_config`, `rate_limit_specification`, and `event_persistence_config`. None of them carry a *type*: the analyzer cannot tell a workflow step that charges a card from one that writes a log line. Every reliability feature below is therefore advisory metadata rather than a checked property.

### 1. Effect annotations on workflow steps

**Missing primitive:** a lattice of effects (`read`, `write`, `external`, `irreversible`, `idempotent`, `compensable`) attached to each step and checked at compile time.

**Why it cannot be done today:** `_execute_workflow_steps` (`compiler/code_generator.py:1877`) runs steps in list order and consults `compensation` (`:1639-1666`) only when a workflow declares `is_saga` (`:1998`). Nothing records whether a step is safe to re-run.

**What it unlocks:** 2, 3, 4, 5, 6, 7 all reduce to a lattice join once this exists. A step marked `irreversible` with no `compensation` becomes a compile error, not a production incident.

**Derivation:** effects form a join-semilattice with `irreversible ⊔ idempotent = irreversible`. A workflow's effect is the join over its steps. The checker is `O(S · |lattice|)` per workflow, linear and free at compile time.

### 2. Compile-time compensation completeness

**Missing primitive:** the analyzer proving that every `irreversible` step has a reachable compensation.

**Why it cannot be done today:** `_compensation_actions` (`:1659`) returns whatever the workflow declared; an empty declaration is not an error. `_trigger_saga_compensation` (`:1864`) then runs against a dict that may not cover the step that failed.

**What it unlocks:** sagas that are correct by construction rather than correct by review.

### 3. Idempotency keys as a first-class DSL modifier

**Missing primitive:** `idempotent on (entity, request_key)` — a deduplication store the runtime consults before executing a mutating step.

**Why it cannot be done today:** `grep -c idempot compiler/code_generator.py` returns `0`. There is no `Idempotency-Key` handling anywhere in the generated runtime.

**What it unlocks:** retry safety for 3 of the 4 failure modes in distributed systems. Without it, feature 4 is unsound.

### 4. Automatic retry with effect-derived policy

**Missing primitive:** retry permitted only where the effect lattice says re-execution is safe, with jittered backoff.

**Why it cannot be done today:** `retry` appears 25 times in the generated runtime but only in the webhook sender (`:11709`, fixed `0/1/2/4` second sleeps) and in `_pg_*` helpers. No workflow step is retried.

**Complexity note:** naive retry on a non-idempotent step converts one failure into `k` duplicate writes. This is why 3 must precede 4.

### 5. Transaction boundary declaration

**Missing primitive:** `transaction { ... }` spanning record writes, events, and outbox rows.

**Why it cannot be done today:** `_persist_record_store` (`:3047`) writes the file and then calls `_pg_save_entity_records` per entity (`:11654`) with no transaction around the pair. A crash between them leaves the two stores divergent, and nothing detects it.

**What it unlocks:** the outbox pattern (feature 6) is meaningless without it.

### 6. Transactional outbox for events and webhooks

**Missing primitive:** events written to a table in the same transaction as the state change, then relayed.

**Why it cannot be done today:** `_apg_deliver_webhook` (`:11709`) starts a daemon thread and retries in memory. `_publish_live_event` (`:1156`) fans out to SSE subscribers. Both lose their payload on process death, and neither is in the write path's transaction. `grep -c outbox compiler/code_generator.py` returns `0`.

**What it unlocks:** at-least-once delivery with a durable log, which is the precondition for feature 44 (event replay) and for any integration a customer will pay for.

### 7. Dead-letter queue with operator replay

**Missing primitive:** a failed-effect store with a bounded retry count, an inspection surface, and a replay action.

**Why it cannot be done today:** `_apg_deliver_webhook` logs and gives up (`:11730`). There is no `dead letter`, `dlq`, or equivalent in the generated runtime.

---

## Cluster B — The durable data plane (features 8–17)

`RECORD_STORE` is a Python dict. `list_records` (`:2575`) copies it, `query_records` (`:2747`) filters and sorts it in Python, `_record_by_id` (`:10365`) scans it linearly, and `_persist_record_store` (`:3047`) rewrites the whole thing on every mutation. Every feature below requires replacing that substrate. None of them can be bolted on above it.

### 8. Predicate pushdown to the store

**Missing primitive:** a query IR the runtime lowers to SQL rather than to list comprehensions.

**Why it cannot be done today:** `query_records` (`:2747-2799`) does `list_records` → comprehension filter → `sorted(...)` → `records[start:start+limit]`. The store is touched before the predicate is known.

**Complexity:** today `O(N log N)` per request for a page of `k` rows; the correct class is `O(log N + k)` with an index. At `N = 10⁶` and `k = 25` that is a factor of roughly `4·10⁴` in rows touched, and it grows with the dataset rather than staying flat.

**What it unlocks:** 9, 10, 11, 12 all follow.

### 9. Cursor pagination that is stable under concurrent insert

**Missing primitive:** keyset pagination on `(sort_key, id)` rather than offset or id-membership.

**Why it cannot be done today:** `:2808-2816` locates `after` by scanning for a matching id, then slices by index. An insert before the cursor shifts every subsequent page; a delete makes the cursor unfindable.

**Complexity:** the current scan is `O(N)` to *find* the cursor, on top of the `O(N log N)` sort. Keyset is `O(log N + k)`.

### 10. Server-side aggregation and grouping

**Missing primitive:** `group by` / `count` / `sum` evaluated in the store.

**Why it cannot be done today:** `metrics_snapshot` (`:3088`) calls `len(list_records(entity))` per entity, materialising every record to count it. The same shape recurs in the UI payload builders.

**Complexity:** `O(N)` per entity per metrics scrape, times the scrape rate — a monitoring system that costs more the more you monitor.

### 11. Index declaration honoured by the generated schema

**Missing primitive:** `index` in the DSL producing an actual index in SQLite and PostgreSQL.

**Why it cannot be done today:** `IndexDeclaration` exists in `ast_builder.py`, and `_sqlite_init_entity_table` (`:6974`) creates tables and FTS tables but no secondary indexes. The declaration is parsed and dropped.

**What it unlocks:** 8's asymptotic gain is unrealisable without it.

### 12. Migration ledger with applied-revision tracking

**Missing primitive:** a table of applied migrations, so `_sqlite_auto_migrate_entity` (`:6995`) can be incremental and idempotent across processes.

**Why it cannot be done today:** auto-migration compares `information_schema` to the expected column set at startup and issues `ALTER TABLE` on the fly. There is no record of what ran, so two workers starting concurrently both attempt the same DDL, and a rollback is undefined.

**Constraint from the project's own history:** a migration revision that has been applied must never be rewritten in place — a new revision is added. A ledger is what makes that rule enforceable rather than remembered.

### 13. Optimistic concurrency at the HTTP layer

**Missing primitive:** `ETag` / `If-Match` on record mutations, returning `412`.

**Why it cannot be done today:** `_revision` is tracked (65 references, seeded at `:3883`) and `expected_revision` is accepted as a query parameter (`:2874-2889`). But `grep -c If-Match` and `grep -c 412` both return `0`: there is no HTTP-level precondition, so the guard is opt-in by a client that already knows to ask.

### 14. Time-travel reads over the event log

**Missing primitive:** `as_of=<timestamp>` resolving a record's state from the append-only event stream.

**Why it cannot be done today:** `EVENT_LOG` exists and `_record_event` is called on mutations, but the log stores summaries, not field-level diffs, and no read path reconstructs state from it.

**What it unlocks:** dispute resolution, audit replay, and the "why did this change" question every enterprise buyer asks in the first demo.

### 15. Columnar analytics sidecar

**Missing primitive:** an append-only columnar projection for analytical queries, separate from the transactional store.

**Why it cannot be done today:** every analytical query in the generated UI runs against `RECORD_STORE` through `query_records`. There is no second read path.

**Derivation:** transactional row storage is `O(N)` for a full scan regardless of column count; columnar is `O(N · c_selected / c_total)`. For the dashboard queries the generator emits — which select 2–4 of 30+ columns over the whole table — the constant is 10–15×, and it compounds with 10.

### 16. Soft-delete compaction and retention policy

**Missing primitive:** a retention rule per entity with a compaction job.

**Why it cannot be done today:** soft deletes are implemented (`deleted_at`, `_record_deleted`) and nothing ever removes a row. `list_records` filters them out but `_persist_record_store` still serialises them forever.

**Complexity:** storage grows without bound while every read pays the filter cost — the worst of both.

### 17. Referential integrity enforcement

**Missing primitive:** foreign keys in the generated DDL, with cascade policy.

**Why it cannot be done today:** relationships are modelled (`_relationship_specs`, `:6438`) and resolved at query time (`_relationship_records_payload`, `:10430`), but `_sqlite_init_entity_table` (`:6974`) emits no `FOREIGN KEY` clause. A relationship to a deleted row resolves to nothing, silently.

---

## Cluster C — The governance plane (features 18–25)

The generated runtime has row ownership, column ACLs, tenant scoping, and audit logging. Each was added as a filter at a call site. None is enforced at a boundary, which is why the CSV and upload routes bypass all of them.

### 18. A single authorisation chokepoint

**Missing primitive:** one function through which every read and write passes, with the policy decision inside it.

**Why it cannot be done today:** `_flask_api_get` (`:12885`) applies `_APG_GET_PUBLIC` and `_check_mutation_auth`; but only 4 of the 24 `@_flask_app.route` decorators are the `api_path` catch-all, so 20 routes bypass the dispatcher entirely. `/entities/<entity>/records.csv` (`:12765`) and `/uploads/<entity>/<filename>` (`:12770`) are among them. The ACL filters live in `list_records` (`:2575`) and `_record_public_copy` (`:2523`), below the route, so a route that does not call them is unprotected by construction.

**Derivation:** the number of ways to reach a record is the number of routes, not the number of handlers. A control that exists at 19 of 27 paths is not a control. This is the exact failure mode the project's own lessons file records.

### 19. Policy-as-data, evaluated by a real engine

**Missing primitive:** the DSL's `access_policy_specification` (`spec/apg.g4:1807`) compiled to Rego or Cedar and evaluated by an engine, rather than to `if` statements.

**Why it cannot be done today:** the grammar production exists and produces properties; the generator emits Python conditionals. The capability surface has `opa` (15 references) but the generated app does not use it.

**What it unlocks:** policy review by a non-programmer, which is the difference between a compliance team that can sign off and one that cannot.

### 20. Field-level provenance

**Missing primitive:** for each field, who set it, when, from which request, and under which policy.

**Why it cannot be done today:** audit events record entity-level actions. There is no per-field lineage, so "who changed the bank account number" is answerable only by diffing snapshots that are not retained.

### 21. Consent and purpose limitation enforced at read time

**Missing primitive:** a record tagged with a purpose; a read declaring its purpose; a mismatch returning nothing.

**Why it cannot be done today:** `CONS` exists as a capability (`capabilities/common/cons`), and the generated runtime has no notion of purpose. Consent is a service you call, not a property of the data.

### 22. Data residency routing

**Missing primitive:** residency declared per entity, enforced at connection selection.

**Why it cannot be done today:** `_pg_connection` (`:11564`) connects to one `_APG_PG_URL` regardless of tenant or entity. Africa-first deployment means multi-jurisdiction storage is a product requirement, not an edge case.

### 23. Verifiable audit chain across all mutations

**Missing primitive:** the hash chain currently applied to workflow journals extended to every record mutation.

**Why it cannot be done today:** `_journal_append` (`:1236`) chains `prev_hash` over workflow events and writes to PostgreSQL when configured. Record mutations go through `_record_event`, which is not chained. So the one part of the system that is tamper-evident is the part that touches nothing valuable.

**What it unlocks:** the difference between "we log" and "we can prove we did not alter the log".

### 24. Right-to-erasure that reaches the backups

**Missing primitive:** a crypto-shredding key per subject, so erasure is key destruction rather than row deletion across a replicated store.

**Why it cannot be done today:** there is no per-subject key. Deletion is a soft-delete flag, and the file store is rewritten wholesale, so a deleted record may persist in any retained snapshot.

### 25. Policy conformance test generation

**Missing primitive:** the compiler emitting tests that assert each declared policy denies what it should.

**Why it cannot be done today:** the generated test scaffold (`:13081`) exercises CRUD. A policy with no test is a comment.

**Derivation:** for `n` entities and `r` roles the matrix is `n·r` cells; generating and asserting the deny cases is `O(n·r)` at compile time and catches the class of bug where a filter is added to one of two code paths.

---

## Cluster D — Bounded telemetry (features 26–32)

`_APG_HTTP_REQUESTS_TOTAL` (`:12128`) is keyed by `(method, path_template, status)`. `_apg_path_template` (`:12133`) replaces numeric and UUID segments with `:id`, but leaves arbitrary path segments intact. `_APG_RATE_BUCKETS` (`:12525`) is keyed by `remote_addr` and never evicted. `_APG_EMAIL_THREADS` (`:959`) appends a thread object per send and never prunes.

### 26. Cardinality-bounded metric labels

**Missing primitive:** a label allowlist with overflow bucketing, so an attacker-chosen path cannot mint an unbounded number of series.

**Why it cannot be done today:** `_apg_path_template` (`:12133`) normalises ids but not arbitrary segments. Every unique unknown path creates a permanent `_APG_HTTP_REQUESTS_TOTAL` entry (`:12170`) and a `_APG_HTTP_REQUEST_DURATION` entry (`:12171`). Memory is `O(distinct paths)` under adversarial input, and `/metrics` exposition is `O(distinct series)` — the scrape cost grows with the attack.

**Derivation:** this is a remote memory-exhaustion primitive, not a metrics-quality issue.

### 27. Evicting rate-limit buckets

**Missing primitive:** a TTL sweep or a bounded LRU over `_APG_RATE_BUCKETS`.

**Why it cannot be done today:** `:12547-12557` writes a bucket per `remote_addr` and never deletes. There is no `pop`, `clear`, or `del` in the range. Memory is `O(distinct source addresses)` and unbounded behind a proxy.

### 28. Thread-pool executor for email and webhooks

**Missing primitive:** a bounded worker pool replacing per-send daemon threads.

**Why it cannot be done today:** `_apg_send_email` (`:962`) creates a thread and appends it to `_APG_EMAIL_THREADS` (`:993`), which is never pruned. `_apg_deliver_webhook` (`:11709`) does the same without even the list. A burst of sends creates a burst of threads with no backpressure.

### 29. W3C trace context propagation

**Missing primitive:** `traceparent` parsing and propagation across requests, jobs, and outbound webhooks.

**Why it cannot be done today:** `grep -c traceparent compiler/code_generator.py` returns `0`. `X-Request-ID` is generated and propagated (Wave B), which correlates one hop; it does not join a distributed trace, and it does not survive a job.

### 30. Job trace continuity

**Missing primitive:** a job carrying its originating trace and span, so the async half of a request is visible.

**Why it cannot be done today:** `_apg_job_worker_loop` (`:11930`) dequeues from SQLite with no context beyond the payload. The request that enqueued the job is a separate, unlinked trace.

### 31. Structured event schema with versioning

**Missing primitive:** a schema registry for emitted events, with a compatibility rule.

**Why it cannot be done today:** `emit_apg_event` (`:2370`) takes an arbitrary payload dict. Consumers — SSE subscribers, webhook receivers, the workflow event bus — parse it by convention.

### 32. Redaction at the emission boundary

**Missing primitive:** classification tags on fields, with the logger, metrics, and error handler consulting them.

**Why it cannot be done today:** structured JSON logging is shipped (Wave B), and it logs whatever is in scope. `PHI` exists as a capability and the generated runtime has no notion of a sensitive field. A `_logging.info` of a record is a disclosure.

---

## Cluster E — The compiler as a queryable object (features 33–39)

The compiler produces an app. It does not produce a *model of the app* that other tools can consume and verify. `compiler/code_generator.py` is 15,617 lines with a single 12,224-line method (`_generate_python_app`, `:793-13016`) that emits a 713 KB `app.py` for a two-entity module by concatenating f-strings.

### 33. Emission provenance

**Missing primitive:** every generated line tagged with the source construct that produced it, so a stack trace in production maps back to a `.apg` line.

**Why it cannot be done today:** the generator emits strings. `compiler/diagnostics.py` maps parse errors to source; nothing maps generated code to source.

**What it unlocks:** 34, 35, and any serious debugging story.

### 34. Generated-code round-trip verification

**Missing primitive:** parse the emitted `app.py` back to an AST and assert it matches the semantic model the generator intended.

**Why it cannot be done today:** the only check is `compile_string` (`compiler/compiler.py:223`) executing the result (`:582`). Executing proves it is syntactically valid Python; it proves nothing about what it does.

**Derivation:** the generator's failure mode is a silently dropped construct — `IndexDeclaration` (feature 11) is the existing example. A round-trip check catches exactly that class, and costs `O(size of emitted code)` per build.

### 35. Generator decomposition with a stable interface

**Missing primitive:** the emission pipeline split into composable emitters over a shared IR, so a change to one surface cannot alter another.

**Why it cannot be done today:** one method, 12,224 lines. Any edit risks every surface, which is why the anchors above cluster in a single file.

**Note on sequencing:** this is the most invasive item in the document and the least valuable to do early. It is on the roadmap as a *precondition* for parallel work, not as a defect fix.

### 36. Incremental compilation

**Missing primitive:** a dependency graph from source construct to emitted file, so an unchanged construct is not re-emitted.

**Why it cannot be done today:** `compile_string` (`:223`) regenerates everything. The CLI writes all artifacts each run.

**Complexity:** `O(total output)` per build today. For a 40-entity module that is seconds; for a large one it is minutes, and it is paid on every save in the LSP path.

### 37. A language server that uses the real parser

**Missing primitive:** the LSP consuming `compiler/parser.py` output.

**Why it cannot be done today:** `language_server/server.py:225` defines `parse_entities` with `re.compile(r"\b(entity|table)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{")` (`:227`), and `_parse_fields` (`:249`) and `_parse_relationships` (`:265`) are likewise regex. The editor's model of the language is a different model from the compiler's.

**Consequence:** every grammar feature added after the regex was written is invisible to completion, hover, and diagnostics. This is why the LSP feels like a different product.

### 38. Machine-readable capability dependency resolution

**Missing primitive:** the manifest's `requires` field (declared by 339 of 351 capabilities) resolved into a dependency graph at compile time, with cycles rejected.

**Why it cannot be done today:** `capabilities/MANIFEST.json` carries `requires` per capability and `provides`. Nothing resolves it into a closure, detects a cycle, or verifies that a `requires` target exists.

**What it unlocks:** 39, and honest composition — today a capability can require something the generated app never installs.

### 39. Composition proof

**Missing primitive:** a check that the union of two capabilities' provided routes, models, and event names is consistent — no duplicate route, no conflicting model prefix, no orphaned event subscriber.

**Why it cannot be done today:** capability composition is additive. `compiler/capability_implementation.py` and `capability_materializer.py` copy files; nothing checks the union.

---

## Cluster F — Agent provenance (features 40–45)

The platform has `agnt` (agents), `aicr` (AI control plane), agent teams, handoff flows, memory, a tool-call inspector (`compiler/templates/agent_console.html.j2:125`), a prompt library with versions (`:142`), and OpenUI Lang rendering of model output. What it does not have is a record of what the model actually did.

### 40. Prompt and model version pinned per run

**Missing primitive:** every agent run records `(prompt_version, model_id, model_digest, temperature, tool_set)` immutably.

**Why it cannot be done today:** the console displays `prompt.version` (`agent_console.html.j2:147`) from the current library. A run does not carry the version it used.

**Consequence:** no run is reproducible, so no evaluation is valid — the model you evaluate is not the model that produced last week's output.

### 41. Tool-call provenance with arguments and results

**Missing primitive:** the tool-call inspector (`:125`) reads *declared* tools. The missing half is the executed call: name, arguments, result, latency, and cost.

**Why it cannot be done today:** `agent_intelligence.tool_calls` is built from declarations, not from execution.

### 42. Evaluation harness with a golden set

**Missing primitive:** versioned inputs with expected properties, run against a pinned agent configuration, with a regression gate.

**Why it cannot be done today:** `eval` appears 1,296 times across `compiler/` and `capabilities/`, none of it an agent evaluation harness.

**Derivation:** an eval without pinned versions (40) measures noise. This is why 42 depends on 40 and cannot be reordered.

### 43. Cost and latency budget enforcement per agent

**Missing primitive:** a declared budget, enforced at the control plane, with the run terminating rather than overrunning.

**Why it cannot be done today:** the console *displays* `stream_meter.cost` (`:35`). Display is not enforcement.

### 44. Deterministic replay of an agent run

**Missing primitive:** recorded model responses and tool results, replayed to reproduce a run without re-calling the model.

**Why it cannot be done today:** nothing is recorded. This depends on 40 and 41.

### 45. Cross-agent handoff ledger

**Missing primitive:** handoffs (`agent_console.html.j2:110`) recorded as edges with payload, so a multi-agent failure is attributable to an agent rather than to "the system".

**Why it cannot be done today:** `team_flow` is rendered from declarations.

---

## Cluster G — Offline and edge (features 46–48)

### 46. Conflict-free replicated records

**Missing primitive:** a CRDT or a server-authoritative merge policy per field, with the policy declared in the DSL.

**Why it cannot be done today:** `grep -c CRDT compiler/code_generator.py` returns `0`. There is a PWA shell with a service worker cache, and the app is offline-capable for *reads of cached pages*. It is not offline-capable for writes.

**Derivation:** last-write-wins over a clock the client controls is not a merge policy, it is a data-loss policy. Africa-first deployment makes intermittent connectivity the normal case, not the exception.

### 47. Offline write queue with server reconciliation

**Missing primitive:** a durable client-side queue, replayed on reconnect, deduplicated by feature 3.

**Why it cannot be done today:** no queue exists. This depends on 3 (idempotency) and 46 (merge).

### 48. Edge execution of declared compute

**Missing primitive:** computed fields (which exist) executable at the edge from the same definition.

**Why it cannot be done today:** computed fields are Python `eval` in the server process (`:6262`). The same expression cannot run in the browser or at a CDN edge.

**Note:** the current `eval` is also a sandbox-escape vulnerability — see `defects.md` D-03. A restricted expression IR fixes both.

---

## Cluster H — Supply chain and release integrity (features 49–50)

### 49. Signed, reproducible build artifacts

**Missing primitive:** deterministic output plus a signature, so a customer can verify that the app they run is the app the compiler produced.

**Why it cannot be done today:** `compiler/release.py` and `evidence_bundle.py` assemble release metadata; nothing signs the artifact. Determinism is also unproven rather than disproven: `_load_static_assets` (`:719`) iterates a fixed tuple, which is stable, but no test asserts that two compilations of the same source produce byte-identical output, so the property is assumed rather than checked.

**Derivation:** "ownable generated code" is APG's core differentiator against Retool. Ownership without verification is a claim, not a property.

### 50. SBOM and dependency provenance in the generated app

**Missing primitive:** a generated SBOM pinned to the exact versions the app resolves, plus a check that the app's dependencies are a subset of the declared set.

**Why it cannot be done today:** `pyproject.toml:9-22` declares the compiler's dependencies; the *generated* app's `requirements.txt` is emitted separately (`:13020`) with no cross-check.

---

## Why these 50 and not another 50

Every feature above is derivable from one of eight missing primitives. The test for inclusion was: *if this were added alone, would it be correct?* A feature that is only correct given another feature is still included, because the dependency is stated — but a feature that would be a screen over a broken substrate is not, because that is what the platform already has.

The failure mode to avoid is visible in the current architecture. `compiler/code_generator.py` accumulated 15,617 lines because each wave added behaviour at the point of use rather than to a substrate. The result is a generator whose cost model is `O(N log N)` per read and `O(N)` per write, whose auth is applied at 19 of 27 paths, and whose audit chain covers the one subsystem nothing depends on. Fifty more screens would reproduce that. Fifty primitives cannot.
