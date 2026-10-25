# Defect Catalog

Anchors are `file:line` in the tree at `381e7050`. Each entry states its **evidence class**: *static* (read from the code), *derived* (complexity argument from the code path), or *probe* (observed by running a generated app). Where a probe is cited the observable is given so it can be reproduced.

Severity is assigned by blast radius, not by ease of fix.

| ID | Severity | Area | One line |
| --- | --- | --- | --- |
| D-01 | Critical | Security | Unauthenticated cross-tenant CSV export |
| D-02 | Critical | Security | Cross-tenant file download without a valid key |
| D-03 | Critical | Security | `eval` sandbox escape in computed fields |
| D-04 | High | Security | Tenant identity from an unverified header |
| D-05 | High | Security | Session secret minted per process when unconfigured |
| D-06 | High | Security | 20 of 24 explicit routes bypass the auth dispatcher |
| D-07 | High | Concurrency | Single SQLite connection shared across threads |
| D-08 | High | Concurrency | Unlocked read-modify-write on record ids |
| D-09 | High | Scalability | Full-store rewrite on every mutation |
| D-10 | High | Scalability | Every record read is a Python-side full scan |
| D-11 | High | Build | CI has been red for at least five consecutive runs |
| D-12 | Medium | Scalability | New PostgreSQL connection per call |
| D-13 | Medium | Correctness | PostgreSQL full-text search returns `[]` |
| D-14 | Medium | Observability | Unbounded metric label cardinality |
| D-15 | Medium | Observability | Rate-limit buckets never evicted |
| D-16 | Medium | Observability | Email threads never pruned |
| D-17 | Medium | Reliability | Webhooks fire-and-forget with in-memory retry |
| D-18 | Medium | Build | Generated Dockerfile runs the Flask dev server |
| D-19 | Medium | Build | `requires-python = ">=3.9"` but PEP 604 at def time |
| D-20 | Medium | Structure | 260 duplicate store implementations |
| D-21 | Medium | Structure | LSP re-parses the language with regex |
| D-22 | Low | Structure | Dead code: `_apg_db_connect` leaks a semaphore |
| D-23 | Low | Structure | `compiler/widget_rules.py` has no importers |
| D-24 | Low | Correctness | Declared indexes are parsed and dropped |
| D-25 | Low | Correctness | Auto-migration has no applied-revision ledger |

---

## Security

### D-01 — Unauthenticated cross-tenant CSV export

**Anchor:** route `compiler/code_generator.py:12765` → `_flask_csv_export` `:12766` → `_csv_export_body` `:11533`.

**Mechanism.** The `_flask_api_get` dispatcher authenticates any path not in `_APG_GET_PUBLIC` (`:12881`, gate at `:12887`), and it *does* have a CSV branch at `:12907` that routes through `_records_csv_response` (`:12849`) behind that gate. But the explicit route at `:12765` shadows it: Flask dispatches to `_flask_csv_export` directly, which calls `_csv_export_body(entity_name)` with no query and no auth check, then returns `200`.

Two handlers exist for the same path. One is protected and one is not, and the unprotected one wins. That is the project's own recorded failure mode — "a canary whose patch applies to the *other* copy of a duplicated guard has not run" — in the shipped runtime.

The tenant filter that would otherwise limit the export lives in `list_records` (`:2591-2597`), which reads `_tenant_id()` from `_TENANT_LOCAL`, which `_setup_tenant` (`:12586`) fills from the unverified header (D-04).

**Evidence: probe.** A generated app with two tenants; `GET /entities/<entity>/records.csv` carrying tenant-a's header and **no** API key returned `200` with tenant-a's rows in the body.

**Complexity of the exposure:** `O(N)` rows per request, unauthenticated. `_csv_export_body` passes no query, so `query_records` runs with `paginate=False` (`:11541`) and the entire entity leaves in one response.

**Fix:** route through `_check_mutation_auth` for any non-public path regardless of which decorator matched. Better: D-06.

---

### D-02 — Cross-tenant file download without a valid key

**Anchor:** route `:12770`; handler `_uploaded_file_response` `:6412`.

**Mechanism.** The handler validates the entity name against `ENTITY_NAMES` and the filename shape (`:6415-6418`), then reads the file from `_upload_root()/entity/filename` (`:6419-6423`). It never calls an authorisation check and never consults the owning record's tenant or owner. The upload root is shared across tenants.

**Evidence: probe.** `GET /uploads/<entity>/<filename>` with tenant-b's header and a deliberately wrong API key returned `200` with tenant-a's file bytes.

**Note:** the ETag is computed from the content (`:6424`), so the response also leaks a content hash, and `If-None-Match` (`:6426-6432`) lets an unauthenticated caller confirm a file's presence and identity without downloading it.

---

### D-03 — `eval` sandbox escape in computed fields

**Anchor:** `_apply_computed_fields` `:6253`; the `eval` at `:6262`:
`public[field_name] = eval(expression, {"__builtins__": {}}, dict(local_values))`.

**Mechanism.** Clearing `__builtins__` does not sandbox CPython. An expression reaches object graphs through any literal — `().__class__.__bases__[0].__subclasses__()` — and from there to `os.system` or file I/O. Computed field expressions come from the `.apg` source, so the threat model is "a source file the operator compiles", not "remote input" — but APG's own premise is that generated apps are handed to third parties, and a capability or template that emits a computed field is a remote-code path by construction.

**Evidence: probe.** An expression built from `().__class__.__bases__[0].__subclasses__()` resolved 240 subclasses inside the generated app's evaluation context.

**Fix:** a restricted expression IR evaluated by a tree-walking interpreter over an allowlisted node set. This is also feature 48's precondition.

---

### D-04 — Tenant identity from an unverified header

**Anchor:** `_tenant_header_value` `:2314`; consumed by `_setup_tenant` `:12586` and `_record_tenant_visible` `:2357`.

**Mechanism.** The tenant is whatever the client sends in `X-APG-Tenant` (default name at `:900`). There is no binding between the authenticated principal and the tenant. `_tenant_admin_bypass` (`:2329`) requires `X-APG-Admin: 1` plus a matching admin key, which is a *different* key from the API key — so possessing the API key does not restrict you to your tenant, and not possessing any key does not restrict you either if the route is unprotected (D-01).

**Consequence.** Every tenant-scoping feature built on this header — row visibility (`:2357`), search filtering (`:10602`), cache keys — is advisory.

**Fix:** derive the tenant from the authenticated principal (JWT claim, API-key record, or session user). Reject a header that disagrees.

---

### D-05 — Session secret minted per process when unconfigured

**Anchor:** `_generated_session_secret` `:937-941`: returns `secrets.token_urlsafe(48)` when nothing is configured.

**Mechanism.** Every process gets a different secret, so sessions do not survive a restart and are not shared across workers. Behind a load balancer with two workers, a user is logged out roughly every other request. In a single-worker deployment it silently works, which is why it is not caught before production.

**Note:** this is a fail-open in the sense that the app starts and appears healthy. A missing secret in production should refuse to start.

---

### D-06 — 20 of 24 explicit routes bypass the auth dispatcher

**Anchor:** `grep -c '@_flask_app.route'` = 24; only 4 of them are the `api_path` catch-all (`:12884`, `:12920`, `:12943`, `:12966`). The other 20 decorators sit on routes that never reach it.

**Mechanism.** `_flask_api_get` (`:12885`) and `_flask_ui_post` (`:12785`) apply auth. The remaining routes — `/metrics` (`:12310`), `/theme.css` (`:12633`), `/locales/<lang>.json` (`:12638`), `/login` (`:12646`), `/logout` (`:12706`), `/locale` (`:12719`), `/api-docs` (`:12738`), the CSV route (`:12765`), the upload route (`:12770`), and the `/ui` family (`:12775-12787`) — apply whatever they individually remember to apply. Some do (`_flask_ui_post` calls `_check_mutation_auth` at `:12790`); the CSV and upload routes do not.

**Derivation.** The number of ways to reach data is the number of routes. A control applied at the dispatcher protects the dispatcher's routes only. This is the project's own recorded lesson — "a control that exists is not a control that is applied" — occurring in the generated runtime rather than in the compiler.

**Fix:** a `before_request` hook that denies by default and requires each route to declare its public status. Flask supports this cleanly; the current design is allowlist-at-the-dispatcher, which inverts the default.

---

## Concurrency

### D-07 — Single SQLite connection shared across threads

**Anchor:** `_sqlite_connection` `:6958-6972`; the module global `_APG_SQLITE_CONN` with `check_same_thread=False` (`:6966`).

**Mechanism.** One connection object is returned to every caller. `sqlite3.Connection` is not thread-safe for concurrent use; `check_same_thread=False` disables the guard that would have caught it. The only lock in the job paths is `_APG_JOB_LOCK`; record paths take `APG_RECORD_LOCK` (`:889`) at exactly one site (`:11394`).

**Evidence: probe.** 200 concurrent threads creating records produced `sqlite3.DatabaseError: no more rows available`, `sqlite3.InterfaceError: bad parameter or other API misuse`, and `SystemError: error return without exception set`, with 190/200 then 197/200 creates succeeding across runs.

**Reading the signature.** The failure rate is not 100%, and it changes between runs. That non-determinism is the finding, not flake: it is what an unlocked shared connection looks like. The project's own lesson applies — "a defect with an alternating signature reads as flake, and flake gets retried instead of investigated."

**Fix:** thread-local connections or a pool, plus `PRAGMA journal_mode=WAL` and a busy timeout.

---

### D-08 — Unlocked read-modify-write on record ids

**Anchor:** `NEXT_RECORD_IDS[entity_name]` read and incremented at `:11082-11083`; initialised at `:2917`.

**Mechanism.** `record["id"] = NEXT_RECORD_IDS[entity_name]` followed by `NEXT_RECORD_IDS[entity_name] += 1` is two bytecodes. Under the GIL each is atomic; the pair is not. Two threads can read the same value and both write the same id.

**Evidence: probe.** The same 200-thread run lost 3–10 creates against the 200 issued — consistent with id collisions silently overwriting.

**Fix:** hold a lock across the pair, or move id allocation into the store (`AUTOINCREMENT` / a sequence).

---

## Scalability

### D-09 — Full-store rewrite on every mutation

**Anchor:** `_persist_record_store` `:3047-3078`; called from 12 sites.

**Mechanism.** Every mutation serialises `records` for every entity, plus `record_metadata`, `events`, `workflow_runs`, all three next-id counters (`:3056-3065`), writes the whole document to a temp file, and `os.replace`s it (`:3070-3073`). With PostgreSQL configured it *also* rewrites every entity's rows (`:3049-3051` → `_pg_save_entity_records` `:11654`).

**Derived complexity.** `O(N)` bytes written per mutation, where `N` is total records across all entities. A single-record update to a 10⁶-row store rewrites the entire store. Aggregate write cost for `m` mutations is `O(m·N)` — quadratic in the workload.

**Secondary effect:** JSON encoding is `O(N)` CPU under the GIL, so it serialises all requests in the process, not just the writer.

---

### D-10 — Every record read is a Python-side full scan

**Anchor:** `query_records` `:2747`; `list_records` `:2575`; `_record_by_id` `:10365`; `_record_public_copy` `:2523`.

**Mechanism.** `query_records` calls `list_records` (`:2761`), which materialises and copies every visible record (`:2589-2597`), then filters with a comprehension (`:2766-2770`), then optionally filters again for `q` (`:2772-2778`), then `sorted(...)` (`:2794`), then slices (`:2815`). `_record_by_id` scans linearly for a single id (`:10366`).

**Derived complexity.** `O(N)` copy + `O(N)` filter + `O(N log N)` sort + `O(k)` slice = `O(N log N)` per request, for a page of `k` rows. At `N = 10⁶`, `k = 25`: roughly 10⁶ copies and 2·10⁷ comparisons to return 25 rows.

**Compounding:** `_records_payload_with_query` (`:10614`) and the UI payload builders call this per page render; `metrics_snapshot` (`:3088`) calls `len(list_records(entity))` per entity per scrape (D-14).

---

### D-12 — New PostgreSQL connection per call

**Anchor:** `_pg_connection` `:11564-11570`; 5 call sites (`:1262`, `:11591`, `:11612`, `:11655`, `:11681`).

**Mechanism.** Each call executes `psycopg2.connect(_APG_PG_URL)` — a TCP handshake, TLS negotiation, authentication, and backend fork — then the caller closes it. Every one of the 5 call sites is wrapped in a bare `except Exception` that returns `None`, so a connection failure is indistinguishable from "no database configured."

**Derived complexity.** Connection setup is `O(1)` but with a constant of tens of milliseconds. At 100 requests/second that is thousands of backends per second, against PostgreSQL's `max_connections` (default 100). The system fails at a load it should handle trivially.

**Secondary defect:** the bare excepts (`:11569`, and around each call site) mean a PostgreSQL outage degrades silently to in-memory behaviour rather than surfacing.

---

## Correctness

### D-13 — PostgreSQL full-text search returns `[]`

**Anchor:** `search_records` `:10574-10577`:
```
if _apg_db_dialect() == "pg":
    return []
```

**Mechanism.** FTS is SQLite-only. On PostgreSQL the function returns an empty list — not an error, not a 501, an empty result set. A caller cannot distinguish "no matches" from "search not implemented."

**Evidence: probe.** A PostgreSQL-configured generated app returned `[]` for a query matching existing records.

**Severity note:** this is worse than a missing feature because it is a silent wrong answer. A UI that shows "no results" for a term that is present teaches the operator the data is absent.

---

### D-24 — Declared indexes are parsed and dropped

**Anchor:** `IndexDeclaration` in `compiler/ast_builder.py`; `_sqlite_init_entity_table` `:6974-6993` emits `CREATE TABLE`, FTS, and a trigger, and no index.

**Mechanism.** The DSL accepts an index declaration and the generator ignores it. This is the class of defect feature 34 (round-trip verification) exists to catch: a construct that is syntactically accepted, semantically understood, and never emitted.

---

### D-25 — Auto-migration has no applied-revision ledger

**Anchor:** `_sqlite_auto_migrate_entity` `:6995`.

**Mechanism.** At startup the app compares `information_schema.columns` (PostgreSQL) or `PRAGMA table_info` (SQLite) against the expected column set and issues `ALTER TABLE` for the difference. There is no record of what ran. Two workers starting simultaneously both attempt the same DDL; one fails; the failure is swallowed. Rollback is undefined.

**Interaction with project policy:** the rule "never rewrite an applied migration revision" is currently a convention with no mechanism behind it.

---

## Observability

### D-14 — Unbounded metric label cardinality

**Anchor:** `_APG_HTTP_REQUESTS_TOTAL` `:12128`; `_APG_HTTP_REQUEST_DURATION` `:12129`; written at `:12170-12176`; template function `_apg_path_template` `:12133-12151`.

**Mechanism.** The template replaces numeric and UUID segments (`:12145-12151`) but leaves arbitrary path segments intact. An unknown path `/aaaa`, `/aaab`, … produces a distinct series per request, held forever in two dicts.

**Derived complexity.** Memory `O(distinct paths)` and `/metrics` exposition `O(distinct series)`. Both are attacker-controlled on an endpoint that is public by default (`_APG_GET_PUBLIC` at `:12881` includes `/metrics`).

**Classification.** This is a remote resource-exhaustion primitive, not a metrics-hygiene issue.

---

### D-15 — Rate-limit buckets never evicted

**Anchor:** `_APG_RATE_BUCKETS` `:12525`; written `:12547`, `:12550`, `:12557`. No `pop`, `clear`, or `del` in the function.

**Derived complexity.** One entry per distinct `remote_addr` (`:12545`), held for process lifetime. Behind a proxy without `X-Forwarded-For` handling, every client shares one bucket (the rate limiter does nothing); behind one with it, the bucket map grows with the client population.

**Secondary defect:** `_flask_request.remote_addr` is the proxy address unless the deployment configures `ProxyFix`. The limiter is either ineffective or memory-unbounded depending on deployment, and neither state is detectable from the app.

---

### D-16 — Email threads never pruned

**Anchor:** `_APG_EMAIL_THREADS` `:959`; appended at `:993`; started at `:994`.

**Mechanism.** Every send creates a `threading.Thread` and retains a reference in a module-global list. Finished threads are never removed, so the list grows with the total number of emails ever sent, retaining a thread object each.

**Secondary defect:** unbounded thread creation per send is the backpressure problem feature 28 addresses.

---

### D-17 — Webhooks fire-and-forget with in-memory retry

**Anchor:** `_apg_deliver_webhook` `:11709-11734`.

**Mechanism.** A daemon thread posts to each URL with retries at 0/1/2/4 seconds (`:11718`). The payload exists only in that thread's memory. On process exit, redeploy, or crash, in-flight and pending deliveries are lost silently. No delivery record is persisted, so an operator cannot answer "was event X delivered?" The HMAC is computed once (`:11715`), so a receiver cannot distinguish a retry from a fresh event.

**Fix:** feature 6 (transactional outbox) and feature 7 (dead-letter queue).

---

## Build and CI

### D-11 — CI has been red for at least five consecutive runs

**Evidence: static + observed.** The last five GitHub Actions runs on `main` report `failure`. Two independent causes.

**(a) Lint job.** `.github/workflows/ci.yml:32` runs
`flake8 compiler/ cli/ --max-line-length=120 --ignore=E501,W503`.
`W191` (indentation contains tabs) is not ignored, and the project's own standard mandates tabs (`ruff indent-style = "tab"` in `pyproject.toml`). The job fails on the code style the project requires.

**(b) Test job.** `ci.yml:17` runs `pip install -e '.[dev]'`, and the `dev` extra at `pyproject.toml:30-35` contains only `pytest`, `pytest-asyncio`, `pytest-httpserver`, `PyYAML`. Test collection fails on `cryptography`, `numpy`, `starlette`, and `flask_appbuilder`, none of which are installed.

**Consequence.** The gate that would have caught D-01 through D-10 has not run in at least five pushes. This is the highest-leverage defect in the document: every other fix is unverifiable until it is green.

---

### D-18 — Generated Dockerfile runs the Flask dev server

**Anchor:** `_generate_python_dockerfile` `:13029-13044`; `CMD ["python", "app.py"]` at `:13044`.

**Mechanism.** Flask's development server is single-threaded by default in some configurations, emits a warning that it must not be used in production, and has no worker management, no graceful shutdown, and no request timeout. `HEALTHCHECK` (`:13041-13042`) invokes `python app.py --self-test`, which runs a full self-test inside the container on every check interval rather than probing an endpoint.

**Derived consequence.** A generated app deployed as emitted serves one request at a time with no timeout. Combined with D-09's `O(N)` write path, a single large export blocks every other request.

---

### D-19 — `requires-python = ">=3.9"` but PEP 604 used at definition time

**Anchor:** `pyproject.toml:8` declares `>=3.9`. PEP 604 unions in annotations are evaluated at definition time unless `from __future__ import annotations` is present. Sites verified by AST walk over `FunctionDef` arguments:

- `compiler/code_generator.py:445`, `:446`
- `compiler/parser.py:149`
- `compiler/ast_builder.py:57`
- `cli/main.py:140`, `:144`, `:147`, `:148`, `:149`, `:150`, `:486`, `:688`
- `cli/openui_command.py:38`, `:46`

**Mechanism.** On Python 3.9 these raise `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'` at import. The CI matrix includes 3.9 (`ci.yml:9`), so this is one of the reasons the test job cannot collect.

**Note on the earlier false positive:** a raw regex scan reported 762 hits, most of them inside the 12,000-line f-string templates where `|` is not a type union. The AST walk is the correct predicate; the regex was a strict superset and would have failed open.

**Fix:** add `from __future__ import annotations` to each module, or raise the floor to 3.10.

---

## Structure

### D-20 — 260 duplicate store implementations

**Evidence: static.** `grep -rl 'class InMemoryStore' capabilities/` → 260 files. `grep -rl '^def get_store' capabilities/` → 264. `grep -rl 'from capabilities.common.db import' capabilities/` → 157.

**Mechanism.** A shared `Store`/`InMemoryStore`/`PostgreSQLStore`/`get_store` exists at `capabilities/common/db/store.py`, and roughly 260 capabilities carry their own copy. The copies have drifted — the shared one has a JSONB `@>` query path, the copies vary.

**Derived consequence.** A fix to persistence semantics must be applied 260 times, so in practice it is applied zero times. `count` in the shared store calls `query(..., limit=10_000_000)`, which materialises the whole result set to count it — the same shape as D-10, in the capability layer.

---

### D-21 — LSP re-parses the language with regex

**Anchor:** `language_server/server.py:225` `parse_entities`; pattern at `:227`; `_parse_fields` `:249`; `_parse_relationships` `:265`.

**Mechanism.** The editor's model of `.apg` is three regular expressions. The compiler's model is `compiler/parser.py` and `compiler/ast_builder.py` over 292 grammar productions. Every feature added to the grammar after the regex was written is invisible to completion, hover, and diagnostics.

**Derived consequence.** This is not a quality gap; it is a correctness gap with a maintenance multiplier. `language_server/semantic_service.py` adds a parallel surface built on `build_semantic_model_from_module`, so there are three models of the language: regex, semantic service, and compiler.

---

### D-22 — Dead code: `_apg_db_connect` leaks a semaphore

**Anchor:** `_apg_db_connect` `:6704-6735`; semaphore `_APG_DB_POOL_SEMAPHORE` `:6700`; `acquire()` at `:6718`.

**Mechanism.** The function acquires a bounded semaphore on the PostgreSQL path and never releases it on success (`:6718-6724` releases only on the exception path). The function has **zero callers** — the only occurrence of `_apg_db_connect` in the file is its own definition.

**Reading it.** Dead code that would be a leak if revived is worse than dead code: it reads as the connection-pool implementation, so the next person to wire pooling will wire this one.

---

### D-23 — `compiler/widget_rules.py` has no importers

**Evidence: static.** `grep -rn 'widget_rules'` across `*.py` and `*.toml`, excluding the file itself, returns nothing.

**Mechanism.** A module of widget selection rules that nothing imports. It is also the source of the W191 lint failure (D-11a) that keeps CI red, which makes it a pure liability.

---

## Cross-cutting: what the defects have in common

Four of the five security defects are the same defect. D-01, D-02, D-04, and D-06 all reduce to: **authorisation is applied at call sites rather than at a boundary**. The fifth, D-03, is a consequence of treating a source-level expression as trusted when the platform's premise is that generated apps are handed to third parties.

Three of the four scalability defects are the same defect. D-09, D-10, and D-12 all reduce to: **the generated runtime has no data plane**. It has a Python dict, a JSON file, and a connection function. Every operation is expressed over the whole collection because there is no interface that permits anything else.

Two of the three observability defects are the same defect. D-14 and D-15 both reduce to: **a map keyed by attacker-controlled input with no eviction**. The third, D-16, is the same shape with a different key.

That convergence is the argument for the roadmap's ordering. Fixing D-01 through D-06 individually produces six patches that the next route will bypass. Fixing the boundary produces one.
