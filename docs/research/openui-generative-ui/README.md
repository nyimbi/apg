# OpenUI Generative UI — research and APG seam

**Date:** 2026-10-24
**Scope:** What OpenUI (thesysdev/openui, MIT) is, what its language actually specifies, and where it meets APG's compiler and generated UI.
**Status:** research complete; prototype at `compiler/openui_library.py` + `compiler/openui_prompt.py`; 0 runtime dependencies added.

---

## 1. What OpenUI is

A **generative UI** framework. An LLM emits a program in **OpenUI Lang**; a client parses it incrementally and renders it with components from a library the host registered. Two ideas carry the design:

1. **Streaming-native, line-oriented.** One statement per line, `name = expression`. Every newline completes a statement that never changes again, so the client renders a valid tree after every chunk. The spec's invariant is explicit: *streaming equals parsing* — the end-of-stream tree MUST equal a batch parse of the same text.
2. **The component library is the schema, and the schema is the prompt.** You register components with typed props (JSON Schema). The prompt is *derived* from that schema, so the model can only emit components you can render. Constraint lives in the library, not in post-hoc validation.

Frameworks: `lang-core` (parser, prompt generation, evaluation — framework-agnostic) plus runtimes for React, Vue, Svelte, Angular, email, and a no-build browser bundle. Integrations for LangChain/LangGraph, assistant-ui, A2UI, Vercel AI SDK.

## 2. What the language actually is

The spec (v1.0) is precise and worth reading before designing against it. Key mechanics:

| Mechanism | Shape | Why it matters to APG |
|---|---|---|
| Statements | `name = expression`, newline-terminated | flat, hoisted, order-independent |
| Entry | `root` statement; fallback to the library's `root` component | one tree per program |
| References | bare name → statement, hoisted, cycles unresolved | no nesting syntax needed |
| Positional args | key order of `properties` in `$defs` **is** the argument order | prompt stays compact; adding props is append-only |
| Calls | `Component(...)`, `@Builtin(...)`, `@Action(...)` | `@` distinguishes calls from references |
| Streaming | pending tail re-parsed each chunk; implicit string/bracket closing; pending never overwrites completed | render after every token |
| State | `$name`, flat store, defaults never overwrite user values | two-way binding via `$binding<T>` props |
| Data | `Query(tool, args, defaults)`, `Mutation(tool, args)`, `@Run` | tools are host-supplied async functions |
| Actions | `@Set`, `@Reset`, `@Run`, `@ToAssistant`, `@OpenUrl`, custom | events reach the host as JSON |
| Editing | patch = changed statements only; merge by name; prune unreachable | cheap multi-turn refinement |
| Errors | drop-and-render with typed codes; never crash the host | partial UI beats blank screen |

Numbers claimed by the project: ~1,362 output tokens per screen across 46 benchmarked interfaces, 52–58% below JSON-based UI formats. That is the whole economic argument for a line grammar over JSON — position is meaning, so no keys are paid for twice.

## 3. Where APG already is

APG and OpenUI are both "generate the interface, don't hand-write it," but they sit at different layers:

| | APG | OpenUI |
|---|---|---|
| Generates | a whole application: schema, migrations, auth, routes, UI, evidence | one screen's structure, per request |
| When | compile time, deterministic, reviewable | runtime, model-driven, streaming |
| Substrate | Jinja templates + HTMX, server-rendered | component tree, client-rendered |
| Constraint | DSL grammar + semantic analyzer | component library → derived prompt |
| Persistence | SQLAlchemy models, migrations | none |

APG's relevant existing assets:

- `compiler/templates/*.j2` — 15 templates that *are* a component vocabulary (entity list, record detail, kanban, dashboard, agent console, workflow wizard).
- `compiler/assets/apg.css` — a design-token system (`--apg-primary`, `--apg-chart-0..5`, `.apg-card`, `.apg-badge`, `.apg-table`, `.apg-btn`).
- `compiler/widget_rules.py` — field-name → widget inference (map, kanban, chart, progress, heatmap). **Currently dead code**: no module imports it.
- `compiler/semantic_model.py` — `apg.semantic-model.v1`, exposing `tables` (fields, types, relationships), `agents`, `flows`, `capabilities`, `diagnostics`.
- SSE plumbing in the generated app (`/events`, `apg-sse.js`, `_publish_live_event`) — an existing streaming channel that currently carries *events*, not UI.

## 4. The seam

The natural integration is **not** "replace Jinja with OpenUI." It is: let a generated app's **agent console** emit interface structure instead of prose, using APG's own design system as the registered library.

```
semantic_model.json  ──►  LibrarySpec (APG components)  ──►  system prompt
                                                                    │
                                              Ollama/OpenAI ◄───────┘
                                                    │
                                        OpenUI Lang stream
                                                    │
                              parse (incremental) ──┴──► render (APG CSS classes)
```

Concretely, four steps, each independently shippable:

1. **Library from design system** — `compiler/openui_library.py`. Derive a `LibrarySpec` from a fixed APG component set, with entity-aware variants. *Prototyped.*
2. **Prompt generation** — `compiler/openui_prompt.py`. Deterministic, byte-stable prompt from the spec, following the spec's section 4.4 ordering. *Prototyped.*
3. **Parser** — `compiler/openui_lang/`. Lexer, expression parser, statement splitter with the three continuation rules, evaluator, and streaming parser. *Built and conformant* against the spec's own 58-case fixture suite (`tests/ci/test_openui_conformance.py`).
4. **Renderer** — two emitters over one component-to-class table. `compiler/openui_lang/render.py` paints server-side; `compiler/assets/apg-openui.js` parses and paints in the browser, so a streamed screen and a server-rendered one agree. *Built.*

Step 4 is where the risk sat, and the split above is the answer: OpenUI's own renderers are client-side JS, and APG's generated apps are deliberately no-build, no-CDN. The browser half is vanilla JS vendored as a static asset (no build, no CDN, no framework), held to the same fixture suite as the Python parser (`tests/ci/openui_render_js.test.js`, invoked from the Python suite so CI runs it).

**What a generated app ships.** `generate_deployment_bundle` emits `openui_library.json` and `openui_prompt.txt` beside `semantic_model.json`, and `app.py` serves the library at `/openui-library.json`. The agent console fetches that document and renders a reply written in OpenUI Lang through `APGOpenUI`, falling back to the sanitised text when the reply is prose. The library is emitted only when `validate_library` passes: a contract no client can satisfy is worse than none.

### Why the entity-aware library matters

A generic `Table(columns, rows)` library asks the model to re-derive what the compiler already knows. APG knows that `Patient.date_of_birth` is a `date`, that `Appointment.status` is a status field (kanban-eligible), that `Bill.total_amount` is currency. Encoding that as **typed, pre-shaped components** —

```
PatientTable(rows: PatientRow[], title?: string)
PatientRow(full_name: string, date_of_birth: date, gender: string, ...)
AppointmentKanban(columns: AppointmentColumn[])
```

— means the model picks a component and fills it, rather than choosing among generic primitives and getting the shape wrong. This is the same move APG already makes at compile time, moved to request time.

## 5. Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Adopt OpenUI's React runtime? | **No** | violates the no-build/no-CDN contract of generated apps |
| Adopt the language? | **Yes, as a wire format** | line grammar is cheaper than JSON and streams cleanly |
| Library source of truth | `compiler/openui_library.py` constants | derived from `apg.css` + template vocabulary; no drift-prone duplication |
| Component granularity | generic primitives **+** entity-specific composites | generic alone wastes the semantic model; entity-only alone cannot express novel screens |
| Where it runs | agent console, behind a flag | leaves the deterministic compile-time UI untouched |
| Ollama default | yes | matches APG's local-first model strategy; OpenUI is model-agnostic |

## 6. Rejected options

- **A2UI** — Google's alternative protocol, JSON-payload shaped. Loses the token advantage that motivates the whole exercise; OpenUI ships an adapter if it ever matters.
- **Vercel AI SDK `streamUI`** — requires React + a Node server; both absent by design.
- **Hand-rolled JSON UI schema** — this is exactly what the benchmark numbers argue against, and it forfeits the streaming guarantees the spec already worked out.
- **LLM-emitted raw HTML** — unbounded output surface, no constraint, XSS-adjacent, untestable.

## 7. Open questions

1. **Tool binding.** OpenUI's `Query(tool, args)` assumes a host tool registry. APG's generated apps have REST endpoints. The mapping is mechanical (`tool` → route path, `args` → query params) but needs a naming convention.
2. **State and the URL.** OpenUI state is in-memory per program. APG's generated UI is link-driven. Decide whether `$state` mirrors query params.
3. **Determinism budget.** A model-generated screen cannot be snapshot-tested the way compile-time output is. Bound it: the *library* and the *prompt* are snapshot-tested (done); the *program* is validated, not compared.
4. **Failure mode.** Spec §8 says drop-and-render. Confirm the renderer never emits a blank page — a partial card beats nothing.

## 8. Prototype

`compiler/openui_library.py` — builds a `LibrarySpec` (`id`, `root`, `schema`, `components`) from APG's component set, optionally specialised per entity from `apg.semantic-model.v1`.

`compiler/openui_prompt.py` — `generate_system_prompt(spec)` → deterministic prompt text: syntax rules, component signatures, built-ins, streaming guidance, in the spec's prescribed order.

`tests/ci/test_openui_library.py` — asserts library conformance (naming, required-before-optional, no reserved names), prompt determinism, entity specialisation, and that every generated signature parses back to its schema entry.

Sources archived in `sources/`; index in `sources/INDEX.md`.
