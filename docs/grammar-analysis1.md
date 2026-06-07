# APG Language: Grammar Analysis — Shortcomings and Design Flaws

*Author: Nyimbi Odero / Datacraft*
*Date: 2026-06-06*
*Scope: spec/apg.g4 v11, compiler/ast_builder.py, compiler/semantic_analyzer.py, examples/*

> **Living document.** This analysis is updated as issues are resolved. The Resolution Status
> section below tracks progress against commits 9c547d7c–6a661a29 (Phases 1–7). Open issues
> remain on the roadmap; partial fixes are noted with caveats.

---

## Methodology

This analysis examines the grammar specification (`spec/apg.g4`, 4,375 lines), the actual compiler
implementation (`compiler/ast_builder.py`, 1,424 lines; `compiler/parser.py`, 354 lines;
`compiler/semantic_analyzer.py`, 838 lines), and representative APG programs from the `examples/`
directory. Issues are ordered by architectural severity.

## Resolution Status

Progress as of commits 9c547d7c–6a661a29 (Phases 1–7):

| Issue | Severity | Status | Notes |
|-------|----------|--------|-------|
| §1 Dual-parser architecture | Critical | Partial | `compiler/antlr_ast_visitor.py`: ANTLR drives entity boundary detection; body parsing still uses regex. `antlr_clean` gate prevents broken trees from corrupting AST. **Known blocker**: implicit keyword tokens in ANTLR parser rules (e.g. `'id'` in `capability_contract_member`) shadow IDENTIFIER for common field names, causing `antlr_clean=False` on all practical APG files — the visitor path is never reached in practice. Next step: disambiguate implicit keywords in `spec/apg.g4`. Full migration is Phase 8. |
| §2 100+ entity_type keywords | High | Partial | `spec/apg.g4`: added an `IDENTIFIER` alternative — any identifier is now a valid entity kind. Historic keyword list retained for backward compatibility. Grammar regenerated. |
| §3 Rule conditions as strings | High | Done | `compiler/rule_expr.py`: recursive descent parser for `when` conditions. `parse_rule_expr`, `extract_fields`, `validate_rule_fields`, `expr_to_dict`. Integrated into semantic_analyzer; conditions stored as `when_ast` dicts in semantic model. |
| §4 Workflow as string | High | Done | `compiler/ast_builder.py`: `WorkflowDeclaration` with typed `states`, `transitions`, `human_tasks`, `guards`, `assignments`, `timers`, `waits`, `retry_policy`, `compensation`. Validated in semantic_analyzer. Exposed in semantic_model flows. |
| §5 No cross-entity validation | High | Done | `compiler/semantic_analyzer.py`: `_resolve_references()` Phase 5 pass. Validates `requires`, `provides`, `capabilities`, `agents` against local symbols and MANIFEST.json. Emits warnings (not errors). |
| §6 Inheritance declared not implemented | Medium | Open | `extends` keyword accepted by grammar but not semantically resolved. Planned for a later phase. |
| §7 Import system grammar-only | Medium | Partial | `compiler/compiler.py`: `_resolve_imports()` resolves dot-separated module names to relative `.apg` file paths. Lenient (missing files silently skipped). No semver constraints or lock file. |
| §8 contract_separator ambiguity | Medium | Open | Both `;` and `,` remain valid separators. Formatter normalization planned. |
| §9 Agent memory parsing ambiguous | Medium | Open | `memory: {kind: vector, name: sales_memory}` structured form not yet enforced. |
| §10 Generated code is config only | Medium | Partial | `compiler/code_generator.py`: `_generate_agent_stubs()` produces typed `AgentBase` subclasses in `agent_stubs.py`. Capability router stubs not yet generated. |
| §11 Streaming declared not expressed | Medium | Open | `streaming: {processor: bytewax}` remains a declaration tag without topology. |
| §12 Physical literals discarded | Low | Open | `NUMBER PHYS_UNIT` tokens parsed but type-erased in property values. |
| §13 Form layout sub-language stranded | Low | Open | `form` entity bodies capture ANTLR context but form-field validation against table schema not implemented. |
| §14 No versioning for generated artifacts | Low | Open | Module version strings stored but not cross-validated. No lock file generated. |

---

## 1. The Dual-Parser Architecture Is the Root Problem

**Severity: Critical**

The grammar (`spec/apg.g4`) is 4,375 lines. The compiler ignores ~95% of it. After ANTLR parses
the source, the result is handed to `_build_source_ast()` — a 1,000-line regex engine that
re-parses the raw source text. The ANTLR parse tree is discarded.

This means every language feature requires two implementations:

1. An ANTLR grammar rule (in `spec/apg.g4`)
2. A regex handler (in `compiler/ast_builder.py`)

The consequences are severe:

- The regex parser cannot handle deeply nested constructs. The ERP platform bug (`capability`
  with nested `components: {}`) failed because `_parse_value()` choked on empty parts from
  trailing commas in nested objects — a problem that would not exist with a proper parse tree.
- Error messages are generic: `Failed to build AST`, `Missing semicolon at 468:87`. The exact
  position reported is the character count into the raw string, not a meaningful source location.
- Any new grammar feature produces a false sense of progress — the grammar rule is added but
  nothing actually parses it.
- Two separate parsers for the same language (`looks_like_ai_agent_composition()` → regex agent
  parser; everything else → ANTLR → regex ast builder) diverge over time and create edge cases
  that are expensive to diagnose.

**Recommended fix:** Commit to ANTLR. Delete `_build_source_ast()` and implement a proper
ANTLR `ParseTreeVisitor` subclass. The regex approach cannot scale.

---

## 2. `entity_type` Has 100+ Reserved Keywords

**Severity: High**

```antlr
entity_type
    : 'agent' | 'team' | 'agent_team' | 'robot' | 'sensor' | 'camera' | 'actuator' | 'drone'
    | 'chat' | 'llm' | 'db' | 'table' | 'biz' | 'flow' | 'rule'
    | 'monitor' | 'alert' | 'map' | 'classify' | 'profile' | 'context'
    ... // 80+ more
```

Over 100 domain-specific keywords are reserved at the grammar level. Problems:

- **Namespace exhaustion.** `alert`, `map`, `monitor`, `classify`, `profile`, `context` are all
  reserved. An enterprise domain model with an `Alert` entity that is *not* an intel/monitoring
  alert cannot be expressed without fighting the lexer.
- **Lexer performance.** Every identifier token must be checked against this list at scan time.
- **No user extensibility.** You cannot define `entity_kind: supply_chain_order` without modifying
  the grammar source.
- **False homogeneity.** `biz`, `erp`, `crm`, `sales`, `hr`, `payroll`, `manufacturing`,
  `procurement` are all first-class entity types with identical grammar bodies — they convey
  domain intent but have no distinct semantic treatment in the compiler.

**Recommended fix:** Keep 8–10 first-class primitives: `table`, `workflow`, `agent`, `capability`,
`app`, `enum`, `event`, `rule`. Domain specificity is expressed via decorator annotations:
`@intel table Alert { ... }`, `@hr workflow Onboarding { ... }`. This is the approach used by
Protocol Buffers, Smithy, and Avro.

---

## 3. Rule Conditions Are Opaque Strings

**Severity: High**

```apg
rules: [
    {name: "large_deal", when: "amount > 50000 and stage == 'qualification'", action: require_review},
    {name: "cross_tenant", when: "contact_tenant != actor_tenant", action: deny}
]
```

`when` conditions are stored verbatim as strings and evaluated at runtime by whatever the
capability's rule engine does. Compile-time consequences:

- No validation that `amount` is a field on the ambient table.
- No type checking (`amount > "hello"` is syntactically valid APG).
- No IDE autocomplete for field names within condition strings.
- No refactoring support — renaming `amount` to `deal_value` silently breaks every rule that
  references it.
- Cannot compose conditions — `when: "A and B"` where `A` and `B` are independently named and
  reused predicates.

The grammar already contains a full expression language (`comparison`, `conditional_expr`,
`bitwise_or`, `pipeline_expr`, etc.) but it is never applied to rule conditions. The two
subsystems coexist without meeting.

**Recommended fix:** Parse `when` clauses using the full APG expression grammar against the
ambient symbol table. Type-check operands at compile time using the field types from the
enclosing `table` or `capability` context. This is the single highest-ROI improvement available.

---

## 4. Workflow Steps Are a String, Not a Graph

**Severity: High**

```apg
workflow LeadQualification {
    steps: str = "new_lead -> researched -> contacted -> qualified -> opportunity_created";
    human_tasks: [contacted, qualified];
    guards: {qualified: "budget_confirmed and timeline_defined"};
}
```

The state machine is stored as a string. Compile-time consequences:

- No validation that `human_tasks` entries (`contacted`, `qualified`) are actual states in the
  `steps` string.
- No validation that `guards` keys are actual state names.
- No support for parallel gateways, XOR branches, sub-processes, error events, or timers —
  all of which appear in realistic business workflows.
- `guards` conditions are also opaque strings (same problem as §3).
- The grammar *does* have `flow_definition` with `conditional_flow_step` and `parallel_flow_step`,
  but the parser discards them.

**Recommended fix:** Parse `steps` as a proper transition graph. Represent each `source -> target`
edge as a typed `Transition(source: str, target: str, guard: Expression | None)`. Validate
`human_tasks` and `guards` against the resolved state set. Extend to support branching:

```apg
workflow DealApproval {
    states: [submitted, manager_review, finance_review, approved, rejected];
    transitions: [
        submitted    -> manager_review  [human: sales_manager],
        manager_review -> finance_review [when: "amount > 100000", human: finance_controller],
        manager_review -> approved       [when: "amount <= 100000"],
        finance_review -> approved,
        finance_review -> rejected
    ];
}
```

---

## 5. No Cross-Entity Reference Validation

**Severity: High**

The semantic analyzer validates entity definitions in isolation. It does not validate
cross-entity references:

```apg
capability CRMCore {
    contract: {
        requires: [auth, audl, ntfy, wflo],        // Do these exist? Not checked.
        provides: [contact_lifecycle, opportunity_pipeline]
    }
}
agent SalesAssistant {
    capabilities: [contact_lifecycle, opportunity_pipeline],  // Match CRMCore.provides? Not checked.
    tools: [contact_search, deal_analysis]                    // Defined anywhere? Not checked.
}
```

Broken references produce no compile-time error. This is the core failure of a language whose
primary value proposition is composability.

**Recommended fix:** After the entity symbol table is built, run a second-pass resolver that
validates all `requires`, `provides`, `capabilities`, `agents`, `binds`, and `tools` identifiers
against declared entities and known capability contracts.

---

## 6. Inheritance Is Declared but Not Implemented

**Severity: Medium**

```apg
entity Customer extends BaseEntity { ... }
```

The grammar has `inheritance: 'extends' IDENTIFIER`. The `EntityDeclaration` dataclass has no
`parent` field. The semantic analyzer never:

- Checks that the parent entity (`BaseEntity`) is declared in scope.
- Copies parent properties into the child's `properties` list.
- Validates that overridden fields are compatible with the parent definition.
- Produces a type hierarchy usable by downstream code generation.

`extends` is pure decorative syntax — it compiles to nothing.

**Recommended fix:** Either implement single-table inheritance (merge parent fields into child
`properties` during semantic analysis, marking inherited fields with an `inherited` flag), or
remove `extends` from the grammar entirely to avoid creating false expectations.

---

## 7. Module/Import System Is Grammar-Only

**Severity: Medium**

```apg
module crm_platform version 1.0.0 {
    description: "Composable CRM platform";
    dependencies: [common.types @>=1.2.0, sales.contracts @^2.0.0];
}
import sales.contracts;
from common.types import CustomerStatus;
```

The grammar has `import_statement`, `include_statement`, `export_statement`, and
`module_declaration` with full semver dependency constraints. None of these are handled by
`_build_source_ast()`. An `import` statement in an APG file is silently ignored or causes a
parse error depending on placement.

Multi-file APG programs are impossible: everything must live in one file. For complex enterprise
applications this is a practical blocker.

**Recommended fix:** Implement `import` as the first priority after the ANTLR visitor migration
(§1). Process imports before entity resolution. Build a module dependency graph with cycle
detection. Generate a lock file (`apg.lock`) that pins resolved module versions.

---

## 8. The `contract_separator` Semicolon/Comma Ambiguity

**Severity: Medium**

```antlr
contract_separator: ';' | ',';
```

Inside contract bodies, `;` and `,` are both valid separators. Writers have no consistent
mental model for which to use where:

```apg
// Both of these are valid in the same contract:
provides: [contact_lifecycle, account_management],    // comma-terminated
requires: [auth, audl];                               // semicolon-terminated
configuration: {tenant_id: "default"},                // comma (like JSON)
rules: [...]                                          // no separator needed?
```

The formatter cannot normalize without a declared preference. Generated examples are inconsistent,
making the language look unpolished and confusing copy-paste patterns.

**Recommended fix:** Pick one separator. Inside object/array literals: `,`. At the statement
level: `;`. Document this as a hard rule and enforce it in the formatter and semantic analyzer.
`contract_separator: ';' | ','` is a symptom of accumulated debt from early iterative design.

---

## 9. Agent Memory Parsing Is Ambiguous

**Severity: Medium**

```antlr
agent_memory_value: IDENTIFIER IDENTIFIER?
```

`vector sales_memory` is parsed as two consecutive IDENTIFIERs. This is ambiguous with:
- `vector` on a line immediately followed by a field name on the next line
- Multi-word memory specs: `redis cluster sales_memory` produces wrong results

The regex fallback:
```python
parts = str(mem_raw).split()
memory = AgentMemory(kind=parts[0], name=parts[-1])
```
silently drops the middle word and produces misleading results for any spec with more than
two space-separated tokens.

**Recommended fix:** Make memory a structured declaration:
```apg
memory: {kind: vector, name: sales_memory, ttl: 7d, backend: pgvector}
```
or a keyword-prefixed literal with mandatory braces:
```apg
memory vector { name: sales_memory; ttl: 7d; }
```

---

## 10. Generated Code Is Configuration Serialization, Not Code Generation

**Severity: Medium**

Compiling the following APG:

```apg
agent SalesAssistant {
    role: "sales assistant";
    model: "openai:gpt-4.1-mini";
    runtime: codex;
    capabilities: [contact_lifecycle, opportunity_pipeline];
}
```

Produces:

```python
# apg_application.py (generated)
APG_AGENTS = [{
    "name": "SalesAssistant",
    "role": "sales assistant",
    "model": "openai:gpt-4.1-mini",
    "runtime": "codex",
    ...
}]
```

This is YAML-equivalent metadata serialization. The `runtime: codex` declaration has no effect
on the generated output — it is never wired to an actual agent runtime invocation. No generated
class exists, no interface is emitted, no integration code routes to the declared model.

A language that compiles to pure configuration has limited value over YAML or TOML. The claim
of APG as a "programming language" is only valid if it generates executable code.

**Recommended fix:** Generate concrete Python stubs:

```python
# generated/agents/sales_assistant.py
from apg.runtime.agent import AgentBase, AgentContext

class SalesAssistant(AgentBase):
    name = "SalesAssistant"
    model = "openai:gpt-4.1-mini"
    role = "sales assistant"

    async def invoke(self, prompt: str, context: AgentContext) -> str:
        # Routed to declared runtime
        return await self._runtime.chat(prompt, context)
```

For capabilities: generate FastAPI/Flask endpoint stubs that call the corresponding
`capability_contract.py` method.

---

## 11. Streaming Is Declared, Not Expressed

**Severity: Medium**

```apg
streaming: {processor: bytewax, state: crm_event_state}
```

This declares that Bytewax is used, but specifies nothing about the topology:

- What events are produced (sources)
- What transformations apply (operators: map, filter, aggregate, join)
- What downstream capabilities consume the stream (sinks)
- What the windowing or watermarking strategy is
- What happens on processing failure (dead-letter queue, retry, discard)

This is equivalent to writing `database: postgres` without specifying any schema or queries.

**Recommended fix:** Add a first-class `stream` entity with source/operator/sink topology:

```apg
stream CRMEventStream {
    source: crm_event_bus;
    operators: [
        filter:    "event_type in ['opportunity_created', 'deal_closed']",
        aggregate: {window: 5min, by: ["account_id"], compute: ["sum(amount)", "count(*)"]}
    ];
    sink: crm_analytics_store;
    on_error: dead_letter_queue;
}
```

---

## 12. Physical Literals Are Parsed but Discarded

**Severity: Low**

```apg
threshold: 80°C
pressure: 150psi
```

`physical_literal: NUMBER PHYS_UNIT` generates a `PHYS_UNIT` lexer token. However,
`_parse_value()` in `ai_agent_composition.py` has no case for physical literals — values
containing unit suffixes are treated as bare identifiers or cause parse failures. The unit
information (°C, psi, kPa, rpm) is silently dropped.

This matters for digital twin and industrial monitoring use cases where unit correctness is
safety-critical — storing `80` when the programmer wrote `80°C` is a type erasure bug.

**Recommended fix:** Add `PhysicalLiteral(value: float, unit: str, si_equivalent: float)` to
the AST and represent it in the semantic model. The semantic analyzer should type-check that
physical literals are only assigned to fields annotated with compatible unit types.

---

## 13. The Form Layout Sub-Language Is Stranded

**Severity: Low**

The form layout grammar (lines 1,989–2,458 of `spec/apg.g4`) defines a rich UI DSL with
containers, fields, components, responsive breakpoints, validation rules, animations, and
accessibility attributes. However:

- `form` entities are parsed as generic `EntityDeclaration` with `entity_type = EntityType.FORM`.
- The layout body is not parsed — only the outer form declaration is captured.
- There is no semantic link between a `form` entity and the `table` it edits.
- No validation that field names in the form match field names in the ambient table.

Nearly 500 lines of grammar are unreachable.

**Recommended fix:** After implementing the ANTLR visitor (§1), resolve `form` layout fields
against the table declared in the form's `binds` or `table` property. Validate field types
match the declared input widget type (e.g., `currency` field → `decimal` table column).

---

## 14. No Versioning Semantics for Generated Artifacts

**Severity: Low**

The grammar has rich version constraint syntax:

```antlr
version_range: SEMVER | '>=' SEMVER | '~' SEMVER | '^' SEMVER | SEMVER '..' SEMVER
```

But:

- Dependency version constraints between modules are never resolved or validated.
- No lock file is generated.
- `module foo version 1.2.3` has no semantic meaning in the compiler beyond storing the string.
- `capability_contract.py` files contain hardcoded version strings that are never cross-validated
  against the version declared by any module that depends on them.

---

## Priority Matrix

| # | Issue | Severity | Effort | Value Unlock |
|---|-------|----------|--------|--------------|
| 1 | Switch to ANTLR visitor | Critical | High | All fixes below |
| 2 | Parse rule conditions as APG expressions | High | Medium | Composability correctness |
| 3 | Cross-entity reference validation | High | Medium | Composability correctness |
| 4 | Implement import statements | High | High | Multi-file programs |
| 5 | Workflow as typed state graph | High | Medium | ERP/BPM viability |
| 6 | Reduce entity_type keywords | High | Medium | Extensibility |
| 7 | Generate executable stubs | Medium | High | Language viability claim |
| 8 | Implement extends inheritance | Medium | Medium | Type system completeness |
| 9 | Fix contract_separator ambiguity | Medium | Low | DX and formatter |
| 10 | Physical literals | Low | Low | Digital twin accuracy |

---

## Structural Observation

Many of these issues share a common root: the grammar was extended aspirationally while the
implementation remained conservative. The grammar is a vision document; the compiler is a
prototype. The gap between them is the primary technical debt.

The priority is not to extend the grammar further — it is to close the implementation gap
on the features already specified. Items 1–5 above would produce a compiler that correctly
validates the features already demonstrated in the `examples/` directory.

Items 6–14 are design improvements for the grammar itself, best addressed after the
implementation baseline is solid.
