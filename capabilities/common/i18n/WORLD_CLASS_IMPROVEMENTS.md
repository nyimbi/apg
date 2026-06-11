# World-Class Improvements: APG i18n Capability

**Capability**: Internationalisation (i18n)
**Path**: `capabilities/common/i18n`
**Date**: 2026-06-11
**Author**: Nyimbi Odero, Datacraft

---

## 1. CLDR-Backed Plural Rules Engine

Replace the naive `{"other": "n != 1"}` default with a full Unicode CLDR
plural-category engine.  Store CLDR rule strings per locale (`zero`, `one`,
`two`, `few`, `many`, `other`), expose a `pluralise(key, count, locale)` helper
that selects the right category at runtime, and validate incoming rule strings
against the CLDR DSL grammar.  This eliminates broken plurals in Arabic (6
forms), Russian (4 forms), Polish (3 forms), and dozens of African languages.

## 2. ICU MessageFormat 2.0 Template Support

Introduce a lightweight MessageFormat 2 parser/resolver so translation strings
can carry variable interpolation, select expressions, and plural selectors in a
single key — e.g. `{count, plural, one{# item} other{# items}}`.  The resolver
is pure-Python, dependency-free, and falls back gracefully on unparseable
strings.  This replaces ad-hoc `str.format()` patching scattered across
consumer code.

## 3. Babel/Locale-Data Integration for Date/Number/Currency Formatting

Replace `_apply_date_format` and `_apply_number_format` with real Babel
`format_date`, `format_number`, and `format_currency` calls backed by the CLDR
locale database.  Keeps the same async method signatures but eliminates the
heuristic separator-swap logic and produces correct output for locales whose
conventions differ fundamentally from Western European defaults (e.g. Ethiopian,
Amharic, Hindi number grouping).

## 4. Streaming Translation Export (AsyncIterator)

Convert `translation_export` to a true async generator that yields serialised
chunks (NDJSON lines, PO paragraphs, CSV rows) instead of accumulating the
entire payload in memory.  Large catalogs (100 k+ keys) currently exhaust
heap; streaming export lets consumers pipe directly to S3, GCS, or an HTTP
response without buffering.

## 5. Locale-Aware Collation and Sort Keys

Add `async locale_sort(tenant_id, locale_code, items)` that returns items
ordered by CLDR collation rules for the locale.  Naïve `sorted()` produces
wrong alphabetical order for Swahili, Yoruba, Czech, and most CJK scripts.
Backed by `icu4c` via `PyICU` with an optional pure-Python `pyuca` fallback.

## 6. Translation Memory Fuzzy-Match Score Threshold

The current `TranslationMemoryMatcher` returns the first match.  Replace with
a scored approach: compute normalised edit distance (Levenshtein / Jaro-Winkler)
between source_text and candidate entries, expose `min_score` (0–100) as a
parameter, and return the best-scoring match above the threshold plus its score.
This matches CAT-tool industry standard TM leverage behaviour and prevents
low-confidence re-use.

## 7. Differential Coverage Alerts

Extend `coverage_report` to compare the new snapshot against the previous
stored report and emit a structured delta: keys gained, keys lost, coverage
% change, and a severity band (stable / degraded / critical).  Publish deltas
as CloudEvents to the `bytewax` event stream so CI pipelines and dashboards can
react to regressions without polling.

## 8. Pseudo-Localisation Mode

Add `async pseudo_localise(tenant_id, locale_code, key_pattern)` that produces
synthetic translations by padding strings with extra characters (inflating length
by 30-40% as per W3C internationalisation guidance), wrapping text in `[!!!`
and `!!!]` markers, and replacing ASCII letters with visually similar Unicode
equivalents.  This lets QA engineers test UI layout breakage before real
translations exist, a standard practice in enterprise i18n toolchains.

## 9. Namespace-Scoped Translation Keys

Impose a two-part key schema `<namespace>.<key>` with namespace registration,
per-namespace ownership, and per-namespace coverage reports.  Current flat keys
become unmaintainable in large apps; namespaces enable team-level ownership,
bulk export per product area, and fine-grained RBAC ("marketing team can only
translate `marketing.*` keys").

## 10. AI-Assisted Glossary Consistency Checker

Add `async glossary_consistency_check(tenant_id, locale_code)` that scans all
published translations for the locale, detects where a source term appears in
`source_text` but the corresponding glossary term's `localized_terms` entry is
not present in `translated_text`, and returns a ranked inconsistency report.
Surfaces terminology drift without requiring a full MT pass.

## 11. Locale Lifecycle State Machine

Replace the implicit "a locale exists or it doesn't" model with an explicit
state machine: `draft → active → deprecated → archived`.  Add transitions
`async locale_activate`, `async locale_deprecate`, `async locale_archive` with
guard rules (cannot publish to deprecated, cannot translate to archived).
Enables safe sunset of obsolete language variants without data deletion.

## 12. Bulk Translation Diff / Merge

Add `async translation_diff(tenant_id, locale_code, incoming_entries)` that
computes a three-way diff between the current store, the reference locale, and
an incoming import payload, producing a per-key action plan: `keep`, `update`,
`conflict`, `add`, `delete`.  Callers can review the plan before committing
with `translation_merge_apply(plan_id, approved_keys)`.  This mirrors how
professional CAT tools handle catalog synchronisation.

## 13. Fallback Chain Audit Trail

Enrich `resolve_text` to record which locale in the fallback chain actually
served the text, the chain length, and the number of misses traversed.  Emit
this as a structured event so dashboards can identify "hot" fallback locales
that are carrying disproportionate traffic — a signal to prioritise translation
effort.

## 14. Locale-Specific Timezone DST Awareness

Extend `date_localise` to accept an `aware` flag that expands the timezone
field from a bare IANA zone name to a full DST-aware `ZoneInfo`-backed
conversion before formatting.  Currently the `timezone` field is stored but
never used in formatting, meaning scheduled content (announcements, deadlines)
renders in wrong wall-clock time for users in non-UTC zones.

## 15. Translation Linting Pipeline

Add `async translation_lint(tenant_id, locale_code, rules)` where `rules` is a
list of linting directives: `no_untranslated` (source == translated),
`html_tag_parity` (same HTML tags in source and translation),
`placeholder_parity` (same `{var}` tokens), `max_length(n)` (translated text ≤
n chars), `no_double_spaces`, and `required_punctuation_end`.  Returns
per-entry violations with severity and suggested fixes.  This is the
quality-gate layer missing from the current workbench.
