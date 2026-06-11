# loc_mlg — 15 World-Class Improvements

## 1. Translation Memory with Fuzzy Matching

**Problem**: Repeated near-identical strings are translated from scratch each time, wasting cost and introducing inconsistency.

**Improvement**: Add a `TranslationMemory` store that indexes all approved translations. On `create_translation`, run a Jaccard/cosine similarity check against the memory store and surface matches above a configurable threshold (e.g. ≥0.85) as suggestions. Reduces Ollama inference calls by an estimated 30–60% for iterative content.

**New methods**: `translation_memory_suggest`, `translation_memory_import`, `translation_memory_export`

---

## 2. Namespace-Scoped Translation Versioning with Conflict Detection

**Problem**: Multiple translators can overwrite the same key in the same namespace without awareness of concurrent edits.

**Improvement**: Add an optimistic locking field (`version: int`) to `TranslationResponse`. `update_translation` rejects writes with stale `version` values and surfaces a `TranslationConflictError`. Combine with `translation_diff` to show what changed between versions.

**New methods**: `translation_diff`, `update_translation` (with version guard)

---

## 3. Automated RTL Layout Validation

**Problem**: RTL compliance is checked only at locale-config time. UI strings added later may contain hard-coded LTR punctuation, directional marks, or mixed-direction numbers.

**Improvement**: Add `validate_rtl_string` that uses Unicode BiDi algorithm rules to check a string for directional anomalies and returns a list of character-level issues with byte offsets.

**New methods**: `validate_rtl_string`, `validate_rtl_batch`

---

## 4. Locale-Aware Number and Date Formatting Engine

**Problem**: `FormattingRuleResponse` stores format *metadata* but the service never applies it — downstream renderers must re-implement the same logic.

**Improvement**: Add `format_number`, `format_date`, `format_currency` methods that take a raw value and a `locale_id` and return a formatted string using the stored rules. Single source of truth for formatting behaviour.

**New methods**: `format_number`, `format_date`, `format_currency`

---

## 5. Translation Coverage Heatmap Aggregation

**Problem**: `missing_translations_report` only compares to `en` and produces a flat list. Product managers need a language × namespace coverage matrix.

**Improvement**: Add `coverage_matrix` that returns a `dict[language, dict[namespace, float]]` of completion percentages across all active locales. Feeds directly into monitoring dashboards without client-side aggregation.

**New methods**: `coverage_matrix`

---

## 6. Glossary-Enforced Translation Validation

**Problem**: Translators can use forbidden terms or ignore preferred translations; there is no enforcement path.

**Improvement**: Add `validate_against_glossary` that checks a translated text against `TerminologyResponse` entries for the target language and returns a `GlossaryViolation` list with position, forbidden term used, and suggested replacement.

**New methods**: `validate_against_glossary`

---

## 7. Locale Inheritance / Overrides

**Problem**: `en_KE` and `en_GB` share most strings. Storing full copies for each region locale wastes storage and creates synchronisation drift.

**Improvement**: Add `parent_locale` to `LocaleConfigResponse`. `lookup_translation` walks the inheritance chain (child → parent → default) before returning `None`. Similar to CSS cascade — regional locales only store overrides.

**New methods**: `set_locale_parent`, `lookup_translation` (updated chain walk)

---

## 8. Batch Translation Status Transition

**Problem**: Approving or publishing translations one-by-one is O(n) API calls for bulk content releases.

**Improvement**: Add `batch_approve_translations` and `batch_publish_translations` that accept a list of IDs and execute the workflow transition atomically, returning a per-ID success/failure map. Failed IDs do not block successful ones (partial commit with error report).

**New methods**: `batch_approve_translations`, `batch_publish_translations`

---

## 9. Webhook / Event Sink Registration

**Problem**: External systems (CMS, headless front-end) poll for translation status. The bytewax stream is internal; external consumers need a push mechanism.

**Improvement**: Add `register_webhook` that stores a URL + event filter. `_emit` becomes async-fanout — after writing the audit event it fires registered webhooks with the event payload using `httpx.AsyncClient`. Retries with exponential backoff up to 3 attempts.

**New methods**: `register_webhook`, `list_webhooks`, `delete_webhook`

---

## 10. Translator Workload and SLA Tracking

**Problem**: There is no visibility into translator queue depth or SLA compliance. Overloaded translators cause silent review bottlenecks.

**Improvement**: Add `translator_workload` that returns per-translator counts of `draft`, `pending_review`, and `approved` translations plus average days-to-review. Supports deadline SLA enforcement by flagging translations older than a configurable threshold.

**New methods**: `translator_workload`, `sla_violations_report`

---

## 11. Context-Aware Machine Translation Prompting

**Problem**: `machine_translate_batch` submits raw text with no domain or terminology context. Ollama models produce generic translations that ignore glossary entries.

**Improvement**: Before sending to Ollama, prepend a system prompt that includes (a) the relevant `TerminologyResponse` entries for the target language/domain and (b) any existing approved translations for sibling keys in the same namespace. This is retrieval-augmented translation.

**New methods**: `machine_translate_with_context`, updated `machine_translate_batch`

---

## 12. Translation Rollback

**Problem**: Publishing a bad translation has no recovery path except manually creating a new translation — breaking the audit trail.

**Improvement**: Add `rollback_translation` that clones a prior approved version of a translation key (identified by version number) back into `published` status while archiving the current published entry as `deprecated`. Full audit trail preserved.

**New methods**: `rollback_translation`, `get_translation_history`

---

## 13. Locale-Specific Plural Rule Engine

**Problem**: `plural_rule_define` stores a rule expression but never evaluates it. Applications must implement plural selection themselves.

**Improvement**: Add `resolve_plural_form` that accepts a count and a language code, evaluates the stored CLDR-compatible plural rule expression, and returns the correct plural category (`zero`, `one`, `two`, `few`, `many`, `other`). Integrates with `lookup_translation` via a `count` parameter.

**New methods**: `resolve_plural_form`, updated `lookup_translation`

---

## 14. AI-Assisted Translation Review Scoring

**Problem**: Human reviewers receive translations cold with no quality signal. Inconsistent reviews increase cycle time.

**Improvement**: Add `score_translation_quality` that sends the source + translated text pair to Ollama with a structured QA prompt returning scores across five dimensions: accuracy, fluency, terminology adherence, style consistency, and cultural appropriateness. Scores are stored on `TranslationResponse.quality_score` and surfaced in the review UI.

**New methods**: `score_translation_quality`

---

## 15. Locale Sync Across Tenants (Super-Admin)

**Problem**: Platform operators maintain dozens of tenants that should share a common baseline locale configuration. Any update must be manually replicated.

**Improvement**: Add `sync_locale_baseline` (super-admin only) that takes a source `tenant_id` and a list of target `tenant_id` values and copies all `is_default=True` locale configs and global formatting rules to the targets, skipping entries that already exist (idempotent). Returns a per-tenant sync report.

**New methods**: `sync_locale_baseline`
