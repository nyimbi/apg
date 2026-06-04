"""Deterministic domain rules for NLP Core.

Every rule is a pure function.  RuleViolation is the single exception type
used for governance failures.  Service methods call assert_* functions at
entry; no rule logic lives in service.py.
"""
from __future__ import annotations

from typing import Any

try:
	from ..models import LanguageCode
except ImportError:
	from models import LanguageCode  # type: ignore[no-redef]


class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"Rule '{rule_name}' violated: {reason}")


# ---------------------------------------------------------------------------
# Tenant / access rules
# ---------------------------------------------------------------------------

def assert_tenant_context(context: dict[str, Any]) -> None:
	"""All operations require a non-empty tenant_id in context."""
	if not context.get("tenant_id"):
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required",
			"attach_tenant_context",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant access is always denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			"cross-tenant access is not permitted",
			"use_own_tenant_resources",
		)


def assert_write_policy(context: dict[str, Any]) -> None:
	"""Write operations require an attached policy."""
	if context.get("operation_type") == "write" and not context.get("policy_attached"):
		raise RuleViolation(
			"write_requires_policy",
			"write operations require an attached policy",
			"attach_policy",
		)


# ---------------------------------------------------------------------------
# Input validation rules
# ---------------------------------------------------------------------------

def assert_text_not_empty(text: str) -> None:
	"""Input text must be non-empty and contain printable characters."""
	if not text or not text.strip():
		raise RuleViolation(
			"text_not_empty",
			"input text must not be empty or blank",
			"provide_non_empty_text",
		)


def assert_text_length(text: str, max_chars: int = 10_000_000) -> None:
	"""Input text must not exceed max_chars."""
	if len(text) > max_chars:
		raise RuleViolation(
			"text_too_long",
			f"text length {len(text)} exceeds maximum {max_chars}",
			"truncate_input_text",
		)


def assert_target_language_not_auto(target_lang: Any) -> None:
	"""Translation target language must be a specific code, not 'auto'."""
	val = target_lang.value if hasattr(target_lang, "value") else str(target_lang)
	if val == "auto":
		raise RuleViolation(
			"target_language_not_auto",
			"target_language must be a specific language code, not 'auto'",
			"specify_target_language",
		)


def assert_embedding_dimensions(vector: list[float]) -> None:
	"""Embedding vector must be non-empty."""
	if not vector:
		raise RuleViolation(
			"empty_embedding_vector",
			"embedding vector must contain at least one dimension",
			"check_embedding_model",
		)


def assert_intents_not_empty(intents: list[str]) -> None:
	"""Intent list must contain at least one label."""
	if not intents:
		raise RuleViolation(
			"intents_not_empty",
			"intent list must not be empty",
			"provide_intent_labels",
		)


def assert_max_words_positive(max_words: int) -> None:
	"""Summary max_words must be >= 1."""
	if max_words < 1:
		raise RuleViolation(
			"max_words_positive",
			f"max_words must be >= 1, got {max_words}",
			"increase_max_words",
		)


# ---------------------------------------------------------------------------
# Lifecycle rules
# ---------------------------------------------------------------------------

def assert_document_not_deleted(is_deleted: bool, document_id: str) -> None:
	"""Operations on soft-deleted documents are not allowed."""
	if is_deleted:
		raise RuleViolation(
			"document_not_deleted",
			f"document {document_id} has been deleted",
			"restore_or_recreate_document",
		)


def assert_batch_not_running(status: str) -> None:
	"""Cannot restart a batch job that is currently processing."""
	if status == "processing":
		raise RuleViolation(
			"batch_not_running",
			"cannot restart a batch job that is currently processing",
			"wait_for_completion",
		)
