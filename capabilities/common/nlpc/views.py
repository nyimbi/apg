"""
NLPC UI Views — Natural Language Processing Core

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Website: www.datacraft.co.ke

Plain Flask Blueprint for browser-facing views.  No flask_appbuilder.
Templates live in templates/nlpc/.

Mount with:
    from capabilities.common.nlpc.views import nlpc_views_bp
    app.register_blueprint(nlpc_views_bp)
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from flask import (
	Blueprint,
	abort,
	flash,
	jsonify,
	redirect,
	render_template,
	request,
	url_for,
)

from .domain.rules import RuleViolation
from .models import (
	ClassificationTaxonomy,
	EntityType,
	LanguageCode,
	NLPDocumentCreate,
	NLPTask,
	SummaryMethod,
)
from .service import NLPCoreService


nlpc_views_bp = Blueprint(
	"nlpc_views",
	__name__,
	url_prefix="/nlpc",
	template_folder="templates",
	static_folder="static",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _svc() -> NLPCoreService:
	tenant_id = request.cookies.get("tenant_id") or request.args.get("tenant_id") or "default"
	actor_id = request.cookies.get("actor_id") or "web"
	return NLPCoreService(tenant_id=tenant_id, actor_id=actor_id)


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@nlpc_views_bp.route("/", methods=["GET"])
@nlpc_views_bp.route("/dashboard", methods=["GET"])
def dashboard() -> str:
	"""
	NLP Core dashboard.

	KPIs: total documents, tasks run, avg sentiment, top languages.
	"""
	svc = _svc()
	docs = _run(svc.list_documents(limit=1000))
	now = datetime.utcnow()
	report = _run(svc.usage_report(
		period_start=datetime(now.year, now.month, 1),
		period_end=now,
	))
	kpis = {
		"total_documents": report.total_documents,
		"total_requests": report.total_requests,
		"top_languages": sorted(
			report.language_breakdown.items(), key=lambda x: x[1], reverse=True
		)[:5],
		"task_breakdown": report.task_breakdown,
		"error_rate": report.error_rate,
		"cache_hit_rate": report.cache_hit_rate,
	}
	return render_template("nlpc/dashboard.html", kpis=kpis, docs=docs[:10])


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@nlpc_views_bp.route("/documents", methods=["GET"])
def list_documents() -> str:
	"""List all documents for current tenant."""
	svc = _svc()
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	language = request.args.get("language")
	docs = _run(svc.list_documents(limit=limit, offset=offset, language=language))
	return render_template(
		"nlpc/document_list.html",
		docs=docs,
		limit=limit,
		offset=offset,
		language=language,
		languages=[lc.value for lc in LanguageCode],
	)


@nlpc_views_bp.route("/documents/new", methods=["GET", "POST"])
def create_document() -> Any:
	"""Create a new document."""
	svc = _svc()
	if request.method == "POST":
		content = request.form.get("content", "").strip()
		title = request.form.get("title", "").strip() or None
		source = request.form.get("source", "").strip() or None
		language = request.form.get("language") or None
		is_sensitive = bool(request.form.get("is_sensitive"))
		if not content:
			flash("Content is required.", "danger")
			return render_template("nlpc/document_form.html", languages=[lc.value for lc in LanguageCode])
		tenant_id = request.cookies.get("tenant_id") or "default"
		try:
			payload = NLPDocumentCreate(
				tenant_id=tenant_id,
				content=content,
				title=title,
				source=source,
				language=language,
				is_sensitive=is_sensitive,
			)
			doc = _run(svc.create_document(payload))
			flash(f"Document {doc.id[:8]}… created.", "success")
			return redirect(url_for("nlpc_views.detail_document", document_id=doc.id))
		except (RuleViolation, ValueError) as exc:
			flash(str(exc), "danger")
	return render_template(
		"nlpc/document_form.html",
		languages=[lc.value for lc in LanguageCode],
	)


@nlpc_views_bp.route("/documents/<document_id>", methods=["GET"])
def detail_document(document_id: str) -> Any:
	"""Document detail view with all NLP results."""
	svc = _svc()
	doc = _run(svc.get_document(document_id))
	if doc is None:
		abort(404)
	return render_template("nlpc/document_detail.html", doc=doc)


@nlpc_views_bp.route("/documents/<document_id>/delete", methods=["POST"])
def delete_document(document_id: str) -> Any:
	"""Soft-delete a document."""
	svc = _svc()
	ok = _run(svc.delete_document(document_id))
	if ok:
		flash("Document deleted.", "success")
	else:
		flash("Document not found.", "warning")
	return redirect(url_for("nlpc_views.list_documents"))


# ---------------------------------------------------------------------------
# Analysis views (form → result on same page)
# ---------------------------------------------------------------------------

@nlpc_views_bp.route("/analyse/language", methods=["GET", "POST"])
def analyse_language() -> Any:
	"""Detect language for arbitrary text."""
	svc = _svc()
	result = None
	text = ""
	if request.method == "POST":
		text = request.form.get("text", "").strip()
		if text:
			try:
				result = _run(svc.detect_language(text))
			except RuleViolation as exc:
				flash(str(exc), "danger")
		else:
			flash("Please enter some text.", "warning")
	return render_template("nlpc/analyse_language.html", result=result, text=text)


@nlpc_views_bp.route("/analyse/entities", methods=["GET", "POST"])
def analyse_entities() -> Any:
	"""Extract named entities from arbitrary text."""
	svc = _svc()
	results = []
	text = ""
	selected_types: list[str] = []
	if request.method == "POST":
		text = request.form.get("text", "").strip()
		selected_types = request.form.getlist("entity_types")
		entity_types: list[EntityType] | None = None
		if selected_types:
			try:
				entity_types = [EntityType(t) for t in selected_types]
			except ValueError:
				flash("Invalid entity type selected.", "danger")
		if text:
			try:
				results = _run(svc.extract_entities(text, entity_types=entity_types))
			except RuleViolation as exc:
				flash(str(exc), "danger")
		else:
			flash("Please enter some text.", "warning")
	return render_template(
		"nlpc/analyse_entities.html",
		results=results,
		text=text,
		entity_type_choices=[et.value for et in EntityType],
		selected_types=selected_types,
	)


@nlpc_views_bp.route("/analyse/sentiment", methods=["GET", "POST"])
def analyse_sentiment() -> Any:
	"""Sentiment analysis for arbitrary text."""
	svc = _svc()
	result = None
	text = ""
	if request.method == "POST":
		text = request.form.get("text", "").strip()
		if text:
			try:
				result = _run(svc.sentiment_analysis(text))
			except RuleViolation as exc:
				flash(str(exc), "danger")
		else:
			flash("Please enter some text.", "warning")
	return render_template("nlpc/analyse_sentiment.html", result=result, text=text)


@nlpc_views_bp.route("/analyse/intent", methods=["GET", "POST"])
def analyse_intent() -> Any:
	"""Intent classification for arbitrary text."""
	svc = _svc()
	result = None
	text = ""
	intents_raw = ""
	if request.method == "POST":
		text = request.form.get("text", "").strip()
		intents_raw = request.form.get("intents", "").strip()
		intents = [i.strip() for i in intents_raw.split(",") if i.strip()]
		if text and intents:
			try:
				result = _run(svc.intent_classification(text, intents=intents))
			except RuleViolation as exc:
				flash(str(exc), "danger")
		else:
			flash("Text and at least one intent are required.", "warning")
	return render_template(
		"nlpc/analyse_intent.html",
		result=result,
		text=text,
		intents_raw=intents_raw,
	)


@nlpc_views_bp.route("/analyse/summarise", methods=["GET", "POST"])
def analyse_summarise() -> Any:
	"""Text summarisation."""
	svc = _svc()
	result = None
	text = ""
	max_words = 100
	method = "extractive"
	if request.method == "POST":
		text = request.form.get("text", "").strip()
		max_words = int(request.form.get("max_words", 100))
		method = request.form.get("method", "extractive")
		if text:
			try:
				result = _run(
					svc.text_summarisation(
						text,
						max_words=max_words,
						method=SummaryMethod(method),
					)
				)
			except (RuleViolation, ValueError) as exc:
				flash(str(exc), "danger")
		else:
			flash("Please enter some text.", "warning")
	return render_template(
		"nlpc/analyse_summarise.html",
		result=result,
		text=text,
		max_words=max_words,
		method=method,
		methods=[m.value for m in SummaryMethod],
	)


@nlpc_views_bp.route("/analyse/translate", methods=["GET", "POST"])
def analyse_translate() -> Any:
	"""Text translation."""
	svc = _svc()
	result = None
	text = ""
	target_lang = "sw"
	if request.method == "POST":
		text = request.form.get("text", "").strip()
		target_lang = request.form.get("target_lang", "sw")
		if text:
			try:
				result = _run(
					svc.translate(text, target_lang=LanguageCode(target_lang))
				)
			except (RuleViolation, ValueError) as exc:
				flash(str(exc), "danger")
		else:
			flash("Please enter some text.", "warning")
	return render_template(
		"nlpc/analyse_translate.html",
		result=result,
		text=text,
		target_lang=target_lang,
		languages=[lc.value for lc in LanguageCode if lc != LanguageCode.AUTO],
	)


@nlpc_views_bp.route("/analyse/classify", methods=["GET", "POST"])
def analyse_classify() -> Any:
	"""Document classification."""
	svc = _svc()
	result = None
	text = ""
	taxonomy = "topics"
	labels_raw = ""
	if request.method == "POST":
		text = request.form.get("text", "").strip()
		taxonomy = request.form.get("taxonomy", "topics")
		labels_raw = request.form.get("labels", "").strip()
		labels: list[str] | None = [l.strip() for l in labels_raw.split(",") if l.strip()] or None
		if text:
			try:
				result = _run(
					svc.classify_document(text, taxonomy=ClassificationTaxonomy(taxonomy), labels=labels)
				)
			except (RuleViolation, ValueError) as exc:
				flash(str(exc), "danger")
		else:
			flash("Please enter some text.", "warning")
	return render_template(
		"nlpc/analyse_classify.html",
		result=result,
		text=text,
		taxonomy=taxonomy,
		labels_raw=labels_raw,
		taxonomies=[t.value for t in ClassificationTaxonomy],
	)


@nlpc_views_bp.route("/analyse/african-id", methods=["GET", "POST"])
def analyse_african_id() -> Any:
	"""African language identification."""
	svc = _svc()
	result = None
	text = ""
	if request.method == "POST":
		text = request.form.get("text", "").strip()
		if text:
			try:
				result = _run(svc.language_id_for_african_languages(text))
			except RuleViolation as exc:
				flash(str(exc), "danger")
		else:
			flash("Please enter some text.", "warning")
	return render_template("nlpc/analyse_african_id.html", result=result, text=text)


# ---------------------------------------------------------------------------
# Batch jobs
# ---------------------------------------------------------------------------

@nlpc_views_bp.route("/batch", methods=["GET"])
def list_batch_jobs() -> str:
	"""List batch jobs for current tenant (stub — reads from in-process store)."""
	from .service import _STORE
	tenant_id = request.cookies.get("tenant_id") or "default"
	jobs = [
		v for v in _STORE.get("nlpc_batch_jobs", {}).values()
		if v.get("tenant_id") == tenant_id and not v.get("is_deleted")
	]
	jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
	return render_template("nlpc/batch_list.html", jobs=jobs)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@nlpc_views_bp.route("/reports/usage", methods=["GET"])
def report_usage() -> str:
	"""Usage report view for current month."""
	svc = _svc()
	now = datetime.utcnow()
	report = _run(svc.usage_report(
		period_start=datetime(now.year, now.month, 1),
		period_end=now,
	))
	return render_template("nlpc/report_usage.html", report=report)
