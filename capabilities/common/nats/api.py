"""NATS JetStream capability — REST API endpoints."""
from __future__ import annotations

import logging
import os
from typing import Any

from flask import Blueprint, jsonify, request

from .service import NATSService
from .models import CreateStreamRequest, CreateConsumerRequest, PublishRequest, PublishBatchRequest

_log = logging.getLogger(__name__)

nats_api = Blueprint("nats_api", __name__, url_prefix="/api/nats")

_NATS_URL = os.environ.get("NATS_URL", "nats://localhost:4222")


def _svc(tenant_id: str) -> NATSService:
	return NATSService(nats_url=_NATS_URL, tenant_id=tenant_id)


def _tenant() -> str:
	return request.headers.get("X-Tenant-Id", "default")


# ── Health ────────────────────────────────────────────────────────────────────

@nats_api.get("/health")
async def health():
	svc = _svc(_tenant())
	try:
		await svc.connect()
		result = await svc.health_check()
		await svc.disconnect()
	except Exception as exc:
		result = {"status": "error", "error": str(exc)}
	return jsonify(result)


# ── Publish ───────────────────────────────────────────────────────────────────

@nats_api.post("/publish")
async def publish():
	body = PublishRequest.model_validate(request.get_json(force=True))
	svc = _svc(_tenant())
	try:
		await svc.connect()
		result = await svc.publish(
			body.capability_id,
			body.event_type,
			body.payload,
			actor_id=body.actor_id,
		)
		await svc.disconnect()
	except Exception as exc:
		_log.exception("publish failed")
		return jsonify({"error": str(exc)}), 500
	return jsonify(result), 201


@nats_api.post("/publish/batch")
async def publish_batch():
	body = PublishBatchRequest.model_validate(request.get_json(force=True))
	svc = _svc(_tenant())
	try:
		await svc.connect()
		result = await svc.publish_batch(body.capability_id, body.events)
		await svc.disconnect()
	except Exception as exc:
		_log.exception("publish_batch failed")
		return jsonify({"error": str(exc)}), 500
	return jsonify(result), 201


# ── Streams ───────────────────────────────────────────────────────────────────

@nats_api.get("/streams")
async def list_streams():
	svc = _svc(_tenant())
	try:
		await svc.connect()
		streams = await svc.list_streams()
		await svc.disconnect()
	except Exception as exc:
		return jsonify({"error": str(exc)}), 500
	return jsonify({"streams": streams, "total": len(streams)})


@nats_api.post("/streams")
async def create_stream():
	body = CreateStreamRequest.model_validate(request.get_json(force=True))
	svc = _svc(_tenant())
	try:
		await svc.connect()
		result = await svc.create_stream(
			body.stream_name,
			body.subjects,
			max_age_seconds=body.max_age_seconds,
			max_bytes=body.max_bytes,
			replicas=body.replicas,
		)
		await svc.disconnect()
	except Exception as exc:
		return jsonify({"error": str(exc)}), 500
	return jsonify(result), 201


@nats_api.get("/streams/<stream_name>")
async def get_stream(stream_name: str):
	svc = _svc(_tenant())
	try:
		await svc.connect()
		info = await svc.get_stream(stream_name)
		await svc.disconnect()
	except Exception as exc:
		return jsonify({"error": str(exc)}), 404
	return jsonify(info)


@nats_api.delete("/streams/<stream_name>")
async def delete_stream(stream_name: str):
	svc = _svc(_tenant())
	try:
		await svc.connect()
		result = await svc.delete_stream(stream_name)
		await svc.disconnect()
	except Exception as exc:
		return jsonify({"error": str(exc)}), 500
	return jsonify(result)


@nats_api.post("/streams/<stream_name>/purge")
async def purge_stream(stream_name: str):
	svc = _svc(_tenant())
	try:
		await svc.connect()
		result = await svc.purge_stream(stream_name)
		await svc.disconnect()
	except Exception as exc:
		return jsonify({"error": str(exc)}), 500
	return jsonify(result)


# ── Consumers ─────────────────────────────────────────────────────────────────

@nats_api.get("/streams/<stream_name>/consumers")
async def list_consumers(stream_name: str):
	svc = _svc(_tenant())
	try:
		await svc.connect()
		consumers = await svc.list_consumers(stream_name)
		await svc.disconnect()
	except Exception as exc:
		return jsonify({"error": str(exc)}), 500
	return jsonify({"consumers": consumers, "total": len(consumers)})


@nats_api.post("/streams/<stream_name>/consumers")
async def create_consumer(stream_name: str):
	body = CreateConsumerRequest.model_validate(request.get_json(force=True))
	svc = _svc(_tenant())
	try:
		await svc.connect()
		result = await svc.create_consumer(
			stream_name,
			body.consumer_name,
			filter_subject=body.filter_subject,
		)
		await svc.disconnect()
	except Exception as exc:
		return jsonify({"error": str(exc)}), 500
	return jsonify(result), 201


@nats_api.delete("/streams/<stream_name>/consumers/<consumer_name>")
async def delete_consumer(stream_name: str, consumer_name: str):
	svc = _svc(_tenant())
	try:
		await svc.connect()
		result = await svc.delete_consumer(stream_name, consumer_name)
		await svc.disconnect()
	except Exception as exc:
		return jsonify({"error": str(exc)}), 500
	return jsonify(result)


# ── Subject utilities ─────────────────────────────────────────────────────────

@nats_api.get("/subjects")
async def list_subjects():
	svc = _svc(_tenant())
	try:
		await svc.connect()
		subjects = await svc.list_subjects()
		await svc.disconnect()
	except Exception as exc:
		return jsonify({"error": str(exc)}), 500
	return jsonify({"subjects": subjects})


@nats_api.get("/subjects/resolve")
def resolve_subject():
	cap = request.args.get("capability_id", "")
	evt = request.args.get("event_type", ">")
	from .subject_registry import subject_for
	return jsonify({"subject": subject_for(cap, evt)})


# ── Connection info ───────────────────────────────────────────────────────────

@nats_api.get("/connection")
def connection_info():
	return jsonify({
		"nats_url": _NATS_URL,
		"tenant_id": _tenant(),
		"configured": bool(os.environ.get("NATS_URL")),
	})


# ── APG stream setup ──────────────────────────────────────────────────────────

@nats_api.post("/setup")
async def setup_apg_streams():
	svc = _svc(_tenant())
	try:
		await svc.connect()
		result = await svc.setup_apg_streams()
		await svc.disconnect()
	except Exception as exc:
		return jsonify({"error": str(exc)}), 500
	return jsonify(result)
