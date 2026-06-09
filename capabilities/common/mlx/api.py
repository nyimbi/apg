"""MLX capability — REST API endpoints."""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from .service import MLCapability
from .views import (
	ScoreRequest, ClassifyRequest, PredictRequest,
	SummarizeRequest, ExtractRequest, EmbedRequest, RankRequest,
)

_log = logging.getLogger(__name__)

mlx_api = Blueprint("mlx_api", __name__, url_prefix="/api/mlx")


def _svc() -> MLCapability:
	return MLCapability()


@mlx_api.get("/health")
async def health():
	svc = _svc()
	result = await svc.health_check()
	return jsonify(result)


@mlx_api.get("/models")
async def list_models():
	svc = _svc()
	models = await svc.list_models()
	return jsonify({"models": models, "total": len(models)})


@mlx_api.post("/models/<model_name>/pull")
async def pull_model(model_name: str):
	svc = _svc()
	result = await svc.pull_model(model_name)
	return jsonify(result)


@mlx_api.post("/score")
async def score():
	body = ScoreRequest.model_validate(request.get_json(force=True))
	svc = _svc()
	try:
		result = await svc.score(body.features, model=body.model, task_description=body.task_description)
		return jsonify(result.model_dump())
	except Exception as exc:
		_log.exception("score failed")
		return jsonify({"error": str(exc)}), 500


@mlx_api.post("/classify")
async def classify():
	body = ClassifyRequest.model_validate(request.get_json(force=True))
	svc = _svc()
	try:
		result = await svc.classify(body.text, body.labels, model=body.model)
		return jsonify(result.model_dump())
	except Exception as exc:
		_log.exception("classify failed")
		return jsonify({"error": str(exc)}), 500


@mlx_api.post("/predict")
async def predict():
	body = PredictRequest.model_validate(request.get_json(force=True))
	svc = _svc()
	try:
		result = await svc.predict(body.series, horizon=body.horizon, model=body.model)
		return jsonify(result.model_dump())
	except Exception as exc:
		_log.exception("predict failed")
		return jsonify({"error": str(exc)}), 500


@mlx_api.post("/summarize")
async def summarize():
	body = SummarizeRequest.model_validate(request.get_json(force=True))
	svc = _svc()
	try:
		result = await svc.summarize(body.text, max_words=body.max_words, model=body.model)
		return jsonify(result.model_dump())
	except Exception as exc:
		_log.exception("summarize failed")
		return jsonify({"error": str(exc)}), 500


@mlx_api.post("/extract")
async def extract():
	body = ExtractRequest.model_validate(request.get_json(force=True))
	svc = _svc()
	try:
		result = await svc.extract(body.document, schema=body.schema, model=body.model)
		return jsonify(result.model_dump())
	except Exception as exc:
		_log.exception("extract failed")
		return jsonify({"error": str(exc)}), 500


@mlx_api.post("/embed")
async def embed():
	body = EmbedRequest.model_validate(request.get_json(force=True))
	svc = _svc()
	try:
		result = await svc.embed(body.text, model=body.model)
		return jsonify(result)
	except Exception as exc:
		return jsonify({"error": str(exc)}), 500


@mlx_api.post("/rank")
async def rank():
	body = RankRequest.model_validate(request.get_json(force=True))
	svc = _svc()
	try:
		result = await svc.rank_documents(body.query, body.documents, model=body.model)
		return jsonify(result)
	except Exception as exc:
		return jsonify({"error": str(exc)}), 500


@mlx_api.get("/stats")
async def stats():
	svc = _svc()
	return jsonify(await svc.get_inference_stats())
