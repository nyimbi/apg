import pytest

from capabilities.composition.gateway.ai_engine import NaturalLanguagePolicyModel
from capabilities.composition.gateway.service import ASMService


@pytest.mark.asyncio
async def test_natural_language_policy_model_uses_keyword_fallback_offline() -> None:
	model = NaturalLanguagePolicyModel()

	intent = await model.classify_intent("route traffic for service orders to billing on path /api/orders")
	rules = await model.generate_policy_rules(intent, intent["parameters"])

	assert intent["intent"] == "route"
	assert intent["primary_intent"] == "route"
	assert intent["parameters"]["service"] == "orders"
	assert intent["parameters"]["target_service"] == "billing"
	assert rules[0]["type"] == "routing"
	assert rules[0]["destination"]["service"] == "billing"


@pytest.mark.asyncio
async def test_asm_service_ai_helpers_compile_fallback_rules() -> None:
	service = object.__new__(ASMService)

	processed = await service._process_natural_language_intent(
		"scale service checkout when cpu load is high"
	)
	compiled = await service._compile_intent_to_rules(processed, "canary")

	assert processed["intent_type"] == "scaling"
	assert processed["extracted_entities"]["service"] == "checkout"
	assert compiled["deployment_strategy"] == "canary"
	assert compiled["route_rules"][0]["type"] == "scaling"
	assert compiled["affected_services"] == ["checkout"]
