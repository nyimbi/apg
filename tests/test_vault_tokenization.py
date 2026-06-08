"""Tests for PCI DSS cardholder data tokenization service."""
import pytest
from capabilities.common.vault import TokenizationService, TokenRecord


VISA = "4111111111111111"
MASTERCARD = "5500005555555559"
AMEX = "378282246310005"


def make_svc(vault_key: str = "test-key") -> TokenizationService:
	return TokenizationService(tenant_id="pci-tenant", vault_key=vault_key)


# ── tokenize_pan ──────────────────────────────────────────────────────────────

async def test_tokenize_returns_record():
	svc = make_svc()
	record = await svc.tokenize_pan(VISA)
	assert isinstance(record, TokenRecord)
	assert record.token is not None
	assert record.last_four == "1111"
	assert record.bin == "411111"
	assert record.card_type == "visa"
	assert record.tenant_id == "pci-tenant"
	assert record.masked_pan.startswith("411111")
	assert record.masked_pan.endswith("1111")


async def test_token_has_same_length_as_pan():
	svc = make_svc()
	record = await svc.tokenize_pan(VISA)
	assert len(record.token) == len(VISA)


async def test_token_preserves_bin():
	svc = make_svc()
	record = await svc.tokenize_pan(VISA)
	assert record.token[:6] == VISA[:6]


async def test_token_preserves_last_three_display_digits():
	"""Last 3 of last-4 are preserved; the 4th (Luhn check digit) is recomputed."""
	svc = make_svc()
	record = await svc.tokenize_pan(VISA)
	# positions [-4:-1] = last 3 display digits, position [-1] = new Luhn check digit
	assert record.token[-4:-1] == VISA[-4:-1]
	assert record.last_four == VISA[-4:]  # TokenRecord.last_four is from original PAN


async def test_token_passes_luhn_check():
	svc = make_svc()
	record = await svc.tokenize_pan(VISA)
	assert svc.luhn_valid(record.token) is True


async def test_token_differs_from_pan():
	svc = make_svc()
	record = await svc.tokenize_pan(VISA)
	assert record.token != VISA


async def test_tokenize_twice_produces_different_tokens():
	"""Each tokenization produces a unique token."""
	svc = make_svc()
	r1 = await svc.tokenize_pan(VISA)
	r2 = await svc.tokenize_pan(VISA)
	assert r1.token != r2.token


async def test_card_type_detection():
	svc = make_svc()
	assert (await svc.tokenize_pan(VISA)).card_type == "visa"
	assert (await svc.tokenize_pan(MASTERCARD)).card_type == "mastercard"
	assert (await svc.tokenize_pan(AMEX)).card_type == "amex"


async def test_invalid_pan_raises_value_error():
	svc = make_svc()
	with pytest.raises(ValueError, match="Invalid PAN"):
		await svc.tokenize_pan("not-a-card")

	with pytest.raises(ValueError, match="Invalid PAN"):
		await svc.tokenize_pan("123")  # too short


# ── detokenize_pan ────────────────────────────────────────────────────────────

async def test_detokenize_returns_original_pan():
	svc = make_svc()
	record = await svc.tokenize_pan(VISA)
	pan = await svc.detokenize_pan(record.token, requester_role="pci_authorized")
	assert pan == VISA


async def test_detokenize_unauthorized_role_raises():
	svc = make_svc()
	record = await svc.tokenize_pan(VISA)
	with pytest.raises(PermissionError, match="not authorized"):
		await svc.detokenize_pan(record.token, requester_role="sales")


async def test_detokenize_unknown_token_raises():
	svc = make_svc()
	with pytest.raises(KeyError):
		await svc.detokenize_pan("4111111111112222", requester_role="pci_authorized")


async def test_tokenize_detokenize_roundtrip_mastercard():
	svc = make_svc()
	record = await svc.tokenize_pan(MASTERCARD)
	pan = await svc.detokenize_pan(record.token, requester_role="payment_processor")
	assert pan == MASTERCARD


# ── Luhn validation ───────────────────────────────────────────────────────────

def test_luhn_valid_known_cards():
	svc = make_svc()
	assert svc.luhn_valid(VISA) is True
	assert svc.luhn_valid(MASTERCARD) is True
	assert svc.luhn_valid(AMEX) is True


def test_luhn_invalid_modified_card():
	svc = make_svc()
	modified = VISA[:-1] + str((int(VISA[-1]) + 1) % 10)
	assert svc.luhn_valid(modified) is False


# ── PAN format with spaces/dashes ────────────────────────────────────────────

async def test_tokenize_handles_formatted_pan():
	"""PAN with spaces or dashes is normalized before tokenization."""
	svc = make_svc()
	record = await svc.tokenize_pan("4111 1111 1111 1111")
	assert len(record.token) == 16
	pan = await svc.detokenize_pan(record.token, requester_role="pci_authorized")
	assert pan == VISA
