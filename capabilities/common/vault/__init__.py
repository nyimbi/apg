"""APG Vault — PCI DSS tokenization and secrets management.

Provides cardholder data tokenization (PCI DSS requirement 3.5) for
fintech capabilities. Tokens replace Primary Account Numbers (PANs) in
application storage so cardholder data never appears in logs or databases.

The tokenization engine uses format-preserving tokenization (FPT):
  - Tokens preserve PAN length and the first 6 (BIN) + last 4 digits
  - Middle 6 digits are replaced with a random token suffix
  - Tokens pass Luhn check (prevents trivial detection)

Usage::

    from capabilities.common.vault import TokenizationService
    svc = TokenizationService(tenant_id="fintech-co")

    tok = await svc.tokenize_pan("4111111111111111")
    print(tok.token)    # e.g. "4111111111119999"  (same BIN, random middle)

    pan = await svc.detokenize_pan(tok.token, requester_role="pci_authorized")
    print(pan)          # "4111111111111111"
"""
from .service import TokenizationService, TokenRecord

__all__ = ["TokenizationService", "TokenRecord"]
