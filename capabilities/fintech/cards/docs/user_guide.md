# Digital Cards

**Capability ID**: `fintech_cards` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Digital Cards provides executable card issuing and operations workflows: program governance, cardholder onboarding, virtual and physical card issuance, token provisioning (wallet, device, merchant, network tokens), authorization decisions with fraud and AML controls, and dispute intake. It is the issuing layer that sits between a payment wallet and the card network, enforcing per-authorization fraud scoring and AML result checks before any card transaction is approved.

## Installation

```bash
pip install apg-fintech-cards
```

## Provides

- `card_program_governance`
- `cardholder_card_lifecycle`
- `tokenized_card_credentialing`
- `card_authorization_control`
- `card_dispute_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-cards/dashboard` | `fintech_cards:view` | Overview |
| `/fintech-cards/programs` | `fintech_cards:manage_programs` | Programs |
| `/fintech-cards/cardholders` | `fintech_cards:manage_cardholders` | Cards |
| `/fintech-cards/cards` | `fintech_cards:issue` | Cards |
| `/fintech-cards/tokens` | `fintech_cards:tokenize` | Tokens |
| `/fintech-cards/authorizations` | `fintech_cards:authorize` | Controls |
| `/fintech-cards/disputes` | `fintech_cards:dispute` | Exceptions |
| `/fintech-cards/agents` | `fintech_cards:admin` | Automation |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_program()`
- `onboard_cardholder()`
- `issue_card()`
- `provision_token()`
- `authorize_transaction()`
- `file_dispute()`
- `register_card_agent()`
- `validate_batch()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_cards` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_cards;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_CARDS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
