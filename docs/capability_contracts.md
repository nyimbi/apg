# Executable Capability Contracts

APG capabilities expose a common executable contract so composition, governance,
UI generation, and theming can reason about capabilities consistently.

Every contract provides five required surfaces:

- `configuration`: tenant-specific runtime configuration defaults.
- `configuration_schema`: the shape accepted by configuration overrides.
- `rule_engine`: deterministic governance rules and effects.
- `ui`: routes, permissions, view shell, and theme requirement metadata.
- `theme`: visual tokens and component-level styling metadata.

## Registry API

Use the registry when application code needs to discover, validate, or evaluate
capability contracts.

```python
from capabilities import (
    evaluate_capability_contract_rules,
    get_capability_contract,
    load_contract_registry,
)

registry = load_contract_registry()
contract = get_capability_contract("composition_events")
decision = evaluate_capability_contract_rules(
    "composition_events",
    {
        "tenant_context_present": False,
        "operation_type": "write",
        "policy_attached": False,
    },
)
```

The registry discovers every `capability_contract.py` under `capabilities/`,
validates the required shape, and indexes contracts by capability id.

## CLI Validation

Use the CLI for a lightweight developer or CI gate:

```bash
python cli.py capabilities validate-contracts
```

Expected success output:

```text
✓ Validated 101 capability contracts
```

List contracts for inspection:

```bash
python cli.py capabilities contracts
python cli.py capabilities contracts --json
```

## Adding a Contract

For a spec-backed capability, add a thin wrapper beside `cap_spec.md`:

```python
from pathlib import Path
from typing import Any

from capabilities.capability_contract_factory import (
    build_spec_capability_contract,
    evaluate_contract_rules,
)

CAPABILITY_PATH = Path(__file__).resolve().parent


def get_capability_contract(
    tenant_id: str = "default",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_spec_capability_contract(CAPABILITY_PATH, tenant_id, overrides)


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
    return evaluate_contract_rules(
        get_capability_contract()["rule_engine"]["rules"],
        context,
    )
```

For a bespoke capability, implement the same function names directly. The
registry requires `get_capability_contract`; `evaluate_capability_rules` is
optional because the registry can evaluate standard deterministic rules itself.

## Focused Tests

Run these checks after changing contracts:

```bash
python -m pytest -q capabilities/test_capability_contract_registry.py \
  capabilities/test_spec_capability_contracts.py \
  capabilities/common/test_capability_contracts.py \
  tests/test_cli_capability_contracts.py
```
