# Identity Federation

**Capability ID**: `idfd` | **Domain**: `common` | **Version**: `1.0.0`

## Description

IDFD is APG's generated-application capability for tenant-scoped identity federation. It gives composed applications a deterministic, dependency-light surface for SAML, OIDC, LDAP, SCIM, claim mapping, federated sessions,

## Installation

```bash
pip install apg-common-idfd
```

## Provides

- `identity_federation`
- `federated_sso`
- `federation_agent_composition`

## Requires

- `auth`
- `mfau`
- `encr`
- `audl`
- `secu`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/idfd/dashboard` | `idfd:view` | Overview |
| `/idfd/providers` | `idfd:manage_providers` | Providers |
| `/idfd/protocols` | `idfd:manage_providers` | Providers |
| `/idfd/mappings` | `idfd:manage_mappings` | Mappings |
| `/idfd/sessions` | `idfd:view` | Operations |
| `/idfd/certificates` | `idfd:rotate_keys` | Security |
| `/idfd/scim` | `idfd:manage_providers` | Directory |
| `/idfd/risk` | `idfd:view` | Operations |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_provider()`
- `refresh_provider_metadata()`
- `add_claim_mapping()`
- `issue_session()`
- `revoke_session()`
- `register_certificate()`
- `health_report()`
- `register_federation_agent()`

_(See `service.py` for complete API.)_

## Interoperability

`idfd` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use idfd;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `IDFD_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
