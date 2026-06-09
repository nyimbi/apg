# Mobile App Platform

**Capability ID**: `mob_map` | **Domain**: `mob` | **Version**: `1.0.0`

## Description

The Mobile App Platform (MAP) capability provides a complete cross-platform mobile application lifecycle management runtime. It covers app registration across iOS, Android, PWA and desktop targets; offline data sync with configurable conflict resolution; push notification dispatch via APNS/FCM/Web; biometric authentication enrollment and revocation; app version publishing with phased rollouts and rollbacks; granular permission scope governance; and an analytics event pipeline — all governed by tenant-scoped deterministic policy rules.

## Installation

```bash
pip install apg-mob-map
```

## Provides

- `mobile_app_registry`
- `cross_platform_build_workflow`
- `offline_sync_workflow`
- `push_notification_dispatch`
- `biometric_auth_enrollment`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/mob-map/dashboard` | `mob_map:view` | Overview |
| `/mob-map/apps` | `mob_map:apps:list` | Applications |
| `/mob-map/apps/<app_id>` | `mob_map:apps:view` | Applications |
| `/mob-map/versions` | `mob_map:versions:list` | Releases |
| `/mob-map/versions/<version_id>/deploy` | `mob_map:versions:deploy` | Releases |
| `/mob-map/sync` | `mob_map:sync:list` | Sync |
| `/mob-map/sync/conflicts` | `mob_map:sync:resolve` | Sync |
| `/mob-map/notifications` | `mob_map:notifications:list` | Notifications |

## Key Service Methods

- `uuid7str()`
- `describe()`
- `evaluate()`
- `register_app()`
- `get_app()`
- `list_apps()`
- `update_app()`
- `retire_app()`
- `publish_version()`
- `deploy_version()`

_(See `service.py` for complete API.)_

## Interoperability

`mob_map` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use mob_map;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `MOB_MAP_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
