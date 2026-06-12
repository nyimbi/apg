# Mobile Maps (mob_map)

## Overview
The Mobile Maps (mob_map) capability provides a complete cross-platform mobile application lifecycle management runtime with offline maps, turn-by-turn navigation, and POI management. It covers app registration across iOS, Android, PWA and desktop targets; offline data sync with configurable conflict resolution; push notification dispatch via APNS/FCM/Web; biometric authentication enrollment and revocation; app version publishing with phased rollouts and rollbacks; granular permission scope governance; and an analytics event pipeline — all governed by tenant-scoped deterministic policy rules.

## Capability ID
`mob_map`

## Provides
| Service | Description |
|---------|-------------|
| mobile_app_registry | Register and lifecycle-manage apps across platforms |
| cross_platform_build_workflow | Track builds per platform/channel/environment |
| offline_sync_workflow | Manage offline sync sessions with encryption and conflict policies |
| push_notification_dispatch | Send push notifications via APNS, FCM, and Web Push |
| biometric_auth_enrollment | Enroll and revoke fingerprint/face/passkey auth per device |
| app_version_management | Publish, deploy, and rollback app versions |
| phased_rollout_workflow | Control rollout channels (alpha/beta/canary/stable) |
| permission_scope_governance | Grant and revoke device/app permission scopes |
| app_analytics_pipeline | Collect and summarise in-app analytics events |
| sync_conflict_resolution | Resolve offline sync conflicts via configurable policies |

## Requires
| Capability | Reason |
|------------|--------|
| auth | User authentication context and token validation |
| audl | Audit trail for all state-changing operations |
| mten | Multi-tenancy enforcement and tenant isolation |
| conf | Runtime configuration management |
| ntfy | Notification dispatch integration |
| moni | Operational monitoring and alerting |
| mqeb | Event streaming via Bytewax for sync and lifecycle events |
| mob_mdm | Device enrolment status required before biometric enrollment |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| apps.deployment_approval_required | true | Require explicit approval before version deployment |
| sync.encryption_required | true | Enforce encrypted sync sessions |
| notifications.rate_limit_per_device_per_hour | 50 | Max push notifications per device per hour |
| auth.mfa_required_for_sensitive | true | Require MFA for sensitive operations |
| versions.rollback_supported | true | Enable version rollback |
| governance.cross_tenant_access_denied | true | Prevent cross-tenant data access |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /api/mob/map/contract | GET | Return capability contract | mob_map:view |
| /api/mob/map/apps | GET | List apps | mob_map:apps:list |
| /api/mob/map/apps | POST | Register app | mob_map:apps:create |
| /api/mob/map/apps/<id> | GET | Get app detail | mob_map:apps:view |
| /api/mob/map/apps/<id> | PUT | Update app | mob_map:apps:edit |
| /api/mob/map/apps/<id> | DELETE | Retire app | mob_map:apps:retire |
| /api/mob/map/versions | GET | List versions | mob_map:versions:list |
| /api/mob/map/versions | POST | Publish version | mob_map:versions:publish |
| /api/mob/map/versions/<id>/deploy | POST | Deploy version | mob_map:versions:deploy |
| /api/mob/map/apps/<id>/rollback | POST | Rollback version | mob_map:versions:deploy |
| /api/mob/map/sync | GET | List sync sessions | mob_map:sync:list |
| /api/mob/map/sync | POST | Start sync session | mob_map:sync:start |
| /api/mob/map/sync/<id>/complete | POST | Complete sync | mob_map:sync:manage |
| /api/mob/map/sync/<id>/resolve | POST | Resolve conflicts | mob_map:sync:resolve |
| /api/mob/map/notifications | GET | List notifications | mob_map:notifications:list |
| /api/mob/map/notifications | POST | Send notification | mob_map:notifications:send |
| /api/mob/map/auth/biometric | GET | List enrollments | mob_map:auth:manage |
| /api/mob/map/auth/biometric | POST | Enroll biometric | mob_map:auth:manage |
| /api/mob/map/auth/biometric/<id> | DELETE | Revoke biometric | mob_map:auth:manage |
| /api/mob/map/permissions | GET | List permissions | mob_map:permissions:manage |
| /api/mob/map/permissions | POST | Grant permission | mob_map:permissions:manage |
| /api/mob/map/permissions/<id> | DELETE | Revoke permission | mob_map:permissions:manage |
| /api/mob/map/analytics/<app_id> | GET | Get analytics summary | mob_map:analytics:view |
| /api/mob/map/analytics | POST | Record analytics event | mob_map:analytics:write |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | tenant_context_present=False | deny — attach tenant context |
| write_requires_policy | operation_type=write, policy_attached=False | deny — attach policy |
| platform_must_be_supported | operation=register_app, platform_supported=False | deny — select supported platform |
| deployment_requires_approval | operation=deploy_version, approval_present=False | deny — obtain approval |
| sync_encryption_mandatory | operation=start_sync, encryption_enabled=False | deny — enable encryption |
| notification_rate_limit | operation=send_notification, rate_limit_exceeded=True | deny — wait rate-limit window |
| biometric_requires_device_enrollment | operation=enroll_biometric, device_enrolled=False | deny — enrol device first |
| retired_app_blocks_deployment | operation=deploy_version, app_state=retired | deny — reinstate app first |
| rollback_requires_previous_version | operation=rollback_version, previous_version_exists=False | deny — check version history |
| cross_tenant_access_denied | cross_tenant_access=True | deny — use tenant-scoped context |

## Data Models
| Model | Key Fields |
|-------|-----------|
| MobileAppResponse | id, tenant_id, name, bundle_id, platform, category, state |
| AppVersionResponse | id, app_id, version_string, channel, update_policy, state, deployed_at |
| SyncSessionResponse | id, app_id, device_id, sync_strategy, state, conflicts_detected, conflicts_resolved |
| PushNotificationResponse | id, app_id, channel, title, target_reference, state, sent_at |
| BiometricEnrollmentResponse | id, device_id, user_id, auth_method, biometric_state |
| PermissionScopeResponse | id, app_id, device_id, scope, state, granted_by |
| AppAnalyticsEventResponse | id, app_id, device_id, event_type, event_payload, session_id |

## Streaming Events
- `app_registered` — new app registered
- `app_state_changed` — app state transition
- `app_version_published` — version published
- `app_version_deployed` — version deployed to environment
- `sync_session_started` — offline sync initiated
- `sync_session_completed` — sync completed
- `sync_conflict_detected` — conflict found during sync
- `sync_conflict_resolved` — conflict resolved
- `push_notification_sent` — notification dispatched
- `biometric_enrolled` / `biometric_revoked` — enrollment lifecycle
- `permission_scope_granted` / `permission_scope_revoked` — scope lifecycle
- `app_analytics_event` — in-app analytics

## Edge Cases Handled
- Retired apps block further version deployments
- Biometric enrollment blocked if device not enrolled in MDM
- Push notification rate limiting enforced per device per hour
- Sync sessions denied if encryption disabled
- Rollbacks create a new version record rather than mutating the original
- Cross-tenant access denied at rule-engine level, not just service layer
- Conflict resolution tracked separately from conflict detection (partial resolution possible)
- MFA required for sensitive operations (biometric changes, permission grants)

## Composability Notes
- Integrates with `mob_mdm` to check device enrollment status before biometric operations
- Emits events to `mqeb` (Bytewax) for downstream analytics and monitoring
- Notification dispatch delegates to `ntfy` for channel-specific delivery
- All state changes audited via `audl`
- Tenant isolation enforced through `mten` — no cross-tenant data leakage
- Analytics events can feed `moni` dashboards for real-time app health monitoring

---

## World-Class Enhancements (v2.0)

1. **Vector Tile Caching with Differential Sync** — Merkle-tree delta sync reduces tile update bandwidth 60–90%; integrates with existing `offline_sync` encryption pipeline.
2. **Turn-by-Turn Navigation Engine** — Async `RouteService` wrapping OSRM/Valhalla with local Contraction Hierarchies fallback; returns GeoJSON geometry, manoeuvre list, and traffic-aware ETA.
3. **Geofence Lifecycle Management** — PostGIS circular/polygon geofences with Bytewax streaming evaluation; emits `geofence_entered`/`geofence_exited` CloudEvents composable with `ntfy`.
4. **POI Semantic Search with Embedding Index** — Ollama `nomic-embed-text` embeddings stored in pgvector; degrades to trigram similarity when index is cold.
5. **Offline Route Pre-computation & Packaging** — `TripPackage` bundles route geometry, tile references, turn instructions, and POI stops into a single zstd/AES-256-GCM archive for fully offline trips.
6. **Real-Time Location Sharing (Presence Layer)** — WebSocket/SSE presence channel with configurable publish interval, time-series TTL storage, and `list_nearby_devices` bounding-box query.
7. **Elevation-Aware Route Profiles** — SRTM/Copernicus DEM integration annotates route segments with grade (%); adds elevation profile charts and `min_elevation_gain`/`scenic_route` optimisation objectives.
8. **Map Style Engine (Tenant-Brandable Cartography)** — Per-tenant MapLibre GL style JSON stored in `conf`; supports colour palette, font, layer, and icon overrides with server-side thumbnail preview.
9. **Fleet Tracking & ETA Broadcasting** — `FleetVehicle` entities ingest location telemetry, compute live ETA to `WaypointStop` sets via routing engine, broadcast via `mqeb` with delay confidence intervals.
10. **Spatial Analytics Aggregations** — PostGIS heatmap density grids (`ST_SquareGrid`), corridor clustering (`ST_ClusterDBSCAN`), and dwell-time polygons returned as GeoJSON `FeatureCollection`.
11. **Map Data Freshness & Auto-Update Scheduling** — CronCreate-registered jobs compare tile manifest hashes; triggers background Wi-Fi/charging downloads for regions exceeding configurable staleness threshold.
12. **Multi-Modal Transit Integration** — GTFS feed ingestion for `TransitRoute`/`TransitStop`/`ServiceCalendar`; extends routing engine with intermodal legs including boarding times, headways, and fare estimates.
13. **Privacy-Preserving Location Anonymisation** — k-anonymity via OSRM map-matching, configurable timestamp quantisation, and home/work zone redaction; non-destructive with encrypted cold-store option.
14. **Indoor Mapping & Venue Navigation** — IMDF venue package ingestion; PostGIS `Venue`/`Floor`/`Unit`/`Opening` entities with indoor routing graphs across elevators and stairs.
15. **Predictive Caching via ML Trip Prediction** — Markov/LSTM trip prediction via Ollama; UCB1 bandit policy pre-fetches top-3 destination tile packs on Wi-Fi connect.

---

## New Methods

### `push_campaign` — Segmented push notification broadcast

Send a targeted notification to all devices matching a segment filter. Returns a list of dispatched `PushNotificationResponse` objects.

```python
svc = MobileAppPlatformService()

results = await svc.push_campaign(
    tenant_id="acme",
    app_id="app-uuid",
    title="New map tiles available",
    body="Download offline pack for Nairobi CBD.",
    channel="fcm",
    segment_filter={"platform": "android", "region": "nairobi"},
    sent_by="ops-bot",
)
print(f"Dispatched to {len(results)} devices")
```

### `offline_data_sync` — Full offline sync session lifecycle

Initiates a sync session, transfers records, resolves any conflicts, and completes — all in one call. Returns the finalised `SyncSessionResponse`.

```python
session = await svc.offline_data_sync(
    tenant_id="acme",
    app_id="app-uuid",
    device_id="device-uuid",
    sync_strategy="incremental",
    conflict_policy="server_wins",
    encryption_key="base64-encoded-key",
    records=[{"id": "r1", "payload": {...}}, ...],
)
print(f"Sync {session.state}: {session.records_synced} records, {session.conflicts_resolved} conflicts resolved")
```

### `get_analytics_summary` — Aggregated in-app event summary

Returns event-type counts, unique device count, and session metrics for an app over the stored event window. Feeds directly into `moni` dashboards.

```python
summary = await svc.get_analytics_summary(
    tenant_id="acme",
    app_id="app-uuid",
)
# summary keys: total_events, unique_devices, event_breakdown, top_sessions
print(summary["event_breakdown"])
# {"map_tile_load": 4821, "route_calculated": 312, "poi_searched": 98, ...}
```
