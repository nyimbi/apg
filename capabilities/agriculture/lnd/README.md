# Land Management (agr_lnd)

Parcel cadastre, tenure registry, GPS boundary capture, title issuance, land transfer,
dispute adjudication, encumbrance registry, subdivision/amalgamation, valuation roll,
rate billing, tenure formalisation, geospatial search, chain of title, and webhook streaming.

## API Endpoints

### Core Registry

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/lnd/health | Health check |
| GET | /api/agriculture/lnd/describe | Capability metadata |
| GET | /api/agriculture/lnd/registry-summary | Registry statistics |
| GET | /api/agriculture/lnd/audit | Audit log |

### Parcels

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/lnd/parcels | List parcels (filter by owner, county, tenure, status) |
| POST | /api/agriculture/lnd/parcels | Register parcel |
| GET | /api/agriculture/lnd/parcels/{id} | Get parcel |
| PUT | /api/agriculture/lnd/parcels/{id} | Update parcel |
| DELETE | /api/agriculture/lnd/parcels/{id} | Delete parcel |
| GET | /api/agriculture/lnd/owners/{id}/holdings | Owner land holdings summary |

### GPS Boundaries

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/lnd/boundaries | List boundaries |
| POST | /api/agriculture/lnd/boundaries | Capture GPS boundary (auto-computes area) |
| DELETE | /api/agriculture/lnd/boundaries/{id} | Delete boundary |

### Titles

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/lnd/titles | List titles |
| POST | /api/agriculture/lnd/titles | Issue title |
| POST | /api/agriculture/lnd/titles/{id}/invalidate | Invalidate title |
| DELETE | /api/agriculture/lnd/titles/{id} | Delete title |

### Transfers

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/lnd/transfers | List transfers |
| POST | /api/agriculture/lnd/transfers | Initiate transfer |
| PUT | /api/agriculture/lnd/transfers/{id} | Update transfer status |
| DELETE | /api/agriculture/lnd/transfers/{id} | Delete transfer |
| POST | /api/agriculture/lnd/transfers/{id}/sign | Multi-sig approval signature |

### Disputes & Adjudication

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/lnd/disputes | List disputes |
| POST | /api/agriculture/lnd/disputes | File dispute |
| PUT | /api/agriculture/lnd/disputes/{id}/advance | Advance dispute stage |

### Encumbrances

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/lnd/encumbrances | List encumbrances |
| POST | /api/agriculture/lnd/encumbrances | Register encumbrance |
| POST | /api/agriculture/lnd/encumbrances/{id}/discharge | Discharge encumbrance |

### Subdivision & Amalgamation

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/agriculture/lnd/parcels/{id}/subdivide | Subdivide parcel into children |
| POST | /api/agriculture/lnd/parcels/amalgamate | Merge parcels into one |

### Valuation & Rate Bills

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/agriculture/lnd/valuations | Record parcel valuation |
| POST | /api/agriculture/lnd/parcels/{id}/rate-bill | Generate rate bill |
| GET | /api/agriculture/lnd/rate-bills | List rate bills |

### Tenure Formalisation

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/agriculture/lnd/formalisations | Initiate formalisation workflow |
| POST | /api/agriculture/lnd/formalisations/{id}/advance | Advance formalisation stage |
| GET | /api/agriculture/lnd/formalisations/{id} | Get formalisation status |

### Geospatial Search

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/lnd/parcels/search/location | Radius search by lat/lng |
| GET | /api/agriculture/lnd/parcels/search/point | Point-in-polygon lookup |

### Chain of Title

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/lnd/parcels/{id}/chain-of-title | Full ownership history |

### Webhooks

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/lnd/webhooks | List webhooks |
| POST | /api/agriculture/lnd/webhooks | Register webhook |
| DELETE | /api/agriculture/lnd/webhooks/{id} | Delete webhook |

## Key Features

- **Monetary precision**: All financial values (transfer values, encumbrance amounts, assessments, rate bills) stored and computed as `Decimal` strings — no float rounding errors.
- **Dispute adjudication**: Structured 5-stage workflow locks parcel on filing, unlocks on resolution.
- **Encumbrance registry**: Full mortgage/caveat/lien/easement lifecycle with discharge workflow; satisfies Kenya Land Registration Act §59.
- **Multi-sig transfers**: High-value transfers (≥ KES 10M) require 2-approver quorum; each signature is HMAC-protected with chain-of-custody hash.
- **Subdivision/amalgamation**: Area conservation validated with `Decimal` arithmetic; full parent→child lineage preserved in audit trail.
- **Tenure formalisation**: 6-stage customary→statutory workflow per Kenya Community Land Act 2016.
- **Geospatial search**: Haversine radius search and ray-casting point-in-polygon against all stored boundaries.
- **Chain of title**: Full provenance reconstruction with encumbrance overlay per ownership period.
- **Webhook streaming**: HMAC-SHA-256 signed event delivery to registered endpoints; county offices, banks, and KRA can subscribe to parcel events.

## Tenure Types

`freehold` | `leasehold` | `customary` | `communal` | `government`

## Encumbrance Types

`mortgage` | `caveat` | `lien` | `easement` | `charge`

## Transfer Status Lifecycle

`initiated` → (multi-sig) → `approved` → `registered` | `rejected`

## Dispute Stage Lifecycle

`filed` → `evidence_collection` → `hearing_scheduled` → `adjudicated` → `appealed`

## Formalisation Stage Lifecycle

`community_consent` → `demarcation` → `survey` → `adjudication` → `registration` → `title_issued`
