# Land Registry (gov_lnd)

Parcel cadastre, title issuance, land transfer, adjudication, encumbrance registry, valuation rolls.

## Overview

Complete land administration system for managing land parcels, title deeds, ownership transfers, dispute adjudication, encumbrances/charges, and property valuations. Designed for national/county land registries in East Africa.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/government/lnd/health | Service health check |
| GET | /api/government/lnd/dashboard | Dashboard metrics |
| GET | /api/government/lnd/parcels | List parcels |
| GET | /api/government/lnd/parcels/{id} | Get parcel |
| POST | /api/government/lnd/parcels | Register parcel |
| PUT | /api/government/lnd/parcels/{id} | Update parcel |
| DELETE | /api/government/lnd/parcels/{id} | Deregister parcel |
| GET | /api/government/lnd/titles | List titles |
| GET | /api/government/lnd/titles/{id} | Get title |
| POST | /api/government/lnd/titles | Issue title |
| PUT | /api/government/lnd/titles/{id} | Update title |
| GET | /api/government/lnd/transfers | List transfers |
| POST | /api/government/lnd/transfers | Initiate transfer |
| POST | /api/government/lnd/transfers/{id}/complete | Complete transfer |
| GET | /api/government/lnd/adjudications | List adjudications |
| POST | /api/government/lnd/adjudications | Submit adjudication |
| GET | /api/government/lnd/encumbrances | List encumbrances |
| POST | /api/government/lnd/encumbrances | Register encumbrance |
| POST | /api/government/lnd/encumbrances/{id}/discharge | Discharge encumbrance |
| GET | /api/government/lnd/valuations | List valuations |
| POST | /api/government/lnd/valuations | Record valuation |
| GET | /api/government/lnd/audit-events | List audit events |
