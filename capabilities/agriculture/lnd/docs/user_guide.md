# Land Management — User Guide

## Overview

agr_lnd provides a digital land registry: registering parcels with tenure and ownership,
capturing GPS boundaries with automatic area computation, issuing titles, and managing
the full land transfer workflow.

## Key Concepts

- **Parcel**: Base unit of land with parcel number, area, tenure type, owner, and location.
- **GPS Boundary**: A polygon defined by lat/lng waypoints. Area is computed via the
  Shoelace formula on an equirectangular projection.
- **Title**: Issued document proving ownership. Can be invalidated without deletion for audit trail.
- **Transfer**: Workflow: initiated → pending_approval → approved → registered.
  On registration, parcel ownership updates automatically.

## Example Workflows

### Register a Parcel
```
POST /api/agriculture/lnd/parcels
{
  "parcel_number": "NAKURU/NAIVASHA/1234",
  "area_ha": 3.5,
  "tenure_type": "freehold",
  "owner_id": "farmer-001",
  "owner_name": "Jane Wanjiku",
  "location_county": "Nakuru",
  "location_sub_county": "Naivasha",
  "land_use": "arable"
}
```

### Capture GPS Boundary
```
POST /api/agriculture/lnd/boundaries
{
  "parcel_id": "lnd-abc",
  "captured_by": "surveyor-001",
  "waypoints": [
    {"lat": -0.7167, "lng": 36.4333},
    {"lat": -0.7180, "lng": 36.4350},
    {"lat": -0.7160, "lng": 36.4360},
    {"lat": -0.7167, "lng": 36.4333}
  ],
  "accuracy_m": 2.5
}
```

### Initiate Land Transfer
```
POST /api/agriculture/lnd/transfers
{
  "parcel_id": "lnd-abc",
  "from_owner_id": "farmer-001",
  "to_owner_id": "farmer-002",
  "to_owner_name": "Peter Mwangi",
  "transfer_value": 1500000,
  "reason": "sale"
}
```
