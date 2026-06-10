# RSV User Guide

## Overview

The Reservations & Channel Manager (hos_rsv) provides a Central Reservations System with multi-channel distribution, GDS connectivity, real-time availability management, and an integrated booking engine.

## Key Concepts

- **Channels**: Distribution channels (OTA, GDS, direct, booking engine) with commission rates.
- **Bookings**: Reservations received from any channel; commission and net revenue computed automatically.
- **Availability**: Per room-type, per-date inventory counts with stop-sell controls.
- **GDS Connections**: Amadeus / Sabre / Travelport integrations for global distribution.
- **Waitlist**: Demand capture when rooms are unavailable.

## Workflow

1. Register channels (Booking.com, Expedia, GDS, direct) with commission percentages.
2. Set availability for room types and dates.
3. Bookings arrive via channel API or manual entry; availability decrements automatically.
4. Sync availability to GDS connections.
5. Monitor channel performance analytics.
