# Agri-Marketplace — User Guide

## Overview

agr_mkt connects farmers to buyers through a structured digital marketplace with price
transparency, escrow-based payment protection, and auction mechanisms for perishable produce.

## Key Use Cases

- **Produce Listings**: Farmers post harvested produce with quantity, price, quality grade,
  and availability window. Listings start as drafts and are published when ready.
- **Buyer Matching**: The system identifies buyers who have previously bid on the same product type.
- **Price Discovery**: Real-time market prices derived from accepted bids per product type and region.
- **Escrow**: Funds are held in escrow when a bid is accepted; released to farmer on delivery confirmation.
- **Auctions**: Time-bound ascending-price auctions for high-value or perishable produce.
  Reserve price protects the seller; minimum increment ensures fair progression.

## Example Workflows

### Farmer Posts Maize
```
POST /api/agriculture/mkt/listings
{
  "farmer_id": "farmer-001",
  "product_type": "maize",
  "quantity_kg": 5000,
  "asking_price_per_kg": 42,
  "harvest_date": "2025-08-10",
  "available_from": "2025-08-12",
  "available_to": "2025-09-30",
  "location": "Nakuru"
}
```

### Buyer Places Bid
```
POST /api/agriculture/mkt/bids
{"listing_id": "lst-abc", "buyer_id": "buyer-001", "offered_price_per_kg": 40, "quantity_kg": 3000}
```

### Price Discovery
```
GET /api/agriculture/mkt/prices?product_type=maize&region=Nakuru
```
