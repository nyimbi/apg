# RMS User Guide

## Overview

The Revenue Management & Rates capability (hos_rms) provides dynamic pricing intelligence, demand-driven rate recommendations, competitor benchmarking, and yield optimisation strategies.

## Key Concepts

- **Rate Plans**: Named rate packages (BAR, Advance Purchase, etc.) with base rates, meal plans, and cancellation policies.
- **Demand Tiers**: low / medium / high / peak / super_peak — automatically derived from forecasted occupancy, applied as multipliers to base rates.
- **Seasonal Rules**: Date-range multipliers that overlay on all rate plans (e.g., Christmas +40%).
- **Price Overrides**: One-off date-specific rate overrides for special circumstances.
- **Yield Optimisation**: Gap analysis between current and target occupancy driving discount or premium strategies.
- **Rate Parity**: Automated alerts when our published rates deviate ≥5% from tracked competitor rates.

## Workflow

1. Create rate plans with base rates per room type.
2. Set seasonal rules for high-demand periods.
3. Create demand forecasts; the system recommends rates automatically.
4. Monitor competitor rates; parity alerts fire when deviation exceeds threshold.
5. Run yield optimisation to receive pricing strategy recommendations.
