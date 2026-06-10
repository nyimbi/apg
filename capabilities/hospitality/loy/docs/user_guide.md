# LOY User Guide

## Overview

The Guest Loyalty Programme (hos_loy) drives repeat business through points accrual, tier-based privileges, partner network rewards, and personalised recognition.

## Tier Structure

| Tier | Lifetime Points | Qualifying Nights | Points Multiplier |
|------|----------------|-------------------|-------------------|
| Bronze | 0+ | 0+ | 1.0x |
| Silver | 5,000+ | 10+ | 1.25x |
| Gold | 15,000+ | 25+ | 1.5x |
| Platinum | 50,000+ | 50+ | 2.0x |

## Points Economy

- **Earn**: 1 point per KES spent (multiplied by tier and active bonus campaigns).
- **Redeem**: 1 point = 0.05 KES (minimum 100 points per redemption).
- **Enrollment Bonus**: 500 points on signup.
- **Expiry**: Points expire per configured policy; use the `expire_points` endpoint.

## Recognition Preferences

Store guest preferences (room floor, pillow type, newspaper, dietary restrictions, etc.) via the preferences endpoint. These are surfaced to front desk and housekeeping on each stay.

## Partner Programme

Hotels can link airline, car rental, restaurant, and retail partners. Guests earn additional points on partner spend and can redeem at partner venues.
