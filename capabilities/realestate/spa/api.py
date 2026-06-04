"""REST API Blueprint for Space Planning & Management (spa)."""

from __future__ import annotations

from .views import bp as views_bp

api_bp = views_bp

__all__ = ["api_bp"]

"""
OpenAPI summary — Space Planning & Management (realestate_spa)
=============================================================

Base path: /realestate/spa

GET    /dashboard                         → spaces summary
GET    /floor-plans                       → list floor plans (?property_id)
POST   /floor-plans                       → upload floor plan
GET    /floor-plans/<id>                  → get floor plan
GET    /spaces                            → list spaces (?property_id, ?space_type, ?status)
POST   /spaces                            → create space
GET    /spaces/<id>                       → get space
PUT    /spaces/<id>                       → update space
GET    /spaces/available                  → available spaces (?property_id, ?space_type, ?min_capacity)
GET    /allocations                       → list allocations (?space_id)
POST   /allocations                       → allocate space
DELETE /allocations/<id>                  → deallocate space
GET    /moves                             → list moves (?status)
POST   /moves                             → create move request
POST   /moves/<id>/approve                → approve move
POST   /moves/<id>/complete               → complete move
GET    /bookings                          → list bookings (?space_id, ?booking_type)
POST   /bookings                          → create booking
DELETE /bookings/<id>                     → cancel booking
POST   /occupancy                         → ingest sensor occupancy data
GET    /occupancy/<property_id>           → occupancy metrics
POST   /density                           → create density plan
GET    /density/<property_id>             → density analysis
GET    /chargeback/<property_id>          → chargeback calculation (?period, ?rate_per_sqm, ?verified)

Permissions:
  realestate_spa:view          → dashboard
  realestate_spa:floor_plans   → floor plan management
  realestate_spa:spaces        → space registry
  realestate_spa:allocations   → space allocation
  realestate_spa:moves         → move management
  realestate_spa:bookings      → space bookings
  realestate_spa:occupancy     → occupancy analytics
  realestate_spa:density       → density planning
  realestate_spa:departments   → department space view
  realestate_spa:chargeback    → chargeback accounting
  realestate_spa:reports       → reporting
  realestate_spa:admin         → settings, sensor integration
"""
