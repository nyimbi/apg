"""REST API Blueprint for Property Management (prm).

All endpoints return JSON: {"status": "ok"|"error", "data": ..., "message": ...}
"""

from __future__ import annotations

from .views import bp as views_bp

api_bp = views_bp

__all__ = ["api_bp"]

"""
OpenAPI summary — Property Management (realestate_prm)
======================================================

Base path: /realestate/prm

GET    /dashboard                         → portfolio summary
GET    /owners                            → list owners
POST   /owners                            → register owner
GET    /owners/<id>                       → get owner
PUT    /owners/<id>                       → update owner
GET    /properties                        → list properties (?portfolio_tier, ?status)
POST   /properties                        → register property
GET    /properties/<id>                   → get property
PUT    /properties/<id>                   → update property
DELETE /properties/<id>                   → delete property (board_approved required)
GET    /properties/search                 → search properties (?q)
GET    /units                             → list units (?property_id, ?status)
POST   /units                             → create unit
GET    /units/<id>                        → get unit
PUT    /units/<id>                        → update unit
GET    /units/void                        → list void units (?property_id)
POST   /kpis                              → calculate KPIs
GET    /distributions                     → list distributions (?owner_id)
POST   /distributions                     → create distribution
POST   /distributions/<id>/approve        → approve distribution (dual control)
POST   /handovers                         → create handover
POST   /handovers/<id>/complete           → complete handover

Permissions:
  realestate_prm:view          → dashboard
  realestate_prm:portfolio     → portfolio overview
  realestate_prm:properties    → property CRUD
  realestate_prm:units         → unit management
  realestate_prm:owners        → owner management
  realestate_prm:owner_portal  → owner portal
  realestate_prm:performance   → KPI dashboard
  realestate_prm:kpis          → KPI builder
  realestate_prm:handovers     → handover management
  realestate_prm:distributions → owner distributions
  realestate_prm:data_room     → property data room
  realestate_prm:reports       → reporting
  realestate_prm:admin         → settings
"""
