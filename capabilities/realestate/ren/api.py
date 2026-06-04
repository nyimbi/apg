"""REST API Blueprint for Rental Operations (ren)."""

from __future__ import annotations

from .views import bp as views_bp

api_bp = views_bp

__all__ = ["api_bp"]

"""
OpenAPI summary — Rental Operations (realestate_ren)
====================================================

Base path: /realestate/ren

GET    /dashboard                         → rent roll count, active arrears, renewals due
GET    /tenancies                         → list tenancies (?unit_id, ?status)
POST   /tenancies                         → create tenancy
GET    /tenancies/<id>                    → get tenancy
PUT    /tenancies/<id>                    → update tenancy
POST   /tenancies/<id>/activate           → activate tenancy (pre-conditions enforced)
GET    /rent-collection                   → list payments (?tenancy_id, ?period)
POST   /rent-collection                   → record rent payment
GET    /arrears                           → list active arrears
POST   /arrears/<id>/legal                → escalate arrears to legal
POST   /deposits                          → register deposit
GET    /deposits/<id>                     → get deposit
POST   /deposits/<id>/deduct              → deduct from deposit (evidence required)
POST   /deposits/<id>/release             → release deposit
GET    /notices                           → list notices (?tenancy_id)
POST   /notices                           → serve notice
POST   /renewals                          → initiate renewal
POST   /renewals/<id>/accept              → accept renewal
GET    /renewals/pipeline                 → renewal pipeline (?months=3)
POST   /referencing                       → run referencing
POST   /referencing/<id>/complete         → complete referencing
GET    /rent-roll                         → rent roll (?property_id)

Permissions:
  realestate_ren:view          → dashboard
  realestate_ren:tenancies     → tenancy management
  realestate_ren:referencing   → referencing
  realestate_ren:rent_collection → rent collection
  realestate_ren:arrears       → arrears management
  realestate_ren:deposits      → deposit accounting
  realestate_ren:renewals      → renewal pipeline
  realestate_ren:notices       → notice management
  realestate_ren:legal         → legal actions
  realestate_ren:rent_roll     → rent roll
  realestate_ren:reports       → reporting
  realestate_ren:admin         → settings
"""
