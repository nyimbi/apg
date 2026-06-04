"""REST API Blueprint for Lease Management (lea)."""

from __future__ import annotations

from .views import bp as views_bp

api_bp = views_bp

__all__ = ["api_bp"]

"""
OpenAPI summary — Lease Management (realestate_lea)
===================================================

Base path: /realestate/lea

GET    /dashboard                         → expiry pipeline + expiring options
GET    /leases                            → list leases (?property_id, ?status)
POST   /leases                            → create lease
GET    /leases/<id>                       → get lease
PUT    /leases/<id>                       → update lease
POST   /leases/<id>/activate              → activate lease
POST   /leases/<id>/surrender             → surrender lease
POST   /abstraction                       → create abstraction
POST   /abstraction/<id>/verify           → verify abstraction
GET    /escalations                       → list escalations (?lease_id)
POST   /escalations                       → create escalation
POST   /escalations/<id>/apply            → apply escalation to lease
POST   /options                           → create lease option
POST   /options/<id>/exercise             → exercise option
GET    /options/expiring                  → expiring options (?days=180)
POST   /rent-reviews                      → commence rent review
POST   /rent-reviews/<id>/agree           → agree rent review outcome
POST   /ifrs16                            → generate IFRS 16 schedule
POST   /ifrs16/<id>/reclassify            → reclassify IFRS 16 category
POST   /assignments                       → create lease assignment
POST   /assignments/<id>/complete         → complete assignment
GET    /expiry                            → expiry pipeline (?months=12)

Permissions:
  realestate_lea:view          → dashboard, expiry pipeline
  realestate_lea:leases        → lease CRUD
  realestate_lea:abstraction   → lease abstraction
  realestate_lea:escalations   → rent escalations
  realestate_lea:rent_reviews  → rent reviews
  realestate_lea:options       → option tracking
  realestate_lea:ifrs16        → IFRS 16 / ASC 842
  realestate_lea:assignments   → lease assignments
  realestate_lea:dilapidations → dilapidations
  realestate_lea:renewals      → renewal pipeline
  realestate_lea:reports       → reporting
  realestate_lea:admin         → settings
"""
