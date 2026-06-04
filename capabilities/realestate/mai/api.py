"""REST API Blueprint for Facilities Maintenance (mai)."""

from __future__ import annotations

from .views import bp as views_bp

api_bp = views_bp

__all__ = ["api_bp"]

"""
OpenAPI summary — Facilities Maintenance (realestate_mai)
=========================================================

Base path: /realestate/mai

GET    /dashboard                         → SLA dashboard
GET    /assets                            → list assets (?property_id, ?category, ?status)
POST   /assets                            → register asset
GET    /assets/<id>                       → get asset
PUT    /assets/<id>                       → update asset
GET    /assets/end-of-life                → end-of-life assets (?property_id)
GET    /ppm                               → list PPM schedules (?asset_id, ?status)
POST   /ppm                               → create PPM schedule
POST   /ppm/<id>/complete                 → complete PPM
GET    /ppm/overdue                       → overdue PPM schedules
GET    /work-orders                       → list work orders (?property_id, ?status, ?priority)
POST   /work-orders                       → raise work order
POST   /work-orders/<id>/assign           → assign contractor
PUT    /work-orders/<id>                  → update work order
POST   /work-orders/<id>/close            → close work order (verification required)
GET    /contractors                       → list contractors (?contractor_type)
POST   /contractors                       → register contractor
POST   /inspections                       → schedule inspection
POST   /inspections/<id>/complete         → complete inspection
GET    /inspections/overdue               → overdue inspections
GET    /defects                           → list defects (?property_id, ?severity)
POST   /defects                           → raise defect
POST   /defects/<id>/resolve              → resolve defect
GET    /sla                               → SLA dashboard
POST   /sla                               → create SLA definition

Permissions:
  realestate_mai:view          → dashboard
  realestate_mai:work_orders   → work order management
  realestate_mai:ppm           → PPM schedules
  realestate_mai:assets        → asset register
  realestate_mai:contractors   → contractor management
  realestate_mai:inspections   → inspection management
  realestate_mai:defects       → defect tracking
  realestate_mai:sla           → SLA monitoring
  realestate_mai:costs         → maintenance costs
  realestate_mai:compliance    → compliance
  realestate_mai:reports       → reporting
  realestate_mai:admin         → settings, CAFM integration
"""
