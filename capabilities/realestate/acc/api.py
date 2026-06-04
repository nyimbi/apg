"""REST API Blueprint for Real Estate Accounting (acc).

All endpoints return JSON: {"status": "ok"|"error", "data": ..., "message": ...}
Authentication via X-Tenant-ID header (replace with real JWT middleware in production).
"""

from __future__ import annotations

from flask import Blueprint
from .views import bp as views_bp

# Re-export the views blueprint as the public API surface.
# In production, register this blueprint on the Flask app:
#   app.register_blueprint(api_bp)
api_bp = views_bp

__all__ = ["api_bp"]

"""
OpenAPI summary — Real Estate Accounting (realestate_acc)
=========================================================

Base path: /realestate/acc

GET    /dashboard                         → financial summary
GET    /accounts                          → list accounts (?property_id)
POST   /accounts                          → create account
GET    /accounts/<id>                     → get account
PUT    /accounts/<id>                     → update account
GET    /journals                          → list journals (?period, ?property_id)
POST   /journals                          → create journal entry
POST   /journals/<id>/approve             → approve journal
POST   /journals/<id>/post                → post journal to ledger
POST   /journals/<id>/reverse             → create reversal entry
GET    /service-charges                   → list charges (?property_id, ?period)
POST   /service-charges                   → raise service charge
POST   /service-charges/<id>/approve      → approve service charge
GET    /cam                               → list CAM reconciliations
POST   /cam                               → start CAM reconciliation
POST   /cam/<id>/approve                  → approve CAM reconciliation
POST   /cam/<id>/settle                   → settle CAM reconciliation
POST   /ifrs16                            → generate IFRS 16 schedule
GET    /ifrs16/<id>                       → get IFRS 16 schedule
POST   /revenue                           → create revenue schedule
POST   /periods                           → open accounting period
POST   /periods/<id>/close                → close period (dual control)
POST   /statements                        → generate tenant statement
GET    /statements/<id>                   → get tenant statement
GET    /reports/trial-balance             → trial balance (?period)

Permissions:
  realestate_acc:view          → dashboard
  realestate_acc:ledger        → accounts
  realestate_acc:journals      → journal entries
  realestate_acc:service_charges → service charges
  realestate_acc:cam           → CAM reconciliation
  realestate_acc:ifrs16        → IFRS 16 schedules
  realestate_acc:revenue       → revenue schedules
  realestate_acc:period_close  → periods
  realestate_acc:statements    → tenant statements
  realestate_acc:reports       → reports
  realestate_acc:admin         → settings
"""
