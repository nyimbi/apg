"""REST API Blueprint for Property Valuation (val)."""

from __future__ import annotations

from .views import bp as views_bp

api_bp = views_bp

__all__ = ["api_bp"]

"""
OpenAPI summary — Property Valuation (realestate_val)
=====================================================

Base path: /realestate/val

GET    /dashboard                         → valuation portfolio summary
GET    /valuers                           → list valuers (?grade, ?independent=true)
POST   /valuers                           → register valuer
GET    /valuers/<id>                      → get valuer
GET    /comparables                       → list comparables (?comparable_type, ?verified_only)
POST   /comparables                       → add comparable
POST   /comparables/<id>/verify           → verify comparable
GET    /valuations                        → list valuations (?property_id, ?status)
POST   /valuations                        → instruct valuation
GET    /valuations/<id>                   → get valuation
PUT    /valuations/<id>                   → update valuation
POST   /valuations/<id>/sign-off          → sign off valuation
POST   /valuations/<id>/publish           → publish valuation (immutable after)
POST   /dcf                               → run DCF model
GET    /dcf/<id>                          → get DCF model
POST   /mass-appraisal                    → run mass appraisal
GET    /mass-appraisal/<id>               → get mass appraisal run
GET    /roll                              → valuation roll (?property_id)
POST   /roll                              → add to valuation roll
GET    /yields/<property_id>              → calculate yield (?annual_rent, ?purchase_price, ?yield_type)
GET    /challenges                        → list challenges (?valuation_id)
POST   /challenges                        → raise challenge (counter evidence required)
POST   /challenges/<id>/resolve           → resolve challenge
GET    /benchmarking                      → benchmarking dashboard

Permissions:
  realestate_val:view          → dashboard
  realestate_val:valuations    → valuation management
  realestate_val:comparables   → comparable database
  realestate_val:dcf           → DCF model builder
  realestate_val:mass_appraisal → mass appraisal
  realestate_val:roll          → valuation roll
  realestate_val:cycles        → revaluation cycles
  realestate_val:yields        → yield analysis
  realestate_val:valuers       → valuer panel registry
  realestate_val:challenges    → challenge workflow
  realestate_val:benchmarking  → benchmarking
  realestate_val:reports       → reporting
  realestate_val:admin         → settings
"""
