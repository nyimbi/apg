"""REST API Blueprint for Property Insurance (ins)."""

from __future__ import annotations

from .views import bp as views_bp

api_bp = views_bp

__all__ = ["api_bp"]

"""
OpenAPI summary — Property Insurance (realestate_ins)
=====================================================

Base path: /realestate/ins

GET    /dashboard                         → insurance portfolio summary
GET    /insurers                          → list insurers (?grade)
POST   /insurers                          → register insurer
GET    /insurers/<id>                     → get insurer
GET    /policies                          → list policies (?property_id, ?status)
POST   /policies                          → create policy
GET    /policies/<id>                     → get policy
PUT    /policies/<id>                     → update policy
POST   /policies/<id>/bind                → bind (activate) policy
GET    /renewals                          → renewal pipeline (?days=90)
GET    /assets                            → policy asset schedule (?policy_id required)
POST   /assets                            → add asset to schedule
DELETE /assets/<id>                       → remove asset from schedule
GET    /claims                            → list claims (?policy_id, ?status)
POST   /claims                            → lodge claim
GET    /claims/<id>                       → get claim
POST   /claims/<id>/approve               → approve claim
POST   /claims/<id>/settle                → settle claim
GET    /endorsements                      → list endorsements (?policy_id)
POST   /endorsements                      → issue endorsement
POST   /premiums                          → run premium allocation
GET    /gaps                              → list coverage gaps (?property_id)
POST   /gaps/detect/<property_id>         → detect coverage gaps for property

Permissions:
  realestate_ins:view          → dashboard
  realestate_ins:policies      → policy management
  realestate_ins:assets        → asset schedule
  realestate_ins:claims        → claims processing
  realestate_ins:premiums      → premium allocation
  realestate_ins:gaps          → coverage gap analysis
  realestate_ins:endorsements  → endorsements
  realestate_ins:insurers      → insurer registry
  realestate_ins:brokers       → broker registry
  realestate_ins:renewals      → renewal pipeline
  realestate_ins:certificates  → compliance certificates
  realestate_ins:reports       → reporting
  realestate_ins:admin         → settings
"""
