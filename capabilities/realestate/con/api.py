"""REST API Blueprint for Property Contracts (con)."""

from __future__ import annotations

from .views import bp as views_bp

api_bp = views_bp

__all__ = ["api_bp"]

"""
OpenAPI summary — Property Contracts (realestate_con)
=====================================================

Base path: /realestate/con

GET    /dashboard                         → contract portfolio summary
GET    /contracts                         → list contracts (?contract_type, ?status)
POST   /contracts                         → create contract
GET    /contracts/<id>                    → get contract
PUT    /contracts/<id>                    → update contract
POST   /contracts/<id>/execute            → execute contract
POST   /contracts/<id>/terminate          → terminate contract
POST   /contracts/<id>/sign/<party_id>    → record party signature
GET    /expiry                            → expiry pipeline (?days=90)
GET    /contractors                       → list contractors (?grade)
POST   /contractors                       → register contractor
GET    /contractors/<id>                  → get contractor
POST   /contractors/<id>/grade            → grade contractor
GET    /milestones                        → list milestones (?contract_id)
POST   /milestones                        → create milestone
POST   /milestones/<id>/complete          → complete milestone
GET    /variations                        → list variation orders (?contract_id)
POST   /variations                        → raise variation order
POST   /variations/<id>/approve           → approve variation order
GET    /disputes                          → list disputes (?contract_id)
POST   /disputes                          → raise dispute
POST   /disputes/<id>/resolve             → resolve dispute
POST   /retention                         → create retention record
POST   /retention/<id>/release            → release retention
GET    /clauses                           → search clause library (?clause_type, ?q)
POST   /clauses                           → add clause to library

Permissions:
  realestate_con:view          → dashboard, expiry pipeline
  realestate_con:contracts     → contract CRUD
  realestate_con:contractors   → contractor registry
  realestate_con:milestones    → milestone tracking
  realestate_con:variations    → variation orders
  realestate_con:disputes      → dispute resolution
  realestate_con:retention     → retention management
  realestate_con:clauses       → clause library
  realestate_con:approvals     → approval queue
  realestate_con:reports       → reporting
  realestate_con:admin         → settings
"""
