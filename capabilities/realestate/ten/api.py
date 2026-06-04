"""REST API Blueprint for Tenant Management (ten)."""

from __future__ import annotations

from .views import bp as views_bp

api_bp = views_bp

__all__ = ["api_bp"]

"""
OpenAPI summary — Tenant Management (realestate_ten)
====================================================

Base path: /realestate/ten

GET    /dashboard                         → tenant portfolio summary
GET    /tenants                           → list tenants (?status, ?tenant_type)
POST   /tenants                           → register tenant
GET    /tenants/<id>                      → get tenant (access always logged)
PUT    /tenants/<id>                      → update tenant
POST   /tenants/<id>/activate             → activate tenant
POST   /tenants/<id>/blacklist            → blacklist tenant
POST   /tenants/<id>/grade                → assign credit grade
GET    /onboarding/<tenant_entity_id>     → onboarding progress
POST   /onboarding                        → complete onboarding step
GET    /service-requests                  → list service requests (?tenant_entity_id, ?status)
POST   /service-requests                  → raise service request
GET    /service-requests/<id>             → get service request
PUT    /service-requests/<id>             → update service request
POST   /service-requests/<id>/resolve     → resolve service request
GET    /communications                    → list communications (?tenant_entity_id, ?channel)
POST   /communications                    → send communication
GET    /satisfaction                      → list satisfaction surveys (?tenant_entity_id)
POST   /satisfaction                      → record satisfaction survey
GET    /satisfaction/<tenant_entity_id>/trend → satisfaction trend
POST   /scoring                           → calculate tenant score
GET    /escalations                       → list escalations (?tenant_entity_id)
POST   /escalations                       → raise escalation
POST   /escalations/<id>/resolve          → resolve escalation
GET    /retention/at-risk                 → tenants at retention risk

Permissions:
  realestate_ten:view          → dashboard
  realestate_ten:tenants       → tenant management
  realestate_ten:onboarding    → onboarding workflow
  realestate_ten:service_requests → service requests
  realestate_ten:communications → communication portal
  realestate_ten:satisfaction  → satisfaction tracking
  realestate_ten:scoring       → tenant scoring
  realestate_ten:escalations   → escalation management
  realestate_ten:documents     → document management
  realestate_ten:timeline      → event timeline
  realestate_ten:retention     → retention analytics
  realestate_ten:reports       → reporting
  realestate_ten:admin         → settings
"""
