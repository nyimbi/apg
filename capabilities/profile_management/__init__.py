"""
Profile Management & Registration Capability

This capability provides comprehensive user profile management and registration
functionality with GDPR compliance, multi-tenant support, and Flask-AppBuilder integration.
"""

from .models import PMUser, PMProfile, PMConsent, PMRegistration, PMPreferences
from .services import ProfileService, RegistrationService, ConsentService
from .views import register_profile_views
from .events import ProfileEventEmitter, ProfileEvents

__version__ = "1.0.0"
__capability_code__ = "PROFILE_MGMT"
__capability_name__ = "Profile Management & Registration"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
    "requires_profile_capability",
    "emits_profile_events", 
    "consumes_profile_events",
    "integrates_with_profiles"
]

# Export main interfaces for capability composition
__all__ = [
    "PMUser",
    "PMProfile", 
    "PMConsent",
    "PMRegistration",
    "PMPreferences",
    "ProfileService",
    "RegistrationService",
    "ConsentService",
    "register_profile_views",
    "ProfileEventEmitter",
    "ProfileEvents"
]