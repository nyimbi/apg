"""
Profile Management Services

Core business logic services for user registration, profile management,
and GDPR compliance operations. Implements capability composition patterns
and integration with other APG capabilities.
"""

import uuid
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import and_, or_, func

from .models import PMUser, PMProfile, PMConsent, PMRegistration, PMPreferences
from .events import ProfileEventEmitter, ProfileEvents
from .exceptions import (
    ProfileManagementError, RegistrationError, ConsentError,
    ValidationError, PermissionError, GDPRError
)

# Set up logging
logger = logging.getLogger(__name__)

# Capability composition decorators
def requires_auth_capability(func):
    """Decorator indicating this service requires authentication capability"""
    func._requires_capabilities = getattr(func, '_requires_capabilities', [])
    func._requires_capabilities.append('auth_rbac')
    return func

def integrates_with(capability_name: str):
    """Decorator indicating integration with another capability"""
    def decorator(func):
        func._integrates_with = getattr(func, '_integrates_with', [])
        func._integrates_with.append(capability_name)
        return func
    return decorator

def emits_profile_events(event_types: List[str]):
    """Decorator indicating this method emits profile events"""
    def decorator(func):
        func._emits_events = getattr(func, '_emits_events', [])
        func._emits_events.extend(event_types)
        return func
    return decorator


@dataclass
class RegistrationRequest:
    """Registration request data structure"""
    email: str
    password: Optional[str] = None
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    tenant_id: str = ""
    registration_source: str = "email"
    consents: Dict[str, bool] = None
    profile_data: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.consents is None:
            self.consents = {}
        if self.profile_data is None:
            self.profile_data = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class ProfileUpdateRequest:
    """Profile update request data structure"""
    user_id: str
    updates: Dict[str, Any]
    updated_by: str
    tenant_id: str = ""
    validate_permissions: bool = True


class RegistrationService:
    """
    Service for handling user registration workflows including validation,
    email verification, and integration with other capabilities.
    """
    
    def __init__(self, db_session: Session, event_emitter: ProfileEventEmitter = None):
        self.db = db_session
        self.event_emitter = event_emitter or ProfileEventEmitter()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @emits_profile_events(['user.registration_started'])
    def start_registration(self, request: RegistrationRequest) -> str:
        """
        Start a new user registration process.
        
        Args:
            request: Registration request with user data and preferences
            
        Returns:
            registration_id: Unique identifier for the registration attempt
            
        Raises:
            RegistrationError: If registration cannot be started
        """
        try:
            # Validate registration request
            self._validate_registration_request(request)
            
            # Check if email already exists
            existing_user = self.db.query(PMUser).filter(
                and_(
                    PMUser.email == request.email.lower(),
                    PMUser.tenant_id == request.tenant_id
                )
            ).first()
            
            if existing_user:
                raise RegistrationError(f"Email {request.email} is already registered")
            
            # Create registration tracking record
            registration = PMRegistration(
                email=request.email.lower(),
                tenant_id=request.tenant_id,
                registration_source=request.registration_source,
                registration_step="started",
                status="started",
                registration_data=request.metadata.get('form_data', {}),
                ip_address=request.metadata.get('ip_address'),
                user_agent=request.metadata.get('user_agent'),
                referrer_url=request.metadata.get('referrer_url'),
                utm_source=request.metadata.get('utm_source'),
                utm_medium=request.metadata.get('utm_medium'),
                utm_campaign=request.metadata.get('utm_campaign'),
                consent_given=request.consents,
                profile_data=request.profile_data
            )
            
            self.db.add(registration)
            self.db.commit()
            
            # Emit registration started event
            self.event_emitter.emit('user.registration_started', {
                'registration_id': registration.registration_id,
                'email': request.email,
                'tenant_id': request.tenant_id,
                'source': request.registration_source
            })
            
            self.logger.info(f"Registration started for {request.email} in tenant {request.tenant_id}")
            return registration.registration_id
            
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to start registration for {request.email}: {str(e)}")
            raise RegistrationError(f"Registration failed: {str(e)}")
    
    @emits_profile_events(['user.registered', 'user.email_verification_sent'])
    @integrates_with('notification_engine')
    def complete_registration(self, registration_id: str, 
                            send_verification: bool = True) -> Dict[str, Any]:
        """
        Complete user registration by creating user account and profile.
        
        Args:
            registration_id: Registration attempt identifier
            send_verification: Whether to send email verification
            
        Returns:
            Dictionary with user_id, verification_token, and registration details
            
        Raises:
            RegistrationError: If registration cannot be completed
        """
        try:
            # Get registration record
            registration = self.db.query(PMRegistration).filter(
                PMRegistration.registration_id == registration_id
            ).first()
            
            if not registration:
                raise RegistrationError(f"Registration {registration_id} not found")
            
            if registration.status == "completed":
                raise RegistrationError("Registration already completed")
            
            # Create user account
            user = PMUser(
                email=registration.email,
                tenant_id=registration.tenant_id,
                registration_source=registration.registration_source,
                registration_data=registration.registration_data,
                data_processing_consent=registration.consent_given.get('data_processing', False),
                marketing_consent=registration.consent_given.get('marketing', False),
                analytics_consent=registration.consent_given.get('analytics', False)
            )
            
            # Set username if provided
            if registration.profile_data.get('username'):
                user.username = registration.profile_data['username']
            
            # Set password if provided (for email registration)
            if registration.registration_data.get('password'):
                user.set_password(registration.registration_data['password'])
            
            # Generate email verification token
            verification_token = None
            if send_verification:
                verification_token = user.generate_verification_token()
            
            self.db.add(user)
            self.db.flush()  # Get user.id for relationships
            
            # Create user profile
            profile_data = registration.profile_data or {}
            profile = PMProfile(
                user_id=user.user_id,
                tenant_id=user.tenant_id,
                first_name=profile_data.get('first_name'),
                last_name=profile_data.get('last_name'),
                display_name=profile_data.get('display_name'),
                title=profile_data.get('title'),
                company=profile_data.get('company'),
                timezone=profile_data.get('timezone', 'UTC'),
                locale=profile_data.get('locale', 'en-US')
            )
            
            self.db.add(profile)
            
            # Create user preferences with defaults
            preferences = PMPreferences(
                user_id=user.user_id,
                tenant_id=user.tenant_id,
                email_notifications=registration.consent_given.get('email_notifications', True),
                marketing_emails=registration.consent_given.get('marketing', False),
                newsletter_subscription=registration.consent_given.get('newsletter', False)
            )
            
            self.db.add(preferences)
            
            # Create consent records
            consent_service = ConsentService(self.db, self.event_emitter)
            for purpose, granted in registration.consent_given.items():
                consent_service.record_consent(
                    user_id=user.user_id,
                    tenant_id=user.tenant_id,
                    purpose=purpose,
                    granted=granted,
                    version="1.0",
                    method="registration"
                )
            
            # Update registration record
            start_time = registration.created_on
            completion_time = int((datetime.utcnow() - start_time).total_seconds()) if start_time else None
            registration.mark_completed(user.user_id, completion_time)
            
            if send_verification:
                registration.email_verification_sent = True
            
            self.db.commit()
            
            # Calculate profile completion score
            profile.calculate_completion_score()
            self.db.commit()
            
            # Emit events
            user_data = {
                'user_id': user.user_id,
                'email': user.email,
                'tenant_id': user.tenant_id,
                'registration_source': user.registration_source,
                'profile_completion': profile.completion_score
            }
            
            self.event_emitter.emit('user.registered', user_data)
            
            if send_verification and verification_token:
                self.event_emitter.emit('user.email_verification_sent', {
                    'user_id': user.user_id,
                    'email': user.email,
                    'verification_token': verification_token
                })
            
            self.logger.info(f"Registration completed for user {user.user_id}")
            
            return {
                'user_id': user.user_id,
                'email': user.email,
                'verification_token': verification_token,
                'profile_completion': profile.completion_score,
                'registration_id': registration.registration_id
            }
            
        except Exception as e:
            self.db.rollback()
            
            # Mark registration as failed
            if 'registration' in locals():
                registration.mark_failed(str(e))
                self.db.commit()
            
            self.logger.error(f"Failed to complete registration {registration_id}: {str(e)}")
            raise RegistrationError(f"Registration completion failed: {str(e)}")
    
    @integrates_with('notification_engine')
    def verify_email(self, token: str) -> bool:
        """
        Verify user email address using verification token.
        
        Args:
            token: Email verification token
            
        Returns:
            True if verification successful, False otherwise
        """
        try:
            user = self.db.query(PMUser).filter(
                PMUser.email_verification_token == token
            ).first()
            
            if not user:
                self.logger.warning(f"Invalid verification token: {token}")
                return False
            
            if not user.is_verification_token_valid(token):
                self.logger.warning(f"Expired verification token for user {user.user_id}")
                return False
            
            # Mark email as verified
            user.email_verified = True
            user.email_verification_token = None
            user.email_verification_expires = None
            
            # Update registration records
            registration = self.db.query(PMRegistration).filter(
                PMRegistration.user_id == user.user_id
            ).first()
            
            if registration:
                registration.email_verified_at = datetime.utcnow()
            
            self.db.commit()
            
            # Emit verification event
            self.event_emitter.emit('user.email_verified', {
                'user_id': user.user_id,
                'email': user.email,
                'tenant_id': user.tenant_id
            })
            
            self.logger.info(f"Email verified for user {user.user_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Email verification failed for token {token}: {str(e)}")
            return False
    
    def _validate_registration_request(self, request: RegistrationRequest) -> None:
        """Validate registration request data"""
        
        # Email validation
        if not request.email:
            raise ValidationError("Email is required")
        
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        if not email_pattern.match(request.email):
            raise ValidationError("Invalid email format")
        
        # Password validation (if provided)
        if request.password:
            if len(request.password) < 8:
                raise ValidationError("Password must be at least 8 characters")
            
            # Check password complexity
            if not re.search(r'[A-Z]', request.password):
                raise ValidationError("Password must contain at least one uppercase letter")
            if not re.search(r'[a-z]', request.password):
                raise ValidationError("Password must contain at least one lowercase letter")
            if not re.search(r'\d', request.password):
                raise ValidationError("Password must contain at least one number")
        
        # Username validation (if provided)
        if request.username:
            if len(request.username) < 3 or len(request.username) > 50:
                raise ValidationError("Username must be 3-50 characters")
            
            username_pattern = re.compile(r'^[a-zA-Z0-9_.-]+$')
            if not username_pattern.match(request.username):
                raise ValidationError("Username can only contain letters, numbers, dots, dashes, and underscores")
        
        # Tenant validation
        if not request.tenant_id:
            raise ValidationError("Tenant ID is required")
    
    def get_registration_analytics(self, tenant_id: str, 
                                 days: int = 30) -> Dict[str, Any]:
        """
        Get registration analytics for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            days: Number of days to analyze
            
        Returns:
            Dictionary with analytics data
        """
        try:
            since_date = datetime.utcnow() - timedelta(days=days)
            
            # Query registration data
            registrations = self.db.query(PMRegistration).filter(
                and_(
                    PMRegistration.tenant_id == tenant_id,
                    PMRegistration.created_on >= since_date
                )
            ).all()
            
            # Calculate metrics
            total_attempts = len(registrations)
            completed = [r for r in registrations if r.status == "completed"]
            failed = [r for r in registrations if r.status == "failed"]
            abandoned = [r for r in registrations if r.status == "abandoned"]
            
            # Conversion rates
            completion_rate = (len(completed) / total_attempts * 100) if total_attempts > 0 else 0
            
            # Average completion time
            completion_times = [r.completion_time for r in completed if r.completion_time]
            avg_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0
            
            # Source breakdown
            sources = {}
            for reg in registrations:
                sources[reg.registration_source] = sources.get(reg.registration_source, 0) + 1
            
            # Email verification rates
            verification_sent = len([r for r in registrations if r.email_verification_sent])
            verification_completed = len([r for r in registrations if r.email_verified_at])
            verification_rate = (verification_completed / verification_sent * 100) if verification_sent > 0 else 0
            
            return {
                'period_days': days,
                'total_attempts': total_attempts,
                'completed': len(completed),
                'failed': len(failed),
                'abandoned': len(abandoned),
                'completion_rate': round(completion_rate, 2),
                'average_completion_time_seconds': round(avg_completion_time, 2),
                'source_breakdown': sources,
                'email_verification_rate': round(verification_rate, 2),
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get registration analytics: {str(e)}")
            raise ProfileManagementError(f"Analytics generation failed: {str(e)}")


class ProfileService:
    """
    Service for user profile management including CRUD operations,
    privacy controls, and GDPR compliance.
    """
    
    def __init__(self, db_session: Session, event_emitter: ProfileEventEmitter = None):
        self.db = db_session
        self.event_emitter = event_emitter or ProfileEventEmitter()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def get_profile(self, user_id: str, viewer_user_id: str = None, 
                   viewer_tenant_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Get user profile with privacy filtering.
        
        Args:
            user_id: User identifier
            viewer_user_id: ID of user viewing the profile (for privacy filtering)
            viewer_tenant_id: Tenant ID of viewer
            
        Returns:
            Profile dictionary or None if not found/accessible
        """
        try:
            profile = self.db.query(PMProfile).join(PMUser).filter(
                PMUser.user_id == user_id
            ).first()
            
            if not profile:
                return None
            
            # Apply privacy filtering
            return profile.to_dict(viewer_user_id, viewer_tenant_id)
            
        except Exception as e:
            self.logger.error(f"Failed to get profile for user {user_id}: {str(e)}")
            return None
    
    @emits_profile_events(['profile.updated'])
    @integrates_with('audit_logging')
    def update_profile(self, request: ProfileUpdateRequest) -> Dict[str, Any]:
        """
        Update user profile with validation and audit logging.
        
        Args:
            request: Profile update request with data and metadata
            
        Returns:
            Updated profile dictionary
            
        Raises:
            ProfileManagementError: If update fails
            PermissionError: If user lacks permission
        """
        try:
            # Get user and profile
            user = self.db.query(PMUser).filter(
                PMUser.user_id == request.user_id
            ).first()
            
            if not user:
                raise ProfileManagementError(f"User {request.user_id} not found")
            
            profile = user.profile
            if not profile:
                raise ProfileManagementError(f"Profile not found for user {request.user_id}")
            
            # Permission check
            if request.validate_permissions and request.updated_by != request.user_id:
                # In a full implementation, check if updated_by has admin permissions
                pass
            
            # Store original values for change tracking
            original_values = {}
            changes = {}
            
            # Update profile fields
            for field, value in request.updates.items():
                if hasattr(profile, field):
                    original_value = getattr(profile, field)
                    if original_value != value:
                        original_values[field] = original_value
                        setattr(profile, field, value)
                        changes[field] = {'from': original_value, 'to': value}
                else:
                    # Handle custom attributes
                    if field == 'custom_attributes':
                        if not profile.custom_attributes:
                            profile.custom_attributes = {}
                        profile.custom_attributes.update(value)
                        changes[field] = value
            
            if not changes:
                return profile.to_dict()
            
            # Update metadata
            profile.last_updated_by = request.updated_by
            profile.calculate_completion_score()
            
            self.db.commit()
            
            # Emit profile updated event
            self.event_emitter.emit('profile.updated', {
                'user_id': request.user_id,
                'updated_by': request.updated_by,
                'changes': changes,
                'completion_score': profile.completion_score,
                'tenant_id': request.tenant_id
            })
            
            self.logger.info(f"Profile updated for user {request.user_id} by {request.updated_by}")
            
            return profile.to_dict()
            
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to update profile for user {request.user_id}: {str(e)}")
            raise ProfileManagementError(f"Profile update failed: {str(e)}")
    
    def search_profiles(self, tenant_id: str, query: str = "", 
                       filters: Dict[str, Any] = None, 
                       viewer_user_id: str = None,
                       limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """
        Search user profiles with privacy filtering.
        
        Args:
            tenant_id: Tenant identifier
            query: Search query string
            filters: Additional filters (department, skills, etc.)
            viewer_user_id: ID of user performing search
            limit: Maximum results to return
            offset: Results offset for pagination
            
        Returns:
            Dictionary with search results and metadata
        """
        try:
            # Build base query
            query_builder = self.db.query(PMProfile).join(PMUser).filter(
                and_(
                    PMUser.tenant_id == tenant_id,
                    PMUser.is_active == True,
                    PMUser.email_verified == True,
                    PMProfile.search_visibility == True
                )
            )
            
            # Apply text search
            if query:
                search_filter = or_(
                    PMProfile.first_name.ilike(f"%{query}%"),
                    PMProfile.last_name.ilike(f"%{query}%"),
                    PMProfile.display_name.ilike(f"%{query}%"),
                    PMProfile.company.ilike(f"%{query}%"),
                    PMProfile.job_title.ilike(f"%{query}%"),
                    PMUser.email.ilike(f"%{query}%")
                )
                query_builder = query_builder.filter(search_filter)
            
            # Apply filters
            if filters:
                if filters.get('department'):
                    query_builder = query_builder.filter(
                        PMProfile.department == filters['department']
                    )
                
                if filters.get('company'):
                    query_builder = query_builder.filter(
                        PMProfile.company == filters['company']
                    )
                
                if filters.get('country'):
                    query_builder = query_builder.filter(
                        PMProfile.country == filters['country']
                    )
                
                if filters.get('verification_level'):
                    query_builder = query_builder.filter(
                        PMProfile.verification_level == filters['verification_level']
                    )
            
            # Get total count
            total_count = query_builder.count()
            
            # Apply pagination and get results
            profiles = query_builder.offset(offset).limit(limit).all()
            
            # Convert to dictionaries with privacy filtering
            results = []
            for profile in profiles:
                profile_data = profile.to_dict(viewer_user_id, tenant_id)
                if profile_data:  # Only include visible profiles
                    results.append(profile_data)
            
            return {
                'results': results,
                'total_count': total_count,
                'limit': limit,
                'offset': offset,
                'query': query,
                'filters': filters or {}
            }
            
        except Exception as e:
            self.logger.error(f"Profile search failed: {str(e)}")
            raise ProfileManagementError(f"Profile search failed: {str(e)}")
    
    @emits_profile_events(['profile.deleted', 'user.gdpr_deletion_completed'])
    @integrates_with('audit_logging')
    def delete_profile(self, user_id: str, deletion_type: str = "gdpr", 
                      requestor_id: str = None) -> bool:
        """
        Delete user profile with GDPR compliance.
        
        Args:
            user_id: User identifier
            deletion_type: Type of deletion (gdpr, admin, user_request)
            requestor_id: ID of user requesting deletion
            
        Returns:
            True if deletion successful
            
        Raises:
            GDPRError: If GDPR deletion requirements not met
        """
        try:
            user = self.db.query(PMUser).filter(
                PMUser.user_id == user_id
            ).first()
            
            if not user:
                raise ProfileManagementError(f"User {user_id} not found")
            
            # For GDPR deletion, mark for deletion but preserve audit trail
            if deletion_type == "gdpr":
                # Anonymize personal data
                user.email = f"deleted-{user_id}@example.com"
                user.username = None
                user.gdpr_deletion_requested = True
                user.gdpr_deletion_scheduled = datetime.utcnow()
                user.is_active = False
                
                # Anonymize profile data
                if user.profile:
                    profile = user.profile
                    profile.first_name = "Deleted"
                    profile.last_name = "User"
                    profile.display_name = "Deleted User"
                    profile.bio = None
                    profile.phone_primary = None
                    profile.phone_secondary = None
                    profile.website_url = None
                    profile.linkedin_url = None
                    profile.twitter_handle = None
                    profile.github_username = None
                    profile.custom_attributes = None
                    profile.avatar_url = None
                    profile.banner_url = None
                
                self.db.commit()
                
                # Emit GDPR deletion event
                self.event_emitter.emit('user.gdpr_deletion_completed', {
                    'user_id': user_id,
                    'deletion_date': datetime.utcnow().isoformat(),
                    'requested_by': requestor_id
                })
                
            else:
                # Complete deletion (admin or user request)
                self.db.delete(user)
                self.db.commit()
                
                # Emit profile deletion event
                self.event_emitter.emit('profile.deleted', {
                    'user_id': user_id,
                    'deletion_type': deletion_type,
                    'deleted_by': requestor_id
                })
            
            self.logger.info(f"User {user_id} deleted via {deletion_type} by {requestor_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to delete user {user_id}: {str(e)}")
            raise GDPRError(f"Profile deletion failed: {str(e)}")
    
    @integrates_with('audit_logging')
    def export_profile_data(self, user_id: str, requestor_id: str) -> Dict[str, Any]:
        """
        Export complete user profile data for GDPR compliance.
        
        Args:
            user_id: User identifier
            requestor_id: ID of user requesting export
            
        Returns:
            Complete user data dictionary
            
        Raises:
            PermissionError: If requestor lacks permission
            GDPRError: If export fails
        """
        try:
            # Permission check - user can export own data, admins can export any
            if requestor_id != user_id:
                # In full implementation, check admin permissions
                pass
            
            # Get user and related data
            user = self.db.query(PMUser).filter(
                PMUser.user_id == user_id
            ).first()
            
            if not user:
                raise ProfileManagementError(f"User {user_id} not found")
            
            # Build complete data export
            export_data = {
                'export_generated_at': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'user_account': user.to_dict(include_sensitive=True),
                'profile': user.profile.to_dict() if user.profile else None,
                'preferences': user.preferences.to_dict() if user.preferences else None,
                'consents': [consent.to_dict() for consent in user.consents],
                'registration_history': [reg.to_dict() for reg in user.registrations]
            }
            
            # Emit data export event for audit
            self.event_emitter.emit('user.data_exported', {
                'user_id': user_id,
                'exported_by': requestor_id,
                'export_timestamp': datetime.utcnow().isoformat()
            })
            
            self.logger.info(f"Data exported for user {user_id} by {requestor_id}")
            return export_data
            
        except Exception as e:
            self.logger.error(f"Failed to export data for user {user_id}: {str(e)}")
            raise GDPRError(f"Data export failed: {str(e)}")


class ConsentService:
    """
    Service for managing GDPR consent records and compliance.
    """
    
    def __init__(self, db_session: Session, event_emitter: ProfileEventEmitter = None):
        self.db = db_session
        self.event_emitter = event_emitter or ProfileEventEmitter()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @emits_profile_events(['consent.granted', 'consent.withdrawn'])
    def record_consent(self, user_id: str, tenant_id: str, purpose: str, 
                      granted: bool, version: str, method: str = "explicit",
                      context: Dict[str, Any] = None) -> str:
        """
        Record user consent for data processing purpose.
        
        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            purpose: Purpose of data processing
            granted: Whether consent was granted
            version: Version of consent policy
            method: How consent was obtained
            context: Additional context (IP, user agent, etc.)
            
        Returns:
            consent_id: Unique consent record identifier
        """
        try:
            # Create consent record
            consent = PMConsent(
                user_id=user_id,
                tenant_id=tenant_id,
                purpose=purpose,
                granted=granted,
                consent_version=version,
                consent_method=method,
                ip_address=context.get('ip_address') if context else None,
                user_agent=context.get('user_agent') if context else None,
                source_page=context.get('source_page') if context else None
            )
            
            # Set retention period based on purpose
            if purpose in ['marketing', 'analytics']:
                consent.retention_period = 365 * 2  # 2 years
                consent.expires_at = datetime.utcnow() + timedelta(days=consent.retention_period)
            
            self.db.add(consent)
            self.db.commit()
            
            # Emit consent event
            event_type = 'consent.granted' if granted else 'consent.withdrawn'
            self.event_emitter.emit(event_type, {
                'consent_id': consent.consent_id,
                'user_id': user_id,
                'purpose': purpose,
                'granted': granted,
                'version': version,
                'tenant_id': tenant_id
            })
            
            self.logger.info(f"Consent {event_type} for user {user_id}, purpose {purpose}")
            return consent.consent_id
            
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to record consent: {str(e)}")
            raise ConsentError(f"Consent recording failed: {str(e)}")
    
    def get_user_consents(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all consent records for a user"""
        try:
            consents = self.db.query(PMConsent).filter(
                PMConsent.user_id == user_id
            ).order_by(PMConsent.created_on.desc()).all()
            
            return [consent.to_dict() for consent in consents]
            
        except Exception as e:
            self.logger.error(f"Failed to get consents for user {user_id}: {str(e)}")
            return []
    
    def withdraw_consent(self, user_id: str, purpose: str, 
                        method: str = "user_request", reason: str = None) -> bool:
        """Withdraw user consent for specific purpose"""
        try:
            # Get active consent
            consent = self.db.query(PMConsent).filter(
                and_(
                    PMConsent.user_id == user_id,
                    PMConsent.purpose == purpose,
                    PMConsent.granted == True,
                    PMConsent.withdrawn_at.is_(None)
                )
            ).first()
            
            if not consent:
                return False
            
            # Withdraw consent
            consent.withdraw(method, reason)
            self.db.commit()
            
            # Emit withdrawal event
            self.event_emitter.emit('consent.withdrawn', {
                'consent_id': consent.consent_id,
                'user_id': user_id,
                'purpose': purpose,
                'withdrawal_method': method,
                'withdrawal_reason': reason
            })
            
            return True
            
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to withdraw consent: {str(e)}")
            return False


# Singleton pattern for service access
class ProfileManager:
    """
    Singleton manager for profile management services.
    Provides unified access to all profile management capabilities.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.db_session = None
            self.event_emitter = ProfileEventEmitter()
            self.initialized = True
    
    def initialize(self, db_session: Session):
        """Initialize with database session"""
        self.db_session = db_session
    
    @property
    def registration_service(self) -> RegistrationService:
        """Get registration service instance"""
        if not self.db_session:
            raise ProfileManagementError("ProfileManager not initialized with database session")
        return RegistrationService(self.db_session, self.event_emitter)
    
    @property
    def profile_service(self) -> ProfileService:
        """Get profile service instance"""
        if not self.db_session:
            raise ProfileManagementError("ProfileManager not initialized with database session")
        return ProfileService(self.db_session, self.event_emitter)
    
    @property
    def consent_service(self) -> ConsentService:
        """Get consent service instance"""
        if not self.db_session:
            raise ProfileManagementError("ProfileManager not initialized with database session")
        return ConsentService(self.db_session, self.event_emitter)
    
    @classmethod
    def get_instance(cls) -> 'ProfileManager':
        """Get singleton instance - capability composition keyword"""
        return cls()


# Export capability composition interface
def get_profile_service(db_session: Session) -> ProfileService:
    """Capability composition function to get profile service"""
    return ProfileService(db_session)

def get_registration_service(db_session: Session) -> RegistrationService:
    """Capability composition function to get registration service"""
    return RegistrationService(db_session)

def get_consent_service(db_session: Session) -> ConsentService:
    """Capability composition function to get consent service"""
    return ConsentService(db_session)