"""
Profile Management Exceptions

Custom exception classes for profile management capability with detailed
error information and proper categorization for different error types.
"""

class ProfileManagementError(Exception):
    """Base exception for all profile management errors"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or 'PROFILE_MANAGEMENT_ERROR'
        self.details = details or {}
    
    def to_dict(self):
        return {
            'error': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details
        }


class RegistrationError(ProfileManagementError):
    """Errors related to user registration process"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message, error_code or 'REGISTRATION_ERROR', details)


class ValidationError(ProfileManagementError):
    """Errors related to data validation"""
    
    def __init__(self, message: str, field: str = None, error_code: str = None, details: dict = None):
        super().__init__(message, error_code or 'VALIDATION_ERROR', details)
        self.field = field
        if field:
            self.details['field'] = field


class ConsentError(ProfileManagementError):
    """Errors related to GDPR consent management"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message, error_code or 'CONSENT_ERROR', details)


class PermissionError(ProfileManagementError):
    """Errors related to insufficient permissions"""
    
    def __init__(self, message: str, required_permission: str = None, error_code: str = None, details: dict = None):
        super().__init__(message, error_code or 'PERMISSION_ERROR', details)
        self.required_permission = required_permission
        if required_permission:
            self.details['required_permission'] = required_permission


class GDPRError(ProfileManagementError):
    """Errors related to GDPR compliance operations"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message, error_code or 'GDPR_ERROR', details)


class TenantError(ProfileManagementError):
    """Errors related to multi-tenant operations"""
    
    def __init__(self, message: str, tenant_id: str = None, error_code: str = None, details: dict = None):
        super().__init__(message, error_code or 'TENANT_ERROR', details)
        self.tenant_id = tenant_id
        if tenant_id:
            self.details['tenant_id'] = tenant_id