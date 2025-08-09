"""
Profile Management Database Models

All models use the PM prefix to identify Profile Management capability components.
Models are designed for multi-tenant architecture with GDPR compliance.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import sqlalchemy as sa
from sqlalchemy import Column, String, Boolean, DateTime, Text, JSON, Integer, ForeignKey
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.declarative import declarative_base
from flask_appbuilder import Model
from flask_appbuilder.models.mixins import AuditMixin, BaseMixin
import bcrypt
import re

Base = declarative_base()

class PMUser(Model, AuditMixin, BaseMixin):
	"""
	Primary user account model with authentication and core identity information.
	
	This model represents the core user entity with authentication credentials,
	verification status, and basic account information. Designed for multi-tenant
	architecture with tenant isolation.
	"""
	
	__tablename__ = 'pm_user'
	
	# Primary identification
	id = Column(Integer, primary_key=True)
	user_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()),
					index=True, comment="Unique user identifier across all tenants")
	tenant_id = Column(String(36), nullable=False, index=True,
					  comment="Tenant identifier for multi-tenant isolation")
	
	# Authentication credentials
	email = Column(String(255), nullable=False, index=True,
				  comment="Primary email address for authentication")
	username = Column(String(100), nullable=True, index=True,
					 comment="Optional username for authentication")
	password_hash = Column(String(255), nullable=True,
						  comment="Bcrypt hashed password")
	
	# Account status and verification
	is_active = Column(Boolean, default=True, nullable=False,
					  comment="Account active status")
	email_verified = Column(Boolean, default=False, nullable=False,
						   comment="Email verification status")
	email_verification_token = Column(String(255), nullable=True,
									 comment="Token for email verification")
	email_verification_expires = Column(DateTime, nullable=True,
									   comment="Email verification token expiration")
	phone_number = Column(String(20), nullable=True,
						 comment="Phone number for SMS verification")
	phone_verified = Column(Boolean, default=False, nullable=False,
						   comment="Phone verification status")
	
	# Account security and compliance
	password_changed_at = Column(DateTime, nullable=True,
								comment="Last password change timestamp")
	failed_login_attempts = Column(Integer, default=0, nullable=False,
								  comment="Failed login attempt counter")
	account_locked_until = Column(DateTime, nullable=True,
								 comment="Account lockout expiration")
	last_login_at = Column(DateTime, nullable=True,
						  comment="Last successful login timestamp")
	last_login_ip = Column(String(45), nullable=True,
						  comment="IP address of last login")
	
	# Registration and lifecycle
	registration_source = Column(String(50), default='email', nullable=False,
								comment="Registration method (email, oauth, sso, invitation)")
	registration_data = Column(JSON, nullable=True,
							  comment="Additional registration metadata")
	terms_accepted_at = Column(DateTime, nullable=True,
							  comment="Terms and conditions acceptance timestamp")
	terms_version = Column(String(20), nullable=True,
						  comment="Version of accepted terms")
	
	# GDPR and privacy
	data_processing_consent = Column(Boolean, default=False, nullable=False,
									comment="Consent for data processing")
	marketing_consent = Column(Boolean, default=False, nullable=False,
							  comment="Consent for marketing communications")
	analytics_consent = Column(Boolean, default=False, nullable=False,
							 comment="Consent for analytics tracking")
	gdpr_deletion_requested = Column(Boolean, default=False, nullable=False,
									comment="GDPR deletion request flag")
	gdpr_deletion_scheduled = Column(DateTime, nullable=True,
									comment="Scheduled deletion timestamp")
	
	# Relationships
	profile = relationship("PMProfile", back_populates="user", uselist=False,
						  cascade="all, delete-orphan")
	consents = relationship("PMConsent", back_populates="user",
						   cascade="all, delete-orphan")
	preferences = relationship("PMPreferences", back_populates="user", uselist=False,
							  cascade="all, delete-orphan")
	registrations = relationship("PMRegistration", back_populates="user",
								cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<PMUser(user_id='{self.user_id}', email='{self.email}')>"
	
	@validates('email')
	def validate_email(self, key, email):
		"""Validate email format"""
		if not email:
			raise ValueError("Email is required")
		
		email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
		if not email_pattern.match(email):
			raise ValueError("Invalid email format")
		
		return email.lower().strip()
	
	@validates('username')
	def validate_username(self, key, username):
		"""Validate username format"""
		if username:
			if len(username) < 3 or len(username) > 50:
				raise ValueError("Username must be 3-50 characters")
			
			username_pattern = re.compile(r'^[a-zA-Z0-9_.-]+$')
			if not username_pattern.match(username):
				raise ValueError("Username can only contain letters, numbers, dots, dashes, and underscores")
		
		return username
	
	def set_password(self, password: str) -> None:
		"""Set password with bcrypt hashing"""
		if not password or len(password) < 8:
			raise ValueError("Password must be at least 8 characters")
		
		# Generate salt and hash password
		salt = bcrypt.gensalt(rounds=12)
		self.password_hash = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
		self.password_changed_at = datetime.utcnow()
	
	def check_password(self, password: str) -> bool:
		"""Verify password against stored hash"""
		if not self.password_hash:
			return False
		
		return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))
	
	def generate_verification_token(self) -> str:
		"""Generate email verification token"""
		token = str(uuid.uuid4())
		self.email_verification_token = token
		self.email_verification_expires = datetime.utcnow() + timedelta(hours=24)
		return token
	
	def is_verification_token_valid(self, token: str) -> bool:
		"""Check if verification token is valid"""
		return (self.email_verification_token == token and
				self.email_verification_expires and
				self.email_verification_expires > datetime.utcnow())
	
	def increment_failed_login(self) -> None:
		"""Increment failed login attempts and lock account if necessary"""
		self.failed_login_attempts += 1
		
		# Lock account after 5 failed attempts for 30 minutes
		if self.failed_login_attempts >= 5:
			self.account_locked_until = datetime.utcnow() + timedelta(minutes=30)
	
	def reset_failed_login(self) -> None:
		"""Reset failed login attempts after successful login"""
		self.failed_login_attempts = 0
		self.account_locked_until = None
		self.last_login_at = datetime.utcnow()
	
	def is_account_locked(self) -> bool:
		"""Check if account is currently locked"""
		return (self.account_locked_until is not None and
				self.account_locked_until > datetime.utcnow())
	
	def can_login(self) -> bool:
		"""Check if user can login (active, verified, not locked)"""
		return (self.is_active and 
				self.email_verified and 
				not self.is_account_locked() and
				not self.gdpr_deletion_requested)
	
	def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
		"""Convert user to dictionary for API responses"""
		data = {
			'user_id': self.user_id,
			'email': self.email,
			'username': self.username,
			'is_active': self.is_active,
			'email_verified': self.email_verified,
			'phone_verified': self.phone_verified,
			'registration_source': self.registration_source,
			'created_on': self.created_on.isoformat() if self.created_on else None,
			'changed_on': self.changed_on.isoformat() if self.changed_on else None,
			'last_login_at': self.last_login_at.isoformat() if self.last_login_at else None
		}
		
		if include_sensitive:
			data.update({
				'phone_number': self.phone_number,
				'failed_login_attempts': self.failed_login_attempts,
				'account_locked_until': self.account_locked_until.isoformat() if self.account_locked_until else None,
				'terms_accepted_at': self.terms_accepted_at.isoformat() if self.terms_accepted_at else None,
				'terms_version': self.terms_version
			})
		
		return data


class PMProfile(Model, AuditMixin, BaseMixin):
	"""
	Extended user profile information including personal, professional, and preference data.
	
	Stores comprehensive user profile information with privacy controls and customizable
	fields. Supports multi-tenant customization and GDPR compliance.
	"""
	
	__tablename__ = 'pm_profile'
	
	# Primary identification
	id = Column(Integer, primary_key=True)
	profile_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()),
					   index=True, comment="Unique profile identifier")
	user_id = Column(String(36), ForeignKey('pm_user.user_id'), nullable=False,
					index=True, comment="Reference to associated user")
	tenant_id = Column(String(36), nullable=False, index=True,
					  comment="Tenant identifier for multi-tenant isolation")
	
	# Personal information
	first_name = Column(String(100), nullable=True,
					   comment="User's first name")
	last_name = Column(String(100), nullable=True,
					  comment="User's last name")
	display_name = Column(String(200), nullable=True,
						 comment="Preferred display name")
	title = Column(String(100), nullable=True,
				  comment="Professional title or honorific")
	bio = Column(Text, nullable=True,
				comment="User biography or description")
	
	# Avatar and media
	avatar_url = Column(String(500), nullable=True,
					   comment="URL to user's profile picture")
	avatar_updated_at = Column(DateTime, nullable=True,
							  comment="Avatar last update timestamp")
	banner_url = Column(String(500), nullable=True,
					   comment="URL to user's profile banner")
	
	# Contact information
	phone_primary = Column(String(20), nullable=True,
						  comment="Primary phone number")
	phone_secondary = Column(String(20), nullable=True,
							comment="Secondary phone number")
	website_url = Column(String(500), nullable=True,
						comment="Personal website URL")
	
	# Professional information
	company = Column(String(200), nullable=True,
					comment="Current company or organization")
	department = Column(String(100), nullable=True,
					   comment="Department within organization")
	job_title = Column(String(200), nullable=True,
					  comment="Current job title")
	manager_user_id = Column(String(36), nullable=True,
							comment="User ID of manager (if applicable)")
	employee_id = Column(String(50), nullable=True,
						comment="Employee ID within organization")
	hire_date = Column(DateTime, nullable=True,
					  comment="Employment start date")
	
	# Location information
	timezone = Column(String(50), default='UTC', nullable=False,
					 comment="User's timezone preference")
	locale = Column(String(10), default='en-US', nullable=False,
				   comment="User's locale preference")
	country = Column(String(2), nullable=True,
					comment="Country code (ISO 3166-1 alpha-2)")
	region = Column(String(100), nullable=True,
				   comment="State, province, or region")
	city = Column(String(100), nullable=True,
				 comment="City of residence")
	
	# Social links
	linkedin_url = Column(String(500), nullable=True,
						 comment="LinkedIn profile URL")
	twitter_handle = Column(String(100), nullable=True,
						   comment="Twitter username (without @)")
	github_username = Column(String(100), nullable=True,
							comment="GitHub username")
	
	# Custom attributes and metadata
	custom_attributes = Column(JSON, nullable=True,
							  comment="Tenant-specific custom profile fields")
	skills = Column(JSON, nullable=True,
				   comment="List of skills and expertise areas")
	languages = Column(JSON, nullable=True,
					  comment="Spoken languages with proficiency levels")
	
	# Privacy and visibility settings
	profile_visibility = Column(String(20), default='public', nullable=False,
							   comment="Profile visibility: public, organization, private")
	search_visibility = Column(Boolean, default=True, nullable=False,
							  comment="Include profile in search results")
	activity_visibility = Column(Boolean, default=True, nullable=False,
								comment="Show user activity and presence")
	contact_visibility = Column(String(20), default='organization', nullable=False,
							   comment="Contact information visibility level")
	
	# Profile completion and quality
	completion_score = Column(Integer, default=0, nullable=False,
							 comment="Profile completion percentage (0-100)")
	last_updated_by = Column(String(36), nullable=True,
							comment="User ID who last updated profile")
	profile_verified = Column(Boolean, default=False, nullable=False,
							 comment="Profile verification status")
	verification_level = Column(String(20), default='none', nullable=False,
							   comment="Verification level: none, email, phone, document, full")
	
	# Relationships
	user = relationship("PMUser", back_populates="profile")
	
	def __repr__(self):
		return f"<PMProfile(profile_id='{self.profile_id}', display_name='{self.display_name}')>"
	
	@validates('timezone')
	def validate_timezone(self, key, timezone):
		"""Validate timezone format"""
		# Basic timezone validation - in production, use pytz for full validation
		if timezone and len(timezone) > 50:
			raise ValueError("Timezone identifier too long")
		return timezone or 'UTC'
	
	@validates('locale')
	def validate_locale(self, key, locale):
		"""Validate locale format"""
		if locale:
			locale_pattern = re.compile(r'^[a-z]{2}(-[A-Z]{2})?$')
			if not locale_pattern.match(locale):
				raise ValueError("Invalid locale format (expected: en-US)")
		return locale or 'en-US'
	
	@validates('profile_visibility')
	def validate_profile_visibility(self, key, visibility):
		"""Validate profile visibility setting"""
		allowed_values = ['public', 'organization', 'private']
		if visibility not in allowed_values:
			raise ValueError(f"Profile visibility must be one of: {', '.join(allowed_values)}")
		return visibility
	
	@validates('contact_visibility')
	def validate_contact_visibility(self, key, visibility):
		"""Validate contact visibility setting"""
		allowed_values = ['public', 'organization', 'private']
		if visibility not in allowed_values:
			raise ValueError(f"Contact visibility must be one of: {', '.join(allowed_values)}")
		return visibility
	
	def get_full_name(self) -> str:
		"""Get formatted full name"""
		parts = []
		if self.first_name:
			parts.append(self.first_name)
		if self.last_name:
			parts.append(self.last_name)
		return ' '.join(parts)
	
	def get_display_name(self) -> str:
		"""Get preferred display name with fallback"""
		if self.display_name:
			return self.display_name
		
		full_name = self.get_full_name()
		if full_name:
			return full_name
		
		# Fallback to user email or username
		if hasattr(self, 'user') and self.user:
			return self.user.username or self.user.email
		
		return "Unknown User"
	
	def calculate_completion_score(self) -> int:
		"""Calculate profile completion percentage"""
		total_fields = 20  # Total weighted fields
		completed_fields = 0
		
		# Core fields (weight: 2 points each)
		core_fields = [self.first_name, self.last_name]
		completed_fields += sum(2 for field in core_fields if field)
		
		# Important fields (weight: 1 point each)
		important_fields = [
			self.title, self.bio, self.avatar_url, self.company, 
			self.job_title, self.timezone, self.locale, self.phone_primary
		]
		completed_fields += sum(1 for field in important_fields if field)
		
		# Additional fields (weight: 0.5 points each)
		additional_fields = [
			self.website_url, self.linkedin_url, self.twitter_handle,
			self.github_username, self.country, self.city
		]
		completed_fields += sum(0.5 for field in additional_fields if field)
		
		# Custom attributes and skills
		if self.custom_attributes:
			completed_fields += min(2, len(self.custom_attributes) * 0.5)
		if self.skills:
			completed_fields += min(2, len(self.skills) * 0.5)
		
		score = min(100, int((completed_fields / total_fields) * 100))
		self.completion_score = score
		return score
	
	def is_visible_to_user(self, viewer_user_id: str, viewer_tenant_id: str) -> bool:
		"""Check if profile is visible to specified user"""
		# Profile owner can always see their own profile
		if self.user.user_id == viewer_user_id:
			return True
		
		# Check visibility settings
		if self.profile_visibility == 'public':
			return True
		elif self.profile_visibility == 'organization':
			return self.tenant_id == viewer_tenant_id
		else:  # private
			return False
	
	def get_public_fields(self) -> List[str]:
		"""Get list of fields visible based on privacy settings"""
		public_fields = ['display_name', 'title', 'company', 'avatar_url']
		
		if self.contact_visibility == 'public':
			public_fields.extend(['phone_primary', 'website_url', 'linkedin_url'])
		
		if self.profile_visibility == 'public':
			public_fields.extend(['bio', 'skills', 'languages', 'city', 'country'])
		
		return public_fields
	
	def to_dict(self, viewer_user_id: str = None, viewer_tenant_id: str = None) -> Dict[str, Any]:
		"""Convert profile to dictionary with privacy filtering"""
		# Get base profile data
		data = {
			'profile_id': self.profile_id,
			'user_id': self.user_id,
			'display_name': self.get_display_name(),
			'avatar_url': self.avatar_url,
			'completion_score': self.completion_score,
			'profile_verified': self.profile_verified,
			'verification_level': self.verification_level,
			'created_on': self.created_on.isoformat() if self.created_on else None,
			'changed_on': self.changed_on.isoformat() if self.changed_on else None
		}
		
		# Apply privacy filtering if viewer is specified
		if viewer_user_id is not None and viewer_tenant_id is not None:
			if not self.is_visible_to_user(viewer_user_id, viewer_tenant_id):
				# Return minimal public data for private profiles
				return {
					'profile_id': self.profile_id,
					'display_name': self.get_display_name(),
					'avatar_url': self.avatar_url if self.profile_visibility != 'private' else None
				}
			
			# Add fields based on visibility settings
			public_fields = self.get_public_fields()
			
			if 'bio' in public_fields:
				data['bio'] = self.bio
			if 'title' in public_fields:
				data['title'] = self.title
			if 'company' in public_fields:
				data['company'] = self.company
			if 'job_title' in public_fields:
				data['job_title'] = self.job_title
			if 'skills' in public_fields and self.skills:
				data['skills'] = self.skills
			if 'languages' in public_fields and self.languages:
				data['languages'] = self.languages
			if 'city' in public_fields:
				data['city'] = self.city
			if 'country' in public_fields:
				data['country'] = self.country
			
			# Contact information based on contact visibility
			if self.contact_visibility in ['public', 'organization']:
				if 'phone_primary' in public_fields:
					data['phone_primary'] = self.phone_primary
				if 'website_url' in public_fields:
					data['website_url'] = self.website_url
				if 'linkedin_url' in public_fields:
					data['linkedin_url'] = self.linkedin_url
		else:
			# Full profile data (for owner or admin access)
			data.update({
				'first_name': self.first_name,
				'last_name': self.last_name,
				'title': self.title,
				'bio': self.bio,
				'phone_primary': self.phone_primary,
				'phone_secondary': self.phone_secondary,
				'website_url': self.website_url,
				'company': self.company,
				'department': self.department,
				'job_title': self.job_title,
				'timezone': self.timezone,
				'locale': self.locale,
				'country': self.country,
				'region': self.region,
				'city': self.city,
				'linkedin_url': self.linkedin_url,
				'twitter_handle': self.twitter_handle,
				'github_username': self.github_username,
				'custom_attributes': self.custom_attributes,
				'skills': self.skills,
				'languages': self.languages,
				'profile_visibility': self.profile_visibility,
				'search_visibility': self.search_visibility,
				'activity_visibility': self.activity_visibility,
				'contact_visibility': self.contact_visibility
			})
		
		return data


class PMConsent(Model, AuditMixin, BaseMixin):
	"""
	GDPR consent tracking for various data processing purposes.
	
	Tracks user consent for different types of data processing activities
	with versioning and historical record keeping for compliance.
	"""
	
	__tablename__ = 'pm_consent'
	
	# Primary identification
	id = Column(Integer, primary_key=True)
	consent_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()),
					   index=True, comment="Unique consent record identifier")
	user_id = Column(String(36), ForeignKey('pm_user.user_id'), nullable=False,
					index=True, comment="Reference to associated user")
	tenant_id = Column(String(36), nullable=False, index=True,
					  comment="Tenant identifier for multi-tenant isolation")
	
	# Consent details
	purpose = Column(String(100), nullable=False,
					comment="Purpose of data processing (marketing, analytics, etc.)")
	purpose_description = Column(Text, nullable=True,
								comment="Detailed description of data processing purpose")
	granted = Column(Boolean, nullable=False,
					comment="Whether consent was granted (True) or denied (False)")
	
	# Consent metadata
	consent_method = Column(String(50), default='explicit', nullable=False,
						   comment="How consent was obtained (explicit, implied, required)")
	consent_version = Column(String(20), nullable=False,
							comment="Version of consent policy when granted")
	consent_text = Column(Text, nullable=True,
						 comment="Exact consent text shown to user")
	
	# Legal basis and retention
	legal_basis = Column(String(50), default='consent', nullable=False,
						comment="GDPR legal basis for processing")
	retention_period = Column(Integer, nullable=True,
							 comment="Data retention period in days")
	expires_at = Column(DateTime, nullable=True,
					   comment="Consent expiration date")
	
	# Consent source and context
	ip_address = Column(String(45), nullable=True,
					   comment="IP address when consent was given")
	user_agent = Column(String(500), nullable=True,
					   comment="User agent when consent was given")
	source_page = Column(String(500), nullable=True,
						comment="Page where consent was obtained")
	campaign_id = Column(String(100), nullable=True,
						comment="Marketing campaign associated with consent")
	
	# Withdrawal information
	withdrawn_at = Column(DateTime, nullable=True,
						 comment="Timestamp when consent was withdrawn")
	withdrawal_method = Column(String(50), nullable=True,
							  comment="How consent was withdrawn")
	withdrawal_reason = Column(Text, nullable=True,
							  comment="Reason for consent withdrawal")
	
	# Relationships
	user = relationship("PMUser", back_populates="consents")
	
	def __repr__(self):
		return f"<PMConsent(consent_id='{self.consent_id}', purpose='{self.purpose}', granted={self.granted})>"
	
	@validates('purpose')
	def validate_purpose(self, key, purpose):
		"""Validate consent purpose"""
		allowed_purposes = [
			'marketing', 'analytics', 'personalization', 'communication',
			'functionality', 'security', 'legal', 'research', 'improvement'
		]
		if purpose not in allowed_purposes:
			raise ValueError(f"Invalid consent purpose. Must be one of: {', '.join(allowed_purposes)}")
		return purpose
	
	@validates('legal_basis')
	def validate_legal_basis(self, key, legal_basis):
		"""Validate GDPR legal basis"""
		allowed_bases = [
			'consent', 'contract', 'legal_obligation', 'vital_interests',
			'public_task', 'legitimate_interests'
		]
		if legal_basis not in allowed_bases:
			raise ValueError(f"Invalid legal basis. Must be one of: {', '.join(allowed_bases)}")
		return legal_basis
	
	def is_active(self) -> bool:
		"""Check if consent is currently active"""
		if not self.granted:
			return False
		
		if self.withdrawn_at:
			return False
		
		if self.expires_at and self.expires_at < datetime.utcnow():
			return False
		
		return True
	
	def withdraw(self, method: str = 'user_request', reason: str = None) -> None:
		"""Withdraw consent"""
		self.withdrawn_at = datetime.utcnow()
		self.withdrawal_method = method
		if reason:
			self.withdrawal_reason = reason
	
	def renew(self, version: str, text: str = None) -> None:
		"""Renew expired consent with new version"""
		if self.is_active():
			raise ValueError("Cannot renew active consent")
		
		self.granted = True
		self.consent_version = version
		if text:
			self.consent_text = text
		
		# Clear withdrawal information
		self.withdrawn_at = None
		self.withdrawal_method = None
		self.withdrawal_reason = None
		
		# Set new expiration if retention period is specified
		if self.retention_period:
			self.expires_at = datetime.utcnow() + timedelta(days=self.retention_period)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert consent to dictionary"""
		return {
			'consent_id': self.consent_id,
			'user_id': self.user_id,
			'purpose': self.purpose,
			'purpose_description': self.purpose_description,
			'granted': self.granted,
			'consent_method': self.consent_method,
			'consent_version': self.consent_version,
			'legal_basis': self.legal_basis,
			'is_active': self.is_active(),
			'created_on': self.created_on.isoformat() if self.created_on else None,
			'expires_at': self.expires_at.isoformat() if self.expires_at else None,
			'withdrawn_at': self.withdrawn_at.isoformat() if self.withdrawn_at else None,
			'withdrawal_method': self.withdrawal_method,
			'withdrawal_reason': self.withdrawal_reason
		}


class PMRegistration(Model, AuditMixin, BaseMixin):
	"""
	Registration attempt tracking and analytics.
	
	Tracks all registration attempts including successful and failed attempts
	for analytics, security monitoring, and user experience optimization.
	"""
	
	__tablename__ = 'pm_registration'
	
	# Primary identification
	id = Column(Integer, primary_key=True)
	registration_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()),
							index=True, comment="Unique registration attempt identifier")
	user_id = Column(String(36), ForeignKey('pm_user.user_id'), nullable=True,
					index=True, comment="Reference to created user (if successful)")
	tenant_id = Column(String(36), nullable=False, index=True,
					  comment="Tenant identifier for multi-tenant isolation")
	
	# Registration attempt details
	email = Column(String(255), nullable=False, index=True,
				  comment="Email address used for registration")
	registration_source = Column(String(50), nullable=False,
								comment="Registration method (email, oauth, sso, invitation)")
	registration_step = Column(String(50), nullable=False,
							  comment="Registration step where attempt ended")
	
	# Status and outcome
	status = Column(String(20), default='started', nullable=False,
				   comment="Registration status: started, completed, failed, abandoned")
	completion_time = Column(Integer, nullable=True,
							comment="Time to complete registration in seconds")
	failure_reason = Column(String(200), nullable=True,
						   comment="Reason for registration failure")
	
	# Technical context
	ip_address = Column(String(45), nullable=True,
					   comment="IP address of registration attempt")
	user_agent = Column(String(500), nullable=True,
					   comment="User agent of registration attempt")
	referrer_url = Column(String(500), nullable=True,
						 comment="Referrer URL for registration")
	utm_source = Column(String(100), nullable=True,
					   comment="UTM source parameter")
	utm_medium = Column(String(100), nullable=True,
					   comment="UTM medium parameter")
	utm_campaign = Column(String(100), nullable=True,
						 comment="UTM campaign parameter")
	
	# Registration data and preferences
	registration_data = Column(JSON, nullable=True,
							  comment="Additional registration form data")
	consent_given = Column(JSON, nullable=True,
						  comment="Consent preferences provided during registration")
	profile_data = Column(JSON, nullable=True,
						 comment="Profile information provided during registration")
	
	# A/B testing and experimentation
	experiment_variant = Column(String(50), nullable=True,
							   comment="A/B test variant for registration flow")
	feature_flags = Column(JSON, nullable=True,
						  comment="Feature flags active during registration")
	
	# Verification and security
	email_verification_sent = Column(Boolean, default=False, nullable=False,
									comment="Whether email verification was sent")
	email_verified_at = Column(DateTime, nullable=True,
							  comment="When email was verified")
	fraud_score = Column(Integer, nullable=True,
						comment="Fraud detection score (0-100)")
	security_flags = Column(JSON, nullable=True,
						   comment="Security flags and warnings")
	
	# Relationships
	user = relationship("PMUser", back_populates="registrations")
	
	def __repr__(self):
		return f"<PMRegistration(registration_id='{self.registration_id}', email='{self.email}', status='{self.status}')>"
	
	@validates('status')
	def validate_status(self, key, status):
		"""Validate registration status"""
		allowed_statuses = ['started', 'in_progress', 'completed', 'failed', 'abandoned']
		if status not in allowed_statuses:
			raise ValueError(f"Invalid status. Must be one of: {', '.join(allowed_statuses)}")
		return status
	
	@validates('registration_source')
	def validate_registration_source(self, key, source):
		"""Validate registration source"""
		allowed_sources = ['email', 'google', 'microsoft', 'linkedin', 'github', 'saml', 'invitation']
		if source not in allowed_sources:
			raise ValueError(f"Invalid registration source. Must be one of: {', '.join(allowed_sources)}")
		return source
	
	def mark_completed(self, user_id: str, completion_time: int = None) -> None:
		"""Mark registration as completed"""
		self.status = 'completed'
		self.user_id = user_id
		if completion_time:
			self.completion_time = completion_time
		elif self.created_on:
			self.completion_time = int((datetime.utcnow() - self.created_on).total_seconds())
	
	def mark_failed(self, reason: str) -> None:
		"""Mark registration as failed"""
		self.status = 'failed'
		self.failure_reason = reason
	
	def mark_abandoned(self, step: str) -> None:
		"""Mark registration as abandoned at specific step"""
		self.status = 'abandoned'
		self.registration_step = step
	
	def calculate_conversion_metrics(self) -> Dict[str, Any]:
		"""Calculate conversion metrics for this registration"""
		metrics = {
			'completed': self.status == 'completed',
			'completion_time_seconds': self.completion_time,
			'steps_completed': self.registration_step,
			'source': self.registration_source,
			'had_email_verification': self.email_verification_sent,
			'email_verified': self.email_verified_at is not None
		}
		
		if self.utm_source:
			metrics['attribution'] = {
				'source': self.utm_source,
				'medium': self.utm_medium,
				'campaign': self.utm_campaign
			}
		
		return metrics
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert registration to dictionary"""
		return {
			'registration_id': self.registration_id,
			'user_id': self.user_id,
			'email': self.email,
			'registration_source': self.registration_source,
			'registration_step': self.registration_step,
			'status': self.status,
			'completion_time': self.completion_time,
			'failure_reason': self.failure_reason,
			'ip_address': self.ip_address,
			'referrer_url': self.referrer_url,
			'utm_source': self.utm_source,
			'utm_medium': self.utm_medium,
			'utm_campaign': self.utm_campaign,
			'email_verification_sent': self.email_verification_sent,
			'email_verified_at': self.email_verified_at.isoformat() if self.email_verified_at else None,
			'fraud_score': self.fraud_score,
			'experiment_variant': self.experiment_variant,
			'created_on': self.created_on.isoformat() if self.created_on else None,
			'changed_on': self.changed_on.isoformat() if self.changed_on else None
		}


class PMPreferences(Model, AuditMixin, BaseMixin):
	"""
	User preferences and settings for personalization and user experience.
	
	Stores user preferences for notifications, privacy, UI customization,
	and other personalization settings with tenant isolation.
	"""
	
	__tablename__ = 'pm_preferences'
	
	# Primary identification
	id = Column(Integer, primary_key=True)
	preferences_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()),
						   index=True, comment="Unique preferences identifier")
	user_id = Column(String(36), ForeignKey('pm_user.user_id'), nullable=False,
					index=True, comment="Reference to associated user")
	tenant_id = Column(String(36), nullable=False, index=True,
					  comment="Tenant identifier for multi-tenant isolation")
	
	# Notification preferences
	email_notifications = Column(Boolean, default=True, nullable=False,
								comment="Enable email notifications")
	sms_notifications = Column(Boolean, default=False, nullable=False,
							   comment="Enable SMS notifications")
	push_notifications = Column(Boolean, default=True, nullable=False,
							   comment="Enable push notifications")
	marketing_emails = Column(Boolean, default=False, nullable=False,
							 comment="Enable marketing emails")
	newsletter_subscription = Column(Boolean, default=False, nullable=False,
									comment="Subscribe to newsletter")
	
	# Communication preferences
	notification_frequency = Column(String(20), default='immediate', nullable=False,
								   comment="Notification frequency: immediate, daily, weekly")
	digest_frequency = Column(String(20), default='weekly', nullable=False,
							 comment="Email digest frequency: daily, weekly, monthly, never")
	communication_language = Column(String(10), default='en', nullable=False,
								   comment="Preferred language for communications")
	
	# Privacy preferences
	profile_indexing = Column(Boolean, default=True, nullable=False,
							 comment="Allow profile to be indexed by search engines")
	activity_tracking = Column(Boolean, default=True, nullable=False,
							  comment="Allow activity tracking for analytics")
	personalized_content = Column(Boolean, default=True, nullable=False,
								 comment="Enable personalized content recommendations")
	data_sharing = Column(Boolean, default=False, nullable=False,
						 comment="Allow data sharing with partners")
	
	# UI and experience preferences
	theme = Column(String(20), default='auto', nullable=False,
				  comment="UI theme preference: light, dark, auto")
	density = Column(String(20), default='comfortable', nullable=False,
					comment="UI density: compact, comfortable, spacious")
	sidebar_collapsed = Column(Boolean, default=False, nullable=False,
							  comment="Sidebar collapsed by default")
	show_tooltips = Column(Boolean, default=True, nullable=False,
						  comment="Show UI tooltips and help text")
	
	# Accessibility preferences
	high_contrast = Column(Boolean, default=False, nullable=False,
						  comment="Enable high contrast mode")
	large_text = Column(Boolean, default=False, nullable=False,
					   comment="Enable large text size")
	reduced_motion = Column(Boolean, default=False, nullable=False,
						   comment="Reduce animations and motion")
	keyboard_navigation = Column(Boolean, default=False, nullable=False,
								comment="Optimize for keyboard navigation")
	
	# Feature preferences
	beta_features = Column(Boolean, default=False, nullable=False,
						  comment="Enable beta features and experiments")
	advanced_features = Column(Boolean, default=False, nullable=False,
							  comment="Show advanced features and options")
	feature_hints = Column(Boolean, default=True, nullable=False,
						  comment="Show feature hints and onboarding")
	
	# Custom preferences
	custom_preferences = Column(JSON, nullable=True,
							   comment="Tenant-specific custom preferences")
	
	# Dashboard and workspace preferences
	dashboard_layout = Column(JSON, nullable=True,
							 comment="Dashboard widget layout and configuration")
	default_workspace = Column(String(100), nullable=True,
							  comment="Default workspace or view")
	favorite_items = Column(JSON, nullable=True,
						   comment="User's favorite items, pages, or features")
	
	# Integration preferences
	calendar_integration = Column(Boolean, default=False, nullable=False,
								 comment="Enable calendar integration")
	task_management_integration = Column(Boolean, default=False, nullable=False,
										comment="Enable task management integration")
	social_sharing = Column(Boolean, default=False, nullable=False,
						   comment="Enable social media sharing")
	
	# Relationships
	user = relationship("PMUser", back_populates="preferences")
	
	def __repr__(self):
		return f"<PMPreferences(preferences_id='{self.preferences_id}', user_id='{self.user_id}')>"
	
	@validates('theme')
	def validate_theme(self, key, theme):
		"""Validate theme preference"""
		allowed_themes = ['light', 'dark', 'auto']
		if theme not in allowed_themes:
			raise ValueError(f"Invalid theme. Must be one of: {', '.join(allowed_themes)}")
		return theme
	
	@validates('density')
	def validate_density(self, key, density):
		"""Validate UI density preference"""
		allowed_densities = ['compact', 'comfortable', 'spacious']
		if density not in allowed_densities:
			raise ValueError(f"Invalid density. Must be one of: {', '.join(allowed_densities)}")
		return density
	
	@validates('notification_frequency')
	def validate_notification_frequency(self, key, frequency):
		"""Validate notification frequency"""
		allowed_frequencies = ['immediate', 'daily', 'weekly', 'never']
		if frequency not in allowed_frequencies:
			raise ValueError(f"Invalid frequency. Must be one of: {', '.join(allowed_frequencies)}")
		return frequency
	
	@validates('digest_frequency')
	def validate_digest_frequency(self, key, frequency):
		"""Validate digest frequency"""
		allowed_frequencies = ['daily', 'weekly', 'monthly', 'never']
		if frequency not in allowed_frequencies:
			raise ValueError(f"Invalid digest frequency. Must be one of: {', '.join(allowed_frequencies)}")
		return frequency
	
	def get_notification_settings(self) -> Dict[str, Any]:
		"""Get comprehensive notification settings"""
		return {
			'email_notifications': self.email_notifications,
			'sms_notifications': self.sms_notifications,
			'push_notifications': self.push_notifications,
			'marketing_emails': self.marketing_emails,
			'newsletter_subscription': self.newsletter_subscription,
			'notification_frequency': self.notification_frequency,
			'digest_frequency': self.digest_frequency,
			'communication_language': self.communication_language
		}
	
	def get_privacy_settings(self) -> Dict[str, Any]:
		"""Get comprehensive privacy settings"""
		return {
			'profile_indexing': self.profile_indexing,
			'activity_tracking': self.activity_tracking,
			'personalized_content': self.personalized_content,
			'data_sharing': self.data_sharing
		}
	
	def get_ui_settings(self) -> Dict[str, Any]:
		"""Get UI and accessibility settings"""
		return {
			'theme': self.theme,
			'density': self.density,
			'sidebar_collapsed': self.sidebar_collapsed,
			'show_tooltips': self.show_tooltips,
			'high_contrast': self.high_contrast,
			'large_text': self.large_text,
			'reduced_motion': self.reduced_motion,
			'keyboard_navigation': self.keyboard_navigation
		}
	
	def update_preferences(self, preferences: Dict[str, Any]) -> None:
		"""Update multiple preferences at once"""
		for key, value in preferences.items():
			if hasattr(self, key):
				setattr(self, key, value)
	
	def reset_to_defaults(self) -> None:
		"""Reset preferences to default values"""
		# Notification defaults
		self.email_notifications = True
		self.sms_notifications = False
		self.push_notifications = True
		self.marketing_emails = False
		self.newsletter_subscription = False
		self.notification_frequency = 'immediate'
		self.digest_frequency = 'weekly'
		
		# Privacy defaults
		self.profile_indexing = True
		self.activity_tracking = True
		self.personalized_content = True
		self.data_sharing = False
		
		# UI defaults
		self.theme = 'auto'
		self.density = 'comfortable'
		self.sidebar_collapsed = False
		self.show_tooltips = True
		
		# Accessibility defaults
		self.high_contrast = False
		self.large_text = False
		self.reduced_motion = False
		self.keyboard_navigation = False
		
		# Feature defaults
		self.beta_features = False
		self.advanced_features = False
		self.feature_hints = True
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert preferences to dictionary"""
		return {
			'preferences_id': self.preferences_id,
			'user_id': self.user_id,
			'notification_settings': self.get_notification_settings(),
			'privacy_settings': self.get_privacy_settings(),
			'ui_settings': self.get_ui_settings(),
			'feature_preferences': {
				'beta_features': self.beta_features,
				'advanced_features': self.advanced_features,
				'feature_hints': self.feature_hints
			},
			'integration_preferences': {
				'calendar_integration': self.calendar_integration,
				'task_management_integration': self.task_management_integration,
				'social_sharing': self.social_sharing
			},
			'custom_preferences': self.custom_preferences,
			'dashboard_layout': self.dashboard_layout,
			'default_workspace': self.default_workspace,
			'favorite_items': self.favorite_items,
			'created_on': self.created_on.isoformat() if self.created_on else None,
			'changed_on': self.changed_on.isoformat() if self.changed_on else None
		}