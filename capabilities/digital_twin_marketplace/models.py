"""
Digital Twin Marketplace Models

Database models for marketplace management, item publishing, user management,
transactions, reviews, and ecosystem analytics with comprehensive trading features.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ..auth_rbac.models import BaseMixin, AuditMixin, Model


class DTMUser(Model, AuditMixin, BaseMixin):
	"""
	Marketplace user with profile, reputation, and contribution tracking.
	
	Represents individual users, organizations, and enterprises
	participating in the digital twin marketplace ecosystem.
	"""
	__tablename__ = 'dtm_user'
	
	# Identity
	user_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# User Information
	username = Column(String(100), nullable=False, unique=True, index=True)
	email = Column(String(200), nullable=False, unique=True, index=True)
	display_name = Column(String(200), nullable=True)
	bio = Column(Text, nullable=True)
	avatar_url = Column(String(500), nullable=True)
	
	# Organization Details
	organization = Column(String(200), nullable=True)
	organization_role = Column(String(100), nullable=True)
	website = Column(String(500), nullable=True)
	location = Column(String(200), nullable=True)
	
	# User Classification
	user_type = Column(String(20), nullable=False, default='individual', index=True)  # individual, organization, enterprise
	subscription_tier = Column(String(20), default='basic', index=True)  # basic, premium, enterprise
	account_status = Column(String(20), default='active', index=True)  # active, suspended, banned
	
	# Reputation and Activity
	reputation_score = Column(Float, default=0.0, index=True)  # 0-5.0 reputation score
	contribution_count = Column(Integer, default=0)  # Number of published items
	download_count = Column(Integer, default=0)  # Items downloaded by user
	review_count = Column(Integer, default=0)  # Reviews written
	helpful_review_votes = Column(Integer, default=0)  # Helpful votes received
	
	# Verification and Trust
	is_verified = Column(Boolean, default=False, index=True)
	verification_date = Column(DateTime, nullable=True)
	verified_by = Column(String(36), nullable=True)
	trust_level = Column(String(20), default='new', index=True)  # new, trusted, expert, authority
	
	# API and Access
	api_key = Column(String(64), nullable=True, unique=True)
	api_rate_limit = Column(Integer, default=1000)  # Requests per hour
	api_quota_used = Column(Integer, default=0)
	api_quota_reset = Column(DateTime, nullable=True)
	
	# Preferences and Settings
	notification_preferences = Column(JSON, default=dict)  # Notification settings
	privacy_settings = Column(JSON, default=dict)  # Privacy preferences
	marketplace_preferences = Column(JSON, default=dict)  # Display and search preferences
	
	# Financial Information
	total_earnings = Column(Float, default=0.0)  # Total earnings from sales
	total_spent = Column(Float, default=0.0)  # Total spent on purchases
	payout_settings = Column(JSON, default=dict)  # Payment method settings
	
	# Activity Tracking
	last_login = Column(DateTime, nullable=True)
	last_activity = Column(DateTime, nullable=True)
	login_count = Column(Integer, default=0)
	
	# Relationships
	published_items = relationship("DTMItem", back_populates="author", foreign_keys="DTMItem.author_id")
	purchases = relationship("DTMTransaction", back_populates="buyer", foreign_keys="DTMTransaction.buyer_id")
	sales = relationship("DTMTransaction", back_populates="seller", foreign_keys="DTMTransaction.seller_id")
	reviews = relationship("DTMReview", back_populates="reviewer")
	
	def __repr__(self):
		return f"<DTMUser {self.username} ({self.user_type})>"
	
	def update_reputation(self, new_rating: float):
		"""Update user reputation based on new rating"""
		if self.review_count == 0:
			self.reputation_score = new_rating
		else:
			# Weighted average with previous ratings
			total_score = self.reputation_score * self.review_count
			self.reputation_score = (total_score + new_rating) / (self.review_count + 1)
		
		# Update trust level based on reputation and activity
		self.update_trust_level()
	
	def update_trust_level(self):
		"""Update trust level based on reputation and activity"""
		if self.reputation_score >= 4.5 and self.contribution_count >= 10:
			self.trust_level = 'authority'
		elif self.reputation_score >= 4.0 and self.contribution_count >= 5:
			self.trust_level = 'expert'
		elif self.reputation_score >= 3.5 and self.contribution_count >= 2:
			self.trust_level = 'trusted'
		else:
			self.trust_level = 'new'
	
	def can_publish_premium(self) -> bool:
		"""Check if user can publish premium items"""
		return self.is_verified and self.trust_level in ['expert', 'authority']


class DTMCategory(Model, AuditMixin, BaseMixin):
	"""
	Marketplace categories for organizing digital twin items.
	
	Hierarchical category system for organizing digital twins
	by industry, application, and technical characteristics.
	"""
	__tablename__ = 'dtm_category'
	
	# Identity
	category_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Category Information
	name = Column(String(100), nullable=False, index=True)
	slug = Column(String(100), nullable=False, unique=True, index=True)
	description = Column(Text, nullable=True)
	icon = Column(String(50), nullable=True)  # Icon class or name
	color = Column(String(7), nullable=True)  # Hex color code
	
	# Hierarchy
	parent_category_id = Column(String(36), ForeignKey('dtm_category.category_id'), nullable=True)
	level = Column(Integer, default=0)  # Depth in hierarchy
	sort_order = Column(Integer, default=0)  # Display order
	
	# Status and Visibility
	is_active = Column(Boolean, default=True, index=True)
	is_featured = Column(Boolean, default=False)
	item_count = Column(Integer, default=0)  # Number of items in category
	
	# Metadata
	keywords = Column(JSON, default=list)  # Search keywords
	attributes = Column(JSON, default=dict)  # Category-specific attributes
	
	# Relationships
	parent_category = relationship("DTMCategory", remote_side=[category_id])
	child_categories = relationship("DTMCategory")
	items = relationship("DTMItem", back_populates="category")
	
	def __repr__(self):
		return f"<DTMCategory {self.name}>"
	
	def get_full_path(self) -> str:
		"""Get full category path (e.g., 'Manufacturing > Automotive > Assembly')"""
		if self.parent_category:
			return f"{self.parent_category.get_full_path()} > {self.name}"
		return self.name


class DTMItem(Model, AuditMixin, BaseMixin):
	"""
	Marketplace item representing digital twins, algorithms, or services.
	
	Comprehensive item model supporting various types of digital twin
	assets with versioning, pricing, licensing, and quality metrics.
	"""
	__tablename__ = 'dtm_item'
	
	# Identity
	item_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Basic Information
	name = Column(String(200), nullable=False, index=True)
	slug = Column(String(200), nullable=False, unique=True, index=True)
	description = Column(Text, nullable=False)
	short_description = Column(String(500), nullable=True)
	
	# Classification
	category_id = Column(String(36), ForeignKey('dtm_category.category_id'), nullable=False, index=True)
	item_type = Column(String(50), nullable=False, index=True)  # digital_twin, algorithm, data_model, etc.
	industry_tags = Column(JSON, default=list)  # Industry classifications
	technical_tags = Column(JSON, default=list)  # Technical specifications
	keywords = Column(JSON, default=list)  # Search keywords
	
	# Authorship
	author_id = Column(String(36), ForeignKey('dtm_user.user_id'), nullable=False, index=True)
	organization = Column(String(200), nullable=True)
	contributors = Column(JSON, default=list)  # Additional contributors
	
	# Versioning
	version = Column(String(20), nullable=False, default='1.0.0')
	version_history = Column(JSON, default=list)  # Previous versions
	is_latest_version = Column(Boolean, default=True, index=True)
	parent_item_id = Column(String(36), ForeignKey('dtm_item.item_id'), nullable=True)  # For version chains
	
	# Content and Resources
	source_code_url = Column(String(500), nullable=True)
	documentation_url = Column(String(500), nullable=True)
	demo_url = Column(String(500), nullable=True)
	api_endpoint = Column(String(500), nullable=True)
	model_files = Column(JSON, default=list)  # List of model file URLs
	sample_data_url = Column(String(500), nullable=True)
	tutorial_url = Column(String(500), nullable=True)
	
	# Media
	thumbnail_url = Column(String(500), nullable=True)
	screenshots = Column(JSON, default=list)  # List of screenshot URLs
	videos = Column(JSON, default=list)  # Video demonstrations
	
	# Licensing and Pricing
	license_type = Column(String(50), nullable=False, index=True)  # open_source, commercial, freemium, etc.
	license_details = Column(Text, nullable=True)
	price = Column(Float, default=0.0, index=True)
	currency = Column(String(3), default='USD')
	pricing_model = Column(String(20), default='free', index=True)  # free, one_time, subscription, pay_per_use
	pricing_tiers = Column(JSON, default=dict)  # Different pricing options
	
	# Quality and Certification
	quality_tier = Column(String(20), default='community', index=True)  # community, verified, certified, enterprise
	is_tested = Column(Boolean, default=False)
	is_certified = Column(Boolean, default=False)
	is_security_scanned = Column(Boolean, default=False)
	is_performance_benchmarked = Column(Boolean, default=False)
	quality_score = Column(Float, nullable=True)  # 0-100 quality score
	
	# Technical Specifications
	requirements = Column(JSON, default=dict)  # System requirements
	compatibility = Column(JSON, default=list)  # Compatible platforms/frameworks
	performance_metrics = Column(JSON, default=dict)  # Performance benchmarks
	supported_formats = Column(JSON, default=list)  # Supported file formats
	api_specification = Column(JSON, default=dict)  # API documentation
	
	# Status and Visibility
	status = Column(String(20), default='draft', index=True)  # draft, review, published, archived
	is_public = Column(Boolean, default=True, index=True)
	is_featured = Column(Boolean, default=False)
	is_trending = Column(Boolean, default=False)
	publication_date = Column(DateTime, nullable=True, index=True)
	
	# Metrics and Analytics
	download_count = Column(Integer, default=0, index=True)
	view_count = Column(Integer, default=0)
	clone_count = Column(Integer, default=0)
	fork_count = Column(Integer, default=0)
	star_count = Column(Integer, default=0)
	
	# Rating and Reviews
	average_rating = Column(Float, default=0.0, index=True)  # 0-5.0 average rating
	rating_count = Column(Integer, default=0)
	review_count = Column(Integer, default=0)
	rating_distribution = Column(JSON, default=dict)  # Distribution of 1-5 star ratings
	
	# Popularity and Engagement
	popularity_score = Column(Float, default=0.0, index=True)  # Calculated popularity metric
	engagement_score = Column(Float, default=0.0)  # User engagement metric
	last_activity = Column(DateTime, nullable=True)
	
	# Content Moderation
	is_approved = Column(Boolean, default=False)
	approved_by = Column(String(36), nullable=True)
	approval_date = Column(DateTime, nullable=True)
	moderation_flags = Column(JSON, default=list)  # Content flags
	
	# Relationships
	category = relationship("DTMCategory", back_populates="items")
	author = relationship("DTMUser", back_populates="published_items", foreign_keys=[author_id])
	parent_item = relationship("DTMItem", remote_side=[item_id])
	versions = relationship("DTMItem", remote_side=[parent_item_id])
	reviews = relationship("DTMReview", back_populates="item")
	transactions = relationship("DTMTransaction", back_populates="item")
	
	def __repr__(self):
		return f"<DTMItem {self.name} v{self.version}>"
	
	def calculate_popularity_score(self) -> float:
		"""Calculate popularity score based on various metrics"""
		# Weighted formula considering downloads, ratings, and recency
		download_score = min(self.download_count / 1000, 10)  # Cap at 10
		rating_score = self.average_rating * 2  # 0-10 scale
		recency_days = (datetime.utcnow() - (self.publication_date or self.created_at)).days
		recency_score = max(0, 10 - recency_days / 30)  # Decay over months
		
		self.popularity_score = (download_score * 0.4 + rating_score * 0.4 + recency_score * 0.2)
		return self.popularity_score
	
	def update_rating(self, new_rating: int):
		"""Update average rating with new rating"""
		if self.rating_count == 0:
			self.average_rating = float(new_rating)
			self.rating_count = 1
		else:
			total_rating = self.average_rating * self.rating_count
			self.rating_count += 1
			self.average_rating = (total_rating + new_rating) / self.rating_count
		
		# Update rating distribution
		if not self.rating_distribution:
			self.rating_distribution = {str(i): 0 for i in range(1, 6)}
		
		self.rating_distribution[str(new_rating)] = self.rating_distribution.get(str(new_rating), 0) + 1
	
	def is_free(self) -> bool:
		"""Check if item is free"""
		return self.price == 0.0 or self.pricing_model == 'free'
	
	def can_be_downloaded_by(self, user_id: str) -> bool:
		"""Check if user can download this item"""
		if self.is_free():
			return True
		
		# Check if user has purchased this item
		for transaction in self.transactions:
			if transaction.buyer_id == user_id and transaction.status == 'completed':
				return True
		
		return False


class DTMReview(Model, AuditMixin, BaseMixin):
	"""
	User reviews and ratings for marketplace items.
	
	Comprehensive review system with ratings, comments,
	helpfulness voting, and moderation capabilities.
	"""
	__tablename__ = 'dtm_review'
	
	# Identity
	review_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	item_id = Column(String(36), ForeignKey('dtm_item.item_id'), nullable=False, index=True)
	reviewer_id = Column(String(36), ForeignKey('dtm_user.user_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Review Content
	rating = Column(Integer, nullable=False, index=True)  # 1-5 star rating
	title = Column(String(200), nullable=True)
	content = Column(Text, nullable=False)
	
	# Review Metadata
	is_verified_purchase = Column(Boolean, default=False)
	version_reviewed = Column(String(20), nullable=True)  # Version of item reviewed
	use_case = Column(String(100), nullable=True)  # How reviewer used the item
	experience_level = Column(String(20), nullable=True)  # beginner, intermediate, advanced
	
	# Engagement
	helpful_votes = Column(Integer, default=0)
	unhelpful_votes = Column(Integer, default=0)
	total_votes = Column(Integer, default=0)
	helpfulness_score = Column(Float, default=0.0)  # Calculated helpfulness
	
	# Status and Moderation
	status = Column(String(20), default='published', index=True)  # draft, published, hidden, flagged
	is_flagged = Column(Boolean, default=False)
	flag_reasons = Column(JSON, default=list)  # Reasons for flagging
	moderated_by = Column(String(36), nullable=True)
	moderation_date = Column(DateTime, nullable=True)
	
	# Response from Author
	author_response = Column(Text, nullable=True)
	author_response_date = Column(DateTime, nullable=True)
	
	# Relationships
	item = relationship("DTMItem", back_populates="reviews")
	reviewer = relationship("DTMUser", back_populates="reviews")
	
	def __repr__(self):
		return f"<DTMReview {self.rating}/5 for {self.item.name}>"
	
	def calculate_helpfulness_score(self) -> float:
		"""Calculate helpfulness score based on votes"""
		if self.total_votes == 0:
			return 0.0
		
		self.helpfulness_score = self.helpful_votes / self.total_votes
		return self.helpfulness_score
	
	def add_helpful_vote(self):
		"""Add a helpful vote"""
		self.helpful_votes += 1
		self.total_votes += 1
		self.calculate_helpfulness_score()
	
	def add_unhelpful_vote(self):
		"""Add an unhelpful vote"""
		self.unhelpful_votes += 1
		self.total_votes += 1
		self.calculate_helpfulness_score()


class DTMTransaction(Model, AuditMixin, BaseMixin):
	"""
	Marketplace transactions for item purchases and subscriptions.
	
	Comprehensive transaction tracking with payment processing,
	licensing, and refund management capabilities.
	"""
	__tablename__ = 'dtm_transaction'
	
	# Identity
	transaction_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	item_id = Column(String(36), ForeignKey('dtm_item.item_id'), nullable=False, index=True)
	buyer_id = Column(String(36), ForeignKey('dtm_user.user_id'), nullable=False, index=True)
	seller_id = Column(String(36), ForeignKey('dtm_user.user_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Transaction Details
	transaction_type = Column(String(20), nullable=False, index=True)  # purchase, subscription, renewal, refund
	amount = Column(Float, nullable=False)
	currency = Column(String(3), default='USD')
	payment_method = Column(String(50), nullable=True)  # credit_card, paypal, crypto, etc.
	
	# Status and Processing
	status = Column(String(20), default='pending', index=True)  # pending, processing, completed, failed, refunded
	payment_status = Column(String(20), default='pending')  # pending, authorized, captured, failed
	processing_fee = Column(Float, default=0.0)
	net_amount = Column(Float, nullable=False)  # Amount after fees
	
	# Timing
	initiated_at = Column(DateTime, nullable=False, index=True)
	completed_at = Column(DateTime, nullable=True)
	expires_at = Column(DateTime, nullable=True)  # For subscriptions
	
	# Payment Processing
	payment_processor = Column(String(50), nullable=True)  # stripe, paypal, etc.
	external_transaction_id = Column(String(200), nullable=True)
	payment_reference = Column(String(200), nullable=True)
	
	# Licensing
	license_key = Column(String(200), nullable=True, unique=True)
	license_type = Column(String(50), nullable=True)
	license_expires_at = Column(DateTime, nullable=True)
	download_limit = Column(Integer, nullable=True)
	download_count = Column(Integer, default=0)
	
	# Refunds and Disputes
	is_refundable = Column(Boolean, default=True)
	refund_requested_at = Column(DateTime, nullable=True)
	refund_reason = Column(String(500), nullable=True)
	refunded_at = Column(DateTime, nullable=True)
	refunded_amount = Column(Float, nullable=True)
	
	# Metadata
	user_agent = Column(String(500), nullable=True)
	ip_address = Column(String(45), nullable=True)
	metadata = Column(JSON, default=dict)  # Additional transaction data
	
	# Relationships
	item = relationship("DTMItem", back_populates="transactions")
	buyer = relationship("DTMUser", back_populates="purchases", foreign_keys=[buyer_id])
	seller = relationship("DTMUser", back_populates="sales", foreign_keys=[seller_id])
	
	def __repr__(self):
		return f"<DTMTransaction {self.transaction_type} ${self.amount}>"
	
	def generate_license_key(self) -> str:
		"""Generate unique license key for transaction"""
		import hashlib
		key_data = f"{self.transaction_id}:{self.item_id}:{self.buyer_id}:{datetime.utcnow().timestamp()}"
		key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:32]
		self.license_key = f"DTMP-{key_hash[:8]}-{key_hash[8:16]}-{key_hash[16:24]}-{key_hash[24:32]}"
		return self.license_key
	
	def is_license_valid(self) -> bool:
		"""Check if license is still valid"""
		if self.status != 'completed':
			return False
		
		if self.license_expires_at and datetime.utcnow() > self.license_expires_at:
			return False
		
		if self.download_limit and self.download_count >= self.download_limit:
			return False
		
		return True
	
	def can_download(self) -> bool:
		"""Check if user can download the item"""
		return self.is_license_valid()


class DTMAnalytics(Model, AuditMixin, BaseMixin):
	"""
	Analytics and metrics for marketplace performance tracking.
	
	Aggregated analytics data for marketplace insights,
	trends, performance monitoring, and business intelligence.
	"""
	__tablename__ = 'dtm_analytics'
	
	# Identity
	analytics_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Time Period
	date = Column(DateTime, nullable=False, index=True)
	period_type = Column(String(20), nullable=False, index=True)  # daily, weekly, monthly, yearly
	
	# User Metrics
	total_users = Column(Integer, default=0)
	new_users = Column(Integer, default=0)
	active_users = Column(Integer, default=0)
	verified_users = Column(Integer, default=0)
	
	# Item Metrics
	total_items = Column(Integer, default=0)
	new_items = Column(Integer, default=0)
	published_items = Column(Integer, default=0)
	featured_items = Column(Integer, default=0)
	
	# Transaction Metrics
	total_transactions = Column(Integer, default=0)
	transaction_volume = Column(Float, default=0.0)
	successful_transactions = Column(Integer, default=0)
	failed_transactions = Column(Integer, default=0)
	refunded_transactions = Column(Integer, default=0)
	
	# Revenue Metrics
	gross_revenue = Column(Float, default=0.0)
	net_revenue = Column(Float, default=0.0)
	processing_fees = Column(Float, default=0.0)
	average_transaction_value = Column(Float, default=0.0)
	
	# Engagement Metrics
	total_downloads = Column(Integer, default=0)
	total_views = Column(Integer, default=0)
	total_reviews = Column(Integer, default=0)
	average_rating = Column(Float, default=0.0)
	
	# Category Performance
	category_metrics = Column(JSON, default=dict)  # Performance by category
	top_items = Column(JSON, default=list)  # Top performing items
	top_contributors = Column(JSON, default=list)  # Top contributing users
	
	# Quality Metrics
	quality_distribution = Column(JSON, default=dict)  # Distribution by quality tier
	certification_rate = Column(Float, default=0.0)  # Percentage of certified items
	security_scan_rate = Column(Float, default=0.0)  # Percentage of security scanned items
	
	def __repr__(self):
		return f"<DTMAnalytics {self.period_type} {self.date.strftime('%Y-%m-%d')}>"
	
	def calculate_conversion_rate(self) -> float:
		"""Calculate transaction conversion rate"""
		if self.total_views == 0:
			return 0.0
		return (self.total_transactions / self.total_views) * 100
	
	def calculate_success_rate(self) -> float:
		"""Calculate transaction success rate"""
		if self.total_transactions == 0:
			return 0.0
		return (self.successful_transactions / self.total_transactions) * 100