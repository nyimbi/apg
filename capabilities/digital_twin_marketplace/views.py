"""
Digital Twin Marketplace Views

Flask-AppBuilder views for comprehensive marketplace management,
item publishing, user management, transactions, and ecosystem analytics.
"""

from flask import request, jsonify, flash, redirect, url_for, render_template
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from flask_appbuilder.widgets import FormWidget, ListWidget, SearchWidget
from flask_appbuilder.forms import DynamicForm
from wtforms import StringField, TextAreaField, SelectField, BooleanField, FloatField, IntegerField, validators
from wtforms.validators import DataRequired, Length, Optional, NumberRange, URL
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from .models import (
	DTMUser, DTMCategory, DTMItem, DTMReview,
	DTMTransaction, DTMAnalytics
)


class DigitalTwinMarketplaceBaseView(BaseView):
	"""Base view for digital twin marketplace functionality"""
	
	def __init__(self):
		super().__init__()
		self.default_view = 'dashboard'
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID from security context"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"
	
	def _format_price(self, price: float, currency: str = 'USD') -> str:
		"""Format price for display"""
		if price == 0.0:
			return "Free"
		return f"${price:,.2f} {currency}"
	
	def _format_rating(self, rating: float) -> str:
		"""Format rating for display"""
		if rating is None:
			return "No ratings"
		return f"{rating:.1f}/5.0"


class DTMUserModelView(ModelView):
	"""Marketplace user management view"""
	
	datamodel = SQLAInterface(DTMUser)
	
	# List view configuration
	list_columns = [
		'username', 'email', 'organization', 'user_type',
		'subscription_tier', 'reputation_score', 'contribution_count',
		'is_verified', 'account_status'
	]
	show_columns = [
		'user_id', 'username', 'email', 'display_name', 'bio', 'avatar_url',
		'organization', 'organization_role', 'website', 'location',
		'user_type', 'subscription_tier', 'account_status', 'reputation_score',
		'contribution_count', 'download_count', 'review_count', 'helpful_review_votes',
		'is_verified', 'verification_date', 'trust_level', 'total_earnings',
		'total_spent', 'last_login', 'last_activity', 'login_count'
	]
	edit_columns = [
		'username', 'email', 'display_name', 'bio', 'avatar_url',
		'organization', 'organization_role', 'website', 'location',
		'user_type', 'subscription_tier', 'account_status',
		'is_verified', 'trust_level', 'notification_preferences',
		'privacy_settings', 'marketplace_preferences'
	]
	add_columns = [
		'username', 'email', 'display_name', 'bio', 'avatar_url',
		'organization', 'organization_role', 'website', 'location',
		'user_type', 'subscription_tier'
	]
	
	# Search and filtering
	search_columns = ['username', 'email', 'organization', 'display_name']
	base_filters = [['account_status', lambda: 'active', lambda: True]]
	
	# Ordering
	base_order = ('reputation_score', 'desc')
	
	# Form validation
	validators_columns = {
		'username': [DataRequired(), Length(min=3, max=100)],
		'email': [DataRequired(), validators.Email()],
		'reputation_score': [NumberRange(min=0, max=5)],
		'contribution_count': [NumberRange(min=0)]
	}
	
	# Custom labels
	label_columns = {
		'user_id': 'User ID',
		'display_name': 'Display Name',
		'organization_role': 'Role in Organization',
		'user_type': 'User Type',
		'subscription_tier': 'Subscription Tier',
		'account_status': 'Account Status',
		'reputation_score': 'Reputation Score',
		'contribution_count': 'Published Items',
		'download_count': 'Downloads',
		'review_count': 'Reviews Written',
		'helpful_review_votes': 'Helpful Votes',
		'is_verified': 'Verified User',
		'verification_date': 'Verified Date',
		'trust_level': 'Trust Level',
		'total_earnings': 'Total Earnings',
		'total_spent': 'Total Spent',
		'last_login': 'Last Login',
		'last_activity': 'Last Activity',
		'login_count': 'Login Count',
		'api_rate_limit': 'API Rate Limit',
		'notification_preferences': 'Notification Settings',
		'privacy_settings': 'Privacy Settings',
		'marketplace_preferences': 'Marketplace Preferences'
	}
	
	@expose('/verify_user/<int:pk>')
	@has_access
	def verify_user(self, pk):
		"""Verify user account"""
		user = self.datamodel.get(pk)
		if not user:
			flash('User not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			user.is_verified = True
			user.verification_date = datetime.utcnow()
			user.verified_by = self._get_current_user_id()
			user.update_trust_level()
			self.datamodel.edit(user)
			flash(f'User {user.username} verified successfully', 'success')
		except Exception as e:
			flash(f'Error verifying user: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/suspend_user/<int:pk>')
	@has_access
	def suspend_user(self, pk):
		"""Suspend user account"""
		user = self.datamodel.get(pk)
		if not user:
			flash('User not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			user.account_status = 'suspended'
			self.datamodel.edit(user)
			flash(f'User {user.username} suspended', 'success')
		except Exception as e:
			flash(f'Error suspending user: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/user_analytics/<int:pk>')
	@has_access
	def user_analytics(self, pk):
		"""View user analytics dashboard"""
		user = self.datamodel.get(pk)
		if not user:
			flash('User not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			analytics_data = self._get_user_analytics(user)
			
			return render_template('digital_twin_marketplace/user_analytics.html',
								   user=user,
								   analytics_data=analytics_data,
								   page_title=f"Analytics: {user.username}")
		except Exception as e:
			flash(f'Error loading user analytics: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new user"""
		item.tenant_id = self._get_tenant_id()
		
		# Set default values
		if not item.reputation_score:
			item.reputation_score = 0.0
		if not item.trust_level:
			item.trust_level = 'new'
		if not item.account_status:
			item.account_status = 'active'
	
	def _get_user_analytics(self, user: DTMUser) -> Dict[str, Any]:
		"""Get analytics data for user"""
		return {
			'reputation_trend': 'stable',
			'contribution_growth': [],
			'download_history': [],
			'earnings_trend': [],
			'review_statistics': {
				'total_reviews': user.review_count,
				'helpful_votes': user.helpful_review_votes,
				'average_helpfulness': 0.0
			},
			'activity_heatmap': {}
		}
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class DTMCategoryModelView(ModelView):
	"""Marketplace category management view"""
	
	datamodel = SQLAInterface(DTMCategory)
	
	# List view configuration
	list_columns = [
		'name', 'slug', 'parent_category', 'level',
		'item_count', 'is_active', 'is_featured', 'sort_order'
	]
	show_columns = [
		'category_id', 'name', 'slug', 'description', 'icon', 'color',
		'parent_category', 'level', 'sort_order', 'is_active', 'is_featured',
		'item_count', 'keywords', 'attributes'
	]
	edit_columns = [
		'name', 'slug', 'description', 'icon', 'color', 'parent_category',
		'sort_order', 'is_active', 'is_featured', 'keywords', 'attributes'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['name', 'description', 'slug']
	base_filters = [['is_active', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('sort_order', 'asc')
	
	# Form validation
	validators_columns = {
		'name': [DataRequired(), Length(min=2, max=100)],
		'slug': [DataRequired(), Length(min=2, max=100)],
		'sort_order': [NumberRange(min=0)]
	}
	
	# Custom labels
	label_columns = {
		'category_id': 'Category ID',
		'parent_category': 'Parent Category',
		'sort_order': 'Sort Order',
		'is_active': 'Active',
		'is_featured': 'Featured',
		'item_count': 'Items Count',
		'color': 'Color (Hex)',
		'icon': 'Icon Class'
	}
	
	@expose('/category_stats/<int:pk>')
	@has_access
	def category_stats(self, pk):
		"""View category statistics"""
		category = self.datamodel.get(pk)
		if not category:
			flash('Category not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			stats_data = self._get_category_statistics(category)
			
			return render_template('digital_twin_marketplace/category_stats.html',
								   category=category,
								   stats_data=stats_data,
								   page_title=f"Statistics: {category.name}")
		except Exception as e:
			flash(f'Error loading category statistics: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new category"""
		item.tenant_id = self._get_tenant_id()
		
		# Calculate level based on parent
		if item.parent_category:
			item.level = item.parent_category.level + 1
		else:
			item.level = 0
		
		# Generate slug from name if not provided
		if not item.slug:
			item.slug = item.name.lower().replace(' ', '-').replace('_', '-')
	
	def _get_category_statistics(self, category: DTMCategory) -> Dict[str, Any]:
		"""Get statistics for category"""
		return {
			'total_items': category.item_count,
			'active_items': 0,
			'total_downloads': 0,
			'average_rating': 0.0,
			'top_items': [],
			'growth_trend': 'stable',
			'subcategories': len(category.child_categories)
		}
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class DTMItemModelView(ModelView):
	"""Marketplace item management view"""
	
	datamodel = SQLAInterface(DTMItem)
	
	# List view configuration
	list_columns = [
		'name', 'category', 'author', 'item_type', 'version',
		'status', 'average_rating', 'download_count', 'price'
	]
	show_columns = [
		'item_id', 'name', 'slug', 'description', 'category', 'item_type',
		'author', 'organization', 'version', 'license_type', 'price',
		'currency', 'pricing_model', 'quality_tier', 'status',
		'average_rating', 'rating_count', 'download_count', 'view_count',
		'popularity_score', 'is_featured', 'publication_date'
	]
	edit_columns = [
		'name', 'slug', 'description', 'short_description', 'category',
		'item_type', 'industry_tags', 'technical_tags', 'keywords',
		'version', 'license_type', 'license_details', 'price', 'currency',
		'pricing_model', 'quality_tier', 'status', 'is_public', 'is_featured',
		'source_code_url', 'documentation_url', 'demo_url', 'api_endpoint',
		'thumbnail_url', 'requirements', 'compatibility'
	]
	add_columns = [
		'name', 'description', 'short_description', 'category', 'item_type',
		'industry_tags', 'technical_tags', 'keywords', 'version',
		'license_type', 'license_details', 'price', 'currency', 'pricing_model',
		'source_code_url', 'documentation_url', 'demo_url'
	]
	
	# Search and filtering
	search_columns = ['name', 'description', 'author.username', 'organization']
	base_filters = [['status', lambda: 'published', lambda: True]]
	
	# Ordering
	base_order = ('popularity_score', 'desc')
	
	# Form validation
	validators_columns = {
		'name': [DataRequired(), Length(min=5, max=200)],
		'description': [DataRequired(), Length(min=50)],
		'version': [DataRequired()],
		'price': [NumberRange(min=0)],
		'source_code_url': [Optional(), URL()],
		'documentation_url': [Optional(), URL()],
		'demo_url': [Optional(), URL()]
	}
	
	# Custom labels
	label_columns = {
		'item_id': 'Item ID',
		'short_description': 'Short Description',
		'item_type': 'Item Type',
		'industry_tags': 'Industry Tags',
		'technical_tags': 'Technical Tags',
		'license_type': 'License Type',
		'license_details': 'License Details',
		'pricing_model': 'Pricing Model',
		'quality_tier': 'Quality Tier',
		'is_public': 'Public',
		'is_featured': 'Featured',
		'source_code_url': 'Source Code URL',
		'documentation_url': 'Documentation URL',
		'demo_url': 'Demo URL',
		'api_endpoint': 'API Endpoint',
		'thumbnail_url': 'Thumbnail URL',
		'average_rating': 'Average Rating',
		'rating_count': 'Rating Count',
		'download_count': 'Downloads',
		'view_count': 'Views',
		'popularity_score': 'Popularity Score',
		'publication_date': 'Published Date',
		'is_tested': 'Tested',
		'is_certified': 'Certified',
		'is_security_scanned': 'Security Scanned',
		'performance_metrics': 'Performance Metrics'
	}
	
	@expose('/publish_item/<int:pk>')
	@has_access
	def publish_item(self, pk):
		"""Publish marketplace item"""
		item = self.datamodel.get(pk)
		if not item:
			flash('Item not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			item.status = 'published'
			item.publication_date = datetime.utcnow()
			item.is_public = True
			self.datamodel.edit(item)
			flash(f'Item "{item.name}" published successfully', 'success')
		except Exception as e:
			flash(f'Error publishing item: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/feature_item/<int:pk>')
	@has_access
	def feature_item(self, pk):
		"""Feature marketplace item"""
		item = self.datamodel.get(pk)
		if not item:
			flash('Item not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			item.is_featured = not item.is_featured
			self.datamodel.edit(item)
			status = "featured" if item.is_featured else "unfeatured"
			flash(f'Item "{item.name}" {status} successfully', 'success')
		except Exception as e:
			flash(f'Error updating item: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/item_analytics/<int:pk>')
	@has_access
	def item_analytics(self, pk):
		"""View item analytics dashboard"""
		item = self.datamodel.get(pk)
		if not item:
			flash('Item not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			analytics_data = self._get_item_analytics(item)
			
			return render_template('digital_twin_marketplace/item_analytics.html',
								   item=item,
								   analytics_data=analytics_data,
								   page_title=f"Analytics: {item.name}")
		except Exception as e:
			flash(f'Error loading item analytics: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/quality_check/<int:pk>')
	@has_access
	def quality_check(self, pk):
		"""Perform quality check on item"""
		item = self.datamodel.get(pk)
		if not item:
			flash('Item not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Simulate quality check
			quality_results = self._perform_quality_check(item)
			flash(f'Quality check completed. Score: {quality_results["score"]}/100', 'success')
		except Exception as e:
			flash(f'Error performing quality check: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new item"""
		item.tenant_id = self._get_tenant_id()
		item.author_id = self._get_current_user_id()
		
		# Set default values
		if not item.status:
			item.status = 'draft'
		if not item.quality_tier:
			item.quality_tier = 'community'
		if not item.version:
			item.version = '1.0.0'
		
		# Generate slug from name if not provided
		if not item.slug:
			item.slug = item.name.lower().replace(' ', '-').replace('_', '-')
	
	def post_add(self, item):
		"""Post-process after adding item"""
		# Calculate initial popularity score
		item.calculate_popularity_score()
		self.datamodel.edit(item)
	
	def _get_item_analytics(self, item: DTMItem) -> Dict[str, Any]:
		"""Get analytics data for item"""
		return {
			'download_trend': [],
			'rating_trend': [],
			'view_history': [],
			'revenue_data': [],
			'geographic_distribution': {},
			'user_engagement': {
				'total_views': item.view_count,
				'downloads': item.download_count,
				'conversion_rate': 0.0
			},
			'performance_metrics': item.performance_metrics or {}
		}
	
	def _perform_quality_check(self, item: DTMItem) -> Dict[str, Any]:
		"""Perform quality check on item"""
		score = 100
		issues = []
		
		if len(item.description) < 100:
			score -= 10
			issues.append("Description too short")
		
		if not item.documentation_url:
			score -= 15
			issues.append("No documentation provided")
		
		if not item.keywords:
			score -= 10
			issues.append("No keywords provided")
		
		return {
			'score': max(0, score),
			'issues': issues,
			'recommendations': []
		}
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class DTMReviewModelView(ModelView):
	"""Marketplace review management view"""
	
	datamodel = SQLAInterface(DTMReview)
	
	# List view configuration
	list_columns = [
		'item', 'reviewer', 'rating', 'title',
		'is_verified_purchase', 'helpful_votes', 'status'
	]
	show_columns = [
		'review_id', 'item', 'reviewer', 'rating', 'title', 'content',
		'is_verified_purchase', 'version_reviewed', 'use_case', 'experience_level',
		'helpful_votes', 'unhelpful_votes', 'helpfulness_score', 'status',
		'author_response', 'author_response_date'
	]
	edit_columns = [
		'status', 'is_flagged', 'flag_reasons', 'author_response'
	]
	add_columns = [
		'rating', 'title', 'content', 'use_case', 'experience_level'
	]
	
	# Search and filtering
	search_columns = ['item.name', 'reviewer.username', 'title', 'content']
	base_filters = [['status', lambda: 'published', lambda: True]]
	
	# Ordering
	base_order = ('created_at', 'desc')
	
	# Form validation
	validators_columns = {
		'rating': [DataRequired(), NumberRange(min=1, max=5)],
		'title': [Optional(), Length(max=200)],
		'content': [DataRequired(), Length(min=10)]
	}
	
	# Custom labels
	label_columns = {
		'review_id': 'Review ID',
		'is_verified_purchase': 'Verified Purchase',
		'version_reviewed': 'Version Reviewed',
		'use_case': 'Use Case',
		'experience_level': 'Experience Level',
		'helpful_votes': 'Helpful Votes',
		'unhelpful_votes': 'Unhelpful Votes',
		'helpfulness_score': 'Helpfulness Score',
		'is_flagged': 'Flagged',
		'flag_reasons': 'Flag Reasons',
		'author_response': 'Author Response',
		'author_response_date': 'Response Date'
	}
	
	@expose('/moderate_review/<int:pk>')
	@has_access
	def moderate_review(self, pk):
		"""Moderate review content"""
		review = self.datamodel.get(pk)
		if not review:
			flash('Review not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			action = request.args.get('action', 'hide')
			if action == 'hide':
				review.status = 'hidden'
			elif action == 'publish':
				review.status = 'published'
			elif action == 'flag':
				review.is_flagged = True
				review.status = 'flagged'
			
			review.moderated_by = self._get_current_user_id()
			review.moderation_date = datetime.utcnow()
			self.datamodel.edit(review)
			flash(f'Review moderated: {action}', 'success')
		except Exception as e:
			flash(f'Error moderating review: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/helpful_vote/<int:pk>')
	@has_access
	def helpful_vote(self, pk):
		"""Add helpful vote to review"""
		review = self.datamodel.get(pk)
		if not review:
			flash('Review not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			review.add_helpful_vote()
			self.datamodel.edit(review)
			flash('Helpful vote added', 'success')
		except Exception as e:
			flash(f'Error adding vote: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new review"""
		item.tenant_id = self._get_tenant_id()
		item.reviewer_id = self._get_current_user_id()
		
		# Set default values
		if not item.status:
			item.status = 'published'
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class DTMTransactionModelView(ModelView):
	"""Marketplace transaction management view"""
	
	datamodel = SQLAInterface(DTMTransaction)
	
	# List view configuration
	list_columns = [
		'transaction_id', 'item', 'buyer', 'seller', 'transaction_type',
		'amount', 'currency', 'status', 'completed_at'
	]
	show_columns = [
		'transaction_id', 'item', 'buyer', 'seller', 'transaction_type',
		'amount', 'currency', 'payment_method', 'status', 'payment_status',
		'processing_fee', 'net_amount', 'initiated_at', 'completed_at',
		'license_key', 'license_type', 'download_count', 'download_limit'
	]
	# Read-only view for transactions
	edit_columns = ['status', 'payment_status']
	add_columns = []
	can_create = False
	
	# Search and filtering
	search_columns = ['transaction_id', 'item.name', 'buyer.username', 'seller.username']
	base_filters = [['status', lambda: 'completed', lambda: True]]
	
	# Ordering
	base_order = ('initiated_at', 'desc')
	
	# Custom labels
	label_columns = {
		'transaction_id': 'Transaction ID',
		'transaction_type': 'Type',
		'payment_method': 'Payment Method',
		'payment_status': 'Payment Status',
		'processing_fee': 'Processing Fee',
		'net_amount': 'Net Amount',
		'initiated_at': 'Initiated At',
		'completed_at': 'Completed At',
		'expires_at': 'Expires At',
		'license_key': 'License Key',
		'license_type': 'License Type',
		'license_expires_at': 'License Expires',
		'download_limit': 'Download Limit',
		'download_count': 'Downloads Used',
		'is_refundable': 'Refundable',
		'refund_requested_at': 'Refund Requested',
		'refunded_at': 'Refunded At',
		'refunded_amount': 'Refund Amount'
	}
	
	@expose('/process_refund/<int:pk>')
	@has_access
	def process_refund(self, pk):
		"""Process transaction refund"""
		transaction = self.datamodel.get(pk)
		if not transaction:
			flash('Transaction not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			if not transaction.is_refundable:
				flash('Transaction is not refundable', 'warning')
				return redirect(self.get_redirect())
			
			transaction.status = 'refunded'
			transaction.refunded_at = datetime.utcnow()
			transaction.refunded_amount = transaction.amount
			self.datamodel.edit(transaction)
			flash(f'Refund processed: ${transaction.refunded_amount:.2f}', 'success')
		except Exception as e:
			flash(f'Error processing refund: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/view_license/<int:pk>')
	@has_access
	def view_license(self, pk):
		"""View transaction license details"""
		transaction = self.datamodel.get(pk)
		if not transaction:
			flash('Transaction not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			license_data = self._get_license_details(transaction)
			
			return render_template('digital_twin_marketplace/license_details.html',
								   transaction=transaction,
								   license_data=license_data,
								   page_title=f"License: {transaction.transaction_id}")
		except Exception as e:
			flash(f'Error loading license details: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	def _get_license_details(self, transaction: DTMTransaction) -> Dict[str, Any]:
		"""Get license details for transaction"""
		return {
			'is_valid': transaction.is_license_valid(),
			'can_download': transaction.can_download(),
			'remaining_downloads': (transaction.download_limit - transaction.download_count) if transaction.download_limit else 'Unlimited',
			'expiry_date': transaction.license_expires_at,
			'terms_and_conditions': f"License for {transaction.item.name} v{transaction.item.version}"
		}


class DigitalTwinMarketplaceDashboardView(DigitalTwinMarketplaceBaseView):
	"""Digital twin marketplace dashboard"""
	
	route_base = "/digital_twin_marketplace_dashboard"
	default_view = "index"
	
	@expose('/')
	@has_access
	def index(self):
		"""Marketplace dashboard main page"""
		try:
			# Get dashboard metrics
			metrics = self._get_dashboard_metrics()
			
			return render_template('digital_twin_marketplace/dashboard.html',
								   metrics=metrics,
								   page_title="Digital Twin Marketplace Dashboard")
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return render_template('digital_twin_marketplace/dashboard.html',
								   metrics={},
								   page_title="Digital Twin Marketplace Dashboard")
	
	@expose('/browse/')
	@has_access
	def browse_items(self):
		"""Browse marketplace items"""
		try:
			# Get filters from request
			category = request.args.get('category')
			item_type = request.args.get('item_type')
			license_type = request.args.get('license_type')
			min_rating = request.args.get('min_rating', type=float)
			
			browse_data = self._get_browse_data(category, item_type, license_type, min_rating)
			
			return render_template('digital_twin_marketplace/browse.html',
								   browse_data=browse_data,
								   page_title="Browse Marketplace")
		except Exception as e:
			flash(f'Error loading browse page: {str(e)}', 'error')
			return redirect(url_for('DigitalTwinMarketplaceDashboardView.index'))
	
	@expose('/analytics/')
	@has_access
	def marketplace_analytics(self):
		"""Marketplace analytics and insights"""
		try:
			period_days = int(request.args.get('period', 30))
			analytics_data = self._get_marketplace_analytics(period_days)
			
			return render_template('digital_twin_marketplace/analytics.html',
								   analytics_data=analytics_data,
								   period_days=period_days,
								   page_title="Marketplace Analytics")
		except Exception as e:
			flash(f'Error loading analytics: {str(e)}', 'error')
			return redirect(url_for('DigitalTwinMarketplaceDashboardView.index'))
	
	@expose('/trending/')
	@has_access
	def trending_items(self):
		"""Trending items in marketplace"""
		try:
			trending_data = self._get_trending_data()
			
			return render_template('digital_twin_marketplace/trending.html',
								   trending_data=trending_data,
								   page_title="Trending Items")
		except Exception as e:
			flash(f'Error loading trending items: {str(e)}', 'error')
			return redirect(url_for('DigitalTwinMarketplaceDashboardView.index'))
	
	def _get_dashboard_metrics(self) -> Dict[str, Any]:
		"""Get marketplace dashboard metrics"""
		# Implementation would calculate real metrics from database
		return {
			'total_items': 1256,
			'active_items': 1089,
			'total_users': 3450,
			'verified_users': 892,
			'total_downloads': 89450,
			'total_revenue': 445750.50,
			'average_rating': 4.3,
			'featured_items': 45,
			'new_items_this_week': 23,
			'active_transactions': 156,
			'top_categories': [
				{'name': 'Manufacturing', 'items': 456, 'downloads': 23450},
				{'name': 'Healthcare', 'items': 234, 'downloads': 18900},
				{'name': 'Smart Cities', 'items': 189, 'downloads': 15670}
			],
			'recent_items': [
				{'name': 'Factory Optimizer Pro', 'author': 'TechCorp', 'rating': 4.8},
				{'name': 'Medical Device Twin', 'author': 'HealthTech', 'rating': 4.6},
				{'name': 'Traffic Flow Manager', 'author': 'CityPlan', 'rating': 4.5}
			],
			'growth_trends': {
				'users': [120, 145, 178, 203, 234, 278, 312],
				'items': [89, 102, 118, 134, 156, 178, 203],
				'revenue': [15600, 18900, 22100, 25800, 29200, 34500, 39800]
			}
		}
	
	def _get_browse_data(self, category=None, item_type=None, license_type=None, min_rating=None) -> Dict[str, Any]:
		"""Get browse page data with filters"""
		return {
			'items': [
				{
					'name': 'Smart Factory Digital Twin',
					'category': 'Manufacturing',
					'author': 'Industrial AI',
					'rating': 4.8,
					'downloads': 1250,
					'price': 299.99,
					'thumbnail': '/static/img/factory-twin.jpg'
				}
			],
			'categories': [
				{'name': 'Manufacturing', 'count': 456},
				{'name': 'Healthcare', 'count': 234},
				{'name': 'Smart Cities', 'count': 189}
			],
			'filters': {
				'category': category,
				'item_type': item_type,
				'license_type': license_type,
				'min_rating': min_rating
			},
			'total_results': 156
		}
	
	def _get_marketplace_analytics(self, period_days: int) -> Dict[str, Any]:
		"""Get marketplace analytics data"""
		return {
			'period_days': period_days,
			'overview': {
				'total_revenue': 445750.50,
				'total_transactions': 2340,
				'average_transaction': 190.49,
				'top_selling_category': 'Manufacturing'
			},
			'revenue_trend': [23400, 28900, 31200, 35600, 39800, 42100, 44575],
			'category_performance': {
				'Manufacturing': {'revenue': 156780, 'items': 456, 'growth': 15.2},
				'Healthcare': {'revenue': 123450, 'items': 234, 'growth': 22.8},
				'Smart Cities': {'revenue': 89320, 'items': 189, 'growth': 8.4}
			},
			'user_engagement': {
				'daily_active_users': 892,
				'session_duration': 24.5,
				'bounce_rate': 0.23,
				'conversion_rate': 0.045
			},
			'quality_metrics': {
				'certified_items': 234,
				'security_scanned': 789,
				'average_quality_score': 87.5
			}
		}
	
	def _get_trending_data(self) -> Dict[str, Any]:
		"""Get trending items data"""
		return {
			'trending_items': [
				{
					'name': 'AI-Powered Factory Optimizer',
					'author': 'AutomationCorp',
					'category': 'Manufacturing',
					'downloads': 456,
					'growth_rate': 45.2,
					'rating': 4.9
				},
				{
					'name': 'Hospital Resource Manager',
					'author': 'MedTech Solutions',
					'category': 'Healthcare',
					'downloads': 234,
					'growth_rate': 38.7,
					'rating': 4.7
				},
				{
					'name': 'Smart Grid Controller',
					'author': 'EnergyTech',
					'category': 'Energy',
					'downloads': 189,
					'growth_rate': 32.1,
					'rating': 4.6
				}
			],
			'trending_categories': [
				{'name': 'AI/ML Algorithms', 'growth': 56.3},
				{'name': 'IoT Integrations', 'growth': 42.8},
				{'name': 'Simulation Engines', 'growth': 38.5}
			],
			'hot_keywords': [
				'artificial intelligence', 'predictive maintenance',
				'real-time analytics', 'edge computing', 'blockchain'
			]
		}


# Register views with AppBuilder
def register_views(appbuilder):
	"""Register all digital twin marketplace views with Flask-AppBuilder"""
	
	# Model views
	appbuilder.add_view(
		DTMUserModelView,
		"Users",
		icon="fa-users",
		category="Digital Twin Marketplace",
		category_icon="fa-store"
	)
	
	appbuilder.add_view(
		DTMCategoryModelView,
		"Categories",
		icon="fa-tags",
		category="Digital Twin Marketplace"
	)
	
	appbuilder.add_view(
		DTMItemModelView,
		"Items",
		icon="fa-cube",
		category="Digital Twin Marketplace"
	)
	
	appbuilder.add_view(
		DTMReviewModelView,
		"Reviews",
		icon="fa-star",
		category="Digital Twin Marketplace"
	)
	
	appbuilder.add_view(
		DTMTransactionModelView,
		"Transactions",
		icon="fa-credit-card",
		category="Digital Twin Marketplace"
	)
	
	# Dashboard views
	appbuilder.add_view_no_menu(DigitalTwinMarketplaceDashboardView)
	
	# Menu links
	appbuilder.add_link(
		"Marketplace Dashboard",
		href="/digital_twin_marketplace_dashboard/",
		icon="fa-dashboard",
		category="Digital Twin Marketplace"
	)
	
	appbuilder.add_link(
		"Browse Items",
		href="/digital_twin_marketplace_dashboard/browse/",
		icon="fa-search",
		category="Digital Twin Marketplace"
	)
	
	appbuilder.add_link(
		"Marketplace Analytics",
		href="/digital_twin_marketplace_dashboard/analytics/",
		icon="fa-chart-line",
		category="Digital Twin Marketplace"
	)
	
	appbuilder.add_link(
		"Trending Items",
		href="/digital_twin_marketplace_dashboard/trending/",
		icon="fa-fire",
		category="Digital Twin Marketplace"
	)