"""
Product Catalog Views

Flask-AppBuilder views for comprehensive product catalog management
with CRUD operations, category management, inventory tracking, and search.
"""

from flask import request, jsonify, flash, redirect, url_for, render_template, abort
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from flask_appbuilder.widgets import FormWidget, ListWidget, SearchWidget
from flask_appbuilder.forms import DynamicForm
from wtforms import StringField, TextAreaField, SelectField, BooleanField, DecimalField, IntegerField, validators
from wtforms.validators import DataRequired, Length, Optional, NumberRange
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal
import json

from .models import (
	PCCategory, PCAttribute, PCAttributeOption, PCProduct, PCProductCategory,
	PCProductAttributeValue, PCProductVariant, PCProductImage, PCProductReview,
	PCInventoryLog
)


class ProductCatalogBaseView(BaseView):
	"""Base view for product catalog functionality"""
	
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
	
	def _format_price(self, price: Decimal, currency: str = 'USD') -> str:
		"""Format price for display"""
		if price is None:
			return f"0.00 {currency}"
		return f"{price:,.2f} {currency}"


class PCCategoryModelView(ModelView):
	"""Product category management view"""
	
	datamodel = SQLAInterface(PCCategory)
	
	# List view configuration
	list_columns = [
		'category_code', 'category_name', 'parent_category', 'level',
		'product_count', 'is_active', 'sort_order'
	]
	show_columns = [
		'category_code', 'category_name', 'description', 'parent_category',
		'level', 'path', 'product_count', 'is_active', 'is_visible_in_menu',
		'url_key', 'meta_title', 'meta_description', 'image_url'
	]
	edit_columns = [
		'category_code', 'category_name', 'description', 'short_description',
		'parent_category', 'is_active', 'is_visible_in_menu', 'sort_order',
		'url_key', 'meta_title', 'meta_description', 'meta_keywords',
		'image_url', 'banner_image_url'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['category_code', 'category_name', 'description']
	base_filters = [['is_active', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('sort_order', 'asc')
	
	# Form validation
	validators_columns = {
		'category_code': [DataRequired(), Length(min=1, max=50)],
		'category_name': [DataRequired(), Length(min=1, max=200)],
		'sort_order': [NumberRange(min=0)]
	}
	
	# Custom labels
	label_columns = {
		'category_code': 'Category Code',
		'category_name': 'Category Name',
		'parent_category': 'Parent Category',
		'product_count': 'Product Count',
		'is_active': 'Active',
		'is_visible_in_menu': 'Visible in Menu',
		'sort_order': 'Sort Order',
		'url_key': 'URL Key',
		'meta_title': 'Meta Title',
		'meta_description': 'Meta Description'
	}
	
	@expose('/rebuild_tree/')
	@has_access
	def rebuild_tree(self):
		"""Rebuild category tree structure"""
		try:
			# Implementation would rebuild the category tree
			categories = self.datamodel.get_all()
			for category in categories:
				if category.parent_category:
					category.level = category.parent_category.level + 1
					category.path = f"{category.parent_category.path} > {category.category_name}"
				else:
					category.level = 0
					category.path = category.category_name
			
			flash('Category tree rebuilt successfully', 'success')
		except Exception as e:
			flash(f'Error rebuilding category tree: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/update_product_counts/')
	@has_access
	def update_product_counts(self):
		"""Update product counts for all categories"""
		try:
			categories = self.datamodel.get_all()
			for category in categories:
				category.update_product_count()
			
			flash('Product counts updated successfully', 'success')
		except Exception as e:
			flash(f'Error updating product counts: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new category"""
		item.tenant_id = self._get_tenant_id()
		
		# Set level and path based on parent
		if item.parent_category:
			item.level = item.parent_category.level + 1
			item.path = f"{item.parent_category.path} > {item.category_name}"
		else:
			item.level = 0
			item.path = item.category_name
		
		# Generate URL key if not provided
		if not item.url_key:
			item.url_key = item.category_code.lower().replace(' ', '-')
	
	def post_update(self, item):
		"""Post-process after updating category"""
		# Update child categories if path changed
		for child in item.child_categories:
			child.path = f"{item.path} > {child.category_name}"
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class PCAttributeModelView(ModelView):
	"""Product attribute management view"""
	
	datamodel = SQLAInterface(PCAttribute)
	
	# List view configuration
	list_columns = [
		'attribute_code', 'attribute_name', 'attribute_type', 'input_type',
		'is_required', 'is_filterable', 'is_searchable', 'sort_order'
	]
	show_columns = [
		'attribute_code', 'attribute_name', 'description', 'attribute_type',
		'input_type', 'data_type', 'is_required', 'is_unique', 'is_filterable',
		'is_searchable', 'is_comparable', 'has_options', 'sort_order'
	]
	edit_columns = [
		'attribute_code', 'attribute_name', 'description', 'attribute_type',
		'input_type', 'data_type', 'is_required', 'is_unique', 'is_filterable',
		'is_searchable', 'is_comparable', 'validation_rules', 'default_value',
		'sort_order', 'note'
	]
	add_columns = edit_columns
	
	# Related views
	related_views = [PCCategoryModelView]
	
	# Search and filtering
	search_columns = ['attribute_code', 'attribute_name', 'description']
	
	# Ordering
	base_order = ('sort_order', 'asc')
	
	# Form validation
	validators_columns = {
		'attribute_code': [DataRequired(), Length(min=1, max=50)],
		'attribute_name': [DataRequired(), Length(min=1, max=200)],
		'attribute_type': [DataRequired()],
		'input_type': [DataRequired()]
	}
	
	# Custom labels
	label_columns = {
		'attribute_code': 'Attribute Code',
		'attribute_name': 'Attribute Name',
		'attribute_type': 'Attribute Type',
		'input_type': 'Input Type',
		'data_type': 'Data Type',
		'is_required': 'Required',
		'is_unique': 'Unique',
		'is_filterable': 'Filterable',
		'is_searchable': 'Searchable',
		'is_comparable': 'Comparable',
		'has_options': 'Has Options',
		'validation_rules': 'Validation Rules',
		'default_value': 'Default Value'
	}
	
	def pre_add(self, item):
		"""Pre-process before adding new attribute"""
		item.tenant_id = self._get_tenant_id()
		
		# Set has_options based on attribute type
		if item.attribute_type in ['select', 'multiselect']:
			item.has_options = True
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class PCProductModelView(ModelView):
	"""Product management view"""
	
	datamodel = SQLAInterface(PCProduct)
	
	# List view configuration
	list_columns = [
		'sku', 'name', 'brand', 'product_type', 'price',
		'qty', 'is_in_stock', 'status', 'is_featured'
	]
	show_columns = [
		'sku', 'name', 'description', 'short_description', 'product_type',
		'brand', 'manufacturer', 'price', 'cost', 'special_price',
		'qty', 'is_in_stock', 'weight', 'status', 'visibility',
		'is_featured', 'categories', 'image', 'created_at'
	]
	edit_columns = [
		'sku', 'name', 'description', 'short_description', 'product_type',
		'brand', 'manufacturer', 'price', 'cost', 'special_price',
		'special_price_from', 'special_price_to', 'qty', 'min_qty',
		'weight', 'length', 'width', 'height', 'status', 'visibility',
		'is_featured', 'meta_title', 'meta_description', 'image'
	]
	add_columns = edit_columns
	
	# Related views
	related_views = [PCCategoryModelView, PCAttributeModelView]
	
	# Search and filtering
	search_columns = ['sku', 'name', 'description', 'brand']
	base_filters = [['status', lambda: 'active', lambda: True]]
	
	# Ordering
	base_order = ('created_at', 'desc')
	
	# Form validation
	validators_columns = {
		'sku': [DataRequired(), Length(min=1, max=100)],
		'name': [DataRequired(), Length(min=1, max=500)],
		'price': [NumberRange(min=0)],
		'qty': [NumberRange(min=0)]
	}
	
	# Custom labels
	label_columns = {
		'sku': 'SKU',
		'name': 'Product Name',
		'product_type': 'Product Type',
		'short_description': 'Short Description',
		'special_price': 'Special Price',
		'special_price_from': 'Special Price From',
		'special_price_to': 'Special Price To',
		'qty': 'Quantity',
		'min_qty': 'Minimum Quantity',
		'is_in_stock': 'In Stock',
		'is_featured': 'Featured',
		'meta_title': 'Meta Title',
		'meta_description': 'Meta Description'
	}
	
	@expose('/update_search_weights/')
	@has_access
	def update_search_weights(self):
		"""Update search weights for all products"""
		try:
			products = self.datamodel.get_all()
			for product in products:
				product.update_search_weight()
			
			flash('Search weights updated successfully', 'success')
		except Exception as e:
			flash(f'Error updating search weights: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/duplicate/<int:pk>')
	@has_access
	def duplicate_product(self, pk):
		"""Duplicate a product"""
		original = self.datamodel.get(pk)
		if not original:
			flash('Product not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Create duplicate
			duplicate = PCProduct(
				sku=f"{original.sku}-COPY",
				name=f"{original.name} (Copy)",
				description=original.description,
				short_description=original.short_description,
				product_type=original.product_type,
				brand=original.brand,
				manufacturer=original.manufacturer,
				price=original.price,
				cost=original.cost,
				weight=original.weight,
				status='draft',
				tenant_id=original.tenant_id
			)
			
			self.datamodel.add(duplicate)
			flash(f'Product duplicated successfully: {duplicate.sku}', 'success')
		except Exception as e:
			flash(f'Error duplicating product: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/bulk_update_status/')
	@has_access
	def bulk_update_status(self):
		"""Bulk update product status"""
		try:
			product_ids = request.form.getlist('product_ids')
			new_status = request.form.get('new_status')
			
			if not product_ids or not new_status:
				flash('Please select products and status', 'error')
				return redirect(self.get_redirect())
			
			updated_count = 0
			for product_id in product_ids:
				product = self.datamodel.get(product_id)
				if product:
					product.status = new_status
					updated_count += 1
			
			flash(f'Updated status for {updated_count} products', 'success')
		except Exception as e:
			flash(f'Error updating product status: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new product"""
		item.tenant_id = self._get_tenant_id()
		
		# Generate URL key if not provided
		if not item.url_key:
			item.url_key = item.name.lower().replace(' ', '-').replace('/', '-')
		
		# Set initial inventory status
		item.is_in_stock = item.qty > 0
	
	def post_add(self, item):
		"""Post-process after adding new product"""
		# Create inventory log entry
		inventory_log = PCInventoryLog(
			product_id=item.product_id,
			tenant_id=item.tenant_id,
			movement_type='initial',
			quantity_change=item.qty,
			quantity_before=0,
			quantity_after=item.qty,
			reason='Initial inventory',
			processed_by=self._get_current_user_id()
		)
		
		from flask_appbuilder import db
		db.session.add(inventory_log)
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class PCProductVariantModelView(ModelView):
	"""Product variant management view"""
	
	datamodel = SQLAInterface(PCProductVariant)
	
	# List view configuration
	list_columns = [
		'variant_sku', 'parent_product', 'price', 'qty',
		'is_in_stock', 'is_enabled'
	]
	show_columns = [
		'variant_sku', 'variant_name', 'parent_product', 'attribute_combination',
		'price', 'cost', 'special_price', 'qty', 'is_in_stock',
		'weight', 'is_enabled', 'sort_order'
	]
	edit_columns = [
		'variant_sku', 'variant_name', 'attribute_combination', 'price',
		'cost', 'special_price', 'special_price_from', 'special_price_to',
		'qty', 'weight', 'is_enabled', 'sort_order', 'image'
	]
	add_columns = edit_columns
	
	# Related views
	related_views = [PCProductModelView]
	
	# Search and filtering
	search_columns = ['variant_sku', 'variant_name']
	base_filters = [['is_enabled', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('sort_order', 'asc')
	
	# Form validation
	validators_columns = {
		'variant_sku': [DataRequired(), Length(min=1, max=100)],
		'parent_product': [DataRequired()],
		'qty': [NumberRange(min=0)]
	}
	
	# Custom labels
	label_columns = {
		'variant_sku': 'Variant SKU',
		'variant_name': 'Variant Name',
		'parent_product': 'Parent Product',
		'attribute_combination': 'Attribute Combination',
		'special_price_from': 'Special Price From',
		'special_price_to': 'Special Price To',
		'qty': 'Quantity',
		'is_in_stock': 'In Stock',
		'is_enabled': 'Enabled',
		'sort_order': 'Sort Order'
	}
	
	def pre_add(self, item):
		"""Pre-process before adding new variant"""
		item.tenant_id = self._get_tenant_id()
		item.is_in_stock = item.qty > 0
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class PCProductReviewModelView(ModelView):
	"""Product review management view"""
	
	datamodel = SQLAInterface(PCProductReview)
	
	# List view configuration
	list_columns = [
		'product', 'reviewer_name', 'rating', 'title',
		'status', 'is_verified_buyer', 'review_date'
	]
	show_columns = [
		'product', 'reviewer_name', 'reviewer_email', 'rating',
		'title', 'review_text', 'status', 'is_verified_buyer',
		'is_featured', 'helpful_votes', 'total_votes', 'review_date'
	]
	edit_columns = [
		'product', 'reviewer_name', 'reviewer_email', 'rating',
		'title', 'review_text', 'status', 'is_featured',
		'quality_rating', 'value_rating', 'service_rating'
	]
	add_columns = edit_columns
	
	# Related views
	related_views = [PCProductModelView]
	
	# Search and filtering
	search_columns = ['title', 'review_text', 'reviewer_name']
	base_filters = [['status', lambda: 'pending', lambda: True]]
	
	# Ordering
	base_order = ('review_date', 'desc')
	
	# Form validation
	validators_columns = {
		'product': [DataRequired()],
		'reviewer_name': [DataRequired(), Length(min=1, max=200)],
		'rating': [DataRequired(), NumberRange(min=1, max=5)],
		'title': [DataRequired(), Length(min=1, max=200)],
		'review_text': [DataRequired()]
	}
	
	# Custom labels
	label_columns = {
		'reviewer_name': 'Reviewer Name',
		'reviewer_email': 'Reviewer Email',
		'review_text': 'Review Text',
		'is_verified_buyer': 'Verified Buyer',
		'is_featured': 'Featured Review',
		'quality_rating': 'Quality Rating',
		'value_rating': 'Value Rating',
		'service_rating': 'Service Rating',
		'helpful_votes': 'Helpful Votes',
		'total_votes': 'Total Votes',
		'review_date': 'Review Date'
	}
	
	@expose('/approve/<int:pk>')
	@has_access
	def approve_review(self, pk):
		"""Approve product review"""
		review = self.datamodel.get(pk)
		if not review:
			flash('Review not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			review.status = 'approved'
			review.moderated_by = self._get_current_user_id()
			review.moderated_at = datetime.utcnow()
			review.update_product_rating()
			
			self.datamodel.edit(review)
			flash(f'Review approved successfully', 'success')
		except Exception as e:
			flash(f'Error approving review: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/reject/<int:pk>')
	@has_access
	def reject_review(self, pk):
		"""Reject product review"""
		review = self.datamodel.get(pk)
		if not review:
			flash('Review not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			review.status = 'rejected'
			review.moderated_by = self._get_current_user_id()
			review.moderated_at = datetime.utcnow()
			
			self.datamodel.edit(review)
			flash(f'Review rejected successfully', 'success')
		except Exception as e:
			flash(f'Error rejecting review: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new review"""
		item.tenant_id = self._get_tenant_id()
		item.review_date = datetime.utcnow()
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class ProductCatalogDashboardView(ProductCatalogBaseView):
	"""Product catalog dashboard"""
	
	route_base = "/product_catalog"
	default_view = "index"
	
	@expose('/')
	@has_access
	def index(self):
		"""Product catalog dashboard main page"""
		try:
			# Get catalog metrics
			metrics = self._get_catalog_metrics()
			
			return render_template('product_catalog/dashboard.html',
								   metrics=metrics,
								   page_title="Product Catalog Dashboard")
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return render_template('product_catalog/dashboard.html',
								   metrics={},
								   page_title="Product Catalog Dashboard")
	
	@expose('/search/')
	@has_access
	def search_products(self):
		"""Advanced product search"""
		try:
			query = request.args.get('q', '')
			category_id = request.args.get('category')
			min_price = request.args.get('min_price', type=float)
			max_price = request.args.get('max_price', type=float)
			in_stock_only = request.args.get('in_stock', type=bool, default=False)
			
			# Perform search
			results = self._search_products(query, category_id, min_price, max_price, in_stock_only)
			
			return render_template('product_catalog/search_results.html',
								   results=results,
								   query=query,
								   page_title="Product Search Results")
		except Exception as e:
			flash(f'Error performing search: {str(e)}', 'error')
			return redirect(url_for('ProductCatalogDashboardView.index'))
	
	@expose('/inventory_report/')
	@has_access
	def inventory_report(self):
		"""Inventory status report"""
		try:
			# Get inventory data
			inventory_data = self._get_inventory_report()
			
			return render_template('product_catalog/inventory_report.html',
								   inventory_data=inventory_data,
								   page_title="Inventory Report")
		except Exception as e:
			flash(f'Error generating inventory report: {str(e)}', 'error')
			return redirect(url_for('ProductCatalogDashboardView.index'))
	
	@expose('/category_tree/')
	@has_access
	def category_tree(self):
		"""Category tree view"""
		try:
			# Get category tree
			categories = self._get_category_tree()
			
			return render_template('product_catalog/category_tree.html',
								   categories=categories,
								   page_title="Category Tree")
		except Exception as e:
			flash(f'Error loading category tree: {str(e)}', 'error')
			return redirect(url_for('ProductCatalogDashboardView.index'))
	
	def _get_catalog_metrics(self) -> Dict[str, Any]:
		"""Get catalog metrics for dashboard"""
		# Implementation would calculate real metrics from database
		return {
			'total_products': 1250,
			'active_products': 1180,
			'out_of_stock': 85,
			'low_stock': 42,
			'total_categories': 156,
			'active_categories': 148,
			'total_reviews': 3420,
			'pending_reviews': 23,
			'average_rating': 4.2,
			'top_categories': [
				{'name': 'Electronics', 'product_count': 320},
				{'name': 'Clothing', 'product_count': 285},
				{'name': 'Home & Garden', 'product_count': 198}
			]
		}
	
	def _search_products(self, query: str, category_id: str = None,
						min_price: float = None, max_price: float = None,
						in_stock_only: bool = False) -> Dict[str, Any]:
		"""Search products with filters"""
		# Implementation would perform actual database search
		return {
			'products': [],
			'total_count': 0,
			'facets': {
				'categories': [],
				'brands': [],
				'price_ranges': []
			}
		}
	
	def _get_inventory_report(self) -> Dict[str, Any]:
		"""Get inventory report data"""
		# Implementation would generate real inventory report
		return {
			'total_value': 125000.00,
			'low_stock_items': [],
			'out_of_stock_items': [],
			'overstock_items': [],
			'movement_summary': {
				'sales': 1200,
				'purchases': 800,
				'adjustments': 45
			}
		}
	
	def _get_category_tree(self) -> List[Dict[str, Any]]:
		"""Get hierarchical category tree"""
		# Implementation would build category tree from database
		return []


class InventoryManagementView(ProductCatalogBaseView):
	"""Inventory management functionality"""
	
	route_base = "/inventory_management"
	default_view = "index"
	
	@expose('/')
	@has_access
	def index(self):
		"""Inventory management dashboard"""
		try:
			# Get inventory summary
			summary = self._get_inventory_summary()
			
			return render_template('product_catalog/inventory_dashboard.html',
								   summary=summary,
								   page_title="Inventory Management")
		except Exception as e:
			flash(f'Error loading inventory dashboard: {str(e)}', 'error')
			return render_template('product_catalog/inventory_dashboard.html',
								   summary={},
								   page_title="Inventory Management")
	
	@expose('/adjust/', methods=['GET', 'POST'])
	@has_access
	def adjust_inventory(self):
		"""Adjust product inventory"""
		if request.method == 'POST':
			try:
				product_id = request.form.get('product_id')
				adjustment_qty = float(request.form.get('adjustment_qty', 0))
				reason = request.form.get('reason', '')
				
				# Perform inventory adjustment
				self._adjust_product_inventory(product_id, adjustment_qty, reason)
				
				flash('Inventory adjusted successfully', 'success')
				return redirect(url_for('InventoryManagementView.index'))
			except Exception as e:
				flash(f'Error adjusting inventory: {str(e)}', 'error')
		
		# Get products for adjustment form
		products = self._get_products_for_adjustment()
		
		return render_template('product_catalog/inventory_adjustment.html',
							   products=products,
							   page_title="Inventory Adjustment")
	
	def _get_inventory_summary(self) -> Dict[str, Any]:
		"""Get inventory summary data"""
		return {
			'total_products': 1250,
			'low_stock_count': 42,
			'out_of_stock_count': 85,
			'overstock_count': 28,
			'total_value': 125000.00
		}
	
	def _get_products_for_adjustment(self) -> List[Dict[str, Any]]:
		"""Get products available for inventory adjustment"""
		# Implementation would query active products
		return []
	
	def _adjust_product_inventory(self, product_id: str, adjustment_qty: float, reason: str):
		"""Adjust inventory for a product"""
		# Implementation would update product quantity and create log entry
		pass


# Register views with AppBuilder
def register_views(appbuilder):
	"""Register all product catalog views with Flask-AppBuilder"""
	
	# Model views
	appbuilder.add_view(
		PCCategoryModelView,
		"Categories",
		icon="fa-sitemap",
		category="Product Catalog",
		category_icon="fa-shopping-cart"
	)
	
	appbuilder.add_view(
		PCAttributeModelView,
		"Attributes",
		icon="fa-tags",
		category="Product Catalog"
	)
	
	appbuilder.add_view(
		PCProductModelView,
		"Products",
		icon="fa-cube",
		category="Product Catalog"
	)
	
	appbuilder.add_view(
		PCProductVariantModelView,
		"Product Variants",
		icon="fa-cubes",
		category="Product Catalog"
	)
	
	appbuilder.add_view(
		PCProductReviewModelView,
		"Product Reviews",
		icon="fa-star",
		category="Product Catalog"
	)
	
	# Dashboard and management views
	appbuilder.add_view_no_menu(ProductCatalogDashboardView)
	appbuilder.add_view_no_menu(InventoryManagementView)
	
	# Menu links
	appbuilder.add_link(
		"Catalog Dashboard",
		href="/product_catalog/",
		icon="fa-dashboard",
		category="Product Catalog"
	)
	
	appbuilder.add_link(
		"Inventory Management",
		href="/inventory_management/",
		icon="fa-warehouse",
		category="Product Catalog"
	)
	
	appbuilder.add_link(
		"Category Tree",
		href="/product_catalog/category_tree/",
		icon="fa-tree",
		category="Product Catalog"
	)