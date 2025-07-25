"""
Product Catalog Models

Database models for comprehensive product catalog management including
multi-variant products, advanced search, inventory integration, and recommendations.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey, Numeric
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
from decimal import Decimal
import json

from ..auth_rbac.models import BaseMixin, AuditMixin, Model


def uuid7str():
	"""Generate UUID7 string for consistent ID generation"""
	from uuid_extensions import uuid7
	return str(uuid7())


class PCCategory(Model, AuditMixin, BaseMixin):
	"""
	Product categories with hierarchical structure.
	
	Manages product categorization with SEO optimization,
	merchandising features, and multi-level hierarchy support.
	"""
	__tablename__ = 'pc_category'
	
	# Identity
	category_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Category Information
	category_code = Column(String(50), nullable=False, unique=True, index=True)
	category_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	short_description = Column(String(500), nullable=True)
	
	# Hierarchy
	parent_category_id = Column(String(36), ForeignKey('pc_category.category_id'), nullable=True, index=True)
	level = Column(Integer, default=0)  # Hierarchy level (0 = root)
	path = Column(String(1000), nullable=True)  # Full hierarchical path
	children_count = Column(Integer, default=0)
	
	# Status and Visibility
	is_active = Column(Boolean, default=True, index=True)
	is_visible_in_menu = Column(Boolean, default=True)
	is_anchor = Column(Boolean, default=False)  # Can be filtered
	include_in_menu = Column(Boolean, default=True)
	
	# Display and Ordering
	sort_order = Column(Integer, default=0)
	display_mode = Column(String(20), default='products')  # products, page, both
	cms_block = Column(Text, nullable=True)  # Custom content block
	
	# SEO and Meta Information
	url_key = Column(String(200), nullable=True, unique=True, index=True)
	meta_title = Column(String(200), nullable=True)
	meta_description = Column(Text, nullable=True)
	meta_keywords = Column(JSON, default=list)
	canonical_url = Column(String(500), nullable=True)
	
	# Images and Media
	image_url = Column(String(500), nullable=True)
	thumbnail_url = Column(String(500), nullable=True)
	banner_image_url = Column(String(500), nullable=True)
	gallery_images = Column(JSON, default=list)
	
	# Category Attributes
	custom_attributes = Column(JSON, default=dict)
	category_filters = Column(JSON, default=list)  # Available filters for this category
	
	# Product Association
	product_count = Column(Integer, default=0)
	automatic_sorting = Column(Boolean, default=True)
	default_product_sort = Column(String(50), default='position')  # position, name, price, created_at
	
	# Merchandising
	featured_products = Column(JSON, default=list)  # Featured product IDs
	cross_sell_categories = Column(JSON, default=list)  # Related category IDs
	
	# Relationships
	parent_category = relationship("PCCategory", remote_side=[category_id], back_populates="child_categories")
	child_categories = relationship("PCCategory", back_populates="parent_category", cascade="all, delete-orphan")
	products = relationship("PCProduct", secondary="pc_product_category", back_populates="categories")
	
	def __repr__(self):
		return f"<PCCategory {self.category_code}: {self.category_name}>"
	
	def get_full_path(self) -> str:
		"""Get full category path"""
		if self.parent_category:
			return f"{self.parent_category.get_full_path()} > {self.category_name}"
		return self.category_name
	
	def get_url_path(self) -> str:
		"""Get SEO-friendly URL path"""
		if self.parent_category and self.parent_category.url_key:
			return f"{self.parent_category.get_url_path()}/{self.url_key}"
		return self.url_key or self.category_code.lower()
	
	def update_product_count(self) -> None:
		"""Update product count for this category"""
		# This would query associated products
		self.product_count = len(self.products)
	
	def get_all_child_categories(self) -> List['PCCategory']:
		"""Get all descendant categories"""
		children = list(self.child_categories)
		for child in self.child_categories:
			children.extend(child.get_all_child_categories())
		return children
	
	def is_leaf_category(self) -> bool:
		"""Check if category is a leaf (no children)"""
		return self.children_count == 0
	
	def can_have_products(self) -> bool:
		"""Check if category can contain products"""
		return self.is_active and (self.is_leaf_category() or self.is_anchor)


class PCAttribute(Model, AuditMixin, BaseMixin):
	"""
	Product attributes for flexible product data modeling.
	
	Defines custom attributes for products with validation,
	filtering, search, and comparison capabilities.
	"""
	__tablename__ = 'pc_attribute'
	
	# Identity
	attribute_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Attribute Definition
	attribute_code = Column(String(50), nullable=False, unique=True, index=True)
	attribute_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Attribute Type and Validation
	attribute_type = Column(String(20), nullable=False, index=True)  # text, number, boolean, date, select, etc.
	input_type = Column(String(20), nullable=False)  # text, textarea, select, multiselect, boolean, date, etc.
	data_type = Column(String(20), default='varchar')  # varchar, text, int, decimal, datetime, boolean
	
	# Validation Rules
	is_required = Column(Boolean, default=False)
	is_unique = Column(Boolean, default=False)
	validation_rules = Column(JSON, default=dict)  # Min/max length, regex, etc.
	default_value = Column(Text, nullable=True)
	
	# Display and Behavior
	is_visible_on_front = Column(Boolean, default=True)
	is_filterable = Column(Boolean, default=False)
	is_filterable_in_search = Column(Boolean, default=False)
	is_searchable = Column(Boolean, default=False)
	is_comparable = Column(Boolean, default=False)
	is_used_for_promo_rules = Column(Boolean, default=False)
	
	# Attribute Options (for select/multiselect)
	has_options = Column(Boolean, default=False)
	
	# Display Configuration
	sort_order = Column(Integer, default=0)
	frontend_class = Column(String(100), nullable=True)  # CSS class for frontend
	note = Column(Text, nullable=True)  # Help text
	
	# Global vs Store-specific
	is_global = Column(Boolean, default=True)
	scope = Column(String(20), default='global')  # global, website, store
	
	# System Attributes
	is_system = Column(Boolean, default=False)  # System-defined attribute
	is_user_defined = Column(Boolean, default=True)
	
	# Relationships
	attribute_options = relationship("PCAttributeOption", back_populates="attribute", cascade="all, delete-orphan")
	product_values = relationship("PCProductAttributeValue", back_populates="attribute")
	
	def __repr__(self):
		return f"<PCAttribute {self.attribute_code}: {self.attribute_name}>"
	
	def get_options_list(self) -> List[Dict[str, Any]]:
		"""Get attribute options as list"""
		return [
			{
				'option_id': option.option_id,
				'value': option.value,
				'label': option.label,
				'sort_order': option.sort_order
			}
			for option in sorted(self.attribute_options, key=lambda x: x.sort_order)
		]
	
	def validate_value(self, value: Any) -> bool:
		"""Validate attribute value against rules"""
		if self.is_required and (value is None or value == ''):
			return False
		
		if value is None:
			return True
		
		# Type-specific validation
		if self.attribute_type == 'number':
			try:
				float(value)
			except (ValueError, TypeError):
				return False
		
		elif self.attribute_type == 'boolean':
			if not isinstance(value, bool):
				return False
		
		elif self.attribute_type == 'select' and self.has_options:
			valid_values = [opt.value for opt in self.attribute_options]
			if value not in valid_values:
				return False
		
		# Validation rules
		rules = self.validation_rules or {}
		
		if 'min_length' in rules and len(str(value)) < rules['min_length']:
			return False
		
		if 'max_length' in rules and len(str(value)) > rules['max_length']:
			return False
		
		if 'regex' in rules:
			import re
			if not re.match(rules['regex'], str(value)):
				return False
		
		return True


class PCAttributeOption(Model, AuditMixin, BaseMixin):
	"""
	Options for select-type attributes.
	
	Stores predefined values for select and multiselect attributes
	with multilingual support and ordering.
	"""
	__tablename__ = 'pc_attribute_option'
	
	# Identity
	option_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	attribute_id = Column(String(36), ForeignKey('pc_attribute.attribute_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Option Details
	value = Column(String(255), nullable=False)  # Internal value
	label = Column(String(255), nullable=False)  # Display label
	sort_order = Column(Integer, default=0)
	
	# Additional Properties
	is_default = Column(Boolean, default=False)
	color_code = Column(String(7), nullable=True)  # For color attributes
	image_url = Column(String(500), nullable=True)  # For visual options
	
	# Relationships
	attribute = relationship("PCAttribute", back_populates="attribute_options")
	
	def __repr__(self):
		return f"<PCAttributeOption {self.label} ({self.value})>"


class PCProduct(Model, AuditMixin, BaseMixin):
	"""
	Core product entity with comprehensive product management.
	
	Stores product information with multi-variant support,
	pricing, inventory integration, and SEO optimization.
	"""
	__tablename__ = 'pc_product'
	
	# Identity
	product_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Product Identification
	sku = Column(String(100), nullable=False, unique=True, index=True)
	name = Column(String(500), nullable=False)
	short_description = Column(Text, nullable=True)
	description = Column(Text, nullable=True)
	
	# Product Classification
	product_type = Column(String(20), default='simple', index=True)  # simple, configurable, grouped, bundle, etc.
	attribute_set_id = Column(String(36), nullable=True, index=True)
	manufacturer = Column(String(200), nullable=True)
	brand = Column(String(200), nullable=True, index=True)
	
	# Status and Visibility
	status = Column(String(20), default='active', index=True)  # active, inactive, draft, discontinued
	visibility = Column(String(20), default='catalog_and_search')  # catalog_and_search, catalog, search, not_visible
	is_featured = Column(Boolean, default=False, index=True)
	is_new = Column(Boolean, default=False)
	
	# Pricing
	price = Column(Numeric(12, 4), nullable=True)
	cost = Column(Numeric(12, 4), nullable=True)
	msrp = Column(Numeric(12, 4), nullable=True)  # Manufacturer's Suggested Retail Price
	special_price = Column(Numeric(12, 4), nullable=True)
	special_price_from = Column(DateTime, nullable=True)
	special_price_to = Column(DateTime, nullable=True)
	
	# Pricing Configuration
	pricing_type = Column(String(20), default='fixed')  # fixed, dynamic, tiered
	tax_class_id = Column(String(36), nullable=True)
	price_rules = Column(JSON, default=list)  # Complex pricing rules
	
	# Inventory
	qty = Column(Numeric(12, 4), default=0)
	min_qty = Column(Numeric(12, 4), default=0)
	use_config_min_qty = Column(Boolean, default=True)
	is_qty_decimal = Column(Boolean, default=False)
	backorders = Column(String(20), default='no')  # no, yes, notify
	min_sale_qty = Column(Numeric(12, 4), default=1)
	max_sale_qty = Column(Numeric(12, 4), nullable=True)
	is_in_stock = Column(Boolean, default=True, index=True)
	
	# Physical Properties
	weight = Column(Numeric(12, 4), nullable=True)
	length = Column(Numeric(12, 4), nullable=True)
	width = Column(Numeric(12, 4), nullable=True)
	height = Column(Numeric(12, 4), nullable=True)
	volume = Column(Numeric(12, 4), nullable=True)
	
	# SEO and URLs
	url_key = Column(String(200), nullable=True, unique=True, index=True)
	url_path = Column(String(500), nullable=True, index=True)
	meta_title = Column(String(200), nullable=True)
	meta_description = Column(Text, nullable=True)
	meta_keywords = Column(JSON, default=list)
	canonical_url = Column(String(500), nullable=True)
	
	# Images and Media
	image = Column(String(500), nullable=True)  # Main product image
	small_image = Column(String(500), nullable=True)
	thumbnail = Column(String(500), nullable=True)
	gallery_images = Column(JSON, default=list)
	video_url = Column(String(500), nullable=True)
	
	# Product Relations
	related_products = Column(JSON, default=list)  # Related product IDs
	upsell_products = Column(JSON, default=list)   # Upsell product IDs
	cross_sell_products = Column(JSON, default=list)  # Cross-sell product IDs
	
	# Dates
	news_from_date = Column(DateTime, nullable=True)
	news_to_date = Column(DateTime, nullable=True)
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	
	# Performance Metrics
	view_count = Column(Integer, default=0)
	sale_count = Column(Integer, default=0)
	review_count = Column(Integer, default=0)
	rating_average = Column(Numeric(3, 2), default=0.00)
	
	# Search and Filtering
	search_weight = Column(Integer, default=0)  # Search ranking weight
	filter_attributes = Column(JSON, default=dict)  # Cached filterable attributes
	
	# Configuration
	has_options = Column(Boolean, default=False)  # Has configurable options
	required_options = Column(Boolean, default=False)  # All options required
	custom_options = Column(JSON, default=list)  # Custom product options
	
	# Relationships
	categories = relationship("PCCategory", secondary="pc_product_category", back_populates="products")
	attribute_values = relationship("PCProductAttributeValue", back_populates="product", cascade="all, delete-orphan")
	variants = relationship("PCProductVariant", back_populates="parent_product", cascade="all, delete-orphan")
	images = relationship("PCProductImage", back_populates="product", cascade="all, delete-orphan")
	reviews = relationship("PCProductReview", back_populates="product", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<PCProduct {self.sku}: {self.name}>"
	
	def is_available(self) -> bool:
		"""Check if product is available for purchase"""
		return (self.status == 'active' and 
				self.is_in_stock and 
				self.qty > 0)
	
	def is_on_sale(self) -> bool:
		"""Check if product is currently on sale"""
		if not self.special_price:
			return False
		
		now = datetime.utcnow()
		
		if self.special_price_from and now < self.special_price_from:
			return False
		
		if self.special_price_to and now > self.special_price_to:
			return False
		
		return self.special_price < self.price
	
	def get_effective_price(self) -> Decimal:
		"""Get current effective price (considering sales)"""
		if self.is_on_sale():
			return Decimal(str(self.special_price))
		return Decimal(str(self.price or 0))
	
	def get_discount_percentage(self) -> float:
		"""Get discount percentage if on sale"""
		if not self.is_on_sale() or not self.price:
			return 0.0
		
		return float(((self.price - self.special_price) / self.price) * 100)
	
	def update_search_weight(self) -> None:
		"""Update search weight based on performance metrics"""
		# Simple algorithm - can be enhanced
		weight = 0
		weight += self.view_count * 0.1
		weight += self.sale_count * 5
		weight += self.review_count * 2
		weight += float(self.rating_average) * 10
		
		if self.is_featured:
			weight += 100
		
		self.search_weight = int(weight)
	
	def get_attribute_value(self, attribute_code: str) -> Any:
		"""Get value for specific attribute"""
		for attr_value in self.attribute_values:
			if attr_value.attribute.attribute_code == attribute_code:
				return attr_value.get_typed_value()
		return None
	
	def has_variant_options(self) -> bool:
		"""Check if product has configurable variants"""
		return self.product_type == 'configurable' and len(self.variants) > 0
	
	def get_price_range(self) -> Dict[str, Decimal]:
		"""Get price range for configurable products"""
		if not self.has_variant_options():
			price = self.get_effective_price()
			return {'min_price': price, 'max_price': price}
		
		prices = [variant.get_effective_price() for variant in self.variants if variant.is_enabled]
		if not prices:
			price = self.get_effective_price()
			return {'min_price': price, 'max_price': price}
		
		return {'min_price': min(prices), 'max_price': max(prices)}


# Association table for product-category many-to-many relationship
class PCProductCategory(Model):
	"""Association table for product-category relationships"""
	__tablename__ = 'pc_product_category'
	
	product_id = Column(String(36), ForeignKey('pc_product.product_id'), primary_key=True)
	category_id = Column(String(36), ForeignKey('pc_category.category_id'), primary_key=True)
	position = Column(Integer, default=0)  # Sort position within category
	is_primary = Column(Boolean, default=False)  # Primary category for product


class PCProductAttributeValue(Model, AuditMixin, BaseMixin):
	"""
	Product attribute values storage.
	
	Stores actual attribute values for products with type-specific
	handling and efficient querying support.
	"""
	__tablename__ = 'pc_product_attribute_value'
	
	# Identity
	value_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	product_id = Column(String(36), ForeignKey('pc_product.product_id'), nullable=False, index=True)
	attribute_id = Column(String(36), ForeignKey('pc_attribute.attribute_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Value Storage (polymorphic based on attribute type)
	value_varchar = Column(String(500), nullable=True, index=True)
	value_text = Column(Text, nullable=True)
	value_int = Column(Integer, nullable=True, index=True)
	value_decimal = Column(Numeric(12, 4), nullable=True, index=True)
	value_datetime = Column(DateTime, nullable=True, index=True)
	value_boolean = Column(Boolean, nullable=True, index=True)
	value_json = Column(JSON, nullable=True)
	
	# Relationships
	product = relationship("PCProduct", back_populates="attribute_values")
	attribute = relationship("PCAttribute", back_populates="product_values")
	
	def __repr__(self):
		return f"<PCProductAttributeValue {self.attribute.attribute_code}: {self.get_value()}>"
	
	def get_value(self) -> Any:
		"""Get typed value based on attribute type"""
		attr_type = self.attribute.data_type
		
		if attr_type == 'varchar':
			return self.value_varchar
		elif attr_type == 'text':
			return self.value_text
		elif attr_type == 'int':
			return self.value_int
		elif attr_type == 'decimal':
			return self.value_decimal
		elif attr_type == 'datetime':
			return self.value_datetime
		elif attr_type == 'boolean':
			return self.value_boolean
		elif attr_type == 'json':
			return self.value_json
		
		return self.value_varchar
	
	def set_value(self, value: Any) -> None:
		"""Set value in appropriate column based on attribute type"""
		attr_type = self.attribute.data_type
		
		# Clear all value fields first
		self.value_varchar = None
		self.value_text = None
		self.value_int = None
		self.value_decimal = None
		self.value_datetime = None
		self.value_boolean = None
		self.value_json = None
		
		# Set value in appropriate field
		if attr_type == 'varchar':
			self.value_varchar = str(value) if value is not None else None
		elif attr_type == 'text':
			self.value_text = str(value) if value is not None else None
		elif attr_type == 'int':
			self.value_int = int(value) if value is not None else None
		elif attr_type == 'decimal':
			self.value_decimal = Decimal(str(value)) if value is not None else None
		elif attr_type == 'datetime':
			self.value_datetime = value if isinstance(value, datetime) else None
		elif attr_type == 'boolean':
			self.value_boolean = bool(value) if value is not None else None
		elif attr_type == 'json':
			self.value_json = value
		else:
			self.value_varchar = str(value) if value is not None else None
	
	def get_typed_value(self) -> Any:
		"""Get value with proper type conversion"""
		value = self.get_value()
		
		if value is None:
			return None
		
		attr_type = self.attribute.attribute_type
		
		if attr_type == 'number':
			return float(value) if self.attribute.data_type == 'decimal' else int(value)
		elif attr_type == 'boolean':
			return bool(value)
		elif attr_type == 'date':
			return value if isinstance(value, datetime) else None
		elif attr_type in ['select', 'multiselect']:
			return value
		else:
			return str(value)


class PCProductVariant(Model, AuditMixin, BaseMixin):
	"""
	Product variants for configurable products.
	
	Manages product variations with attribute combinations,
	individual pricing, and inventory tracking.
	"""
	__tablename__ = 'pc_product_variant'
	
	# Identity
	variant_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	parent_product_id = Column(String(36), ForeignKey('pc_product.product_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Variant Identification
	variant_sku = Column(String(100), nullable=False, unique=True, index=True)
	variant_name = Column(String(500), nullable=True)
	
	# Variant Attributes (combination that defines this variant)
	attribute_combination = Column(JSON, nullable=False)  # {attribute_id: option_value}
	
	# Pricing (overrides parent if specified)
	price = Column(Numeric(12, 4), nullable=True)
	cost = Column(Numeric(12, 4), nullable=True)
	special_price = Column(Numeric(12, 4), nullable=True)
	special_price_from = Column(DateTime, nullable=True)
	special_price_to = Column(DateTime, nullable=True)
	
	# Inventory (independent of parent)
	qty = Column(Numeric(12, 4), default=0)
	is_in_stock = Column(Boolean, default=True, index=True)
	
	# Physical Properties (overrides parent if specified)
	weight = Column(Numeric(12, 4), nullable=True)
	length = Column(Numeric(12, 4), nullable=True)
	width = Column(Numeric(12, 4), nullable=True)
	height = Column(Numeric(12, 4), nullable=True)
	
	# Status
	is_enabled = Column(Boolean, default=True, index=True)
	sort_order = Column(Integer, default=0)
	
	# Images specific to this variant
	image = Column(String(500), nullable=True)
	gallery_images = Column(JSON, default=list)
	
	# Performance
	view_count = Column(Integer, default=0)
	sale_count = Column(Integer, default=0)
	
	# Relationships
	parent_product = relationship("PCProduct", back_populates="variants")
	
	def __repr__(self):
		return f"<PCProductVariant {self.variant_sku}>"
	
	def get_effective_price(self) -> Decimal:
		"""Get effective price (variant or parent)"""
		if self.is_on_sale() and self.special_price:
			return Decimal(str(self.special_price))
		elif self.price:
			return Decimal(str(self.price))
		else:
			return self.parent_product.get_effective_price()
	
	def is_on_sale(self) -> bool:
		"""Check if variant is on sale"""
		if not self.special_price:
			return False
		
		now = datetime.utcnow()
		
		if self.special_price_from and now < self.special_price_from:
			return False
		
		if self.special_price_to and now > self.special_price_to:
			return False
		
		base_price = self.price or self.parent_product.price
		return self.special_price < base_price
	
	def is_available(self) -> bool:
		"""Check if variant is available"""
		return (self.is_enabled and 
				self.is_in_stock and 
				self.qty > 0 and
				self.parent_product.status == 'active')
	
	def get_attribute_label(self, attribute_id: str) -> str:
		"""Get display label for attribute value"""
		if attribute_id not in self.attribute_combination:
			return ""
		
		# This would look up the actual attribute option label
		# Implementation would query PCAttributeOption
		return str(self.attribute_combination[attribute_id])
	
	def get_variant_display_name(self) -> str:
		"""Get display name with attribute labels"""
		if self.variant_name:
			return self.variant_name
		
		# Build name from attribute combinations
		labels = []
		for attr_id, value in self.attribute_combination.items():
			label = self.get_attribute_label(attr_id)
			if label:
				labels.append(label)
		
		base_name = self.parent_product.name
		if labels:
			return f"{base_name} - {' / '.join(labels)}"
		
		return base_name


class PCProductImage(Model, AuditMixin, BaseMixin):
	"""
	Product image management with metadata and optimization.
	
	Stores product images with detailed metadata, alt text,
	and optimization information for SEO and performance.
	"""
	__tablename__ = 'pc_product_image'
	
	# Identity
	image_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	product_id = Column(String(36), ForeignKey('pc_product.product_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Image Information
	filename = Column(String(500), nullable=False)
	original_filename = Column(String(500), nullable=False)
	file_path = Column(String(1000), nullable=False)
	url = Column(String(1000), nullable=False)
	
	# Image Properties
	file_size = Column(Integer, nullable=False)  # Size in bytes
	width = Column(Integer, nullable=True)
	height = Column(Integer, nullable=True)
	mime_type = Column(String(100), nullable=False)
	file_hash = Column(String(64), nullable=True)  # For deduplication
	
	# Image Types and Usage
	image_type = Column(String(20), nullable=False)  # main, thumbnail, gallery, swatch
	is_main = Column(Boolean, default=False, index=True)
	is_visible = Column(Boolean, default=True)
	sort_order = Column(Integer, default=0)
	
	# SEO and Accessibility
	alt_text = Column(String(200), nullable=True)
	title = Column(String(200), nullable=True)
	caption = Column(Text, nullable=True)
	
	# Optimization
	is_optimized = Column(Boolean, default=False)
	optimization_score = Column(Integer, nullable=True)  # 0-100
	
	# Variants (for different sizes/formats)
	thumbnail_url = Column(String(1000), nullable=True)
	small_url = Column(String(1000), nullable=True)
	medium_url = Column(String(1000), nullable=True)
	large_url = Column(String(1000), nullable=True)
	webp_url = Column(String(1000), nullable=True)  # WebP format for modern browsers
	
	# Usage Statistics
	view_count = Column(Integer, default=0)
	
	# Relationships
	product = relationship("PCProduct", back_populates="images")
	
	def __repr__(self):
		return f"<PCProductImage {self.filename} ({self.image_type})>"
	
	def get_file_size_formatted(self) -> str:
		"""Get formatted file size"""
		size = self.file_size
		if size < 1024:
			return f"{size} B"
		elif size < 1024 * 1024:
			return f"{size / 1024:.1f} KB"
		else:
			return f"{size / (1024 * 1024):.1f} MB"
	
	def get_aspect_ratio(self) -> float:
		"""Get image aspect ratio"""
		if self.width and self.height:
			return self.width / self.height
		return 1.0
	
	def is_landscape(self) -> bool:
		"""Check if image is landscape orientation"""
		return self.get_aspect_ratio() > 1.0
	
	def is_portrait(self) -> bool:
		"""Check if image is portrait orientation"""
		return self.get_aspect_ratio() < 1.0
	
	def get_optimized_url(self, size: str = 'medium') -> str:
		"""Get optimized URL for specific size"""
		size_urls = {
			'thumbnail': self.thumbnail_url,
			'small': self.small_url,
			'medium': self.medium_url,
			'large': self.large_url,
			'webp': self.webp_url
		}
		
		return size_urls.get(size) or self.url


class PCProductReview(Model, AuditMixin, BaseMixin):
	"""
	Product reviews and ratings system.
	
	Manages customer reviews with ratings, verification status,
	moderation workflow, and helpful votes.
	"""
	__tablename__ = 'pc_product_review'
	
	# Identity
	review_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	product_id = Column(String(36), ForeignKey('pc_product.product_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Review Content
	title = Column(String(200), nullable=False)
	review_text = Column(Text, nullable=False)
	rating = Column(Integer, nullable=False, index=True)  # 1-5 stars
	
	# Reviewer Information
	reviewer_id = Column(String(36), nullable=True, index=True)  # User ID if logged in
	reviewer_name = Column(String(200), nullable=False)
	reviewer_email = Column(String(200), nullable=True)
	is_verified_buyer = Column(Boolean, default=False, index=True)
	
	# Review Status
	status = Column(String(20), default='pending', index=True)  # pending, approved, rejected, spam
	is_featured = Column(Boolean, default=False)
	
	# Detailed Ratings (optional)
	quality_rating = Column(Integer, nullable=True)  # 1-5
	value_rating = Column(Integer, nullable=True)    # 1-5
	service_rating = Column(Integer, nullable=True)  # 1-5
	
	# Review Metadata
	review_date = Column(DateTime, default=datetime.utcnow, index=True)
	purchase_date = Column(DateTime, nullable=True)
	
	# Moderation
	moderated_by = Column(String(36), nullable=True)  # Moderator user ID
	moderated_at = Column(DateTime, nullable=True)
	moderation_notes = Column(Text, nullable=True)
	
	# Interaction Metrics
	helpful_votes = Column(Integer, default=0)
	total_votes = Column(Integer, default=0)
	
	# Additional Information
	pros = Column(Text, nullable=True)  # What the reviewer liked
	cons = Column(Text, nullable=True)  # What the reviewer didn't like
	usage_context = Column(String(200), nullable=True)  # How they used the product
	
	# Review Images
	image_urls = Column(JSON, default=list)  # Customer uploaded images
	
	# Relationships
	product = relationship("PCProduct", back_populates="reviews")
	
	def __repr__(self):
		return f"<PCProductReview {self.title} ({self.rating} stars)>"
	
	def get_helpfulness_percentage(self) -> float:
		"""Get percentage of helpful votes"""
		if self.total_votes == 0:
			return 0.0
		return (self.helpful_votes / self.total_votes) * 100
	
	def is_helpful(self) -> bool:
		"""Check if review is considered helpful"""
		return self.get_helpfulness_percentage() >= 60  # 60% threshold
	
	def can_be_displayed(self) -> bool:
		"""Check if review can be displayed publicly"""
		return self.status == 'approved'
	
	def get_review_age_days(self) -> int:
		"""Get review age in days"""
		return (datetime.utcnow() - self.review_date).days
	
	def is_recent(self, days: int = 30) -> bool:
		"""Check if review is recent"""
		return self.get_review_age_days() <= days
	
	def update_product_rating(self) -> None:
		"""Update parent product's average rating"""
		# This would be called after review approval/rejection
		if self.product:
			approved_reviews = [r for r in self.product.reviews if r.status == 'approved']
			if approved_reviews:
				total_rating = sum(r.rating for r in approved_reviews)
				self.product.rating_average = Decimal(str(total_rating / len(approved_reviews)))
				self.product.review_count = len(approved_reviews)
			else:
				self.product.rating_average = Decimal('0')
				self.product.review_count = 0


class PCInventoryLog(Model, AuditMixin, BaseMixin):
	"""
	Inventory movement tracking for products.
	
	Records all inventory changes with detailed context,
	source tracking, and audit capabilities.
	"""
	__tablename__ = 'pc_inventory_log'
	
	# Identity
	log_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	product_id = Column(String(36), ForeignKey('pc_product.product_id'), nullable=True, index=True)
	variant_id = Column(String(36), ForeignKey('pc_product_variant.variant_id'), nullable=True, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Movement Information
	movement_type = Column(String(20), nullable=False, index=True)  # sale, purchase, adjustment, return
	quantity_change = Column(Numeric(12, 4), nullable=False)  # Positive = increase, Negative = decrease
	quantity_before = Column(Numeric(12, 4), nullable=False)
	quantity_after = Column(Numeric(12, 4), nullable=False)
	
	# Context Information
	reference_type = Column(String(50), nullable=True)  # order, purchase_order, adjustment, etc.
	reference_id = Column(String(36), nullable=True, index=True)
	reference_number = Column(String(100), nullable=True)
	
	# Movement Details
	unit_cost = Column(Numeric(12, 4), nullable=True)
	total_cost = Column(Numeric(12, 4), nullable=True)
	reason = Column(String(200), nullable=True)
	notes = Column(Text, nullable=True)
	
	# Tracking
	movement_date = Column(DateTime, default=datetime.utcnow, index=True)
	processed_by = Column(String(36), nullable=True, index=True)  # User ID
	warehouse_id = Column(String(36), nullable=True, index=True)
	location = Column(String(100), nullable=True)  # Warehouse location
	
	# Batch/Lot Tracking
	batch_number = Column(String(100), nullable=True)
	lot_number = Column(String(100), nullable=True)
	expiry_date = Column(DateTime, nullable=True)
	
	def __repr__(self):
		return f"<PCInventoryLog {self.movement_type}: {self.quantity_change}>"
	
	def is_increase(self) -> bool:
		"""Check if this is an inventory increase"""
		return self.quantity_change > 0
	
	def is_decrease(self) -> bool:
		"""Check if this is an inventory decrease"""
		return self.quantity_change < 0
	
	def get_absolute_change(self) -> Decimal:
		"""Get absolute value of quantity change"""
		return abs(self.quantity_change)