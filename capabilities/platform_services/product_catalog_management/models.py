"""
Product Catalog Management Models

Database models for comprehensive product catalog management including
products, categories, variants, pricing, and inventory.
"""

from sqlalchemy import Column, String, Text, Boolean, Integer, DateTime, Numeric, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum

Base = declarative_base()

class ProductStatus(str, Enum):
	DRAFT = "draft"
	ACTIVE = "active"
	INACTIVE = "inactive"
	DISCONTINUED = "discontinued"

class ProductType(str, Enum):
	SIMPLE = "simple"
	VARIABLE = "variable"
	GROUPED = "grouped"
	EXTERNAL = "external"
	BUNDLE = "bundle"
	DIGITAL = "digital"

class CategoryType(str, Enum):
	STANDARD = "standard"
	BRAND = "brand"
	COLLECTION = "collection"
	TAG = "tag"

class AttributeType(str, Enum):
	TEXT = "text"
	NUMBER = "number"
	SELECT = "select"
	MULTISELECT = "multiselect"
	BOOLEAN = "boolean"
	DATE = "date"
	COLOR = "color"
	IMAGE = "image"

class PricingType(str, Enum):
	FIXED = "fixed"
	TIERED = "tiered"
	DYNAMIC = "dynamic"
	AUCTION = "auction"

# SQLAlchemy Models
class PSProductCategory(Base):
	__tablename__ = 'ps_product_categories'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	
	# Category Information
	name = Column(String(255), nullable=False)
	slug = Column(String(255), nullable=False)
	description = Column(Text)
	category_type = Column(String(20), nullable=False, default=CategoryType.STANDARD.value)
	
	# Hierarchy
	parent_id = Column(String(36), ForeignKey('ps_product_categories.id'), nullable=True)
	level = Column(Integer, default=0)
	sort_order = Column(Integer, default=0)
	
	# Display
	image_url = Column(String(500))
	icon = Column(String(100))
	color = Column(String(20))
	is_featured = Column(Boolean, default=False)
	is_active = Column(Boolean, default=True)
	
	# SEO
	meta_title = Column(String(255))
	meta_description = Column(Text)
	meta_keywords = Column(Text)
	
	# Configuration
	attributes = Column(JSON)  # Default attributes for products in this category
	filters = Column(JSON)    # Available filters for this category
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(36))
	updated_by = Column(String(36))
	
	# Relationships
	parent = relationship("PSProductCategory", remote_side=[id])
	children = relationship("PSProductCategory", back_populates="parent")
	products = relationship("PSProduct", back_populates="category")
	
	# Indexes
	__table_args__ = (
		Index('idx_category_tenant_slug', 'tenant_id', 'slug'),
		Index('idx_category_parent', 'parent_id'),
		Index('idx_category_active', 'is_active'),
	)

class PSProduct(Base):
	__tablename__ = 'ps_products'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	
	# Basic Information
	name = Column(String(255), nullable=False)
	slug = Column(String(255), nullable=False)
	sku = Column(String(100), nullable=False)
	barcode = Column(String(100))
	description = Column(Text)
	short_description = Column(Text)
	
	# Product Type and Status
	product_type = Column(String(20), nullable=False, default=ProductType.SIMPLE.value)
	status = Column(String(20), nullable=False, default=ProductStatus.DRAFT.value)
	
	# Categorization
	category_id = Column(String(36), ForeignKey('ps_product_categories.id'), nullable=False)
	brand = Column(String(255))
	tags = Column(JSON)  # List of tags
	
	# Pricing (base price)
	base_price = Column(Numeric(10, 2), nullable=False, default=0.00)
	sale_price = Column(Numeric(10, 2))
	cost_price = Column(Numeric(10, 2))
	currency = Column(String(3), default='USD')
	
	# Physical Properties
	weight = Column(Numeric(8, 3))
	length = Column(Numeric(8, 3))
	width = Column(Numeric(8, 3))
	height = Column(Numeric(8, 3))
	weight_unit = Column(String(10), default='kg')
	dimension_unit = Column(String(10), default='cm')
	
	# Inventory
	manage_stock = Column(Boolean, default=True)
	stock_quantity = Column(Integer, default=0)
	low_stock_threshold = Column(Integer, default=5)
	stock_status = Column(String(20), default='in_stock')
	
	# Shipping
	requires_shipping = Column(Boolean, default=True)
	shipping_class = Column(String(100))
	shipping_weight = Column(Numeric(8, 3))
	
	# Digital Product Properties
	is_digital = Column(Boolean, default=False)
	download_url = Column(String(500))
	download_limit = Column(Integer)
	download_expiry = Column(Integer)  # Days
	
	# SEO and Marketing
	meta_title = Column(String(255))
	meta_description = Column(Text)
	meta_keywords = Column(Text)
	featured_image_url = Column(String(500))
	gallery_images = Column(JSON)  # List of image URLs
	
	# Features and Configuration
	is_featured = Column(Boolean, default=False)
	is_virtual = Column(Boolean, default=False)
	is_downloadable = Column(Boolean, default=False)
	allow_reviews = Column(Boolean, default=True)
	
	# Statistics
	view_count = Column(Integer, default=0)
	purchase_count = Column(Integer, default=0)
	rating_average = Column(Numeric(3, 2), default=0.00)
	rating_count = Column(Integer, default=0)
	
	# Dates
	published_at = Column(DateTime)
	sale_start_date = Column(DateTime)
	sale_end_date = Column(DateTime)
	
	# Vendor/Seller (if marketplace)
	vendor_id = Column(String(36))  # Links to vendor management
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(36))
	updated_by = Column(String(36))
	
	# Relationships
	category = relationship("PSProductCategory", back_populates="products")
	attributes = relationship("PSProductAttribute", back_populates="product", cascade="all, delete-orphan")
	variants = relationship("PSProductVariant", back_populates="parent_product", cascade="all, delete-orphan")
	images = relationship("PSProductImage", back_populates="product", cascade="all, delete-orphan")
	pricing = relationship("PSProductPricing", back_populates="product", cascade="all, delete-orphan")
	inventory = relationship("PSProductInventory", back_populates="product", cascade="all, delete-orphan")
	related_from = relationship("PSProductRelation", foreign_keys="PSProductRelation.product_id", back_populates="product")
	related_to = relationship("PSProductRelation", foreign_keys="PSProductRelation.related_product_id", back_populates="related_product")
	
	# Indexes
	__table_args__ = (
		Index('idx_product_tenant_sku', 'tenant_id', 'sku', unique=True),
		Index('idx_product_tenant_slug', 'tenant_id', 'slug'),
		Index('idx_product_category', 'category_id'),
		Index('idx_product_status', 'status'),
		Index('idx_product_featured', 'is_featured'),
		Index('idx_product_vendor', 'vendor_id'),
	)

class PSProductAttribute(Base):
	__tablename__ = 'ps_product_attributes'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	product_id = Column(String(36), ForeignKey('ps_products.id'), nullable=False)
	
	# Attribute Information
	name = Column(String(255), nullable=False)
	slug = Column(String(255), nullable=False)
	attribute_type = Column(String(20), nullable=False, default=AttributeType.TEXT.value)
	
	# Value
	value = Column(Text)
	numeric_value = Column(Numeric(15, 6))
	boolean_value = Column(Boolean)
	date_value = Column(DateTime)
	
	# Configuration
	is_required = Column(Boolean, default=False)
	is_variation = Column(Boolean, default=False)  # Used for creating variants
	is_visible = Column(Boolean, default=True)
	is_filterable = Column(Boolean, default=False)
	
	# Options (for select/multiselect)
	options = Column(JSON)  # List of available options
	
	# Display
	sort_order = Column(Integer, default=0)
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	
	# Relationships
	product = relationship("PSProduct", back_populates="attributes")
	
	# Indexes
	__table_args__ = (
		Index('idx_attribute_product', 'product_id'),
		Index('idx_attribute_name', 'name'),
		Index('idx_attribute_variation', 'is_variation'),
	)

class PSProductVariant(Base):
	__tablename__ = 'ps_product_variants'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	parent_product_id = Column(String(36), ForeignKey('ps_products.id'), nullable=False)
	
	# Variant Information
	name = Column(String(255), nullable=False)
	sku = Column(String(100), nullable=False)
	barcode = Column(String(100))
	
	# Variation Attributes
	variation_attributes = Column(JSON)  # {"color": "red", "size": "large"}
	
	# Pricing
	price = Column(Numeric(10, 2))
	sale_price = Column(Numeric(10, 2))
	cost_price = Column(Numeric(10, 2))
	
	# Inventory
	stock_quantity = Column(Integer, default=0)
	stock_status = Column(String(20), default='in_stock')
	
	# Physical Properties
	weight = Column(Numeric(8, 3))
	length = Column(Numeric(8, 3))
	width = Column(Numeric(8, 3))
	height = Column(Numeric(8, 3))
	
	# Media
	image_url = Column(String(500))
	gallery_images = Column(JSON)
	
	# Status
	is_active = Column(Boolean, default=True)
	is_default = Column(Boolean, default=False)
	
	# Statistics
	purchase_count = Column(Integer, default=0)
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	
	# Relationships
	parent_product = relationship("PSProduct", back_populates="variants")
	
	# Indexes
	__table_args__ = (
		Index('idx_variant_tenant_sku', 'tenant_id', 'sku', unique=True),
		Index('idx_variant_product', 'parent_product_id'),
		Index('idx_variant_active', 'is_active'),
	)

class PSProductImage(Base):
	__tablename__ = 'ps_product_images'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	product_id = Column(String(36), ForeignKey('ps_products.id'), nullable=False)
	
	# Image Information
	url = Column(String(500), nullable=False)
	alt_text = Column(String(255))
	title = Column(String(255))
	caption = Column(Text)
	
	# Image Properties
	width = Column(Integer)
	height = Column(Integer)
	file_size = Column(Integer)  # bytes
	mime_type = Column(String(100))
	
	# Organization
	sort_order = Column(Integer, default=0)
	is_primary = Column(Boolean, default=False)
	is_featured = Column(Boolean, default=False)
	
	# Usage
	image_type = Column(String(50), default='gallery')  # gallery, thumbnail, featured, etc.
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	
	# Relationships
	product = relationship("PSProduct", back_populates="images")
	
	# Indexes
	__table_args__ = (
		Index('idx_image_product', 'product_id'),
		Index('idx_image_primary', 'is_primary'),
	)

class PSProductPricing(Base):
	__tablename__ = 'ps_product_pricing'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	product_id = Column(String(36), ForeignKey('ps_products.id'), nullable=False)
	
	# Pricing Information
	pricing_type = Column(String(20), nullable=False, default=PricingType.FIXED.value)
	name = Column(String(255))  # Pricing rule name
	
	# Base Pricing
	price = Column(Numeric(10, 2), nullable=False)
	compare_at_price = Column(Numeric(10, 2))  # Original price for discounts
	cost_price = Column(Numeric(10, 2))
	currency = Column(String(3), default='USD')
	
	# Tiered Pricing
	quantity_min = Column(Integer, default=1)
	quantity_max = Column(Integer)
	
	# Customer Segmentation
	customer_group = Column(String(100))  # Premium, wholesale, etc.
	geographic_region = Column(String(100))
	
	# Time-based Pricing
	start_date = Column(DateTime)
	end_date = Column(DateTime)
	
	# Channel-specific Pricing
	channel = Column(String(100))  # website, mobile_app, marketplace, etc.
	storefront_id = Column(String(36))
	
	# Status
	is_active = Column(Boolean, default=True)
	priority = Column(Integer, default=0)  # For multiple pricing rules
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(36))
	updated_by = Column(String(36))
	
	# Relationships
	product = relationship("PSProduct", back_populates="pricing")
	
	# Indexes
	__table_args__ = (
		Index('idx_pricing_product', 'product_id'),
		Index('idx_pricing_active', 'is_active'),
		Index('idx_pricing_dates', 'start_date', 'end_date'),
	)

class PSProductInventory(Base):
	__tablename__ = 'ps_product_inventory'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	product_id = Column(String(36), ForeignKey('ps_products.id'), nullable=False)
	
	# Location/Warehouse
	location = Column(String(255), nullable=False, default='default')
	warehouse_id = Column(String(36))
	
	# Stock Information
	quantity_available = Column(Integer, default=0)
	quantity_reserved = Column(Integer, default=0)
	quantity_incoming = Column(Integer, default=0)
	quantity_damaged = Column(Integer, default=0)
	
	# Thresholds
	low_stock_threshold = Column(Integer, default=5)
	out_of_stock_threshold = Column(Integer, default=0)
	
	# Status
	stock_status = Column(String(20), default='in_stock')  # in_stock, low_stock, out_of_stock
	is_tracked = Column(Boolean, default=True)
	allow_backorder = Column(Boolean, default=False)
	
	# Last Updated
	last_count_date = Column(DateTime)
	last_movement_date = Column(DateTime)
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	
	# Relationships
	product = relationship("PSProduct", back_populates="inventory")
	
	# Indexes
	__table_args__ = (
		Index('idx_inventory_product_location', 'product_id', 'location', unique=True),
		Index('idx_inventory_status', 'stock_status'),
		Index('idx_inventory_warehouse', 'warehouse_id'),
	)

class PSProductRelation(Base):
	__tablename__ = 'ps_product_relations'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	
	# Relationship
	product_id = Column(String(36), ForeignKey('ps_products.id'), nullable=False)
	related_product_id = Column(String(36), ForeignKey('ps_products.id'), nullable=False)
	relation_type = Column(String(50), nullable=False)  # related, upsell, cross_sell, bundle, accessory
	
	# Configuration
	sort_order = Column(Integer, default=0)
	is_active = Column(Boolean, default=True)
	
	# Bundle-specific (if relation_type is bundle)
	quantity = Column(Integer, default=1)
	discount_percentage = Column(Numeric(5, 2), default=0.00)
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	
	# Relationships
	product = relationship("PSProduct", foreign_keys=[product_id], back_populates="related_from")
	related_product = relationship("PSProduct", foreign_keys=[related_product_id], back_populates="related_to")
	
	# Indexes
	__table_args__ = (
		Index('idx_relation_product', 'product_id'),
		Index('idx_relation_type', 'relation_type'),
		Index('idx_relation_active', 'is_active'),
	)

class PSProductBundle(Base):
	__tablename__ = 'ps_product_bundles'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	product_id = Column(String(36), ForeignKey('ps_products.id'), nullable=False)
	
	# Bundle Information
	name = Column(String(255), nullable=False)
	description = Column(Text)
	
	# Bundle Items (JSON array of products with quantities)
	bundle_items = Column(JSON, nullable=False)  # [{"product_id": "...", "quantity": 2, "discount": 10}]
	
	# Pricing
	bundle_price = Column(Numeric(10, 2))
	discount_type = Column(String(20), default='percentage')  # percentage, fixed
	discount_value = Column(Numeric(10, 2), default=0.00)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	allow_individual_purchase = Column(Boolean, default=True)
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(36))
	updated_by = Column(String(36))

class PSProductImport(Base):
	__tablename__ = 'ps_product_imports'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	
	# Import Information
	name = Column(String(255), nullable=False)
	file_name = Column(String(255))
	file_url = Column(String(500))
	import_type = Column(String(50), default='csv')  # csv, xml, json, api
	
	# Status
	status = Column(String(20), default='pending')  # pending, processing, completed, failed
	progress = Column(Integer, default=0)  # Percentage
	
	# Results
	total_records = Column(Integer, default=0)
	processed_records = Column(Integer, default=0)
	success_records = Column(Integer, default=0)
	error_records = Column(Integer, default=0)
	
	# Configuration
	mapping_config = Column(JSON)  # Field mapping configuration
	import_options = Column(JSON)  # Import-specific options
	
	# Error Log
	error_log = Column(Text)
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	completed_at = Column(DateTime)
	created_by = Column(String(36))

# Pydantic Models for API/Views
class ProductCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	name: str = Field(..., min_length=1, max_length=255)
	slug: str = Field(..., min_length=1, max_length=255)
	sku: str = Field(..., min_length=1, max_length=100)
	description: str | None = None
	short_description: str | None = None
	product_type: ProductType = Field(default=ProductType.SIMPLE)
	category_id: str = Field(..., min_length=1)
	base_price: Decimal = Field(..., ge=0)
	weight: Decimal | None = Field(None, ge=0)

class ProductUpdate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	name: str | None = Field(None, min_length=1, max_length=255)
	description: str | None = None
	short_description: str | None = None
	status: ProductStatus | None = None
	base_price: Decimal | None = Field(None, ge=0)
	sale_price: Decimal | None = Field(None, ge=0)
	category_id: str | None = None
	is_featured: bool | None = None

class CategoryCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	name: str = Field(..., min_length=1, max_length=255)
	slug: str = Field(..., min_length=1, max_length=255)
	description: str | None = None
	parent_id: str | None = None
	category_type: CategoryType = Field(default=CategoryType.STANDARD)
	sort_order: int = Field(default=0, ge=0)